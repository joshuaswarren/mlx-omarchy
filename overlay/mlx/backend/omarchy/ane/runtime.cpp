// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/ane/runtime.h"

#include "mlx/backend/omarchy/ane/bundle.h"
#include "mlx/backend/omarchy/ane/runtime_detail.h"

#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <linux/memfd.h>
#include <poll.h>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

extern char** environ;

#if !defined(MLX_OMARCHY_ANE_WORKER_BUILD_PATH) || \
    !defined(MLX_OMARCHY_ANE_WORKER_INSTALL_PATH)
#error "private ANE worker build and install paths must be defined"
#endif

namespace mlx::core::omarchy::ane {
namespace {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

std::string system_error(const std::string& operation, int error = errno) {
  return operation + " failed: " + std::strerror(error);
}

size_t staging_size(const AneManifest& manifest) {
  uint64_t size = 0;
  auto add = [&](const std::vector<AneTensor>& tensors) {
    for (const auto& tensor : tensors) {
      if (tensor.byte_size > std::numeric_limits<uint64_t>::max() - size) {
        throw detail::runtime_error("host staging size overflows uint64");
      }
      size += tensor.byte_size;
    }
  };
  add(manifest.inputs);
  add(manifest.outputs);
  return detail::checked_size(size, "host staging size");
}

int remaining_milliseconds(TimePoint deadline) {
  auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
      deadline - Clock::now());
  if (remaining.count() <= 0) {
    return 0;
  }
  if (remaining.count() > std::numeric_limits<int>::max()) {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(remaining.count());
}

bool wait_ready(int fd, short events, TimePoint deadline) {
  while (true) {
    pollfd descriptor{fd, events, 0};
    int result = ::poll(&descriptor, 1, remaining_milliseconds(deadline));
    if (result > 0) {
      return (descriptor.revents & events) != 0;
    }
    if (result == 0) {
      return false;
    }
    if (errno != EINTR) {
      throw detail::runtime_error(system_error("poll"));
    }
  }
}

void send_command(int fd, const detail::WorkerCommand& command, TimePoint deadline) {
  if (!wait_ready(fd, POLLOUT, deadline)) {
    throw detail::runtime_error("deadline expired before worker command could be sent");
  }
  ssize_t sent = ::send(fd, &command, sizeof(command), MSG_NOSIGNAL);
  if (sent != static_cast<ssize_t>(sizeof(command))) {
    throw detail::runtime_error(system_error("worker command send"));
  }
}

detail::WorkerReply receive_reply(int fd, TimePoint deadline) {
  if (!wait_ready(fd, POLLIN, deadline)) {
    throw detail::runtime_error("worker completion deadline expired");
  }
  detail::WorkerReply reply;
  ssize_t received = ::recv(fd, &reply, sizeof(reply), 0);
  if (received == 0) {
    throw detail::runtime_error("worker exited without a completion reply");
  }
  if (received != static_cast<ssize_t>(sizeof(reply))) {
    throw detail::runtime_error(system_error("worker reply receive"));
  }
  if (reply.version != detail::kWorkerProtocolVersion) {
    throw detail::runtime_error("worker protocol version mismatch");
  }
  reply.detail[detail::kWorkerDetailBytes - 1] = '\0';
  return reply;
}

bool wait_for_exit(pid_t pid, TimePoint deadline) {
  while (Clock::now() < deadline) {
    int status = 0;
    pid_t result = ::waitpid(pid, &status, WNOHANG);
    if (result == pid) {
      return true;
    }
    if (result < 0) {
      return errno == ECHILD;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return false;
}

int duplicate_owned_fd(int fd) {
  int duplicate = ::fcntl(fd, F_DUPFD_CLOEXEC, 10);
  if (duplicate < 0) {
    throw detail::runtime_error(system_error("file descriptor duplication"));
  }
  return duplicate;
}

} // namespace

struct AneRuntime::Impl {
  AneBundle bundle;
  std::filesystem::path diagnostic_path;
  int control_fd{-1};
  int staging_fd{-1};
  uint8_t* staging{nullptr};
  size_t staging_bytes{0};
  pid_t pid{-1};
  uint64_t serial{0};
  bool is_usable{false};
  bool submitted{false};
  std::string identity;

  ~Impl() {
    if (control_fd >= 0) {
      ::close(control_fd);
    }
    if (staging != nullptr) {
      ::munmap(staging, staging_bytes);
    }
    if (staging_fd >= 0) {
      ::close(staging_fd);
    }
    if (pid > 0) {
      int status = 0;
      ::waitpid(pid, &status, WNOHANG);
    }
  }

  void preserve_diagnostic(const std::string& reason, bool uncertain) noexcept {
    try {
      if (!diagnostic_path.parent_path().empty()) {
        std::filesystem::create_directories(diagnostic_path.parent_path());
      }
      std::ofstream output(diagnostic_path, std::ios::app);
      output << "reason=" << reason << '\n'
             << "graph_hash=" << bundle.manifest.graph_hash << '\n'
             << "worker_pid=" << pid << '\n'
             << "submitted=" << (submitted ? "true" : "false") << '\n'
             << "runtime_identity=" << identity << '\n'
             << "recovery="
             << (uncertain
                     ? "stop ANE submissions; uncertain completion requires reboot"
                     : "worker stopped before another submission; hardware recovery not required")
             << '\n';
    } catch (...) {
    }
  }

  void mark_unusable(const std::string& reason, bool uncertain) {
    is_usable = false;
    preserve_diagnostic(reason, uncertain);
  }

  static std::unique_ptr<Impl> start(
      AneBundle loaded_bundle,
      std::chrono::milliseconds startup_deadline,
      const std::filesystem::path& diagnostic) {
    detail::validate_deadline(startup_deadline);
    auto implementation = std::make_unique<Impl>();
    implementation->bundle = std::move(loaded_bundle);
    implementation->diagnostic_path = diagnostic;
    if (implementation->diagnostic_path.empty()) {
      throw std::invalid_argument("ANE diagnostic path must not be empty");
    }
    if (!implementation->diagnostic_path.parent_path().empty()) {
      std::filesystem::create_directories(
          implementation->diagnostic_path.parent_path());
    }
    {
      std::ofstream output(implementation->diagnostic_path, std::ios::app);
      if (!output) {
        throw detail::runtime_error("ANE diagnostic path is not writable");
      }
      output << "runtime_load_graph_hash="
             << implementation->bundle.manifest.graph_hash << '\n';
      if (!output) {
        throw detail::runtime_error("ANE diagnostic path write failed");
      }
    }
    implementation->staging_bytes = staging_size(implementation->bundle.manifest);

    int memory = ::memfd_create("mlx-omarchy-ane", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (memory < 0) {
      throw detail::runtime_error(system_error("memfd_create"));
    }
    implementation->staging_fd = duplicate_owned_fd(memory);
    ::close(memory);
    if (::ftruncate(
            implementation->staging_fd,
            static_cast<off_t>(implementation->staging_bytes)) != 0) {
      throw detail::runtime_error(system_error("host staging allocation"));
    }
    void* mapping = ::mmap(
        nullptr,
        implementation->staging_bytes,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        implementation->staging_fd,
        0);
    if (mapping == MAP_FAILED) {
      throw detail::runtime_error(system_error("host staging mmap"));
    }
    implementation->staging = static_cast<uint8_t*>(mapping);

    std::string executable = MLX_OMARCHY_ANE_WORKER_BUILD_PATH;
    if (::access(executable.c_str(), X_OK) != 0) {
      executable = MLX_OMARCHY_ANE_WORKER_INSTALL_PATH;
    }
    if (::access(executable.c_str(), X_OK) != 0) {
      throw detail::runtime_error("private ANE worker executable not found");
    }

    int sockets[2];
    if (::socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, sockets) != 0) {
      throw detail::runtime_error(system_error("worker socketpair"));
    }
    implementation->control_fd = duplicate_owned_fd(sockets[0]);
    int child_control = duplicate_owned_fd(sockets[1]);
    ::close(sockets[0]);
    ::close(sockets[1]);

    posix_spawn_file_actions_t actions;
    int spawn_error = ::posix_spawn_file_actions_init(&actions);
    if (spawn_error != 0) {
      ::close(child_control);
      throw detail::runtime_error(system_error("posix_spawn_file_actions_init", spawn_error));
    }
    auto destroy_actions = [&] { ::posix_spawn_file_actions_destroy(&actions); };
    spawn_error = ::posix_spawn_file_actions_adddup2(
        &actions, child_control, detail::kWorkerControlFd);
    if (spawn_error == 0) {
      spawn_error = ::posix_spawn_file_actions_adddup2(
          &actions, implementation->staging_fd, detail::kWorkerStagingFd);
    }
    if (spawn_error == 0) {
      spawn_error = ::posix_spawn_file_actions_addclose(
          &actions, implementation->control_fd);
    }
    if (spawn_error != 0) {
      destroy_actions();
      ::close(child_control);
      throw detail::runtime_error(system_error("worker file actions", spawn_error));
    }

    std::string size = std::to_string(implementation->staging_bytes);
    std::string bundle_path = implementation->bundle.manifest.name.empty()
        ? std::string()
        : implementation->bundle.programs.front().anec.parent_path().string();
    char* arguments[] = {
        executable.data(), bundle_path.data(), size.data(), nullptr};
    spawn_error = ::posix_spawn(
        &implementation->pid,
        executable.c_str(),
        &actions,
        nullptr,
        arguments,
        environ);
    destroy_actions();
    ::close(child_control);
    if (spawn_error != 0) {
      implementation->pid = -1;
      throw detail::runtime_error(system_error("ANE worker spawn", spawn_error));
    }

    const TimePoint deadline = Clock::now() + startup_deadline;
    detail::WorkerReply reply;
    try {
      reply = receive_reply(implementation->control_fd, deadline);
    } catch (const std::exception& error) {
      implementation->mark_unusable(
          std::string("worker startup failed: ") + error.what(), false);
      throw;
    }
    if (reply.kind != detail::WorkerReplyKind::ready) {
      implementation->mark_unusable(reply.detail, false);
      if (wait_for_exit(implementation->pid, deadline)) {
        implementation->pid = -1;
      }
      throw detail::runtime_error(std::string(reply.detail));
    }
    const std::string graph_prefix = "graph_hash=" +
        implementation->bundle.manifest.graph_hash + "\n";
    const std::string worker_detail(reply.detail);
    if (worker_detail.rfind(graph_prefix, 0) != 0) {
      implementation->mark_unusable(
          "worker loaded a different bundle identity", false);
      ::close(implementation->control_fd);
      implementation->control_fd = -1;
      if (wait_for_exit(implementation->pid, deadline)) {
        implementation->pid = -1;
      }
      throw detail::runtime_error("worker loaded a different bundle identity");
    }
    implementation->identity = worker_detail.substr(graph_prefix.size());
    implementation->is_usable = true;
    return implementation;
  }
};

AneRuntime::AneRuntime(std::unique_ptr<Impl> implementation)
    : implementation_(std::move(implementation)) {}

std::unique_ptr<AneRuntime> AneRuntime::load(
    const std::filesystem::path& bundle,
    std::chrono::milliseconds startup_deadline,
    const std::filesystem::path& diagnostic_path) {
  AneBundle validated = load_bundle(bundle);
  return std::unique_ptr<AneRuntime>(new AneRuntime(
      Impl::start(std::move(validated), startup_deadline, diagnostic_path)));
}

AneRuntime::~AneRuntime() {
  if (implementation_ && implementation_->is_usable) {
    try {
      shutdown(std::chrono::seconds(5));
    } catch (const std::exception& error) {
      implementation_->mark_unusable(
          std::string("automatic shutdown failed: ") + error.what(), true);
    }
  }
}

AneBufferMap AneRuntime::execute(
    const AneBufferMap& inputs,
    std::chrono::milliseconds deadline_duration) {
  detail::validate_deadline(deadline_duration);
  if (!implementation_->is_usable) {
    throw detail::runtime_error("runtime is unusable; no ANE work was submitted");
  }
  const auto& expected = implementation_->bundle.manifest.inputs;
  if (inputs.size() != expected.size()) {
    throw detail::runtime_error("inputs do not exactly match the bundle manifest");
  }

  size_t offset = 0;
  for (const auto& tensor : expected) {
    auto found = inputs.find(tensor.name);
    if (found == inputs.end() || found->second.size() != tensor.byte_size) {
      throw detail::runtime_error(
          "input '" + tensor.name + "' byte count does not match the bundle manifest");
    }
    std::memcpy(
        implementation_->staging + offset,
        found->second.data(),
        found->second.size());
    offset += found->second.size();
  }

  const TimePoint deadline = Clock::now() + deadline_duration;
  detail::WorkerCommand command;
  command.operation = detail::WorkerOperation::execute;
  command.serial = ++implementation_->serial;
  command.deadline_monotonic_nanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          deadline.time_since_epoch()).count();
  try {
    send_command(implementation_->control_fd, command, deadline);
  } catch (const std::exception& error) {
    implementation_->mark_unusable(
        std::string("worker command was not sent: ") + error.what(), false);
    throw;
  }
  implementation_->submitted = true;

  detail::WorkerReply reply;
  try {
    reply = receive_reply(implementation_->control_fd, deadline);
  } catch (const std::exception& error) {
    implementation_->mark_unusable(
        std::string("completion became uncertain: ") + error.what(), true);
    throw detail::runtime_error(
        std::string(error.what()) +
        "; runtime marked unusable and no worker signal was sent; reboot is required before further ANE use");
  }
  if (reply.serial != command.serial) {
    implementation_->mark_unusable("worker reply serial mismatch", true);
    throw detail::runtime_error("worker reply serial mismatch; runtime marked unusable");
  }
  if (reply.kind == detail::WorkerReplyKind::uncertain) {
    implementation_->mark_unusable(reply.detail, true);
    if (wait_for_exit(implementation_->pid, deadline)) {
      implementation_->pid = -1;
    }
    throw detail::runtime_error(
        std::string(reply.detail) +
        "; runtime marked unusable and reboot is required before further ANE use");
  }
  if (reply.kind != detail::WorkerReplyKind::executed) {
    implementation_->mark_unusable(reply.detail, false);
    throw detail::runtime_error(
        std::string(reply.detail) + "; runtime marked unusable");
  }

  AneBufferMap outputs;
  for (const auto& tensor : implementation_->bundle.manifest.outputs) {
    AneBuffer data(detail::checked_size(tensor.byte_size, "output byte count"));
    std::memcpy(data.data(), implementation_->staging + offset, data.size());
    offset += data.size();
    outputs.emplace(tensor.name, std::move(data));
  }
  return outputs;
}

AneShutdownReceipt AneRuntime::shutdown(
    std::chrono::milliseconds deadline_duration) {
  detail::validate_deadline(deadline_duration);
  if (!implementation_->is_usable) {
    throw detail::runtime_error("runtime is unusable and cannot claim clean shutdown");
  }

  const TimePoint deadline = Clock::now() + deadline_duration;
  detail::WorkerCommand command;
  command.operation = detail::WorkerOperation::shutdown;
  command.serial = ++implementation_->serial;
  command.deadline_monotonic_nanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          deadline.time_since_epoch()).count();
  try {
    send_command(implementation_->control_fd, command, deadline);
  } catch (const std::exception& error) {
    implementation_->mark_unusable(
        std::string("shutdown command was not sent: ") + error.what(), false);
    throw;
  }
  detail::WorkerReply reply;
  try {
    reply = receive_reply(implementation_->control_fd, deadline);
  } catch (const std::exception& error) {
    implementation_->mark_unusable(
        std::string("shutdown completion became uncertain: ") + error.what(), true);
    throw detail::runtime_error(
        std::string(error.what()) +
        "; runtime marked unusable and no worker signal was sent; reboot is required before further ANE use");
  }
  if (reply.kind != detail::WorkerReplyKind::stopped || reply.serial != command.serial) {
    implementation_->mark_unusable(
        "worker did not confirm clean resource release", true);
    throw detail::runtime_error("worker did not confirm clean resource release");
  }
  if (!wait_for_exit(implementation_->pid, deadline)) {
    implementation_->mark_unusable(
        "worker confirmed release but did not exit by deadline", true);
    throw detail::runtime_error("worker confirmed release but did not exit by deadline");
  }

  AneShutdownReceipt receipt{
      static_cast<int>(implementation_->pid), reply.released_programs};
  implementation_->pid = -1;
  implementation_->is_usable = false;
  return receipt;
}

bool AneRuntime::usable() const {
  return implementation_->is_usable;
}

int AneRuntime::worker_pid() const {
  return static_cast<int>(implementation_->pid);
}

const std::string& AneRuntime::runtime_identity() const {
  return implementation_->identity;
}

} // namespace mlx::core::omarchy::ane
