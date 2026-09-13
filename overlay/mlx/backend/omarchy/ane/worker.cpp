// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/ane/runtime_detail.h"

#include "mlx/backend/omarchy/ane/bundle.h"

#include <ane.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <unistd.h>
#include <vector>

namespace mlx::core::omarchy::ane::detail {
namespace {

constexpr const char* kQualifiedLibaneCommit =
    "f261a6cb537aca62f267ad3d01beda0d6877544c";
constexpr const char* kQualifiedDriverVersion = "f2a3e5e+lifecycle6";
using Clock = std::chrono::steady_clock;

class UncertainCompletion : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

std::string trim_system_value(std::string value) {
  while (!value.empty() &&
         (value.back() == '\0' || value.back() == '\n' || value.back() == '\r')) {
    value.pop_back();
  }
  return value;
}

std::string read_value(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw runtime_error("cannot read hardware identity " + path.string());
  }
  return trim_system_value(
      std::string(std::istreambuf_iterator<char>(input), {}));
}

bool has_compatible(const std::string& values, const std::string& expected) {
  size_t offset = 0;
  while (offset < values.size()) {
    size_t end = values.find('\0', offset);
    if (end == std::string::npos) {
      end = values.size();
    }
    if (values.substr(offset, end - offset) == expected) {
      return true;
    }
    offset = end + 1;
  }
  return false;
}

std::filesystem::path single_entry(
    const std::filesystem::path& directory,
    const std::string& prefix,
    const std::string& suffix) {
  std::filesystem::path match;
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    const std::string name = entry.path().filename().string();
    if (name.rfind(prefix, 0) != 0 ||
        (name.size() < suffix.size() ||
         name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0)) {
      continue;
    }
    if (!match.empty()) {
      throw runtime_error("hardware identity has multiple matching entries in " +
                          directory.string());
    }
    match = entry.path();
  }
  if (match.empty()) {
    throw runtime_error("hardware identity has no matching entry in " +
                        directory.string());
  }
  return match;
}

std::string verify_hardware_eligibility() {
  utsname system{};
  if (::uname(&system) != 0) {
    throw runtime_error("uname failed: " + std::string(std::strerror(errno)));
  }
  if (std::string(system.sysname) != "Linux" ||
      std::string(system.machine) != "aarch64") {
    throw runtime_error("ANE execution requires Linux aarch64 on the qualified base M1");
  }

  const auto node = single_entry("/proc/device-tree/soc", "ane@", "");
  const std::string compatible = read_value(node / "compatible");
  if (!has_compatible(compatible, "apple,t8103-ane")) {
    throw runtime_error("device-tree ANE node is not apple,t8103-ane");
  }
  const std::string status = read_value(node / "status");
  if (status != "okay") {
    throw runtime_error("device-tree ANE status is '" + status + "', not 'okay'");
  }

  const std::string module_version = read_value("/sys/module/ane/version");
  if (module_version != kQualifiedDriverVersion) {
    throw runtime_error(
        "qualified driver " + std::string(kQualifiedDriverVersion) +
        " required, found '" + module_version + "'");
  }
  const std::string module_source = read_value("/sys/module/ane/srcversion");

  const auto platform = single_entry("/sys/bus/platform/devices", "", ".ane");
  const auto driver = std::filesystem::canonical(platform / "driver").filename();
  if (driver != "ane") {
    throw runtime_error("ANE platform device is not bound to the ane driver");
  }
  const std::string power_control = read_value(platform / "power/control");
  const std::string runtime_status = read_value(platform / "power/runtime_status");
  if (power_control != "on" || runtime_status != "active") {
    throw runtime_error(
        "ANE runtime power must be on and active, found control='" +
        power_control + "' status='" + runtime_status + "'");
  }

  struct stat device_status {};
  if (::stat("/dev/accel/accel0", &device_status) != 0 ||
      !S_ISCHR(device_status.st_mode) ||
      ::access("/dev/accel/accel0", R_OK | W_OK) != 0) {
    throw runtime_error("qualified /dev/accel/accel0 is not readable and writable");
  }

  std::ostringstream identity;
  identity << "host=" << system.nodename << " kernel=" << system.release
           << " machine=" << system.machine
           << " dt_node=" << node.filename().string()
           << " dt_compatible=apple,t8103-ane"
           << " driver_version=" << module_version
           << " driver_srcversion=" << module_source
           << " runtime_pm=" << power_control << '/' << runtime_status
           << " libane_commit=" << kQualifiedLibaneCommit
           << " driver_abi=1";
  return identity.str();
}

std::vector<uint8_t> read_kernel(const AneValidatedProgram& program) {
  const uint64_t aligned_task =
      (program.anec_header.task_size + 15) & ~uint64_t{15};
  const uint64_t offset = kAnecPayloadOffset + aligned_task;
  const size_t size = checked_size(program.anec_header.kernel_size, "kernel byte count");
  std::vector<uint8_t> kernel(size);
  if (kernel.empty()) {
    return kernel;
  }
  std::ifstream input(program.anec, std::ios::binary);
  input.seekg(static_cast<std::streamoff>(offset));
  input.read(reinterpret_cast<char*>(kernel.data()), kernel.size());
  if (input.gcount() != static_cast<std::streamsize>(kernel.size())) {
    throw runtime_error("cannot read validated kernel section from " + program.anec.string());
  }
  return kernel;
}

struct AneHandleDeleter {
  void operator()(ane_nn* handle) const {
    if (handle != nullptr) {
      __ane_free(handle);
    }
  }
};

struct LoadedProgram {
  const AneProgram* manifest{nullptr};
  size_t manifest_index{0};
  std::unique_ptr<ane_nn, AneHandleDeleter> handle;
  std::vector<std::vector<uint8_t>> inputs;
  std::vector<std::vector<uint8_t>> outputs;
};

std::vector<LoadedProgram> load_programs(const AneBundle& bundle) {
  std::vector<LoadedProgram> loaded;
  loaded.reserve(bundle.programs.size());
  for (const auto& program : bundle.programs) {
    errno = 0;
    std::unique_ptr<ane_nn, AneHandleDeleter> handle(
        __ane_init(program.anec.c_str(), 0));
    if (!handle) {
      throw runtime_error(
          "libane rejected program " + std::to_string(program.manifest_index) +
          " or the ANE device; qualified driver ABI 1 is required (errno=" +
          std::to_string(errno) + ")");
    }
    const auto& manifest = bundle.manifest.programs.at(program.manifest_index);
    for (size_t i = 0; i < manifest.inputs.size(); ++i) {
      const uint64_t size = __ane_src_size(handle.get(), static_cast<uint32_t>(i));
      if (size != manifest.inputs[i].allocation_bytes) {
        throw runtime_error(
            "libane input allocation differs for program " +
            std::to_string(program.manifest_index));
      }
    }
    for (size_t i = 0; i < manifest.outputs.size(); ++i) {
      const uint64_t size = __ane_dst_size(handle.get(), static_cast<uint32_t>(i));
      if (size != manifest.outputs[i].allocation_bytes) {
        throw runtime_error(
            "libane output allocation differs for program " +
            std::to_string(program.manifest_index));
      }
    }

    auto kernel = read_kernel(program);
    if (!kernel.empty()) {
      if (ane_kernel_capacity(handle.get()) < kernel.size()) {
        throw runtime_error(
            "libane kernel capacity differs for program " +
            std::to_string(program.manifest_index));
      }
      errno = 0;
      int result = ane_bind_kernel(handle.get(), kernel.data(), kernel.size());
      if (result != 0) {
        throw runtime_error(
            "ane_bind_kernel failed for program " +
            std::to_string(program.manifest_index) + " with result " +
            std::to_string(result) + " errno=" + std::to_string(errno));
      }
    }
    std::vector<std::vector<uint8_t>> inputs;
    for (const auto& binding : manifest.inputs) {
      inputs.emplace_back(
          checked_size(binding.allocation_bytes, "input allocation"));
    }
    std::vector<std::vector<uint8_t>> outputs;
    for (const auto& binding : manifest.outputs) {
      outputs.emplace_back(
          checked_size(binding.allocation_bytes, "output allocation"));
    }
    loaded.push_back({
        &manifest,
        program.manifest_index,
        std::move(handle),
        std::move(inputs),
        std::move(outputs)});
  }
  return loaded;
}

size_t staging_size(const AneManifest& manifest) {
  uint64_t total = 0;
  auto add = [&](const std::vector<AneTensor>& tensors) {
    for (const auto& tensor : tensors) {
      if (tensor.byte_size > std::numeric_limits<uint64_t>::max() - total) {
        throw runtime_error("host staging size overflows uint64");
      }
      total += tensor.byte_size;
    }
  };
  add(manifest.inputs);
  add(manifest.outputs);
  return checked_size(total, "host staging size");
}

void read_inputs(
    const AneManifest& manifest,
    const uint8_t* staging,
    std::map<std::string, std::vector<uint8_t>>& dense) {
  size_t offset = 0;
  for (const auto& tensor : manifest.inputs) {
    auto& data = dense.at(tensor.name);
    std::memcpy(data.data(), staging + offset, data.size());
    offset += data.size();
  }
  for (const auto& tensor : manifest.intermediates) {
    std::fill(dense.at(tensor.name).begin(), dense.at(tensor.name).end(), 0);
  }
  for (const auto& tensor : manifest.outputs) {
    std::fill(dense.at(tensor.name).begin(), dense.at(tensor.name).end(), 0);
  }
}

void write_outputs(
    const AneManifest& manifest,
    uint8_t* staging,
    const std::map<std::string, std::vector<uint8_t>>& dense) {
  size_t offset = 0;
  for (const auto& tensor : manifest.inputs) {
    offset += checked_size(tensor.byte_size, "input byte count");
  }
  for (const auto& tensor : manifest.outputs) {
    const auto& data = dense.at(tensor.name);
    std::memcpy(staging + offset, data.data(), data.size());
    offset += data.size();
  }
}

void execute_program(LoadedProgram& program,
                     std::map<std::string, std::vector<uint8_t>>& dense) {
  for (size_t i = 0; i < program.manifest->inputs.size(); ++i) {
    const auto& binding = program.manifest->inputs[i];
    auto& packed = program.inputs[i];
    pack_binding(binding, dense.at(binding.tensor), packed);
    __ane_send(program.handle.get(), packed.data(), static_cast<uint32_t>(i));
  }

  errno = 0;
  int result = ane_exec(program.handle.get());
  if (result != 0) {
    throw UncertainCompletion(
        "ane_exec failed for program " + std::to_string(program.manifest_index) +
        " with result " + std::to_string(result) + " errno=" +
        std::to_string(errno));
  }

  for (size_t i = 0; i < program.manifest->outputs.size(); ++i) {
    const auto& binding = program.manifest->outputs[i];
    auto& packed = program.outputs[i];
    __ane_read(program.handle.get(), packed.data(), static_cast<uint32_t>(i));
    unpack_binding(binding, packed, dense.at(binding.tensor));
  }
}

std::string error_reason(std::string value) {
  constexpr const char* prefix = "[omarchy-ane] runtime: ";
  if (value.rfind(prefix, 0) == 0) {
    value.erase(0, std::strlen(prefix));
  }
  if (!value.empty() && value.back() == '.') {
    value.pop_back();
  }
  return value;
}

void set_detail(WorkerReply& reply, const std::string& value) {
  std::strncpy(reply.detail, value.c_str(), sizeof(reply.detail) - 1);
}

bool send_reply(int fd, const WorkerReply& reply) {
  return ::send(fd, &reply, sizeof(reply), MSG_NOSIGNAL) ==
      static_cast<ssize_t>(sizeof(reply));
}

bool receive_command(int fd, WorkerCommand& command) {
  ssize_t received = ::recv(fd, &command, sizeof(command), 0);
  return received == static_cast<ssize_t>(sizeof(command));
}

std::map<std::string, std::vector<uint8_t>> allocate_dense(
    const AneManifest& manifest) {
  std::map<std::string, std::vector<uint8_t>> dense;
  auto add = [&](const std::vector<AneTensor>& tensors) {
    for (const auto& tensor : tensors) {
      dense.emplace(
          tensor.name,
          std::vector<uint8_t>(checked_size(tensor.byte_size, "tensor byte count")));
    }
  };
  add(manifest.inputs);
  add(manifest.outputs);
  add(manifest.state);
  add(manifest.intermediates);
  return dense;
}

} // namespace

int run_worker(
    int control_fd,
    int staging_fd,
    size_t expected_staging_size,
    const std::filesystem::path& manifest_path,
    const std::map<std::string, std::filesystem::path>& payload_paths) {
  uint8_t* staging = nullptr;
  std::vector<LoadedProgram> programs;
  try {
    std::string identity = verify_hardware_eligibility();
    AneBundle bundle = load_bundle_snapshot(manifest_path, payload_paths);
    if (staging_size(bundle.manifest) != expected_staging_size) {
      throw runtime_error("parent and worker staging sizes differ");
    }
    void* mapping = ::mmap(
        nullptr,
        expected_staging_size,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        staging_fd,
        0);
    if (mapping == MAP_FAILED) {
      throw runtime_error("worker host staging mmap failed: " +
                          std::string(std::strerror(errno)));
    }
    staging = static_cast<uint8_t*>(mapping);
    programs = load_programs(bundle);
    auto dense = allocate_dense(bundle.manifest);

    WorkerReply ready;
    ready.kind = WorkerReplyKind::ready;
    set_detail(
        ready,
        "graph_hash=" + bundle.manifest.graph_hash + "\ncontract_sha256=" +
            sha256_file(manifest_path) + "\nmodel_sha256=" +
            bundle.manifest.release_asset.model_sha256 + "\n" + identity);
    if (!send_reply(control_fd, ready)) {
      throw runtime_error("parent closed before worker readiness reply");
    }

    while (true) {
      WorkerCommand command;
      if (!receive_command(control_fd, command)) {
        break;
      }
      WorkerReply reply;
      reply.serial = command.serial;
      if (command.version != kWorkerProtocolVersion ||
          command.deadline_monotonic_nanoseconds <= 0) {
        reply.kind = WorkerReplyKind::failed;
        set_detail(reply, "invalid worker command or deadline");
        send_reply(control_fd, reply);
        break;
      }
      if (command.operation == WorkerOperation::shutdown) {
        const uint64_t count = programs.size();
        programs.clear();
        reply.kind = WorkerReplyKind::stopped;
        reply.released_programs = count;
        send_reply(control_fd, reply);
        break;
      }
      if (command.operation != WorkerOperation::execute) {
        reply.kind = WorkerReplyKind::failed;
        set_detail(reply, "unknown worker operation");
        send_reply(control_fd, reply);
        break;
      }

      try {
        const auto deadline = Clock::time_point(
            std::chrono::nanoseconds(command.deadline_monotonic_nanoseconds));
        read_inputs(bundle.manifest, staging, dense);
        for (auto& program : programs) {
          if (Clock::now() >= deadline) {
            throw runtime_error(
                "deadline expired before program " +
                std::to_string(program.manifest_index) +
                "; no further ANE work was submitted");
          }
          execute_program(program, dense);
        }
        write_outputs(bundle.manifest, staging, dense);
        reply.kind = WorkerReplyKind::executed;
        set_detail(reply, "ordered dispatch completed");
        if (!send_reply(control_fd, reply)) {
          break;
        }
      } catch (const UncertainCompletion& error) {
        reply.kind = WorkerReplyKind::uncertain;
        set_detail(reply, error_reason(error.what()));
        send_reply(control_fd, reply);
        break;
      } catch (const std::exception& error) {
        reply.kind = WorkerReplyKind::failed;
        set_detail(reply, error_reason(error.what()));
        send_reply(control_fd, reply);
        break;
      }
    }
    programs.clear();
    ::munmap(staging, expected_staging_size);
    return 0;
  } catch (const std::exception& error) {
    WorkerReply reply;
    reply.kind = WorkerReplyKind::failed;
    set_detail(reply, error_reason(error.what()));
    send_reply(control_fd, reply);
    programs.clear();
    if (staging != nullptr) {
      ::munmap(staging, expected_staging_size);
    }
    return 1;
  }
}

} // namespace mlx::core::omarchy::ane::detail
