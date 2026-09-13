// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/ane/worker.h"

#include <sys/wait.h>
#include <unistd.h>

#include <csignal>

#include <cerrno>
#include <cstring>
#include <string>
#include <utility>

namespace mlx::core::omarchy::ane {

namespace {

// Wire protocol from child to supervisor:
//   "loaded\n"                        after every program is on the device
//   "iter\n"                          after each completed iteration
//   "out <name> <len>\n" + len bytes  final-iteration output payloads
//   "released\n"                      programs released, device closed
//   "failed:<reason>\n"               clean named failure (exit code 1)
constexpr char kTokenLoaded[] = "loaded\n";
constexpr char kTokenIteration[] = "iter\n";
constexpr char kTokenOut[] = "out ";
constexpr char kTokenReleased[] = "released\n";
constexpr char kTokenFailed[] = "failed:";

std::chrono::milliseconds now_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch());
}

void write_all(int fd, const void* data, size_t size) {
  const char* cursor = static_cast<const char*>(data);
  while (size > 0) {
    auto written = ::write(fd, cursor, size);
    if (written <= 0) {
      if (errno == EINTR) continue;
      return; // supervisor closed the pipe; nothing more to report
    }
    cursor += written;
    size -= static_cast<size_t>(written);
  }
}

size_t count_token(const std::string& received, const char* token) {
  size_t count = 0;
  for (auto pos = received.find(token); pos != std::string::npos;
       pos = received.find(token, pos + 1)) {
    ++count;
  }
  return count;
}

std::map<std::string, AneWorker::Buffer> parse_outputs(
    const std::string& received) {
  std::map<std::string, AneWorker::Buffer> outputs;
  size_t cursor = 0;
  const size_t prefix = std::strlen(kTokenOut);
  while (true) {
    auto header = received.find(kTokenOut, cursor);
    if (header == std::string::npos) break;
    auto line_end = received.find('\n', header);
    if (line_end == std::string::npos) break;
    std::string header_line =
        received.substr(header + prefix, line_end - header - prefix);
    auto space = header_line.find(' ');
    if (space == std::string::npos) break;
    std::string name = header_line.substr(0, space);
    size_t length = 0;
    try {
      length = static_cast<size_t>(
          std::stoull(header_line.substr(space + 1)));
    } catch (...) {
      break;
    }
    size_t payload_start = line_end + 1;
    if (received.size() < payload_start + length) break;
    outputs[name].assign(
        received.begin() + static_cast<long>(payload_start),
        received.begin() + static_cast<long>(payload_start + length));
    cursor = payload_start + length;
  }
  return outputs;
}

// Executes the dispatch plan once. Intermediates route through child
// memory; the worker owns every byte between programs.
void execute_plan(
    AneDevice& device,
    const AneBundle& bundle,
    const std::map<std::string, AneWorker::Buffer>& inputs,
    std::map<std::string, AneWorker::Buffer>& outputs) {
  std::map<std::string, AneWorker::Buffer> values(inputs);
  for (auto index : bundle.manifest.dispatch_plan) {
    if (index >= bundle.programs.size() ||
        index >= bundle.manifest.programs.size()) {
      throw AneDeviceError("dispatch plan references unknown program");
    }
    const auto& validated = bundle.programs[index];
    const auto& program = bundle.manifest.programs[index];
    for (size_t position = 0; position < program.inputs.size(); ++position) {
      const auto& binding = program.inputs[position];
      auto found = values.find(binding.tensor);
      if (found == values.end()) {
        throw AneDeviceError(
            "program '" + program.payload + "' input tensor '" +
            binding.tensor + "' has no staged value");
      }
      if (found->second.size() < binding.logical_bytes) {
        throw AneDeviceError(
            "tensor '" + binding.tensor + "' payload is " +
            std::to_string(found->second.size()) + " bytes, manifest "
            "requires " + std::to_string(binding.logical_bytes));
      }
      device.send(validated.manifest_index, static_cast<uint32_t>(position),
                  binding, found->second.data(), found->second.size());
    }
    device.exec(validated.manifest_index);
    for (size_t position = 0; position < program.outputs.size(); ++position) {
      const auto& binding = program.outputs[position];
      AneWorker::Buffer buffer(binding.logical_bytes);
      device.read(validated.manifest_index, static_cast<uint32_t>(position),
                  binding, buffer.data(), buffer.size());
      values[binding.tensor] = std::move(buffer);
    }
  }
  for (const auto& tensor : bundle.manifest.outputs) {
    auto found = values.find(tensor.name);
    if (found == values.end()) {
      throw AneDeviceError(
          "manifest output '" + tensor.name + "' was never produced");
    }
    outputs[tensor.name] = found->second;
  }
}

} // namespace

AneWorker::AneWorker(DeviceFactory factory, AneWorkerOptions options)
    : factory_(std::move(factory)), options_(options) {
  if (options_.deadline.count() <= 0) {
    throw std::invalid_argument("ANE worker deadline must be positive");
  }
  if (options_.iterations <= 0) {
    throw std::invalid_argument(
        "ANE worker iteration count must be positive");
  }
  if (!factory_) {
    throw std::invalid_argument("ANE worker requires a device factory");
  }
}

AneWorkerReport AneWorker::run(
    const AneBundle& bundle,
    const std::map<std::string, Buffer>& inputs,
    std::map<std::string, Buffer>* outputs) {
  if (quarantined()) {
    return AneWorkerReport{
        AneWorkerStatus::QuarantinedRefused,
        0,
        0,
        std::chrono::milliseconds(0),
        "quarantined: " + quarantine_reason_,
        std::string()};
  }

  int fds[2];
  if (::pipe(fds) != 0) {
    quarantine_reason_ =
        std::string("pipe creation failed: ") + std::strerror(errno);
    return AneWorkerReport{
        AneWorkerStatus::WorkerDied, 0, 0, std::chrono::milliseconds(0),
        quarantine_reason_, std::string()};
  }

  const auto started = now_ms();
  pid_t child = ::fork();
  if (child < 0) {
    ::close(fds[0]);
    ::close(fds[1]);
    quarantine_reason_ =
        std::string("fork failed: ") + std::strerror(errno);
    return AneWorkerReport{
        AneWorkerStatus::WorkerDied, 0, 0, std::chrono::milliseconds(0),
        quarantine_reason_, std::string()};
  }

  if (child == 0) {
    ::close(fds[0]);
    // The worker child must die by real signals so the supervisor can
    // classify the death; inherited handlers are reset to defaults.
    ::signal(SIGABRT, SIG_DFL);
    ::signal(SIGSEGV, SIG_DFL);
    ::signal(SIGINT, SIG_DFL);
    ::signal(SIGTERM, SIG_DFL);
    int code = 0;
    try {
      std::unique_ptr<AneDevice> device = factory_();
      for (const auto& program : bundle.programs) {
        device->load(program);
      }
      write_all(fds[1], kTokenLoaded, std::strlen(kTokenLoaded));
      for (int iteration = 0; iteration < options_.iterations; ++iteration) {
        std::map<std::string, Buffer> produced;
        execute_plan(*device, bundle, inputs, produced);
        write_all(fds[1], kTokenIteration, std::strlen(kTokenIteration));
        if (iteration + 1 == options_.iterations) {
          for (const auto& entry : produced) {
            std::string header = std::string(kTokenOut) + entry.first +
                                 " " +
                                 std::to_string(entry.second.size()) +
                                 "\n";
            write_all(fds[1], header.data(), header.size());
            write_all(fds[1], entry.second.data(), entry.second.size());
          }
        }
      }
      device->release();
      write_all(fds[1], kTokenReleased, std::strlen(kTokenReleased));
    } catch (const std::exception& error) {
      std::string reason = error.what();
      for (char& c : reason) {
        if (c == '\n' || c == '\r') c = ' ';
      }
      std::string line = std::string(kTokenFailed) + reason + "\n";
      write_all(fds[1], line.data(), line.size());
      code = 1;
    }
    ::close(fds[1]);
    ::_exit(code);
  }

  ::close(fds[1]);
  AneWorkerReport report = supervise(child, fds[0]);
  report.elapsed = now_ms() - started;
  if (report.status == AneWorkerStatus::Completed) {
    report.released_programs = static_cast<int>(bundle.programs.size());
    if (outputs != nullptr) {
      *outputs = parse_outputs(report.detail_extra);
    }
  }
  report.detail_extra.clear();
  return report;
}

AneWorkerReport AneWorker::supervise(pid_t child, int report_fd) {
  std::string received;
  char buffer[4096];
  const auto deadline = now_ms() + options_.deadline;

  for (;;) {
    if (now_ms() >= deadline) {
      int status = 0;
      ::kill(child, SIGKILL);
      while (::waitpid(child, &status, 0) < 0 && errno == EINTR) {
      }
      ::close(report_fd);
      quarantine_reason_ =
          "deadline of " + std::to_string(options_.deadline.count()) +
          " ms exceeded; worker killed; device completion state uncertain";
      AneWorkerReport out;
      out.iterations =
          static_cast<int>(count_token(received, kTokenIteration));
      out.status = AneWorkerStatus::DeadlineExceeded;
      out.detail = quarantine_reason_;
      return out;
    }

    int status = 0;
    pid_t reaped = ::waitpid(child, &status, WNOHANG);
    if (reaped == child) {
      while (true) {
        auto n = ::read(report_fd, buffer, sizeof buffer);
        if (n <= 0) break;
        received.append(buffer, static_cast<size_t>(n));
      }
      ::close(report_fd);

      AneWorkerReport out;
      out.iterations =
          static_cast<int>(count_token(received, kTokenIteration));

      if (WIFSIGNALED(status)) {
        quarantine_reason_ = "worker killed by signal " +
                             std::to_string(WTERMSIG(status)) +
                             "; device completion state uncertain";
        out.status = AneWorkerStatus::WorkerDied;
        out.detail = quarantine_reason_;
        return out;
      }
      if (WEXITSTATUS(status) == 0 &&
          received.find(kTokenReleased) != std::string::npos) {
        out.status = AneWorkerStatus::Completed;
        out.detail = "worker completed " + std::to_string(out.iterations) +
                     " iteration(s) and released its programs";
        out.detail_extra = std::move(received);
        return out;
      }
      auto failed_at = received.find(kTokenFailed);
      if (failed_at != std::string::npos) {
        auto end = received.find('\n', failed_at);
        out.status = AneWorkerStatus::DeviceFailed;
        out.detail = received.substr(
            failed_at + std::strlen(kTokenFailed),
            end == std::string::npos
                ? std::string::npos
                : end - failed_at - std::strlen(kTokenFailed));
        return out;
      }
      quarantine_reason_ =
          "worker exited mid-protocol (status " +
          std::to_string(WEXITSTATUS(status)) +
          ") without a terminal report; device completion state uncertain";
      out.status = AneWorkerStatus::WorkerDied;
      out.detail = quarantine_reason_;
      return out;
    }

    auto n = ::read(report_fd, buffer, sizeof buffer);
    if (n > 0) {
      received.append(buffer, static_cast<size_t>(n));
      continue;
    }
    if (n == 0) {
      // Child closed its end; it is reaped on a later pass, still inside
      // the deadline loop.
      ::usleep(1000);
      continue;
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
      ::usleep(1000);
      continue;
    }
    int kill_status = 0;
    ::kill(child, SIGKILL);
    while (::waitpid(child, &kill_status, 0) < 0 && errno == EINTR) {
    }
    ::close(report_fd);
    quarantine_reason_ =
        std::string("report pipe read failed: ") + std::strerror(errno);
    AneWorkerReport out;
    out.status = AneWorkerStatus::WorkerDied;
    out.detail = quarantine_reason_;
    return out;
  }
}

} // namespace mlx::core::omarchy::ane
