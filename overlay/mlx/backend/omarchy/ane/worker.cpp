// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/ane/worker.h"

#include <poll.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <csignal>

#include <cerrno>
#include <algorithm>
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

// Device-phase wall split for one dispatch-plan execution: what the
// host spent copying payloads into the device (pack), waiting on
// execute ioctls (exec), and copying results back (read). The resident
// loop reports it per submit; the one-shot path ignores it.
struct DevicePhaseStats {
  long long pack_ns{0};
  long long exec_ns{0};
  long long read_ns{0};
};

std::chrono::steady_clock::time_point device_phase_mark() {
  return std::chrono::steady_clock::now();
}

void device_phase_add(
    long long* bucket,
    const std::chrono::steady_clock::time_point& started) {
  *bucket += std::chrono::duration_cast<std::chrono::nanoseconds>(
                 std::chrono::steady_clock::now() - started)
                 .count();
}

// A non-owning payload: points at an owning Buffer or at a span of a
// shared-memory region. data()/size() match AneWorker::Buffer so
// execute_plan can consume both interchangeably.
struct PayloadView {
  const uint8_t* ptr{nullptr};
  size_t len{0};
  const uint8_t* data() const { return ptr; }
  size_t size() const { return len; }
};

void execute_plan_views(
    AneDevice& device,
    const AneBundle& bundle,
    const std::map<std::string, PayloadView>& inputs,
    std::map<std::string, AneWorker::Buffer>& outputs,
    DevicePhaseStats* stats,
    const std::map<std::string, uint8_t*>* sinks);

// Executes the dispatch plan once. Inputs may be owning buffers or
// views (shared-memory spans); intermediates route through child
// memory; the worker owns every byte between programs. `sinks`, when
// provided, routes matching manifest outputs straight into caller
// memory (the device read unpacks into the sink) instead of an
// intermediate buffer; sunk outputs are not copied into `outputs`.
void execute_plan(
    AneDevice& device,
    const AneBundle& bundle,
    const std::map<std::string, AneWorker::Buffer>& buffer_inputs,
    std::map<std::string, AneWorker::Buffer>& outputs,
    DevicePhaseStats* stats = nullptr,
    const std::map<std::string, uint8_t*>* sinks = nullptr) {
  std::map<std::string, PayloadView> view_inputs;
  for (const auto& entry : buffer_inputs) {
    view_inputs[entry.first] =
        PayloadView{entry.second.data(), entry.second.size()};
  }
  execute_plan_views(device, bundle, view_inputs, outputs, stats, sinks);
}

void execute_plan_views(
    AneDevice& device,
    const AneBundle& bundle,
    const std::map<std::string, PayloadView>& inputs,
    std::map<std::string, AneWorker::Buffer>& outputs,
    DevicePhaseStats* stats,
    const std::map<std::string, uint8_t*>* sinks) {
  std::map<std::string, AneWorker::Buffer> values;
  auto resolve = [&](const std::string& name, const uint8_t** data,
                     size_t* size) -> bool {
    auto value = values.find(name);
    if (value != values.end()) {
      *data = value->second.data();
      *size = value->second.size();
      return true;
    }
    auto input = inputs.find(name);
    if (input != inputs.end()) {
      *data = input->second.data();
      *size = input->second.size();
      return true;
    }
    return false;
  };
  for (auto index : bundle.manifest.dispatch_plan) {
    if (index >= bundle.programs.size() ||
        index >= bundle.manifest.programs.size()) {
      throw AneDeviceError("dispatch plan references unknown program");
    }
    const auto& validated = bundle.programs[index];
    const auto& program = bundle.manifest.programs[index];
    for (size_t position = 0; position < program.inputs.size(); ++position) {
      const auto& binding = program.inputs[position];
      const uint8_t* data = nullptr;
      size_t size = 0;
      if (!resolve(binding.tensor, &data, &size)) {
        throw AneDeviceError(
            "program '" + program.payload + "' input tensor '" +
            binding.tensor + "' has no staged value");
      }
      if (size < binding.logical_bytes) {
        throw AneDeviceError(
            "tensor '" + binding.tensor + "' payload is " +
            std::to_string(size) + " bytes, manifest " +
            std::to_string(binding.logical_bytes));
      }
      auto mark = device_phase_mark();
      device.send(validated.manifest_index, static_cast<uint32_t>(position),
                  binding, data, size);
      if (stats) device_phase_add(&stats->pack_ns, mark);
    }
    {
      auto mark = device_phase_mark();
      device.exec(validated.manifest_index);
      if (stats) device_phase_add(&stats->exec_ns, mark);
    }
    for (size_t position = 0; position < program.outputs.size(); ++position) {
      const auto& binding = program.outputs[position];
      const bool have_sink =
          sinks != nullptr && sinks->count(binding.tensor) != 0;
      auto mark = device_phase_mark();
      if (have_sink) {
        device.read(validated.manifest_index, static_cast<uint32_t>(position),
                    binding, sinks->at(binding.tensor),
                    binding.logical_bytes);
      } else {
        AneWorker::Buffer buffer(binding.logical_bytes);
        device.read(validated.manifest_index, static_cast<uint32_t>(position),
                    binding, buffer.data(), buffer.size());
        values[binding.tensor] = std::move(buffer);
      }
      if (stats) device_phase_add(&stats->read_ns, mark);
    }
  }
  for (const auto& tensor : bundle.manifest.outputs) {
    if (sinks != nullptr && sinks->count(tensor.name) != 0) {
      continue; // landed directly in the sink memory
    }
    auto found = values.find(tensor.name);
    if (found == values.end()) {
      throw AneDeviceError(
          "manifest output '" + tensor.name + "' was never produced");
    }
    outputs[tensor.name] = found->second;
  }
}

// Resident-session wire protocol, one bounded request/response pair per
// submit over a single socketpair. Strictly sequential framing: a
// header line, then exactly its payload byte count -- so binary fp16
// payloads can never be mistaken for a control token.
//
//   parent -> child   "submit <bundle>\n"
//                     ("in <name> <len>\n" + len bytes)*
//                     "run\n"
//                     "close\n"                     end of session
//   child  -> parent  "loaded <programs>\n"         once, after open
//                     "iter\n"                      per completed iteration
//                     ("out <name> <len>\n" + len bytes)*
//                     "done\n"                      submit complete
//                     "released\n"                  programs released
//                     "failed:<reason>\n"           clean named failure
constexpr char kTokenDone[] = "done\n";
constexpr char kRequestRun[] = "run";
constexpr char kRequestClose[] = "close";
constexpr char kTokenShmOut[] = "shmout ";

// MSG_NOSIGNAL keeps a dead peer from raising SIGPIPE in either
// direction: the library never changes the process signal disposition.
bool send_frame(int fd, const void* data, size_t size) {
  const char* cursor = static_cast<const char*>(data);
  while (size > 0) {
    auto written = ::send(fd, cursor, size, MSG_NOSIGNAL);
    if (written <= 0) {
      if (written < 0 && errno == EINTR) continue;
      return false;
    }
    cursor += written;
    size -= static_cast<size_t>(written);
  }
  return true;
}

bool send_frame(int fd, const std::string& text) {
  return send_frame(fd, text.data(), text.size());
}

// Child-side blocking reads. `carry` holds bytes received past the end
// of the current frame; it only ever holds control lines, because
// payload frames are read straight into their destination buffer. A
// payload is megabytes: routing it through a growing string costs a
// reallocation and a copy of everything received so far on every
// recv(), which is quadratic in the payload size and measurably slower
// than the fork the resident worker is meant to save.
ssize_t child_recv(int fd, void* destination, size_t size) {
  for (;;) {
    auto n = ::recv(fd, destination, size, 0);
    if (n < 0 && errno == EINTR) continue;
    return n; // 0 or negative: the supervisor is gone
  }
}

bool child_read_line(int fd, std::string& carry, std::string& line) {
  char buffer[4096];
  for (;;) {
    auto end = carry.find('\n');
    if (end != std::string::npos) {
      line = carry.substr(0, end);
      carry.erase(0, end + 1);
      return true;
    }
    auto n = child_recv(fd, buffer, sizeof buffer);
    if (n <= 0) return false;
    carry.append(buffer, static_cast<size_t>(n));
  }
}

bool child_read_bytes(
    int fd,
    std::string& carry,
    size_t count,
    AneWorker::Buffer& out) {
  out.resize(count);
  size_t filled = std::min(carry.size(), count);
  if (filled > 0) {
    std::memcpy(out.data(), carry.data(), filled);
    carry.erase(0, filled);
  }
  while (filled < count) {
    auto n = child_recv(fd, out.data() + filled, count - filled);
    if (n <= 0) return false;
    filled += static_cast<size_t>(n);
  }
  return true;
}

// "in <name> <len>" / "out <name> <len>" payload headers.
bool parse_payload_header(
    const std::string& line,
    const char* keyword,
    std::string& name,
    size_t& length) {
  const std::string prefix = std::string(keyword) + " ";
  if (line.compare(0, prefix.size(), prefix) != 0) return false;
  auto space = line.find(' ', prefix.size());
  if (space == std::string::npos) return false;
  name = line.substr(prefix.size(), space - prefix.size());
  if (name.empty()) return false;
  try {
    size_t consumed = 0;
    length = static_cast<size_t>(
        std::stoull(line.substr(space + 1), &consumed));
    return consumed > 0;
  } catch (...) {
    return false;
  }
}

// "<kw> <name> <offset> <length>" shared-memory payload headers.
bool parse_shm_header(
    const std::string& line,
    const char* keyword,
    std::string& name,
    size_t& offset,
    size_t& length) {
  const std::string prefix = std::string(keyword) + " ";
  if (line.compare(0, prefix.size(), prefix) != 0) return false;
  auto first = line.find(' ', prefix.size());
  if (first == std::string::npos) return false;
  name = line.substr(prefix.size(), first - prefix.size());
  if (name.empty()) return false;
  auto second = line.find(' ', first + 1);
  if (second == std::string::npos) return false;
  try {
    size_t consumed = 0;
    offset = static_cast<size_t>(
        std::stoull(line.substr(first + 1, second - first - 1), &consumed));
    if (consumed == 0) return false;
    length = static_cast<size_t>(
        std::stoull(line.substr(second + 1), &consumed));
    return consumed > 0;
  } catch (...) {
    return false;
  }
}

std::string one_line(const std::string& text) {
  std::string flat = text;
  for (char& c : flat) {
    if (c == '\n' || c == '\r') c = ' ';
  }
  return flat;
}

// One resident child: serves bounded submits against already-loaded
// programs until the supervisor closes the session or a submit fails.
// With shm regions configured, "inshm"/"shout" request frames move
// payloads through shared memory instead of inline socket bytes: the
// device side reads inputs straight from shm_in and unpacks outputs
// straight into shm_out, answering with "shmout <name> <off> <len>".
int resident_child_loop(
    int fd,
    AneDevice& device,
    const std::vector<AneBundle>& bundles,
    int iterations,
    const AneWorkerOptions::ShmRegion& shm_in,
    const AneWorkerOptions::ShmRegion& shm_out) {
  std::string carry;
  for (;;) {
    std::string line;
    if (!child_read_line(fd, carry, line)) {
      return 0; // supervisor closed the channel; nothing left to serve
    }
    if (line == kRequestClose) {
      device.release();
      send_frame(fd, kTokenReleased, std::strlen(kTokenReleased));
      return 0;
    }
    size_t index = bundles.size();
    std::string selector;
    size_t parsed = 0;
    if (line.compare(0, 7, "submit ") == 0) {
      try {
        index = static_cast<size_t>(std::stoull(line.substr(7), &parsed));
      } catch (...) {
        parsed = 0;
      }
    }
    if (parsed == 0 || index >= bundles.size()) {
      send_frame(
          fd,
          std::string(kTokenFailed) + "unknown request '" + one_line(line) +
              "'\n");
      return 1;
    }
    const AneBundle& bundle = bundles[index];

    auto loop_mark = device_phase_mark();
    std::map<std::string, AneWorker::Buffer> owned_inputs;
    std::map<std::string, PayloadView> input_views;
    std::map<std::string, uint8_t*> sinks;
    std::map<std::string, size_t> sink_offsets;
    long long crecv_ns = 0;
    bool request_complete = false;
    while (child_read_line(fd, carry, line)) {
      if (line == kRequestRun) {
        request_complete = true;
        break;
      }
      std::string name;
      size_t length = 0;
      size_t offset = 0;
      if (parse_payload_header(line, "in", name, length)) {
        AneWorker::Buffer payload;
        auto mark = device_phase_mark();
        if (!child_read_bytes(fd, carry, length, payload)) {
          return 0;
        }
        device_phase_add(&crecv_ns, mark);
        owned_inputs[name] = std::move(payload);
      } else if (parse_shm_header(line, "inshm", name, offset, length)) {
        if (shm_in.base == nullptr || offset > shm_in.size ||
            length > shm_in.size - offset) {
          send_frame(
              fd,
              std::string(kTokenFailed) + "input span for '" + name +
                  "' is outside the shm region\n");
          return 1;
        }
        input_views[name] = PayloadView{shm_in.base + offset, length};
      } else if (parse_payload_header(line, "shout", name, offset)) {
        // Reserve an output sink: the logical size comes from the
        // bundle's output bindings so the span check is exact.
        size_t logical = 0;
        bool known = false;
        for (const auto& validated : bundle.programs) {
          const auto& manifest_program =
              bundle.manifest.programs[validated.manifest_index];
          for (const auto& binding : manifest_program.outputs) {
            if (binding.tensor == name) {
              logical = std::max(logical, binding.logical_bytes);
              known = true;
            }
          }
        }
        if (!known || shm_out.base == nullptr || offset > shm_out.size ||
            logical > shm_out.size - offset) {
          send_frame(
              fd,
              std::string(kTokenFailed) + "output span for '" + name +
                  "' is outside the shm region\n");
          return 1;
        }
        sinks[name] = shm_out.base + offset;
        sink_offsets[name] = offset;
      } else {
        send_frame(
            fd,
            std::string(kTokenFailed) + "malformed input frame '" +
                one_line(line) + "'\n");
        return 1;
      }
    }
    if (!request_complete) {
      return 0;
    }
    device_phase_add(&crecv_ns, loop_mark);
    for (const auto& entry : owned_inputs) {
      input_views[entry.first] =
          PayloadView{entry.second.data(), entry.second.size()};
    }

    try {
      DevicePhaseStats stats;
      for (int iteration = 0; iteration < iterations; ++iteration) {
        std::map<std::string, AneWorker::Buffer> produced;
        execute_plan_views(device, bundle, input_views, produced, &stats, &sinks);
        if (!send_frame(fd, kTokenIteration, std::strlen(kTokenIteration))) {
          return 0;
        }
        if (iteration + 1 == iterations) {
          for (const auto& tensor : bundle.manifest.outputs) {
            auto sink = sinks.find(tensor.name);
            if (sink != sinks.end()) {
              size_t logical = 0;
              for (const auto& validated : bundle.programs) {
                const auto& manifest_program =
                    bundle.manifest.programs[validated.manifest_index];
                for (const auto& binding : manifest_program.outputs) {
                  if (binding.tensor == tensor.name) {
                    logical = std::max(logical, binding.logical_bytes);
                  }
                }
              }
              std::string header = std::string(kTokenShmOut) + tensor.name +
                                   " " + std::to_string(sink_offsets[tensor.name]) +
                                   " " + std::to_string(logical) + "\n";
              if (!send_frame(fd, header)) {
                return 0;
              }
              continue;
            }
            auto produced_entry = produced.find(tensor.name);
            if (produced_entry == produced.end()) {
              throw AneDeviceError(
                  "manifest output '" + tensor.name + "' was never produced");
            }
            std::string header = std::string(kTokenOut) + tensor.name + " " +
                                 std::to_string(produced_entry->second.size()) +
                                 "\n";
            if (!send_frame(fd, header) ||
                !send_frame(
                    fd, produced_entry->second.data(),
                    produced_entry->second.size())) {
              return 0;
            }
          }
        }
      }
      std::string perf =
          "perf pack_us=" + std::to_string(stats.pack_ns / 1000) +
          " exec_us=" + std::to_string(stats.exec_ns / 1000) +
          " read_us=" + std::to_string(stats.read_ns / 1000) +
          " crecv_us=" + std::to_string(crecv_ns / 1000) + "\n";
      if (!send_frame(fd, perf) ||
          !send_frame(fd, kTokenDone, std::strlen(kTokenDone))) {
        return 0;
      }
    } catch (const std::exception& error) {
      send_frame(
          fd, std::string(kTokenFailed) + one_line(error.what()) + "\n");
      return 1;
    }
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

AneWorker::~AneWorker() {
  teardown();
}

void AneWorker::teardown() {
  if (child_ > 0) {
    int status = 0;
    ::kill(child_, SIGKILL);
    while (::waitpid(child_, &status, 0) < 0 && errno == EINTR) {
    }
    child_ = -1;
  }
  if (channel_ >= 0) {
    ::close(channel_);
    channel_ = -1;
  }
  inbox_.clear();
  inbox_cursor_ = 0;
  resident_programs_ = 0;
  batch_until_ = std::chrono::milliseconds(0);
  batch_rounds_ = 0;
}

AneWorkerReport AneWorker::end_session(const Wait& failure) {
  AneWorkerReport out;
  out.status = failure.status;
  if (failure.status == AneWorkerStatus::DeadlineExceeded ||
      failure.status == AneWorkerStatus::WorkerDied) {
    // Completion is uncertain, so the worker refuses everything after
    // this point -- the same rule the one-shot path applies.
    quarantine_reason_ = failure.detail;
  }
  out.detail = failure.detail;
  teardown();
  return out;
}

// Blocks until the channel is readable or writable, or the submit's
// deadline passes. This is a real wait, not a sleep quantum: a fixed
// sleep between socket drains caps payload throughput at
// chunk-size/sleep, which on this path measured about 320 MB/s and made
// a resident submit slower than a fresh process whose inputs arrive by
// fork.
AneWorker::Wait AneWorker::await_channel(
    short events,
    std::chrono::milliseconds until,
    const char* what) {
  auto remaining = until - now_ms();
  if (remaining.count() > 0) {
    struct pollfd descriptor {};
    descriptor.fd = channel_;
    descriptor.events = events;
    int ready = ::poll(&descriptor, 1, static_cast<int>(remaining.count()));
    if (ready > 0) {
      return Wait{true, AneWorkerStatus::Completed, std::string()};
    }
    if (ready < 0 && errno != EINTR) {
      return Wait{
          false, AneWorkerStatus::WorkerDied,
          std::string("resident channel poll failed: ") +
              std::strerror(errno) + "; device completion state uncertain"};
    }
    if (ready < 0) {
      return Wait{true, AneWorkerStatus::Completed, std::string()}; // EINTR
    }
  }
  if (now_ms() < until) {
    return Wait{true, AneWorkerStatus::Completed, std::string()};
  }
  return Wait{
      false, AneWorkerStatus::DeadlineExceeded,
      "deadline of " + std::to_string(options_.deadline.count()) +
          " ms exceeded " + what + "; worker killed; device completion "
          "state uncertain"};
}

// One successful recv into `destination`, or a classified failure. The
// inbox only ever buffers control lines: payload frames are received
// straight into the caller's buffer, because funnelling megabytes
// through a growing string reallocates and recopies the whole payload
// on every recv.
AneWorker::Wait AneWorker::recv_raw(
    void* destination,
    size_t size,
    size_t& got,
    std::chrono::milliseconds until) {
  got = 0;
  for (;;) {
    auto received = ::recv(channel_, destination, size, MSG_DONTWAIT);
    if (received > 0) {
      got = static_cast<size_t>(received);
      return Wait{true, AneWorkerStatus::Completed, std::string()};
    }
    if (received == 0) {
      // The child closed its end without a terminal frame: reap it and
      // classify the death.
      int status = 0;
      while (::waitpid(child_, &status, 0) < 0 && errno == EINTR) {
      }
      child_ = -1;
      if (WIFSIGNALED(status)) {
        return Wait{
            false, AneWorkerStatus::WorkerDied,
            "resident worker killed by signal " +
                std::to_string(WTERMSIG(status)) +
                "; device completion state uncertain"};
      }
      return Wait{
          false, AneWorkerStatus::WorkerDied,
          "resident worker exited mid-protocol (status " +
              std::to_string(WEXITSTATUS(status)) +
              ") without a terminal frame; device completion state "
              "uncertain"};
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
      auto ready = await_channel(POLLIN, until, "on a resident submit");
      if (!ready.ok) return ready;
      continue;
    }
    return Wait{
        false, AneWorkerStatus::WorkerDied,
        std::string("resident channel read failed: ") + std::strerror(errno) +
            "; device completion state uncertain"};
  }
}

AneWorker::Wait AneWorker::recv_more(std::chrono::milliseconds until) {
  char buffer[4096];
  size_t got = 0;
  auto received = recv_raw(buffer, sizeof buffer, got, until);
  if (!received.ok) return received;
  inbox_.append(buffer, got);
  return received;
}

AneWorker::Wait AneWorker::recv_line(
    std::string& line,
    std::chrono::milliseconds until) {
  for (;;) {
    auto end = inbox_.find('\n', inbox_cursor_);
    if (end != std::string::npos) {
      line = inbox_.substr(inbox_cursor_, end - inbox_cursor_);
      inbox_cursor_ = end + 1;
      if (inbox_cursor_ == inbox_.size()) {
        inbox_.clear();
        inbox_cursor_ = 0;
      }
      return Wait{true, AneWorkerStatus::Completed, std::string()};
    }
    auto more = recv_more(until);
    if (!more.ok) return more;
  }
}

AneWorker::Wait AneWorker::recv_bytes(
    size_t count,
    Buffer& out,
    std::chrono::milliseconds until) {
  out.resize(count);
  size_t filled = std::min(inbox_.size() - inbox_cursor_, count);
  if (filled > 0) {
    std::memcpy(out.data(), inbox_.data() + inbox_cursor_, filled);
    inbox_cursor_ += filled;
    if (inbox_cursor_ == inbox_.size()) {
      inbox_.clear();
      inbox_cursor_ = 0;
    }
  }
  while (filled < count) {
    size_t got = 0;
    auto received =
        recv_raw(out.data() + filled, count - filled, got, until);
    if (!received.ok) return received;
    filled += got;
  }
  return Wait{true, AneWorkerStatus::Completed, std::string()};
}

AneWorker::Wait AneWorker::send_request(
    const void* data,
    size_t size,
    std::chrono::milliseconds until) {
  const char* cursor = static_cast<const char*>(data);
  while (size > 0) {
    auto written =
        ::send(channel_, cursor, size, MSG_NOSIGNAL | MSG_DONTWAIT);
    if (written > 0) {
      cursor += written;
      size -= static_cast<size_t>(written);
      continue;
    }
    if (written < 0 &&
        (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) {
      auto ready = await_channel(
          POLLOUT, until, "while staging a resident submit");
      if (!ready.ok) return ready;
      continue;
    }
    return Wait{
        false, AneWorkerStatus::WorkerDied,
        std::string("resident channel write failed: ") +
            std::strerror(errno) +
            "; device completion state uncertain"};
  }
  return Wait{true, AneWorkerStatus::Completed, std::string()};
}

AneWorkerReport AneWorker::open(const std::vector<AneBundle>& bundles) {
  if (resident()) {
    throw std::invalid_argument("resident ANE session is already open");
  }
  if (bundles.empty()) {
    throw std::invalid_argument(
        "resident ANE session requires at least one bundle");
  }
  if (quarantined()) {
    return AneWorkerReport{
        AneWorkerStatus::QuarantinedRefused, 0, 0,
        std::chrono::milliseconds(0), "quarantined: " + quarantine_reason_,
        std::string()};
  }

  // Every program of every bundle is loaded into one device, so the
  // device keys must be unique across bundles: each bundle's programs
  // are offset by the programs already claimed.
  std::vector<AneBundle> resident;
  resident.reserve(bundles.size());
  size_t base = 0;
  for (const auto& bundle : bundles) {
    AneBundle copy = bundle;
    for (auto& program : copy.programs) {
      program.manifest_index += base;
    }
    base += copy.programs.size();
    resident.push_back(std::move(copy));
  }
  if (base == 0) {
    throw std::invalid_argument(
        "resident ANE session requires at least one program");
  }

  int fds[2];
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
    quarantine_reason_ =
        std::string("socketpair creation failed: ") + std::strerror(errno);
    return AneWorkerReport{
        AneWorkerStatus::WorkerDied, 0, 0, std::chrono::milliseconds(0),
        quarantine_reason_, std::string()};
  }

  const auto started = now_ms();
  pid_t child = ::fork();
  if (child < 0) {
    ::close(fds[0]);
    ::close(fds[1]);
    quarantine_reason_ = std::string("fork failed: ") + std::strerror(errno);
    return AneWorkerReport{
        AneWorkerStatus::WorkerDied, 0, 0, std::chrono::milliseconds(0),
        quarantine_reason_, std::string()};
  }

  if (child == 0) {
    ::close(fds[0]);
    ::signal(SIGABRT, SIG_DFL);
    ::signal(SIGSEGV, SIG_DFL);
    ::signal(SIGINT, SIG_DFL);
    ::signal(SIGTERM, SIG_DFL);
    int code = 0;
    try {
      std::unique_ptr<AneDevice> device = factory_();
      size_t loaded = 0;
      for (const auto& bundle : resident) {
        for (const auto& program : bundle.programs) {
          device->load(program);
          ++loaded;
        }
      }
      send_frame(fds[1], "loaded " + std::to_string(loaded) + "\n");
      code = resident_child_loop(
          fds[1], *device, resident, options_.iterations, options_.shm_in,
          options_.shm_out);
    } catch (const std::exception& error) {
      send_frame(
          fds[1], std::string(kTokenFailed) + one_line(error.what()) + "\n");
      code = 1;
    }
    ::close(fds[1]);
    ::_exit(code);
  }

  ::close(fds[1]);
  child_ = child;
  channel_ = fds[0];
  resident_programs_ = base;

  const auto until = now_ms() + options_.deadline;
  std::string line;
  auto wait = recv_line(line, until);
  if (!wait.ok) {
    auto out = end_session(wait);
    out.elapsed = now_ms() - started;
    return out;
  }
  if (line.compare(0, std::strlen(kTokenFailed), kTokenFailed) == 0) {
    Wait failure{
        false, AneWorkerStatus::DeviceFailed,
        line.substr(std::strlen(kTokenFailed))};
    auto out = end_session(failure);
    out.elapsed = now_ms() - started;
    return out;
  }
  if (line.compare(0, 7, "loaded ") != 0) {
    Wait failure{
        false, AneWorkerStatus::WorkerDied,
        "resident worker sent '" + one_line(line) +
            "' instead of a load report; device completion state uncertain"};
    auto out = end_session(failure);
    out.elapsed = now_ms() - started;
    return out;
  }

  AneWorkerReport out;
  out.status = AneWorkerStatus::Completed;
  out.elapsed = now_ms() - started;
  out.detail = "resident worker loaded " + std::to_string(base) +
               " program(s) from " + std::to_string(resident.size()) +
               " bundle(s)";
  return out;
}

AneWorkerReport AneWorker::submit(
    size_t bundle,
    const std::map<std::string, Buffer>& inputs,
    std::map<std::string, Buffer>* outputs) {
  if (quarantined()) {
    return AneWorkerReport{
        AneWorkerStatus::QuarantinedRefused, 0, 0,
        std::chrono::milliseconds(0), "quarantined: " + quarantine_reason_,
        std::string()};
  }
  if (!resident()) {
    throw std::invalid_argument("no resident ANE session is open");
  }

  const auto started = now_ms();
  const auto until =
      batch_until_.count() != 0 ? batch_until_ : started + options_.deadline;
  if (batch_until_.count() != 0) {
    ++batch_rounds_;
  }

  std::string header = "submit " + std::to_string(bundle) + "\n";
  auto wait = send_request(header.data(), header.size(), until);
  for (auto entry = inputs.begin(); wait.ok && entry != inputs.end();
       ++entry) {
    std::string frame = "in " + entry->first + " " +
                        std::to_string(entry->second.size()) + "\n";
    wait = send_request(frame.data(), frame.size(), until);
    if (!wait.ok) break;
    wait = send_request(
        entry->second.data(), entry->second.size(), until);
  }
  if (wait.ok) {
    wait = send_request(kRequestRun, std::strlen(kRequestRun), until);
  }
  if (wait.ok) {
    wait = send_request("\n", 1, until);
  }
  if (!wait.ok) {
    auto out = end_session(wait);
    out.elapsed = now_ms() - started;
    return out;
  }

  AneWorkerReport out;
  std::map<std::string, Buffer> produced;
  for (;;) {
    std::string line;
    wait = recv_line(line, until);
    if (!wait.ok) {
      auto failed = end_session(wait);
      failed.iterations = out.iterations;
      failed.elapsed = now_ms() - started;
      return failed;
    }
    if (line == "iter") {
      ++out.iterations;
      continue;
    }
    if (line.compare(0, 5, "perf ") == 0) {
      out.perf = line.substr(5);
      continue;
    }
    if (line + "\n" == kTokenDone) {
      break;
    }
    if (line.compare(0, std::strlen(kTokenFailed), kTokenFailed) == 0) {
      Wait failure{
          false, AneWorkerStatus::DeviceFailed,
          line.substr(std::strlen(kTokenFailed))};
      auto failed = end_session(failure);
      failed.iterations = out.iterations;
      failed.elapsed = now_ms() - started;
      return failed;
    }
    std::string name;
    size_t length = 0;
    if (parse_payload_header(line, "out", name, length)) {
      Buffer payload;
      wait = recv_bytes(length, payload, until);
      if (!wait.ok) {
        auto failed = end_session(wait);
        failed.iterations = out.iterations;
        failed.elapsed = now_ms() - started;
        return failed;
      }
      produced[name] = std::move(payload);
      continue;
    }
    Wait failure{
        false, AneWorkerStatus::WorkerDied,
        "resident worker sent unknown frame '" + one_line(line) +
            "'; device completion state uncertain"};
    auto failed = end_session(failure);
    failed.iterations = out.iterations;
    failed.elapsed = now_ms() - started;
    return failed;
  }

  out.status = AneWorkerStatus::Completed;
  out.elapsed = now_ms() - started;
  out.detail = "resident worker completed " + std::to_string(out.iterations) +
               " iteration(s) on resident programs";
  if (outputs != nullptr) {
    *outputs = std::move(produced);
  }
  return out;
}

// Shared-memory submit: control framing on the socketpair, payload
// bytes in the regions. Same safety model as submit(): one bounded
// request/response pair, deadline or failure ends the session.
AneWorkerReport AneWorker::submit_shm(
    size_t bundle,
    const std::map<std::string, AneShmSpan>& inputs,
    const std::map<std::string, size_t>& output_offsets,
    std::map<std::string, AneShmSpan>* outputs) {
  if (quarantined()) {
    return AneWorkerReport{
        AneWorkerStatus::QuarantinedRefused, 0, 0,
        std::chrono::milliseconds(0), "quarantined: " + quarantine_reason_,
        std::string()};
  }
  if (!resident()) {
    throw std::invalid_argument("no resident ANE session is open");
  }
  if (options_.shm_in.base == nullptr || options_.shm_out.base == nullptr) {
    throw std::invalid_argument(
        "submit_shm requires configured shm regions");
  }

  const auto started = now_ms();
  const auto until =
      batch_until_.count() != 0 ? batch_until_ : started + options_.deadline;
  if (batch_until_.count() != 0) {
    ++batch_rounds_;
  }

  std::string header = "submit " + std::to_string(bundle) + "\n";
  auto wait = send_request(header.data(), header.size(), until);
  for (const auto& entry : inputs) {
    if (entry.second.offset > options_.shm_in.size ||
        entry.second.size > options_.shm_in.size - entry.second.offset) {
      throw std::invalid_argument(
          "input span for '" + entry.first + "' is outside shm_in");
    }
    std::string frame = "inshm " + entry.first + " " +
                        std::to_string(entry.second.offset) + " " +
                        std::to_string(entry.second.size) + "\n";
    wait = send_request(frame.data(), frame.size(), until);
    if (!wait.ok) break;
  }
  if (wait.ok) {
    for (const auto& entry : output_offsets) {
      if (entry.second > options_.shm_out.size) {
        throw std::invalid_argument(
            "output offset for '" + entry.first + "' is outside shm_out");
      }
      std::string frame = "shout " + entry.first + " " +
                          std::to_string(entry.second) + "\n";
      wait = send_request(frame.data(), frame.size(), until);
      if (!wait.ok) break;
    }
  }
  if (wait.ok) {
    wait = send_request(kRequestRun, std::strlen(kRequestRun), until);
  }
  if (wait.ok) {
    wait = send_request("\n", 1, until);
  }
  if (!wait.ok) {
    auto out = end_session(wait);
    out.elapsed = now_ms() - started;
    return out;
  }

  AneWorkerReport out;
  std::map<std::string, AneShmSpan> produced;
  for (;;) {
    std::string line;
    wait = recv_line(line, until);
    if (!wait.ok) {
      auto failed = end_session(wait);
      failed.iterations = out.iterations;
      failed.elapsed = now_ms() - started;
      return failed;
    }
    if (line == "iter") {
      ++out.iterations;
      continue;
    }
    if (line.compare(0, 5, "perf ") == 0) {
      out.perf = line.substr(5);
      continue;
    }
    if (line + "\n" == kTokenDone) {
      break;
    }
    if (line.compare(0, std::strlen(kTokenFailed), kTokenFailed) == 0) {
      Wait failure{
          false, AneWorkerStatus::DeviceFailed,
          line.substr(std::strlen(kTokenFailed))};
      auto failed = end_session(failure);
      failed.iterations = out.iterations;
      failed.elapsed = now_ms() - started;
      return failed;
    }
    std::string name;
    size_t offset = 0;
    size_t length = 0;
    if (parse_shm_header(line, kTokenShmOut, name, offset, length) ||
        parse_payload_header(line, "out", name, length)) {
      // "shmout" carries offset+length; a plain "out" on the shm path
      // would mean the child had no sink for the tensor -- that is a
      // slot-layout bug, not a payload, so refuse it.
      if (line.compare(0, std::strlen(kTokenShmOut), kTokenShmOut) != 0) {
        Wait failure{
            false, AneWorkerStatus::WorkerDied,
            "resident worker sent inline output '" + name +
                "' on the shm path; device completion state uncertain"};
        auto failed = end_session(failure);
        failed.iterations = out.iterations;
        failed.elapsed = now_ms() - started;
        return failed;
      }
      if (offset > options_.shm_out.size ||
          length > options_.shm_out.size - offset) {
        Wait failure{
            false, AneWorkerStatus::WorkerDied,
            "resident worker reported an output span outside shm_out"};
        auto failed = end_session(failure);
        failed.iterations = out.iterations;
        failed.elapsed = now_ms() - started;
        return failed;
      }
      produced[name] = AneShmSpan{offset, length};
      continue;
    }
    Wait failure{
        false, AneWorkerStatus::WorkerDied,
        "resident worker sent unknown frame '" + one_line(line) +
            "'; device completion state uncertain"};
    auto failed = end_session(failure);
    failed.iterations = out.iterations;
    failed.elapsed = now_ms() - started;
    return failed;
  }

  out.status = AneWorkerStatus::Completed;
  out.elapsed = now_ms() - started;
  out.detail = "resident worker completed " + std::to_string(out.iterations) +
               " iteration(s) on resident programs over shm";
  if (outputs != nullptr) {
    *outputs = std::move(produced);
  }
  return out;
}

// Opens a batch scope: one absolute deadline bounds every submit until
// close_batch(). The bounded unit is the batch, so the caller can turn
// a whole island pass of per-layer submits into one deadline-bounded
// unit without weakening anything else: the child stays private, a
// deadline miss or a failed round still quarantines and ends the
// session, and nothing is retried.
AneWorkerReport AneWorker::open_batch(std::chrono::milliseconds deadline) {
  if (deadline.count() <= 0) {
    throw std::invalid_argument("ANE worker batch deadline must be positive");
  }
  if (!resident()) {
    throw std::invalid_argument("no resident ANE session is open");
  }
  if (batching()) {
    throw std::invalid_argument("ANE worker batch scope is already open");
  }
  AneWorkerReport out;
  if (quarantined()) {
    out.status = AneWorkerStatus::QuarantinedRefused;
    out.detail = "quarantined: " + quarantine_reason_;
    return out;
  }
  batch_until_ = now_ms() + deadline;
  batch_rounds_ = 0;
  out.status = AneWorkerStatus::Completed;
  out.detail = "batch opened deadline_ms=" + std::to_string(deadline.count());
  return out;
}

AneWorkerReport AneWorker::close_batch() {
  if (!batching()) {
    throw std::invalid_argument("no ANE worker batch scope is open");
  }
  AneWorkerReport out;
  if (quarantined()) {
    out.status = AneWorkerStatus::QuarantinedRefused;
    out.detail = "quarantined: " + quarantine_reason_;
    batch_until_ = std::chrono::milliseconds(0);
    return out;
  }
  batch_until_ = std::chrono::milliseconds(0);
  out.status = AneWorkerStatus::Completed;
  out.iterations = batch_rounds_;
  out.detail = "batch closed rounds=" + std::to_string(batch_rounds_);
  return out;
}

AneWorkerReport AneWorker::close() {
  if (!resident()) {
    throw std::invalid_argument("no resident ANE session is open");
  }
  const auto started = now_ms();
  const auto until = started + options_.deadline;
  const size_t programs = resident_programs_;

  std::string request = std::string(kRequestClose) + "\n";
  auto wait = send_request(request.data(), request.size(), until);
  if (wait.ok) {
    std::string line;
    wait = recv_line(line, until);
    if (wait.ok && line + "\n" != kTokenReleased) {
      wait = Wait{
          false, AneWorkerStatus::WorkerDied,
          "resident worker sent '" + one_line(line) +
              "' instead of a release report; device completion state "
              "uncertain"};
    }
  }
  if (!wait.ok) {
    auto out = end_session(wait);
    out.elapsed = now_ms() - started;
    return out;
  }

  int status = 0;
  while (::waitpid(child_, &status, 0) < 0 && errno == EINTR) {
  }
  child_ = -1;
  teardown();

  AneWorkerReport out;
  out.elapsed = now_ms() - started;
  if (WIFSIGNALED(status) || WEXITSTATUS(status) != 0) {
    quarantine_reason_ =
        "resident worker released its programs but exited abnormally; "
        "device completion state uncertain";
    out.status = AneWorkerStatus::WorkerDied;
    out.detail = quarantine_reason_;
    return out;
  }
  out.status = AneWorkerStatus::Completed;
  out.released_programs = static_cast<int>(programs);
  out.detail = "resident worker released " + std::to_string(programs) +
               " program(s) and exited";
  return out;
}

} // namespace mlx::core::omarchy::ane
