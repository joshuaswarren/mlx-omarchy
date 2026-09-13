// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT
#pragma once

// Bounded ANE worker boundary (plan sections 22-27).
//
// A worker run is a supervised child process: the child loads the
// validated bundle programs onto a device, executes the dispatch plan,
// reads results, releases the programs, and reports through a pipe. The
// parent enforces a wall-clock deadline; a deadline expiry or an
// abnormal child death leaves the device completion state uncertain, so
// the worker enters quarantine and refuses every later submission.
//
// This is not an inference server (section 25): one run() is one bounded
// execution of one bundle, nothing persists between runs except the
// quarantine state, and no listening socket or external request surface
// exists.

#include "mlx/backend/omarchy/ane/bundle.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlx::core::omarchy::ane {

// Device seam. Bindings and validated programs are passed through so a
// backend can map tensors onto device channels; the generic worker never
// interprets channel indices itself.
struct AneDevice {
  virtual ~AneDevice() = default;
  virtual std::string describe() const = 0;
  virtual void load(const AneValidatedProgram& program) = 0;
  virtual void send(
      size_t manifest_index,
      uint32_t channel,
      const AneProgramBinding& binding,
      const uint8_t* data,
      size_t size) = 0;
  virtual void exec(size_t manifest_index) = 0;
  virtual void read(
      size_t manifest_index,
      uint32_t channel,
      const AneProgramBinding& binding,
      uint8_t* out,
      size_t size) = 0;
  virtual void release() = 0;
};

struct AneDeviceError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

enum class AneWorkerStatus {
  Completed,
  DeviceFailed,      // clean child failure with a named reason
  DeadlineExceeded,  // killed at the deadline; device state uncertain
  WorkerDied,        // signal death or mid-protocol exit; uncertain
  QuarantinedRefused // a prior run quarantined this worker
};

struct AneWorkerReport {
  AneWorkerStatus status{AneWorkerStatus::Completed};
  int iterations{0};
  int released_programs{0};
  std::chrono::milliseconds elapsed{0};
  std::string detail;
  // Internal: raw child report bytes carried out of supervise() so run()
  // can extract output payloads. Empty in every report handed to callers.
  std::string detail_extra;
};

struct AneWorkerOptions {
  std::chrono::milliseconds deadline{std::chrono::milliseconds(2000)};
  int iterations{1};
};

class AneWorker {
 public:
  using DeviceFactory = std::function<std::unique_ptr<AneDevice>()>;
  using Buffer = std::vector<uint8_t>;

  // Throws std::invalid_argument on a non-positive deadline or
  // iteration count: an unbounded worker is a contract violation, not a
  // configuration.
  AneWorker(DeviceFactory factory, AneWorkerOptions options);

  // Executes one bounded run. `inputs` maps manifest input tensor names
  // to payload bytes; on Completed, `outputs` receives the manifest
  // output tensors of the final iteration.
  AneWorkerReport run(
      const AneBundle& bundle,
      const std::map<std::string, Buffer>& inputs,
      std::map<std::string, Buffer>* outputs = nullptr);

  bool quarantined() const {
    return !quarantine_reason_.empty();
  }
  const std::string& quarantine_reason() const {
    return quarantine_reason_;
  }

 private:
  AneWorkerReport supervise(pid_t child, int report_fd);

  DeviceFactory factory_;
  AneWorkerOptions options_;
  std::string quarantine_reason_;
};

} // namespace mlx::core::omarchy::ane
