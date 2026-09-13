// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Host lifecycle tests for the bounded ANE worker. The device is a
// scripted fake; no ANE hardware, no libane, no GPU lock is touched.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "mlx/backend/omarchy/ane/worker.h"

#include <unistd.h>

#include <chrono>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace mlx::core::omarchy::ane;

namespace {

// Behavior is selected through statics before fork(); the child inherits
// the selection, so no cross-process coordination is needed.
struct FakeDevice : AneDevice {
  enum class Behavior {
    Succeed,
    HangPastDeadline,
    CrashInExec,
    RefuseLoad,
    FailExecNamed,
  };
  static Behavior behavior;
  static inline const char* kFailReason = "fake device refused exec";

  std::string describe() const override {
    return "fake-device";
  }

  void load(const AneValidatedProgram& program) override {
    if (behavior == Behavior::RefuseLoad) {
      throw AneDeviceError(
          "fake device refused to load " + program.anec.string());
    }
  }

  void send(
      size_t /*manifest_index*/,
      uint32_t /*channel*/,
      const AneProgramBinding& binding,
      const uint8_t* /*data*/,
      size_t size) override {
    sent_.push_back({binding.tensor, size});
  }

  void exec(size_t manifest_index) override {
    if (behavior == Behavior::HangPastDeadline) {
      // Sleep far past any test deadline; the supervisor must kill us.
      ::sleep(30);
      return;
    }
    if (behavior == Behavior::CrashInExec) {
      ::abort(); // a real signal death (SIGABRT), not a clean _exit
    }
    if (behavior == Behavior::FailExecNamed) {
      throw AneDeviceError(kFailReason);
    }
    execs_.push_back(manifest_index);
  }

  void read(
      size_t /*manifest_index*/,
      uint32_t /*channel*/,
      const AneProgramBinding& binding,
      uint8_t* out,
      size_t size) override {
    // Deterministic payload: fill with the tensor name's bytes so the
    // parent can prove output bytes really crossed the pipe.
    for (size_t i = 0; i < size; ++i) {
      out[i] = static_cast<uint8_t>(binding.tensor.empty()
                                        ? '?'
                                        : binding.tensor[i % binding.tensor.size()]);
    }
  }

  void release() override {
    released_ = true;
  }

  struct Sent {
    std::string tensor;
    size_t size;
  };
  std::vector<Sent> sent_;
  std::vector<size_t> execs_;
  bool released_{false};
};

FakeDevice::Behavior FakeDevice::behavior = FakeDevice::Behavior::Succeed;

AneBundle toy_bundle() {
  AneBundle bundle;
  bundle.manifest.manifest_version = 4;
  bundle.manifest.name = "toy-add";
  bundle.manifest.graph_hash = "deadbeef";
  bundle.manifest.driver_abi_major = 1;

  AneTensor a;
  a.name = "a";
  a.dtype = "fp16";
  a.shape = {1, 64, 1, 1};
  a.byte_size = 128;
  bundle.manifest.inputs = {a};
  AneTensor b = a;
  b.name = "b";
  bundle.manifest.inputs.push_back(b);

  AneTensor y;
  y.name = "y";
  y.dtype = "fp16";
  y.shape = {1, 64, 1, 1};
  y.byte_size = 128;
  bundle.manifest.logical_results = {};
  bundle.manifest.outputs = {y};

  AneProgram program;
  program.payload = "program-0.anec";
  program.operation = "add";
  AneProgramBinding in;
  in.tensor = "a";
  in.logical_bytes = 128;
  in.allocation_bytes = 128;
  program.inputs = {in};
  AneProgramBinding in_b = in;
  in_b.tensor = "b";
  program.inputs.push_back(in_b);
  AneProgramBinding out = in;
  out.tensor = "y";
  program.outputs = {out};
  bundle.manifest.programs = {program};
  bundle.manifest.dispatch_plan = {0};

  AneValidatedProgram validated;
  validated.manifest_index = 0;
  validated.anec = "/dev/null";
  bundle.programs = {validated};
  return bundle;
}

AneWorker::Buffer input_bytes(char fill) {
  return AneWorker::Buffer(128, static_cast<uint8_t>(fill));
}

} // namespace

TEST_CASE("a completed run reports iterations, outputs, and release") {
  FakeDevice::behavior = FakeDevice::Behavior::Succeed;
  AneWorkerOptions options;
  options.deadline = std::chrono::milliseconds(4000);
  options.iterations = 2;
  AneWorker worker([] { return std::unique_ptr<AneDevice>(new FakeDevice); },
                   options);

  std::map<std::string, AneWorker::Buffer> inputs;
  inputs["a"] = input_bytes(0x11);
  inputs["b"] = input_bytes(0x22);
  std::map<std::string, AneWorker::Buffer> outputs;
  auto report = worker.run(toy_bundle(), inputs, &outputs);

  REQUIRE(report.status == AneWorkerStatus::Completed);
  CHECK(report.iterations == 2);
  CHECK(report.released_programs == 1);
  CHECK(!worker.quarantined());
  CHECK(report.elapsed < options.deadline);
  REQUIRE(outputs.count("y") == 1);
  CHECK(outputs["y"].size() == 128);
  std::string expected(128, 'y');
  CHECK(std::memcmp(outputs["y"].data(), expected.data(), 128) == 0);
  // detail_extra never leaks to callers.
  CHECK(report.detail_extra.empty());
}

TEST_CASE("a hanging device is killed at the deadline and quarantined") {
  FakeDevice::behavior = FakeDevice::Behavior::HangPastDeadline;
  AneWorkerOptions options;
  options.deadline = std::chrono::milliseconds(250);
  options.iterations = 1;
  AneWorker worker([] { return std::unique_ptr<AneDevice>(new FakeDevice); },
                   options);

  auto report = worker.run(toy_bundle(), {{"a", input_bytes(1)}, {"b", input_bytes(2)}});
  REQUIRE(report.status == AneWorkerStatus::DeadlineExceeded);
  CHECK(report.elapsed >= std::chrono::milliseconds(200));
  CHECK(worker.quarantined());
  CHECK(worker.quarantine_reason().find("deadline") != std::string::npos);
  CHECK(
      worker.quarantine_reason().find("uncertain") != std::string::npos);

  // The quarantined worker refuses without spawning anything.
  auto refused = worker.run(toy_bundle(), {});
  REQUIRE(refused.status == AneWorkerStatus::QuarantinedRefused);
  CHECK(refused.iterations == 0);
  CHECK(refused.detail.find("quarantined") == 0);
}

TEST_CASE("a worker that dies in exec is quarantined with the signal") {
  FakeDevice::behavior = FakeDevice::Behavior::CrashInExec;
  AneWorkerOptions options;
  options.deadline = std::chrono::milliseconds(4000);
  AneWorker worker([] { return std::unique_ptr<AneDevice>(new FakeDevice); },
                   options);

  auto report = worker.run(toy_bundle(), {{"a", input_bytes(1)}, {"b", input_bytes(2)}});
  REQUIRE(report.status == AneWorkerStatus::WorkerDied);
  CHECK(worker.quarantined());
  CHECK(worker.quarantine_reason().find("signal 6") != std::string::npos);
}

TEST_CASE("a named device failure is clean and does not quarantine") {
  std::map<std::string, AneWorker::Buffer> inputs = {
      {"a", input_bytes(1)}, {"b", input_bytes(2)}};
  FakeDevice::behavior = FakeDevice::Behavior::RefuseLoad;
  AneWorkerOptions options;
  options.deadline = std::chrono::milliseconds(4000);
  AneWorker worker([] { return std::unique_ptr<AneDevice>(new FakeDevice); },
                   options);

  auto report = worker.run(toy_bundle(), inputs);
  REQUIRE(report.status == AneWorkerStatus::DeviceFailed);
  CHECK(!worker.quarantined());
  CHECK(report.detail.find("refused to load") != std::string::npos);

  FakeDevice::behavior = FakeDevice::Behavior::FailExecNamed;
  report = worker.run(toy_bundle(), inputs);
  REQUIRE(report.status == AneWorkerStatus::DeviceFailed);
  CHECK(!worker.quarantined());
  CHECK(report.detail == FakeDevice::kFailReason);

  // A clean-failed worker stays usable.
  FakeDevice::behavior = FakeDevice::Behavior::Succeed;
  report = worker.run(toy_bundle(), inputs);
  REQUIRE(report.status == AneWorkerStatus::Completed);
}

TEST_CASE("missing staging input is a named failure, not a crash") {
  FakeDevice::behavior = FakeDevice::Behavior::Succeed;
  AneWorkerOptions options;
  options.deadline = std::chrono::milliseconds(4000);
  AneWorker worker([] { return std::unique_ptr<AneDevice>(new FakeDevice); },
                   options);

  std::map<std::string, AneWorker::Buffer> inputs;
  inputs["a"] = input_bytes(1);
  // "b" intentionally missing.
  auto report = worker.run(toy_bundle(), inputs);
  REQUIRE(report.status == AneWorkerStatus::DeviceFailed);
  CHECK(report.detail.find("'b' has no staged value") != std::string::npos);
  CHECK(!worker.quarantined());
}

TEST_CASE("worker construction rejects unbounded configurations") {
  auto factory = [] {
    return std::unique_ptr<AneDevice>(new FakeDevice);
  };
  AneWorkerOptions bad_deadline;
  bad_deadline.deadline = std::chrono::milliseconds(0);
  CHECK_THROWS_AS(AneWorker(factory, bad_deadline), std::invalid_argument);

  AneWorkerOptions bad_iterations;
  bad_iterations.iterations = 0;
  CHECK_THROWS_AS(AneWorker(factory, bad_iterations), std::invalid_argument);
}
