// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Focused runtime tests for the Omarchy Vulkan backend slice:
//   1. discovery classification (supported / unsupported hardware),
//   2. device selection through the MLX device model,
//   3. a real Vulkan buffer round trip through the encoder,
//   4. event lifetime, cross-stream ordering, handler-only commits, and
//      the completion-wait drain join when the dispatcher thread wins the
//      pending_ race,
//   5. temporary ownership across asynchronous (handler-free) completion,
//   6. device_info reference stability under concurrent access,
//   7. measured concurrency of independent streams on a hardware GPU,
//   8. fresh-process device reopen and bounded failed-submit errors,
//      proven in process-isolated child runs,
// 10. safe command-buffer reuse across many asynchronous commits.
// 11. in-order-stream contract: a small eager output crosses deep
//     submit boundaries into a later consumer dispatch.
//
// The suite needs MLX_BUILD_OMARCHY=ON and compiles against Vulkan 1.3
// headers (the Honeykrisp driver id is pinned in device.h). Round-trip and
// device-selection tests need a qualifying Vulkan device; on non-Omarchy
// machines set MLX_OMARCHY_ALLOW_NON_APPLE=1 to exercise them against a
// development driver (reported as dev-only by mlx-omarchy-info).

#define DOCTEST_CONFIG_IMPLEMENT
#include <sys/wait.h>
#include <unistd.h>
#include <chrono>
#include <cmath>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include "doctest/doctest.h"
#include "mlx/backend/cpu/device_info.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/backend/omarchy/vulkan.h"
#include "mlx/device.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

using namespace mlx::core;

namespace {

omarchy::DeviceSupport classify(
    const char* name,
    uint32_t vendor_id,
    uint32_t api_version,
    int32_t driver_id,
    bool allow_non_apple) {
  VkPhysicalDeviceProperties props{};
  std::strncpy(props.deviceName, name, VK_MAX_PHYSICAL_DEVICE_NAME_SIZE - 1);
  props.vendorID = vendor_id;
  props.deviceID = 0x6001;
  props.apiVersion = api_version;
  return omarchy::classify_physical_device(props, driver_id, allow_non_apple);
}

void skip(const char* reason) {
  std::cout << "Skipping: " << reason << "\n";
}

// --- Process-isolated failure-mode scenarios ------------------------------
//
// The parent test re-execs this same binary with MLX_OMARCHY_TEST_CHILD set.
// The child runs one scenario from a fresh process (fresh Vulkan discovery
// and a fresh VkDevice) and exits with a code the parent asserts on. This
// isolates wedged queues and device teardown from the test process, per the
// hardware-safety rules in AGENTS.md.

int run_child_scenario(const std::string& mode) {
  if (!gpu::is_available()) {
    return 77;
  }
  if (mode == "reopen") {
    // Fresh process: discovery, VkDevice creation, work, and full teardown.
    auto& dev = omarchy::device(0);
    if (dev.handle() == VK_NULL_HANDLE || dev.queue() == VK_NULL_HANDLE) {
      return 1;
    }
    Stream s = new_stream(Device::gpu);
    auto& enc = omarchy::get_command_encoder(s);
    auto buf = omarchy::allocator().malloc(4096);
    auto* p = static_cast<omarchy::VulkanBuffer*>(buf.ptr());
    enc.fill_buffer(p->buffer, 0x2a, 4096);
    enc.commit();
    enc.synchronize();
    auto* words = static_cast<uint32_t*>(p->data);
    for (size_t i = 0; i < 4096 / sizeof(uint32_t); ++i) {
      if (words[i] != 0x2a) {
        omarchy::allocator().free(buf);
        return 2;
      }
    }
    omarchy::allocator().free(buf);
    std::cout << "[child/reopen] fresh process device reopen ok\n";
    return 0;
  }
  if (mode == "bounded_submit") {
    // Queue a wait that can never be satisfied, then block on completion.
    // The submit must return control with a typed Omarchy error instead of
    // hanging forever (plan R16). No CPU rescue is attempted for a hung GPU.
    Stream a = new_stream(Device::gpu);
    Stream b = new_stream(Device::gpu);
    Event e{a};
    e.set_value(42);
    auto& enc = omarchy::get_command_encoder(b);
    e.wait(b);
    auto buf = omarchy::allocator().malloc(4096);
    auto* p = static_cast<omarchy::VulkanBuffer*>(buf.ptr());
    enc.fill_buffer(p->buffer, 1, 4096);
    enc.commit();
    auto t0 = std::chrono::steady_clock::now();
    try {
      enc.synchronize();
      std::cout << "[child/bounded_submit] ERROR: hung submit never threw\n";
      omarchy::allocator().free(buf);
      return 3;
    } catch (const std::exception& ex) {
      auto elapsed = std::chrono::steady_clock::now() - t0;
      std::string msg = ex.what();
      std::cout << "[child/bounded_submit] typed error: " << msg << "\n";
      if (msg.find("[omarchy]") == std::string::npos) {
        omarchy::allocator().free(buf);
        return 4;
      }
      if (msg.find("no CPU fallback") == std::string::npos) {
        omarchy::allocator().free(buf);
        return 5;
      }
      if (elapsed > std::chrono::seconds(30)) {
        omarchy::allocator().free(buf);
        return 6;
      }
      // _Exit skips destruction on purpose: the queue keeps the wedged
      // submission, so teardown would block. The parent bounds this child.
      std::_Exit(0);
    }
  }
  return 126;
}

struct ChildRun {
  int code;
  bool timed_out;
  bool signaled;
};

ChildRun run_child(const char* mode, int timeout_s) {
  char self[4096];
  ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
  if (n <= 0) {
    return {127, false, false};
  }
  self[n] = '\0';
  pid_t pid = ::fork();
  if (pid == 0) {
    ::setenv("MLX_OMARCHY_TEST_CHILD", mode, 1);
    ::execl(self, self, static_cast<char*>(nullptr));
    ::_exit(127);
  }
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
  int status = 0;
  for (;;) {
    pid_t done = ::waitpid(pid, &status, WNOHANG);
    if (done == pid) {
      break;
    }
    if (std::chrono::steady_clock::now() > deadline) {
      ::kill(pid, SIGKILL);
      ::waitpid(pid, &status, 0);
      return {-1, true, false};
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  if (WIFEXITED(status)) {
    return {WEXITSTATUS(status), false, false};
  }
  return {-2, false, true};
}

} // namespace

int main(int argc, char** argv) {
  if (const char* mode = std::getenv("MLX_OMARCHY_TEST_CHILD")) {
    return run_child_scenario(mode);
  }
  doctest::Context ctx;
  ctx.applyCommandLine(argc, argv);
  return ctx.run();
}

TEST_CASE("discovery classifies supported and unsupported hardware") {
  // kMesaHoneykrispDriverId is pinned in device.h so the suite compiles
  // against Vulkan 1.3 headers that predate the enum entry.
  constexpr auto honeykrisp_id = omarchy::kMesaHoneykrispDriverId;

  SUBCASE("Apple GPU with Honeykrisp driver is supported") {
    auto s = classify(
        "Apple M1 (G13G B1)", 0x106b, VK_API_VERSION_1_3, honeykrisp_id, false);
    CHECK(s.supported);
    CHECK_FALSE(s.non_apple_dev);
    CHECK(s.reason.empty());
  }

  SUBCASE("M1 target is supported by driver identity alone") {
    // Real M1 receipt: vendor 0x10005, name "Apple M1",
    // driverID MESA_HONEYKRISP. Driver id is the authoritative signal.
    auto s =
        classify("Apple M1", 0x10005, VK_API_VERSION_1_3, honeykrisp_id, false);
    CHECK(s.supported);
    CHECK_FALSE(s.non_apple_dev);
  }

  SUBCASE("Apple vendor remains an alternate signal") {
    auto s = classify(
        "Apple M1 (G13G B1)",
        0x106b,
        VK_MAKE_API_VERSION(0, 1, 4, 0),
        -1,
        false);
    CHECK(s.supported);
  }
  SUBCASE("llvmpipe is refused without the development override") {
    auto s = classify(
        "llvmpipe (LLVM 19.1.0, 256 bits)",
        0x10005,
        VK_API_VERSION_1_3,
        VK_DRIVER_ID_MESA_LLVMPIPE,
        false);
    CHECK_FALSE(s.supported);
    CHECK(s.reason.find("MLX_OMARCHY_ALLOW_NON_APPLE") != std::string::npos);
  }

  SUBCASE("llvmpipe is accepted only as a development device") {
    auto s = classify(
        "llvmpipe (LLVM 19.1.0, 256 bits)",
        0x10005,
        VK_API_VERSION_1_3,
        VK_DRIVER_ID_MESA_LLVMPIPE,
        true);
    CHECK(s.supported);
    CHECK(s.non_apple_dev);
  }

  SUBCASE("devices below Vulkan 1.3 are refused") {
    auto s = classify(
        "Apple M1 (G13G B1)", 0x106b, VK_API_VERSION_1_2, honeykrisp_id, false);
    CHECK_FALSE(s.supported);
    CHECK(s.reason.find("Vulkan 1.3") != std::string::npos);
  }

  SUBCASE("non-Apple 1.3 device without override is refused") {
    auto s = classify(
        "NVIDIA GeForce RTX 4090", 0x10de, VK_API_VERSION_1_3, -1, false);
    CHECK_FALSE(s.supported);
  }
}

TEST_CASE("gpu is unavailable reports a reason and refuses selection") {
  if (gpu::is_available()) {
    skip(
        "a qualifying GPU is present; the refusal path is exercised by"
        " the classification tests and this test is meaningless there.");
    return;
  }
  CHECK(omarchy::device_count() == 0);
  CHECK_FALSE(omarchy::init_error().empty());
  CHECK_THROWS(omarchy::device(0));
  CHECK_THROWS(set_default_device(Device::gpu));
  CHECK_THROWS(new_stream(Device::gpu));
}

TEST_CASE("mx.gpu selects the Omarchy device") {
  if (!gpu::is_available()) {
    skip(
        "no qualifying Vulkan device (set MLX_OMARCHY_ALLOW_NON_APPLE=1 on"
        " a development machine).");
    return;
  }

  CHECK(gpu::device_count() >= 1);
  set_default_device(Device::gpu);
  CHECK(default_device() == Device::gpu);
  CHECK(is_available(Device::gpu));

  const auto& info = gpu::device_info(0);
  auto name_it = info.find("device_name");
  REQUIRE(name_it != info.end());
  auto name = std::get<std::string>(name_it->second);
  CHECK_FALSE(name.empty());

  auto& dev = omarchy::device(0);
  CHECK(dev.handle() != VK_NULL_HANDLE);
  CHECK(dev.queue() != VK_NULL_HANDLE);
  std::cout << "[receipt] compute queue families expose "
            << dev.capabilities().queue_count << " queue(s)\n";

  // Stream creation routes through gpu::new_stream and allocates an encoder.
  Stream s = new_stream(Device::gpu);
  auto& encoder = omarchy::get_command_encoder(s);
  CHECK_FALSE(encoder.needs_commit());
  encoder.commit(); // no-op with nothing pending
}

TEST_CASE("allocator tracks buffers and reuses the cache") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  auto& alloc = omarchy::allocator();
  size_t before = alloc.get_active_memory();

  auto* a = static_cast<omarchy::VulkanBuffer*>(alloc.malloc(4096).ptr());
  auto* b = static_cast<omarchy::VulkanBuffer*>(alloc.malloc(4096).ptr());
  REQUIRE(a->data != nullptr);
  REQUIRE(b->data != nullptr);
  CHECK(alloc.get_active_memory() == before + 2 * 4096);
  alloc.free(allocator::Buffer{a});
  alloc.free(allocator::Buffer{b});
  CHECK(alloc.get_active_memory() == before);

  // A same-size malloc after frees must succeed and keep accounting exact
  // (the page came from cache or the device).
  auto* c = static_cast<omarchy::VulkanBuffer*>(alloc.malloc(4096).ptr());
  REQUIRE(c->data != nullptr);
  CHECK(alloc.get_active_memory() == before + 4096);
  alloc.free(allocator::Buffer{c});
  CHECK(alloc.get_active_memory() == before);
}

TEST_CASE("buffer round trip through the Vulkan encoder") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = new_stream(Device::gpu);
  auto& encoder = omarchy::get_command_encoder(s);

  constexpr size_t kBytes = 1 << 16;
  auto src = omarchy::allocator().malloc(kBytes);
  auto dst = omarchy::allocator().malloc(kBytes);
  REQUIRE(src.raw_ptr() != nullptr);
  REQUIRE(dst.raw_ptr() != nullptr);

  auto* src_buf = static_cast<omarchy::VulkanBuffer*>(src.ptr());
  auto* dst_buf = static_cast<omarchy::VulkanBuffer*>(dst.ptr());

  auto* src_bytes = static_cast<uint8_t*>(src_buf->data);
  auto* dst_bytes = static_cast<uint8_t*>(dst_buf->data);
  for (size_t i = 0; i < kBytes; ++i) {
    src_bytes[i] = static_cast<uint8_t>(i % 251);
  }
  std::memset(dst_bytes, 0, kBytes);

  uint64_t submissions_before =
      omarchy::trace::counters().vk_submissions.load();
  uint64_t copies_before = omarchy::trace::counters().vk_buffer_copies.load();

  encoder.copy_buffer(src_buf->buffer, dst_buf->buffer, kBytes);
  encoder.commit();
  encoder.synchronize();

  CHECK(
      omarchy::trace::counters().vk_submissions.load() ==
      submissions_before + 1);
  CHECK(
      omarchy::trace::counters().vk_buffer_copies.load() == copies_before + 1);

  size_t mismatches = 0;
  for (size_t i = 0; i < kBytes; ++i) {
    if (src_bytes[i] != dst_bytes[i]) {
      mismatches++;
    }
  }
  CHECK(mismatches == 0);
}

TEST_CASE("events synchronize between streams") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream a = new_stream(Device::gpu);
  Stream b = new_stream(Device::gpu);

  Event e{a};
  e.set_value(1);
  CHECK_FALSE(e.is_signaled());

  e.signal(a); // asynchronous signal submission
  e.wait(); // host wait on the device timeline semaphore
  CHECK(e.is_signaled());

  // A wait recorded on another stream must not hang and must be honored by
  // that stream's submissions.
  auto& enc_b = omarchy::get_command_encoder(b);
  e.wait(b);
  auto buf = omarchy::allocator().malloc(4096);
  auto* buf_ptr = static_cast<omarchy::VulkanBuffer*>(buf.ptr());
  enc_b.fill_buffer(buf_ptr->buffer, 0, 4096);
  enc_b.commit();
  enc_b.synchronize();
  CHECK(e.is_signaled());
  omarchy::allocator().free(buf);
}

TEST_CASE("queued event wait survives destruction of the Event") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream a = new_stream(Device::gpu);
  Stream b = new_stream(Device::gpu);
  {
    Event e{a};
    e.set_value(1);
    e.signal(a);
    e.wait(b); // queued on encoder b as a raw wait before the fix
  } // e destroyed here: the encoder must keep the semaphore alive
  auto& enc_b = omarchy::get_command_encoder(b);
  auto buf = omarchy::allocator().malloc(4096);
  auto* p = static_cast<omarchy::VulkanBuffer*>(buf.ptr());
  enc_b.fill_buffer(p->buffer, 7, 4096);
  enc_b.commit(); // must never submit a destroyed VkSemaphore
  enc_b.synchronize();
  CHECK(*static_cast<uint32_t*>(p->data) == 7u);
  omarchy::allocator().free(buf);
}

TEST_CASE("cross-stream event ordering is preserved") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream a = new_stream(Device::gpu);
  Stream b = new_stream(Device::gpu);

  constexpr size_t kBytes = 1 << 16;
  auto src = omarchy::allocator().malloc(kBytes);
  auto mid = omarchy::allocator().malloc(kBytes);
  auto dst = omarchy::allocator().malloc(kBytes);
  auto* src_bytes = static_cast<uint8_t*>(
      static_cast<omarchy::VulkanBuffer*>(src.ptr())->data);
  auto* mid_bytes = static_cast<uint8_t*>(
      static_cast<omarchy::VulkanBuffer*>(mid.ptr())->data);
  auto* dst_bytes = static_cast<uint8_t*>(
      static_cast<omarchy::VulkanBuffer*>(dst.ptr())->data);
  for (size_t i = 0; i < kBytes; ++i) {
    src_bytes[i] = static_cast<uint8_t>(i % 251);
  }
  std::memset(mid_bytes, 0, kBytes);
  std::memset(dst_bytes, 0, kBytes);

  // MLX order: the consumer records its wait before the producer signals.
  Event e{a};
  e.set_value(1);
  e.wait(b);

  auto& enc_a = omarchy::get_command_encoder(a);
  auto& enc_b = omarchy::get_command_encoder(b);
  auto* src_buf = static_cast<omarchy::VulkanBuffer*>(src.ptr());
  auto* mid_buf = static_cast<omarchy::VulkanBuffer*>(mid.ptr());
  auto* dst_buf = static_cast<omarchy::VulkanBuffer*>(dst.ptr());

  enc_a.copy_buffer(src_buf->buffer, mid_buf->buffer, kBytes);
  e.signal(a); // signal submission follows the copy on stream a
  enc_b.copy_buffer(mid_buf->buffer, dst_buf->buffer, kBytes);
  enc_b.commit(); // gated on the event by the queued timeline wait
  enc_b.synchronize();

  size_t mismatches = 0;
  for (size_t i = 0; i < kBytes; ++i) {
    if (src_bytes[i] != dst_bytes[i]) {
      mismatches++;
    }
  }
  CHECK(mismatches == 0);

  omarchy::allocator().free(src);
  omarchy::allocator().free(mid);
  omarchy::allocator().free(dst);
}

TEST_CASE("handler-only commit runs handlers after prior queue work") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = new_stream(Device::gpu);
  auto& enc = omarchy::get_command_encoder(s);
  auto buf = omarchy::allocator().malloc(4096);
  auto* p = static_cast<omarchy::VulkanBuffer*>(buf.ptr());

  // Prior GPU work, submitted asynchronously.
  enc.fill_buffer(p->buffer, 0x5a, 4096);

  // A submission with only a completion handler (the fill_zero edge: no
  // aligned 4-byte word to record). It must still be submitted and ordered
  // after the fill.
  std::atomic<bool> ran{false};
  enc.add_completed_handler([&ran]() { ran.store(true); });
  enc.commit();
  enc.synchronize();
  // synchronize() joins the dispatcher drain: the handler must have run
  // before it returns (pins the drain-join contract).
  CHECK(ran.load());
  CHECK(*static_cast<uint32_t*>(p->data) == 0x5au);
  omarchy::allocator().free(buf);
}

TEST_CASE("wait joins a handler the background drain is still running") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = new_stream(Device::gpu);
  auto& enc = omarchy::get_command_encoder(s);

  // Race under test: the background dispatcher thread takes a submission
  // out of pending_ and starts its handler while another thread calls
  // wait() for the same value. The caller's inline drain then finds an
  // empty pending_, and wait() must still not return until the handler
  // finishes (the drained_value_ join). Each round blocks the handler
  // until this test releases it, so a wait() that returns first is
  // detected by state, not by timing.
  constexpr int kRaces = 20;
  for (int i = 0; i < kRaces; ++i) {
    std::promise<void> started_promise;
    std::future<void> started = started_promise.get_future();
    std::promise<void> release_promise;
    std::future<void> release = release_promise.get_future();
    std::atomic<bool> handler_finished{false};
    std::atomic<bool> sync_returned{false};
    std::atomic<bool> finished_at_return{false};
    std::string sync_error;

    enc.add_completed_handler([&]() {
      started_promise.set_value();
      release.wait();
      handler_finished.store(true);
    });
    enc.commit();

    // The handler signals only after the dispatcher moved the entry out
    // of pending_, so every synchronize() below is guaranteed to hit the
    // raced path: the caller-side drain finds nothing to run.
    auto started_status = started.wait_for(std::chrono::seconds(10));
    CHECK(started_status == std::future_status::ready);
    if (started_status != std::future_status::ready) {
      release_promise.set_value();
      enc.synchronize();
      continue;
    }

    std::thread waiter([&]() {
      try {
        enc.synchronize();
      } catch (const std::exception& ex) {
        sync_error = ex.what();
      }
      // Handler state at the instant synchronize returned.
      finished_at_return.store(handler_finished.load());
      sync_returned.store(true);
    });

    // Bounded detection window, not the proof: a broken wait() returns
    // immediately once pending_ is empty, so sync_returned lands here
    // within microseconds of the thread starting. The proof is the state
    // check below: synchronize returned while the blocked handler
    // provably had not finished (this thread has not released it yet).
    auto window =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
    while (!sync_returned.load() && std::chrono::steady_clock::now() < window) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Unblock the handler and join. On correct code the waiter is still
    // inside the drained_value_ join here; the release is what lets it
    // return.
    release_promise.set_value();
    waiter.join();

    CHECK(sync_error.empty());
    CHECK(finished_at_return.load());
    CHECK(handler_finished.load());
  }
}

TEST_CASE(
    "wait serializes against the background drain to preserve handler order") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = new_stream(Device::gpu);
  auto& enc = omarchy::get_command_encoder(s);
  auto buf = omarchy::allocator().malloc(64);
  auto* p = static_cast<omarchy::VulkanBuffer*>(buf.ptr());

  // V1's handler blocks until the test releases it. V2 is enqueued and
  // waited on from a second thread while V1 is still executing on the
  // background dispatcher. drain_through is serialized end to end
  // through drain_mutex_, so the waiter's drain cannot run V2's handler
  // before V1's drain finishes.
  std::promise<void> v1_started_promise;
  std::future<void> v1_started = v1_started_promise.get_future();
  std::promise<void> v1_release_promise;
  std::future<void> v1_release = v1_release_promise.get_future();
  std::atomic<bool> v1_done{false};
  std::atomic<bool> wait_returned{false};
  std::atomic<bool> wait_returned_after_v1{false};
  std::string wait_error;

  enc.fill_buffer(p->buffer, 0xa5, 64);
  enc.add_completed_handler([&]() {
    v1_started_promise.set_value();
    v1_release.wait();
    v1_done.store(true);
  });
  // This test targets the CompletionDispatcher's drain serialization, so
  // the submission must be on the queue immediately: commit_now() (plain
  // commit() defers into the open batch by design).
  enc.commit();

  auto v1_status = v1_started.wait_for(std::chrono::seconds(10));
  CHECK(v1_status == std::future_status::ready);
  if (v1_status != std::future_status::ready) {
    v1_release_promise.set_value();
    enc.synchronize();
    omarchy::allocator().free(buf);
    return;
  }

  // Submit V2 (handler-only) on this thread, then synchronize it from a
  // second thread. The waiter enters drain_through(V2) and must block
  // until V1's drain releases drain_mutex_.
  enc.add_completed_handler([]() {});
  enc.commit();

  std::thread waiter([&]() {
    try {
      enc.synchronize();
    } catch (const std::exception& ex) {
      wait_error = ex.what();
    }
    wait_returned_after_v1.store(v1_done.load());
    wait_returned.store(true);
  });

  // Bounded window for the waiter to reach drain_through. A broken
  // serializer returns here within microseconds: V2 was enqueued, the
  // GPU signaled it, and the waiter's drain_through would find nothing
  // blocking it.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  CHECK_FALSE(wait_returned.load());

  // Release V1; the waiter now makes progress and synchronize returns.
  v1_release_promise.set_value();
  waiter.join();

  CHECK(wait_error.empty());
  CHECK(wait_returned.load());
  CHECK(wait_returned_after_v1.load());
  omarchy::allocator().free(buf);
}

TEST_CASE("temporaries release one completion after their submission") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = new_stream(Device::gpu);
  auto& enc = omarchy::get_command_encoder(s);

  auto arr = zeros({1024}, float32);
  arr.eval();
  std::weak_ptr<void> observed = arr.data_shared_ptr();
  auto* arr_buf = static_cast<omarchy::VulkanBuffer*>(arr.buffer().ptr());

  // A large fill stretches execution so the submission is still running
  // while the checks below run. (Gating via an unsatisfied timeline wait
  // would stall the queue: the satisfying signal itself travels on the
  // same single queue.)
  constexpr VkDeviceSize kBigBytes = 256ull << 20;
  auto scratch = omarchy::allocator().malloc(kBigBytes);
  auto* scratch_buf = static_cast<omarchy::VulkanBuffer*>(scratch.ptr());
  enc.fill_buffer(scratch_buf->buffer, 0x11, kBigBytes);
  enc.fill_buffer(arr_buf->buffer, 0x33, 4096);
  enc.add_temporary(arr); // the ownership under test
  enc.commit(); // in flight

  // Drop the caller's reference while work is queued (arrays have no
  // default ctor: reassign to a fresh array).
  arr = zeros({1}, float32);
  CHECK_FALSE(observed.expired()); // retention outlived destruction

  enc.synchronize(); // bounded completion wait; joins handler execution
  // Mesa's queue thread signals a submission's semaphores (including the
  // completion timeline) before its submit-final cleanup retires the
  // submission's timeline points, so a payload of this submission must
  // still be alive here: destroying it now races the driver.
  CHECK_FALSE(observed.expired());

  // The next completion proves the driver finished this submission's
  // cleanup; the retired payload releases then, and never leaks.
  enc.fill_buffer(scratch_buf->buffer, 0x44, 4096);
  enc.commit();
  enc.synchronize();
  CHECK(observed.expired());
  omarchy::allocator().free(scratch);
}

TEST_CASE(
    "eager per-node commits reuse ring slots without host joins") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = new_stream(Device::gpu);
  auto& enc = omarchy::get_command_encoder(s);
  constexpr size_t kBytes = 16u << 20;
  auto buf = omarchy::allocator().malloc(kBytes);
  auto* p = static_cast<omarchy::VulkanBuffer*>(buf.ptr());

  // Eager-flush contract (kBatchNodeBudget = 1, measured: deeper batching
  // defeats the evaluator's task pipeline and regresses tok/s heavily).
  // Every commit submits, but the in-flight ring means a commit never
  // host-joins the previous submission the way the pre-ring encoder did
  // (that join was the dominant decode cost in the 2026-09-02 profile).
  // Ordering still holds end to end (each node carries its full barrier
  // pair), so the final bytes carry the last iteration's value.
  constexpr int kIters = 8;
  uint64_t submissions_before =
      omarchy::trace::counters().vk_submissions.load();
  for (int i = 0; i < kIters; ++i) {
    enc.fill_buffer(p->buffer, (i % 2) ? 0x5a5a5a5au : 0xa5a5a5a5u, kBytes);
    enc.commit();
  }
  CHECK(
      omarchy::trace::counters().vk_submissions.load() ==
      submissions_before + kIters);
  enc.synchronize();
  CHECK(enc.synchronized());
  const uint32_t expected = ((kIters - 1) % 2) ? 0x5a5a5a5au : 0xa5a5a5a5u;
  const auto* words = static_cast<const uint32_t*>(p->data);
  size_t mismatches = 0;
  for (size_t i = 0; i < kBytes / sizeof(uint32_t); ++i) {
    if (words[i] != expected) {
      mismatches++;
    }
  }
  CHECK(mismatches == 0);
  omarchy::allocator().free(buf);
}

TEST_CASE("Event::wait joins completion handlers without polling") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = new_stream(Device::gpu);
  auto buf = omarchy::allocator().malloc(8);
  auto* p = static_cast<omarchy::VulkanBuffer*>(buf.ptr());
  std::memset(p->data, 0, 8);
  auto& enc = omarchy::get_command_encoder(s);
  enc.fill_buffer(p->buffer, 0x2a, 8);
  uint8_t* bytes = static_cast<uint8_t*>(p->data);
  enc.add_completed_handler([&bytes]() { bytes[0] = 0x5a; });
  enc.commit();
  Event e{s};
  e.set_value(1);
  e.signal(s);
  e.wait(); // direct host wait; must join the handler boundary
  CHECK(bytes[0] == 0x5a);
  omarchy::allocator().free(buf);
}

TEST_CASE("sub-word zero fill then copy stays zero without syncs") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  set_default_device(Device::gpu);
  // 3 bytes: sub-word, so fill_zero writes host-side edges; the follow-up
  // copy must never observe stale bytes despite no intermediate sync.
  auto a = zeros({3}, uint8);
  a.eval();
  auto b = copy(a);
  b.eval();
  synchronize(default_stream(default_device()));
  const auto* d = b.data<uint8_t>();
  for (int i = 0; i < 3; ++i) {
    CHECK_EQ(d[i], uint8_t(0));
  }
}

TEST_CASE("zero-scalar fast path refuses GPU-produced scalars") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  set_default_device(Device::gpu);
  Stream s = new_stream(Device::gpu);

  auto lit = array(0.0f);
  auto copied = copy(lit, s);
  copied.eval();
  synchronize(s);
  CHECK_EQ(copied.data<float>()[0], 0.0f);

  auto produced = zeros({1}, float32, s);
  array out({1.0f, 1.0f, 1.0f, 1.0f});
  bool threw = false;
  try {
    fill_gpu(produced, out, s);
  } catch (const std::exception& ex) {
    threw = true;
    std::string msg = ex.what();
    CHECK(msg.find("[omarchy]") != std::string::npos);
    CHECK(msg.find("GPU-in-flight fill") != std::string::npos);
  }
  CHECK(threw);
}

TEST_CASE("host event bridges a GPU-stream signal") {
  if (!cpu::is_available()) {
    skip("requires a development build with the CPU backend enabled.");
    return;
  }
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream cs = new_stream(Device::cpu);
  Stream gs = new_stream(Device::gpu);
  Event e{cs};
  e.set_value(1);
  e.signal(cs); // host counter path: queued behind cpu-stream work
  e.wait(); // bounded host wait; joins the queue behind the signal
  CHECK(e.is_signaled());

  e.set_value(2);
  e.signal(gs); // GPU stream: the host counter advances at completion
  // No negative check here: the submission may already have completed by
  // the time this thread looks, so "not yet signaled" is unobservable.
  e.wait(); // bounded host wait; joins the completion handler
  CHECK(e.is_signaled());
}
TEST_CASE("device_info returns stable references under concurrent access") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  const auto& info0 = gpu::device_info(0);
  auto name_it = info0.find("device_name");
  REQUIRE(name_it != info0.end());
  const auto* name0 = std::get_if<std::string>(&name_it->second);
  REQUIRE(name0 != nullptr);

  std::atomic<bool> stop{false};
  std::vector<std::thread> hammers;
  for (int t = 0; t < 4; ++t) {
    hammers.emplace_back([&stop]() {
      while (!stop.load()) {
        (void)gpu::device_info(0);
      }
    });
  }
  // The same map object and the same entries must come back every time.
  for (int i = 0; i < 2000; ++i) {
    const auto& info = gpu::device_info(0);
    CHECK(&info == &info0);
    auto it = info.find("device_name");
    REQUIRE(it != info.end());
    CHECK(&it->second == &name_it->second);
  }
  stop.store(true);
  for (auto& t : hammers) {
    t.join();
  }
  CHECK_FALSE(name0->empty());
}

TEST_CASE("independent streams measure against the serialized sum") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  auto& dev = omarchy::device(0);
  if (dev.capabilities().driver_id == VK_DRIVER_ID_MESA_LLVMPIPE) {
    skip("software Vulkan has no independent GPU execution.");
    return;
  }
  std::cout << "[receipt] device exposes " << dev.capabilities().queue_count
            << " queue(s); measuring overlap on one VkQueue\n";

  constexpr size_t kBytes = 8 << 20;
  Stream s1 = new_stream(Device::gpu);
  Stream s2 = new_stream(Device::gpu);
  auto a1 = omarchy::allocator().malloc(kBytes);
  auto b1 = omarchy::allocator().malloc(kBytes);
  auto a2 = omarchy::allocator().malloc(kBytes);
  auto b2 = omarchy::allocator().malloc(kBytes);
  auto* x1 = static_cast<omarchy::VulkanBuffer*>(a1.ptr());
  auto* y1 = static_cast<omarchy::VulkanBuffer*>(b1.ptr());
  auto* x2 = static_cast<omarchy::VulkanBuffer*>(a2.ptr());
  auto* y2 = static_cast<omarchy::VulkanBuffer*>(b2.ptr());
  std::memset(x1->data, 0x11, kBytes);
  std::memset(x2->data, 0x22, kBytes);

  auto& enc1 = omarchy::get_command_encoder(s1);
  auto& enc2 = omarchy::get_command_encoder(s2);
  auto clock = [](std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
        .count();
  };

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < 2; ++i) {
    enc1.copy_buffer(x1->buffer, y1->buffer, kBytes);
    enc1.commit();
  }
  enc1.synchronize();
  double per_iter = clock(t0) / 2.0;
  int iters = static_cast<int>(0.35 / per_iter);
  iters = std::max(2, std::min(iters, 400));

  auto run_serialized = [&]() {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
      enc1.copy_buffer(x1->buffer, y1->buffer, kBytes);
      enc1.commit();
    }
    enc1.synchronize();
    for (int i = 0; i < iters; ++i) {
      enc2.copy_buffer(x2->buffer, y2->buffer, kBytes);
      enc2.commit();
    }
    enc2.synchronize();
    return clock(start);
  };
  auto run_concurrent = [&]() {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
      enc1.copy_buffer(x1->buffer, y1->buffer, kBytes);
      enc1.commit();
      enc2.copy_buffer(x2->buffer, y2->buffer, kBytes);
      enc2.commit();
    }
    enc1.synchronize();
    enc2.synchronize();
    return clock(start);
  };

  std::array<double, 3> serialized{};
  std::array<double, 3> concurrent{};
  for (size_t trial = 0; trial < serialized.size(); ++trial) {
    if ((trial & 1) == 0) {
      serialized[trial] = run_serialized();
      concurrent[trial] = run_concurrent();
    } else {
      concurrent[trial] = run_concurrent();
      serialized[trial] = run_serialized();
    }
  }
  auto median = [](std::array<double, 3> values) {
    std::sort(values.begin(), values.end());
    return values[1];
  };
  double serialized_median = median(serialized);
  double concurrent_median = median(concurrent);
  double ratio = serialized_median / concurrent_median;
  std::cout << "[receipt] iters=" << iters
            << " serialized_median=" << serialized_median
            << "s concurrent_median=" << concurrent_median
            << "s ratio=" << ratio << "\n";
  // Stream independence: interleaving two streams' batches must not cost
  // more than running them alone; a global serialization lock would push
  // the ratio well below 1, and that is what the lower bound guards. On
  // real async hardware the interleaved run can legitimately FINISH
  // FASTER than serialized: the serialized pattern exposes a host-GPU
  // sync round trip between its two loops that interleaving hides
  // (measured ratio 1.21 on the M1/Honeykrisp device; llvmpipe's
  // synchronous QueueSubmit lands at ratio ~1, which is why this upper
  // direction passed there). The old `ratio < 1.10` upper bound asserted
  // that the overlap win was structurally subsumed, and that assumption
  // is false on the hardware we ship for, so it is not asserted here.
  // Interleaved being faster is only legitimate if every copy actually
  // ran, so the destination buffers are verified byte-for-byte instead.
  CHECK(ratio > 0.95);
  CHECK(std::memcmp(y1->data, x1->data, kBytes) == 0);
  CHECK(std::memcmp(y2->data, x2->data, kBytes) == 0);

  omarchy::allocator().free(a1);
  omarchy::allocator().free(b1);
  omarchy::allocator().free(a2);
  omarchy::allocator().free(b2);
}

TEST_CASE("a fresh process reopens the Vulkan device") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  auto r = run_child("reopen", 60);
  REQUIRE_FALSE(r.timed_out);
  CHECK_MESSAGE(
      r.code == 0, "child reopen scenario failed with code " << r.code);
}

TEST_CASE("a hung submit returns a bounded Omarchy error") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  // The child queues an unsatisfiable timeline wait. The bounded wait must
  // throw a typed Omarchy error (the 10 s bound is the behavior under
  // test); process isolation keeps the wedged queue away from this process.
  auto r = run_child("bounded_submit", 45);
  REQUIRE_FALSE(r.timed_out);
  CHECK_MESSAGE(
      r.code == 0, "child bounded_submit scenario failed with code " << r.code);
}

TEST_CASE("tensor ops dispatch on Vulkan and never silently on CPU") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }

  set_default_device(Device::gpu);

  // Default-device work must land on the Vulkan stream. The GPU counters
  // prove the kernel ran there; a silent CPU substitution would leave
  // them flat.
  auto x = array({1.0f, 2.0f}, float32);
  auto y = array({3.0f, 4.0f}, float32);
  auto z = add(x, y);
  uint64_t gpu_dispatches_before =
      omarchy::trace::counters().gpu_primitive_dispatches.load();
  uint64_t compute_dispatches_before =
      omarchy::trace::counters().vk_compute_dispatches.load();
  z.eval();
  synchronize(default_stream(default_device()));
  CHECK_EQ(z.data<float>()[0], 4.0f);
  CHECK_EQ(z.data<float>()[1], 6.0f);
  CHECK(
      omarchy::trace::counters().gpu_primitive_dispatches.load() >
      gpu_dispatches_before);
  CHECK(
      omarchy::trace::counters().vk_compute_dispatches.load() >
      compute_dispatches_before);

  if (cpu::is_available()) {
    // This build carries a CPU tensor backend. The stream boundary is the
    // contract: an op created on an explicit CPU stream runs there and
    // leaves the GPU counters flat. Nothing else may move work onto CPU.
    uint64_t gpu_dispatches_before_cpu =
        omarchy::trace::counters().gpu_primitive_dispatches.load();
    uint64_t compute_dispatches_before_cpu =
        omarchy::trace::counters().vk_compute_dispatches.load();
    Stream cpu_stream = new_stream(Device::cpu);
    auto c = add(x, y, cpu_stream);
    c.eval();
    synchronize(cpu_stream);
    CHECK_EQ(c.data<float>()[0], 4.0f);
    CHECK_EQ(c.data<float>()[1], 6.0f);
    CHECK_EQ(
        omarchy::trace::counters().gpu_primitive_dispatches.load(),
        gpu_dispatches_before_cpu);
    CHECK_EQ(
        omarchy::trace::counters().vk_compute_dispatches.load(),
        compute_dispatches_before_cpu);
  } else {
    // No CPU backend in this build: the CPU device does not exist at all.
    CHECK_FALSE(is_available(Device::cpu));
    CHECK_EQ(device_count(Device::cpu), 0);
  }

  auto zero_array = mlx::core::zeros({2, 3}, float32);
  zero_array.eval();
  synchronize(default_stream(default_device()));
  const auto* data = zero_array.data<float>();
  for (int i = 0; i < 6; ++i) {
    CHECK_EQ(data[i], 0.0f);
  }
}

TEST_CASE("small eager output stays ordered across deep submit boundaries") {
  // Pins the in-order-stream contract: an eager one-element f32 output
  // committed as its own submission must be visible to a consumer
  // dispatch in a LATER submission, even when long work fills the queue
  // in between. This is the shape of the compiled 4-bit decode abort:
  // the RoPE positions chain lost its device write on Honeykrisp when a
  // consumer submission ran without a dependency on the producer
  // submission (Vulkan defines no cross-submission ordering without a
  // wait; encoder submit() now waits on the stream's previous completion
  // value). llvmpipe executes submissions synchronously in order, so
  // this test pins the contract there and guards regressions of the
  // wait itself; the pre-fix failure needs an out-of-order queue and is
  // validated on Honeykrisp hardware.
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  auto s = default_stream(default_device());
  for (int iter = 0; iter < 8; ++iter) {
    // Tiny eager producer: one element, its own submission.
    auto positions = arange(iter, iter + 1, float32, s);
    positions.eval();
    // Long work fills the queue so the consumer below is submitted
    // while earlier submissions are still executing.
    auto heavy = astype(ones({256, 256}, float32, s), float32);
    for (int i = 0; i < 3; ++i) {
      heavy = matmul(heavy, heavy, s);
      heavy.eval();
    }
    // Consumer submission reads the producer's buffer across the queue.
    auto theta = positions * ones({1}, float32, s);
    auto c = cos(theta);
    c.eval();
    synchronize(s);
    CHECK_EQ(theta.item<float>(), static_cast<float>(iter));
    CHECK(std::fabs(c.item<float>() - std::cos(static_cast<float>(iter))) <
          1e-5f);
  }
}
