// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Regression guard for the single-full-sequence-eval wedge
// (docs/known-defects.md, v0.3.4 "A single large evaluation can wedge
// the GPU queue", receipts/2026-09-04-hang-watchdog-hardware.md,
// receipts/2026-09-04-wedge-candidates-and-bisect-protocol.md).
//
// The wedge is Honeykrisp-specific: a single mx.eval over a full-sequence
// forward at a specific sequence length (currently believed to be 2,048
// tokens for Qwen 0.5B eager, but the exact threshold is what the M1
// bisect on branch wedge/q1-2048 commits will pin). At that length the
// timeline counter stays frozen at 0 and the watchdog fails by name.
//
// What this test verifies once the threshold is known:
//   1. The wedge shape fails with a typed [omarchy] error (the watchdog
//      catches it) within the no-progress interval + 2 s margin.
//   2. The watchdog's typed error names the failing condition
//      (last_observed, target) so a future regression that hangs the
//      host instead of wedging the queue still fails loudly.
//
// Until the M1 bisect pins the exact shape, this test runs the
// discriminator from the receipt (single mx.eval over a Qwen-shape SDPA
// at the believed wedge length) and asserts either success-with-correct-
// values or named-error-with-watchdog. On dev-box lavapipe the SDPA
// completes; on Honeykrisp it wedges and the watchdog catches it.
//
// The test re-execs itself in a fresh child process because the wedge
// poisons state; an in-process child would observe residual wedged-queue
// contamination. Pattern mirrors test_runtime.cpp's run_child helper.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/fast.h"
#include "mlx/ops.h"

using namespace mlx::core;

// Sequence length at which the wedge first reproduces on Honeykrisp.
// Pre-bisect default: 2,048 (matches the receipt). Once the M1 bisect
// on branch wedge/q1-2048 pins the exact value, change this constant
// to match; the test will continue to enforce the bound.
static constexpr int kWedgingSequenceLength = 2048;

// Qwen 0.5B-Instruct shape: 14 query heads, 2 kv heads, head_dim 64.
// This is the SDPA shape the wedge reproducer hits.
static constexpr int kQwenQHeads = 14;
static constexpr int kQwenKVHeads = 2;
static constexpr int kQwenHeadDim = 64;

// Margin added to the watchdog interval when measuring child wall time.
// Watchdog trips at exactly 10 s; child must exit in less than 10 s + 2.
static constexpr int kWatchdogMarginSeconds = 2;

namespace {

void skip(const char* reason) {
  std::cout << "Skipping: " << reason << "\n";
}

// Child branch: run the wedge reproducer. Returns:
//   0  = success (no wedge on this device at this length)
//   4  = non-[omarchy] exception
//   5  = exception is not from the watchdog
//   6  = elapsed > watchdog + margin (the watchdog is too slow)
//   77 = no qualifying device
int run_wedge_child(int seq_len) {
  if (!gpu::is_available()) {
    return 77;
  }
  set_default_device(Device::gpu);
  // Eager path matches the receipt's discriminator.
  setenv("MLX_DISABLE_COMPILE", "1", 1);
  // 10 s watchdog; the wedge will be caught at this interval.
  setenv("MLX_OMARCHY_HANG_NO_PROGRESS_NS", "10000000000", 1);
  setenv("MLX_OMARCHY_MAX_WALL_NS", "10000000000", 1);

  std::vector<float> q_data(
      1 * kQwenQHeads * seq_len * kQwenHeadDim, 0.0f);
  std::vector<float> k_data(
      1 * kQwenKVHeads * seq_len * kQwenHeadDim, 0.0f);
  std::vector<float> v_data(
      1 * kQwenKVHeads * seq_len * kQwenHeadDim, 0.0f);
  auto q = array(
      q_data.data(), Shape{1, kQwenQHeads, seq_len, kQwenHeadDim}, float32);
  auto k = array(
      k_data.data(), Shape{1, kQwenKVHeads, seq_len, kQwenHeadDim}, float32);
  auto v = array(
      v_data.data(), Shape{1, kQwenKVHeads, seq_len, kQwenHeadDim}, float32);

  Stream stream = new_stream(Device::gpu);
  float scale = 1.0f / static_cast<float>(std::sqrt(kQwenHeadDim));

  auto t0 = std::chrono::steady_clock::now();
  try {
    auto out = fast::scaled_dot_product_attention(
        q,
        k,
        v,
        scale,
        std::string(""),
        std::nullopt,
        std::nullopt,
        false,
        stream);
    out.eval();
    auto elapsed = std::chrono::steady_clock::now() - t0;
    std::cout << "[child/wedge] seq_len=" << seq_len
              << " completed in "
              << std::chrono::duration<double>(elapsed).count()
              << " s (no wedge reproduced on this device)\n";
    return 0;
  } catch (const std::exception& ex) {
    auto elapsed = std::chrono::steady_clock::now() - t0;
    std::string msg = ex.what();
    std::cout << "[child/wedge] seq_len=" << seq_len
              << " threw after "
              << std::chrono::duration<double>(elapsed).count()
              << " s: " << msg << "\n";
    if (msg.find("[omarchy]") == std::string::npos) {
      return 4;
    }
    if (msg.find("failed to advance") == std::string::npos) {
      return 5;
    }
    if (elapsed > std::chrono::seconds(10 + kWatchdogMarginSeconds)) {
      return 6;
    }
    return 0; // watchdog caught it correctly
  }
}

// Fork + execlp the current test binary with MLX_OMARCHY_TEST_CHILD_WEDGE
// set. Bounded by 60 s (well above watchdog + margin). Returns the
// child's exit code, or -1 if hard-capped.
int run_wedge(int seq_len) {
  char self[4096];
  ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
  if (n <= 0) {
    return 127;
  }
  self[n] = '\0';
  char seq_buf[32];
  std::snprintf(seq_buf, sizeof(seq_buf), "%d", seq_len);
  pid_t pid = fork();
  if (pid == 0) {
    setenv("MLX_OMARCHY_TEST_CHILD_WEDGE", "1", 1);
    setenv("MLX_OMARCHY_TEST_WEDGE_SEQ_LEN", seq_buf, 1);
    execl(self, self, static_cast<char*>(nullptr));
    _exit(127);
  }
  auto deadline = std::chrono::steady_clock::now() +
      std::chrono::seconds(60);
  int status = 0;
  for (;;) {
    pid_t done = waitpid(pid, &status, WNOHANG);
    if (done == pid) {
      break;
    }
    if (std::chrono::steady_clock::now() > deadline) {
      kill(pid, SIGKILL);
      waitpid(pid, &status, 0);
      return -1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  return -2;
}

} // namespace

int main(int argc, char** argv) {
  // Child branch: do the work and exit with a code the parent reads.
  if (std::getenv("MLX_OMARCHY_TEST_CHILD_WEDGE")) {
    int seq_len = kWedgingSequenceLength;
    if (const char* sl = std::getenv("MLX_OMARCHY_TEST_WEDGE_SEQ_LEN")) {
      seq_len = std::atoi(sl);
    }
    return run_wedge_child(seq_len);
  }
  doctest::Context ctx;
  ctx.applyCommandLine(argc, argv);
  return ctx.run();
}

TEST_CASE("a single full-sequence eager SDPA at the wedge threshold "
          "either completes or fails by name within the watchdog") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  // Run the wedge reproducer in a fresh child process. The wedge poisons
  // state, so an in-process child would observe residual contamination.
  // The child runs at kWedgingSequenceLength; on Honeykrisp the watchdog
  // should catch the wedge and emit a typed error within the watchdog
  // interval. On lavapipe or any device where the wedge does not
  // reproduce, the child returns 0 (success). Either outcome is
  // acceptable; a hard hang or a non-watchdog error is a regression.
  auto r = run_wedge(kWedgingSequenceLength);
  REQUIRE_MESSAGE(
      r != -1,
      "child wedge scenario hard-capped (>60 s); the watchdog is not "
      "catching the wedge within the bound");
  CHECK_MESSAGE(
      r == 0,
      "child wedge scenario failed with code " << r
      << "; expected watchdog-caught named error or successful "
         "completion. If the bisect on branch wedge/q1-2048 has "
         "pinned a different wedge shape, update kWedgingSequenceLength "
         "and the SDPA shape constants.");
}
