// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Weight-streaming bandwidth of the affine q4 decode GEMV, uninstrumented:
// N back-to-back quantized_matmul dispatches on one activation row, one
// eval, wall time / N. The shapes are the real Qwen2.5-0.5B-4bit decode
// layers. Prints one JSON row per shape (best of three) so the kernel can
// be iterated from a libmlx rebuild without a wheel. Every shape is first
// checked against the dequantized dense matmul so a broken kernel never
// reports a bandwidth.
//
//   MLX_OMARCHY_QMM_BENCH_REPS=100 ./omarchy_qmm_bandwidth_bench
//
// Not registered with ctest: the 68 MB lm_head shape is a benchmark, not
// a test, and takes minutes on llvmpipe.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "mlx/backend/omarchy/encoder.h"
#include "mlx/mlx.h"

using namespace mlx::core;

namespace {

struct Case {
  const char* name;
  int n;
  int k;
};

constexpr Case kCases[] = {
    {"k_proj 57KB", 128, 896},
    {"q_proj 401KB", 896, 896},
    {"gate 2.18MB", 4864, 896},
    {"down 2.18MB", 896, 4864},
    {"lm_head 68MB", 151936, 896},
};

} // namespace

int main(int argc, char** argv) {
  if (!gpu::is_available()) {
    std::fprintf(stderr, "no qualifying Vulkan device\n");
    return 1;
  }
  set_default_device(Device::gpu);
  Stream stream = new_stream(Device::gpu);
  auto& encoder = omarchy::get_command_encoder(stream);
  int reps = 100;
  if (const char* env = std::getenv("MLX_OMARCHY_QMM_BENCH_REPS")) {
    reps = std::atoi(env);
  }
  const std::string filter = argc > 1 ? argv[1] : "";
  constexpr int group = 64;
  constexpr int bits = 4;
  int failures = 0;
  for (const Case& c : kCases) {
    if (!filter.empty() && std::string(c.name).find(filter) == std::string::npos) {
      continue;
    }
    array w = astype(
        random::normal(Shape{c.n, c.k}, random::key(1), stream),
        float16,
        stream);
    auto q = quantize(w, group, bits, "affine", std::nullopt, stream);
    array x = astype(
        random::normal(Shape{1, c.k}, random::key(2), stream), float16, stream);
    eval(q[0], q[1], q[2], x);
    size_t weight_bytes = q[0].nbytes() + q[1].nbytes() + q[2].nbytes();

    // Correctness gate: dequantized dense matmul on the same device.
    array dense = dequantize(q[0], q[1], q[2], group, bits, "affine",
        std::nullopt, std::nullopt, stream);
    array ref = astype(
        matmul(astype(x, float32, stream),
            transpose(astype(dense, float32, stream), stream), stream),
        float32, stream);
    array got = astype(
        quantized_matmul(x, q[0], q[1], q[2], true, group, bits, "affine",
            stream),
        float32, stream);
    eval(ref, got);
    encoder.synchronize();
    float max_abs = 0.0f;
    float max_ref = 0.0f;
    for (int i = 0; i < c.n; ++i) {
      max_abs = std::fmax(
          max_abs, std::fabs(ref.data<float>()[i] - got.data<float>()[i]));
      max_ref = std::fmax(max_ref, std::fabs(ref.data<float>()[i]));
    }
    // f16 output rounding of |ref| up to ~100 plus f32 sum-order noise.
    bool ok = max_abs <= 0.15f * std::fmax(1.0f, max_ref / 32.0f);
    if (!ok) {
      ++failures;
    }

    auto run = [&](int count) {
      std::vector<array> ys;
      ys.reserve(count);
      for (int i = 0; i < count; ++i) {
        ys.push_back(quantized_matmul(
            x, q[0], q[1], q[2], true, group, bits, "affine", stream));
      }
      eval(ys);
      encoder.synchronize();
    };
    run(5);
    double best = 1e30;
    for (int trial = 0; trial < 3; ++trial) {
      auto t0 = std::chrono::steady_clock::now();
      run(reps);
      double s = std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t0)
                     .count();
      best = std::fmin(best, s);
    }
    double per = best / reps;
    std::printf(
        "{\"case\": \"%s\", \"n\": %d, \"k\": %d, \"weight_bytes\": %zu, "
        "\"us_per_dispatch\": %.2f, \"GBps\": %.2f, \"max_abs_err\": %.4g, "
        "\"max_abs_ref\": %.4g, \"ok\": %s}\n",
        c.name,
        c.n,
        c.k,
        weight_bytes,
        per * 1e6,
        weight_bytes / per / 1e9,
        max_abs,
        max_ref,
        ok ? "true" : "false");
    std::fflush(stdout);
  }
  return failures == 0 ? 0 : 2;
}
