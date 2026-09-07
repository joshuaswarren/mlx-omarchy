// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Decode attention wall time, fused (the default, one or two
// dispatches; fused_tree = MLX_OMARCHY_SDPA_SUBGROUP=0) against the composed matmul -> softmax -> matmul path, on
// the Qwen2.5-0.5B decode shape (14 q heads over 2 kv heads, head_dim 64,
// one query) at several KV lengths read as cache slices. N back-to-back
// calls, one eval, wall time / N; best of three; one JSON row per
// (k_len, path). Both paths are checked against each other first.
//
//   MLX_OMARCHY_SDPA_BENCH_REPS=48 ./omarchy_sdpa_decode_bench
//
// Not registered with ctest: a benchmark, not a test.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "mlx/backend/omarchy/encoder.h"
#include "mlx/mlx.h"

using namespace mlx::core;

int main(int argc, char** argv) {
  if (!gpu::is_available()) {
    std::fprintf(stderr, "no qualifying Vulkan device\n");
    return 1;
  }
  set_default_device(Device::gpu);
  Stream stream = new_stream(Device::gpu);
  auto& encoder = omarchy::get_command_encoder(stream);
  int reps = 48;
  if (const char* env = std::getenv("MLX_OMARCHY_SDPA_BENCH_REPS")) {
    reps = std::atoi(env);
  }
  const int B = 1, H = 14, KV = 2, D = 64;
  int q_len = 1;
  if (const char* env = std::getenv("MLX_OMARCHY_SDPA_BENCH_QLEN")) {
    q_len = std::atoi(env);
  }
  const float scale = 1.0f / std::sqrt(float(D));
  std::vector<int> k_lens = {45, 280, 390, 1024, 4096};
  if (argc > 1) {
    k_lens.clear();
    for (int i = 1; i < argc; ++i) {
      k_lens.push_back(std::atoi(argv[i]));
    }
  }
  int failures = 0;
  for (int k_len : k_lens) {
    const int cache_len = ((k_len + 255) / 256) * 256;
    array q = astype(
        random::normal(Shape{B, H, q_len, D}, random::key(1), stream),
        float16,
        stream);
    array kw = astype(
        random::normal(Shape{B, KV, cache_len, D}, random::key(2), stream),
        float16,
        stream);
    array vw = astype(
        random::normal(Shape{B, KV, cache_len, D}, random::key(3), stream),
        float16,
        stream);
    eval(q, kw, vw);
    array k = slice(kw, {0, 0, 0, 0}, {B, KV, k_len, D}, stream);
    array v = slice(vw, {0, 0, 0, 0}, {B, KV, k_len, D}, stream);
    auto sdpa = [&]() {
      return fast::scaled_dot_product_attention(
          q, k, v, scale, q_len > 1 ? "causal" : "", {}, std::nullopt,
          false, stream);
    };
    auto to_host = [&](array x) {
      array wide = astype(x, float32, stream);
      eval(wide);
      encoder.synchronize();
      return std::vector<float>(wide.data<float>(), wide.data<float>() + wide.size());
    };
    setenv("MLX_OMARCHY_SDPA_FUSED", "0", 1);
    auto composed = to_host(sdpa());
    unsetenv("MLX_OMARCHY_SDPA_FUSED");
    auto fused = to_host(sdpa());
    float max_abs = 0.0f;
    for (size_t i = 0; i < composed.size(); ++i) {
      max_abs = std::fmax(max_abs, std::fabs(composed[i] - fused[i]));
    }
    // Outputs are weighted means of N(0,1) values: float16 output
    // rounding plus the composed path's float16 score storage.
    bool ok = max_abs <= 2e-2f;
    if (!ok) {
      ++failures;
    }
    for (const char* path : {"composed", "fused", "fused_tree"}) {
      if (std::string(path) == "composed") {
        setenv("MLX_OMARCHY_SDPA_FUSED", "0", 1);
      } else {
        unsetenv("MLX_OMARCHY_SDPA_FUSED");
      }
      if (std::string(path) == "fused_tree") {
        setenv("MLX_OMARCHY_SDPA_SUBGROUP", "0", 1);
      } else {
        unsetenv("MLX_OMARCHY_SDPA_SUBGROUP");
      }
      auto run = [&](int count) {
        std::vector<array> ys;
        ys.reserve(count);
        for (int i = 0; i < count; ++i) {
          ys.push_back(sdpa());
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
      std::printf(
          "{\"k_len\": %d, \"q_len\": %d, \"path\": \"%s\", \"reps\": %d, "
          "\"us_per_call\": %.2f, \"max_abs_diff\": %.4g, \"ok\": %s}\n",
          k_len,
          q_len,
          path,
          reps,
          best / reps * 1e6,
          max_abs,
          ok ? "true" : "false");
      std::fflush(stdout);
    }
  }
  return failures == 0 ? 0 : 2;
}
