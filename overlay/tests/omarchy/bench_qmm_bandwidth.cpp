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
//   MLX_OMARCHY_QMM_BENCH_REPS=100 MLX_OMARCHY_QMM_BENCH_TRIALS=3 \
//       ./omarchy_qmm_bandwidth_bench [case-substring]
//
// Not registered with ctest: the 68 MB lm_head shape is a benchmark, not
// a test, and takes minutes on llvmpipe.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    // Not a model shape: one 8-row GEMV so the per-dispatch floor (host
    // graph + record + launch) can be read off the same harness.
    {"floor 8x64", 8, 64},
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
  int trials = 3;
  if (const char* env = std::getenv("MLX_OMARCHY_QMM_BENCH_TRIALS")) {
    trials = std::atoi(env);
  }
  const std::string filter = argc > 1 ? argv[1] : "";
  constexpr int group = 64;
  constexpr int bits = 4;
  // MLX_OMARCHY_QMM_BENCH_PAIR=1 interleaves every timed dispatch with
  // one 8x64 dispatch: if the per-op host floor overlaps GPU execution
  // the pair costs about max(2 * floor, kernel), if it serializes it
  // costs floor + kernel.
  const char* pair_env = std::getenv("MLX_OMARCHY_QMM_BENCH_PAIR");
  const bool pair = pair_env != nullptr && std::strcmp(pair_env, "1") == 0;
  array tiny_w = astype(
      random::normal(Shape{8, 64}, random::key(3), stream), float16, stream);
  auto tiny_q = quantize(tiny_w, group, bits, "affine", std::nullopt, stream);
  array tiny_x = astype(
      random::normal(Shape{1, 64}, random::key(4), stream), float16, stream);
  eval(tiny_q[0], tiny_q[1], tiny_q[2], tiny_x);
  // MLX_OMARCHY_QMM_BENCH_CONFIGS: comma-separated kernel configurations
  // timed side by side: "old" (MLX_OMARCHY_QMM_VEC_Q4_V2=0), "env" (the
  // process environment as is, so after another configuration it
  // inherits that one's variables), or "rN[tT]" for the DecodeQ4Vec kernel
  // with N rows per slot and T workgroup threads (MLX_OMARCHY_QMM_Q4_ROWS
  // / MLX_OMARCHY_QMM_Q4_WG; "r2t128" is the shipped default).
  struct Config {
    std::string name;
    void apply() const {
      if (name == "env") {
        return;
      }
      if (name == "old") {
        setenv("MLX_OMARCHY_QMM_VEC_Q4_V2", "0", 1);
        return;
      }
      setenv("MLX_OMARCHY_QMM_VEC_Q4_V2", "1", 1);
      unsetenv("MLX_OMARCHY_QMM_Q4_ROWS");
      unsetenv("MLX_OMARCHY_QMM_Q4_WG");
      std::string s = name;
      size_t t = s.find('t');
      if (t != std::string::npos) {
        setenv("MLX_OMARCHY_QMM_Q4_WG", s.substr(t + 1).c_str(), 1);
        s = s.substr(0, t);
      }
      if (s.rfind("r", 0) == 0) {
        setenv("MLX_OMARCHY_QMM_Q4_ROWS", s.substr(1).c_str(), 1);
      }
    }
  };
  std::vector<Config> configs;
  if (const char* env = std::getenv("MLX_OMARCHY_QMM_BENCH_CONFIGS")) {
    std::string list = env;
    size_t start = 0;
    while (start <= list.size()) {
      size_t comma = list.find(',', start);
      if (comma == std::string::npos) {
        comma = list.size();
      }
      if (comma > start) {
        configs.push_back({list.substr(start, comma - start)});
      }
      start = comma + 1;
    }
  }
  if (configs.empty()) {
    configs.push_back({"env"});
  }
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

    // Correctness gate: every configuration against the dequantized
    // dense matmul on the same device. Each configuration is checked on
    // its own scaled copy of x (distinct sign and magnitude), so a kernel
    // that leaves rows unwritten cannot pass on values an earlier
    // configuration left in the recycled output buffer.
    array dense_t = transpose(
        astype(dequantize(q[0], q[1], q[2], group, bits, "affine",
                   std::nullopt, std::nullopt, stream),
            float32, stream),
        stream);
    std::vector<float> max_abs(configs.size(), 0.0f);
    std::vector<bool> ok(configs.size(), true);
    float max_ref = 0.0f;
    for (size_t cfg = 0; cfg < configs.size(); ++cfg) {
      float scale = (cfg % 2 == 0 ? 1.0f : -1.0f) * (1.0f + 0.5f * (cfg / 2));
      array x_cfg = astype(
          multiply(astype(x, float32, stream), array(scale), stream),
          float16, stream);
      array ref = matmul(astype(x_cfg, float32, stream), dense_t, stream);
      configs[cfg].apply();
      array got = astype(
          quantized_matmul(x_cfg, q[0], q[1], q[2], true, group, bits,
              "affine", stream),
          float32, stream);
      eval(ref, got);
      encoder.synchronize();
      float ref_max = 0.0f;
      for (int i = 0; i < c.n; ++i) {
        max_abs[cfg] = std::fmax(
            max_abs[cfg], std::fabs(ref.data<float>()[i] - got.data<float>()[i]));
        ref_max = std::fmax(ref_max, std::fabs(ref.data<float>()[i]));
      }
      // f16 output rounding of |ref| plus f32 sum-order noise.
      ok[cfg] = max_abs[cfg] <= 0.15f * std::fmax(1.0f, ref_max / 32.0f);
      max_ref = std::fmax(max_ref, ref_max);
      if (!ok[cfg]) {
        ++failures;
      }
    }

    auto run = [&](int count) {
      std::vector<array> ys;
      ys.reserve(count);
      for (int i = 0; i < count; ++i) {
        ys.push_back(quantized_matmul(
            x, q[0], q[1], q[2], true, group, bits, "affine", stream));
        if (pair) {
          ys.push_back(quantized_matmul(
              tiny_x, tiny_q[0], tiny_q[1], tiny_q[2], true, group, bits,
              "affine", stream));
        }
      }
      eval(ys);
      encoder.synchronize();
    };
    // Configurations are interleaved trial by trial so GPU clock state
    // and host noise hit every one alike; the reported number is the
    // best trial per configuration.
    std::vector<double> best(configs.size(), 1e30);
    for (int trial = 0; trial < trials + 1; ++trial) {
      for (size_t cfg = 0; cfg < configs.size(); ++cfg) {
        configs[cfg].apply();
        auto t0 = std::chrono::steady_clock::now();
        run(reps);
        double s = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - t0)
                       .count();
        // Trial 0 is the untimed warm-up that ramps the GPU clock.
        if (trial > 0) {
          best[cfg] = std::fmin(best[cfg], s);
        }
      }
    }
    for (size_t cfg = 0; cfg < configs.size(); ++cfg) {
      double per = best[cfg] / reps;
      std::printf(
          "{\"case\": \"%s\", \"config\": \"%s\", \"n\": %d, \"k\": %d, "
          "\"weight_bytes\": %zu, \"us_per_dispatch\": %.2f, \"GBps\": %.2f, "
          "\"max_abs_err\": %.4g, \"max_abs_ref\": %.4g, \"ok\": %s}\n",
          c.name,
          configs[cfg].name.c_str(),
          c.n,
          c.k,
          weight_bytes,
          per * 1e6,
          weight_bytes / per / 1e9,
          max_abs[cfg],
          max_ref,
          ok[cfg] ? "true" : "false");
    }
    std::fflush(stdout);
  }
  return failures == 0 ? 0 : 2;
}
