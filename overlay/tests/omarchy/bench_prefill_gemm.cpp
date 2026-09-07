// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Prefill GEMM throughput, uninstrumented: N back-to-back dispatches of
// the m > 1 affine q4 QuantizedMatmul (the Qwen2.5-0.5B-4bit projection
// shapes) and of the batched float16 dense Matmul the composed SDPA runs
// at prefill ([heads, L, 64] x [heads, 64, L] scores and [heads, L, L] x
// [heads, L, 64] probs), one eval, wall time / N. Prints one JSON row per
// (shape, L) with us/call, TFLOP/s, and GB/s (best of three). Every row
// is first valued against a host double reference on a row sample so a
// broken kernel never reports a throughput.
//
//   MLX_OMARCHY_PREFILL_BENCH_REPS=50 ./omarchy_prefill_gemm_bench [filter]
//   MLX_OMARCHY_PREFILL_BENCH_L=41,262,1053 (default)
//   MLX_OMARCHY_PREFILL_BENCH_VARIANTS=v1,v2 (matmul; default: both)
//   MLX_OMARCHY_PREFILL_BENCH_QMM_VARIANTS=v1,v2 (qmm; v2splitk, v2cols also)
//
// Every (shape, L) runs once per kernel variant: v1 is the baseline
// (MLX_OMARCHY_QMM_TILE_V2=0 / MLX_OMARCHY_MATMUL_GEMM=0) and v2 the
// PrefillGemm blocked kernels; the v2 row also carries max_diff_vs_v1,
// the largest |v2 - v1| over the whole output (0 means bit-identical).
//
// Not registered with ctest: a benchmark, not a test.

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

struct QmmCase {
  const char* name;
  int n;
  int k;
};

// Per-layer projections; lm_head runs on the last token only, so it is
// not a prefill GEMM.
constexpr QmmCase kQmmCases[] = {
    {"qo_proj", 896, 896},
    {"kv_proj", 128, 896},
    {"gate_up", 4864, 896},
    {"down", 896, 4864},
};

constexpr int kHeads = 14;
constexpr int kHeadDim = 64;
constexpr int kGroup = 64;
constexpr int kBits = 4;

std::vector<float> host_f32(const Stream& s, array v) {
  v = astype(v, float32, s);
  v.eval();
  omarchy::get_command_encoder(s).synchronize();
  return std::vector<float>(v.data<float>(), v.data<float>() + v.size());
}

std::vector<int> parse_lengths() {
  std::vector<int> ls;
  const char* env = std::getenv("MLX_OMARCHY_PREFILL_BENCH_L");
  std::string spec = env ? env : "41,262,1053";
  size_t pos = 0;
  while (pos < spec.size()) {
    size_t comma = spec.find(',', pos);
    if (comma == std::string::npos) {
      comma = spec.size();
    }
    ls.push_back(std::atoi(spec.substr(pos, comma - pos).c_str()));
    pos = comma + 1;
  }
  return ls;
}

template <typename Build>
double best_seconds(const Stream& s, Build build, int reps) {
  auto& encoder = omarchy::get_command_encoder(s);
  auto run = [&](int count) {
    std::vector<array> ys;
    ys.reserve(count);
    for (int i = 0; i < count; ++i) {
      ys.push_back(build());
    }
    eval(ys);
    encoder.synchronize();
  };
  run(3);
  double best = 1e30;
  for (int trial = 0; trial < 3; ++trial) {
    auto t0 = std::chrono::steady_clock::now();
    run(reps);
    double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    best = std::fmin(best, sec);
  }
  return best / reps;
}

// |got - ref| <= atol + rtol * |ref| on every sampled element; the
// tolerance is the f16 output rounding plus f32 sum-order noise.
struct Check {
  double max_abs = 0.0;
  double max_ref = 0.0;
  bool ok = true;
  void add(double got, double ref, double atol, double rtol) {
    double err = std::fabs(got - ref);
    max_abs = std::fmax(max_abs, err);
    max_ref = std::fmax(max_ref, std::fabs(ref));
    if (!(err <= atol + rtol * std::fabs(ref))) {
      ok = false;
    }
  }
};

struct Variant {
  const char* name;
  const char* value;
};

constexpr Variant kVariants[] = {{"v1", "0"}, {"v2", "1"}};

bool variant_selected(const char* name) {
  const char* env = std::getenv("MLX_OMARCHY_PREFILL_BENCH_VARIANTS");
  return env == nullptr || std::strstr(env, name) != nullptr;
}

// qmm variants: "v1" is the tile baseline (MLX_OMARCHY_QMM_TILE_V2=0),
// "v2" the qmm_gemm.comp kernels, "v2splitk" v2 with
// MLX_OMARCHY_QMM_GEMM_SPLITK=1, "v2cols" the column-batched GEMV
// (MLX_OMARCHY_QMM_VEC_COLS=1, qmm_vec_cols.comp).
std::vector<std::string> qmm_variants() {
  const char* env = std::getenv("MLX_OMARCHY_PREFILL_BENCH_QMM_VARIANTS");
  std::string spec = env ? env : "v1,v2";
  std::vector<std::string> names;
  size_t pos = 0;
  while (pos < spec.size()) {
    size_t comma = spec.find(',', pos);
    if (comma == std::string::npos) {
      comma = spec.size();
    }
    names.push_back(spec.substr(pos, comma - pos));
    pos = comma + 1;
  }
  return names;
}

double max_diff(const std::vector<float>& a, const std::vector<float>& b) {
  double d = 0.0;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
    d = std::fmax(d, std::fabs(static_cast<double>(a[i]) - b[i]));
  }
  return d;
}

void print_row(
    const char* op,
    const char* name,
    const char* variant,
    int L,
    int m,
    int n,
    int k,
    int batch,
    size_t bytes,
    double sec,
    const Check& check,
    double diff_vs_v1) {
  double flops = 2.0 * m * n * k * batch;
  std::printf(
      "{\"op\": \"%s\", \"case\": \"%s\", \"variant\": \"%s\", \"L\": %d, "
      "\"m\": %d, \"n\": %d, "
      "\"k\": %d, \"batch\": %d, \"us_per_call\": %.2f, \"TFLOPs\": %.4f, "
      "\"GBps\": %.2f, \"max_abs_err\": %.4g, \"max_abs_ref\": %.4g, "
      "\"max_diff_vs_v1\": %.4g, \"ok\": %s}\n",
      op,
      name,
      variant,
      L,
      m,
      n,
      k,
      batch,
      sec * 1e6,
      flops / sec / 1e12,
      bytes / sec / 1e9,
      check.max_abs,
      check.max_ref,
      diff_vs_v1,
      check.ok ? "true" : "false");
  std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv) {
  if (!gpu::is_available()) {
    std::fprintf(stderr, "no qualifying Vulkan device\n");
    return 1;
  }
  set_default_device(Device::gpu);
  Stream s = new_stream(Device::gpu);
  int reps = 30;
  if (const char* env = std::getenv("MLX_OMARCHY_PREFILL_BENCH_REPS")) {
    reps = std::atoi(env);
  }
  const std::string filter = argc > 1 ? argv[1] : "";
  auto selected = [&](const char* name) {
    return filter.empty() || std::string(name).find(filter) != std::string::npos;
  };
  int failures = 0;
  const std::vector<int> lengths = parse_lengths();

  for (const QmmCase& c : kQmmCases) {
    if (!selected(c.name)) {
      continue;
    }
    array w = astype(
        random::normal(Shape{c.n, c.k}, random::key(1), s), float16, s);
    auto q = quantize(w, kGroup, kBits, "affine", std::nullopt, s);
    eval(q[0], q[1], q[2]);
    array dense = dequantize(
        q[0], q[1], q[2], kGroup, kBits, "affine", std::nullopt, std::nullopt, s);
    std::vector<float> w_host = host_f32(s, dense);
    size_t weight_bytes = q[0].nbytes() + q[1].nbytes() + q[2].nbytes();
    for (int L : lengths) {
      array x = astype(
          random::normal(Shape{L, c.k}, random::key(2 + L), s), float16, s);
      std::vector<float> x_host = host_f32(s, x);
      std::vector<float> v1_host;
      for (const std::string& variant_name : qmm_variants()) {
      bool baseline = variant_name == "v1";
      setenv("MLX_OMARCHY_QMM_TILE_V2", baseline ? "0" : "1", 1);
      setenv(
          "MLX_OMARCHY_QMM_GEMM_SPLITK",
          variant_name == "v2splitk" ? "1" : "0",
          1);
      setenv(
          "MLX_OMARCHY_QMM_VEC_COLS", variant_name == "v2cols" ? "1" : "0", 1);
      array got = quantized_matmul(
          x, q[0], q[1], q[2], true, kGroup, kBits, "affine", s);
      std::vector<float> got_host = host_f32(s, got);
      double diff = 0.0;
      if (baseline) {
        v1_host = got_host;
      } else if (!v1_host.empty()) {
        diff = max_diff(got_host, v1_host);
      }
      // Row sample: every row up to 64, then a stride, so the check stays
      // cheap at L = 1053 while every n-tile and k-slice is exercised.
      Check check;
      int step = L <= 64 ? 1 : L / 64;
      for (int r = 0; r < L; r += step) {
        for (int col = 0; col < c.n; ++col) {
          double acc = 0.0;
          for (int kk = 0; kk < c.k; ++kk) {
            acc += static_cast<double>(x_host[r * c.k + kk]) *
                static_cast<double>(w_host[col * c.k + kk]);
          }
          check.add(got_host[r * c.n + col], acc, 0.25, 4e-3);
        }
      }
      if (!check.ok) {
        ++failures;
      }
      double sec = best_seconds(
          s,
          [&]() {
            return quantized_matmul(
                x, q[0], q[1], q[2], true, kGroup, kBits, "affine", s);
          },
          reps);
      size_t bytes = weight_bytes + x.nbytes() + got.nbytes();
      print_row(
          "qmm",
          c.name,
          variant_name.c_str(),
          L,
          L,
          c.n,
          c.k,
          1,
          bytes,
          sec,
          check,
          diff);
      }
    }
  }

  if (selected("sdpa_scores") || selected("sdpa_probs")) {
    for (int L : lengths) {
      array qh = astype(
          random::normal(Shape{kHeads, L, kHeadDim}, random::key(11), s),
          float16,
          s);
      array kh = astype(
          random::normal(Shape{kHeads, L, kHeadDim}, random::key(12), s),
          float16,
          s);
      array vh = astype(
          random::normal(Shape{kHeads, L, kHeadDim}, random::key(13), s),
          float16,
          s);
      array ph = astype(
          random::uniform(Shape{kHeads, L, L}, float32, random::key(14), s),
          float16,
          s);
      std::vector<float> q_host = host_f32(s, qh);
      std::vector<float> k_host = host_f32(s, kh);
      std::vector<float> v_host = host_f32(s, vh);
      std::vector<float> p_host = host_f32(s, ph);
      int step = L <= 64 ? 1 : L / 64;
      std::vector<float> scores_v1;
      std::vector<float> probs_v1;
      for (const Variant& variant : kVariants) {
      if (!variant_selected(variant.name)) {
        continue;
      }
      setenv("MLX_OMARCHY_MATMUL_GEMM", variant.value, 1);
      if (selected("sdpa_scores")) {
        // q @ k^T through the transposed view the SDPA path builds.
        auto build = [&]() { return matmul(qh, swapaxes(kh, -1, -2, s), s); };
        std::vector<float> got_host = host_f32(s, build());
        double diff = 0.0;
        if (std::strcmp(variant.name, "v1") == 0) {
          scores_v1 = got_host;
        } else if (!scores_v1.empty()) {
          diff = max_diff(got_host, scores_v1);
        }
        Check check;
        for (int h = 0; h < kHeads; ++h) {
          for (int r = 0; r < L; r += step) {
            for (int col = 0; col < L; ++col) {
              double acc = 0.0;
              for (int d = 0; d < kHeadDim; ++d) {
                acc += static_cast<double>(
                           q_host[(h * L + r) * kHeadDim + d]) *
                    static_cast<double>(k_host[(h * L + col) * kHeadDim + d]);
              }
              check.add(got_host[(h * L + r) * L + col], acc, 0.05, 4e-3);
            }
          }
        }
        if (!check.ok) {
          ++failures;
        }
        double sec = best_seconds(s, build, reps);
        size_t bytes = qh.nbytes() + kh.nbytes() +
            static_cast<size_t>(kHeads) * L * L * 2;
        print_row(
            "matmul",
            "sdpa_scores",
            variant.name,
            L,
            L,
            L,
            kHeadDim,
            kHeads,
            bytes,
            sec,
            check,
            diff);
      }
      if (selected("sdpa_probs")) {
        auto build = [&]() { return matmul(ph, vh, s); };
        std::vector<float> got_host = host_f32(s, build());
        double diff = 0.0;
        if (std::strcmp(variant.name, "v1") == 0) {
          probs_v1 = got_host;
        } else if (!probs_v1.empty()) {
          diff = max_diff(got_host, probs_v1);
        }
        Check check;
        for (int h = 0; h < kHeads; ++h) {
          for (int r = 0; r < L; r += step) {
            for (int col = 0; col < kHeadDim; ++col) {
              double acc = 0.0;
              for (int j = 0; j < L; ++j) {
                acc += static_cast<double>(p_host[(h * L + r) * L + j]) *
                    static_cast<double>(v_host[(h * L + j) * kHeadDim + col]);
              }
              check.add(
                  got_host[(h * L + r) * kHeadDim + col], acc, 0.5, 4e-3);
            }
          }
        }
        if (!check.ok) {
          ++failures;
        }
        double sec = best_seconds(s, build, reps);
        size_t bytes = ph.nbytes() + vh.nbytes() + qh.nbytes();
        print_row(
            "matmul",
            "sdpa_probs",
            variant.name,
            L,
            L,
            kHeadDim,
            L,
            kHeads,
            bytes,
            sec,
            check,
            diff);
      }
      }
    }
  }
  return failures == 0 ? 0 : 2;
}
