// Receipt probe for 2026-09-11-bf16-alpha-fix (NOT shipped code).
//
// Demonstrates, at the MatmulBF16Coopmat-gated attention-scores shape,
// that the alpha-scaled bf16 scores matmul agrees with a float64
// round-to-nearest reference on both routes (coopmat bf16 fast path and
// the f32 composition), and prints per-route ULP statistics.
//
// Build (see m1_window_s1.sh): g++ against the tree's built mlx + doctest.
// Run on the M1 (fork driver): the coopmat route needs
// cooperative_matrix_f32_8; on llvmpipe the fast path silently falls back
// to the non-coopmat route and the probe records coopmat_device=0.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

using namespace mlx::core;

namespace {

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
}

bool compute_available() {
  return gpu::is_available();
}

std::vector<float> pattern(size_t count, uint32_t seed) {
  std::vector<float> data(count);
  uint32_t state = seed;
  for (size_t i = 0; i < count; ++i) {
    state = state * 1664525u + 1013904223u;
    data[i] = ((state >> 8) & 0xFFFF) / 16384.0f - 2.0f;
  }
  return data;
}

// Round f32 bits to bf16 bits (RNE), the receipts' convention.
uint16_t rne_bf16_bits(float value) {
  uint32_t b;
  std::memcpy(&b, &value, 4);
  return (uint16_t)((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
}

float bf16_bits_to_float(uint16_t bits) {
  uint32_t wide = (uint32_t)bits << 16;
  float value;
  std::memcpy(&value, &wide, 4);
  return value;
}

int ordered_bits(uint16_t bits) {
  return (int16_t)bits;
}

// Host f64 attention truth on bf16-lifted inputs, returned as bf16 bits.
std::vector<uint16_t> f64_attention_bits(
    const std::vector<float>& q_data,
    const std::vector<float>& k_data,
    const std::vector<float>& v_data,
    int heads,
    int kv_heads,
    int qL,
    int kL,
    int D,
    double scale) {
  int rep = heads / kv_heads;
  std::vector<double> q64((size_t)heads * qL * D);
  std::vector<double> k64((size_t)kv_heads * kL * D);
  std::vector<double> v64((size_t)kv_heads * kL * D);
  for (int h = 0; h < heads; ++h) {
    for (int i = 0; i < qL; ++i) {
      for (int d = 0; d < D; ++d) {
        q64[((size_t)h * qL + i) * D + d] =
            bf16_bits_to_float(rne_bf16_bits(q_data[(h * qL + i) * D + d]));
      }
    }
  }
  for (int h = 0; h < kv_heads; ++h) {
    for (int i = 0; i < kL; ++i) {
      for (int d = 0; d < D; ++d) {
        k64[((size_t)h * kL + i) * D + d] =
            bf16_bits_to_float(rne_bf16_bits(k_data[(h * kL + i) * D + d]));
        v64[((size_t)h * kL + i) * D + d] =
            bf16_bits_to_float(rne_bf16_bits(v_data[(h * kL + i) * D + d]));
      }
    }
  }
  std::vector<uint16_t> out_bits((size_t)heads * qL * D);
  std::vector<double> scores(kL);
  for (int h = 0; h < heads; ++h) {
    int kv = h / rep;
    for (int i = 0; i < qL; ++i) {
      double max_s = -1e300;
      for (int j = 0; j < kL; ++j) {
        double dot = 0.0;
        for (int d = 0; d < D; ++d) {
          dot += q64[((size_t)h * qL + i) * D + d] *
              k64[((size_t)kv * kL + j) * D + d];
        }
        scores[j] = dot * scale;
        max_s = std::max(max_s, scores[j]);
      }
      double denom = 0.0;
      for (int j = 0; j < kL; ++j) {
        scores[j] = std::exp(scores[j] - max_s);
        denom += scores[j];
      }
      for (int d = 0; d < D; ++d) {
        double acc = 0.0;
        for (int j = 0; j < kL; ++j) {
          acc += (scores[j] / denom) * v64[((size_t)kv * kL + j) * D + d];
        }
        out_bits[((size_t)h * qL + i) * D + d] = rne_bf16_bits((float)acc);
      }
    }
  }
  return out_bits;
}

}  // namespace

TEST_CASE("scaled bf16 sdpa matches f64 RNE reference on both routes") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const int B = 1, H = 2, KV = 1, qL = 64, kL = 128, D = 8;
  const float scale = 0.25f;
  const auto& caps = omarchy::device(0).capabilities();
  const bool coopmat_device =
      caps.cooperative_matrix_f32_8 && caps.subgroup_size == 32;

  auto q_data = pattern(B * H * qL * D, 401);
  auto k_data = pattern(B * KV * kL * D, 409);
  auto v_data = pattern(B * KV * kL * D, 419);

  auto run_route = [&](bool fast) {
    setenv("MLX_OMARCHY_SDPA_BF16_FAST", fast ? "1" : "0", 1);
    array q = astype(
        array(q_data.begin(), Shape{B, H, qL, D}, float32), bfloat16, stream);
    array k = astype(
        array(k_data.begin(), Shape{B, KV, kL, D}, float32), bfloat16, stream);
    array v = astype(
        array(v_data.begin(), Shape{B, KV, kL, D}, float32), bfloat16, stream);
    auto out = fast::scaled_dot_product_attention(
        q, k, v, scale, "", {}, std::nullopt, false, stream);
    out.eval();
    synchronize(stream);
    const uint16_t* bits = out.data<uint16_t>();
    return std::vector<uint16_t>(bits, bits + out.size());
  };

  auto truth = f64_attention_bits(
      q_data, k_data, v_data, H, KV, qL, kL, D, (double)scale);

  auto stats = [&](const std::vector<uint16_t>& bits) {
    double sum = 0.0;
    int max_ulp = 0;
    size_t exact = 0;
    for (size_t i = 0; i < bits.size(); ++i) {
      int d = std::abs(ordered_bits(bits[i]) - ordered_bits(truth[i]));
      sum += d;
      max_ulp = std::max(max_ulp, d);
      exact += bits[i] == truth[i];
    }
    char line[160];
    std::snprintf(
        line,
        sizeof(line),
        "{\"n\":%zu,\"exact_frac\":%.6f,\"mean_ulp\":%.4f,\"max_ulp\":%d}",
        bits.size(),
        (double)exact / bits.size(),
        sum / bits.size(),
        max_ulp);
    return std::string(line);
  };

  auto fast_bits = run_route(true);
  auto slow_bits = run_route(false);
  unsetenv("MLX_OMARCHY_SDPA_BF16_FAST");

  double fast_mean = 0.0, slow_mean = 0.0;
  int fast_max = 0, slow_max = 0;
  auto parse = [&](const std::string& s, double& mean, int& maxu) {
    std::sscanf(
        s.c_str(),
        "{\"n\":%*d,\"exact_frac\":%*f,\"mean_ulp\":%lf,\"max_ulp\":%d}",
        &mean,
        &maxu);
  };
  parse(stats(fast_bits), fast_mean, fast_max);
  parse(stats(slow_bits), slow_mean, slow_max);

  std::printf(
      "PROBE coopmat_device=%d fast=%s f32comp=%s\n",
      (int)coopmat_device,
      stats(fast_bits).c_str(),
      stats(slow_bits).c_str());
  std::fflush(stdout);

  // The fixed kernel must agree with f64 truth within a few bf16 ULP.
  // A missing alpha scale (the pre-fix trap) displaces every score by
  // the full alpha factor and misses these bounds by orders of magnitude.
  CHECK_LE(fast_mean, 4.0);
  CHECK_LE(fast_max, 32);
}
