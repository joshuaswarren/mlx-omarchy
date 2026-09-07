// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// The fused float16 decode attention kernels (shaders/sdpa_decode.comp,
// sdpa_decode_combine.comp; MLX_OMARCHY_SDPA_FUSED=0 forces the composed path)
// against host double math and against the composed matmul -> softmax ->
// matmul path they replace. The fused path keeps scores and probs in
// float where the composed path stores float16, so it is held to the
// double reference and required to track it at least as well as the
// composed path; each case prints both errors for the record.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/encoder.h"
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
  if (!gpu::is_available()) {
    std::cout << "Skipping: no qualifying Vulkan device (set"
                 " MLX_OMARCHY_ALLOW_NON_APPLE=1 on a development machine).\n";
    return false;
  }
  return true;
}

std::vector<float> flat(const array& value, Stream stream) {
  array copy = astype(value, float32, stream);
  copy.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const float* data = copy.data<float>();
  return std::vector<float>(data, data + copy.size());
}

// Deterministic values in [-1, 1).
std::vector<float> pattern(size_t count, uint32_t seed) {
  std::vector<float> values;
  values.reserve(count);
  uint32_t state = seed;
  for (size_t index = 0; index < count; ++index) {
    state = state * 1664525u + 1013904223u;
    values.push_back(
        static_cast<float>(static_cast<double>(state % 20000u) / 10000.0) -
        1.0f);
  }
  return values;
}

struct Case {
  const char* name;
  int batch;
  int heads;
  int kv_heads;
  int q_len;
  int k_len;
  int head_dim;
  const char* mask_mode; // "", "causal", "additive", "bool"
};

// Upstream fallback semantics in double: scale, GQA unflatten, softmax
// over additive (or bool -> 0 / -inf) masks.
std::vector<double> host_attention(
    const Case& c,
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    double scale,
    const std::vector<float>& additive_mask,
    const std::vector<bool>& bool_mask) {
  int repeats = c.heads / c.kv_heads;
  std::vector<double> out;
  std::vector<double> scores(c.k_len);
  for (int b = 0; b < c.batch; ++b) {
    for (int h = 0; h < c.heads; ++h) {
      int kv = h / repeats;
      for (int qi = 0; qi < c.q_len; ++qi) {
        double max_score = -INFINITY;
        for (int ki = 0; ki < c.k_len; ++ki) {
          double dot = 0.0;
          for (int d = 0; d < c.head_dim; ++d) {
            dot += double(q[(((b * c.heads + h) * c.q_len) + qi) * c.head_dim + d]) *
                double(k[(((b * c.kv_heads + kv) * c.k_len) + ki) * c.head_dim + d]);
          }
          double score = dot * scale;
          size_t mask_index = ((size_t(b) * c.heads + h) * c.q_len + qi) * c.k_len + ki;
          if (std::string(c.mask_mode) == "causal" &&
              ki > c.k_len - c.q_len + qi) {
            score = -INFINITY;
          } else if (!additive_mask.empty()) {
            score += additive_mask[mask_index];
          } else if (!bool_mask.empty() && !bool_mask[qi * c.k_len + ki]) {
            score = -INFINITY;
          }
          scores[ki] = score;
          max_score = std::max(max_score, score);
        }
        double total = 0.0;
        for (int ki = 0; ki < c.k_len; ++ki) {
          scores[ki] = std::exp(scores[ki] - max_score);
          total += scores[ki];
        }
        for (int d = 0; d < c.head_dim; ++d) {
          double acc = 0.0;
          for (int ki = 0; ki < c.k_len; ++ki) {
            acc += scores[ki] *
                double(v[(((b * c.kv_heads + kv) * c.k_len) + ki) * c.head_dim + d]);
          }
          out.push_back(acc / total);
        }
      }
    }
  }
  return out;
}

struct FusedGate {
  // The primitive reads the variable per call, so flipping it between
  // two evaluations selects the path.
  static void composed() {
    setenv("MLX_OMARCHY_SDPA_FUSED", "0", 1);
  }
  static void fused() {
    unsetenv("MLX_OMARCHY_SDPA_FUSED");
  }
};

void run_case(const Case& c, Stream stream) {
  const int cache_len = c.k_len + 5;
  const double scale = 1.0 / std::sqrt(double(c.head_dim));
  auto q_data = pattern(size_t(c.batch) * c.heads * c.q_len * c.head_dim, 11u + c.head_dim);
  auto k_wide = pattern(size_t(c.batch) * c.kv_heads * cache_len * c.head_dim, 23u + c.k_len);
  auto v_wide = pattern(size_t(c.batch) * c.kv_heads * cache_len * c.head_dim, 37u + c.q_len);
  // q as the model produces it: a transposed [B, L, H, D] projection.
  array q_blhd = astype(
      array(q_data.begin(), Shape{c.batch, c.heads, c.q_len, c.head_dim}, float32),
      float16,
      stream);
  array q = transpose(transpose(q_blhd, {0, 2, 1, 3}, stream), {0, 2, 1, 3}, stream);
  // k, v as KV-cache slices: float16 storage sliced without a copy.
  array kw = astype(
      array(k_wide.begin(), Shape{c.batch, c.kv_heads, cache_len, c.head_dim}, float32),
      float16,
      stream);
  array vw = astype(
      array(v_wide.begin(), Shape{c.batch, c.kv_heads, cache_len, c.head_dim}, float32),
      float16,
      stream);
  array k = slice(kw, {0, 0, 0, 0}, {c.batch, c.kv_heads, c.k_len, c.head_dim}, stream);
  array v = slice(vw, {0, 0, 0, 0}, {c.batch, c.kv_heads, c.k_len, c.head_dim}, stream);
  auto compact = [&](const std::vector<float>& wide) {
    std::vector<float> rows;
    for (int b = 0; b < c.batch; ++b) {
      for (int kv = 0; kv < c.kv_heads; ++kv) {
        for (int ki = 0; ki < c.k_len; ++ki) {
          for (int d = 0; d < c.head_dim; ++d) {
            rows.push_back(wide[(((b * c.kv_heads + kv) * cache_len) + ki) * c.head_dim + d]);
          }
        }
      }
    }
    return rows;
  };
  // Round the inputs the way the device holds them.
  auto q_h = flat(q, stream);
  auto k_h = compact(flat(kw, stream));
  auto v_h = compact(flat(vw, stream));

  std::string mode(c.mask_mode);
  std::optional<array> mask;
  std::vector<float> additive;
  std::vector<bool> bools;
  if (mode == "additive") {
    auto raw = pattern(size_t(c.batch) * c.heads * c.q_len * c.k_len, 41u);
    array m = astype(
        array(raw.begin(), Shape{c.batch, c.heads, c.q_len, c.k_len}, float32),
        float16,
        stream);
    additive = flat(m, stream);
    mask = m;
  } else if (mode == "bool") {
    // Broadcast [1, 1, q_len, k_len] bool mask: every third key off,
    // which lands in the primitive as a stride-0 additive mask.
    bools.resize(size_t(c.q_len) * c.k_len);
    std::vector<uint8_t> bytes(bools.size());
    for (size_t index = 0; index < bools.size(); ++index) {
      bool keep = (index % 3) != 2;
      bools[index] = keep;
      bytes[index] = keep ? 1 : 0;
    }
    mask = array(bytes.begin(), Shape{1, 1, c.q_len, c.k_len}, bool_);
  }
  std::string mask_mode = mode == "causal" ? "causal" : "";

  auto run = [&]() {
    return fast::scaled_dot_product_attention(
        q, k, v, float(scale), mask_mode, mask, std::nullopt, false, stream);
  };
  FusedGate::composed();
  auto composed = flat(run(), stream);
  FusedGate::fused();
  auto fused = flat(run(), stream);

  REQUIRE_EQ(fused.size(), composed.size());
  REQUIRE_EQ(fused.size(), size_t(c.batch) * c.heads * c.q_len * c.head_dim);
  auto want = host_attention(c, q_h, k_h, v_h, scale, additive, bools);
  double worst_fused = 0.0;
  double worst_composed = 0.0;
  double sum_fused = 0.0;
  double sum_composed = 0.0;
  size_t nan_count = 0;
  for (size_t index = 0; index < want.size(); ++index) {
    if (std::isnan(fused[index])) {
      ++nan_count;
      continue;
    }
    double fused_diff = std::abs(double(fused[index]) - want[index]);
    double composed_diff = std::abs(double(composed[index]) - want[index]);
    worst_fused = std::max(worst_fused, fused_diff);
    worst_composed = std::max(worst_composed, composed_diff);
    sum_fused += fused_diff;
    sum_composed += composed_diff;
  }
  double mean_fused = sum_fused / double(want.size());
  double mean_composed = sum_composed / double(want.size());
  std::cout << c.name << ": max|err| fused " << worst_fused << " composed "
            << worst_composed << "; mean|err| fused " << mean_fused
            << " composed " << mean_composed << "\n";
  CHECK_MESSAGE(nan_count == 0, c.name, ": ", nan_count, " NaN outputs");
  // Outputs are averages of values in [-1, 1) stored in float16 (ulp
  // 2^-11 below 1): one storage rounding plus float accumulation.
  CHECK_MESSAGE(
      worst_fused <= 2e-3,
      c.name,
      ": fused vs host double worst ",
      worst_fused);
  // Float scores and probs must not lose to the composed path's float16
  // score/prob storage; a float16 output ulp of slack covers ties.
  CHECK_MESSAGE(
      mean_fused <= mean_composed + 5e-4,
      c.name,
      ": fused mean error ",
      mean_fused,
      " exceeds composed ",
      mean_composed);
}

} // namespace

TEST_CASE("fused decode sdpa tracks the double reference at least as well as the composed path") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<Case> cases = {
      // Qwen2.5-0.5B decode: 14 q heads over 2 kv heads, one query.
      {"qwen decode", 1, 14, 2, 1, 53, 64, ""},
      {"qwen decode causal", 1, 14, 2, 1, 53, 64, "causal"},
      // q_len 8, two probs chunks (k_len > 256), head_dim 128 (2 rows/pass).
      {"q8 causal d128", 2, 8, 2, 8, 300, 128, "causal"},
      // head_dim 96: 2 rows per pass with idle lanes; chunk boundary + 1.
      {"d96 causal k257", 1, 6, 3, 5, 257, 96, "causal"},
      // MHA, single key, 8 rows per pass.
      {"mha d32 k1", 1, 3, 3, 2, 1, 32, ""},
      // Additive float16 mask through its strides.
      {"additive mask d80", 1, 4, 1, 3, 70, 80, "additive"},
      // Bool mask: broadcast additive mask with stride-0 axes.
      {"bool mask", 2, 4, 2, 4, 40, 64, "bool"},
      // head_dim 16 (matmul tile padding), three key blocks.
      {"d16 k520", 1, 2, 1, 1, 520, 16, ""},
      // Qwen shapes at four and five key blocks (split-K combine).
      {"qwen k1000", 1, 14, 2, 1, 1000, 64, ""},
      {"qwen q8 k1100 causal", 1, 14, 2, 8, 1100, 64, "causal"},
      // Additive mask across blocks.
      {"additive mask k700", 1, 4, 2, 2, 700, 64, "additive"},
      // Every kv head of batch 3 lands on its own workgroup.
      {"batch3", 3, 4, 2, 2, 17, 64, "causal"},
  };
  for (const auto& c : cases) {
    run_case(c, stream);
  }
}

TEST_CASE("fused decode sdpa keeps a fully masked row uniform") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const int B = 1, H = 2, KV = 1, qL = 1, kL = 9, D = 32;
  auto q_data = pattern(size_t(B) * H * qL * D, 3u);
  auto k_data = pattern(size_t(B) * KV * kL * D, 5u);
  auto v_data = pattern(size_t(B) * KV * kL * D, 7u);
  array q = astype(array(q_data.begin(), Shape{B, H, qL, D}, float32), float16, stream);
  array k = astype(array(k_data.begin(), Shape{B, KV, kL, D}, float32), float16, stream);
  array v = astype(array(v_data.begin(), Shape{B, KV, kL, D}, float32), float16, stream);
  array mask = array(false);
  auto run = [&]() {
    return fast::scaled_dot_product_attention(
        q, k, v, 0.25f, "", mask, std::nullopt, false, stream);
  };
  FusedGate::composed();
  auto composed = flat(run(), stream);
  FusedGate::fused();
  auto fused = flat(run(), stream);
  auto v_h = flat(v, stream);
  REQUIRE_EQ(fused.size(), size_t(B * H * qL * D));
  for (size_t index = 0; index < fused.size(); ++index) {
    CHECK(std::abs(double(fused[index]) - double(composed[index])) < 2e-3);
    double mean = 0.0;
    for (int ki = 0; ki < kL; ++ki) {
      mean += v_h[ki * D + (index % D)];
    }
    mean /= kL;
    CHECK(std::abs(double(fused[index]) - mean) < 2e-2);
  }
}
