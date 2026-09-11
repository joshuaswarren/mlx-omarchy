// Differential alpha instrument for 2026-09-11-bf16-alpha-fix (NOT shipped code).
//
// The receipt's original probe (matmul_alpha_f64_probe.cpp) compares the
// whole attention output against a plain f64 oracle - every intermediate
// in f64 - so its max-ULP leg measures the fast route's documented bf16
// score/prob storage rounding as much as the alpha mechanism. This
// instrument separates the two questions:
//
//   Leg A (alpha == 1 identity): direct bf16 matmuls at coopmat-gated
//     shapes and an sdpa fast run at scale == 1.0 print FNV-1a-64 digests
//     of the raw output bits. The same binary source is linked against
//     the fixed tree and against the trap tree (pre-fix shader + the
//     fix's relaxed gate); identical digests across the two binaries is
//     the empirical alpha==1 bit-identity of the fixed kernel against the
//     pre-fix kernel.
//
//   Leg B (alpha != 1, the shape the fix exists to make correct): sdpa
//     fast at the original probe's exact shape and seeds, measured
//     against TWO oracles side by side:
//       plain   - f64 round-to-nearest everywhere, one final rounding
//                 (the original probe's oracle).
//       storage - the same f64 reference but modelling the route's
//                 documented narrow storage: scores stored bf16 after
//                 the alpha-scaled matmul, softmax in f64 over the
//                 stored scores, probs stored bf16, PV accumulated in
//                 f64 over the stored probs and bf16-lifted v.
//     The f32 composition route is measured the same way as a control.
//
// Measurement only - no CHECKs. Gate evaluation happens in analysis
// (receipt README / digest of DIFF lines), so the same binary can run on
// the fixed and trap builds without flipping exit semantics.
//
// Build (see the S1 rebuild procedure): g++ -DMLX_STATIC against the
// tree's built libmlx.a + doctest on the M1; label via
// MLX_ALPHA_DIFF_LABEL; optional raw dumps via MLX_ALPHA_DIFF_DUMP.

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

uint64_t fnv1a64(const void* data, size_t bytes) {
  const uint8_t* p = (const uint8_t*)data;
  uint64_t h = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < bytes; ++i) {
    h ^= p[i];
    h *= 0x100000001b3ull;
  }
  return h;
}

void dump_bits(const char* leg, const std::vector<uint16_t>& bits) {
  const char* dir = std::getenv("MLX_ALPHA_DIFF_DUMP");
  if (dir == nullptr || dir[0] == '\0') {
    return;
  }
  char path[512];
  std::snprintf(path, sizeof(path), "%s/%s.bin", dir, leg);
  FILE* f = std::fopen(path, "wb");
  if (f != nullptr) {
    std::fwrite(bits.data(), 2, bits.size(), f);
    std::fclose(f);
  }
}

struct UlpStats {
  size_t n = 0;
  double exact_frac = 0;
  double mean_ulp = 0;
  int max_ulp = 0;
};

UlpStats stats_vs(const std::vector<uint16_t>& bits,
                  const std::vector<uint16_t>& ref) {
  UlpStats s;
  s.n = bits.size();
  double sum = 0;
  size_t exact = 0;
  for (size_t i = 0; i < bits.size(); ++i) {
    int d = std::abs(ordered_bits(bits[i]) - ordered_bits(ref[i]));
    sum += d;
    s.max_ulp = std::max(s.max_ulp, d);
    exact += bits[i] == ref[i];
  }
  s.exact_frac = (double)exact / s.n;
  s.mean_ulp = sum / s.n;
  return s;
}

// Host f64 attention truth on bf16-lifted inputs, returned as bf16 bits.
// storage_model=false: everything f64, one final rounding (the original
// probe's plain oracle). storage_model=true: the route's documented
// narrow storage - scores bf16 after the scaled matmul, f64 softmax over
// the stored scores, probs bf16, f64 PV over stored probs.
std::vector<uint16_t> f64_attention_bits(
    const std::vector<float>& q_data,
    const std::vector<float>& k_data,
    const std::vector<float>& v_data,
    int heads,
    int kv_heads,
    int qL,
    int kL,
    int D,
    double scale,
    bool storage_model) {
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
        if (storage_model) {
          scores[j] = bf16_bits_to_float(rne_bf16_bits((float)scores[j]));
        }
        max_s = std::max(max_s, scores[j]);
      }
      double denom = 0.0;
      std::vector<double> probs(kL);
      for (int j = 0; j < kL; ++j) {
        probs[j] = std::exp(scores[j] - max_s);
        denom += probs[j];
      }
      for (int d = 0; d < D; ++d) {
        double acc = 0.0;
        for (int j = 0; j < kL; ++j) {
          double p = probs[j] / denom;
          if (storage_model) {
            p = bf16_bits_to_float(rne_bf16_bits((float)p));
          }
          acc += p * v64[((size_t)kv * kL + j) * D + d];
        }
        out_bits[((size_t)h * qL + i) * D + d] = rne_bf16_bits((float)acc);
      }
    }
  }
  return out_bits;
}

void print_ulp(const char* leg,
               double scale,
               const char* oracle,
               const UlpStats& s) {
  char line[256];
  std::snprintf(
      line,
      sizeof(line),
      "DIFF {\"leg\":\"%s\",\"scale\":%.4f,\"vs\":\"%s\",\"n\":%zu,"
      "\"exact_frac\":%.6f,\"mean_ulp\":%.4f,\"max_ulp\":%d}",
      leg,
      scale,
      oracle,
      s.n,
      s.exact_frac,
      s.mean_ulp,
      s.max_ulp);
  std::printf("%s\n", line);
}

void print_ident(const char* leg, const std::vector<uint16_t>& bits) {
  char line[160];
  std::snprintf(
      line,
      sizeof(line),
      "IDENT {\"leg\":\"%s\",\"n\":%zu,\"fnv1a64\":\"0x%016llx\"}",
      leg,
      bits.size(),
      (unsigned long long)fnv1a64(bits.data(), bits.size() * 2));
  std::printf("%s\n", line);
  dump_bits(leg, bits);
}

std::vector<uint16_t> matmul_bits(
    int m,
    int k,
    int n,
    uint32_t seed,
    Stream stream) {
  auto a_data = pattern((size_t)m * k, seed);
  auto b_data = pattern((size_t)k * n, seed + 101);
  array a = astype(
      array(a_data.begin(), Shape{m, k}, float32), bfloat16, stream);
  array b = astype(
      array(b_data.begin(), Shape{k, n}, float32), bfloat16, stream);
  auto out = matmul(a, b, stream);
  out.eval();
  synchronize(stream);
  const uint16_t* bits = out.data<uint16_t>();
  return std::vector<uint16_t>(bits, bits + out.size());
}

// The original probe's exact attention workload (shape and seeds).
constexpr int kB = 1, kH = 2, kKV = 1, kQL = 64, kKL = 128, kD = 8;

std::vector<uint16_t> sdpa_bits(
    const std::vector<float>& q_data,
    const std::vector<float>& k_data,
    const std::vector<float>& v_data,
    float scale,
    bool fast,
    Stream stream) {
  setenv("MLX_OMARCHY_SDPA_BF16_FAST", fast ? "1" : "0", 1);
  array q = astype(
      array(q_data.begin(), Shape{kB, kH, kQL, kD}, float32), bfloat16,
      stream);
  array k = astype(
      array(k_data.begin(), Shape{kB, kKV, kKL, kD}, float32), bfloat16,
      stream);
  array v = astype(
      array(v_data.begin(), Shape{kB, kKV, kKL, kD}, float32), bfloat16,
      stream);
  auto out = fast::scaled_dot_product_attention(
      q, k, v, scale, "", {}, std::nullopt, false, stream);
  out.eval();
  synchronize(stream);
  const uint16_t* bits = out.data<uint16_t>();
  return std::vector<uint16_t>(bits, bits + out.size());
}

}  // namespace

TEST_CASE("bf16 coopmat alpha differential (identity + dual-oracle)") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& caps = omarchy::device(0).capabilities();
  const bool coopmat_device =
      caps.cooperative_matrix_f32_8 && caps.subgroup_size == 32;
  const char* label = std::getenv("MLX_ALPHA_DIFF_LABEL");
  label = (label == nullptr || label[0] == '\0') ? "unlabeled" : label;
  std::printf(
      "META {\"label\":\"%s\",\"coopmat_device\":%d,"
      "\"subgroup_size\":%d}\n",
      label,
      (int)coopmat_device,
      (int)caps.subgroup_size);

  // ---- Leg A: alpha == 1 identity digests (compare across builds) ----
  struct MatShape {
    int m, k, n;
  };
  const MatShape shapes[] = {
      {32, 16, 32}, {64, 32, 64}, {96, 64, 96}, {128, 8, 64}};
  char leg[64];
  for (const auto& s : shapes) {
    std::snprintf(leg, sizeof(leg), "mat-%dx%dx%d", s.m, s.k, s.n);
    print_ident(
        leg, matmul_bits(s.m, s.k, s.n, 700 + (uint32_t)s.m, stream));
  }
  {
    auto q = pattern(kB * kH * kQL * kD, 401);
    auto k = pattern(kB * kKV * kKL * kD, 409);
    auto v = pattern(kB * kKV * kKL * kD, 419);
    print_ident("sdpa-scale1", sdpa_bits(q, k, v, 1.0f, true, stream));
  }

  // ---- Leg B: alpha != 1 differential at the probe's shape/seeds ----
  const float scale = 0.25f;
  auto q = pattern(kB * kH * kQL * kD, 401);
  auto k = pattern(kB * kKV * kKL * kD, 409);
  auto v = pattern(kB * kKV * kKL * kD, 419);
  auto ref_plain =
      f64_attention_bits(q, k, v, kH, kKV, kQL, kKL, kD, scale, false);
  auto ref_storage =
      f64_attention_bits(q, k, v, kH, kKV, kQL, kKL, kD, scale, true);
  print_ident("ref-plain", ref_plain);
  print_ident("ref-storage", ref_storage);

  auto fast_bits = sdpa_bits(q, k, v, scale, true, stream);
  auto comp_bits = sdpa_bits(q, k, v, scale, false, stream);
  print_ident("fast-scale025", fast_bits);
  print_ident("f32comp-scale025", comp_bits);

  print_ulp("fast", scale, "plain", stats_vs(fast_bits, ref_plain));
  print_ulp("fast", scale, "storage", stats_vs(fast_bits, ref_storage));
  print_ulp("f32comp", scale, "plain", stats_vs(comp_bits, ref_plain));
  print_ulp("f32comp", scale, "storage", stats_vs(comp_bits, ref_storage));

  std::fflush(stdout);
}
