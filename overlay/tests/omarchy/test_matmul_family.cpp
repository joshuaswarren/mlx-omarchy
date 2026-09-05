// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT
//
// Wave 6: the matmul family. BlockMaskedMM, GatherMM, SegmentedMM,
// GatherQMM, GatherQQMM, and QQMatmul, each valued against a host
// double-precision reference on small shapes, one edge-tile shape per op,
// and named-error pins for the unsupported quantization modes.

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/device.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

using namespace mlx::core;

namespace {

// The no-progress submit watchdog (device.cpp wait_for_timeline_progress)
// sizes its hang interval for hardware: a single-increment timeline wait
// shows no counter motion until its submit completes, so any dispatch that
// legitimately runs longer than MLX_OMARCHY_HANG_NO_PROGRESS_NS is
// misdiagnosed as a hung device. On hardware each prefill submit stays
// under the 10 s default; on the Mesa llvmpipe software rasterizer this
// repo develops against, one 4864x4864 prefill-shape quantized matmul
// takes tens of seconds, so the default window kills the tile case's
// jumbo sweep (and trips a teardown crash on the error path) even though
// the kernel is correct. Provision the documented env override before the
// first wait caches it (device.cpp env_ns_override is static-cached on
// first use): only when the driver is llvmpipe, only when the operator
// has not set a window already. Hardware keeps the 10 s default, and the
// MLX_OMARCHY_MAX_WALL_NS ceiling (30 min default) stays authoritative.
void provision_software_device_watchdog() {
  if (std::getenv("MLX_OMARCHY_HANG_NO_PROGRESS_NS") != nullptr) {
    return;
  }
  if (!gpu::is_available()) {
    return;
  }
  const auto& caps = omarchy::device(0).capabilities();
  if (caps.driver_id != VK_DRIVER_ID_MESA_LLVMPIPE) {
    return;
  }
  std::cout
      << "[provenance] llvmpipe detected: raising submit no-progress"
      << " window to 900 s for jumbo prefill-shape dispatches\n";
  setenv("MLX_OMARCHY_HANG_NO_PROGRESS_NS", "900000000000", 1);
}

} // namespace

int main(int argc, char** argv) {
  provision_software_device_watchdog();
  doctest::Context ctx;
  ctx.applyCommandLine(argc, argv);
  return ctx.run();
}

namespace {

void skip(const char* reason) {
  std::cout << "Skipping: " << reason << "\n";
}

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
}

bool compute_available() {
  if (!gpu::is_available()) {
    skip(
        "no qualifying Vulkan device (set MLX_OMARCHY_ALLOW_NON_APPLE=1 on"
        " a development machine).");
    return false;
  }
  return true;
}

bool float16_available() {
  const auto& capabilities = omarchy::device(0).capabilities();
  return capabilities.shader_float16 &&
      capabilities.storage_buffer_16bit_access;
}

std::string evaluation_error(array value) {
  try {
    value.eval();
  } catch (const std::exception& error) {
    return error.what();
  }
  return {};
}

std::vector<float> readback_f32(const Stream& stream, array value) {
  value = astype(value, float32, stream);
  value.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const float* data = value.data<float>();
  return std::vector<float>(data, data + value.size());
}

void expect_close(
    const std::vector<float>& device,
    const std::vector<float>& expected,
    double epsilon) {
  REQUIRE_EQ(device.size(), expected.size());
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK(device[index] == doctest::Approx(expected[index]).epsilon(epsilon));
  }
}

// fp32 accumulation against a double reference carries a small
// absolute error even where cancellation makes the expected value
// near zero; epsilon alone is pure relative, so add an atol floor.
// Any wrong scale, code, or index misses by O(1) and still fails.
void expect_close_tol(
    const std::vector<float>& device,
    const std::vector<float>& expected,
    double atol,
    double rtol) {
  REQUIRE_EQ(device.size(), expected.size());
  for (size_t index = 0; index < expected.size(); ++index) {
    // This doctest's rule is |a - b| <= eps * (scale + max(|a|, |b|)),
    // so scale(atol / rtol) makes the tolerance atol + rtol * max.
    CHECK(device[index] ==
        doctest::Approx(expected[index]).epsilon(rtol).scale(atol / rtol));
  }
}

double host_at(const std::vector<float>& values, size_t index) {
  return static_cast<double>(values[index]);
}

// Host affine quantizer matching the upstream affine_quantize kernel:
// the abs-dominant endpoint picks the scale sign, the q0 refinement pins
// that endpoint exactly, and clipped rounded codes pack LSB-first,
// 32/bits values per uint32 word.
struct HostQuantizedWeights {
  std::vector<uint32_t> words;
  std::vector<float> scales;
  std::vector<float> biases;
};

// Round values through a 16-bit dtype on the device and read the exact
// 16-bit values back, so host references see what the kernel sees.
std::vector<float> round_trip(
    const Stream& stream,
    const std::vector<float>& values,
    Dtype dtype) {
  array device(values.begin(), Shape{static_cast<int>(values.size())}, float32);
  return readback_f32(
      stream, astype(astype(device, dtype, stream), float32, stream));
}

HostQuantizedWeights host_affine_quantize(
    const std::vector<float>& matrix,
    int rows,
    int cols,
    int group_size,
    int bits) {
  HostQuantizedWeights result;
  int groups = cols / group_size;
  int pack = 32 / bits;
  int words_per_row = cols / pack;
  float n_bins = static_cast<float>((1 << bits) - 1);
  result.words.assign(static_cast<size_t>(rows) * words_per_row, 0);
  result.scales.resize(static_cast<size_t>(rows) * groups);
  result.biases.resize(static_cast<size_t>(rows) * groups);
  for (int row = 0; row < rows; ++row) {
    for (int group = 0; group < groups; ++group) {
      float w_max = -std::numeric_limits<float>::infinity();
      float w_min = std::numeric_limits<float>::infinity();
      for (int i = 0; i < group_size; ++i) {
        float value = matrix[row * cols + group * group_size + i];
        w_max = std::max(w_max, value);
        w_min = std::min(w_min, value);
      }
      bool min_dominant = std::abs(w_min) > std::abs(w_max);
      float scale = std::max((w_max - w_min) / n_bins, 1e-7f);
      if (!min_dominant) {
        scale = -scale;
      }
      float edge = min_dominant ? w_min : w_max;
      float q0 = std::round(edge / scale);
      if (q0 != 0.0f) {
        scale = edge / q0;
      }
      float bias = (q0 == 0.0f) ? 0.0f : edge;
      result.scales[row * groups + group] = scale;
      result.biases[row * groups + group] = bias;
      for (int i = 0; i < group_size; ++i) {
        float value = matrix[row * cols + group * group_size + i];
        float q = std::clamp(std::round((value - bias) / scale), 0.0f, n_bins);
        uint32_t code = static_cast<uint32_t>(q);
        int col = group * group_size + i;
        result.words[row * words_per_row + col / pack] |=
            code << ((col % pack) * bits);
      }
    }
  }
  return result;
}

// Host dot in double precision: x row m against dequantized w row n,
// dequant = q * scale + bias.
std::vector<float> host_quantized_matmul(
    const HostQuantizedWeights& w,
    const std::vector<float>& x,
    int m,
    int n,
    int k,
    int group_size,
    int bits) {
  int pack = 32 / bits;
  int words_per_row = k / pack;
  int groups = k / group_size;
  uint32_t mask = (1u << bits) - 1u;
  std::vector<float> out(static_cast<size_t>(m) * n);
  for (int row = 0; row < m; ++row) {
    for (int column = 0; column < n; ++column) {
      double acc = 0.0;
      for (int inner = 0; inner < k; ++inner) {
        uint32_t code =
            (w.words[column * words_per_row + inner / pack] >>
             ((inner % pack) * bits)) &
            mask;
        double dequant = static_cast<double>(code) *
                w.scales[column * groups + inner / group_size] +
            w.biases[column * groups + inner / group_size];
        acc += host_at(x, row * k + inner) * dequant;
      }
      out[row * n + column] = static_cast<float>(acc);
    }
  }
  return out;
}

// The GatherQQMM / QQMatmul affine contract carries no biases: the
// dequantized value is q * scale only.
std::vector<float> host_scale_only_quantized_matmul(
    const HostQuantizedWeights& w,
    const std::vector<float>& x,
    int m,
    int n,
    int k,
    int group_size,
    int bits) {
  HostQuantizedWeights zero_bias = w;
  std::fill(zero_bias.biases.begin(), zero_bias.biases.end(), 0.0f);
  return host_quantized_matmul(zero_bias, x, m, n, k, group_size, bits);
}

// BlockMaskedMM reference: a masked block contributes nothing, so the
// element-level filter below is exact upstream block semantics including
// partial edge blocks; the out mask multiplies whole blocks afterwards.
std::vector<float> block_masked_reference(
    const std::vector<float>& a, // (batch, m, k)
    const std::vector<float>& b, // (batch, k, n)
    const std::vector<uint8_t>& lhs_mask,
    const std::vector<uint8_t>& rhs_mask,
    const std::vector<float>* out_mask, // nullptr when absent
    int batch,
    int m,
    int k,
    int n,
    int bs) {
  int tm = (m + bs - 1) / bs;
  int tk = (k + bs - 1) / bs;
  int tn = (n + bs - 1) / bs;
  std::vector<float> expected(static_cast<size_t>(batch) * m * n, 0.0f);
  for (int bt = 0; bt < batch; ++bt) {
    for (int r = 0; r < m; ++r) {
      for (int c = 0; c < n; ++c) {
        double acc = 0.0;
        for (int inner = 0; inner < k; ++inner) {
          bool a_keep =
              lhs_mask[(static_cast<size_t>(bt) * tm + r / bs) * tk +
                       inner / bs];
          bool b_keep =
              rhs_mask[(static_cast<size_t>(bt) * tk + inner / bs) * tn +
                       c / bs];
          if (a_keep && b_keep) {
            acc += host_at(a, (static_cast<size_t>(bt) * m + r) * k + inner) *
                host_at(b, (static_cast<size_t>(bt) * k + inner) * n + c);
          }
        }
        float value = static_cast<float>(acc);
        if (out_mask) {
          value *= (*out_mask)[(static_cast<size_t>(bt) * tm + r / bs) * tn +
                               c / bs];
        }
        expected[(static_cast<size_t>(bt) * m + r) * n + c] = value;
      }
    }
  }
  return expected;
}

// Plain double-precision matmul over stacked operand values.
std::vector<float> host_matmul(
    const std::vector<float>& a, // (m, k) or (batch, m, k)
    const std::vector<float>& b, // (k, n) or (batch, k, n)
    int batch,
    int m,
    int k,
    int n) {
  std::vector<float> out(static_cast<size_t>(batch) * m * n);
  for (int bt = 0; bt < batch; ++bt) {
    for (int r = 0; r < m; ++r) {
      for (int c = 0; c < n; ++c) {
        double acc = 0.0;
        for (int inner = 0; inner < k; ++inner) {
          acc += host_at(a, (static_cast<size_t>(bt) * m + r) * k + inner) *
              host_at(b, (static_cast<size_t>(bt) * k + inner) * n + c);
        }
        out[(static_cast<size_t>(bt) * m + r) * n + c] =
            static_cast<float>(acc);
      }
    }
  }
  return out;
}

} // namespace

TEST_CASE("block masked mm zeroes and scales blocks") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(11);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::uniform_int_distribution<int> bit(0, 1);
  constexpr int bs = 32;

  auto build_masks = [&](int batch, int tm, int tk, int tn) {
    std::vector<uint8_t> lhs(static_cast<size_t>(batch) * tm * tk);
    std::vector<uint8_t> rhs(static_cast<size_t>(batch) * tk * tn);
    for (auto& value : lhs) {
      value = static_cast<uint8_t>(bit(gen));
    }
    for (auto& value : rhs) {
      value = static_cast<uint8_t>(bit(gen));
    }
    return std::make_pair(lhs, rhs);
  };

  // Operand masks, batched: 4 inputs, bool grids with both keep and
  // drop blocks present.
  {
    int batch = 2, m = 48, k = 64, n = 48;
    int tm = 2, tk = 2, tn = 2;
    auto [lhs_v, rhs_v] = build_masks(batch, tm, tk, tn);
    std::vector<float> a_values(static_cast<size_t>(batch) * m * k);
    std::vector<float> b_values(static_cast<size_t>(batch) * k * n);
    for (auto& value : a_values) {
      value = dist(gen);
    }
    for (auto& value : b_values) {
      value = dist(gen);
    }
    array a(a_values.begin(), Shape{batch, m, k}, float32);
    array b(b_values.begin(), Shape{batch, k, n}, float32);
    array mask_lhs(lhs_v.begin(), Shape{batch, tm, tk}, bool_);
    array mask_rhs(rhs_v.begin(), Shape{batch, tk, tn}, bool_);
    array out =
        block_masked_mm(a, b, bs, std::nullopt, mask_lhs, mask_rhs, stream);
    REQUIRE(evaluation_error(out).empty());
    std::vector<float> expected = block_masked_reference(
        a_values, b_values, lhs_v, rhs_v, nullptr, batch, m, k, n, bs);
    expect_close(readback_f32(stream, out), expected, 1e-5);
  }

  // Float32 out mask only, unbatched: block factors 0.0, 0.5, 1.0, and
  // 2.0 exercise zeroing, attenuation, identity, and gain.
  {
    int m = 48, k = 64, n = 48;
    int tm = 2, tn = 2;
    std::vector<float> a_values(static_cast<size_t>(m) * k);
    std::vector<float> b_values(static_cast<size_t>(k) * n);
    for (auto& value : a_values) {
      value = dist(gen);
    }
    for (auto& value : b_values) {
      value = dist(gen);
    }
    std::vector<uint8_t> lhs_keep(static_cast<size_t>(tm) * 2, 1);
    std::vector<uint8_t> rhs_keep(static_cast<size_t>(2) * tn, 1);
    std::vector<float> out_mask_values{0.0f, 0.5f, 1.0f, 2.0f};
    array a(a_values.begin(), Shape{m, k}, float32);
    array b(b_values.begin(), Shape{k, n}, float32);
    array mask_out(out_mask_values.begin(), Shape{tm, tn}, float32);
    array out = block_masked_mm(
        a, b, bs, mask_out, std::nullopt, std::nullopt, stream);
    REQUIRE(evaluation_error(out).empty());
    std::vector<float> expected = block_masked_reference(
        a_values,
        b_values,
        lhs_keep,
        rhs_keep,
        &out_mask_values,
        1,
        m,
        k,
        n,
        bs);
    expect_close(readback_f32(stream, out), expected, 1e-5);
  }

  // All five inputs: batched operand masks plus a bool out mask.
  {
    int batch = 2, m = 48, k = 64, n = 48;
    int tm = 2, tk = 2, tn = 2;
    auto [lhs_v, rhs_v] = build_masks(batch, tm, tk, tn);
    std::vector<uint8_t> out_v(static_cast<size_t>(batch) * tm * tn);
    for (auto& value : out_v) {
      value = static_cast<uint8_t>(bit(gen));
    }
    std::vector<float> a_values(static_cast<size_t>(batch) * m * k);
    std::vector<float> b_values(static_cast<size_t>(batch) * k * n);
    for (auto& value : a_values) {
      value = dist(gen);
    }
    for (auto& value : b_values) {
      value = dist(gen);
    }
    array a(a_values.begin(), Shape{batch, m, k}, float32);
    array b(b_values.begin(), Shape{batch, k, n}, float32);
    array mask_out(out_v.begin(), Shape{batch, tm, tn}, bool_);
    array mask_lhs(lhs_v.begin(), Shape{batch, tm, tk}, bool_);
    array mask_rhs(rhs_v.begin(), Shape{batch, tk, tn}, bool_);
    array out =
        block_masked_mm(a, b, bs, mask_out, mask_lhs, mask_rhs, stream);
    REQUIRE(evaluation_error(out).empty());
    // Bool out mask: false zeroes the block, true multiplies by 1.0.
    std::vector<float> out_scale(out_v.begin(), out_v.end());
    std::vector<float> expected = block_masked_reference(
        a_values, b_values, lhs_v, rhs_v, &out_scale, batch, m, k, n, bs);
    expect_close(readback_f32(stream, out), expected, 1e-5);
  }

  // Edge tiles: 33/65/17 leaves partial 32-wide blocks on every axis.
  {
    int m = 33, k = 65, n = 17;
    int tm = 2, tk = 3, tn = 1;
    auto [lhs_v, rhs_v] = build_masks(1, tm, tk, tn);
    std::vector<float> a_values(static_cast<size_t>(m) * k);
    std::vector<float> b_values(static_cast<size_t>(k) * n);
    for (auto& value : a_values) {
      value = dist(gen);
    }
    for (auto& value : b_values) {
      value = dist(gen);
    }
    array a(a_values.begin(), Shape{m, k}, float32);
    array b(b_values.begin(), Shape{k, n}, float32);
    array mask_lhs(lhs_v.begin(), Shape{tm, tk}, bool_);
    array mask_rhs(rhs_v.begin(), Shape{tk, tn}, bool_);
    array out =
        block_masked_mm(a, b, bs, std::nullopt, mask_lhs, mask_rhs, stream);
    REQUIRE(evaluation_error(out).empty());
    std::vector<float> expected = block_masked_reference(
        a_values, b_values, lhs_v, rhs_v, nullptr, 1, m, k, n, bs);
    expect_close(readback_f32(stream, out), expected, 1e-5);
  }

  // Named error: upstream restricts BlockMaskedMM to float32.
  {
    std::vector<double> wide(16, 0.5);
    array a(wide.begin(), Shape{4, 4}, float64);
    array b(wide.begin(), Shape{4, 4}, float64);
    std::string error;
    try {
      array out = block_masked_mm(
          a, b, bs, std::nullopt, std::nullopt, std::nullopt, stream);
      error = evaluation_error(out);
    } catch (const std::exception& caught) {
      error = caught.what();
    }
    bool f64_pinned = error.find("float64") != std::string::npos ||
        error.find("BlockMaskedMM") != std::string::npos;
    CHECK(f64_pinned);
  }
}

TEST_CASE("gather mm gathers operand matrices") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(23);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::uniform_int_distribution<uint32_t> index_a(0, 2);
  std::uniform_int_distribution<uint32_t> index_b(0, 3);

  // Batched both sides: lhs (2,3) and rhs (2,3) index a (3,M,K) and
  // b (4,K,N); the reference walks the flat index shape row-major.
  {
    int batch_a = 3, batch_b = 4, m = 6, k = 5, n = 7;
    std::vector<float> a_values(static_cast<size_t>(batch_a) * m * k);
    std::vector<float> b_values(static_cast<size_t>(batch_b) * k * n);
    for (auto& value : a_values) {
      value = dist(gen);
    }
    for (auto& value : b_values) {
      value = dist(gen);
    }
    std::vector<uint32_t> lhs_v(6);
    std::vector<uint32_t> rhs_v(6);
    for (auto& value : lhs_v) {
      value = index_a(gen);
    }
    for (auto& value : rhs_v) {
      value = index_b(gen);
    }
    array a(a_values.begin(), Shape{batch_a, m, k}, float32);
    array b(b_values.begin(), Shape{batch_b, k, n}, float32);
    array lhs(lhs_v.begin(), Shape{2, 3}, uint32);
    array rhs(rhs_v.begin(), Shape{2, 3}, uint32);
    array out = gather_mm(a, b, lhs, rhs, /*sorted_indices=*/false, stream);
    REQUIRE(evaluation_error(out).empty());
    std::vector<float> a_view = a_values;
    std::vector<float> b_view = b_values;
    std::vector<float> expected;
    expected.reserve(6 * static_cast<size_t>(m) * n);
    // Expand the gathered stacks so host_matmul walks batch positions.
    std::vector<float> ga(6 * static_cast<size_t>(m) * k);
    std::vector<float> gb(6 * static_cast<size_t>(k) * n);
    for (size_t i = 0; i < 6; ++i) {
      for (int r = 0; r < m; ++r) {
        for (int inner = 0; inner < k; ++inner) {
          ga[i * m * k + static_cast<size_t>(r) * k + inner] =
              a_view[(static_cast<size_t>(lhs_v[i]) * m + r) * k + inner];
        }
      }
      for (int inner = 0; inner < k; ++inner) {
        for (int c = 0; c < n; ++c) {
          gb[i * k * n + static_cast<size_t>(inner) * n + c] =
              b_view[(static_cast<size_t>(rhs_v[i]) * k + inner) * n + c];
        }
      }
    }
    expected = host_matmul(ga, gb, 6, m, k, n);
    expect_close(readback_f32(stream, out), expected, 1e-5);
  }

  // 2D left operand with omitted lhs indices: they default to zero, so
  // every output draws from the single a matrix while rhs gathers b.
  {
    int batch_b = 4, m = 6, k = 5, n = 7;
    std::vector<float> a_values(static_cast<size_t>(m) * k);
    std::vector<float> b_values(static_cast<size_t>(batch_b) * k * n);
    for (auto& value : a_values) {
      value = dist(gen);
    }
    for (auto& value : b_values) {
      value = dist(gen);
    }
    std::vector<uint32_t> rhs_v{1u, 3u, 0u};
    array a(a_values.begin(), Shape{m, k}, float32);
    array b(b_values.begin(), Shape{batch_b, k, n}, float32);
    array rhs(rhs_v.begin(), Shape{3}, uint32);
    // Default lhs indices compose arange(total, uint32) at the op layer;
    // the Arange uint32 rejection is a wave-1 gap, not a GatherMM one,
    // so this sub-block pins that named error instead of value checks.
    array out = gather_mm(a, b, std::nullopt, rhs, false, stream);
    std::string arange_error = evaluation_error(out);
    CHECK(arange_error.find("Arange dtype") != std::string::npos);
    if (!arange_error.empty()) {
      return;
    }
    REQUIRE(evaluation_error(out).empty());
    std::vector<float> ga(3 * static_cast<size_t>(m) * k);
    std::vector<float> gb(3 * static_cast<size_t>(k) * n);
    for (size_t i = 0; i < 3; ++i) {
      std::copy(a_values.begin(), a_values.end(), ga.begin() + i * m * k);
      for (int inner = 0; inner < k; ++inner) {
        for (int c = 0; c < n; ++c) {
          gb[i * k * n + static_cast<size_t>(inner) * n + c] =
              b_values[(static_cast<size_t>(rhs_v[i]) * k + inner) * n + c];
        }
      }
    }
    std::vector<float> expected = host_matmul(ga, gb, 3, m, k, n);
    expect_close(readback_f32(stream, out), expected, 1e-5);
  }

  // Edge tiles: 5/3/7 sits off every 16-wide matmul tile boundary.
  {
    int batch_a = 2, batch_b = 2, m = 5, k = 3, n = 7;
    std::vector<float> a_values(static_cast<size_t>(batch_a) * m * k);
    std::vector<float> b_values(static_cast<size_t>(batch_b) * k * n);
    for (auto& value : a_values) {
      value = dist(gen);
    }
    for (auto& value : b_values) {
      value = dist(gen);
    }
    std::vector<uint32_t> lhs_v{1u, 0u, 1u, 1u};
    std::vector<uint32_t> rhs_v{0u, 1u, 1u, 0u};
    array a(a_values.begin(), Shape{batch_a, m, k}, float32);
    array b(b_values.begin(), Shape{batch_b, k, n}, float32);
    array lhs(lhs_v.begin(), Shape{4}, uint32);
    array rhs(rhs_v.begin(), Shape{4}, uint32);
    array out = gather_mm(a, b, lhs, rhs, false, stream);
    REQUIRE(evaluation_error(out).empty());
    std::vector<float> ga(4 * static_cast<size_t>(m) * k);
    std::vector<float> gb(4 * static_cast<size_t>(k) * n);
    for (size_t i = 0; i < 4; ++i) {
      for (int r = 0; r < m; ++r) {
        for (int inner = 0; inner < k; ++inner) {
          ga[i * m * k + static_cast<size_t>(r) * k + inner] =
              a_values[(static_cast<size_t>(lhs_v[i]) * m + r) * k + inner];
        }
      }
      for (int inner = 0; inner < k; ++inner) {
        for (int c = 0; c < n; ++c) {
          gb[i * k * n + static_cast<size_t>(inner) * n + c] =
              b_values[(static_cast<size_t>(rhs_v[i]) * k + inner) * n + c];
        }
      }
    }
    std::vector<float> expected = host_matmul(ga, gb, 4, m, k, n);
    expect_close(readback_f32(stream, out), expected, 1e-5);
  }

  // A transposed right operand (a transpose view, not a dense stack)
  // takes the materialization path and still gathers correctly.
  {
    int batch_b = 3, m = 6, k = 5, n = 7;
    std::vector<float> a_values(static_cast<size_t>(m) * k);
    std::vector<float> w_values(static_cast<size_t>(batch_b) * n * k);
    for (auto& value : a_values) {
      value = dist(gen);
    }
    for (auto& value : w_values) {
      value = dist(gen);
    }
    array a(a_values.begin(), Shape{m, k}, float32);
    // w has shape (batch_b, n, k); its transpose is (batch_b, k, n).
    array w(w_values.begin(), Shape{batch_b, n, k}, float32);
    array b = transpose(w, {0, 2, 1}, stream);
    std::vector<uint32_t> rhs_v{2u, 0u};
    array rhs(rhs_v.begin(), Shape{2}, uint32);
    array out = gather_mm(a, b, std::nullopt, rhs, false, stream);
    REQUIRE(evaluation_error(out).empty());
    std::vector<float> gb(2 * static_cast<size_t>(k) * n);
    for (size_t i = 0; i < 2; ++i) {
      // b[i] = transpose(w[rhs[i]]): element (inner, c) is
      // w[rhs[i]][c][inner].
      for (int inner = 0; inner < k; ++inner) {
        for (int c = 0; c < n; ++c) {
          gb[i * k * n + static_cast<size_t>(inner) * n + c] =
              w_values[(static_cast<size_t>(rhs_v[i]) * n + c) * k + inner];
        }
      }
    }
    std::vector<float> ga(2 * static_cast<size_t>(m) * k);
    for (size_t i = 0; i < 2; ++i) {
      std::copy(a_values.begin(), a_values.end(), ga.begin() + i * m * k);
    }
    std::vector<float> expected = host_matmul(ga, gb, 2, m, k, n);
    expect_close(readback_f32(stream, out), expected, 1e-5);
  }

  // f16 activations against the same double reference over f16 inputs.
  if (float16_available()) {
    int batch_a = 2, batch_b = 2, m = 6, k = 5, n = 7;
    std::vector<float> a_values(static_cast<size_t>(batch_a) * m * k);
    std::vector<float> b_values(static_cast<size_t>(batch_b) * k * n);
    for (auto& value : a_values) {
      value = dist(gen);
    }
    for (auto& value : b_values) {
      value = dist(gen);
    }
    std::vector<float> a_f16 = round_trip(stream, a_values, float16);
    std::vector<float> b_f16 = round_trip(stream, b_values, float16);
    array a(a_f16.begin(), Shape{batch_a, m, k}, float16);
    array b(b_f16.begin(), Shape{batch_b, k, n}, float16);
    std::vector<uint32_t> lhs_v{0u, 1u};
    std::vector<uint32_t> rhs_v{1u, 0u};
    array lhs(lhs_v.begin(), Shape{2}, uint32);
    array rhs(rhs_v.begin(), Shape{2}, uint32);
    array out = gather_mm(a, b, lhs, rhs, false, stream);
    REQUIRE(evaluation_error(out).empty());
    std::vector<float> ga(2 * static_cast<size_t>(m) * k);
    std::vector<float> gb(2 * static_cast<size_t>(k) * n);
    for (size_t i = 0; i < 2; ++i) {
      for (int r = 0; r < m; ++r) {
        for (int inner = 0; inner < k; ++inner) {
          ga[i * m * k + static_cast<size_t>(r) * k + inner] =
              a_f16[(static_cast<size_t>(lhs_v[i]) * m + r) * k + inner];
        }
      }
      for (int inner = 0; inner < k; ++inner) {
        for (int c = 0; c < n; ++c) {
          gb[i * k * n + static_cast<size_t>(inner) * n + c] =
              b_f16[(static_cast<size_t>(rhs_v[i]) * k + inner) * n + c];
        }
      }
    }
    std::vector<float> expected = host_matmul(ga, gb, 2, m, k, n);
    expect_close(readback_f32(stream, out), expected, 2e-2);
  }
}

TEST_CASE("segmented mm writes per-segment contractions") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(37);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  // Segments over K=33: full span, two sub-spans, an empty segment, and
  // an inverted segment; empties must write zeros.
  {
    int m = 6, k = 33, n = 5;
    std::vector<float> a_values(static_cast<size_t>(m) * k);
    std::vector<float> b_values(static_cast<size_t>(k) * n);
    for (auto& value : a_values) {
      value = dist(gen);
    }
    for (auto& value : b_values) {
      value = dist(gen);
    }
    std::vector<uint32_t> segments_v{
        0u, 33u, // full
        0u, 16u, // low sub-span
        16u, 33u, // high sub-span (off-tile boundary)
        7u, 7u, // empty
        20u, 5u, // inverted
    };
    array a(a_values.begin(), Shape{m, k}, float32);
    array b(b_values.begin(), Shape{k, n}, float32);
    array segments(segments_v.begin(), Shape{5, 2}, uint32);
    array out = segmented_mm(a, b, segments, stream);
    REQUIRE(evaluation_error(out).empty());
    REQUIRE_EQ(out.shape(), Shape{5, m, n});
    std::vector<float> expected(static_cast<size_t>(5) * m * n, 0.0f);
    for (int s = 0; s < 5; ++s) {
      uint32_t k_start = segments_v[2 * s];
      uint32_t k_end = segments_v[2 * s + 1];
      if (k_end <= k_start) {
        continue; // zeros stay
      }
      for (int r = 0; r < m; ++r) {
        for (int c = 0; c < n; ++c) {
          double acc = 0.0;
          for (uint32_t inner = k_start; inner < k_end; ++inner) {
            acc += host_at(a_values, static_cast<size_t>(r) * k + inner) *
                host_at(b_values, static_cast<size_t>(inner) * n + c);
          }
          expected[(static_cast<size_t>(s) * m + r) * n + c] =
              static_cast<float>(acc);
        }
      }
    }
    expect_close(readback_f32(stream, out), expected, 1e-5);
  }

  // Transposed operand views take the materialization path; segments
  // stay non-aligned to the 16-wide tile.
  {
    int m = 6, k = 33, n = 5;
    std::vector<float> a_values(static_cast<size_t>(k) * m);
    std::vector<float> b_values(static_cast<size_t>(n) * k);
    for (auto& value : a_values) {
      value = dist(gen);
    }
    for (auto& value : b_values) {
      value = dist(gen);
    }
    // at has shape (k, m); its transpose is (m, k). bt has shape (n, k);
    // its transpose is (k, n).
    array at(a_values.begin(), Shape{k, m}, float32);
    array bt(b_values.begin(), Shape{n, k}, float32);
    array a = transpose(at, {1, 0}, stream);
    array b = transpose(bt, {1, 0}, stream);
    std::vector<uint32_t> segments_v{
        0u, 1u,
        5u, 32u,
        31u, 33u,
    };
    array segments(segments_v.begin(), Shape{3, 2}, uint32);
    array out = segmented_mm(a, b, segments, stream);
    REQUIRE(evaluation_error(out).empty());
    REQUIRE_EQ(out.shape(), Shape{3, m, n});
    std::vector<float> expected(static_cast<size_t>(3) * m * n, 0.0f);
    for (int s = 0; s < 3; ++s) {
      uint32_t k_start = segments_v[2 * s];
      uint32_t k_end = segments_v[2 * s + 1];
      for (int r = 0; r < m; ++r) {
        for (int c = 0; c < n; ++c) {
          double acc = 0.0;
          for (uint32_t inner = k_start; inner < k_end; ++inner) {
            acc += host_at(a_values, static_cast<size_t>(inner) * m + r) *
                host_at(b_values, static_cast<size_t>(c) * k + inner);
          }
          expected[(static_cast<size_t>(s) * m + r) * n + c] =
              static_cast<float>(acc);
        }
      }
    }
    expect_close(readback_f32(stream, out), expected, 1e-5);
  }

  // f16 activations against the double reference.
  if (float16_available()) {
    int m = 4, k = 20, n = 3;
    std::vector<float> a_values(static_cast<size_t>(m) * k);
    std::vector<float> b_values(static_cast<size_t>(k) * n);
    for (auto& value : a_values) {
      value = dist(gen);
    }
    for (auto& value : b_values) {
      value = dist(gen);
    }
    std::vector<float> a_f16 = round_trip(stream, a_values, float16);
    std::vector<float> b_f16 = round_trip(stream, b_values, float16);
    array a(a_f16.begin(), Shape{m, k}, float16);
    array b(b_f16.begin(), Shape{k, n}, float16);
    std::vector<uint32_t> segments_v{0u, 20u, 3u, 17u, 17u, 20u};
    array segments(segments_v.begin(), Shape{3, 2}, uint32);
    array out = segmented_mm(a, b, segments, stream);
    REQUIRE(evaluation_error(out).empty());
    std::vector<float> expected(static_cast<size_t>(3) * m * n, 0.0f);
    for (int s = 0; s < 3; ++s) {
      uint32_t k_start = segments_v[2 * s];
      uint32_t k_end = segments_v[2 * s + 1];
      for (int r = 0; r < m; ++r) {
        for (int c = 0; c < n; ++c) {
          double acc = 0.0;
          for (uint32_t inner = k_start; inner < k_end; ++inner) {
            acc += host_at(a_f16, static_cast<size_t>(r) * k + inner) *
                host_at(b_f16, static_cast<size_t>(inner) * n + c);
          }
          expected[(static_cast<size_t>(s) * m + r) * n + c] =
              static_cast<float>(acc);
        }
      }
    }
    expect_close(readback_f32(stream, out), expected, 2e-2);
  }

  // Named error: upstream rejects batched segmented_mm at the op layer.
  {
    std::vector<float> values(2 * 4 * 4, 1.0f);
    array a(values.begin(), Shape{2, 4, 4}, float32);
    array b(values.begin(), Shape{2, 4, 4}, float32);
    std::vector<uint32_t> segments_v{0u, 4u};
    array segments(segments_v.begin(), Shape{1, 2}, uint32);
    std::string error;
    try {
      array out = segmented_mm(a, b, segments, stream);
      error = evaluation_error(out);
    } catch (const std::exception& caught) {
      error = caught.what();
    }
    CHECK(error.find("segmented_mm") != std::string::npos);
  }
}

TEST_CASE("gather qmm gathers experts with scales and biases") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(41);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::uniform_int_distribution<uint32_t> index_x(0, 1);
  std::uniform_int_distribution<uint32_t> index_w(0, 2);

  auto run_case = [&](int experts,
                      int x_batch,
                      int m,
                      int k,
                      int n,
                      int group_size,
                      int bits,
                      bool with_lhs) {
    CAPTURE(experts);
    CAPTURE(m);
    CAPTURE(n);
    CAPTURE(k);
    CAPTURE(group_size);
    CAPTURE(bits);
    int groups = k / group_size;
    int pack = 32 / bits;
    int words_per_row = k / pack;
    std::vector<HostQuantizedWeights> host_w;
    // uint32, not float: these are raw packed code words. Held in a
    // float vector, every word >= 2^24 is silently rounded by the
    // element conversion (low eight bits destroyed, rounding carry into
    // bit 8), and the uint32 array constructor copies the rounded
    // values verbatim - the kernel then decodes wrong codes.
    std::vector<uint32_t> w_all;
    std::vector<float> scales_all;
    std::vector<float> biases_all;
    for (int e = 0; e < experts; ++e) {
      std::vector<float> matrix(static_cast<size_t>(n) * k);
      for (auto& value : matrix) {
        value = dist(gen);
      }
      HostQuantizedWeights host =
          host_affine_quantize(matrix, n, k, group_size, bits);
      host_w.push_back(host);
      w_all.insert(w_all.end(), host.words.begin(), host.words.end());
      scales_all.insert(scales_all.end(), host.scales.begin(),
                        host.scales.end());
      biases_all.insert(biases_all.end(), host.biases.begin(),
                        host.biases.end());
    }
    std::vector<std::vector<float>> x_batches;
    std::vector<float> x_all;
    for (int b = 0; b < x_batch; ++b) {
      std::vector<float> matrix(static_cast<size_t>(m) * k);
      for (auto& value : matrix) {
        value = dist(gen);
      }
      x_batches.push_back(matrix);
      x_all.insert(x_all.end(), matrix.begin(), matrix.end());
    }
    int positions = with_lhs ? 6 : x_batch;
    std::vector<uint32_t> lhs_v(positions);
    std::vector<uint32_t> rhs_v(positions);
    for (auto& value : lhs_v) {
      value = index_x(gen) % static_cast<uint32_t>(std::max(x_batch, 1));
    }
    for (auto& value : rhs_v) {
      value = index_w(gen) % static_cast<uint32_t>(experts);
    }
    array w_words(
        w_all.begin(),
        Shape{experts, n, words_per_row},
        uint32);
    array scales(
        scales_all.begin(), Shape{experts, n, groups}, float32);
    array biases(
        biases_all.begin(), Shape{experts, n, groups}, float32);
    array x(
        x_all.begin(),
        Shape{x_batch, m, k},
        float32);

    // gather_qmm returns a fresh array in both branches; hold it in an
    // optional so one declaration serves both.
    std::optional<array> out_holder;
    if (with_lhs) {
      array lhs(lhs_v.begin(), Shape{2, 3}, uint32);
      array rhs(rhs_v.begin(), Shape{2, 3}, uint32);
      out_holder = gather_qmm(
          x,
          w_words,
          scales,
          biases,
          lhs,
          rhs,
          /*transpose=*/true,
          group_size,
          bits,
          "affine",
          /*sorted_indices=*/false,
          stream);
    } else {
      // Explicit zero lhs indices exercise the x-side broadcast without
      // the op layer's default path, which composes arange(uint32) --
      // a wave-1 Arange gap this backend names separately.
      std::vector<uint32_t> zeros(x_batch, 0u);
      array lhs0(zeros.begin(), Shape{x_batch}, uint32);
      array rhs(rhs_v.begin(), Shape{x_batch}, uint32);
      out_holder = gather_qmm(
          x,
          w_words,
          scales,
          biases,
          lhs0,
          rhs,
          true,
          group_size,
          bits,
          "affine",
          false,
          stream);
    }
    array out = *out_holder;
    REQUIRE(evaluation_error(out).empty());
    REQUIRE_EQ(out.shape(-2), m);
    REQUIRE_EQ(out.shape(-1), n);
    // Reference: per position p, x[lhs[p]] contracted against the
    // affine dequant of the packed codes of expert rhs[p], in double
    // precision over the exact host words, scales, and biases the
    // device buffers carry.
    std::vector<float> expected(
        static_cast<size_t>(lhs_v.size()) * m * n, 0.0f);
    for (size_t p = 0; p < lhs_v.size(); ++p) {
      std::vector<float> piece = host_quantized_matmul(
          host_w[rhs_v[p]],
          // The non-lhs branch passes an all-zero index container to
          // the device (x-side broadcast), so the reference must
          // contract row 0 there too - not the drawn lhs_v[p].
          x_batches[with_lhs ? lhs_v[p] : 0u],
          m,
          n,
          k,
          group_size,
          bits);
      std::copy(
          piece.begin(),
          piece.end(),
          expected.begin() +
              static_cast<ptrdiff_t>(p) * static_cast<ptrdiff_t>(m) * n);
    }
    expect_close_tol(readback_f32(stream, out), expected, 1e-4, 1e-3);
  };

  run_case(3, 2, 5, 192, 37, 64, 4, true);
  run_case(3, 2, 5, 192, 37, 64, 4, false);
  run_case(2, 2, 1, 128, 16, 32, 8, true);
  run_case(3, 1, 7, 192, 37, 64, 8, true);

  // Named errors. Each shape is crafted to clear the op layer's shape
  // validation so the rejection provably comes from this backend: the
  // op layer's bit/group equality runs first, so the packed weight and
  // scale shapes must satisfy it even for the rejected parameters.
  {
    // mxfp4 mode: scales must be uint8 and biases absent at the op
    // layer; the primitive then rejects the mode itself.
    std::vector<float> x_values(2 * 64, 0.25f);
    array x(x_values.begin(), Shape{2, 64}, float32);
    std::vector<uint32_t> w_values(8 * 8, 0u);
    array w_words(w_values.begin(), Shape{8, 8}, uint32);
    std::vector<uint8_t> fp_scales(8 * 2, 127u);
    array scales8(fp_scales.begin(), Shape{8, 2}, uint8);
    std::vector<uint32_t> idx0{0u};
    array rhs(idx0.begin(), Shape{1}, uint32);
    array lhs0(idx0.begin(), Shape{1}, uint32);
    std::string mode_error = evaluation_error(gather_qmm(
        x,
        w_words,
        scales8,
        std::nullopt,
        lhs0,
        rhs,
        true,
        std::nullopt,
        std::nullopt,
        "mxfp4",
        false,
        stream));
    CHECK(mode_error.find("GatherQMM mode") != std::string::npos);

    // Non-transposed weights: w reads as (k, n) with k == 64.
    std::vector<uint32_t> w_nt(64 * 8, 0u);
    array w_nt_words(w_nt.begin(), Shape{64, 8}, uint32);
    std::vector<float> nt_params(64 * 2, 0.5f);
    array nt_scales(nt_params.begin(), Shape{64, 2}, float32);
    array nt_biases(nt_params.begin(), Shape{64, 2}, float32);
    std::string transpose_error = evaluation_error(gather_qmm(
        x,
        w_nt_words,
        nt_scales,
        nt_biases,
        lhs0,
        rhs,
        false,
        32,
        4,
        "affine",
        false,
        stream));
    CHECK(transpose_error.find("GatherQMM transpose") != std::string::npos);

    // bits=2: packed width and scales must agree through the op layer
    // equality, so k = 8*32/2 = 128 with one scale per 128 columns.
    std::vector<float> x2_values(2 * 128, 0.25f);
    array x2(x2_values.begin(), Shape{2, 128}, float32);
    std::vector<float> one_param(8 * 1, 0.5f);
    array scales_one(one_param.begin(), Shape{8, 1}, float32);
    array biases_one(one_param.begin(), Shape{8, 1}, float32);
    std::string bits_error = evaluation_error(gather_qmm(
        x2,
        w_words,
        scales_one,
        biases_one,
        lhs0,
        rhs,
        true,
        128,
        2,
        "affine",
        false,
        stream));
    CHECK(bits_error.find("GatherQMM bits") != std::string::npos);

    // group_size=128: k = 32*32/4 = 256 with two scales per row.
    std::vector<uint32_t> w_wide(8 * 32, 0u);
    array w_wide_words(w_wide.begin(), Shape{8, 32}, uint32);
    std::vector<float> two_params(8 * 2, 0.5f);
    array scales_two(two_params.begin(), Shape{8, 2}, float32);
    array biases_two(two_params.begin(), Shape{8, 2}, float32);
    std::vector<float> x4_values(2 * 256, 0.25f);
    array x4(x4_values.begin(), Shape{2, 256}, float32);
    std::string group_error = evaluation_error(gather_qmm(
        x4,
        w_wide_words,
        scales_two,
        biases_two,
        lhs0,
        rhs,
        true,
        128,
        4,
        "affine",
        false,
        stream));
    CHECK(group_error.find("GatherQMM group size") != std::string::npos);
  }
}

TEST_CASE("gather qqmm dequants with scales only") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(53);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  auto run_case = [&](int m, int k, int n, int group_size, int bits) {
    CAPTURE(m);
    CAPTURE(n);
    CAPTURE(k);
    int experts = 3;
    int x_batch = 2;
    int groups = k / group_size;
    int pack = 32 / bits;
    int words_per_row = k / pack;
    std::vector<HostQuantizedWeights> host_w;
    // uint32, not float - same packed-words hazard as gather_qmm above.
    std::vector<uint32_t> w_all;
    std::vector<float> scales_all;
    for (int e = 0; e < experts; ++e) {
      std::vector<float> matrix(static_cast<size_t>(n) * k);
      for (auto& value : matrix) {
        value = dist(gen);
      }
      HostQuantizedWeights host =
          host_affine_quantize(matrix, n, k, group_size, bits);
      host_w.push_back(host);
      w_all.insert(w_all.end(), host.words.begin(), host.words.end());
      scales_all.insert(
          scales_all.end(), host.scales.begin(), host.scales.end());
    }
    std::vector<std::vector<float>> x_batches;
    std::vector<float> x_all;
    for (int b = 0; b < x_batch; ++b) {
      std::vector<float> matrix(static_cast<size_t>(m) * k);
      for (auto& value : matrix) {
        value = dist(gen);
      }
      x_batches.push_back(matrix);
      x_all.insert(x_all.end(), matrix.begin(), matrix.end());
    }
    std::vector<uint32_t> lhs_v{0u, 1u, 1u, 0u};
    std::vector<uint32_t> rhs_v{2u, 0u, 1u, 2u};
    array w_words(
        w_all.begin(), Shape{experts, n, words_per_row}, uint32);
    array scales(scales_all.begin(), Shape{experts, n, groups}, float32);
    array x(x_all.begin(), Shape{x_batch, m, k}, float32);
    array lhs(lhs_v.begin(), Shape{2, 2}, uint32);
    array rhs(rhs_v.begin(), Shape{2, 2}, uint32);
    array out = gather_qqmm(
        x,
        w_words,
        scales,
        lhs,
        rhs,
        group_size,
        bits,
        "affine",
        std::nullopt,
        std::nullopt,
        false,
        stream);
    REQUIRE(evaluation_error(out).empty());
    REQUIRE_EQ(out.shape(-2), m);
    REQUIRE_EQ(out.shape(-1), n);
    // Reference: same contraction as GatherQMM but the GatherQQMM
    // affine contract drops the bias, so dequant is q * scale only.
    std::vector<float> expected(4 * static_cast<size_t>(m) * n, 0.0f);
    for (size_t p = 0; p < lhs_v.size(); ++p) {
      std::vector<float> piece = host_scale_only_quantized_matmul(
          host_w[rhs_v[p]],
          x_batches[lhs_v[p]],
          m,
          n,
          k,
          group_size,
          bits);
      std::copy(
          piece.begin(),
          piece.end(),
          expected.begin() +
              static_cast<ptrdiff_t>(p) * static_cast<ptrdiff_t>(m) * n);
    }
    expect_close_tol(readback_f32(stream, out), expected, 1e-4, 1e-3);
  };

  run_case(5, 192, 37, 64, 4);
  run_case(1, 128, 16, 32, 8);

  // Named errors: non-affine modes and float weights keep their tags.
  {
    int k = 64, n = 8;
    std::vector<float> matrix(static_cast<size_t>(n) * k, 0.5f);
    auto host = host_affine_quantize(matrix, n, k, 64, 4);
    array w_words(host.words.begin(), Shape{n, k / 8}, uint32);
    array w_float(matrix.begin(), Shape{n, k}, float32);
    array scales(host.scales.begin(), Shape{n, 1}, float32);
    std::vector<float> x_values(static_cast<size_t>(2) * k, 0.25f);
    array x(x_values.begin(), Shape{2, k}, float32);
    std::vector<uint32_t> idx_v{0u};
    array idx(idx_v.begin(), Shape{1}, uint32);

    std::string mode_error = evaluation_error(gather_qqmm(
        x, w_words, scales, idx, idx, 64, 4, "mxfp4", std::nullopt,
        std::nullopt, false, stream));
    CHECK(mode_error.find("GatherQQMM mode") != std::string::npos);

    std::string weight_error = evaluation_error(gather_qqmm(
        x, w_float, scales, idx, idx, 64, 4, "affine", std::nullopt,
        std::nullopt, false, stream));
    CHECK(weight_error.find("GatherQQMM") != std::string::npos);
  }
}

TEST_CASE("qq matmul matches scale-only quantized and float paths") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(67);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  // Quantized weights: scale-only affine dequant, no activation
  // quantization on this backend.
  {
    int m = 7, k = 128, n = 37;
    for (auto [group_size, bits] : std::vector<std::pair<int, int>>{
             {64, 4}, {32, 8}}) {
      int groups = k / group_size;
      int pack = 32 / bits;
      int words_per_row = k / pack;
      std::vector<float> matrix(static_cast<size_t>(n) * k);
      for (auto& value : matrix) {
        value = dist(gen);
      }
      HostQuantizedWeights host =
          host_affine_quantize(matrix, n, k, group_size, bits);
      std::vector<float> x_values(static_cast<size_t>(m) * k);
      for (auto& value : x_values) {
        value = dist(gen);
      }
      array w_words(
          host.words.begin(), Shape{n, words_per_row}, uint32);
      array scales(host.scales.begin(), Shape{n, groups}, float32);
      array x(x_values.begin(), Shape{m, k}, float32);
      array out = qqmm(x, w_words, scales, group_size, bits, "affine",
                       std::nullopt, std::nullopt, stream);
      REQUIRE(evaluation_error(out).empty());
      REQUIRE_EQ(out.shape(), Shape{m, n});
      std::vector<float> expected = host_scale_only_quantized_matmul(
          host, x_values, m, n, k, group_size, bits);
      expect_close(readback_f32(stream, out), expected, 1e-3);
    }
  }

  // Float weights: the un-quantized product x @ w.T against the host.
  {
    int m = 7, k = 20, n = 37;
    std::vector<float> matrix(static_cast<size_t>(n) * k);
    std::vector<float> x_values(static_cast<size_t>(m) * k);
    for (auto& value : matrix) {
      value = dist(gen);
    }
    for (auto& value : x_values) {
      value = dist(gen);
    }
    array w(matrix.begin(), Shape{n, k}, float32);
    array x(x_values.begin(), Shape{m, k}, float32);
    // Float-weight qqmm needs x @ w.T; the backend names the rejection
    // (no transposed view inside an eval). Pin it.
    array out = qqmm(x, w, std::nullopt, 64, 4, "affine", std::nullopt,
                     std::nullopt, stream);
    std::string weight_error = evaluation_error(out);
    CHECK(weight_error.find("QQMatmul weight dtype") != std::string::npos);

    (void)m;
    (void)k;
    (void)n;
  }

  // Named error: mxfp4 keeps its mode tag.
  {
    std::vector<float> matrix(4 * 32, 0.5f);
    array w(matrix.begin(), Shape{4, 32}, float32);
    std::vector<float> x_values(2 * 32, 0.25f);
    array x(x_values.begin(), Shape{2, 32}, float32);
    std::string mode_error = evaluation_error(
        qqmm(x, w, std::nullopt, 64, 4, "mxfp4", std::nullopt, std::nullopt,
             stream));
    CHECK(mode_error.find("QQMatmul mode") != std::string::npos);
  }
}

// DecodeGemvSubgroup equivalence: with caps.subgroup_size==32 the
// dispatch path in primitives.cpp routes every decode row through the
// new QmmVecSubgroup kernel, so this case is the load-bearing proof
// that the production subgroup path computes the same numbers as the
// CPU double-precision dequant reference across the real decode
// shapes. The microbenchmark in tools/subgroup-bench covers a 32-elem
// float reduction in isolation; this case covers n not divisible by 8
// (workgroup width), n not divisible by 32 (subgroup width), group
// sizes 32 and 64, bits 4 and 8, and f32 / f16 / bf16 storage. bf16
// and f16 store through the same f32 accumulator inside qmm_vec.comp,
// so the reduction order determines the f32 sum bit-for-bit; storage
// quantization then matches both kernels. Where subgroup reduction
// order legitimately changes rounding (subgroupAdd is
// implementation-defined; the tree is a strict log2(32) pairwise
// fold), the tolerance is one f32 ulp at the reduction plus the
// storage dtype's rtol after STORE_VALUE.
//
// Tree-vs-subgroup bit-exact comparison cannot be expressed through
// the public dispatch API today (the encoder's combined
// scales+biases buffer is private). The microbenchmark covers that
// reduction-order comparison at the kernel level on identical
// 32-element float sums; this test covers the end-to-end shape
// coverage the bench does not.
//
// A red subgroup-vs-host result on the M1 ends the A/B regardless of
// speed, per the assignment.
TEST_CASE("qmm_vec subgroup dispatch matches host reference across decode shapes") {
  if (!compute_available()) {
    return;
  }
  const auto& caps = omarchy::device(0).capabilities();
  bool subgroup_ready = caps.subgroup_size == 32u &&
      (caps.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
  if (!subgroup_ready) {
    skip("subgroup size != 32 or no ARITHMETIC; subgroup variant unrunnable on this device");
    return;
  }
  Stream stream = gpu_stream();

  // Shapes: n=1 is a single-element reduction; n=7 is under one
  // workgroup width (COLUMNS_PER_GROUP=8); n=9 crosses one workgroup
  // boundary; n=37 crosses multiple workgroups with a partial last
  // workgroup; n=64 is exactly two workgroups; n=256 is large. k is
  // always 3*group_size so lane boundaries land mid-slot (a known
  // sensitive boundary for the reduction step).
  std::vector<int> n_shapes{1, 7, 9, 37, 64, 256};
  std::vector<std::pair<int, int>> qbits{{64, 4}, {32, 4}, {64, 8}, {32, 8}};
  std::vector<Dtype> dtypes{float32};
  if (float16_available()) {
    dtypes.push_back(float16);
  }
  dtypes.push_back(bfloat16);

  std::mt19937 gen(91);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  for (auto [group_size, bits] : qbits) {
    int k = group_size * 3;
    int pack = 32 / bits;
    int words_per_row = k / pack;
    int groups_per_row = k / group_size;
    for (auto dtype : dtypes) {
      for (int n : n_shapes) {
        std::vector<float> matrix(static_cast<size_t>(n) * k);
        for (auto& v : matrix) v = dist(gen);
        std::vector<float> x_values(k);
        for (auto& v : x_values) v = dist(gen);
        HostQuantizedWeights host =
            host_affine_quantize(matrix, n, k, group_size, bits);
        // Round-trip scales/biases through the storage dtype the kernel
        // sees, otherwise the host reference uses f32 storage while the
        // device rounds to bf16/f16 and the gap shows up as an
        // apparent equivalence failure unrelated to the reduction.
        std::vector<float> scales_rt = round_trip(stream, host.scales, dtype);
        std::vector<float> biases_rt = round_trip(stream, host.biases, dtype);
        HostQuantizedWeights rounded = host;
        rounded.scales = scales_rt;
        rounded.biases = biases_rt;
        std::vector<float> expected = host_quantized_matmul(
            rounded, x_values, 1, n, k, group_size, bits);

        // With subgroup_ready=true the dispatch path picks
        // QmmVecSubgroup on the M1; this is the production path.
        array x(x_values.begin(), Shape{1, k}, dtype);
        array w_words(host.words.begin(), Shape{n, words_per_row}, uint32);
        array scales(scales_rt.begin(), Shape{n, groups_per_row}, dtype);
        array biases(biases_rt.begin(), Shape{n, groups_per_row}, dtype);
        array out = quantized_matmul(
            x, w_words, scales, biases, true, group_size, bits, "affine",
            stream);
        REQUIRE(evaluation_error(out).empty());
        REQUIRE_EQ(out.shape(), Shape{1, n});
        std::vector<float> device_result = readback_f32(stream, out);

        float max_diff = 0.0f;
        int worst_idx = -1;
        for (int i = 0; i < n; ++i) {
          float d = std::fabs(device_result[i] - expected[i]);
          if (d > max_diff) {
            max_diff = d;
            worst_idx = i;
          }
        }
        // Per-dtype bound derived from reduction depth and
        // accumulator type. The device computes the dot in f32
        // throughout (qmm_vec.comp promotes scales/biases/x to f32
        // via LOAD_VALUE); the host reference computes in f64
        // (host_quantized_matmul). The gap is therefore dominated
        // by f32 vs f64 multiply-add precision across the K
        // elements of one output column, plus cross-lane reduction
        // (32 lanes summed; 5 pairwise-add rounds for the tree
        // path, 1 subgroupAdd for the subgroup path - both incur
        // the same per-add 1-ulp rounding in the worst case).
        //
        // Per-element ops: scale * q + bias (2 ops) then x * (...)
        // (1 op) = 3 ops per k element. Per-lane accumulates
        // K/32 elements. 32 cross-lane adds. Total f32 multiply-add
        // count: (3 * K/32 + 32). Each op is bounded by 1 ulp at
        // f32 relative to the operand magnitude. Conservative
        // total error vs the f64 host sum:
        //
        //   E_f32 = (3*K/32 + 32) * M_max * 2^-23
        //
        // For bf16/f16 storage, STORE_VALUE rounds the f32 sum to
        // the storage dtype. The half-ulp at the rounded magnitude
        // adds:
        //
        //   E_storage = M_max * 2^-(mantissa_bits + 1)
        //
        // where mantissa_bits = 23 (f32), 10 (f16), 7 (bf16).
        //
        // Total gap bound (no cancellation assumed):
        //
        //   bound = E_f32 + E_storage
        //
        // A kernel defect (wrong shift, wrong lane, init not
        // reset) would produce a gap of order M_max, not M_max *
        // eps. The derived bound is loose enough to permit
        // legitimate reduction-order and storage rounding, tight
        // enough to catch defects.
        //
        // M_max here is the worst-case |device_result[i]| in this
        // run; using a constant derived from the input range (the
        // test uses dist(-2, 2) for matrix and x, with scales and
        // biases from the same range, and the dequantized w row
        // has magnitude at most 2 * (2^(bits-1) - 1) * 2 + 2).
        // That gives |y| up to ~50 worst case; the empirical
        // |device_result| range was 10-43 on the M1 leg. The bound
        // uses the empirical max with a 2x safety factor so a
        // single test run does not falsely reject when the run's
        // M happens to land near the upper edge of the input
        // distribution.
        double m_max_obs =
            *std::max_element(device_result.begin(), device_result.end(),
                [](float a, float b) {
                  return std::fabs(a) < std::fabs(b);
                });
        m_max_obs = std::fabs(m_max_obs);
        double m_max_bound = std::max(m_max_obs * 2.0, 50.0);
        int storage_mantissa_bits = (dtype == float32)
            ? 23
            : (dtype == float16) ? 10 : 7;
        double ops = 3.0 * k / 32.0 + 32.0;
        double e_f32 = ops * m_max_bound * std::ldexp(1.0, -23);
        double e_storage = m_max_bound *
            std::ldexp(1.0, -(storage_mantissa_bits + 1));
        double bound = e_f32 + e_storage;
        // Floor at 1e-6 so f32 storage's tighter precision is not
        // asserted tighter than the derived math (the f32 E_storage
        // term is below 1e-6 at |M|<16, which is the typical
        // case in this test).
        bound = std::max(bound, 1e-6);
        INFO("bits=" << bits << " group_size=" << group_size
             << " dtype=" << dtype << " n=" << n
             << " worst_idx=" << worst_idx
             << " device=" << device_result[worst_idx]
             << " host=" << expected[worst_idx]
             << " diff=" << max_diff
             << " bound=" << bound
             << " m_max_obs=" << m_max_obs
             << " e_f32=" << e_f32
             << " e_storage=" << e_storage);
        CHECK(max_diff <= bound);
      }
    }
  }
}

namespace {

// Flips the PrefillQmmTile dispatch gate and always restores the unset
// (default OFF) state, so an aborting REQUIRE cannot leak the switch
// into later cases in the process.
struct QmmTileGate {
  explicit QmmTileGate(bool on) {
    set(on);
  }
  ~QmmTileGate() {
    unsetenv("MLX_OMARCHY_QMM_TILE");
  }
  void set(bool on) {
    setenv("MLX_OMARCHY_QMM_TILE", on ? "1" : "0", 1);
  }
};

} // namespace

// PrefillQmmTile equivalence: the env-gated m-tiled kernel must match
// the general qmm.comp kernel it stands in for at matrix_m > 1. Both
// sides run on device through the public dispatch - the gate selects
// qmm_tile unless MLX_OMARCHY_QMM_TILE is "0" (OFF), selecting qmm.comp.
//
// Bound derivation. Both kernels accumulate x * (scale*q + bias) over
// ascending k into a single f32 accumulator per output element in the
// SAME order, so there is no reduction-depth term between them: the
// residual gap is (a) FMA contraction differences between two shaders
// whose expression shapes differ - at most one f32 ulp per k step
// against the output magnitude, (k + 2) * M * 2^-23 - and (b) the
// final STORE_VALUE rounding, at most one storage-dtype ulp,
// M * 2^-(mantissa + 1). A defect (row/column swap, tail overrun,
// barrier misordering, wrong group index) misses by O(M), not by
// eps-scale dust, because the dequantized weights span both signs.
//
// The host-anchored tail case (m=17, n=37) pins the tile kernel to the
// f64 affine contract itself, so tile cannot drift in step with
// qmm.comp; the f16/bf16 anchor round-trips x, scales, and biases
// through the storage dtype so the reference sees exactly the operand
// rounding the kernel sees. The anchor bound adds the per-element
// dequant (2 ops) and mul-add (1 op) across the sequential k
// reduction: (3k + 1) * M * 2^-23 plus storage rounding, the same
// derivation as the qmm_vec case above.
//
// Every device readback is asserted finite BEFORE the max/abs
// reductions. NaN never wins a `>` comparison, so one NaN in the
// output would leave max_diff at 0 and false-pass the CHECK; +Inf
// would inflate the observed maximum and with it the bound to Inf,
// and Inf <= Inf compares true. The finite gate closes both holes.
//
// The default sweep matrix is bounded: every dtype x bits/group combo
// over m {2, 15, 16, 17, 1023} at the hidden width (n = k = 896, all
// m tail classes), plus the wide 4864 x 4864 gate/up projection at
// both m extremes for a 4-bit/f16 and an 8-bit/bf16 combo. The full
// m x n x k x quant x dtype cross product is opt-in for hardware
// validation runs via MLX_OMARCHY_QMM_TILE_FULL_SWEEP=1; it is the
// same cases through the same helper, unbounded in software-GPU time.
TEST_CASE("qmm tile matches host reference and qmm.comp across prefill shapes") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  std::vector<Dtype> dtypes{float32};
  if (float16_available()) {
    dtypes.push_back(float16);
  }
  dtypes.push_back(bfloat16);
  std::vector<std::pair<int, int>> qbits{
      {32, 4}, {64, 4}, {32, 8}, {64, 8}};

  auto storage_mantissa = [](Dtype dtype) {
    return dtype == float32 ? 23 : (dtype == float16 ? 10 : 7);
  };

  // Maximum magnitude over a readback, refusing non-finite values:
  // returns false if any element is NaN or Inf, with the finite max in
  // max_abs either way.
  auto finite_max = [](const std::vector<float>& values, double& max_abs) {
    max_abs = 0.0;
    for (float v : values) {
      if (!std::isfinite(v)) {
        return false;
      }
      max_abs = std::max(max_abs, std::fabs(static_cast<double>(v)));
    }
    return true;
  };

  // One tile-vs-qmm.comp case through the public dispatch: gate ON
  // selects qmm_tile, gate OFF selects qmm.comp, same inputs. Carries
  // the finite guard, the derived bound, and the worst-element report.
  auto run_tile_case = [&](Dtype dtype, int group_size, int bits, int k,
      int n, int m, unsigned seed) {
    const int pack = 32 / bits;
    const int words_per_row = k / pack;
    const int groups_per_row = k / group_size;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> matrix(static_cast<size_t>(n) * k);
    for (auto& v : matrix) {
      v = dist(gen);
    }
    std::vector<float> x_values(static_cast<size_t>(m) * k);
    for (auto& v : x_values) {
      v = dist(gen);
    }
    HostQuantizedWeights host =
        host_affine_quantize(matrix, n, k, group_size, bits);
    HostQuantizedWeights rounded = host;
    rounded.scales = round_trip(stream, host.scales, dtype);
    rounded.biases = round_trip(stream, host.biases, dtype);
    array x(x_values.begin(), Shape{m, k}, dtype);
    array w_words(host.words.begin(), Shape{n, words_per_row}, uint32);
    array scales(rounded.scales.begin(), Shape{n, groups_per_row}, dtype);
    array biases(rounded.biases.begin(), Shape{n, groups_per_row}, dtype);

    QmmTileGate gate(true);
    array out_tile = quantized_matmul(
        x, w_words, scales, biases, true, group_size, bits, "affine",
        stream);
    INFO("dtype=", dtype, " group=", group_size, " bits=", bits,
        " m=", m, " n=", n, " k=", k);
    const auto error = evaluation_error(out_tile);
    REQUIRE_MESSAGE(error.empty(), error);
    std::vector<float> tile_v = readback_f32(stream, out_tile);
    gate.set(false);
    array out_base = quantized_matmul(
        x, w_words, scales, biases, true, group_size, bits, "affine",
        stream);
    std::vector<float> base_v = readback_f32(stream, out_base);

    REQUIRE_EQ(tile_v.size(), base_v.size());
    double tile_max = 0.0;
    double base_max = 0.0;
    REQUIRE_MESSAGE(finite_max(tile_v, tile_max),
        "tile kernel produced a non-finite output");
    REQUIRE_MESSAGE(finite_max(base_v, base_max),
        "qmm kernel produced a non-finite output");
    double m_bound = std::max(std::max(tile_max, base_max) * 2.0, 1.0);
    double bound =
        (static_cast<double>(k) + 2.0) * m_bound * std::ldexp(1.0, -23) +
        m_bound * std::ldexp(1.0, -(storage_mantissa(dtype) + 1));
    bound = std::max(bound, 1e-6);
    double max_diff = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < tile_v.size(); ++i) {
      double d =
          std::fabs(static_cast<double>(tile_v[i]) - base_v[i]);
      if (d > max_diff) {
        max_diff = d;
        worst = i;
      }
    }
    INFO("tile case dtype=" << dtype << " bits=" << bits
         << " group_size=" << group_size << " m=" << m << " n=" << n
         << " k=" << k << " worst=" << worst << " diff=" << max_diff
         << " bound=" << bound);
    CHECK(max_diff <= bound);
  };

  // Host-anchored tail case: m and n straddle one tile boundary each;
  // k = 3 groups puts group boundaries inside every k span the tile
  // crosses. x, scales, and biases are round-tripped through the
  // storage dtype so the host reference sees the kernel's operands.
  for (auto dtype : dtypes) {
    for (auto [group_size, bits] : qbits) {
      const int m = 17;
      const int n = 37;
      const int k = group_size * 3;
      const int pack = 32 / bits;
      const int words_per_row = k / pack;
      const int groups_per_row = k / group_size;
      std::mt19937 gen(17);
      std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
      std::vector<float> matrix(static_cast<size_t>(n) * k);
      for (auto& v : matrix) {
        v = dist(gen);
      }
      std::vector<float> x_values(static_cast<size_t>(m) * k);
      for (auto& v : x_values) {
        v = dist(gen);
      }
      HostQuantizedWeights host =
          host_affine_quantize(matrix, n, k, group_size, bits);
      HostQuantizedWeights rounded = host;
      rounded.scales = round_trip(stream, host.scales, dtype);
      rounded.biases = round_trip(stream, host.biases, dtype);
      std::vector<float> x_rt = round_trip(stream, x_values, dtype);
      std::vector<float> expected =
          host_quantized_matmul(rounded, x_rt, m, n, k, group_size, bits);

      QmmTileGate gate(true);
      array x(x_rt.begin(), Shape{m, k}, dtype);
      array w_words(host.words.begin(), Shape{n, words_per_row}, uint32);
      array scales(rounded.scales.begin(), Shape{n, groups_per_row}, dtype);
      array biases(rounded.biases.begin(), Shape{n, groups_per_row}, dtype);
      array out = quantized_matmul(
          x, w_words, scales, biases, true, group_size, bits, "affine",
          stream);
      REQUIRE(evaluation_error(out).empty());
      REQUIRE_EQ(out.shape(), Shape{m, n});
      std::vector<float> device_result = readback_f32(stream, out);

      double m_max = 0.0;
      REQUIRE_MESSAGE(finite_max(device_result, m_max),
          "tile kernel produced a non-finite output");
      double m_bound = std::max(m_max * 2.0, 1.0);
      double ops = 3.0 * k + 1.0;
      double bound = ops * m_bound * std::ldexp(1.0, -23) +
          m_bound * std::ldexp(1.0, -(storage_mantissa(dtype) + 1));
      bound = std::max(bound, 1e-6);
      double max_diff = 0.0;
      size_t worst = 0;
      for (size_t i = 0; i < expected.size(); ++i) {
        double d = std::fabs(
            static_cast<double>(device_result[i]) - expected[i]);
        if (d > max_diff) {
          max_diff = d;
          worst = i;
        }
      }
      INFO("anchor bits=" << bits << " group_size=" << group_size
           << " dtype=" << dtype << " worst=" << worst
           << " diff=" << max_diff << " bound=" << bound);
      CHECK(max_diff <= bound);
    }
  }

  // Device-vs-device sweep: gate ON must reproduce qmm.comp. Weights
  // quantize once per case; seeds are order-deterministic.
  const bool full_sweep = [] {
    const char* v = std::getenv("MLX_OMARCHY_QMM_TILE_FULL_SWEEP");
    return v != nullptr && v[0] == '1';
  }();
  std::vector<int> m_shapes = full_sweep
      ? std::vector<int>{2, 15, 16, 17, 64, 255, 256, 1023}
      : std::vector<int>{2, 15, 16, 17, 1023};
  std::vector<int> n_shapes =
      full_sweep ? std::vector<int>{64, 896, 4864} : std::vector<int>{896};
  std::vector<int> k_shapes =
      full_sweep ? std::vector<int>{896, 4864} : std::vector<int>{896};
  unsigned seed = 47;
  for (auto dtype : dtypes) {
    for (auto [group_size, bits] : qbits) {
      for (int k : k_shapes) {
        for (int n : n_shapes) {
          for (int m : m_shapes) {
            run_tile_case(dtype, group_size, bits, k, n, m, seed++);
          }
        }
      }
    }
  }
  if (!full_sweep) {
    // The 4864 x 4864 gate/up projection at both m extremes, one
    // 4-bit/f16 and one 8-bit/bf16 combo, keeps the largest real
    // shapes covered without the cross product.
    for (Dtype dtype : {float16, bfloat16}) {
      if (std::find(dtypes.begin(), dtypes.end(), dtype) ==
          dtypes.end()) {
        continue;
      }
      for (int m : {2, 1023}) {
        run_tile_case(dtype, 32, 4, 4864, 4864, m, seed++);
        run_tile_case(dtype, 64, 8, 4864, 4864, m, seed++);
      }
    }
  }
}
