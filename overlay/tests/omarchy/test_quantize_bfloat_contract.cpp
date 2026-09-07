// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/ops.h"
#include "mlx/stream.h"
#include "mlx/types/half_types.h"

using namespace mlx::core;

namespace {

void skip(const char* reason) {
  std::cout << "Skipping: " << reason << "\n";
}

bool bfloat16_available() {
  if (!gpu::is_available()) {
    skip(
        "no qualifying Vulkan device (set MLX_OMARCHY_ALLOW_NON_APPLE=1 on"
        " a development machine).");
    return false;
  }
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.storage_buffer_16bit_access || !capabilities.shader_int16) {
    skip("Vulkan device lacks required bfloat16 storage features.");
    return false;
  }
  return true;
}

float to_bfloat16(float value) {
  return static_cast<float>(bfloat16_t(value));
}

std::vector<uint32_t> pack_affine(
    const std::vector<float>& values,
    int bits,
    float scale,
    float bias) {
  const uint32_t max_code = (1u << bits) - 1u;
  std::vector<uint32_t> words(values.size() * bits / 32, 0);
  for (size_t index = 0; index < values.size(); ++index) {
    float rounded = std::round((values[index] - bias) / scale);
    uint32_t code = static_cast<uint32_t>(
        std::clamp(rounded, 0.0f, static_cast<float>(max_code)));
    size_t bit = index * bits;
    uint32_t shift = static_cast<uint32_t>(bit % 32);
    words[bit / 32] |= code << shift;
    if (shift + bits > 32) {
      words[bit / 32 + 1] |= code >> (32 - shift);
    }
  }
  return words;
}

struct AffineReference {
  std::vector<uint32_t> words;
  std::vector<float> dequantized;
  float scale;
  float bias;
  bool packing_depends_on_unrounded_parameters;
};

AffineReference affine_reference(
    const std::vector<float>& values,
    int bits) {
  float low = *std::min_element(values.begin(), values.end());
  float high = std::max(0.0f, *std::max_element(values.begin(), values.end()));
  float bins = static_cast<float>((1u << bits) - 1u);
  float scale = std::max((high - low) / bins, 1e-7f);
  bool low_dominant = std::abs(low) > std::abs(high);
  scale = low_dominant ? scale : -scale;
  float edge = low_dominant ? low : high;
  float q0 = std::round(edge / scale);
  float bias = 0.0f;
  if (q0 != 0.0f) {
    scale = edge / q0;
    bias = edge;
  }

  std::vector<uint32_t> words = pack_affine(values, bits, scale, bias);
  float stored_scale = to_bfloat16(scale);
  float stored_bias = to_bfloat16(bias);
  std::vector<uint32_t> stored_parameter_words =
      pack_affine(values, bits, stored_scale, stored_bias);
  bool packing_depends_on_unrounded_parameters =
      stored_parameter_words != words;
  std::vector<float> dequantized(values.size());
  uint32_t mask = (1u << bits) - 1u;
  for (size_t index = 0; index < values.size(); ++index) {
    size_t bit = index * bits;
    uint32_t shift = static_cast<uint32_t>(bit % 32);
    uint32_t code = words[bit / 32] >> shift;
    if (shift + bits > 32) {
      code |= words[bit / 32 + 1] << (32 - shift);
    }
    code &= mask;
    dequantized[index] =
        to_bfloat16(static_cast<float>(code) * stored_scale + stored_bias);
  }
  return {
      std::move(words),
      std::move(dequantized),
      stored_scale,
      stored_bias,
      packing_depends_on_unrounded_parameters};
}

void synchronize_encoder(const Stream& stream) {
  omarchy::get_command_encoder(stream).synchronize();
}

} // namespace

TEST_CASE("affine bfloat16 quantize and dequantize match the pinned contract") {
  if (!bfloat16_available()) {
    return;
  }
  set_default_device(Device::gpu);
  Stream stream = new_stream(Device::gpu);

  bool exercised_unrounded_parameter_packing = false;
  for (int group_size : {32, 64, 128}) {
    std::vector<float> source_values(group_size);
    for (size_t index = 0; index < source_values.size(); ++index) {
      source_values[index] = -0.7f + static_cast<float>(index) *
          (1.7f / static_cast<float>(group_size - 1));
    }
    source_values.front() = -0.7f;
    source_values.back() = 1.0f;
    std::vector<float> input_values(source_values.size());
    std::transform(
        source_values.begin(),
        source_values.end(),
        input_values.begin(),
        to_bfloat16);

    for (int bits : {2, 3, 4, 5, 6, 8}) {
      CAPTURE(group_size);
      CAPTURE(bits);
      AffineReference expected = affine_reference(input_values, bits);
      exercised_unrounded_parameter_packing |=
          expected.packing_depends_on_unrounded_parameters;

      array source(source_values.begin(), Shape{1, group_size}, float32);
      array input = astype(source, bfloat16, stream);
      input.eval();
      synchronize_encoder(stream);

      auto outputs =
          quantize(input, group_size, bits, "affine", std::nullopt, stream);
      REQUIRE_EQ(outputs.size(), 3);
      uint64_t quantize_before =
          omarchy::trace::counters().vk_compute_dispatches.load();
      outputs[0].eval();
      synchronize_encoder(stream);
      CHECK(
          omarchy::trace::counters().vk_compute_dispatches.load() >
          quantize_before);
      CHECK_EQ(outputs[0].dtype(), uint32);
      CHECK_EQ(outputs[1].dtype(), bfloat16);
      CHECK_EQ(outputs[2].dtype(), bfloat16);
      CHECK_EQ(
          outputs[0].shape(), Shape{1, group_size * bits / 32});
      CHECK_EQ(outputs[1].shape(), Shape{1, 1});
      CHECK_EQ(outputs[2].shape(), Shape{1, 1});

      const uint32_t* words = outputs[0].data<uint32_t>();
      for (size_t index = 0; index < expected.words.size(); ++index) {
        CHECK_EQ(words[index], expected.words[index]);
      }
      CHECK_EQ(
          static_cast<float>(outputs[1].data<bfloat16_t>()[0]),
          expected.scale);
      CHECK_EQ(
          static_cast<float>(outputs[2].data<bfloat16_t>()[0]),
          expected.bias);

      array reconstructed = dequantize(
          outputs[0],
          outputs[1],
          outputs[2],
          group_size,
          bits,
          "affine",
          std::nullopt,
          std::nullopt,
          stream);
      uint64_t dequantize_before =
          omarchy::trace::counters().vk_compute_dispatches.load();
      reconstructed.eval();
      synchronize_encoder(stream);
      CHECK(
          omarchy::trace::counters().vk_compute_dispatches.load() >
          dequantize_before);
      REQUIRE_EQ(reconstructed.dtype(), bfloat16);
      REQUIRE_EQ(reconstructed.shape(), Shape{1, group_size});
      const bfloat16_t* values = reconstructed.data<bfloat16_t>();
      for (size_t index = 0; index < expected.dequantized.size(); ++index) {
        CHECK_EQ(static_cast<float>(values[index]), expected.dequantized[index]);
      }
    }
  }
  CHECK(exercised_unrounded_parameter_packing);
}
