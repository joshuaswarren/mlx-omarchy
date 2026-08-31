// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>

namespace mlx::core::omarchy {

inline constexpr uint32_t kComputeThreadsPerGroup = 256;
inline constexpr uint32_t kMaxComputeGroupCountX = 65535;

constexpr uint32_t compute_dispatch_group_count(uint32_t count) {
  if (count == 0) {
    return 0;
  }
  uint32_t groups = (count - 1) / kComputeThreadsPerGroup + 1;
  return groups < kMaxComputeGroupCountX ? groups : kMaxComputeGroupCountX;
}

constexpr bool compute_index_span_fits(uint64_t offset, uint64_t count) {
  constexpr uint64_t max_index = std::numeric_limits<uint32_t>::max();
  return count <= max_index && offset <= max_index &&
      (count == 0 || count - 1 <= max_index - offset);
}

enum class ComputeKernel : uint8_t {
  ElementwiseF32,
  ElementwiseF16,
  CastF16F32,
  CastF32F16,
  ReduceF32,
  ReduceF16,
  Count,
};

struct ComputeBinding {
  VkBuffer buffer;
  VkDeviceSize offset;
  VkDeviceSize range;
};

struct ComputeParams {
  uint32_t count{0};
  uint32_t operation{0};
  uint32_t lhs_size{0};
  uint32_t rhs_size{0};
  uint32_t reduce_size{0};
  uint32_t output_size{0};
  uint32_t lhs_offset{0};
  uint32_t rhs_offset{0};
  uint32_t output_offset{0};
};

class ComputeRuntime {
 public:
  explicit ComputeRuntime(VkDevice device);
  ~ComputeRuntime();

  ComputeRuntime(const ComputeRuntime&) = delete;
  ComputeRuntime& operator=(const ComputeRuntime&) = delete;

  VkPipeline pipeline(ComputeKernel kernel);
  VkPipelineLayout pipeline_layout() const {
    return pipeline_layout_;
  }
  VkDescriptorSetLayout descriptor_layout() const {
    return descriptor_layout_;
  }

 private:
  VkPipeline create_pipeline(ComputeKernel kernel);

  VkDevice device_;
  VkDescriptorSetLayout descriptor_layout_{VK_NULL_HANDLE};
  VkPipelineLayout pipeline_layout_{VK_NULL_HANDLE};
  std::array<VkPipeline, static_cast<size_t>(ComputeKernel::Count)> pipelines_{};
  std::mutex mutex_;
};

} // namespace mlx::core::omarchy
