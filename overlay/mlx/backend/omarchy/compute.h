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
inline constexpr uint32_t kComputeBindingCount = 4;

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
  ElementwiseBF16,
  CastF16F32,
  CastF32F16,
  CastBF16F32,
  CastF32BF16,
  CastBF16F16,
  CastF16BF16,
  CastI32F32,
  CastF32I32,
  CastI32F16,
  CastF16I32,
  CastI32BF16,
  CastBF16I32,
  ReduceF32,
  ReduceF16,
  ReduceBF16,
  MatmulF32,
  MatmulF16,
  MatmulBF16,
  FillF32,
  FillF16,
  FillBF16,
  SoftmaxF32,
  SoftmaxF16,
  SoftmaxBF16,
  LogSumExpF32,
  LogSumExpF16,
  LogSumExpBF16,
  SelectF32,
  SelectF16,
  SelectBF16,
  GreaterEqualI32,
  GatherF32,
  GatherF16,
  GatherBF16,
  CopyGeneralF32,
  CopyGeneralF16,
  CopyGeneralBF16,
  ArgReduceF32,
  ArgReduceF16,
  ArgReduceBF16,
  ArangeF32,
  ArangeF16,
  ArangeBF16,
  ArangeI32,
  SortF32,
  SortF16,
  SortBF16,
  ArgSortF32,
  ArgSortF16,
  ArgSortBF16,
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
  uint32_t aux_size{0};
  uint32_t aux_offset{0};
  uint32_t matrix_m{0};
  uint32_t matrix_n{0};
  uint32_t matrix_k{0};
  // Matmul flag bits: 1 = rhs transposed, 2 = bias c used,
  // 4 = lhs transposed. Batch routing is data-driven: dims is the batch
  // axis count, shape[] the batch extents, and in_strides/out_strides[]
  // the per-operand batch strides in elements (0 = broadcast axis). The
  // shader unravels workgroup z over shape[] and offsets each operand.
  uint32_t flags{0};
  float alpha{1.0f};
  float beta{0.0f};
  // Broadcast rank for elementwise kernels; batch axis count for Matmul
  // kernels (0 for rank-2).
  uint32_t dims{0};
  uint32_t shape[4]{};
  uint32_t in_strides[4]{};
  uint32_t out_strides[4]{};
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
