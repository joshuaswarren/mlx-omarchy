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
  CastBoolF32,
  CastF32F16,
  CastBF16F32,
  CastF32BF16,
  CastBF16F16,
  CastF16BF16,
  CastI32F32,
  CastU32F32,
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
  CompareF32,
  CompareF16,
  CompareBF16,
  CompareI32,
  LogicalOrBool,
  GatherF32,
  GatherF16,
  GatherBF16,
  GatherU32,
  CopyGeneralF32,
  CopyGeneralF16,
  CopyGeneralBF16,
  CopyGeneralU32,
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
  RandomBitsU32,
  ElementwiseI32,
  ElementwiseU32,
  ScanF32,
  ScanF16,
  ScanBF16,
  SearchSortedF32,
  SearchSortedF16,
  SearchSortedBF16,
  SearchSortedI32,
  SearchSortedU32,
  ReduceGeneralF32,
  ReduceGeneralF16,
  ReduceGeneralBF16,
  ReduceGeneralI32,
  ReduceGeneralU32,
  AnyAllF32,
  AnyAllF16,
  AnyAllBF16,
  AnyAllI32,
  AnyAllU32,
  AnyAllBool,
  ScanGeneralF32,
  ScanGeneralF16,
  ScanGeneralBF16,
  ScanGeneralI32,
  ScanGeneralU32,
  HadamardF32,
  HadamardF16,
  HadamardBF16,
  QmmF32,
  QmmF16,
  QmmBF16,
  DequantF32,
  DequantF16,
  ConvF32,
  ConvF16,
  ConvBF16,
  // Wave 5: indexing and scatter. The U32 kernels carry bitwise word
  // storage, so float32 shares them with int32 and uint32.
  GatherAxisU32,
  GatherAxisF16,
  GatherAxisBF16,
  ScatterU32,
  ScatterF16,
  ScatterBF16,
  ScatterAxisU32,
  ScatterAxisF16,
  ScatterAxisBF16,
  MaskedScatterU32,
  MaskedScatterF16,
  MaskedScatterBF16,
  ClearU32,
  BlockMaskF32,
  GatherMmF32,
  GatherMmF16,
  GatherMmBF16,
  SegmentedMmF32,
  SegmentedMmF16,
  SegmentedMmBF16,
  GatherQmmF32,
  GatherQmmF16,
  GatherQmmBF16,
  GatherQmmNbF32,
  GatherQmmNbF16,
  GatherQmmNbBF16,
  // Wave 8: FFT. FftF32 is the radix-2 Cooley-Tukey pass (complex64 pairs
  // in shared memory); FftRealF32 strips the real part of a complex64
  // buffer into float32 for the irfft tail.
  FftF32,
  FftRealF32,
  // Wave 9: fused and custom kernels. Norm forward and VJP kernels run one
  // workgroup per row with float32 arithmetic; the dw kernel runs a single
  // workgroup with per-column accumulators. ConvertFP8 pairs travel as
  // little-endian uint32 word packs of four E4M3 bytes.
  FastRmsNormF32,
  FastRmsNormF16,
  FastRmsNormBF16,
  FastLayerNormF32,
  FastLayerNormF16,
  FastLayerNormBF16,
  FastRmsNormVjpDxF32,
  FastRmsNormVjpDxF16,
  FastRmsNormVjpDxBF16,
  FastLayerNormVjpDxF32,
  FastLayerNormVjpDxF16,
  FastLayerNormVjpDxBF16,
  FastRmsNormVjpDwF32,
  FastRmsNormVjpDwF16,
  FastRmsNormVjpDwBF16,
  FastLayerNormVjpDwF32,
  FastLayerNormVjpDwF16,
  FastLayerNormVjpDwBF16,
  CrossEntropyVjpF32,
  CrossEntropyVjpF16,
  CrossEntropyVjpBF16,
  CrossEntropyF32,
  CrossEntropyF16,
  CrossEntropyBF16,
  Fp8ToF32,
  Fp8ToF16,
  Fp8ToBF16,
  Fp8FromF32,
  Fp8FromF16,
  Fp8FromBF16,
  // Wave 7: linear algebra. One workgroup per batch matrix, float32
  // only, matching the upstream CPU dtype contract; SVD runs as a
  // sweeps kernel plus a separate finalize kernel.
  LinalgCholeskyF32,
  LinalgInverseF32,
  LinalgLuF32,
  LinalgQrF32,
  LinalgEighF32,
  LinalgSvdF32,
  LinalgSvdFinalizeF32,
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
