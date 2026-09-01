// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/compute.h"

#include <stdexcept>
#include <utility>

#include "mlx/backend/omarchy/vulkan.h"

#include "arange_bf16.h"
#include "arange_f16.h"
#include "arange_f32.h"
#include "arange_i32.h"
#include "argreduce_bf16.h"
#include "argreduce_f16.h"
#include "argreduce_f32.h"
#include "cast_bf16_f16.h"
#include "cast_bf16_f32.h"
#include "cast_f16_f32.h"
#include "cast_f16_bf16.h"
#include "cast_f32_bf16.h"
#include "cast_f32_f16.h"
#include "cast_bf16_i32.h"
#include "cast_f16_i32.h"
#include "cast_f32_i32.h"
#include "cast_i32_bf16.h"
#include "cast_i32_f16.h"
#include "cast_i32_f32.h"
#include "elementwise_bf16.h"
#include "elementwise_f16.h"
#include "elementwise_f32.h"
#include "gather_bf16.h"
#include "gather_f16.h"
#include "gather_f32.h"
#include "copy_general_bf16.h"
#include "copy_general_f16.h"
#include "copy_general_f32.h"
#include "fill_bf16.h"
#include "fill_f16.h"
#include "fill_f32.h"
#include "greater_equal_i32.h"
#include "matmul_bf16.h"
#include "matmul_f16.h"
#include "matmul_f32.h"
#include "select_bf16.h"
#include "select_f16.h"
#include "select_f32.h"
#include "reduce_bf16.h"
#include "reduce_f16.h"
#include "reduce_f32.h"
#include "logsumexp_bf16.h"
#include "logsumexp_f16.h"
#include "logsumexp_f32.h"
#include "softmax_bf16.h"
#include "softmax_f16.h"
#include "softmax_f32.h"
#include "sort_bf16.h"
#include "sort_f16.h"
#include "sort_f32.h"
#include "argsort_bf16.h"
#include "argsort_f16.h"
#include "argsort_f32.h"

namespace mlx::core::omarchy {

namespace {

using ShaderBytes = std::pair<const unsigned char*, size_t>;

ShaderBytes shader_bytes(ComputeKernel kernel) {
  using namespace shaders;
  switch (kernel) {
    case ComputeKernel::ElementwiseF32:
      return {elementwise_f32, elementwise_f32_size};
    case ComputeKernel::ElementwiseF16:
      return {elementwise_f16, elementwise_f16_size};
    case ComputeKernel::ElementwiseBF16:
      return {elementwise_bf16, elementwise_bf16_size};
    case ComputeKernel::CastF16F32:
      return {cast_f16_f32, cast_f16_f32_size};
    case ComputeKernel::CastF32F16:
      return {cast_f32_f16, cast_f32_f16_size};
    case ComputeKernel::CastBF16F32:
      return {cast_bf16_f32, cast_bf16_f32_size};
    case ComputeKernel::CastF32BF16:
      return {cast_f32_bf16, cast_f32_bf16_size};
    case ComputeKernel::CastBF16F16:
      return {cast_bf16_f16, cast_bf16_f16_size};
    case ComputeKernel::CastF16BF16:
      return {cast_f16_bf16, cast_f16_bf16_size};
    case ComputeKernel::CastI32F32:
      return {cast_i32_f32, cast_i32_f32_size};
    case ComputeKernel::CastF32I32:
      return {cast_f32_i32, cast_f32_i32_size};
    case ComputeKernel::CastI32F16:
      return {cast_i32_f16, cast_i32_f16_size};
    case ComputeKernel::CastF16I32:
      return {cast_f16_i32, cast_f16_i32_size};
    case ComputeKernel::CastI32BF16:
      return {cast_i32_bf16, cast_i32_bf16_size};
    case ComputeKernel::CastBF16I32:
      return {cast_bf16_i32, cast_bf16_i32_size};
    case ComputeKernel::ArgReduceF32:
      return {argreduce_f32, argreduce_f32_size};
    case ComputeKernel::ArgReduceF16:
      return {argreduce_f16, argreduce_f16_size};
    case ComputeKernel::ArgReduceBF16:
      return {argreduce_bf16, argreduce_bf16_size};
    case ComputeKernel::ArangeF32:
      return {arange_f32, arange_f32_size};
    case ComputeKernel::ArangeF16:
      return {arange_f16, arange_f16_size};
    case ComputeKernel::ArangeBF16:
      return {arange_bf16, arange_bf16_size};
    case ComputeKernel::ArangeI32:
      return {arange_i32, arange_i32_size};
    case ComputeKernel::ReduceF32:
      return {reduce_f32, reduce_f32_size};
    case ComputeKernel::ReduceF16:
      return {reduce_f16, reduce_f16_size};
    case ComputeKernel::ReduceBF16:
      return {reduce_bf16, reduce_bf16_size};
    case ComputeKernel::MatmulF32:
      return {matmul_f32, matmul_f32_size};
    case ComputeKernel::MatmulF16:
      return {matmul_f16, matmul_f16_size};
    case ComputeKernel::MatmulBF16:
      return {matmul_bf16, matmul_bf16_size};
    case ComputeKernel::FillF32:
      return {fill_f32, fill_f32_size};
    case ComputeKernel::FillF16:
      return {fill_f16, fill_f16_size};
    case ComputeKernel::FillBF16:
      return {fill_bf16, fill_bf16_size};
    case ComputeKernel::SoftmaxF32:
      return {softmax_f32, softmax_f32_size};
    case ComputeKernel::SoftmaxF16:
      return {softmax_f16, softmax_f16_size};
    case ComputeKernel::SoftmaxBF16:
      return {softmax_bf16, softmax_bf16_size};
    case ComputeKernel::LogSumExpF32:
      return {logsumexp_f32, logsumexp_f32_size};
    case ComputeKernel::LogSumExpF16:
      return {logsumexp_f16, logsumexp_f16_size};
    case ComputeKernel::LogSumExpBF16:
      return {logsumexp_bf16, logsumexp_bf16_size};
    case ComputeKernel::SelectF32:
      return {select_f32, select_f32_size};
    case ComputeKernel::SelectF16:
      return {select_f16, select_f16_size};
    case ComputeKernel::SelectBF16:
      return {select_bf16, select_bf16_size};
    case ComputeKernel::GreaterEqualI32:
      return {greater_equal_i32, greater_equal_i32_size};
    case ComputeKernel::GatherF32:
      return {gather_f32, gather_f32_size};
    case ComputeKernel::GatherF16:
      return {gather_f16, gather_f16_size};
    case ComputeKernel::GatherBF16:
      return {gather_bf16, gather_bf16_size};
    case ComputeKernel::CopyGeneralF32:
      return {copy_general_f32, copy_general_f32_size};
    case ComputeKernel::CopyGeneralF16:
      return {copy_general_f16, copy_general_f16_size};
    case ComputeKernel::CopyGeneralBF16:
      return {copy_general_bf16, copy_general_bf16_size};
    case ComputeKernel::ArgSortF32:
      return {argsort_f32, argsort_f32_size};
    case ComputeKernel::ArgSortF16:
      return {argsort_f16, argsort_f16_size};
    case ComputeKernel::ArgSortBF16:
      return {argsort_bf16, argsort_bf16_size};
    case ComputeKernel::SortF32:
      return {sort_f32, sort_f32_size};
    case ComputeKernel::SortF16:
      return {sort_f16, sort_f16_size};
    case ComputeKernel::SortBF16:
      return {sort_bf16, sort_bf16_size};
    case ComputeKernel::Count:
      break;
  }
  throw std::invalid_argument("[omarchy] invalid compute kernel.");
}

} // namespace

ComputeRuntime::ComputeRuntime(VkDevice device) : device_(device) {
  auto& dt = vk::device_table();
  std::array<VkDescriptorSetLayoutBinding, kComputeBindingCount> bindings{};
  for (uint32_t index = 0; index < bindings.size(); ++index) {
    bindings[index].binding = index;
    bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[index].descriptorCount = 1;
    bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }

  VkDescriptorSetLayoutCreateInfo descriptor_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  descriptor_info.bindingCount = static_cast<uint32_t>(bindings.size());
  descriptor_info.pBindings = bindings.data();
  VKX_CHECK(dt.CreateDescriptorSetLayout(
      device_, &descriptor_info, nullptr, &descriptor_layout_));

  VkPushConstantRange push_range{};
  push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push_range.size = sizeof(ComputeParams);
  VkPipelineLayoutCreateInfo pipeline_info{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipeline_info.setLayoutCount = 1;
  pipeline_info.pSetLayouts = &descriptor_layout_;
  pipeline_info.pushConstantRangeCount = 1;
  pipeline_info.pPushConstantRanges = &push_range;
  try {
    VKX_CHECK(dt.CreatePipelineLayout(
        device_, &pipeline_info, nullptr, &pipeline_layout_));
  } catch (...) {
    dt.DestroyDescriptorSetLayout(device_, descriptor_layout_, nullptr);
    descriptor_layout_ = VK_NULL_HANDLE;
    throw;
  }
}

ComputeRuntime::~ComputeRuntime() {
  auto& dt = vk::device_table();
  for (VkPipeline pipeline : pipelines_) {
    if (pipeline != VK_NULL_HANDLE) {
      dt.DestroyPipeline(device_, pipeline, nullptr);
    }
  }
  if (pipeline_layout_ != VK_NULL_HANDLE) {
    dt.DestroyPipelineLayout(device_, pipeline_layout_, nullptr);
  }
  if (descriptor_layout_ != VK_NULL_HANDLE) {
    dt.DestroyDescriptorSetLayout(device_, descriptor_layout_, nullptr);
  }
}

VkPipeline ComputeRuntime::pipeline(ComputeKernel kernel) {
  size_t index = static_cast<size_t>(kernel);
  if (index >= pipelines_.size()) {
    throw std::invalid_argument("[omarchy] invalid compute kernel.");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (pipelines_[index] == VK_NULL_HANDLE) {
    pipelines_[index] = create_pipeline(kernel);
  }
  return pipelines_[index];
}

VkPipeline ComputeRuntime::create_pipeline(ComputeKernel kernel) {
  auto& dt = vk::device_table();
  auto [bytes, size] = shader_bytes(kernel);
  if (size == 0 || size % sizeof(uint32_t) != 0) {
    throw std::runtime_error("[omarchy] embedded SPIR-V has an invalid size.");
  }

  VkShaderModuleCreateInfo shader_info{
      VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  shader_info.codeSize = size;
  shader_info.pCode = reinterpret_cast<const uint32_t*>(bytes);
  VkShaderModule shader{VK_NULL_HANDLE};
  VKX_CHECK(dt.CreateShaderModule(device_, &shader_info, nullptr, &shader));

  VkPipelineShaderStageCreateInfo stage{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage.module = shader;
  stage.pName = "main";
  VkComputePipelineCreateInfo pipeline_info{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pipeline_info.stage = stage;
  pipeline_info.layout = pipeline_layout_;

  VkPipeline pipeline{VK_NULL_HANDLE};
  try {
    VKX_CHECK(dt.CreateComputePipelines(
        device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline));
  } catch (...) {
    dt.DestroyShaderModule(device_, shader, nullptr);
    throw;
  }
  dt.DestroyShaderModule(device_, shader, nullptr);
  return pipeline;
}

} // namespace mlx::core::omarchy
