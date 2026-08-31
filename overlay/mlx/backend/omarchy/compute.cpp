// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/compute.h"

#include <stdexcept>
#include <utility>

#include "mlx/backend/omarchy/vulkan.h"

#include "cast_f16_f32.h"
#include "cast_f32_f16.h"
#include "elementwise_f16.h"
#include "elementwise_f32.h"
#include "matmul_f16.h"
#include "matmul_f32.h"
#include "reduce_f16.h"
#include "reduce_f32.h"

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
    case ComputeKernel::CastF16F32:
      return {cast_f16_f32, cast_f16_f32_size};
    case ComputeKernel::CastF32F16:
      return {cast_f32_f16, cast_f32_f16_size};
    case ComputeKernel::ReduceF32:
      return {reduce_f32, reduce_f32_size};
    case ComputeKernel::ReduceF16:
      return {reduce_f16, reduce_f16_size};
    case ComputeKernel::MatmulF32:
      return {matmul_f32, matmul_f32_size};
    case ComputeKernel::MatmulF16:
      return {matmul_f16, matmul_f16_size};
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
