// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <vulkan/vulkan.h>

#include <stdexcept>
#include <string>

namespace mlx::core::omarchy::vk {

// The backend talks to Vulkan through the system loader only. There is no
// compile-time link against libvulkan: the loader is opened with dlopen and
// every function is resolved with vkGetInstanceProcAddr / vkGetDeviceProcAddr.
// This keeps the mlx library loadable on machines without a Vulkan ICD and
// gives clean, typed errors when the loader or a driver is missing.

// Resolve the loader-level entry points. Idempotent. Returns false when
// libvulkan cannot be opened or lacks vkGetInstanceProcAddr.
bool load_loader();

// Loader-level entry point used to fill the tables.
extern PFN_vkGetInstanceProcAddr GetInstanceProcAddr;

// Instance-level functions, resolved after vkCreateInstance.
struct InstanceTable {
  PFN_vkCreateInstance CreateInstance{nullptr};
  PFN_vkEnumerateInstanceVersion EnumerateInstanceVersion{nullptr};
  PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices{nullptr};
  PFN_vkEnumerateDeviceExtensionProperties EnumerateDeviceExtensionProperties{
      nullptr};
  PFN_vkGetPhysicalDeviceProperties2 GetPhysicalDeviceProperties2{nullptr};
  PFN_vkGetPhysicalDeviceFeatures2 GetPhysicalDeviceFeatures2{nullptr};
  PFN_vkGetPhysicalDeviceMemoryProperties2 GetPhysicalDeviceMemoryProperties2{
      nullptr};
  PFN_vkGetPhysicalDeviceQueueFamilyProperties
      GetPhysicalDeviceQueueFamilyProperties{nullptr};
  PFN_vkCreateDevice CreateDevice{nullptr};
  PFN_vkDestroyInstance DestroyInstance{nullptr};
};

InstanceTable& instance_table();
// Device-level functions, resolved with vkGetDeviceProcAddr at device
// creation.
struct DeviceTable {
  PFN_vkGetDeviceProcAddr GetDeviceProcAddr{nullptr};
  PFN_vkDestroyDevice DestroyDevice{nullptr};
  PFN_vkGetDeviceQueue GetDeviceQueue{nullptr};
  PFN_vkDeviceWaitIdle DeviceWaitIdle{nullptr};
  PFN_vkAllocateMemory AllocateMemory{nullptr};
  PFN_vkFreeMemory FreeMemory{nullptr};
  PFN_vkMapMemory MapMemory{nullptr};
  PFN_vkUnmapMemory UnmapMemory{nullptr};
  PFN_vkCreateBuffer CreateBuffer{nullptr};
  PFN_vkDestroyBuffer DestroyBuffer{nullptr};
  PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements{nullptr};
  PFN_vkBindBufferMemory BindBufferMemory{nullptr};
  PFN_vkCreateCommandPool CreateCommandPool{nullptr};
  PFN_vkDestroyCommandPool DestroyCommandPool{nullptr};
  PFN_vkAllocateCommandBuffers AllocateCommandBuffers{nullptr};
  PFN_vkFreeCommandBuffers FreeCommandBuffers{nullptr};
  PFN_vkResetCommandPool ResetCommandPool{nullptr};
  PFN_vkBeginCommandBuffer BeginCommandBuffer{nullptr};
  PFN_vkEndCommandBuffer EndCommandBuffer{nullptr};
  PFN_vkResetCommandBuffer ResetCommandBuffer{nullptr};
  PFN_vkCmdCopyBuffer CmdCopyBuffer{nullptr};
  PFN_vkCmdFillBuffer CmdFillBuffer{nullptr};
  PFN_vkCreateShaderModule CreateShaderModule{nullptr};
  PFN_vkDestroyShaderModule DestroyShaderModule{nullptr};
  PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout{nullptr};
  PFN_vkDestroyDescriptorSetLayout DestroyDescriptorSetLayout{nullptr};
  PFN_vkCreateDescriptorPool CreateDescriptorPool{nullptr};
  PFN_vkDestroyDescriptorPool DestroyDescriptorPool{nullptr};
  PFN_vkAllocateDescriptorSets AllocateDescriptorSets{nullptr};
  PFN_vkUpdateDescriptorSets UpdateDescriptorSets{nullptr};
  PFN_vkCreatePipelineLayout CreatePipelineLayout{nullptr};
  PFN_vkDestroyPipelineLayout DestroyPipelineLayout{nullptr};
  PFN_vkCreateComputePipelines CreateComputePipelines{nullptr};
  PFN_vkDestroyPipeline DestroyPipeline{nullptr};
  PFN_vkCmdBindPipeline CmdBindPipeline{nullptr};
  PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets{nullptr};
  PFN_vkCmdPushConstants CmdPushConstants{nullptr};
  PFN_vkCmdDispatch CmdDispatch{nullptr};
  PFN_vkCmdPipelineBarrier CmdPipelineBarrier{nullptr};
  PFN_vkQueueSubmit QueueSubmit{nullptr};
  PFN_vkQueueWaitIdle QueueWaitIdle{nullptr};
  PFN_vkCreateFence CreateFence{nullptr};
  PFN_vkDestroyFence DestroyFence{nullptr};
  PFN_vkResetFences ResetFences{nullptr};
  PFN_vkWaitForFences WaitForFences{nullptr};
  PFN_vkCreateSemaphore CreateSemaphore{nullptr};
  PFN_vkDestroySemaphore DestroySemaphore{nullptr};
  PFN_vkGetSemaphoreCounterValue GetSemaphoreCounterValue{nullptr};
  PFN_vkWaitSemaphores WaitSemaphores{nullptr};
  PFN_vkFlushMappedMemoryRanges FlushMappedMemoryRanges{nullptr};
  PFN_vkInvalidateMappedMemoryRanges InvalidateMappedMemoryRanges{nullptr};
};

// Filled by device creation. Valid for the lifetime of the runtime because
// the mlx backend keeps a single VkDevice per supported physical device.
DeviceTable& device_table();

// Printable VkResult for error messages and receipts.
const char* result_string(VkResult result);

} // namespace mlx::core::omarchy::vk

// Run a Vulkan call and turn every non-success VkResult into a descriptive
// std::runtime_error. Every Vulkan entry point in the backend goes through
// this so failures are never silent.
#define VKX_CHECK(call)                                         \
  do {                                                          \
    VkResult vkx_res_ = (call);                                 \
    if (vkx_res_ != VK_SUCCESS) {                               \
      throw std::runtime_error(                                 \
          std::string("[omarchy] ") + #call + " failed with " + \
          mlx::core::omarchy::vk::result_string(vkx_res_));     \
    }                                                           \
  } while (false)
