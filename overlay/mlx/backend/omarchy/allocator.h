// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <mutex>
#include <utility>
#include <vector>
#include "mlx/allocator.h"
#include "mlx/backend/common/buffer_cache.h"

namespace mlx::core::omarchy {

using allocator::Buffer;

// Host-visible buffer. Apple GPUs are UMA: Honeykrisp exposes a device-local
// heap that is host visible and host coherent, so the mapped pointer is both
// the CPU access path and the GPU buffer backing. When a driver offers no
// coherent type the allocator falls back to HOST_VISIBLE memory and the
// runtime performs explicit vkFlushMappedMemoryRanges /
// vkInvalidateMappedMemoryRanges maintenance. Each buffer owns one
// VkDeviceMemory allocation, which keeps mapped-range lifetime exact.
struct VulkanBuffer {
  VkBuffer buffer{VK_NULL_HANDLE};
  VkDeviceMemory memory{VK_NULL_HANDLE};
  void* data{nullptr};
  size_t size{0};
  // False when the memory type lacks HOST_COHERENT.
  bool coherent{true};
};

class VulkanAllocator : public allocator::Allocator {
 public:
  Buffer malloc(size_t size) override;
  void free(Buffer buffer) override;
  size_t size(Buffer buffer) const override;

  size_t get_active_memory() const {
    std::unique_lock lk(mutex_);
    return active_memory_;
  }
  size_t get_peak_memory() const {
    std::unique_lock lk(mutex_);
    return peak_memory_;
  }
  void reset_peak_memory() {
    std::unique_lock lk(mutex_);
    peak_memory_ = 0;
  }
  size_t get_memory_limit() const {
    std::unique_lock lk(mutex_);
    return memory_limit_;
  }
  size_t set_memory_limit(size_t limit) {
    std::unique_lock lk(mutex_);
    std::swap(memory_limit_, limit);
    return limit;
  }
  size_t get_cache_memory() const {
    std::unique_lock lk(mutex_);
    return buffer_cache_.cache_size();
  }
  size_t set_cache_limit(size_t limit);
  void clear_cache();

  // Cache maintenance for non-coherent mapped memory (Khronos guidance:
  // HOST_VISIBLE alone does not guarantee coherence). The encoder flushes
  // before submission and invalidates after the fence wait. Buffers on
  // coherent memory types are skipped.
  void flush_noncoherent(VkDevice device);
  void invalidate_noncoherent(VkDevice device);

 private:
  VulkanAllocator();
  friend VulkanAllocator& allocator();

  void destroy_buffer(VulkanBuffer* buf);
  // Resolve a memory type with the required flags, preferring types backed by
  // a device-local heap (unified memory on Apple GPUs).
  uint32_t find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags required)
      const;

  mutable std::mutex mutex_;
  size_t memory_limit_;
  size_t cache_limit_;
  size_t active_memory_{0};
  size_t peak_memory_{0};
  mutable BufferCache<VulkanBuffer> buffer_cache_;
  std::vector<VulkanBuffer*> noncoherent_;
};

VulkanAllocator& allocator();

// Round a request to the allocator page so cached buffers can be reused for
// any smaller request.
size_t round_size(size_t size);

} // namespace mlx::core::omarchy
