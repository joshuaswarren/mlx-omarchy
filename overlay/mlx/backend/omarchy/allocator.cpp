// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/allocator.h"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>

#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/vulkan.h"

namespace mlx::core {

namespace omarchy {

constexpr size_t kPageSize = 4096;

size_t round_size(size_t size) {
  if (size <= kPageSize) {
    return kPageSize;
  }
  return kPageSize * ((size + kPageSize - 1) / kPageSize);
}

uint32_t VulkanAllocator::find_memory_type(
    uint32_t type_bits,
    VkMemoryPropertyFlags required) const {
  const auto& mem = device().memory_properties();
  uint32_t fallback = UINT32_MAX;
  for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
    const auto& type = mem.memoryTypes[i];
    if (!(type_bits & (1u << i))) {
      continue;
    }
    if ((type.propertyFlags & required) != required) {
      continue;
    }
    bool device_local =
        mem.memoryHeaps[type.heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
    if (device_local) {
      return i;
    }
    if (fallback == UINT32_MAX) {
      fallback = i;
    }
  }
  return fallback;
}

VulkanAllocator::VulkanAllocator()
    : buffer_cache_(
          kPageSize,
          [](VulkanBuffer* buf) { return buf->size; },
          [this](VulkanBuffer* buf) { destroy_buffer(buf); }) {
  size_t total = is_available() ? capability_report(0).total_memory : 0;
  memory_limit_ = total > 0 ? (total / 100) * 90 : (1ull << 32);
  cache_limit_ = 32ul << 20; // 32 MB default cache limit
}

Buffer VulkanAllocator::malloc(size_t size) {
  auto& dt = vk::device_table();
  if (size == 0) {
    return Buffer{new VulkanBuffer{}};
  }
  size = round_size(size);

  std::unique_lock lk(mutex_);
  if (void* cached = buffer_cache_.reuse_from_cache(size)) {
    auto* buf = static_cast<VulkanBuffer*>(cached);
    active_memory_ += buf->size;
    peak_memory_ = std::max(active_memory_, peak_memory_);
    lk.unlock();
    return Buffer{buf};
  }
  lk.unlock();

  auto* buf = new VulkanBuffer{};
  buf->size = size;

  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = size;
  bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VKX_CHECK(dt.CreateBuffer(device().handle(), &bci, nullptr, &buf->buffer));

  VkMemoryRequirements reqs{};
  dt.GetBufferMemoryRequirements(device().handle(), buf->buffer, &reqs);

  // Prefer host-visible coherent memory (unified memory on Apple GPUs). Fall
  // back to plain host-visible memory with explicit cache maintenance; fail
  // closed when even that is unavailable.
  uint32_t type_index = find_memory_type(
      reqs.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (type_index != UINT32_MAX) {
    buf->coherent = true;
  } else {
    type_index = find_memory_type(
        reqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    buf->coherent = false;
  }
  if (type_index == UINT32_MAX) {
    dt.DestroyBuffer(device().handle(), buf->buffer, nullptr);
    delete buf;
    throw std::runtime_error(
        "[omarchy] no host-visible memory type available;"
        " the backend cannot stage CPU data without one.");
  }

  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.allocationSize = reqs.size;
  mai.memoryTypeIndex = type_index;
  VKX_CHECK(dt.AllocateMemory(device().handle(), &mai, nullptr, &buf->memory));
  VKX_CHECK(
      dt.BindBufferMemory(device().handle(), buf->buffer, buf->memory, 0));
  VKX_CHECK(dt.MapMemory(
      device().handle(), buf->memory, 0, VK_WHOLE_SIZE, 0, &buf->data));

  lk.lock();
  active_memory_ += buf->size;
  peak_memory_ = std::max(active_memory_, peak_memory_);
  if (!buf->coherent) {
    noncoherent_.push_back(buf);
  }
  lk.unlock();
  return Buffer{buf};
}

void VulkanAllocator::destroy_buffer(VulkanBuffer* buf) {
  // Cache clear/release funnels through here while callers already hold
  // mutex_, so the non-coherent registry drops the buffer before its
  // memory dies; later flush/invalidate must never see it.
  std::erase(noncoherent_, buf);
  auto& dt = vk::device_table();
  if (buf->memory != VK_NULL_HANDLE) {
    dt.UnmapMemory(device().handle(), buf->memory);
    dt.FreeMemory(device().handle(), buf->memory, nullptr);
  }
  if (buf->buffer != VK_NULL_HANDLE) {
    dt.DestroyBuffer(device().handle(), buf->buffer, nullptr);
  }
  delete buf;
}

void VulkanAllocator::free(Buffer buffer) {
  auto* buf = static_cast<VulkanBuffer*>(buffer.ptr());
  if (!buf) {
    return;
  }
  size_t sz = buf->size;
  std::unique_lock lk(mutex_);
  active_memory_ -= sz;
  if (sz > 0 && buffer_cache_.cache_size() + sz <= cache_limit_) {
    buffer_cache_.recycle_to_cache(buf);
    return;
  }
  // The lock stays held: destroy_buffer deregisters the buffer from
  // noncoherent_ under the same lock the cache paths already hold.
  destroy_buffer(buf);
}

size_t VulkanAllocator::size(Buffer buffer) const {
  auto* buf = static_cast<VulkanBuffer*>(buffer.ptr());
  return buf ? buf->size : 0;
}

size_t VulkanAllocator::set_cache_limit(size_t limit) {
  std::unique_lock lk(mutex_);
  std::swap(cache_limit_, limit);
  if (buffer_cache_.cache_size() > cache_limit_) {
    buffer_cache_.release_cached_buffers(
        buffer_cache_.cache_size() - cache_limit_);
  }
  return limit;
}

void VulkanAllocator::clear_cache() {
  std::unique_lock lk(mutex_);
  buffer_cache_.clear();
}

void VulkanAllocator::flush_noncoherent(VkDevice device) {
  std::vector<VkMappedMemoryRange> ranges;
  {
    std::lock_guard lk(mutex_);
    ranges.reserve(noncoherent_.size());
    for (auto* buf : noncoherent_) {
      if (!buf || !buf->memory) {
        continue;
      }
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = buf->memory;
      range.offset = 0;
      range.size = VK_WHOLE_SIZE;
      ranges.push_back(range);
    }
  }
  if (!ranges.empty()) {
    VKX_CHECK(
        vk::device_table().FlushMappedMemoryRanges(
            device, static_cast<uint32_t>(ranges.size()), ranges.data()));
  }
}

void VulkanAllocator::invalidate_noncoherent(VkDevice device) {
  std::vector<VkMappedMemoryRange> ranges;
  {
    std::lock_guard lk(mutex_);
    ranges.reserve(noncoherent_.size());
    for (auto* buf : noncoherent_) {
      if (!buf || !buf->memory) {
        continue;
      }
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = buf->memory;
      range.offset = 0;
      range.size = VK_WHOLE_SIZE;
      ranges.push_back(range);
    }
  }
  if (!ranges.empty()) {
    VKX_CHECK(
        vk::device_table().InvalidateMappedMemoryRanges(
            device, static_cast<uint32_t>(ranges.size()), ranges.data()));
  }
}

VulkanAllocator& allocator() {
  static VulkanAllocator allocator_;
  return allocator_;
}

} // namespace omarchy

namespace allocator {

Allocator& allocator() {
  return omarchy::allocator();
}

void* Buffer::raw_ptr() {
  if (!ptr_) {
    return nullptr;
  }
  return static_cast<omarchy::VulkanBuffer*>(ptr_)->data;
}

bool can_reuse_alien_buffer(void*) {
  return true;
}

} // namespace allocator

size_t get_active_memory() {
  return omarchy::allocator().get_active_memory();
}
size_t get_peak_memory() {
  return omarchy::allocator().get_peak_memory();
}
void reset_peak_memory() {
  omarchy::allocator().reset_peak_memory();
}
size_t set_memory_limit(size_t limit) {
  return omarchy::allocator().set_memory_limit(limit);
}
size_t get_memory_limit() {
  return omarchy::allocator().get_memory_limit();
}
size_t get_cache_memory() {
  return omarchy::allocator().get_cache_memory();
}
size_t set_cache_limit(size_t limit) {
  return omarchy::allocator().set_cache_limit(limit);
}
void clear_cache() {
  omarchy::allocator().clear_cache();
}

// Wired limits are a Metal feature; Omarchy has no equivalent (same as CUDA).
size_t set_wired_limit(size_t) {
  return 0;
}

} // namespace mlx::core
