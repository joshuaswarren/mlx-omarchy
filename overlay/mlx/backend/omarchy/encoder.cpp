// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/encoder.h"

#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/trace.h"

#include "mlx/backend/omarchy/vulkan.h"
#include "mlx/scheduler.h"
#include "mlx/utils.h"

namespace mlx::core::omarchy {

CommandEncoder::CommandEncoder(Device& device) : device_(device) {
  auto& dt = vk::device_table();
  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
      VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pci.queueFamilyIndex = device_.queue_family();
  VKX_CHECK(dt.CreateCommandPool(device_.handle(), &pci, nullptr, &pool_));

  VkCommandBufferAllocateInfo ai{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = pool_;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  VKX_CHECK(dt.AllocateCommandBuffers(device_.handle(), &ai, &cmd_));
}

CommandEncoder::~CommandEncoder() {
  // Release pending work (and its temporaries) while the device is alive.
  // A wedged queue throws here; swallow so destruction can continue, the
  // bounded error already surfaced through synchronize().
  try {
    synchronize();
  } catch (const std::exception&) {
  }
  auto& dt = vk::device_table();
  if (pool_ != VK_NULL_HANDLE) {
    dt.DestroyCommandPool(device_.handle(), pool_, nullptr);
  }
}

// Join this encoder's newest in-flight submission, if any, and refresh
// host mappings. After this, the command buffer is guaranteed to have
// left the pending state, so it can legally be begun again, and host
// reads see the submission's final bytes.
void CommandEncoder::join_last_completion() {
  if (last_completion_ == 0) {
    return;
  }
  device_.completions().wait(last_completion_);
  last_completion_ = 0;
  omarchy::allocator().invalidate_noncoherent(device_.handle());
}

void CommandEncoder::ensure_recording() {
  if (recording_) {
    return;
  }
  // A prior commit on this stream can still be executing: beginning a
  // pending command buffer is invalid usage
  // (VUID-vkBeginCommandBuffer-commandBuffer-00048). Join the previous
  // submission before reusing cmd_. Commits stay asynchronous; only the
  // next record on the same stream pays the wait.
  join_last_completion();
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VKX_CHECK(vk::device_table().BeginCommandBuffer(cmd_, &bi));
  recording_ = true;
}

void CommandEncoder::copy_buffer(
    VkBuffer src,
    VkBuffer dst,
    VkDeviceSize size,
    VkDeviceSize src_offset,
    VkDeviceSize dst_offset) {
  ensure_recording();
  VkBufferCopy region{};
  region.srcOffset = src_offset;
  region.dstOffset = dst_offset;
  region.size = size;
  vk::device_table().CmdCopyBuffer(cmd_, src, dst, 1, &region);
  node_count_++;
  trace::counters().vk_buffer_copies++;
}

void CommandEncoder::fill_buffer(
    VkBuffer dst,
    uint32_t value,
    VkDeviceSize size,
    VkDeviceSize offset) {
  ensure_recording();
  vk::device_table().CmdFillBuffer(cmd_, dst, offset, size, value);
  node_count_++;
  trace::counters().vk_buffer_fills++;
}

void CommandEncoder::commit() {
  // Handler-only commits (no recorded commands, no user semaphores) still
  // submit and signal the completion timeline, so handlers run after prior
  // queue work and nothing enqueued is ever dropped.
  if (!recording_ && wait_semaphores_.empty() && signal_semaphores_.empty() &&
      completed_handlers_.empty()) {
    return;
  }
  submit();
}

void CommandEncoder::synchronize() {
  commit();
  join_last_completion();
}

void CommandEncoder::submit() {
  auto& dt = vk::device_table();

  if (recording_) {
    VKX_CHECK(dt.EndCommandBuffer(cmd_));
  }

  std::vector<VkSemaphore> wait_sems;
  std::vector<uint64_t> wait_values;
  wait_sems.reserve(wait_semaphores_.size());
  wait_values.reserve(wait_semaphores_.size());
  for (auto& pending : wait_semaphores_) {
    wait_sems.push_back(pending.semaphore);
    wait_values.push_back(pending.value);
  }
  std::vector<VkPipelineStageFlags> wait_stages(
      wait_sems.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

  VkTimelineSemaphoreSubmitInfo timeline{
      VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
  timeline.waitSemaphoreValueCount = static_cast<uint32_t>(wait_values.size());
  timeline.pWaitSemaphoreValues = wait_values.data();

  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.pNext = &timeline;
  si.waitSemaphoreCount = static_cast<uint32_t>(wait_sems.size());
  si.pWaitSemaphores = wait_sems.data();
  si.pWaitDstStageMask = wait_stages.data();
  si.commandBufferCount = recording_ ? 1u : 0u;
  si.pCommandBuffers = recording_ ? &cmd_ : nullptr;

  // Every submission also signals the device completion timeline; user
  // (event) signals ride along in the same submission, in order.
  std::vector<VkSemaphore> signal_sems;
  std::vector<uint64_t> signal_values;
  signal_sems.reserve(signal_semaphores_.size() + 1);
  signal_values.reserve(signal_semaphores_.size() + 1);
  for (auto& pending : signal_semaphores_) {
    signal_sems.push_back(pending.semaphore);
    signal_values.push_back(pending.value);
  }

  // Ownership moves to the dispatcher entry: buffer temporaries keep the
  // arrays' backing alive, each pending-semaphore keepalive keeps its Event
  // alive, and handlers run on the completion thread. All of it releases
  // exactly when the GPU work finishes.
  std::vector<std::shared_ptr<void>> keepalive = std::move(temporaries_);
  temporaries_.clear();
  for (auto& pending : wait_semaphores_) {
    keepalive.push_back(std::move(pending.keepalive));
  }
  for (auto& pending : signal_semaphores_) {
    keepalive.push_back(std::move(pending.keepalive));
  }

  {
    // VkQueue is externally synchronized: the lock covers completion-value
    // assignment (timeline signals must increase along the queue) through
    // QueueSubmit and the dispatcher enqueue. It is never held across a
    // host wait.
    std::lock_guard<std::mutex> lk(device_.queue_mutex());
    // Khronos guidance: HOST_VISIBLE memory without HOST_COHERENT needs an
    // explicit flush before submission.
    omarchy::allocator().flush_noncoherent(device_.handle());
    uint64_t completion_value = device_.completions().reserve();
    VkSemaphore completion_sem = device_.completions().semaphore();
    signal_sems.push_back(completion_sem);
    signal_values.push_back(completion_value);
    timeline.signalSemaphoreValueCount =
        static_cast<uint32_t>(signal_values.size());
    timeline.pSignalSemaphoreValues = signal_values.data();
    si.signalSemaphoreCount = static_cast<uint32_t>(signal_sems.size());
    si.pSignalSemaphores = signal_sems.data();
    try {
      VKX_CHECK(dt.QueueSubmit(device_.queue(), 1, &si, VK_NULL_HANDLE));
    } catch (...) {
      // The submission never reached the driver: the ended command buffer
      // and the pending semaphore lists are dead (their keepalives have
      // already moved into the local payload and die with this frame).
      // Reset the encoder so it can be reused or destroyed cleanly; the
      // typed error propagates to the stream's error handling.
      recording_ = false;
      node_count_ = 0;
      wait_semaphores_.clear();
      signal_semaphores_.clear();
      completed_handlers_.clear();
      throw;
    }
    // Publish only after the submit: the dispatcher must never wait on a
    // value whose submission has not been handed to the driver.
    device_.completions().enqueue(
        completion_value, std::move(keepalive), std::move(completed_handlers_));
    last_completion_ = completion_value;
    trace::counters().vk_submissions++;
  }

  recording_ = false;
  node_count_ = 0;
  wait_semaphores_.clear();
  signal_semaphores_.clear();
  completed_handlers_.clear();

  // No vkResetCommandBuffer: the buffer may still be executing.
  // ensure_recording() joins last_completion_ before the next
  // BeginCommandBuffer, which then resets the buffer implicitly (the
  // pool was created with RESET_COMMAND_BUFFER_BIT).
}

CommandEncoder& get_command_encoder(Stream s) {
  return get_command_encoders().at(s.index);
}

std::unordered_map<int, CommandEncoder>& get_command_encoders() {
  static thread_local std::unordered_map<int, CommandEncoder> encoders;
  return encoders;
}

std::unordered_map<int, CommandEncoder>& get_global_command_encoders() {
  static std::unordered_map<int, CommandEncoder> encoders;
  return encoders;
}

} // namespace mlx::core::omarchy
