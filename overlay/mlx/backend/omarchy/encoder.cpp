// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/encoder.h"
#include <stdexcept>

#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/backend/omarchy/gpu_profiler.h"
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

  std::array<VkCommandBuffer, kInFlightCommandBuffers> buffers{};
  VkCommandBufferAllocateInfo ai{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = pool_;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = kInFlightCommandBuffers;
  VKX_CHECK(dt.AllocateCommandBuffers(device_.handle(), &ai, buffers.data()));
  for (int i = 0; i < kInFlightCommandBuffers; ++i) {
    slots_[i].cmd = buffers[i];
  }
  prof::get().attach(this, device_);
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
  if (desc_pool_ != VK_NULL_HANDLE) {
    dt.DestroyDescriptorPool(device_.handle(), desc_pool_, nullptr);
    desc_pool_ = VK_NULL_HANDLE;
  }
  if (pool_ != VK_NULL_HANDLE) {
    dt.DestroyCommandPool(device_.handle(), pool_, nullptr);
  }
}

// Join this encoder's newest in-flight submission, if any, and refresh
// host mappings. After this, every command buffer this encoder submitted
// has left the pending state (the completion timeline is strictly
// ordered), so they can legally be begun again, and host reads see the
// submissions' final bytes.
void CommandEncoder::join_last_completion() {
  if (last_completion_ == 0) {
    return;
  }
  uint64_t value = last_completion_;
  uint64_t join_t0 = prof::get().profiling() ? prof::host_ns() : 0;
  device_.completions().wait(value);
  uint64_t wait_t1 = prof::get().profiling() ? prof::host_ns() : 0;
  last_completion_ = 0;
  omarchy::allocator().invalidate_noncoherent(device_.handle());
  uint64_t inval_t2 = prof::get().profiling() ? prof::host_ns() : 0;
  prof::get().on_join(this, value, join_t0, wait_t1, inval_t2);
}

void CommandEncoder::ensure_recording() {
  if (recording_) {
    return;
  }
  uint64_t begin_t0 = prof::get().profiling() ? prof::host_ns() : 0;
  // Acquire a ring slot whose submission has completed. Newer submissions
  // keep executing on the device while this batch records; only when all
  // slots are in flight does the host join the oldest. The profiler's
  // begin cost includes this wait: it is the residual host-side stall the
  // 2026-09-02 profile attributed to per-record joins.
  auto& completions = device_.completions();
  int chosen = -1;
  uint64_t oldest_value = UINT64_MAX;
  int oldest = -1;
  for (int i = 0; i < kInFlightCommandBuffers; ++i) {
    uint64_t value = slots_[i].in_flight;
    if (value == 0 || completions.drained_value() >= value) {
      slots_[i].in_flight = 0;
      chosen = i;
      break;
    }
    if (value < oldest_value) {
      oldest_value = value;
      oldest = i;
    }
  }
  if (chosen < 0) {
    completions.wait(oldest_value);
    slots_[oldest].in_flight = 0;
    chosen = oldest;
  }
  current_slot_ = chosen;
  cmd_ = slots_[chosen].cmd;
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VKX_CHECK(vk::device_table().BeginCommandBuffer(cmd_, &bi));
  prof::get().on_begin(
      this, chosen, cmd_, begin_t0 != 0 ? prof::host_ns() - begin_t0 : 0);
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

// Allocate one descriptor set from the cached pool. A pool serves up to
// kDescriptorSetsPerPool dispatches; on exhaustion it is retired into the
// currently-recording submission's temporaries, so it is destroyed only
// after that submission completes — which is strictly after every earlier
// submission whose batches allocated sets from it (completion timeline
// values increase along the queue).
VkDescriptorSet CommandEncoder::acquire_descriptor_set(
    ComputeRuntime& compute) {
  auto& dt = vk::device_table();
  // MLX_OMARCHY_TAPE_NO_REUSE (diagnostic, docs/install-omarchy.md):
  // every dispatch gets its own descriptor pool with exactly one set, so
  // no pool - and therefore no set - is shared with any other dispatch.
  // The pool retires into the current submission's temporaries with the
  // same lifetime rule as a retired cached pool below.
  if (tape_no_reuse()) {
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = compute.binding_limit();
    VkDescriptorPoolCreateInfo pool_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    VkDescriptorPool pool{VK_NULL_HANDLE};
    VKX_CHECK(dt.CreateDescriptorPool(
        device_.handle(), &pool_info, nullptr, &pool));
    temporaries_.push_back(std::shared_ptr<VkDescriptorPool>(
        new VkDescriptorPool(pool),
        [device = device_.handle()](VkDescriptorPool* owned) {
          vk::device_table().DestroyDescriptorPool(device, *owned, nullptr);
          delete owned;
        }));
    VkDescriptorSetLayout descriptor_layout = compute.descriptor_layout();
    VkDescriptorSetAllocateInfo allocate_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate_info.descriptorPool = pool;
    allocate_info.descriptorSetCount = 1;
    allocate_info.pSetLayouts = &descriptor_layout;
    VkDescriptorSet descriptor_set{VK_NULL_HANDLE};
    VKX_CHECK(dt.AllocateDescriptorSets(
        device_.handle(), &allocate_info, &descriptor_set));
    return descriptor_set;
  }
  if (desc_pool_ == VK_NULL_HANDLE || desc_pool_remaining_ == 0) {
    if (desc_pool_ != VK_NULL_HANDLE) {
      temporaries_.push_back(std::shared_ptr<VkDescriptorPool>(
          new VkDescriptorPool(desc_pool_),
          [device = device_.handle()](VkDescriptorPool* owned) {
            vk::device_table().DestroyDescriptorPool(device, *owned, nullptr);
            delete owned;
          }));
      desc_pool_ = VK_NULL_HANDLE;
    }
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount =
        kDescriptorSetsPerPool * compute.binding_limit();
    VkDescriptorPoolCreateInfo pool_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = kDescriptorSetsPerPool;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    VKX_CHECK(dt.CreateDescriptorPool(
        device_.handle(), &pool_info, nullptr, &desc_pool_));
    desc_pool_remaining_ = kDescriptorSetsPerPool;
  }
  VkDescriptorSetLayout descriptor_layout = compute.descriptor_layout();
  VkDescriptorSetAllocateInfo allocate_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  allocate_info.descriptorPool = desc_pool_;
  allocate_info.descriptorSetCount = 1;
  allocate_info.pSetLayouts = &descriptor_layout;
  VkDescriptorSet descriptor_set{VK_NULL_HANDLE};
  VKX_CHECK(dt.AllocateDescriptorSets(
      device_.handle(), &allocate_info, &descriptor_set));
  desc_pool_remaining_--;
  return descriptor_set;
}

void CommandEncoder::dispatch_compute(
    ComputeKernel kernel,
    std::span<const ComputeBinding> bindings,
    const ComputeParams& params,
    uint32_t group_count_x,
    uint32_t group_count_y,
    uint32_t group_count_z) {
  if (group_count_x == 0 || group_count_y == 0 || group_count_z == 0) {
    return;
  }
  auto& compute = device_.compute();
  uint32_t binding_limit = compute.binding_limit();
  if (bindings.empty() || bindings.size() > binding_limit) {
    throw std::invalid_argument(
        "[omarchy] compute dispatch needs " +
        std::to_string(bindings.size()) +
        " storage-buffer bindings; this device allows " +
        std::to_string(binding_limit) + ".");
  }
  group_count_x = std::min(group_count_x, kMaxComputeGroupCountX);
  group_count_y = std::min(group_count_y, kMaxComputeGroupCountX);
  group_count_z = std::min(group_count_z, kMaxComputeGroupCountX);

  auto& dt = vk::device_table();
  VkPipeline pipeline = compute.pipeline(kernel);

  VkDescriptorSet descriptor_set = acquire_descriptor_set(compute);

  std::array<VkDescriptorBufferInfo, kComputeBindingBudget> buffer_info{};
  std::array<VkWriteDescriptorSet, kComputeBindingBudget> writes{};
  for (uint32_t index = 0; index < bindings.size(); ++index) {
    buffer_info[index] = {
        bindings[index].buffer, bindings[index].offset, bindings[index].range};
    writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[index].dstSet = descriptor_set;
    writes[index].dstBinding = index;
    writes[index].descriptorCount = 1;
    writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[index].pBufferInfo = &buffer_info[index];
  }
  dt.UpdateDescriptorSets(
      device_.handle(),
      static_cast<uint32_t>(bindings.size()),
      writes.data(),
      0,
      nullptr);

  ensure_recording();
  uint64_t host_t0 = prof::get().profiling() ? prof::host_ns() : 0;
  // MLX_OMARCHY_TAPE_FULL_BARRIERS (diagnostic, docs/install-omarchy.md):
  // the heaviest correct dependency - all commands, all memory access,
  // both directions - ahead of every dispatch, on top of the regular
  // barriers below. Probes whether the driver drops an in-buffer
  // dependency the regular barriers already express.
  if (tape_full_barriers()) {
    VkMemoryBarrier full{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    full.srcAccessMask =
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    full.dstAccessMask =
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    dt.CmdPipelineBarrier(
        cmd_,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        0,
        1,
        &full,
        0,
        nullptr,
        0,
        nullptr);
  }
  VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  before.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
      VK_ACCESS_SHADER_WRITE_BIT;
  before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  dt.CmdPipelineBarrier(
      cmd_,
      VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0,
      1,
      &before,
      0,
      nullptr,
      0,
      nullptr);
  prof::get().before_dispatch(this, current_slot_, cmd_);

  VkPipelineLayout pipeline_layout = compute.pipeline_layout();
  dt.CmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  dt.CmdBindDescriptorSets(
      cmd_,
      VK_PIPELINE_BIND_POINT_COMPUTE,
      pipeline_layout,
      0,
      1,
      &descriptor_set,
      0,
      nullptr);
  dt.CmdPushConstants(
      cmd_,
      pipeline_layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(params),
      &params);
  dt.CmdDispatch(cmd_, group_count_x, group_count_y, group_count_z);
  prof::get().after_dispatch(
      this,
      current_slot_,
      cmd_,
      kernel,
      params,
      bindings,
      group_count_x,
      group_count_y,
      group_count_z,
      host_t0 != 0 ? prof::host_ns() - host_t0 : 0);

  VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  after.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT |
      VK_ACCESS_HOST_READ_BIT;
  dt.CmdPipelineBarrier(
      cmd_,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
          VK_PIPELINE_STAGE_HOST_BIT,
      0,
      1,
      &after,
      0,
      nullptr,
      0,
      nullptr);
  if (tape_full_barriers()) {
    // Diagnostic: matching full barrier out of this dispatch, so every
    // dependency between two dispatches is the heaviest form (see the
    // pre-dispatch barrier above).
    VkMemoryBarrier full{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    full.srcAccessMask =
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    full.dstAccessMask =
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    dt.CmdPipelineBarrier(
        cmd_,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        0,
        1,
        &full,
        0,
        nullptr,
        0,
        nullptr);
  }

  node_count_++;
  trace::counters().vk_compute_dispatches++;
}

void CommandEncoder::commit() {
  if (!recording_ && wait_semaphores_.empty() && signal_semaphores_.empty() &&
      completed_handlers_.empty()) {
    trace::counters().commit_calls_noop++;
    return;
  }
  trace::counters().commit_calls_with_work++;
  submit();
}

void CommandEncoder::synchronize() {
  commit();
  join_last_completion();
}

void CommandEncoder::submit() {
  auto& dt = vk::device_table();
  bool was_recording = recording_;
  uint64_t submit_t0 = prof::get().profiling() ? prof::host_ns() : 0;
  uint64_t submitted = 0;

  if (recording_) {
    VKX_CHECK(dt.EndCommandBuffer(cmd_));
  }

  std::vector<VkSemaphore> wait_sems;
  std::vector<uint64_t> wait_values;
  wait_sems.reserve(wait_semaphores_.size() + 1);
  wait_values.reserve(wait_semaphores_.size() + 1);
  // In-order stream: this submission waits for the stream's previous
  // submission. Vulkan defines no execution or memory dependency between
  // submissions without a semaphore wait, and the per-dispatch barriers
  // cover hazards inside one command buffer only. Without this wait, a
  // submission queued behind long work can read a prior tiny submission's
  // output before that write lands (Honeykrisp: compiled 4-bit decode
  // read an eager one-element f32 as recycled page garbage, 20/20).
  // Waiting on an already-signaled timeline value is a driver
  // pass-through, so the shallow-queue case pays nothing.
  if (last_completion_ != 0) {
    wait_sems.push_back(device_.completions().semaphore());
    wait_values.push_back(last_completion_);
  }
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
    submitted = completion_value;
    if (was_recording) {
      slots_[current_slot_].in_flight = completion_value;
    }
    trace::counters().vk_submissions++;
  }

  recording_ = false;
  node_count_ = 0;
  wait_semaphores_.clear();
  signal_semaphores_.clear();
  completed_handlers_.clear();
  prof::get().on_submit_end(
      this,
      submitted,
      submit_t0 != 0 ? prof::host_ns() - submit_t0 : 0,
      current_slot_);

  // No vkResetCommandBuffer: the buffer may still be executing. The ring
  // slot is marked in flight above; ensure_recording() only reuses a slot
  // whose submission completed (or joins the oldest), and BeginCommandBuffer
  // then resets the buffer implicitly (the pool was created with
  // RESET_COMMAND_BUFFER_BIT).
}

CommandEncoder& get_command_encoder(Stream s) {
  // Mirrors the CUDA backend: the per-thread table misses for a stream
  // created on another thread, so fall back to the global thread-unsafe
  // table and finally raise the upstream std::runtime_error contract.
  // unordered_map::at would throw std::out_of_range, which escapes the
  // caller's catch(std::runtime_error) and terminates the process.
  auto& encoders = get_command_encoders();
  auto it = encoders.find(s.index);
  if (it == encoders.end()) {
    auto& global_encoders = get_global_command_encoders();
    it = global_encoders.find(s.index);
    if (it == global_encoders.end()) {
      throw std::runtime_error(
          "There is no Stream(gpu, " + std::to_string(s.index) +
          ") in current thread.");
    }
  }
  return it->second;
}

std::unordered_map<int, CommandEncoder>& get_command_encoders() {
  static thread_local std::unordered_map<int, CommandEncoder> encoders;
  return encoders;
}

std::unordered_map<int, CommandEncoder>& get_global_command_encoders() {
  static std::unordered_map<int, CommandEncoder> global_encoders;
  return global_encoders;
}

} // namespace mlx::core::omarchy
