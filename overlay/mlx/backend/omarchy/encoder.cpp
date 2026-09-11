// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/encoder.h"
#include <stdexcept>

#include "mlx/backend/omarchy/host_trace.h"

#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/backend/omarchy/gpu_profiler.h"
#include "mlx/backend/omarchy/vulkan.h"
#include "mlx/scheduler.h"
#include "mlx/utils.h"

namespace mlx::core::omarchy {

namespace {

// Saturating byte-range end for dependency tracking. VK_WHOLE_SIZE and
// overflowing sizes clamp to the address space so a tracked range never
// under-covers the accesses it stands for.
inline VkDeviceSize tracked_range_end(VkDeviceSize offset, VkDeviceSize size) {
  if (size == VK_WHOLE_SIZE) {
    return UINT64_MAX;
  }
  VkDeviceSize end = offset + size;
  return end < offset ? UINT64_MAX : end;
}

} // namespace

bool CommandEncoder::gated_barriers() {
  // MLX_OMARCHY_GATED_BARRIERS (docs/install-omarchy.md): default off.
  // On, dispatch/copy/fill nodes record a barrier only when their buffer
  // ranges overlap unsynced work of the open batch; off, the historic
  // unconditional pre+post dispatch barriers apply. Read once: the gate
  // shapes recorded commands, so flipping it mid-batch would desync the
  // tracker from the command buffer.
  static const bool on = env_flag("MLX_OMARCHY_GATED_BARRIERS");
  return on;
}

bool CommandEncoder::replay_enabled() {
  // MLX_OMARCHY_REPLAY=1 arms the recorded-sequence replay prototype.
  // It reuses recorded command buffers across batches and therefore
  // assumes the default unconditional barrier scheme and the standard
  // descriptor pool; the diagnostic barrier/pool modes change what gets
  // recorded, so replay stays off under them. Read once per process.
  static const bool on = [] {
    if (!env_flag("MLX_OMARCHY_REPLAY")) {
      return false;
    }
    if (gated_barriers() || tape_full_barriers() || tape_no_reuse()) {
      return false;
    }
    return true;
  }();
  return on;
}

bool CommandEncoder::batch_needs_barrier(
    std::span<const TrackedRange> reads,
    std::span<const TrackedRange> writes) const {
  auto overlaps = [](const TrackedRange& a, const TrackedRange& b) {
    return a.buffer == b.buffer && a.offset < b.end && b.offset < a.end;
  };
  // Read after write and write after write.
  for (const auto& r : reads) {
    for (const auto& w : tracked_writes_) {
      if (overlaps(r, w)) {
        return true;
      }
    }
  }
  // Write after write and write after read: compute-to-compute in one
  // queue has no execution dependency without a barrier, so a node
  // writing a range any earlier node read must also wait.
  for (const auto& w : writes) {
    for (const auto& tw : tracked_writes_) {
      if (overlaps(w, tw)) {
        return true;
      }
    }
    for (const auto& tr : tracked_reads_) {
      if (overlaps(w, tr)) {
        return true;
      }
    }
  }
  return false;
}

void CommandEncoder::record_dependency_barrier() {
  // The heaviest correct dependency: all commands, all memory access,
  // both directions. The tracker restarts after it because the barrier
  // orders everything recorded before it.
  VkMemoryBarrier full{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  full.srcAccessMask =
      VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  full.dstAccessMask =
      VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  vk::device_table().CmdPipelineBarrier(
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
  tracked_reads_.clear();
  tracked_writes_.clear();
  head_synced_ = true;
}

void CommandEncoder::reset_dependency_tracking() {
  tracked_reads_.clear();
  tracked_writes_.clear();
  head_synced_ = false;
}

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
  VkEventCreateInfo event_info{VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
  for (int i = 0; i < kInFlightCommandBuffers; ++i) {
    slots_[i].cmd = buffers[i];
    VKX_CHECK(dt.CreateEvent(
        device_.handle(), &event_info, nullptr, &slots_[i].started));
  }
  prof::get().attach(this, device_);
}

CommandEncoder::~CommandEncoder() {
  // Drain pending work (and its temporaries) while the device is alive.
  // A wedged queue throws here; swallow so destruction can continue, the
  // bounded error already surfaced through synchronize().
  //
  // The command and descriptor pools are deliberately NOT destroyed
  // here. After a watchdog throw the newest submission is still in
  // flight, and even after a successful join Mesa signals a submission's
  // semaphores before its submit-final cleanup retires the command
  // buffer (see drain_through), so a pool destroy in this destructor can
  // free state the driver's queue thread still walks - observed as
  // teardown SIGSEGV, pure-virtual aborts, and Khronos-validation-layer
  // crashes after a watchdog throw. The pools are children of the
  // VkDevice; vkDestroyDevice releases them with it.
  try {
    synchronize();
  } catch (const std::exception&) {
  }
}

// Join this encoder's newest in-flight submission, if any, and refresh
// host mappings. After this, every command buffer this encoder submitted
// has left the pending state (the completion timeline is strictly
// ordered), so they can legally be begun again, and host reads see the
// submissions' final bytes.
void CommandEncoder::join_last_completion(const char* reason) {
  if (last_completion_ == 0) {
    return;
  }
  uint64_t value = last_completion_;
  uint64_t join_t0 = prof::get().profiling() ? prof::host_ns() : 0;
  {
    htrace::Scoped _wait(htrace::join_wait);
    device_.completions().wait(value);
  }
  uint64_t wait_t1 = prof::get().profiling() ? prof::host_ns() : 0;
  last_completion_ = 0;
  omarchy::allocator().invalidate_noncoherent(device_.handle());
  uint64_t inval_t2 = prof::get().profiling() ? prof::host_ns() : 0;
  // Replay prototype: replaced command buffers may reset now that every
  // submission through this encoder drained.
  for (auto cmd : replay_retiring_) {
    vk::device_table().ResetCommandBuffer(cmd, 0);
    replay_scratch_.push_back(cmd);
  }
  replay_retiring_.clear();
  prof::get().on_join(this, value, join_t0, wait_t1, inval_t2, reason);
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
  completions.reset_progress_event(slots_[chosen].started);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VKX_CHECK(vk::device_table().BeginCommandBuffer(cmd_, &bi));
  vk::device_table().CmdSetEvent(
      cmd_, slots_[chosen].started, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
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
  if (replay_enabled()) {
    replay_record_transfer(
        true, src, dst, size, src_offset, dst_offset, 0);
    return;
  }
  ensure_recording();
  // The tape-full diagnostic forces the heaviest dependency around the
  // copy and restarts tracking either way.
  if (tape_full_barriers()) {
    record_dependency_barrier();
  }
  if (gated_barriers()) {
    TrackedRange read{src, src_offset, tracked_range_end(src_offset, size)};
    TrackedRange write{dst, dst_offset, tracked_range_end(dst_offset, size)};
    if (!head_synced_ || batch_needs_barrier({&read, 1}, {&write, 1})) {
      record_dependency_barrier();
      trace::counters().barriers_emitted++;
      prof::get().on_barrier(true);
    } else {
      trace::counters().barriers_skipped++;
      prof::get().on_barrier(false);
    }
    tracked_reads_.push_back(read);
    tracked_writes_.push_back(write);
  }
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
  if (replay_enabled()) {
    replay_record_transfer(false, dst, VK_NULL_HANDLE, size, offset, 0, value);
    return;
  }
  ensure_recording();
  bool full_barrier = tape_full_barriers();
  if (full_barrier) {
    record_dependency_barrier();
  }
  if (gated_barriers()) {
    TrackedRange write{dst, offset, tracked_range_end(offset, size)};
    if (!head_synced_ || batch_needs_barrier({}, {&write, 1})) {
      record_dependency_barrier();
      trace::counters().barriers_emitted++;
      prof::get().on_barrier(true);
    } else {
      trace::counters().barriers_skipped++;
      prof::get().on_barrier(false);
    }
    tracked_writes_.push_back(write);
  } else if (!full_barrier) {
    // Host-scalar fills can now share a command buffer. Preserve WAW order
    // between consecutive fills instead of obtaining it from a host drain.
    record_dependency_barrier();
    trace::counters().barriers_emitted++;
    prof::get().on_barrier(true);
  }
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
    retired_pools_.push_back(std::shared_ptr<VkDescriptorPool>(
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
      retired_pools_.push_back(std::shared_ptr<VkDescriptorPool>(
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

// Allocate one descriptor set from the replay prototype's dedicated pool.
// The pool is never retired, so cached sets outlive any single submission
// for the life of the encoder; a genuinely-rotating binding orphans its
// old set here instead of recycling it (noted prototype ceiling).
VkDescriptorSet CommandEncoder::acquire_replay_descriptor_set(
    ComputeRuntime& compute) {
  auto& dt = vk::device_table();
  if (replay_desc_pool_ == VK_NULL_HANDLE || replay_desc_remaining_ == 0) {
    // Chain a fresh pool when the current one runs dry; retired pools stay
    // alive for the encoder's lifetime so cached sets keep their pools.
    if (replay_desc_pool_ != VK_NULL_HANDLE) {
      replay_desc_pools_.push_back(replay_desc_pool_);
    }
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = kReplayPoolSets * compute.binding_limit();
    VkDescriptorPoolCreateInfo pool_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = kReplayPoolSets;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    VKX_CHECK(dt.CreateDescriptorPool(
        device_.handle(), &pool_info, nullptr, &replay_desc_pool_));
    replay_desc_remaining_ = kReplayPoolSets;
  }
  VkDescriptorSetLayout descriptor_layout = compute.descriptor_layout();
  VkDescriptorSetAllocateInfo allocate_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  allocate_info.descriptorPool = replay_desc_pool_;
  allocate_info.descriptorSetCount = 1;
  allocate_info.pSetLayouts = &descriptor_layout;
  VkDescriptorSet descriptor_set{VK_NULL_HANDLE};
  VKX_CHECK(dt.AllocateDescriptorSets(
      device_.handle(), &allocate_info, &descriptor_set));
  replay_desc_remaining_--;
  return descriptor_set;
}

// Allocate a replay command buffer: drained scratch first, else a fresh
// 64-buffer chunk from the replay pool. Handles are never individually
// freed (command-buffer lifetime is pool-scoped); retired ones reset back
// into the scratch list once their last execution drained.
VkCommandBuffer CommandEncoder::acquire_replay_cmd() {
  auto& dt = vk::device_table();
  if (replay_pool_ == VK_NULL_HANDLE) {
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = device_.queue_family();
    VKX_CHECK(dt.CreateCommandPool(
        device_.handle(), &pci, nullptr, &replay_pool_));
  }
  if (replay_scratch_.empty()) {
    std::array<VkCommandBuffer, 64> buffers{};
    VkCommandBufferAllocateInfo ai{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = replay_pool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = buffers.size();
    VKX_CHECK(dt.AllocateCommandBuffers(device_.handle(), &ai, buffers.data()));
    replay_scratch_.assign(buffers.begin(), buffers.end());
  }
  VkCommandBuffer cmd = replay_scratch_.back();
  replay_scratch_.pop_back();
  return cmd;
}

// Copy/fill ops under replay: every op records fresh into its own
// command buffer (rare in decode; no replayable metadata is kept).
void CommandEncoder::replay_record_transfer(
    bool copy,
    VkBuffer a,
    VkBuffer b,
    VkDeviceSize size,
    VkDeviceSize a_offset,
    VkDeviceSize b_offset,
    uint32_t value) {
  auto& dt = vk::device_table();
  VkCommandBuffer cmd = acquire_replay_cmd();
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  VKX_CHECK(dt.BeginCommandBuffer(cmd, &bi));
  if (copy) {
    VkBufferCopy region{};
    region.srcOffset = a_offset;
    region.dstOffset = b_offset;
    region.size = size;
    dt.CmdCopyBuffer(cmd, a, b, 1, &region);
    trace::counters().vk_buffer_copies++;
  } else {
    dt.CmdFillBuffer(cmd, a, a_offset, size, value);
    trace::counters().vk_buffer_fills++;
  }
  VKX_CHECK(dt.EndCommandBuffer(cmd));
  replay_order_.push_back(cmd);
  node_count_++;
}

void CommandEncoder::record_dispatch_locked(
    VkCommandBuffer cmd,
    VkDescriptorSet descriptor_set,
    VkPipeline pipeline,
    VkPipelineLayout pipeline_layout,
    const ComputeParams& params,
    uint32_t group_count_x,
    uint32_t group_count_y,
    uint32_t group_count_z) {
  auto& dt = vk::device_table();
  VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  before.srcAccessMask =
      VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
      VK_ACCESS_SHADER_WRITE_BIT;
  before.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  dt.CmdPipelineBarrier(
      cmd,
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
  dt.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  dt.CmdBindDescriptorSets(
      cmd,
      VK_PIPELINE_BIND_POINT_COMPUTE,
      pipeline_layout,
      0,
      1,
      &descriptor_set,
      0,
      nullptr);
  dt.CmdPushConstants(
      cmd,
      pipeline_layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(params),
      &params);
  dt.CmdDispatch(cmd, group_count_x, group_count_y, group_count_z);
  VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  after.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT |
      VK_ACCESS_HOST_READ_BIT;
  dt.CmdPipelineBarrier(
      cmd,
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
}

bool CommandEncoder::variant_matches(
    const ReplayVariant& v,
    std::span<const ComputeBinding> bindings) {
  if (bindings.size() == 0) {
    return false;
  }
  for (uint32_t i = 0; i < bindings.size(); ++i) {
    if (v.buffers[i] != bindings[i].buffer ||
        v.offsets[i] != bindings[i].offset ||
        v.ranges[i] != bindings[i].range) {
      return false;
    }
  }
  return true;
}

bool CommandEncoder::replay_dispatch(

    VkPipeline pipeline,
    ComputeKernel profile_kernel,
    std::span<const ComputeBinding> bindings,
    const ComputeParams& params,
    uint32_t group_count_x,
    uint32_t group_count_y,
    uint32_t group_count_z) {
  auto& compute = device_.compute();
  auto& dt = vk::device_table();
  if (bindings.empty() || bindings.size() > compute.binding_limit()) {
    throw std::invalid_argument(
        "[omarchy] compute dispatch needs " +
        std::to_string(bindings.size()) +
        " storage-buffer bindings; this device allows " +
        std::to_string(compute.binding_limit()) + ".");
  }
  ReplayEntry* entry =
      replay_cursor_ < replay_entries_.size()
          ? &replay_entries_[replay_cursor_]
          : nullptr;
  bool stable = entry != nullptr && entry->pipeline == pipeline &&
      entry->gx == group_count_x && entry->gy == group_count_y &&
      entry->gz == group_count_z &&
      std::memcmp(&entry->params, &params, sizeof(ComputeParams)) == 0;
  if (stable) {
    // Pipeline, push constants and groups all repeat: look for a binding
    // variant recorded earlier (async decode ping-pongs between two
    // allocator address sets, so up to kReplayMaxVariants are kept).
    for (auto& v : entry->variants) {
      if (variant_matches(v, bindings)) {
        for (const auto& item : bindings) {
          note_binding_owner(item.owner);
        }
        replay_order_.push_back(v.cmd);
        htrace::add(htrace::replay_hits, 1);
        replay_cursor_++;
        return true;
      }
    }
    if (entry->variants.size() < kReplayMaxVariants) {
      // New address set for a stable dispatch: cache it as one more
      // variant (fresh descriptor set + command buffer, recorded once).
      ReplayVariant v;
      v.descriptor_set = acquire_replay_descriptor_set(compute);
      std::array<VkDescriptorBufferInfo, kComputeBindingBudget> info{};
      std::array<VkWriteDescriptorSet, kComputeBindingBudget> writes{};
      for (uint32_t i = 0; i < bindings.size(); ++i) {
        info[i] = {bindings[i].buffer, bindings[i].offset, bindings[i].range};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = v.descriptor_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &info[i];
      }
      dt.UpdateDescriptorSets(
          device_.handle(),
          static_cast<uint32_t>(bindings.size()),
          writes.data(),
          0,
          nullptr);
      trace::counters().vk_descriptor_update_writes += bindings.size();
      v.cmd = acquire_replay_cmd();
      VkCommandBufferBeginInfo bi{
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      VKX_CHECK(dt.BeginCommandBuffer(v.cmd, &bi));
      record_dispatch_locked(
          v.cmd, v.descriptor_set, pipeline, compute.pipeline_layout(),
          params,
          std::min(group_count_x, kMaxComputeGroupCountX),
          std::min(group_count_y, kMaxComputeGroupCountX),
          std::min(group_count_z, kMaxComputeGroupCountX));
      VKX_CHECK(dt.EndCommandBuffer(v.cmd));
      for (uint32_t i = 0; i < bindings.size(); ++i) {
        v.buffers[i] = bindings[i].buffer;
        v.offsets[i] = bindings[i].offset;
        v.ranges[i] = bindings[i].range;
      }
      entry->variants.push_back(v);
      for (const auto& item : bindings) {
        note_binding_owner(item.owner);
      }
      replay_order_.push_back(v.cmd);
      htrace::add(htrace::replay_records, 1);
      replay_cursor_++;
      return true;
    }
    // Variant set full and none matches: fall through to fresh record
    // with a throwaway set (rare; pool chaining absorbs the churn).
  }

  // Fresh record: params differ (rope/KV/attention progress) or the
  // sequence position is new or structurally different.
  ReplayVariant fresh;
  bool reuse_set = false;
  if (entry != nullptr && !entry->variants.empty()) {
    // A params-changing dispatch keeps ONE variant; when the buffers are
    // the same as its recorded set, only the push constants differ, so
    // re-record into a new command buffer against the same set.
    ReplayVariant& last = entry->variants.back();
    reuse_set = entry->variants.size() == 1 &&
        variant_matches(last, bindings);
    if (reuse_set) {
      fresh.descriptor_set = last.descriptor_set;
    }
  }
  if (!reuse_set) {
    fresh.descriptor_set = acquire_replay_descriptor_set(compute);
    std::array<VkDescriptorBufferInfo, kComputeBindingBudget> info{};
    std::array<VkWriteDescriptorSet, kComputeBindingBudget> writes{};
    for (uint32_t i = 0; i < bindings.size(); ++i) {
      info[i] = {bindings[i].buffer, bindings[i].offset, bindings[i].range};
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = fresh.descriptor_set;
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo = &info[i];
    }
    dt.UpdateDescriptorSets(
        device_.handle(),
        static_cast<uint32_t>(bindings.size()),
        writes.data(),
        0,
        nullptr);
    trace::counters().vk_descriptor_update_writes += bindings.size();
  }
  fresh.cmd = acquire_replay_cmd();
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  VKX_CHECK(dt.BeginCommandBuffer(fresh.cmd, &bi));
  record_dispatch_locked(
      fresh.cmd, fresh.descriptor_set, pipeline, compute.pipeline_layout(),
      params,
      std::min(group_count_x, kMaxComputeGroupCountX),
      std::min(group_count_y, kMaxComputeGroupCountX),
      std::min(group_count_z, kMaxComputeGroupCountX));
  VKX_CHECK(dt.EndCommandBuffer(fresh.cmd));

  if (entry == nullptr) {
    replay_entries_.emplace_back();
    entry = &replay_entries_.back();
    entry->pipeline = pipeline;
    entry->params = params;
    entry->gx = group_count_x;
    entry->gy = group_count_y;
    entry->gz = group_count_z;
  } else if (!stable) {
    // Sequence moved on (params changed): drop stale variants, keep the
    // freshly recorded one. The replaced command buffers reset after
    // their drain; replaced sets are orphaned until teardown.
    for (auto& v : entry->variants) {
      if (v.cmd != VK_NULL_HANDLE && v.cmd != fresh.cmd) {
        replay_retiring_.push_back(v.cmd);
      }
    }
    entry->variants.clear();
    entry->pipeline = pipeline;
    entry->params = params;
    entry->gx = group_count_x;
    entry->gy = group_count_y;
    entry->gz = group_count_z;
  } else if (!reuse_set) {
    // stable but variants full: keep variants, fresh is throwaway.
  }
  if (stable || entry->variants.empty()) {
    if (reuse_set) {
      entry->variants.back().cmd = fresh.cmd;
    } else if (entry->variants.size() < kReplayMaxVariants) {
      for (uint32_t i = 0; i < bindings.size(); ++i) {
        fresh.buffers[i] = bindings[i].buffer;
        fresh.offsets[i] = bindings[i].offset;
        fresh.ranges[i] = bindings[i].range;
      }
      entry->variants.push_back(fresh);
    }
  }
  for (const auto& item : bindings) {
    note_binding_owner(item.owner);
  }
  replay_order_.push_back(fresh.cmd);
  htrace::add(htrace::replay_records, 1);
  replay_cursor_++;
  return true;
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
  dispatch_compute_pipeline(
      compute.pipeline(kernel),
      kernel,
      bindings,
      params,
      group_count_x,
      group_count_y,
      group_count_z);
}

void CommandEncoder::dispatch_compute(
    const std::string& cache_key,
    std::span<const uint32_t> spirv,
    std::span<const ComputeBinding> bindings,
    const ComputeParams& params,
    uint32_t group_count_x,
    uint32_t group_count_y,
    uint32_t group_count_z) {
  if (group_count_x == 0 || group_count_y == 0 || group_count_z == 0) {
    return;
  }
  auto& compute = device_.compute();
  dispatch_compute_pipeline(
      compute.pipeline(cache_key, spirv),
      ComputeKernel::Custom,
      bindings,
      params,
      group_count_x,
      group_count_y,
      group_count_z);
}

void CommandEncoder::dispatch_compute_pipeline(
    VkPipeline pipeline,
    ComputeKernel profile_kernel,
    std::span<const ComputeBinding> bindings,
    const ComputeParams& params,
    uint32_t group_count_x,
    uint32_t group_count_y,
    uint32_t group_count_z) {
  htrace::Scoped _disp(htrace::disp_total);
  if (replay_enabled() &&
      replay_dispatch(
          pipeline,
          profile_kernel,
          bindings,
          params,
          group_count_x,
          group_count_y,
          group_count_z)) {
    node_count_++;
    trace::counters().vk_compute_dispatches++;
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
  htrace::Scoped _desc(htrace::desc_setup);
  for (const auto& item : bindings) {
    note_binding_owner(item.owner);
  }
  group_count_x = std::min(group_count_x, kMaxComputeGroupCountX);
  group_count_y = std::min(group_count_y, kMaxComputeGroupCountX);
  group_count_z = std::min(group_count_z, kMaxComputeGroupCountX);

  auto& dt = vk::device_table();
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
  trace::counters().vk_descriptor_update_writes += bindings.size();

  {
    htrace::Scoped _ensure(htrace::ensure_recording_ns);
    ensure_recording();
  }
  uint64_t host_t0 = prof::get().profiling() ? prof::host_ns() : 0;
  // MLX_OMARCHY_TAPE_FULL_BARRIERS (diagnostic, docs/install-omarchy.md):
  // the heaviest correct dependency - all commands, all memory access,
  // both directions - ahead of every dispatch, on top of the regular
  // dependency below. Probes whether the driver drops an in-buffer
  // dependency the regular dependency already expresses. It also
  // restarts the gated tracker: the full barrier orders everything
  // recorded before it.
  if (tape_full_barriers()) {
    record_dependency_barrier();
  }
  // MLX_OMARCHY_GATED_BARRIERS (default off): one full barrier before
  // the dispatch only when a binding overlaps unsynced work of the open
  // batch (RAW/WAW against tracked writes, WAR against tracked reads)
  // or when this is the first node of a fresh command buffer. Off: the
  // historic unconditional pre-dispatch barrier. Bindings carry no
  // read/write split, so each binding is tracked as both read and
  // write - the tracker may barrier a read-read pair, never skip a
  // real hazard.
  htrace::Scoped _bar(htrace::barriers);
  bool barrier_recorded = false;
  if (gated_barriers()) {
    std::array<TrackedRange, kComputeBindingBudget> ranges{};
    for (size_t i = 0; i < bindings.size(); ++i) {
      ranges[i] = {bindings[i].buffer,
                   bindings[i].offset,
                   tracked_range_end(bindings[i].offset, bindings[i].range)};
    }
    std::span<const TrackedRange> view{ranges.data(), bindings.size()};
    if (!head_synced_ || batch_needs_barrier(view, view)) {
      record_dependency_barrier();
      trace::counters().barriers_emitted++;
      prof::get().on_barrier(true);
      barrier_recorded = true;
    } else {
      trace::counters().barriers_skipped++;
      prof::get().on_barrier(false);
    }
    for (size_t i = 0; i < bindings.size(); ++i) {
      tracked_reads_.push_back(ranges[i]);
      tracked_writes_.push_back(ranges[i]);
    }
  } else {
    VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    before.srcAccessMask =
        VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
        VK_ACCESS_SHADER_WRITE_BIT;
    before.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
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
    head_synced_ = true;
    trace::counters().barriers_emitted++;
    prof::get().on_barrier(true);
    barrier_recorded = true;
  }
  {
    htrace::Scoped _vkcmd(htrace::vkcmd);
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
  }
  prof::get().after_dispatch(
      this,
      current_slot_,
      cmd_,
      profile_kernel,
      params,
      bindings,
      group_count_x,
      group_count_y,
      group_count_z,
      host_t0 != 0 ? prof::host_ns() - host_t0 : 0,
      in_tape_recording ? 1u : 0u);

  // Gated mode tracks this dispatch's writes instead of recording a
  // post barrier; the next node's overlap test consumes the tracking.
  {
    htrace::Scoped _bar2(htrace::barriers);
    if (!gated_barriers()) {
    VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    after.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT |
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
    trace::counters().barriers_emitted++;
    prof::get().on_barrier(true);
    }
  }
  if (tape_full_barriers()) {
    // Diagnostic: matching full barrier out of this dispatch, so every
    // dependency between two dispatches is the heaviest form (see the
    // pre-dispatch barrier above).
    record_dependency_barrier();
  }

  node_count_++;
  trace::counters().vk_compute_dispatches++;
}

void CommandEncoder::commit() {
  if (!recording_ && replay_order_.empty() &&
      wait_semaphores_.empty() && signal_semaphores_.empty() &&
      completed_handlers_.empty()) {
    trace::counters().commit_calls_noop++;
    return;
  }
  trace::counters().commit_calls_with_work++;
  submit();
}

void CommandEncoder::synchronize(const char* reason) {
  htrace::Scoped _sync(htrace::sync_total);
  commit();
  join_last_completion(reason);
}

void CommandEncoder::submit() {
  htrace::Scoped _submit(htrace::submit_total);
  auto& dt = vk::device_table();
  bool was_recording = recording_;
  uint64_t submit_t0 = prof::get().profiling() ? prof::host_ns() : 0;
  uint64_t close_t = 0;
  uint64_t queue_t0 = 0;
  uint64_t queue_t1 = 0;
  uint64_t submitted = 0;

  if (recording_) {
    if (gated_barriers()) {
      VkMemoryBarrier readback{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      readback.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
      readback.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
      dt.CmdPipelineBarrier(
          cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
          VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &readback,
          0, nullptr, 0, nullptr);
      trace::counters().barriers_emitted++;
      prof::get().on_barrier(true);
    }
    VKX_CHECK(dt.EndCommandBuffer(cmd_));
    close_t = prof::get().profiling() ? prof::host_ns() : 0;
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
  const bool replay_submit = !replay_order_.empty();
  si.commandBufferCount = recording_
      ? 1u
      : static_cast<uint32_t>(replay_order_.size());
  si.pCommandBuffers = recording_
      ? &cmd_
      : (replay_submit ? replay_order_.data() : nullptr);

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

  // Ownership moves to the dispatcher entry: each pending-semaphore
  // keepalive keeps its Event alive, and handlers run on the completion
  // thread. The batch's buffers need no shared_ptr: they were stamped
  // with the completion value below, and the allocator quarantine keeps
  // any freed-but-in-flight buffer out of the reuse cache until its
  // generation drains.
  std::vector<std::shared_ptr<void>> keepalive;
  for (auto& pool : retired_pools_) {
    keepalive.push_back(std::move(pool));
  }
  retired_pools_.clear();
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
    // Stamp every buffer referenced by this batch with the completion
    // value just reserved. Free-before-drain sends such buffers to the
    // allocator quarantine; release_quarantine recycles them one
    // generation later. Semaphore keepalives still move into the
    // dispatcher payload below.
    omarchy::allocator().stamp_batch(batch_buffers_, completion_value);
    batch_buffers_.clear();
    VkSemaphore completion_sem = device_.completions().semaphore();
    signal_sems.push_back(completion_sem);
    signal_values.push_back(completion_value);
    timeline.signalSemaphoreValueCount =
        static_cast<uint32_t>(signal_values.size());
    timeline.pSignalSemaphoreValues = signal_values.data();
    si.signalSemaphoreCount = static_cast<uint32_t>(signal_sems.size());
    si.pSignalSemaphores = signal_sems.data();
    try {
      queue_t0 = prof::get().profiling() ? prof::host_ns() : 0;
      VKX_CHECK(dt.QueueSubmit(device_.queue(), 1, &si, VK_NULL_HANDLE));
      queue_t1 = prof::get().profiling() ? prof::host_ns() : 0;
    } catch (...) {
      // The submission never reached the driver: the ended command buffer
      // and the pending semaphore lists are dead (their keepalives have
      // already moved into the local payload and die with this frame).
      // Reset the encoder so it can be reused or destroyed cleanly; the
      // typed error propagates to the stream's error handling. The
      // batch's buffer stamps are harmless: those buffers stay alive via
      // their arrays or the quarantine.
      batch_buffers_.clear();
      recording_ = false;
      node_count_ = 0;
      wait_semaphores_.clear();
      signal_semaphores_.clear();
      completed_handlers_.clear();
      reset_dependency_tracking();
      throw;
    }
    // Publish only after the submit: the dispatcher must never wait on a
    // value whose submission has not been handed to the driver.
    device_.completions().enqueue(
        completion_value,
        std::move(keepalive),
        std::move(completed_handlers_),
        was_recording ? slots_[current_slot_].started : VK_NULL_HANDLE);
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
  // Replay batch bookkeeping: the cursor restarts for the next batch and
  // the ordered array is consumed. Replaced command buffers wait for
  // their drain in replay_retiring_ (join_last_completion recycles them);
  // submitted-and-cached ones stay executable for the next batch.
  replay_cursor_ = 0;
  replay_order_.clear();
  // The submission's in-order wait (last_completion_) provides the
  // cross-submission dependency, so the open batch's unsynced ranges
  // die here either way.
  reset_dependency_tracking();
  prof::get().on_submit_boundary(submitted, close_t, queue_t0, queue_t1);
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

extern "C" __attribute__((visibility("default"))) void
mlx_omarchy_host_trace_reset(void) {
  mlx::core::omarchy::htrace::reset();
}

extern "C" __attribute__((visibility("default"))) int
mlx_omarchy_host_trace_dump(const char* path) {
  return mlx::core::omarchy::htrace::dump(path) ? 0 : 1;
}
