// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>
#include "mlx/array.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/stream.h"

namespace mlx::core::omarchy {

// Per-stream command recorder over the device's single VkQueue. Recording
// is BATCHED: primitive evals append to an open command buffer and the
// buffer is submitted when a node/work budget is reached, a flush is
// demanded (semaphore operation, host read), or the in-flight ring is
// exhausted. Batching is order-safe because every dispatch records a full
// pre+post pipeline barrier, every submission waits on the completion-
// timeline value of this stream's previous submission (Vulkan defines no
// cross-submission dependency without a wait), and cross-submission waits
// use ALL_COMMANDS stage masks, so merging submissions weakens no
// dependency; it removes the per-eval host join between them. Temporaries
// and completion handlers released per submission still release exactly
// when that submission's GPU work finishes, via the device completion
// timeline.
//
// Command buffers come from a small ring so the device can execute one
// batch while the host records the next; the host blocks only when every
// ring slot is in flight (then it joins just the oldest) or when a host
// read demands it (synchronize()).
//
// Every submission signals a strictly increasing value on the device
// completion timeline; the device's CompletionDispatcher runs this
// submission's handlers and releases its temporaries and queued-semaphore
// ownership when the GPU work finishes. Cross-stream order comes from
// timeline semaphore waits and signals carried in the submissions;
// handler-only submissions (no recorded commands) still signal the
// timeline, so ordering with prior queue work is preserved.
// Nodes recorded in one open command buffer before the evaluator flushes
// it. Bounds how many temporaries and how much work one batch pins; the
// historical per-op commit was the 1-node extreme.
inline constexpr int kBatchNodeBudget = 100;

class MLX_API CommandEncoder {
 public:
  explicit CommandEncoder(Device& device);
  ~CommandEncoder();

  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;

  // Keep the array's buffer alive until the committed work completes.
  void add_temporary(const array& arr) {
    temporaries_.push_back(arr.data_shared_ptr());
  }

  // Handlers run on the device completion thread when this submission's
  // GPU work finishes.
  void add_completed_handler(std::function<void()> task) {
    completed_handlers_.push_back(std::move(task));
  }

  bool needs_commit() const {
    return node_count_ > 0;
  }

  // Fragmentation prototype (fragmentation-hunt): the evaluator opens ONE
  // scheduler task per graph evaluation, at the eval's first work-carrying
  // primitive, and closes it at this encoder's FIRST commit of any kind
  // (node budget, event flush contract, memory-guard finalize, or the
  // eval-boundary signal). A task whose decrement is still attached when
  // the encoder goes idle with no work host-completes it, so a task never
  // waits on itself.
  bool scheduler_task_open() const {
    return scheduler_task_open_;
  }
  void set_scheduler_task_open(bool v, Stream s) {
    scheduler_task_open_ = v;
    scheduler_task_stream_ = s;
  }

  Stream scheduler_task_stream() const {
    return *scheduler_task_stream_;
  }

  // Nodes recorded in the open batch. The evaluator flushes the batch at
  // kBatchNodeBudget so a long graph cannot pin unbounded temporaries
  // behind one open command buffer.
  int nodes() const {
    return node_count_;
  }

  // True when nothing is recorded, nothing is queued for submission, and
  // no temporaries are pending flush: the encoder owns no GPU work and
  // no lifetime obligations. An Event::signal issued on this state moves
  // the target timeline counter from the host instead of submitting a
  // signal-only command buffer; the temporaries check is what makes the
  // flush contract explicit — a signal on an encoder that still owes a
  // temporaries flush takes the queued path, which submits and releases.
  bool idle() const {
    return !recording_ && wait_semaphores_.empty() &&
        signal_semaphores_.empty() && completed_handlers_.empty() &&
        temporaries_.empty();
  }

  // Record a device-to-device buffer copy. Both buffers must have
  // VK_BUFFER_USAGE_TRANSFER_* usage bits (all backend buffers do).
  void copy_buffer(
      VkBuffer src,
      VkBuffer dst,
      VkDeviceSize size,
      VkDeviceSize src_offset = 0,
      VkDeviceSize dst_offset = 0);

  // Record one compute dispatch. The binding count must not exceed the
  // device's runtime budget (ComputeRuntime::binding_limit()); kernels
  // needing more refuse by name at the primitive that builds them.
  void dispatch_compute(
      ComputeKernel kernel,
      std::span<const ComputeBinding> bindings,
      const ComputeParams& params,
      uint32_t group_count_x,
      uint32_t group_count_y = 1,
      uint32_t group_count_z = 1);

  // Record a four-byte-word fill. Size and offset must be multiples of 4.
  void fill_buffer(
      VkBuffer dst,
      uint32_t value,
      VkDeviceSize size,
      VkDeviceSize offset = 0);

  // Timeline semaphore operations for Event support. Waits apply to this
  // stream's next submission; signals are flushed by commit(). The keepalive
  // token (the owning Event's shared state) must stay held until the
  // submission completes: a queued wait or signal never stores a bare
  // VkSemaphore whose owner could be destroyed first.
  void add_semaphore_wait(
      VkSemaphore semaphore,
      uint64_t value,
      std::shared_ptr<void> keepalive) {
    wait_semaphores_.push_back({semaphore, value, std::move(keepalive)});
  }

  void add_semaphore_signal(
      VkSemaphore semaphore,
      uint64_t value,
      std::shared_ptr<void> keepalive) {
    signal_semaphores_.push_back({semaphore, value, std::move(keepalive)});
  }

  // Submit recorded work, semaphore operations, and completion handlers.
  // Eager: every finalize flushes, one op-granular submission. Deeper
  // batching was measured (2026-09-03) and made generation 4.5x SLOWER:
  // the evaluator throttles at MAX_ACTIVE_TASKS with finalize plus
  // wait_for_one, so any flush granularity coarser than one op converts
  // its pipelined wait into a stop-and-wait proportional to batch
  // execution time — worse with deeper batches and longer models, never
  // better. A future attempt must flush when the scheduler would block,
  // not on a node budget (upstream-shape work). Safe to call repeatedly;
  // a no-op when nothing is pending.
  void commit();

  // Submit pending work and block (bounded) until it completes.
  void synchronize();

  // True when every submission queued through this encoder has completed
  // and its handlers have run, and no batch is open with un-submitted
  // work. Host reads of input bytes are then sound.
  bool synchronized() const {
    return node_count_ == 0 &&
        (last_completion_ == 0 ||
         device_.completions().drained_value() >= last_completion_);
  }

  // Completion timeline value of the newest submission this encoder has
  // on the queue (0 when none). Event signal paths capture it so waiters
  // can join the handler boundary of the generation that signaled them.
  uint64_t last_submitted_completion() const {
    return last_completion_;
  }

  Device& device() {
    return device_;
  }

 private:
  // A semaphore operation pending in this encoder, plus the ownership token
  // that keeps the semaphore's owner alive until GPU completion.
  struct PendingSemaphore {
    VkSemaphore semaphore;
    uint64_t value;
    std::shared_ptr<void> keepalive;
  };

  // One command buffer of the in-flight ring. in_flight is the completion
  // timeline value of the submission currently executing it (0 = free).
  static constexpr int kInFlightCommandBuffers = 4;
  struct Slot {
    VkCommandBuffer cmd{VK_NULL_HANDLE};
    uint64_t in_flight{0};
  };

  // Descriptor-set pool cache: sets are allocated from a large pool that
  // is created once per kDescriptorSetsPerPool dispatches instead of a
  // pool create/destroy per dispatch. A retired pool is kept alive until
  // the submission recording at retirement time completes, which is after
  // every submission whose batches allocated from it (timeline order).
  static constexpr uint32_t kDescriptorSetsPerPool = 2048;
  VkDescriptorSet acquire_descriptor_set(ComputeRuntime& compute);

  // Pick a completed ring slot (joining the oldest only when all are in
  // flight) and begin recording into it.
  void ensure_recording();

  // Join this encoder's newest in-flight submission (if any) through the
  // CompletionDispatcher, clear last_completion_, and invalidate
  // noncoherent host mappings. Guarantees the newest command buffer has
  // left the pending state.
  void join_last_completion();
  void submit();

  Device& device_;
  VkCommandPool pool_{VK_NULL_HANDLE};
  std::array<Slot, kInFlightCommandBuffers> slots_{};
  int current_slot_{0};
  VkCommandBuffer cmd_{VK_NULL_HANDLE};
  bool recording_{false};
  bool scheduler_task_open_{false};
  std::optional<Stream> scheduler_task_stream_;
  int node_count_{0};
  uint64_t last_completion_{0};
  VkDescriptorPool desc_pool_{VK_NULL_HANDLE};
  uint32_t desc_pool_remaining_{0};
  std::vector<PendingSemaphore> wait_semaphores_;
  std::vector<PendingSemaphore> signal_semaphores_;
  std::vector<std::shared_ptr<void>> temporaries_;
  std::vector<std::function<void()>> completed_handlers_;
};

MLX_API CommandEncoder& get_command_encoder(Stream s);
std::unordered_map<int, CommandEncoder>& get_command_encoders();
std::unordered_map<int, CommandEncoder>& get_global_command_encoders();

} // namespace mlx::core::omarchy
