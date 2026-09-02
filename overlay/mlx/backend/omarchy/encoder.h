// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <vulkan/vulkan.h>

#include <functional>
#include <memory>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>
#include "mlx/array.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/stream.h"

namespace mlx::core::omarchy {

// Per-stream command recorder over the device's single VkQueue. Commits
// are asynchronous: a commit ends the command buffer, submits it under the
// queue's external-synchronization lock, and returns without waiting. Every
// submission signals a strictly increasing value on the device completion
// timeline; the device's CompletionDispatcher runs this submission's
// handlers and releases its temporaries and queued-semaphore ownership when
// the GPU work finishes. Cross-stream order comes from timeline semaphore
// waits and signals carried in the submissions; handler-only submissions
// (no recorded commands) still signal the timeline, so ordering with prior
// queue work is preserved.
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
  // Asynchronous: returns when the work is queued, not when it completes.
  // Safe to call repeatedly; a no-op when nothing is pending.
  void commit();

  // Submit pending work and block (bounded) until it completes.
  void synchronize();

  // True when every submission queued through this encoder has completed
  // and its handlers have run. Host reads of input bytes are then sound.
  bool synchronized() const {
    return last_completion_ == 0 ||
        device_.completions().drained_value() >= last_completion_;
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

  // Join this encoder's newest in-flight submission (if any) through the
  // CompletionDispatcher, clear last_completion_, and invalidate
  // noncoherent host mappings. Guarantees cmd_ has left the pending
  // state before the next BeginCommandBuffer.
  void join_last_completion();
  void ensure_recording();
  void submit();

  Device& device_;
  VkCommandPool pool_{VK_NULL_HANDLE};
  VkCommandBuffer cmd_{VK_NULL_HANDLE};
  bool recording_{false};
  int node_count_{0};
  uint64_t last_completion_{0};
  std::vector<PendingSemaphore> wait_semaphores_;
  std::vector<PendingSemaphore> signal_semaphores_;
  std::vector<std::shared_ptr<void>> temporaries_;
  std::vector<std::function<void()>> completed_handlers_;
};

MLX_API CommandEncoder& get_command_encoder(Stream s);
std::unordered_map<int, CommandEncoder>& get_command_encoders();
std::unordered_map<int, CommandEncoder>& get_global_command_encoders();

} // namespace mlx::core::omarchy
