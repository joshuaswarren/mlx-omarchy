// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstdint>

// Backend dispatch trace counters (plan R8). mlx-omarchy-info and the runtime
// tests read these to prove which backend executed work.
namespace mlx::core::omarchy::trace {

struct Counters {
  // Number of tensor primitives dispatched to the Omarchy GPU evaluator.
  std::atomic<uint64_t> gpu_primitive_dispatches{0};
  // Number of vkQueueSubmit calls made by the backend.
  std::atomic<uint64_t> vk_submissions{0};
  // Number of recorded vkCmdCopyBuffer commands.
  std::atomic<uint64_t> vk_buffer_copies{0};
  // Number of recorded vkCmdFillBuffer commands.
  std::atomic<uint64_t> vk_buffer_fills{0};
  // Number of recorded Vulkan compute dispatches.
  std::atomic<uint64_t> vk_compute_dispatches{0};
  // Number of Vulkan compute dispatches recorded from inside
  // eval_compiled_tape, i.e. issued on behalf of a tape node rather
  // than an eager primitive. Attributes per-token dispatch counts
  // between the tape and eager paths.
  std::atomic<uint64_t> compiled_tape_dispatches{0};
  // Number of Compiled-tape nodes the interpreter evaluated, whether
  // they recorded a dispatch or not.
  std::atomic<uint64_t> compiled_tape_node_evaluations{0};
};

inline Counters& counters() {
  static Counters counters_;
  return counters_;
}

} // namespace mlx::core::omarchy::trace
