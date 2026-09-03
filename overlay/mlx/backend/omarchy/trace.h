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
};

inline Counters& counters() {
  static Counters counters_;
  return counters_;
}

// Process-wide snapshot for in-process readers (ctypes from Python, test
// harnesses). Plain C ABI because the C++ symbol for counters() is inlined
// away and never reaches the dynamic symbol table. libmlx.so is already
// loaded by the Python extension, so ctypes.CDLL on the same path resolves
// to the same static instance.
struct MlxOmarchyTraceSnapshot {
  uint64_t gpu_primitive_dispatches;
  uint64_t vk_submissions;
  uint64_t vk_buffer_copies;
  uint64_t vk_buffer_fills;
  uint64_t vk_compute_dispatches;
};

extern "C" __attribute__((visibility("default"))) void
mlx_omarchy_trace_snapshot(MlxOmarchyTraceSnapshot* out) {
  auto& c = mlx::core::omarchy::trace::counters();
  out->gpu_primitive_dispatches = c.gpu_primitive_dispatches.load();
  out->vk_submissions = c.vk_submissions.load();
  out->vk_buffer_copies = c.vk_buffer_copies.load();
  out->vk_buffer_fills = c.vk_buffer_fills.load();
  out->vk_compute_dispatches = c.vk_compute_dispatches.load();
}

} // namespace mlx::core::omarchy::trace
