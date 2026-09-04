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
  // Number of gpu::finalize calls (throttle points and graph ends).
  std::atomic<uint64_t> omarchy_finalize_calls{0};
  // Commits that submitted a real batch (work, semaphores, or handlers).
  std::atomic<uint64_t> commit_calls_with_work{0};
  // Commits that found nothing pending (finalize on an idle encoder).
  std::atomic<uint64_t> commit_calls_noop{0};
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
  uint64_t omarchy_finalize_calls;
  uint64_t commit_calls_with_work;
  uint64_t commit_calls_noop;
};
// Defined once in eval.cpp: an inline header definition is comdat and
// never reaches the dynamic symbol table under -fvisibility=hidden.
extern "C" __attribute__((visibility("default"))) void
mlx_omarchy_trace_snapshot(MlxOmarchyTraceSnapshot* out);

} // namespace mlx::core::omarchy::trace
