// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

// Host-path phase timers for the decode host-overhead measurement wheel
// (branch wave/HostPathOverhead, NEVER MERGED). Runtime-gated by
// MLX_OMARCHY_HOST_TRACE; unset, the hot path pays one predictable branch
// on a cached bool and no clocks. There are no GPU timestamps anywhere in
// this instrument: CLOCK_MONOTONIC (steady_clock) only, so it measures
// exactly the host work - graph walk, primitive eval, descriptor/binding
// setup, command recording, submission, completion handling, allocator -
// without perturbing device-side cadence the way per-dispatch timestamp
// brackets do (see receipts/2026-09-10-dispatch-floor).
//
// Phases are exclusive where practical (desc_setup / barriers / vkcmd
// partition dispatch_compute_pipeline; backend_eval / eval_bookkeeping
// partition the work part of gpu::eval). join_wait is GPU-paced and
// excluded from host-busy sums by the analyzer; completion_work runs on
// the device completion thread.
//
// C ABI for the harness:
//   mlx_omarchy_host_trace_reset()  zero counters (call at decode start)
//   mlx_omarchy_host_trace_dump(path)  write JSON summary and zero

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace mlx::core::omarchy::htrace {

enum Phase : int {
  eval_node_total = 0,  // gpu::eval() entry..exit per scheduler node
  backend_eval,         // eager fusion attempt + primitive eval_gpu
  eval_bookkeeping,     // temporaries pinning + budget check in gpu::eval
  disp_total,           // dispatch_compute_pipeline total
  desc_setup,           // binding owner notes + descriptor acquire + update
  barriers,             // pre/post CmdPipelineBarrier recording
  vkcmd,                // bind pipeline/sets, push constants, dispatch record
  ensure_recording_ns,  // ring slot pick + BeginCommandBuffer (join risk)
  submit_total,         // CommandEncoder::submit total (close+submit+enqueue)
  join_wait,            // completion-timeline wait in join_last_completion
  completion_work,      // drain_through work: invalidate+handlers+quarantine
  alloc_ns,             // VulkanAllocator::malloc
  free_ns,              // VulkanAllocator::free
  finalize_ns,          // gpu::finalize
  sync_total,           // CommandEncoder::synchronize total
  replay_hits,          // count of dispatches replayed from cache (ns unused)
  replay_records,       // count of dispatches freshly recorded under replay
  PHASE_COUNT
};

struct Counters {
  std::atomic<uint64_t> ns[PHASE_COUNT];
  std::atomic<uint64_t> hits[PHASE_COUNT];
};

inline Counters& counters() {
  static Counters c;
  return c;
}

inline bool enabled() {
  static const bool on = std::getenv("MLX_OMARCHY_HOST_TRACE") != nullptr;
  return on;
}

inline uint64_t now_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

inline void add(Phase p, uint64_t dt) {
  counters().ns[p].fetch_add(dt, std::memory_order_relaxed);
  counters().hits[p].fetch_add(1, std::memory_order_relaxed);
}

// Scoped timer: enabled only, zero-cost construction otherwise.
struct Scoped {
  uint64_t t0;
  Phase p;
  inline explicit Scoped(Phase phase)
      : t0(enabled() ? now_ns() : 0), p(phase) {}
  inline ~Scoped() {
    if (t0 != 0) {
      add(p, now_ns() - t0);
    }
  }
};

inline const char* phase_name(int p) {
  static const char* names[PHASE_COUNT] = {
      "eval_node_total", "backend_eval", "eval_bookkeeping", "disp_total",
      "desc_setup",      "barriers",     "vkcmd",            "ensure_recording",
      "submit_total",    "join_wait",    "completion_work",  "alloc",
      "free",            "finalize",     "synchronize",      "replay_hits",
      "replay_records"};
  return names[p];
}

inline void reset() {
  auto& c = counters();
  for (int i = 0; i < PHASE_COUNT; ++i) {
    c.ns[i].store(0, std::memory_order_relaxed);
    c.hits[i].store(0, std::memory_order_relaxed);
  }
}

inline bool dump(const char* path) {
  auto& c = counters();
  std::FILE* f = std::fopen(path, "w");
  if (f == nullptr) {
    return false;
  }
  std::fprintf(f, "{\"schema\":\"mlx-omarchy/host-trace/1\"");
  for (int i = 0; i < PHASE_COUNT; ++i) {
    std::fprintf(f, ",\"%s\":{\"ns\":%llu,\"hits\":%llu}",
                 phase_name(i),
                 static_cast<unsigned long long>(
                     c.ns[i].load(std::memory_order_relaxed)),
                 static_cast<unsigned long long>(
                     c.hits[i].load(std::memory_order_relaxed)));
  }
  std::fprintf(f, "}\n");
  std::fclose(f);
  return true;
}

}  // namespace mlx::core::omarchy::htrace

// The C ABI entry points live in encoder.cpp (non-inline, so the symbols
// are actually emitted for ctypes/dlsym instead of being discarded as
// unreferenced inline functions).
