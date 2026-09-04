// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Scheduler-task accounting for the fragmentation prototype:
//
//   1. ONE task opens per graph evaluation (at the eval's first
//      work-carrying primitive) and its decrement rides the encoder's
//      first commit of any kind, so a single small eval submits exactly
//      one batch - not one task and one submission per batch.
//   2. The transforms memory guard can call gpu::finalize mid-walk. Under
//      the per-eval task this must not hang: the guard's finalize is a
//      commit, the commit carries the decrement, wait_for_one returns.
//      A tiny memory limit makes that path deterministic on llvmpipe.
//   3. A workless eval (view-only) opens no task and submits nothing.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cstring>
#include <span>

#include "mlx/array.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/ops.h"
#include "mlx/scheduler.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace {

// The C-ABI snapshot from trace.h; resolved through the linked library
// directly (the test links mlx statically, but the ABI works either way).
mlx::core::omarchy::trace::MlxOmarchyTraceSnapshot snapshot() {
  mlx::core::omarchy::trace::MlxOmarchyTraceSnapshot s;
  std::memset(&s, 0, sizeof(s));
  extern "C" void mlx_omarchy_trace_snapshot(
      mlx::core::omarchy::trace::MlxOmarchyTraceSnapshot*);
  mlx_omarchy_trace_snapshot(&s);
  return s;
}

} // namespace

using namespace mlx::core;

TEST_CASE("one small eval submits one batch under per-eval task accounting") {
  auto stream = default_stream(default_device());
  synchronize(stream);
  auto before = snapshot();

  auto x = full({32, 32}, 1.0f, float32);
  auto y = x + 1.0f;
  eval(y);
  synchronize(stream);

  auto after = snapshot();
  CHECK_EQ(after.gpu_primitive_dispatches - before.gpu_primitive_dispatches, 3);
  CHECK_EQ(after.vk_submissions - before.vk_submissions, 1);
  CHECK_EQ(after.commit_calls_with_work - before.commit_calls_with_work, 1);
  CHECK_EQ(after.omarchy_finalize_calls - before.omarchy_finalize_calls, 1);
  CHECK_EQ(y.data<float>()[0], 2.0f);
}

TEST_CASE("memory-guard finalize mid-walk does not hang the task accounting") {
  auto stream = default_stream(default_device());
  synchronize(stream);
  auto previous_limit = set_memory_limit(24 * 1024 * 1024);

  // ~112 MiB of live GPU arrays against a 24 MiB limit: the transforms
  // memory guard trips partway through the walk and calls gpu::finalize
  // plus wait_for_one while this eval's task is still open.
  auto x = full({4, 1024, 1024}, 1.0f, float32);
  auto y = x + 1.0f;
  auto z = y + 1.0f;
  auto w = z + 1.0f;
  auto out = eval(w);
  synchronize(stream);

  set_memory_limit(previous_limit);

  CHECK_EQ(out.data<float>()[0], 4.0f);
  CHECK_EQ(out.data<float>()[1023], 4.0f);
}

TEST_CASE("a workless view eval submits nothing and opens no task") {
  auto stream = default_stream(default_device());
  synchronize(stream);
  auto before = snapshot();

  auto x = full({4, 8}, 1.0f, float32);
  auto v = reshape(x, {2, 4, 4});
  eval(v);
  synchronize(stream);

  auto after = snapshot();
  CHECK_EQ(after.gpu_primitive_dispatches - before.gpu_primitive_dispatches, 2);
  CHECK_EQ(after.vk_submissions - before.vk_submissions, 0);
}
