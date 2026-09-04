// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// GPU evaluation entry points (mlx/backend/gpu/eval.h). Mirrors the CUDA
// backend's flow: run the primitive's eval_gpu, register input buffers as
// encoder temporaries, and commit when work was recorded.

#include "mlx/backend/gpu/eval.h"

#include <algorithm>
#include <string_view>
#include <vector>

#ifdef MLX_OMARCHY_GPU_PROFILING
#include <cstdio>
#endif

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

namespace mlx::core::gpu {

namespace {

// True when walking the shape in C order visits consecutive items, so a
// mapped host read from base + offset sees the logical values.
bool is_c_contiguous(const Shape& shape, const Strides& strides) {
  int64_t expected = 1;
  for (int axis = static_cast<int>(shape.size()) - 1; axis >= 0; --axis) {
    if (shape[axis] != 1 && strides[axis] != expected) {
      return false;
    }
    expected *= shape[axis];
  }
  return true;
}

// A Slice result shares its input's storage. Evaluated arrays are read as
// flat host buffers (base plus byte offset), so a slice view with gaps
// between rows or elements is unreadable. Materialize it through the
// general strided-copy engine; offset-only slices stay shared views.
void materialize_strided_slice(array& out) {
  if (out.size() <= 1 || is_c_contiguous(out.shape(), out.strides())) {
    return;
  }
  std::vector<array> inputs = out.inputs();
  // The view's byte offset lives on the slice output itself; it becomes
  // the source item offset of the gather.
  int64_t in_item_offset = out.offset() / out.itemsize();
  Shape shape = out.shape();
  Strides in_strides = out.strides();
  // The plain set_data overload keeps the existing strides, so the
  // contiguous destination layout is written explicitly.
  Strides out_strides(shape.size(), 1);
  for (int axis = static_cast<int>(shape.size()) - 2; axis >= 0; --axis) {
    out_strides[axis] = out_strides[axis + 1] * shape[axis + 1];
  }
  array::Flags flags;
  flags.contiguous = true;
  flags.row_contiguous = true;
  auto max_dim = std::max_element(shape.begin(), shape.end());
  flags.col_contiguous = out.size() <= 1 || out.size() == *max_dim;
  Stream s = out.primitive().stream();
  out.set_data(
      omarchy::allocator().malloc(out.nbytes()),
      out.size(),
      out_strides,
      flags,
      0);
  copy_gpu_inplace(
      inputs.at(0),
      out,
      shape,
      in_strides,
      out.strides(),
      /*i_offset=*/in_item_offset,
      /*o_offset=*/0,
      CopyType::General,
      s);
}

} // namespace

void init() {
  // Discovery errors surface later through omarchy::init_error() when a caller
  // actually asks for the GPU device; initialization must not throw here.
  omarchy::init();
}

void eval(array& arr) {
  omarchy::trace::counters().gpu_primitive_dispatches++;
#ifdef MLX_OMARCHY_GPU_PROFILING
  ++omarchy::trace::prim_counts()[arr.primitive().name()];
#endif
  auto outputs = arr.outputs();
  auto& stream = arr.primitive().stream();
  auto& encoder = omarchy::get_command_encoder(stream);
  // Open-batch state BEFORE this op records: a batch spans every op
  // recorded between commits.
  bool batch_open = encoder.needs_commit();
  {
    // If the array is a tracer hold a reference
    // to its inputs so they don't get donated
    std::vector<array> inputs;
    if (arr.is_tracer()) {
      inputs = arr.inputs();
    }
    arr.primitive().eval_gpu(arr.inputs(), outputs);
  }
  if (arr.primitive().name() == std::string_view("Slice")) {
    for (auto& out : outputs) {
      materialize_strided_slice(out);
    }
  }

  // Temporaries flush contract. A buffer must be pinned against
  // incomplete GPU work if and only if some recorded dispatch references
  // it. The eval that records work pins its inputs, outputs, and
  // siblings here; the batch carries those pins to its submission, and
  // the dispatcher releases them one completion after that submission.
  // An eval whose primitive records nothing (host-materialized scalars,
  // view rearrangements) pins nothing: no in-flight dispatch can
  // reference its buffers, and every consumer that reads them records
  // its own eval, which pins them as inputs. Without this rule the
  // temporaries of workless evals would accumulate until an unrelated
  // submission flushed them.
  if (encoder.needs_commit()) {
    if (!batch_open) {
      // One scheduler task and one completion notification per batch,
      // attached when the batch opens so that every close path
      // (finalize, event flush contract, node budget, host read sync)
      // carries the pairing exactly once.
      scheduler::notify_new_task(stream);
      encoder.add_completed_handler(
          [stream]() { scheduler::notify_task_completion(stream); });
    }
    // Keep used buffers alive until the submitted work completes. The
    // output is retained too (covers donated storage, where the output
    // reuses an input's buffer): every backing buffer of the submitted
    // commands must outlive the caller's references.
    for (auto& in : arr.inputs()) {
      encoder.add_temporary(in);
    }
    for (auto& s : arr.siblings()) {
      encoder.add_temporary(s);
    }
    encoder.add_temporary(arr);
    // Node budget: flush the batch so a long graph cannot pin unbounded
    // temporaries behind one open command buffer.
    if (encoder.nodes() >= omarchy::kBatchNodeBudget) {
      encoder.commit();
    }
  }
}

void finalize(Stream s) {
  omarchy::trace::counters().omarchy_finalize_calls++;
  // Flush contract: the evaluator calls finalize at task-throttle points
  // and at graph end, and then waits on task-completion handlers. The
  // open batch must reach the queue here or those waits never complete.
  // Batching still happens: every dispatch recorded between finalizes
  // (one whole graph evaluation) shares one open command buffer.
  omarchy::get_command_encoder(s).commit();
}


void synchronize(Stream s) {
  omarchy::get_command_encoder(s).synchronize();
}

extern "C" __attribute__((visibility("default"))) void
mlx_omarchy_trace_snapshot(
    mlx::core::omarchy::trace::MlxOmarchyTraceSnapshot* out) {
  auto& c = mlx::core::omarchy::trace::counters();
  out->gpu_primitive_dispatches = c.gpu_primitive_dispatches.load();
  out->vk_submissions = c.vk_submissions.load();
  out->vk_buffer_copies = c.vk_buffer_copies.load();
  out->vk_buffer_fills = c.vk_buffer_fills.load();
  out->vk_compute_dispatches = c.vk_compute_dispatches.load();
  out->omarchy_finalize_calls = c.omarchy_finalize_calls.load();
  out->commit_calls_with_work = c.commit_calls_with_work.load();
  out->commit_calls_noop = c.commit_calls_noop.load();
}

#ifdef MLX_OMARCHY_GPU_PROFILING
extern "C" __attribute__((visibility("default"))) void
mlx_omarchy_prim_dump(const char* path) {
  std::FILE* f = std::fopen(path, "w");
  if (!f) {
    return;
  }
  for (auto& [name, count] : mlx::core::omarchy::trace::prim_counts()) {
    std::fprintf(
        f, "%.*s,%llu\n", static_cast<int>(name.size()), name.data(),
        static_cast<unsigned long long>(count));
  }
  std::fclose(f);
}

extern "C" __attribute__((visibility("default"))) void
mlx_omarchy_prim_reset(void) {
  mlx::core::omarchy::trace::prim_counts().clear();
}
#endif

} // namespace mlx::core::gpu
