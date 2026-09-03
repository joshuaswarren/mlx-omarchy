// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// GPU evaluation entry points (mlx/backend/gpu/eval.h). Mirrors the CUDA
// backend's flow: run the primitive's eval_gpu, register input buffers as
// encoder temporaries, and commit when work was recorded.

#include "mlx/backend/gpu/eval.h"

#include <algorithm>
#include <string_view>
#include <vector>

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
  auto outputs = arr.outputs();
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

  auto& stream = arr.primitive().stream();
  auto& encoder = omarchy::get_command_encoder(stream);
  // Keep used buffers alive until the submitted work completes. The output
  // is retained too (covers donated storage, where the output reuses an
  // input's buffer): with asynchronous commits every backing buffer of the
  // submitted commands must outlive the caller's references.
  for (auto& in : arr.inputs()) {
    encoder.add_temporary(in);
  }
  for (auto& s : arr.siblings()) {
    encoder.add_temporary(s);
  }
  encoder.add_temporary(arr);
  if (encoder.needs_commit()) {
    scheduler::notify_new_task(stream);
    encoder.add_completed_handler(
        [stream]() { scheduler::notify_task_completion(stream); });
    encoder.commit();
  }
}

void finalize(Stream s) {
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

} // namespace mlx::core::gpu
