// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// GPU evaluation entry points (mlx/backend/gpu/eval.h). Mirrors the CUDA
// backend's flow: run the primitive's eval_gpu, register input buffers as
// encoder temporaries, and commit when work was recorded.

#include "mlx/backend/gpu/eval.h"

#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

namespace mlx::core::gpu {

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
  omarchy::get_command_encoder(s).commit();
}

void synchronize(Stream s) {
  omarchy::get_command_encoder(s).synchronize();
}

} // namespace mlx::core::gpu
