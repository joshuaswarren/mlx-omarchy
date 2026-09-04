// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Single-element host fold. On the eager rope composition a decode token
// dispatched hundreds of count==1 kernels (arange(1), the offset cast,
// the position add/mul chain); each cost a full dispatch, descriptor
// update, and submission share to produce one float. This fold computes
// those on the host instead - but only the profitable subset, by rule:
//
//   Fold iff every input's bytes are final on the host at eval time,
//   with no pending GPU dispatch behind them; if any input needs a
//   device read to become final, keep the dispatch.
//
// That rule can never introduce a synchronization this backend cannot
// afford: it reads only constants (written at creation) and outputs of
// this fold's own op set (whose accepted evals are host-written by
// construction). Deliberately absent: any event or status shortcut. A
// dispatched array can report a signaled event while its dispatch still
// sits in an open, unsubmitted command buffer, so event state proves
// nothing here.
//
// Bit-exactness contract, per op (differential battery in
// overlay/tests/omarchy/test_scalar_fold.cpp, against the llvmpipe
// kernels over +/-0, denormals, infs, NaN payloads, and the 2^24
// integer boundary):
//   Add/Multiply f32: IEEE-754 RNE on both sides; same bits. NaN
//     operands are excluded: NaN payload propagation is
//     implementation-defined and differs between host SSE and the
//     shader, so a NaN operand keeps the dispatch.
//   AsType i32->f32:  correctly-rounded conversion on both sides.
//   Arange f32:       the shader computes alpha + beta*float(index);
//                     the host fold replicates that formula literally
//                     at index 0. The + beta*0.0f term is load-bearing:
//                     -0.0 + 0.0 rounds to +0.0, so a raw alpha would
//                     keep -0 where the kernel produces +0.
// Ops outside this set (trig, exp, sigmoid, f16/bf16 math) use device
// polynomial implementations that are not bit-identical to host libm,
// and are excluded rather than accepted with a tolerance.

#include "mlx/backend/omarchy/scalar_fold.h"

#include <cmath>
#include <cstdlib>
#include <string_view>

#include "mlx/allocator.h"
#include "mlx/backend/omarchy/allocator.h"
#include "mlx/primitives.h"

namespace mlx::core::gpu {

namespace {

// Opt-in gate, default OFF. The fold is measured (see the receipt): on
// the rope-composition tree it removes 192 dispatches/token and 2.2x
// decode wall time on llvmpipe. It stays off by default because the
// ordering edge at test_runtime.cpp "small eager output stays ordered
// across deep submit boundaries" is unresolved: a folded host write
// consumed by a dispatch across deep submit boundaries can observe a
// recycled page's stale contents on this backend. Default-off keeps the
// tree's behavior byte-identical while the encoder-level fix lands.
bool fold_enabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("MLX_OMARCHY_SCALAR_FOLD");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
  }();
  return enabled;
}

// True when the array's bytes are final on the host without running GPU
// work. The Arange / AsType / Add / Multiply cases mirror the guards in
// host_fold_scalar exactly: an eval this fold accepts is host-written,
// so an evaluated array of that shape is final; anything else keeps the
// conservative false.
bool host_final(const array& a) {
  if (!a.has_primitive()) {
    // A constant: bytes written at creation (the Load idiom).
    return true;
  }
  std::string_view n = a.primitive().name();
  if (n == "Arange") {
    return a.size() == 1 && a.dtype() == float32;
  }
  if (n == "AsType") {
    auto& prim = static_cast<AsType&>(a.primitive());
    return a.size() == 1 && prim.state() == float32 &&
        a.inputs().at(0).size() == 1 &&
        a.inputs().at(0).dtype() == int32 && host_final(a.inputs().at(0));
  }
  if (n == "Add" || n == "Multiply") {
    if (a.size() != 1 || a.dtype() != float32) {
      return false;
    }
    for (auto& in : a.inputs()) {
      if (in.size() != 1 || in.dtype() != float32 || !host_final(in)) {
        return false;
      }
    }
    // NaN operand payloads propagate differently on host SSE and on
    // llvmpipe shaders, so a NaN operand keeps the dispatch. Inputs are
    // provably final here, so reading them is safe; the fold branch
    // below refuses identically.
    for (auto& in : a.inputs()) {
      if (std::isnan(*in.data<float>())) {
        return false;
      }
    }
    return true;
  }
  return false;
}

// Fresh dense allocation plus a synchronous host write. Never writes
// into an input's buffer: in-flight dispatches may still be reading
// inputs, and the workless-evals-pin-nothing rule protects nothing
// else. A non-coherent memory type publishes the write at the next
// submit's flush, exactly like the Load idiom for constants.
void write_scalar(array& out, float value) {
  out.set_data(omarchy::allocator().malloc(out.nbytes()));
  *out.data<float>() = value;
}

} // namespace

bool host_fold_scalar(array& arr) {
  if (!fold_enabled()) {
    return false;
  }
  if (arr.is_tracer() || !arr.has_primitive() || arr.size() != 1 ||
      arr.dtype() != float32) {
    return false;
  }
  auto& inputs = arr.inputs();
  std::string_view n = arr.primitive().name();

  if (n == "Arange") {
    auto& prim = static_cast<Arange&>(arr.primitive());
    auto [start, stop, step] = prim.state();
    float alpha = static_cast<float>(start);
    float beta = static_cast<float>(step);
    write_scalar(arr, alpha + beta * 0.0f);
    return true;
  }

  if (n == "AsType") {
    auto& prim = static_cast<AsType&>(arr.primitive());
    if (prim.state() != float32 || inputs.size() != 1 ||
        inputs[0].size() != 1 || inputs[0].dtype() != int32 ||
        !host_final(inputs[0])) {
      return false;
    }
    write_scalar(arr, static_cast<float>(*inputs[0].data<int32_t>()));
    return true;
  }

  if ((n == "Add" || n == "Multiply") && inputs.size() == 2 &&
      inputs[0].size() == 1 && inputs[1].size() == 1 &&
      inputs[0].dtype() == float32 && inputs[1].dtype() == float32 &&
      host_final(inputs[0]) && host_final(inputs[1])) {
    float x = *inputs[0].data<float>();
    float y = *inputs[1].data<float>();
    if (std::isnan(x) || std::isnan(y)) {
      // NaN payload propagation is implementation-defined; the kernel
      // keeps this case (see host_final above).
      return false;
    }
    write_scalar(arr, n == "Add" ? x + y : x * y);
    return true;
  }

  return false;
}

} // namespace mlx::core::gpu
