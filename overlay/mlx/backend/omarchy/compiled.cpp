// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/compiled.h"

#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>

#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/primitives.h"

namespace mlx::core::omarchy {
namespace {

// Upstream compile_fuse (mlx/compile.cpp is_fusable) only ever places
// unary, binary, ternary (Select), and Broadcast primitives inside a
// Compiled tape; reductions and everything else stay eager outside it.
// Every class in that set has an Omarchy eval_gpu implementation except
// Conjugate, Real, and Imag (complex), which keep the named-error
// contract here because their backend path is unsupported.
bool tape_op_supported(const Primitive& primitive) {
  const auto& p = primitive;
  // Upstream is_unary: everything fusable the backend implements, minus
  // the complex-only Conjugate/Real/Imag which stay refused below.
  return typeid(p) == typeid(Abs) || typeid(p) == typeid(ArcCos) ||
      typeid(p) == typeid(ArcCosh) || typeid(p) == typeid(ArcSin) ||
      typeid(p) == typeid(ArcSinh) || typeid(p) == typeid(ArcTan) ||
      typeid(p) == typeid(ArcTanh) || typeid(p) == typeid(AsType) ||
      typeid(p) == typeid(BitwiseInvert) || typeid(p) == typeid(Ceil) ||
      typeid(p) == typeid(Cos) || typeid(p) == typeid(Cosh) ||
      typeid(p) == typeid(Erf) || typeid(p) == typeid(ErfInv) ||
      typeid(p) == typeid(Exp) || typeid(p) == typeid(Expm1) ||
      typeid(p) == typeid(Floor) || typeid(p) == typeid(Log) ||
      typeid(p) == typeid(Log1p) || typeid(p) == typeid(LogicalNot) ||
      typeid(p) == typeid(Negative) || typeid(p) == typeid(Round) ||
      typeid(p) == typeid(Sigmoid) || typeid(p) == typeid(Sign) ||
      typeid(p) == typeid(Sin) || typeid(p) == typeid(Sinh) ||
      typeid(p) == typeid(Square) || typeid(p) == typeid(Sqrt) ||
      typeid(p) == typeid(Tan) || typeid(p) == typeid(Tanh) ||
      // Upstream is_binary.
      typeid(p) == typeid(Add) || typeid(p) == typeid(ArcTan2) ||
      typeid(p) == typeid(BitwiseBinary) || typeid(p) == typeid(Divide) ||
      typeid(p) == typeid(Equal) || typeid(p) == typeid(Greater) ||
      typeid(p) == typeid(GreaterEqual) || typeid(p) == typeid(Less) ||
      typeid(p) == typeid(LessEqual) || typeid(p) == typeid(LogAddExp) ||
      typeid(p) == typeid(LogicalAnd) || typeid(p) == typeid(LogicalOr) ||
      typeid(p) == typeid(Maximum) || typeid(p) == typeid(Minimum) ||
      typeid(p) == typeid(Multiply) || typeid(p) == typeid(NotEqual) ||
      typeid(p) == typeid(Power) || typeid(p) == typeid(Remainder) ||
      typeid(p) == typeid(Subtract) ||
      // Upstream is_ternary and is_broadcast.
      typeid(p) == typeid(Select) || typeid(p) == typeid(Broadcast);
}

[[noreturn]] void unsupported_tape_op(const Primitive& primitive) {
  throw std::runtime_error(
      "[omarchy] Compiled tape op " + std::string(primitive.name()) +
      " is not implemented for the Omarchy Vulkan backend. No GPU kernel"
      " exists for it; no silent CPU fallback occurs. Run it on an explicit"
      " CPU stream to use the CPU implementation.");
}

// bf16 compiled tapes corrupt nondeterministically on Honeykrisp: the M1
// mlx-lm bf16 greedy run returns garbage through the compiled swiglu
// fragment (2026-09-01 probe; inputs matched, garbage differed across
// runs), while no_fuse is correct, f16/4-bit tapes are correct, and
// llvmpipe matches eager exactly. No root cause in the interpreter is
// pinned yet, so the failing configuration is refused by name instead of
// silently returning wrong values. See docs/compatibility.md.
bool tape_has_bfloat16(const std::vector<array>& arrays) {
  for (const auto& a : arrays) {
    if (a.dtype() == bfloat16) {
      return true;
    }
  }
  return false;
}

[[noreturn]] void unsupported_tape_bfloat16() {
  throw std::runtime_error(
      "[omarchy] Compiled tape bfloat16 is refused: bf16 fragments corrupt"
      " nondeterministically on Honeykrisp. Re-run with MLX_DISABLE_COMPILE=1."
      " No GPU kernel exists for it; no silent CPU fallback occurs.");
}

// Compiled tapes return wrong values on real Apple GPU targets, and the
// defect is unfixed: a 4-bit mlx-lm greedy run emits garbage at normal
// speed with exit code 0
// (receipts/2026-09-03-dispatcher-compile-and-column-replace.md,
// docs/known-defects.md). This is the refusal point for the whole
// compiled path: eval_compiled_tape is the only tape runner in the
// backend and its single caller is Compiled::eval_gpu, so every
// compiled execution passes through this gate and no outer layer can
// bypass it. The trigonometric domain gate stays: tape nodes still
// dispatch through their own eval_gpu, which carries it.
[[noreturn]] void unsupported_tape_compilation() {
  throw std::runtime_error(
      "[omarchy] Compiled tapes are refused on real Apple GPUs: the tape"
      " interpreter returns wrong values on Honeykrisp and the defect is"
      " unfixed (docs/known-defects.md;"
      " receipts/2026-09-03-dispatcher-compile-and-column-replace.md)."
      " Re-run with MLX_DISABLE_COMPILE=1. Set"
      " MLX_OMARCHY_ALLOW_UNSAFE_COMPILE=1 only to investigate the defect"
      " deliberately; it permits wrong values.");
}

} // namespace

void eval_compiled_tape(
    const std::vector<array>& tape,
    const std::vector<array>& tape_inputs,
    const std::vector<array>& tape_outputs,
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    const Stream& stream) {
  if (compiled_tapes_refused(device(stream.device.index))) {
    unsupported_tape_compilation();
  }
  if (tape_has_bfloat16(tape) || tape_has_bfloat16(tape_inputs) ||
      tape_has_bfloat16(tape_outputs)) {
    unsupported_tape_bfloat16();
  }
  // Tape arrays refer to the tracing graph. Substitute the eval-time inputs
  // positionally and key every tape node to its computed temporary.
  std::unordered_map<std::uintptr_t, array> resolved;
  resolved.reserve(tape_inputs.size() + tape.size());
  for (size_t i = 0; i < tape_inputs.size(); ++i) {
    resolved.emplace(tape_inputs[i].id(), inputs[i]);
  }
  std::unordered_set<std::uintptr_t> output_ids;
  output_ids.reserve(tape_outputs.size());
  for (const auto& out : tape_outputs) {
    output_ids.insert(out.id());
  }

  // Compiled-tape debug switches (docs/install-omarchy.md). All default
  // off; with none set this function runs exactly the batching path it
  // ran before they existed. The scope publishes the barrier and reuse
  // switches to the encoder and the allocator for this recording;
  // per-node submission and the per-tape drain are applied inline below.
  // Recording happens inside this scope, so every recorded command
  // already carries the switched shape; the eventual submission needs no
  // switch state.
  TapeDebugScope debug_scope(
      env_flag("MLX_OMARCHY_TAPE_FULL_BARRIERS"),
      env_flag("MLX_OMARCHY_TAPE_NO_REUSE"));
  const bool per_node_submit = env_flag("MLX_OMARCHY_TAPE_PER_NODE_SUBMIT");
  const bool sync_every = env_flag("MLX_OMARCHY_TAPE_SYNC_EVERY");
  static std::atomic<bool> announced{false};
  if (!announced.load(std::memory_order_relaxed) &&
      (per_node_submit || sync_every || tape_full_barriers() ||
       tape_no_reuse()) &&
      !announced.exchange(true, std::memory_order_relaxed)) {
    std::fprintf(
        stderr,
        "[omarchy] compiled-tape debug switches active (diagnostics only,"
        " not product configuration; docs/install-omarchy.md):%s%s%s%s\n",
        per_node_submit ? " MLX_OMARCHY_TAPE_PER_NODE_SUBMIT" : "",
        sync_every ? " MLX_OMARCHY_TAPE_SYNC_EVERY" : "",
        tape_full_barriers() ? " MLX_OMARCHY_TAPE_FULL_BARRIERS" : "",
        tape_no_reuse() ? " MLX_OMARCHY_TAPE_NO_REUSE" : "");
  }

  auto& encoder = get_command_encoder(stream);
  for (const auto& node : tape) {
    if (!node.has_primitive()) {
      // A constant the tracer stored in the tape carries its own data.
      resolved.emplace(node.id(), node);
      continue;
    }
    Primitive& primitive = node.primitive();
    if (!tape_op_supported(primitive)) {
      unsupported_tape_op(primitive);
    }

    std::vector<array> node_inputs;
    node_inputs.reserve(node.inputs().size());
    for (const auto& in : node.inputs()) {
      auto it = resolved.find(in.id());
      if (it != resolved.end()) {
        node_inputs.push_back(it->second);
      } else if (!in.has_primitive()) {
        // Scalar constants baked into the tape hold their data already.
        node_inputs.push_back(in);
      } else {
        unsupported_tape_op(primitive);
      }
    }

    // One output per tape node. The node's primitive stays attached so the
    // per-node dispatch sees the compiled stream.
    std::vector<array> outs;
    outs.push_back(
        array(node.shape(), node.dtype(), node.primitive_ptr(), node_inputs));
    primitive.eval_gpu(node_inputs, outs);

    resolved.emplace(node.id(), outs[0]);
    if (output_ids.find(node.id()) == output_ids.end()) {
      // The evaluator's temporaries cover only the Compiled inputs and
      // outputs; keep intermediate buffers alive until the submission
      // completes. Output buffers live with the graph instead.
      encoder.add_temporary(outs[0]);
    }
    if (per_node_submit) {
      // MLX_OMARCHY_TAPE_PER_NODE_SUBMIT (diagnostic): submit after every
      // node, matching eager's op-granular command-buffer shape while
      // running the tape's own code path. Each submission waits on this
      // stream's previous completion, so the tape serializes exactly as
      // an eager chain would. If the Honeykrisp corruption disappears
      // under this switch, the defect lives in many-dispatches-per-
      // command-buffer; if it persists, the command buffer is innocent
      // and the tape's resource handling stays suspect.
      encoder.commit();
    }
  }

  for (size_t j = 0; j < tape_outputs.size(); ++j) {
    outputs[j].copy_shared_buffer(resolved.at(tape_outputs[j].id()));
  }
  if (sync_every) {
    // MLX_OMARCHY_TAPE_SYNC_EVERY (diagnostic): drain this stream
    // before returning, so no submission of the tape - and nothing
    // queued behind it - executes while the host runs ahead. Tests
    // whether host run-ahead is load-bearing for the Honeykrisp
    // corruption; the M1 verdict on the first three switches made this
    // the next named suspect (receipts/2026-09-03-tape-layer-isolation-
    // switches.md, MEASURED OUTCOME).
    encoder.synchronize();
  }
}

} // namespace mlx::core::omarchy
