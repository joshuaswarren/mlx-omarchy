// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/compiled.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>

#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/fused_chain.h"
#include "mlx/backend/omarchy/trace.h"
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

} // namespace

void eval_compiled_tape(
    const std::vector<array>& tape,
    const std::vector<array>& tape_inputs,
    const std::vector<array>& tape_outputs,
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    const Stream& stream) {
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
  // FuseDecodeChains gate: read ONCE per tape evaluation. With it off,
  // no chain structures are built and the interpreter below runs the
  // pre-chain per-node loop byte for byte.
  const bool fusion_enabled = env_flag("MLX_OMARCHY_FUSED_CHAIN");
  // Nodes consumed by MORE than one tape consumer (or surfaced as tape
  // outputs) must keep a materialized value, so a fused chain closes
  // after them; only single-consumer nodes may stay interior. This is
  // what makes interior chain members unreachable to non-chain ops by
  // construction.
  std::unordered_set<std::uintptr_t> must_materialize;
  if (fusion_enabled) {
    std::unordered_map<std::uintptr_t, int> consumer_counts;
    for (const auto& node : tape) {
      if (!node.has_primitive()) {
        continue;
      }
      for (const auto& in : node.inputs()) {
        ++consumer_counts[in.id()];
      }
    }
    must_materialize = output_ids;
    for (const auto& [id, count] : consumer_counts) {
      if (count > 1) {
        must_materialize.insert(id);
      }
    }
  }
  // MLX_OMARCHY_TAPE_CENSUS=<path>: append one line per tape evaluation
  // describing the fragment - node count, per-node op name, dtype,
  // tape-wide use count (references by other nodes plus the output
  // flag), and whether the node is a tape output. Measurement
  // instrument for the elementwise-chain census; off by default.
  if (const char* census_path = std::getenv("MLX_OMARCHY_TAPE_CENSUS")) {
    static FILE* census_out = std::fopen(census_path, "a");
    if (census_out) {
      std::unordered_map<std::uintptr_t, int> uses;
      std::unordered_map<std::uintptr_t, int> index;
      for (size_t n = 0; n < tape.size(); ++n) {
        index[tape[n].id()] = static_cast<int>(n);
      }
      for (const auto& node : tape) {
        for (const auto& in : node.inputs()) {
          uses[in.id()]++;
        }
      }
      std::ostringstream line;
      line << "CENSUS nodes=" << tape.size();
      for (size_t n = 0; n < tape.size(); ++n) {
        const auto& node = tape[n];
        int u = uses.count(node.id()) ? uses.at(node.id()) : 0;
        if (output_ids.count(node.id())) {
          u++;
        }
        char kind = '?';
        switch (kindof(node.dtype())) {
          case Dtype::Kind::b: kind = 'b'; break;
          case Dtype::Kind::i: kind = 'i'; break;
          case Dtype::Kind::u: kind = 'u'; break;
          case Dtype::Kind::V: kind = 'V'; break;
          case Dtype::Kind::f: kind = 'f'; break;
          case Dtype::Kind::c: kind = 'c'; break;
        }
        line << " | n" << n << "="
             << (node.has_primitive() ? node.primitive().name() : "Const")
             << " " << kind << " uses=" << u << " in=[";
        size_t ti = 0;
        for (const auto& in : node.inputs()) {
          auto it = index.find(in.id());
          line << (ti ? "," : "")
               << (it != index.end() ? std::to_string(it->second)
                                     : std::string("x"));
          ti++;
        }
        line << "]" << (output_ids.count(node.id()) ? " OUT" : "");
      }
      std::fprintf(census_out, "%s\n", line.str().c_str());
      std::fflush(census_out);
    }
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
  // Fused-chain accumulation (FuseDecodeChains, DEFAULT OFF via
  // MLX_OMARCHY_FUSED_CHAIN, read once above): consecutive same-shape
  // float unary/binary tape nodes collapse into ONE dispatch when the
  // gate is on, and every EXTENSION must consume the open chain's tail
  // so interior members are never consumable from outside. A node the
  // chain cannot carry closes it; the closed chain either fuses (one
  // dispatch) or falls back to the per-node path below, so refusal
  // semantics are unchanged. bf16 tapes are refused above before any
  // chain runs.
  FusedChain chain(fusion_enabled);
  // A node that consumes the OPEN chain's tail passes the tail's
  // tracing array itself; that reference is only valid while the chain
  // stays open, so any fallback below re-resolves after closing.
  auto close_chain = [&]() -> void {
    if (chain.size() == 0) {
      return;
    }
    trace::counters().compiled_tape_node_evaluations += chain.size();
    uint64_t dispatches_before =
        trace::counters().vk_compute_dispatches.load(std::memory_order_relaxed);
    encoder.in_tape_recording = true;
    auto fused = chain.evaluate(stream);
    encoder.in_tape_recording = false;
    trace::counters().compiled_tape_dispatches.fetch_add(
        trace::counters().vk_compute_dispatches.load(std::memory_order_relaxed) -
            dispatches_before,
        std::memory_order_relaxed);
    const auto tail = chain.tail_id();
    chain = FusedChain(fusion_enabled);
    if (fused) {
      resolved.emplace(tail, *fused);
      if (output_ids.find(tail) == output_ids.end()) {
        encoder.add_temporary(*fused);
      }
    }
  };
  auto resolve_inputs = [&](const array& node, std::vector<array>& out_inputs)
      -> bool {
    for (const auto& in : node.inputs()) {
      auto it = resolved.find(in.id());
      if (it != resolved.end()) {
        out_inputs.push_back(it->second);
      } else if (chain.size() > 0 && chain.tail_id() == in.id()) {
        // Continuation of the open chain; `in` is the tail's tracing
        // array and try_add maps it back to the tail register.
        out_inputs.push_back(in);
      } else if (!in.has_primitive()) {
        // Scalar constants baked into the tape hold their data already.
        out_inputs.push_back(in);
      } else {
        return false;
      }
    }
    return true;
  };
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
    if (fusion_enabled) {
      if (!resolve_inputs(node, node_inputs)) {
        // The node consumes the OPEN chain's tail as an INTERIOR
        // operand of a node that will not join: fuse what the chain
        // carries, then resolve against its materialized tail value.
        close_chain();
        node_inputs.clear();
        if (!resolve_inputs(node, node_inputs)) {
          unsupported_tape_op(primitive);
        }
      }

      // Fast path: the node joins the open fused chain. A tape output
      // may only sit at the tail, so the chain closes right after one.
      if (chain.try_add(
              node,
              node_inputs,
              must_materialize.find(node.id()) !=
                  must_materialize.end())) {
        if (chain.tail_is_tape_output()) {
          close_chain();
        }
        continue;
      }

      // The node cannot join (unsupported op class, shape change, no
      // tail dependency, ...). Close the chain - it may itself fuse -
      // and fall back to the per-node path. If any input referenced
      // the closed chain's tail, re-resolve it against the fused
      // output.
      bool needs_reresolve = false;
      for (const auto& in : node.inputs()) {
        if (chain.size() > 0 && chain.tail_id() == in.id()) {
          needs_reresolve = true;
        }
      }
      close_chain();
      if (needs_reresolve) {
        node_inputs.clear();
        if (!resolve_inputs(node, node_inputs)) {
          unsupported_tape_op(primitive);
        }
      }
    } else {
      // Gate off: the pre-chain resolution, byte for byte - no chain
      // objects, no chain scans, no chain structures.
      for (const auto& in : node.inputs()) {
        auto it = resolved.find(in.id());
        if (it != resolved.end()) {
          node_inputs.push_back(it->second);
        } else if (!in.has_primitive()) {
          // Scalar constants baked into the tape hold their data
          // already.
          node_inputs.push_back(in);
        } else {
          unsupported_tape_op(primitive);
        }
      }
    }

    // One output per tape node. The node's primitive stays attached so the
    // per-node dispatch sees the compiled stream.
    //
    // The output shape is derived from the eval-time inputs, not the
    // trace. A shapeless fragment serves every input shape from one
    // trace - the compile cache matches ndim and dtype only - so a
    // decode call at [1,1,...] legally reuses the prefill-traced tape,
    // and node.shape() would be the stale prefill shape. Elementwise
    // outputs are the trailing broadcast of their inputs (the same
    // contract upstream applies to fused fragments); Broadcast keeps its
    // own rule. At the trace shape this equals node.shape(), so
    // exact-shape tapes are byte-for-byte unchanged.
    Shape out_shape;
    if (typeid(primitive) == typeid(Broadcast)) {
      out_shape = primitive.output_shapes(node_inputs).front();
    } else {
      int nd = 0;
      for (const auto& in : node_inputs) {
        nd = std::max(nd, static_cast<int>(in.ndim()));
      }
      out_shape.resize(nd, 0);
      for (const auto& in : node_inputs) {
        auto dd = nd - static_cast<int>(in.ndim());
        for (int i = dd; i < nd; ++i) {
          out_shape[i] = std::max(out_shape[i], in.shape()[i - dd]);
        }
      }
      static std::atomic<bool> shape_notice{false};
      if (out_shape != node.shape() &&
          !shape_notice.exchange(true, std::memory_order_relaxed)) {
        auto fmt = [](const Shape& s) {
          std::ostringstream os;
          os << "[";
          for (size_t i = 0; i < s.size(); ++i) {
            if (i) {
              os << ",";
            }
            os << s[i];
          }
          os << "]";
          return os.str();
        };
        std::fprintf(
            stderr,
            "[omarchy] shapeless compiled fragment reused at a new shape:"
            " traced %s, serving %s\n",
            fmt(node.shape()).c_str(),
            fmt(out_shape).c_str());
      }
    }
    std::vector<array> outs;
    outs.push_back(
        array(std::move(out_shape), node.dtype(), node.primitive_ptr(), node_inputs));
    trace::counters().compiled_tape_node_evaluations++;
    uint64_t dispatches_before =
        trace::counters().vk_compute_dispatches.load(std::memory_order_relaxed);
    encoder.in_tape_recording = true;
    primitive.eval_gpu(node_inputs, outs);
    encoder.in_tape_recording = false;
    trace::counters().compiled_tape_dispatches.fetch_add(
        trace::counters().vk_compute_dispatches.load(std::memory_order_relaxed) -
        dispatches_before,
        std::memory_order_relaxed);

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
  close_chain();

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
