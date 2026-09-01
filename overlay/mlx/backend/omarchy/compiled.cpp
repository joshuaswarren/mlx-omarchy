// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/compiled.h"

#include <stdexcept>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>

#include "mlx/backend/omarchy/encoder.h"
#include "mlx/primitives.h"

namespace mlx::core::omarchy {
namespace {

// The elementwise subset the tape interpreter supports. Everything else
// keeps the named-error contract.
bool tape_op_supported(const Primitive& primitive) {
  const auto& p = primitive;
  return typeid(p) == typeid(Add) || typeid(p) == typeid(Multiply) ||
      typeid(p) == typeid(Divide) || typeid(p) == typeid(Maximum) ||
      typeid(p) == typeid(Exp) || typeid(p) == typeid(Sigmoid) ||
      typeid(p) == typeid(Square) || typeid(p) == typeid(Sqrt) ||
      typeid(p) == typeid(Subtract) || typeid(p) == typeid(Negative) ||
      typeid(p) == typeid(AsType) || typeid(p) == typeid(Broadcast);
}

[[noreturn]] void unsupported_tape_op(const Primitive& primitive) {
  throw std::runtime_error(
      "[omarchy] Compiled tape op " + std::string(primitive.name()) +
      " is not implemented for the Omarchy Vulkan backend. No CPU fallback"
      " is available in Omarchy builds.");
}

} // namespace

void eval_compiled_tape(
    const std::vector<array>& tape,
    const std::vector<array>& tape_inputs,
    const std::vector<array>& tape_outputs,
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    const Stream& stream) {
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
  }

  for (size_t j = 0; j < tape_outputs.size(); ++j) {
    outputs[j].copy_shared_buffer(resolved.at(tape_outputs[j].id()));
  }
}

} // namespace mlx::core::omarchy
