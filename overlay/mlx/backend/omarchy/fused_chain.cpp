// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/fused_chain.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <array>
#include <memory>
#include <optional>
#include <typeinfo>
#include <vector>

#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/primitives.h"

namespace mlx::core::omarchy {
namespace {

// Op codes: lockstep with the switch in shaders/fused_chain.comp. This
// is the chain's own packed-program space, mapped from primitives by
// chain_op_for below; the values are NOT the ElementwiseOperation enum
// from ../primitives.cpp (they diverge from 11 up).
enum ChainOperation : uint32_t {
  ChainAdd = 0,
  ChainMultiply = 1,
  ChainDivide = 2,
  ChainMaximum = 3,
  ChainExp = 4,
  ChainSigmoid = 5,
  ChainSquare = 6,
  ChainSqrt = 7,
  ChainRsqrt = 8,
  ChainSubtract = 9,
  ChainNegative = 10,
  ChainMinimum = 11,
  ChainTanh = 12,
};

// Instruction packing: op[7:0] | a[15:8] | b[23:16] | dst[31:24].
constexpr uint8_t kChainLeafBase = 0x10;
constexpr uint32_t kMaxChainLeaves = 3;
constexpr uint32_t kMaxChainInstrs = 8;

constexpr uint32_t pack_instruction(
    uint32_t op,
    uint32_t a,
    uint32_t b,
    uint32_t dst) {
  return (op & 0xffu) | ((a & 0xffu) << 8) | ((b & 0xffu) << 16) |
      ((dst & 0xffu) << 24);
}

std::optional<uint32_t> chain_op_for(const Primitive& p) {
  const auto& t = typeid(p);
  if (t == typeid(Add)) {
    return ChainAdd;
  }
  if (t == typeid(Multiply)) {
    return ChainMultiply;
  }
  if (t == typeid(Divide)) {
    return ChainDivide;
  }
  if (t == typeid(Maximum)) {
    return ChainMaximum;
  }
  if (t == typeid(Exp)) {
    return ChainExp;
  }
  if (t == typeid(Sigmoid)) {
    return ChainSigmoid;
  }
  if (t == typeid(Square)) {
    return ChainSquare;
  }
  if (t == typeid(Sqrt)) {
    return ChainSqrt;
  }
  if (t == typeid(Subtract)) {
    return ChainSubtract;
  }
  if (t == typeid(Negative)) {
    return ChainNegative;
  }
  if (t == typeid(Minimum)) {
    return ChainMinimum;
  }
  if (t == typeid(Tanh)) {
    return ChainTanh;
  }
  return std::nullopt;
}

ComputeBinding chain_binding(const array& value) {
  auto* buffer = static_cast<const VulkanBuffer*>(value.buffer().ptr());
  return {buffer->buffer, 0, buffer->size};
}

// Leaf broadcast addressing modes; lockstep with fused_chain.comp.
constexpr uint32_t kLeafDirect = 0;
constexpr uint32_t kLeafModLast = 1;
constexpr uint32_t kLeafDivLast = 2;
constexpr uint32_t kLeafScalar = 3;

std::optional<uint32_t> leaf_mode_for(
    const array& leaf,
    uint32_t count,
    uint32_t last_dim) {
  const size_t data_size = leaf.data_size();
  if (data_size == count) {
    return kLeafDirect;
  }
  if (data_size == 1) {
    return kLeafScalar;
  }
  if (last_dim > 0 && data_size == last_dim) {
    return kLeafModLast;
  }
  if (last_dim > 0 && count % last_dim == 0 && data_size == count / last_dim) {
    return kLeafDivLast;
  }
  return std::nullopt;
}

} // namespace

struct FusedChainImpl {
  // Carried nodes and their resolved inputs, in tape order.
  std::vector<const array*> nodes;
  std::vector<std::vector<array>> inputs;
  // Packed program words.
  std::vector<uint32_t> program;
  // Leaf arrays with item offset and addressing mode.
  std::vector<array> leaves;
  std::vector<uint32_t> leaf_offsets;
  std::vector<uint32_t> leaf_modes;
  uint32_t count = 0;
  uint32_t last_dim = 0;
  // Eval-time output shape shared by every carried member (uniform by
  // construction; see try_add).
  Shape eval_shape;
  // Gate read ONCE per tape evaluation: try_add runs per node on the
  // decode hot path, and the gate must cost nothing when off.
  bool gate_enabled = false;
  bool open = false;
  // A tape output may only be the chain tail; extending past one would
  // demote a materialized result into an unmaterialized register.
  bool saw_tape_output = false;
};

FusedChain::FusedChain() : impl_(std::make_unique<FusedChainImpl>()) {
  impl_->gate_enabled = env_flag("MLX_OMARCHY_FUSED_CHAIN");
}
FusedChain::~FusedChain() = default;
FusedChain::FusedChain(FusedChain&& other)
    : impl_(std::move(other.impl_)) {}

FusedChain& FusedChain::operator=(FusedChain&& other) {
  if (this != &other) {
    // Hand the slot over without instantiating unique_ptr's deleter on
    // the incomplete impl type in caller translation units.
    auto* handed_over = other.impl_.release();
    impl_.reset(handed_over);
  }
  return *this;
}

bool FusedChain::can_start(const array& node) {
  // FuseDecodeChains is DEFAULT OFF: the fused chain has not been
  // equivalence-proven on M1 hardware yet. MLX_OMARCHY_FUSED_CHAIN=1
  // opts in (software-Vulkan correctness runs now, the hardware A/B
  // when scheduled); with the gate off the tape runs the per-node path
  // exactly as it did before the chain existed.
  if (!env_flag("MLX_OMARCHY_FUSED_CHAIN")) {
    return false;
  }
  if (!node.has_primitive()) {
    return false;
  }
  if (node.dtype() != float32 && node.dtype() != float16) {
    // bf16 tapes are refused before chains ever run; every other dtype
    // keeps the per-node tape path.
    return false;
  }
  return chain_op_for(node.primitive()).has_value();
}

size_t FusedChain::size() const {
  return impl_->nodes.size();
}

std::uintptr_t FusedChain::tail_id() const {
  return (*impl_->nodes.back()).id();
}

bool FusedChain::carries(std::uintptr_t id) const {
  for (const auto* node : impl_->nodes) {
    if (node->id() == id) {
      return true;
    }
  }
  return false;
}

bool FusedChain::try_add(
    const array& node,
    const std::vector<array>& node_inputs,
    bool is_tape_output) {
  if (impl_->saw_tape_output) {
    return false;
  }
  if (impl_->nodes.size() >= kMaxChainInstrs) {
    return false;
  }
  // Cached gate: read once per tape evaluation (constructor), not once
  // per node on the decode hot path.
  if (!impl_->gate_enabled) {
    return false;
  }
  if (!can_start(node)) {
    return false;
  }
  const auto op = chain_op_for(node.primitive()).value();
  // Eval-time broadcast shape of this node's output, derived from the
  // resolved inputs exactly as the per-node interpreter derives its
  // output shape. A shapeless tape serves decode from a prefill trace,
  // so node.shape() can be a stale trace shape; chain uniformity and
  // the dispatch count must key on what actually evaluates.
  Shape eval_shape;
  {
    int nd = 0;
    for (const auto& in : node_inputs) {
      nd = std::max(nd, static_cast<int>(in.ndim()));
    }
    eval_shape.resize(nd, 0);
    for (const auto& in : node_inputs) {
      auto dd = nd - static_cast<int>(in.ndim());
      for (int i = dd; i < nd; ++i) {
        eval_shape[i] = std::max(eval_shape[i], in.shape()[i - dd]);
      }
    }
  }
  if (impl_->open) {
    if (eval_shape != impl_->eval_shape ||
        node.dtype() != impl_->nodes.front()->dtype()) {
      return false;
    }
  }

  uint32_t count = 0;
  uint32_t last_dim = 0;
  if (impl_->open) {
    count = impl_->count;
    last_dim = impl_->last_dim;
  } else {
    if (eval_shape.empty()) {
      // A scalar chain has nothing to fuse.
      return false;
    }
    int64_t n = 1;
    for (auto d : eval_shape) {
      n *= d;
    }
    if (n > std::numeric_limits<uint32_t>::max() || n == 0) {
      return false;
    }
    count = static_cast<uint32_t>(n);
    last_dim = static_cast<uint32_t>(eval_shape.back());
  }

  auto encode_leaf = [&](const array& in) -> std::optional<uint32_t> {
    if (!in.flags().contiguous) {
      return std::nullopt;
    }
    if (impl_->leaves.size() >= kMaxChainLeaves) {
      return std::nullopt;
    }
    const auto mode = leaf_mode_for(in, count, last_dim);
    if (!mode) {
      return std::nullopt;
    }
    const size_t item_offset = in.offset() / in.itemsize();
    if (item_offset > std::numeric_limits<uint32_t>::max() / 2) {
      return std::nullopt;
    }
    // Every addressed item must sit inside the leaf's buffer. Checked
    // HERE, at add time: a leaf the chain cannot carry must close the
    // chain while it can still fall back, never at dispatch time when
    // carried members are already unmaterialized.
    uint32_t span;
    switch (mode.value()) {
      case kLeafDirect:
        span = count;
        break;
      case kLeafModLast:
        span = last_dim;
        break;
      case kLeafDivLast:
        span = count / last_dim;
        break;
      default:
        span = 1;
        break;
    }
    if (!compute_index_span_fits(item_offset, span)) {
      return std::nullopt;
    }
    auto* leaf_vk = static_cast<const VulkanBuffer*>(in.buffer().ptr());
    const uint64_t byte_end =
        item_offset * in.itemsize() + span * in.itemsize();
    if (byte_end > leaf_vk->size) {
      return std::nullopt;
    }
    impl_->leaves.push_back(in);
    impl_->leaf_offsets.push_back(static_cast<uint32_t>(item_offset));
    impl_->leaf_modes.push_back(mode.value());
    return kChainLeafBase + static_cast<uint32_t>(impl_->leaves.size() - 1);
  };

  // Operand resolution: one operand may be the previous member's output
  // (its register); everything else must be an addressable leaf.
  const uint32_t dst = static_cast<uint32_t>(impl_->nodes.size());
  const uint32_t prev_reg = dst > 0 ? dst - 1 : std::numeric_limits<uint32_t>::max();
  std::optional<uint32_t> a;
  std::optional<uint32_t> b;
  auto is_prev = [&](const array& in) {
    return dst > 0 && in.id() == (*impl_->nodes.back()).id();
  };
  if (node_inputs.size() == 1) {
    a = is_prev(node_inputs[0]) ? prev_reg : encode_leaf(node_inputs[0]);
    b = a;
  } else if (node_inputs.size() == 2) {
    if (is_prev(node_inputs[0])) {
      a = prev_reg;
      b = encode_leaf(node_inputs[1]);
    } else if (is_prev(node_inputs[1])) {
      b = prev_reg;
      a = encode_leaf(node_inputs[0]);
    } else {
      a = encode_leaf(node_inputs[0]);
      b = encode_leaf(node_inputs[1]);
    }
  } else {
    return false;
  }
  if (!a || !b) {
    return false;
  }

  // Float16 chains need the 16-bit storage capabilities. Refused at
  // add time for the same reason as the leaf bounds above.
  if (!impl_->open && node.dtype() == float16) {
    const auto& capabilities = device().capabilities();
    if (!capabilities.shader_float16 ||
        !capabilities.storage_buffer_16bit_access) {
      return false;
    }
  }

  impl_->program.push_back(pack_instruction(op, a.value(), b.value(), dst));
  impl_->nodes.push_back(&node);
  impl_->inputs.push_back(node_inputs);
  if (!impl_->open) {
    impl_->count = count;
    impl_->last_dim = last_dim;
    impl_->eval_shape = eval_shape;
    impl_->open = true;
  }
  if (is_tape_output) {
    impl_->saw_tape_output = true;
  }
  return true;
}

bool FusedChain::tail_is_tape_output() const {
  return impl_->saw_tape_output;
}

std::optional<array> FusedChain::evaluate(const Stream& stream) {
  if (impl_->nodes.empty()) {
    return std::nullopt;
  }
  auto& encoder = get_command_encoder(stream);

  // The fused output carries the LAST node's primitive so the graph
  // above the tape stays valid.
  const array& tail = *impl_->nodes.back();
  // Eval-time shape, not the tail's trace shape (shapeless tapes).
  array out(
      impl_->eval_shape,
      tail.dtype(),
      tail.primitive_ptr(),
      impl_->inputs.back());
  out.set_data(allocator().malloc(out.nbytes()));


  // Upload the program words. The allocator hands back host-visible
  // mapped buffers, so the words land with a plain copy.
  const size_t program_bytes = impl_->program.size() * sizeof(uint32_t);
  Buffer program_buffer = allocator().malloc(program_bytes);
  auto* program_vk = static_cast<VulkanBuffer*>(program_buffer.ptr());
  std::memcpy(program_vk->data, impl_->program.data(), program_bytes);
  // Keep the program buffer alive until the submission completes.
  array program_keeper(
      Shape{static_cast<int>(impl_->program.size())},
      uint32,
      nullptr,
      {});
  array::Flags keeper_flags;
  keeper_flags.contiguous = true;
  keeper_flags.row_contiguous = true;
  keeper_flags.col_contiguous = true;
  program_keeper.set_data(
      program_buffer,
      program_keeper.size(),
      Strides{1},
      keeper_flags,
      0);
  encoder.add_temporary(program_keeper);

  // Leaf bounds and f16 capabilities were refused at add time
  // (try_add); a chain that reaches this point always dispatches.
  const uint32_t count = impl_->count;
  const uint32_t last_dim = impl_->last_dim;

  // Push constants. The backend pushes ComputeParams by value; the chain
  // shader declares ten scalars whose byte offsets are the first ten
  // ComputeParams fields. Lockstep mapping (shader name = host field):
  //   count = count, instr_count = operation, last_dim = lhs_size,
  //   dst_final = rhs_size,
  //   leaf_offset[3] = reduce_size, output_size, lhs_offset
  //   leaf_mode[3] = rhs_offset, output_offset, aux_size
  ComputeParams params;
  params.count = count;
  params.operation = static_cast<uint32_t>(impl_->program.size());
  params.lhs_size = last_dim;
  params.rhs_size = static_cast<uint32_t>(impl_->nodes.size() - 1);
  params.reduce_size = impl_->leaf_offsets.size() > 0 ? impl_->leaf_offsets[0] : 0;
  params.output_size = impl_->leaf_offsets.size() > 1 ? impl_->leaf_offsets[1] : 0;
  params.lhs_offset = impl_->leaf_offsets.size() > 2 ? impl_->leaf_offsets[2] : 0;
  params.rhs_offset = impl_->leaf_modes.size() > 0 ? impl_->leaf_modes[0] : 0;
  params.output_offset = impl_->leaf_modes.size() > 1 ? impl_->leaf_modes[1] : 0;
  params.aux_size = impl_->leaf_modes.size() > 2 ? impl_->leaf_modes[2] : 0;

  // Bindings are positional (shader: 0-2 leaves, 3 program, 4 output).
  // Unused leaf slots are padded with the output buffer; no carried
  // program references them.
  std::array<ComputeBinding, kMaxChainLeaves + 2> bindings{
      chain_binding(out),
      chain_binding(out),
      chain_binding(out),
      chain_binding(program_keeper),
      chain_binding(out)};
  for (size_t i = 0; i < impl_->leaves.size(); ++i) {
    bindings[i] = chain_binding(impl_->leaves[i]);
  }

  const auto kernel = out.dtype() == float16
      ? ComputeKernel::FusedChainF16
      : ComputeKernel::FusedChainF32;
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      compute_dispatch_group_count(count));
  return out;
}

} // namespace mlx::core::omarchy
