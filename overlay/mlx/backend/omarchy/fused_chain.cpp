// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/fused_chain.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core::omarchy {

bool fused_chain_enabled() {
  return std::getenv("MLX_OMARCHY_FUSED_CHAIN") == nullptr ||
      env_flag("MLX_OMARCHY_FUSED_CHAIN");
}
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
  return {buffer->buffer, 0, buffer->size, buffer};
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
  // Stable graph IDs replace references to scheduler tape entries: eager
  // entries are detached immediately after gpu::eval returns.
  std::vector<std::uintptr_t> node_ids;
  std::vector<array> node_arrays;
  std::vector<std::vector<array>> inputs;
  std::vector<uint32_t> program;
  std::vector<array> leaves;
  std::vector<uint32_t> leaf_offsets;
  std::vector<uint32_t> leaf_modes;
  uint32_t count = 0;
  uint32_t last_dim = 0;
  Shape eval_shape;
  Dtype dtype = float32;
  std::shared_ptr<Primitive> tail_primitive;
  std::optional<array> tail_array;
  bool gate_enabled = false;
  bool open = false;
  bool saw_tape_output = false;
};

FusedChain::FusedChain(bool gate_enabled)
    : impl_(std::make_unique<FusedChainImpl>()) {
  // The gate is read ONCE per tape evaluation by the owner
  // (eval_compiled_tape) and handed in, so the off path costs no env
  // lookups at all.
  impl_->gate_enabled = gate_enabled;
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
  // Pure op/dtype check; the DEFAULT-OFF gate lives in the
  // constructor argument (MLX_OMARCHY_FUSED_CHAIN, decided by the tape
  // interpreter).
  if (!node.has_primitive()) {
    return false;
  }
  if (node.dtype() != float32 && node.dtype() != float16 &&
      node.dtype() != bfloat16) {
    return false;
  }
  return chain_op_for(node.primitive()).has_value();
}

size_t FusedChain::size() const {
  return impl_->node_ids.size();
}

std::uintptr_t FusedChain::tail_id() const {
  return impl_->node_ids.back();
}

bool FusedChain::carries(std::uintptr_t id) const {
  for (auto node_id : impl_->node_ids) {
    if (node_id == id) {
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
  if (impl_->node_ids.size() >= kMaxChainInstrs) {
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
  // the dispatch count must key on what actually evaluates. The tail
  // contributes the chain's eval shape, not its tracing shape.
  auto is_prev = [&](const array& in) {
    return !impl_->node_ids.empty() && in.id() == impl_->node_ids.back();
  };
  Shape eval_shape;
  {
    int nd = 0;
    for (const auto& in : node_inputs) {
      nd = std::max(nd, static_cast<int>(
                            is_prev(in) ? impl_->eval_shape.size()
                                        : in.ndim()));
    }
    eval_shape.resize(nd, 0);
    for (const auto& in : node_inputs) {
      const bool prev = is_prev(in);
      const Shape& shape = prev ? impl_->eval_shape : in.shape();
      auto dd = nd - static_cast<int>(shape.size());
      for (int i = dd; i < nd; ++i) {
        eval_shape[i] = std::max(eval_shape[i], shape[i - dd]);
      }
    }
  }
  if (impl_->open) {
    if (eval_shape != impl_->eval_shape || node.dtype() != impl_->dtype) {
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

  if (!impl_->open && node.dtype() != float32) {
    const auto& capabilities = device().capabilities();
    if ((node.dtype() == float16 && !capabilities.shader_float16) ||
        (node.dtype() == bfloat16 && !capabilities.shader_int16) ||
        !capabilities.storage_buffer_16bit_access) {
      return false;
    }
  }

  auto encode_leaf = [&](const array& in) -> std::optional<uint32_t> {
    if (!in.flags().contiguous) {
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
    for (size_t i = 0; i < impl_->leaves.size(); ++i) {
      if (impl_->leaves[i].id() == in.id() &&
          impl_->leaf_offsets[i] == item_offset &&
          impl_->leaf_modes[i] == mode.value()) {
        return kChainLeafBase + static_cast<uint32_t>(i);
      }
    }
    if (impl_->leaves.size() >= kMaxChainLeaves) {
      return std::nullopt;
    }
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
  const uint32_t dst = static_cast<uint32_t>(impl_->node_ids.size());
  const uint32_t prev_reg = dst > 0 ? dst - 1 : std::numeric_limits<uint32_t>::max();
  std::optional<uint32_t> a;
  std::optional<uint32_t> b;
  if (impl_->open && !is_prev(node_inputs[0]) &&
      (node_inputs.size() < 2 || !is_prev(node_inputs[1]))) {
    // EXTENSIONS must consume the tail's register. A same-shape node
    // sharing no data with the tail would ride the chain as a NON-tail
    // interior member; closing the chain materializes the tail only,
    // so a consumer of that member outside the chain would resolve
    // nothing. Refusing here makes the interpreter close the chain
    // first; the sibling then opens a fresh chain of its own.
    return false;
  }
  // Leaves pushed for a REJECTED member are rolled back below, so the
  // chain never carries orphan slots.
  const size_t leaf_base = impl_->leaves.size();
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
      // Chain head only (extensions were refused above): both
      // operands are addressable leaves.
      a = encode_leaf(node_inputs[0]);
      b = encode_leaf(node_inputs[1]);
    }
  } else {
    return false;
  }
  if (!a || !b) {
    // Erase, not resize: array is not default-constructible.
    impl_->leaves.erase(
        impl_->leaves.begin() + static_cast<std::ptrdiff_t>(leaf_base),
        impl_->leaves.end());
    impl_->leaf_offsets.resize(leaf_base);
    impl_->leaf_modes.resize(leaf_base);
    return false;
  }


  impl_->program.push_back(pack_instruction(op, a.value(), b.value(), dst));
  impl_->node_ids.push_back(node.id());
  impl_->inputs.push_back(node_inputs);
  impl_->node_arrays.push_back(node);
  impl_->dtype = node.dtype();
  impl_->tail_primitive = node.primitive_ptr();
  impl_->tail_array = node;
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

namespace {

// The SwiGLU program (r0 = sigmoid(g); r1 = g * r0; out = r1 * u, two
// direct leaves) gets the straight-line four-wide shaders/swiglu.comp
// with the interpreter's exact rounding. Returns the (gate, up) leaf
// slots when the chain has that shape and the alignment the kernel
// needs, else nullopt and the interpreter runs.
std::optional<std::pair<uint32_t, uint32_t>> swiglu_leaves(
    const FusedChainImpl& chain,
    bool materialize_intermediates) {
  if (materialize_intermediates || chain.program.size() != 3 ||
      chain.leaves.size() != 2 || chain.node_ids.size() != 3 ||
      (chain.count & 3u) != 0u || chain.dtype == float32) {
    return std::nullopt;
  }
  for (uint32_t mode : chain.leaf_modes) {
    if (mode != kLeafDirect) {
      return std::nullopt;
    }
  }
  auto field = [&](size_t i, int shift) {
    return (chain.program[i] >> shift) & 0xffu;
  };
  uint32_t op0 = field(0, 0), a0 = field(0, 8), d0 = field(0, 24);
  uint32_t op1 = field(1, 0), a1 = field(1, 8), b1 = field(1, 16),
           d1 = field(1, 24);
  uint32_t op2 = field(2, 0), a2 = field(2, 8), b2 = field(2, 16),
           d2 = field(2, 24);
  if (op0 != ChainSigmoid || op1 != ChainMultiply || op2 != ChainMultiply ||
      a0 < kChainLeafBase || d2 != chain.node_ids.size() - 1) {
    return std::nullopt;
  }
  uint32_t gate = a0 - kChainLeafBase;
  // Multiplication commutes exactly, so either operand order matches.
  bool mul1_ok = (a1 == a0 && b1 == d0) || (a1 == d0 && b1 == a0);
  uint32_t up = a2 == d1 ? b2 : (b2 == d1 ? a2 : 0u);
  if (!mul1_ok || up < kChainLeafBase || up - kChainLeafBase == gate) {
    return std::nullopt;
  }
  up -= kChainLeafBase;
  if (gate >= chain.leaves.size() || up >= chain.leaves.size() ||
      (chain.leaf_offsets[gate] & 3u) != 0u ||
      (chain.leaf_offsets[up] & 3u) != 0u) {
    return std::nullopt;
  }
  return std::make_pair(gate, up);
}

void dispatch_chain(
    FusedChainImpl& chain,
    array& out,
    const Stream& stream,
    bool materialize_intermediates) {
  auto& encoder = get_command_encoder(stream);
  out.set_data(allocator().malloc(out.nbytes()));
  if (static const bool dump = std::getenv("MLX_OMARCHY_CHAIN_DUMP") != nullptr;
      dump) {
    std::fprintf(stderr, "[chain] count=%u dtype=%d materialize=%d leaves=%zu nodes=%zu program=",
        chain.count, static_cast<int>(chain.dtype.val()), materialize_intermediates ? 1 : 0,
        chain.leaves.size(), chain.node_ids.size());
    for (uint32_t w : chain.program) std::fprintf(stderr, "%08x ", w);
    std::fprintf(stderr, "modes=");
    for (uint32_t m : chain.leaf_modes) std::fprintf(stderr, "%u ", m);
    std::fprintf(stderr, "offsets=");
    for (uint32_t o : chain.leaf_offsets) std::fprintf(stderr, "%u ", o);
    std::fprintf(stderr, "\n");
  }
  if (auto leaves = swiglu_leaves(chain, materialize_intermediates)) {
    ComputeParams params;
    params.count = chain.count;
    params.operation = chain.leaf_offsets[leaves->first];
    params.lhs_size = chain.leaf_offsets[leaves->second];
    std::array<ComputeBinding, 3> bindings{
        chain_binding(chain.leaves[leaves->first]),
        chain_binding(chain.leaves[leaves->second]),
        chain_binding(out)};
    encoder.dispatch_compute(
        out.dtype() == float16 ? ComputeKernel::SwigluF16
                               : ComputeKernel::SwigluBF16,
        bindings,
        params,
        compute_dispatch_group_count(chain.count / 4u));
    return;
  }
  if (materialize_intermediates) {
    for (size_t i = 0; i + 1 < chain.node_arrays.size(); ++i) {
      chain.node_arrays[i].set_data(
          allocator().malloc(chain.node_arrays[i].nbytes()));
    }
  }

  const size_t program_bytes = chain.program.size() * sizeof(uint32_t);
  Buffer program_buffer = allocator().malloc(program_bytes);
  auto* program_vk = static_cast<VulkanBuffer*>(program_buffer.ptr());
  std::memcpy(program_vk->data, chain.program.data(), program_bytes);
  array program_keeper(
      Shape{static_cast<int>(chain.program.size())}, uint32, nullptr, {});
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

  ComputeParams params;
  params.count = chain.count;
  params.operation = static_cast<uint32_t>(chain.program.size());
  params.lhs_size = chain.last_dim;
  params.rhs_size = static_cast<uint32_t>(chain.node_ids.size() - 1);
  params.reduce_size = chain.leaf_offsets.size() > 0 ? chain.leaf_offsets[0] : 0;
  params.output_size = chain.leaf_offsets.size() > 1 ? chain.leaf_offsets[1] : 0;
  params.lhs_offset = chain.leaf_offsets.size() > 2 ? chain.leaf_offsets[2] : 0;
  params.rhs_offset = chain.leaf_modes.size() > 0 ? chain.leaf_modes[0] : 0;
  params.output_offset = chain.leaf_modes.size() > 1 ? chain.leaf_modes[1] : 0;
  params.aux_size = chain.leaf_modes.size() > 2 ? chain.leaf_modes[2] : 0;
  params.aux_offset = materialize_intermediates ? 1u : 0u;

  std::array<ComputeBinding, kMaxChainLeaves + 3> bindings{
      chain_binding(out),
      chain_binding(out),
      chain_binding(out),
      chain_binding(program_keeper),
      chain_binding(out),
      chain_binding(out)};
  for (size_t i = 0; i < chain.leaves.size(); ++i) {
    bindings[i] = chain_binding(chain.leaves[i]);
  }
  if (materialize_intermediates && !chain.node_arrays.empty()) {
    bindings[2] = chain_binding(chain.node_arrays[0]);
    if (chain.node_arrays.size() > 2) {
      bindings[5] = chain_binding(chain.node_arrays[1]);
    }
  }

  auto kernel = ComputeKernel::FusedChainF32;
  if (out.dtype() == float16) {
    kernel = ComputeKernel::FusedChainF16;
  } else if (out.dtype() == bfloat16) {
    kernel = ComputeKernel::FusedChainBF16;
  }
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      compute_dispatch_group_count(chain.count));
}

} // namespace

std::optional<array> FusedChain::evaluate(const Stream& stream) {
  if (impl_->node_ids.empty()) {
    return std::nullopt;
  }
  array out(
      impl_->eval_shape,
      impl_->dtype,
      impl_->tail_primitive,
      impl_->inputs.back());
  dispatch_chain(*impl_, out, stream, false);
  return out;
}

void FusedChain::evaluate_tail(const Stream& stream) {
  if (impl_->node_ids.empty() || !impl_->tail_array) {
    return;
  }
  dispatch_chain(*impl_, *impl_->tail_array, stream, true);
}

namespace {

enum class EagerStep : uint8_t { sigmoid, gate_mul, output_mul };

struct EagerRole {
  std::uintptr_t group;
  EagerStep step;
};

struct EagerFusionState {
  std::unordered_map<std::uintptr_t, EagerRole> roles;
  std::unordered_map<std::uintptr_t, FusedChain> chains;
};

thread_local EagerFusionState* eager_state = nullptr;

bool is_op(const array* node, const std::type_info& op) {
  return node && node->has_primitive() && typeid(node->primitive()) == op;
}

} // namespace

EagerFusionScope::EagerFusionScope(const std::deque<array>& tape)
    : previous_(eager_state) {
  eager_state = nullptr;
  if (!fused_chain_enabled()) {
    return;
  }

  auto* state = new EagerFusionState;
  eager_state = state;
  std::unordered_map<std::uintptr_t, const array*> nodes;
  std::unordered_map<std::uintptr_t, size_t> uses;
  nodes.reserve(tape.size());
  for (const auto& node : tape) {
    nodes.emplace(node.id(), &node);
    for (const auto& input : node.inputs()) {
      ++uses[input.id()];
    }
  }
  auto lookup = [&](const array& ref) -> const array* {
    auto it = nodes.find(ref.id());
    return it == nodes.end() ? nullptr : it->second;
  };
  std::unordered_set<std::uintptr_t> claimed;
  for (const auto& tail : tape) {
    if (!is_op(&tail, typeid(Multiply)) || tail.inputs().size() != 2 ||
        claimed.count(tail.id())) {
      continue;
    }
    const array* left = lookup(tail.inputs()[0]);
    const array* right = lookup(tail.inputs()[1]);
    const array* inner = is_op(left, typeid(Multiply)) ? left :
        (is_op(right, typeid(Multiply)) ? right : nullptr);
    if (!inner || inner->inputs().size() != 2 || uses[inner->id()] != 1) {
      continue;
    }
    const array* inner_left = lookup(inner->inputs()[0]);
    const array* inner_right = lookup(inner->inputs()[1]);
    const array* sigmoid = is_op(inner_left, typeid(Sigmoid)) ? inner_left :
        (is_op(inner_right, typeid(Sigmoid)) ? inner_right : nullptr);
    if (!sigmoid || sigmoid->inputs().size() != 1 ||
        uses[sigmoid->id()] != 1) {
      continue;
    }
    const array& gate = sigmoid == inner_left ? inner->inputs()[1]
                                               : inner->inputs()[0];
    if (gate.id() != sigmoid->inputs()[0].id() ||
        tail.dtype() != inner->dtype() || tail.dtype() != sigmoid->dtype() ||
        tail.primitive().stream() != inner->primitive().stream() ||
        tail.primitive().stream() != sigmoid->primitive().stream() ||
        claimed.count(inner->id()) || claimed.count(sigmoid->id())) {
      continue;
    }
    const auto group = tail.id();
    state->roles.emplace(
        sigmoid->id(), EagerRole{group, EagerStep::sigmoid});
    state->roles.emplace(
        inner->id(), EagerRole{group, EagerStep::gate_mul});
    state->roles.emplace(
        tail.id(), EagerRole{group, EagerStep::output_mul});
    claimed.insert(sigmoid->id());
    claimed.insert(inner->id());
    claimed.insert(tail.id());
  }
}

EagerFusionScope::~EagerFusionScope() {
  delete eager_state;
  eager_state = static_cast<EagerFusionState*>(previous_);
}

bool try_eval_eager_fusion(array& node, const Stream& stream) {
  if (!eager_state) {
    return false;
  }
  auto role_it = eager_state->roles.find(node.id());
  if (role_it == eager_state->roles.end()) {
    return false;
  }
  const auto role = role_it->second;
  if (role.step == EagerStep::sigmoid) {
    if (device().compute().binding_limit() < kComputeBindingBudget) {
      return false;
    }
    auto [chain_it, inserted] =
        eager_state->chains.try_emplace(role.group, true);
    if (!inserted ||
        !chain_it->second.try_add(node, node.inputs(), false)) {
      eager_state->chains.erase(role.group);
      return false;
    }
    return true;
  }

  auto chain_it = eager_state->chains.find(role.group);
  if (chain_it == eager_state->chains.end()) {
    return false;
  }
  if (!chain_it->second.try_add(node, node.inputs(), false)) {
    chain_it->second.evaluate_tail(stream);
    eager_state->chains.erase(chain_it);
    return false;
  }
  if (role.step == EagerStep::output_mul) {
    chain_it->second.evaluate_tail(stream);
    eager_state->chains.erase(chain_it);
  }
  return true;
}
} // namespace mlx::core::omarchy
