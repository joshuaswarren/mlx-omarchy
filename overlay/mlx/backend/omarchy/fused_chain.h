// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <vector>

#include "mlx/array.h"
#include "mlx/stream.h"

namespace mlx::core::omarchy {

// Fused elementwise chain: collapses a run of same-shape float
// unary/binary Compiled-tape nodes into ONE compute dispatch. The chain
// interpreter shader (shaders/fused_chain.comp) executes a packed
// instruction program with all intermediates in registers, so a
// k-op chain costs 1 dispatch instead of k. Honeykrisp register
// pressure stays at 8 scalars per invocation.
//
// Any node the chain cannot carry (unsupported op class, shape change,
// exotic broadcast, too many leaves, no tail dependency) makes the
// whole run fall back to the per-node tape path: loud refusal
// semantics are unchanged. Compiled bf16 tapes remain refused, while
// the eager SwiGLU planner below uses the same interpreter with bf16
// intermediate rounding.
//
// Fusion defaults on after exact-ID parity and paired performance validation on
// M1 hardware. MLX_OMARCHY_FUSED_CHAIN=0 restores the per-node path.

struct FusedChainImpl;
bool fused_chain_enabled();

class FusedChain {
 public:
  // gate_enabled carries the fused-chain decision, read once per tape
  // evaluation; false makes every try_add refuse and preserves the per-node path.
  explicit FusedChain(bool gate_enabled);
  ~FusedChain();

  FusedChain(const FusedChain&) = delete;
  FusedChain& operator=(const FusedChain&) = delete;
  FusedChain(FusedChain&&);
  FusedChain& operator=(FusedChain&&);

  // Pure op/dtype check (the gate lives in the constructor argument):
  // a float32/float16/bfloat16 fusable unary or binary elementwise primitive.
  static bool can_start(const array& node);

  // Attempts to append `node` with resolved `inputs`. Returns false if
  // the node cannot be carried; the caller then closes the chain before
  // it. `is_tape_output` members may only be the chain tail. Two hard
  // contracts:
  // - an EXTENSION (chain already open) must consume the tail's
  //   register; a no-dependency same-shape sibling is refused so the
  //   interpreter closes the chain and the sibling opens a fresh one
  //   (a non-tail interior member can be consumed from outside the
  //   chain and would never be materialized);
  // - leaves pushed for a rejected member are rolled back, so a
  //   refused add leaves no orphan slots behind.
  // Every refusal that would make a carried chain undispatchable (f16
  // capabilities, leaf bounds, broadcast form) is decided HERE, before
  // a node is accepted.
  bool try_add(
      const array& node,
      const std::vector<array>& inputs,
      bool is_tape_output);

  // Dispatches the accumulated chain (1 or more nodes) as one fused
  // kernel. Returns the fused output carrying the last node's primitive
  // so downstream graph bookkeeping stays valid. Returns nullopt only
  // for an empty chain: refusals happen in try_add before acceptance.
  std::optional<array> evaluate(const Stream& stream);

  // Dispatch into the current tail's graph array. Used by eager fusion,
  // where the scheduler owns that array and no replacement descriptor may
  // be substituted for it.
  void evaluate_tail(const Stream& stream);

  // Number of tape nodes currently carried.
  size_t size() const;

  // Tracing-graph id of the chain's last member (size() > 0).
  std::uintptr_t tail_id() const;

  // True when `id` is one of the carried members (interior members have
  // no materialized output; consumers of one must close the chain).
  bool carries(std::uintptr_t id) const;

  // True when the CURRENT tail is a tape output: the caller must close
  // the chain before adding anything else (interior tape outputs would
  // lose their materialized results).
  bool tail_is_tape_output() const;

 private:
  std::unique_ptr<FusedChainImpl> impl_;
};


// One eval_impl graph window. The scope preflights exact eager SwiGLU
// shapes (gate * sigmoid(gate) * up) whose two interior values have one
// consumer, then try_eval_eager_fusion defers those interior dispatches
// and records the three operations as one FusedChain dispatch. With the
// gate off this is an empty, allocation-free plan.
class EagerFusionScope {
 public:
  explicit EagerFusionScope(const std::deque<array>& tape);
  ~EagerFusionScope();

  EagerFusionScope(const EagerFusionScope&) = delete;
  EagerFusionScope& operator=(const EagerFusionScope&) = delete;

 private:
  void* previous_;
};

// Returns true when the primitive was recorded (or deliberately deferred)
// by the active eager-fusion scope; false keeps the ordinary eval_gpu path.
bool try_eval_eager_fusion(array& node, const Stream& stream);

// DecodeFusion: one fused decode GEMV group. Up to kQmmVecMultiWeights
// affine transposed 4-bit/group-64 QuantizedMatmul nodes that read one
// single-row x, each optionally followed by the Add that is its only
// consumer (a bias or residual add), recorded as ONE QmmVecQ4Multi
// dispatch when the first member evaluates. Every member output and
// every Add output is materialized, so retained references stay valid.
struct GemvFusionMember {
  array node;
  std::optional<array> epilogue;
  std::optional<array> addend;
};

// Validates the group against the kernel contract, allocates every
// output, and records the dispatch. Returns false having allocated
// nothing when any member falls outside the contract; the caller then
// lets every node take its ordinary eval_gpu path. Defined in
// primitives.cpp beside QuantizedMatmul::eval_gpu.
bool dispatch_quantized_gemv_group(
    std::vector<GemvFusionMember>& members,
    const Stream& stream);

bool dispatch_slice_update_pair(
    std::array<array, 2>& nodes,
    const Stream& stream);

// MLX_OMARCHY_FUSED_GEMV=0 keeps every QuantizedMatmul and Add on the
// per-node path (the MLX_OMARCHY_FUSED_CHAIN gate also covers it).
bool fused_gemv_enabled();
} // namespace mlx::core::omarchy