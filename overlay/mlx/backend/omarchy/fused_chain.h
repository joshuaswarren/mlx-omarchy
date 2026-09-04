// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
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
// semantics are unchanged, and bf16 tapes are refused before chains
// are ever considered.
//
// DEFAULT OFF: the whole mechanism is gated behind
// MLX_OMARCHY_FUSED_CHAIN until it is equivalence-proven on M1
// hardware. With the gate unset the owner passes gate_enabled=false,
// every try_add refuses, and the tape runs the per-node path exactly
// as it did before this class existed.

struct FusedChainImpl;

class FusedChain {
 public:
  // gate_enabled carries the DEFAULT-OFF MLX_OMARCHY_FUSED_CHAIN
  // decision, read ONCE per tape evaluation by the owner; false makes
  // every try_add refuse and the tape run the per-node path.
  explicit FusedChain(bool gate_enabled);
  ~FusedChain();

  FusedChain(const FusedChain&) = delete;
  FusedChain& operator=(const FusedChain&) = delete;
  FusedChain(FusedChain&&);
  FusedChain& operator=(FusedChain&&);

  // Pure op/dtype check (the gate lives in the constructor argument):
  // a float32/float16 fusable unary or binary elementwise primitive.
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
  // for an empty chain: refusals happen in try_add before acceptance,
  // so an accepted chain always dispatches and the caller never has to
  // re-evaluate carried members individually.
  std::optional<array> evaluate(const Stream& stream);

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

} // namespace mlx::core::omarchy
