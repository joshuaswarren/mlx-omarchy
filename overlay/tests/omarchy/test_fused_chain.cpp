// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// FuseDecodeChains fused-chain coverage. Fusion defaults on and
// MLX_OMARCHY_FUSED_CHAIN=0 disables it. Equivalence cases set their intended
// mode explicitly. The fused path must match the
// per-node path BIT-EXACT for float32, float16, and eager bfloat16: the
// chain shader rounds every intermediate to the storage dtype exactly
// like the per-node path materializes them. Compiled bf16 tapes remain
// refused independently.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/compile.h"
#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

using namespace mlx::core;
using mlx::core::omarchy::trace::counters;

namespace {

void skip(const char* reason) {
  std::cout << "Skipping: " << reason << "\n";
}

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
}

bool compute_available() {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device (set MLX_OMARCHY_ALLOW_NON_APPLE=1).");
    return false;
  }
  return true;
}

// The equivalence cases below prove something only when the fused path
// actually runs, so each one opts in explicitly.
void enable_fusion() {
  setenv("MLX_OMARCHY_FUSED_CHAIN", "1", 1);
}

void sync_stream(const Stream& stream) {
  omarchy::get_command_encoder(stream).synchronize();
}

array as_float32(const array& value, const Stream& stream) {
  if (value.dtype() == float32) {
    return value;
  }
  return astype(value, float32, stream);
}

// epsilon > 0: absolute tolerance after widening to float32.
// epsilon == 0: exact float equality (no NaN operands expected).
// epsilon < 0: BITWISE compare of the widened float32 patterns, for
// cases that carry NaN/inf payloads where == would lie.
void check_compiled_matches_eager(
    const std::function<std::vector<array>(const std::vector<array>&)>& fn,
    const std::vector<array>& inputs,
    Dtype dtype,
    const Stream& stream,
    double epsilon,
    bool shapeless = false) {
  setenv("MLX_OMARCHY_FUSED_CHAIN", "0", 1);
  set_compile_mode(CompileMode::disabled);
  std::vector<array> eager_outputs = fn(inputs);
  for (auto& out : eager_outputs) {
    out.eval();
  }
  sync_stream(stream);
  enable_fusion();
  set_compile_mode(CompileMode::enabled);
  auto compiled_fn = compile(fn, shapeless);
  std::vector<array> compiled_outputs = compiled_fn(inputs);
  for (auto& out : compiled_outputs) {
    out.eval();
  }
  sync_stream(stream);
  set_compile_mode(CompileMode::disabled);

  REQUIRE_EQ(eager_outputs.size(), compiled_outputs.size());
  for (size_t j = 0; j < eager_outputs.size(); ++j) {
    REQUIRE_EQ(eager_outputs[j].shape(), compiled_outputs[j].shape());
    array eager32 = as_float32(eager_outputs[j], stream);
    array compiled32 = as_float32(compiled_outputs[j], stream);
    eager32.eval();
    compiled32.eval();
    sync_stream(stream);
    if (epsilon < 0.0) {
      const uint32_t* eager =
          reinterpret_cast<const uint32_t*>(eager32.data<float>());
      const uint32_t* compiled =
          reinterpret_cast<const uint32_t*>(compiled32.data<float>());
      for (size_t index = 0; index < eager32.size(); ++index) {
        INFO("bitwise mismatch at ", index, " eager=0x", std::hex,
             eager[index], " compiled=0x", compiled[index]);
        CHECK_EQ(eager[index], compiled[index]);
      }
    } else {
      const float* eager = eager32.data<float>();
      const float* compiled = compiled32.data<float>();
      for (size_t index = 0; index < eager32.size(); ++index) {
        if (epsilon == 0.0) {
          INFO("exact mismatch at ", index, " eager=", eager[index],
               " compiled=", compiled[index]);
          CHECK_EQ(eager[index], compiled[index]);
        } else {
          double diff = std::abs(
              static_cast<double>(eager[index]) -
              static_cast<double>(compiled[index]));
          INFO("tolerance mismatch at ", index, " eager=", eager[index],
               " compiled=", compiled[index], " diff=", diff);
          CHECK(diff <= epsilon);
        }
      }
    }
  }
}

} // namespace

namespace {

// Pins the compiled 3-op swiglu evaluation to exactly ONE dispatch: a
// silent per-node fallback would also match values in the equivalence
// checks below, so it is ruled out here by count.
void assert_chain_collapses(Dtype dtype, const Stream& stream) {
  std::vector<array> inputs{random::normal(Shape{4, 64}, dtype),
                            random::normal(Shape{4, 64}, dtype)};
  for (auto& in : inputs) {
    in.eval();
  }
  sync_stream(stream);
  set_compile_mode(CompileMode::enabled);
  auto fused_fn = compile([](const std::vector<array>& in) {
    array gate = in[0];
    array up = in[1];
    return std::vector<array>{gate * sigmoid(gate) * up};
  });
  (void)fused_fn(inputs); // warm: traces + compiles
  sync_stream(stream);
  uint64_t before = counters().vk_compute_dispatches.load();
  for (auto& out : fused_fn(inputs)) {
    out.eval();
  }
  sync_stream(stream);
  uint64_t dispatches = counters().vk_compute_dispatches.load() - before;
  set_compile_mode(CompileMode::disabled);
  CHECK_EQ(dispatches, 1);
}

} // namespace

TEST_CASE("fused swiglu-shaped chain matches eager (f32)") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  enable_fusion();
  assert_chain_collapses(float32, stream);
  check_compiled_matches_eager(
      [](const std::vector<array>& in) {
        array gate = in[0];
        array up = in[1];
        return std::vector<array>{gate * sigmoid(gate) * up};
      },
      std::vector<array>{random::normal(Shape{4, 64}, float32),
                         random::normal(Shape{4, 64}, float32)},
      float32,
      stream,
      0.0);
}

TEST_CASE("fused swiglu-shaped chain matches eager (f16, bit-exact)") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  enable_fusion();
  // The chain shader rounds every intermediate to f16 storage exactly
  // like the per-node path materializes them, so the differential check
  // is exact, not a tolerance band.
  assert_chain_collapses(float16, stream);
  check_compiled_matches_eager(
      [](const std::vector<array>& in) {
        array gate = in[0];
        array up = in[1];
        return std::vector<array>{gate * sigmoid(gate) * up};
      },
      std::vector<array>{random::normal(Shape{4, 64}, float16),
                         random::normal(Shape{4, 64}, float16)},
      float16,
      stream,
      0.0);
}

TEST_CASE("chain with tiled row leaf (mod-last) matches eager") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  enable_fusion();
  check_compiled_matches_eager(
      [](const std::vector<array>& in) {
        // scale * (gate * sigmoid(gate)): leaf scale is [K] tiled.
        array gate = in[0];
        array scale = in[1];
        return std::vector<array>{gate * sigmoid(gate) * scale};
      },
      std::vector<array>{random::normal(Shape{4, 64}, float32),
                         random::normal(Shape{64}, float32)},
      float32,
      stream,
      0.0);
}

TEST_CASE("chain with row-broadcast leaf (div-last) matches eager") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  enable_fusion();
  check_compiled_matches_eager(
      [](const std::vector<array>& in) {
        // per-row affine: row [B,1] broadcast against an elementwise run.
        array x = in[0];
        array row = in[1];
        array t = x * sigmoid(x);
        return std::vector<array>{t * row + t};
      },
      std::vector<array>{random::normal(Shape{4, 64}, float32),
                         random::normal(Shape{4, 1}, float32)},
      float32,
      stream,
      0.0);
}

TEST_CASE("chain with scalar leaf matches eager") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  enable_fusion();
  check_compiled_matches_eager(
      [](const std::vector<array>& in) {
        array x = in[0];
        return std::vector<array>{x * sigmoid(x) * x + 1.0f};
      },
      std::vector<array>{random::normal(Shape{8, 32}, float32)},
      float32,
      stream,
      0.0);
}

TEST_CASE("chain with offset (sliced) leaf matches eager") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  enable_fusion();
  // The tape inputs are slices of a larger buffer: contiguous, but with
  // a non-zero storage offset the leaf addressing must carry.
  array base = random::normal(Shape{8, 64}, float32);
  base.eval();
  sync_stream(stream);
  array gate = slice(base, {2, 0}, {6, 64}, {1, 1}, stream);
  array up = slice(base, {4, 0}, {8, 64}, {1, 1}, stream);
  check_compiled_matches_eager(
      [](const std::vector<array>& in) {
        array gate = in[0];
        array up = in[1];
        return std::vector<array>{gate * sigmoid(gate) * up};
      },
      std::vector<array>{gate, up},
      float32,
      stream,
      0.0);
}

TEST_CASE("special values carry through the chain bit-exactly") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  enable_fusion();
  // Covers the two formulas the plain random suites cannot reach: the
  // unguarded sigmoid at exp overflow/underflow magnitudes (a
  // sign-guarded form produces inf/inf NaN here; the per-node path
  // produces 1.0 / 0.0), and NaN-propagating Maximum/Minimum (GLSL
  // max/min would return the non-NaN operand).
  std::vector<float> vals = {
      0.0f,
      1.0f,
      -1.0f,
      20.0f,
      -20.0f,
      100.0f,
      -100.0f,
      88.0f,
      89.0f,
      -88.0f,
      -89.0f,
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
      65504.0f,
      5e-8f};
  array gf = array(vals.begin(), Shape{static_cast<int>(vals.size())}, float32);
  array uf = array(vals.begin(), Shape{static_cast<int>(vals.size())}, float32);
  array gh = astype(gf, float16);
  array uh = astype(uf, float16);
  gf.eval();
  uf.eval();
  gh.eval();
  uh.eval();
  sync_stream(stream);
  auto swiglu = [](const std::vector<array>& in) {
    array gate = in[0];
    array up = in[1];
    return std::vector<array>{gate * sigmoid(gate) * up};
  };
  auto minmax = [](const std::vector<array>& in) {
    // A NaN operand must stay NaN (pattern-checked bitwise), and the
    // max->mul composition must round like per-node.
    array gate = in[0];
    array up = in[1];
    return std::vector<array>{maximum(gate, up) * minimum(gate, up)};
  };
  check_compiled_matches_eager(swiglu, {gf, uf}, float32, stream, -1.0);
  check_compiled_matches_eager(swiglu, {gh, uh}, float16, stream, -1.0);
  check_compiled_matches_eager(minmax, {gf, uf}, float32, stream, -1.0);
  check_compiled_matches_eager(minmax, {gh, uh}, float16, stream, -1.0);
}

TEST_CASE("independent same-shape siblings close and reopen chains") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  enable_fusion();
  // x*x + y*y: the second mul shares no data with the first, so it
  // must be REFUSED as an extension (extensions require the tail's
  // register), closing chain one; it then opens its own chain and the
  // add extends that one. Extending with a no-dependency sibling
  // would strand a non-tail interior member and lose it at close.
  check_compiled_matches_eager(
      [](const std::vector<array>& in) {
        array x = in[0];
        array y = in[1];
        return std::vector<array>{x * x + y * y};
      },
      std::vector<array>{random::normal(Shape{4, 64}, float32),
                         random::normal(Shape{4, 64}, float32)},
      float32,
      stream,
      0.0);
}

TEST_CASE("fused chain kernel strides one forced workgroup") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  enable_fusion();
  // Direct ONE-GROUP dispatch of the fused kernel over MORE elements
  // than one group covers (the elementwise grid-stride test pattern):
  // no large allocations, and every value beyond the first 256 - the
  // wrap region - must match the eager path bit for bit.
  constexpr uint32_t kCount = 600;
  std::vector<float> gv(kCount);
  std::vector<float> uv(kCount);
  for (uint32_t i = 0; i < kCount; ++i) {
    gv[i] = static_cast<float>(i % 7) - 3.0f;
    uv[i] = static_cast<float>(i % 5) - 2.0f;
  }
  array gate = array(gv.begin(), Shape{static_cast<int>(kCount)}, float32);
  array up = array(uv.begin(), Shape{static_cast<int>(kCount)}, float32);
  array reference = gate * sigmoid(gate) * up;
  gate.eval();
  up.eval();
  reference.eval();
  auto& encoder = omarchy::get_command_encoder(stream);
  encoder.synchronize();

  // The swiglu program with two direct leaves:
  // r0 = sigmoid(leaf0); r1 = r0 * leaf0; out = r1 * leaf1.
  std::array<uint32_t, 3> words = {
      5u | (0x10u << 8) | (0x10u << 16) | (0u << 24),
      1u | (0x00u << 8) | (0x10u << 16) | (1u << 24),
      1u | (0x01u << 8) | (0x11u << 16) | (2u << 24)};
  auto program_buffer =
      omarchy::allocator().malloc(words.size() * sizeof(uint32_t));
  auto* program_vk = static_cast<omarchy::VulkanBuffer*>(program_buffer.ptr());
  std::memcpy(program_vk->data, words.data(), words.size() * sizeof(uint32_t));

  array output = zeros({static_cast<int>(kCount)}, float32, stream);
  output.eval();
  encoder.synchronize();
  auto binding = [](const array& value) {
    auto* buffer =
        static_cast<const omarchy::VulkanBuffer*>(value.buffer().ptr());
    return omarchy::ComputeBinding{buffer->buffer, 0, buffer->size};
  };
  omarchy::ComputeParams params;
  params.count = kCount;
  params.operation = static_cast<uint32_t>(words.size());
  params.lhs_size = kCount; // last_dim: direct leaves
  params.rhs_size = 2;      // dst_final register
  params.reduce_size = 0;   // leaf_offset[0]
  params.output_size = 0;   // leaf_offset[1]
  params.lhs_offset = 0;    // leaf_offset[2]
  params.rhs_offset = 0;    // leaf_mode[0] = direct
  params.output_offset = 0; // leaf_mode[1] = direct
  params.aux_size = 0;      // leaf_mode[2] = direct
  std::array<omarchy::ComputeBinding, 5> bindings{
      binding(gate),
      binding(up),
      binding(output),
      omarchy::ComputeBinding{program_vk->buffer, 0, program_vk->size},
      binding(output)};
  // ONE group for 600 elements: each invocation must stride.
  encoder.dispatch_compute(
      omarchy::ComputeKernel::FusedChainF32, bindings, params, 1);
  encoder.synchronize();

  const float* got = output.data<float>();
  const float* want = reference.data<float>();
  for (uint32_t i = 0; i < kCount; ++i) {
    INFO("stride mismatch at ", i, " fused=", got[i], " eager=", want[i]);
    CHECK_EQ(got[i], want[i]);
  }
}

TEST_CASE("residual add chain (tape-input leaf) matches eager") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  enable_fusion();
  check_compiled_matches_eager(
      [](const std::vector<array>& in) {
        array x = in[0];
        return std::vector<array>{x * sigmoid(x) + x};
      },
      std::vector<array>{random::normal(Shape{16, 16}, float32)},
      float32,
      stream,
      0.0);
}

TEST_CASE("shape-changing node closes the chain without breaking values") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  enable_fusion();
  check_compiled_matches_eager(
      [](const std::vector<array>& in) {
        array x = in[0];
        // square changes nothing, but the sum reduction is not fusable:
        // the tape splits around it and both halves must stay correct.
        array t = x * x;
        array s = sum(t, -1, true);
        return std::vector<array>{t * sigmoid(x) + s};
      },
      std::vector<array>{random::normal(Shape{4, 64}, float32)},
      float32,
      stream,
      1e-5);
}

namespace {

// Counts vk dispatches for one more invocation of `step` after `warmups`.
uint64_t counted_dispatches(
    const std::function<array()>& step,
    int warmups,
    const Stream& stream) {
  for (int i = 0; i < warmups; ++i) {
    array out = step();
    out.eval();
  }
  sync_stream(stream);
  uint64_t before = counters().vk_compute_dispatches.load();
  array out = step();
  out.eval();
  sync_stream(stream);
  return counters().vk_compute_dispatches.load() - before;
}

} // namespace

TEST_CASE("gated-in fused chain collapses eager and compiled swiglu") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  enable_fusion();
  std::vector<array> inputs{random::normal(Shape{32, 32}, float32),
                            random::normal(Shape{32, 32}, float32)};
  for (auto& in : inputs) {
    in.eval();
  }
  sync_stream(stream);
  std::function<std::vector<array>(const std::vector<array>&)> fn =
      [](const std::vector<array>& in) {
        array gate = in[0];
        array up = in[1];
        return std::vector<array>{gate * sigmoid(gate) * up};
      };

  set_compile_mode(CompileMode::disabled);
  uint64_t eager = counted_dispatches(
      [&] { return fn(inputs)[0]; }, 1, stream);

  set_compile_mode(CompileMode::enabled);
  auto compiled_fn = compile(fn);
  uint64_t fused = counted_dispatches(
      [&] { return compiled_fn(inputs)[0]; }, 2, stream);
  set_compile_mode(CompileMode::disabled);

  CHECK_EQ(eager, 1);
  CHECK_EQ(fused, 1);
}
TEST_CASE("eager bf16 swiglu is bit-exact and one dispatch") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  set_compile_mode(CompileMode::disabled);
  array gate = astype(random::normal(Shape{32, 32}), bfloat16, stream);
  array up = astype(random::normal(Shape{32, 32}), bfloat16, stream);
  gate.eval();
  up.eval();
  sync_stream(stream);

  setenv("MLX_OMARCHY_FUSED_CHAIN", "0", 1);
  array baseline = gate * sigmoid(gate) * up;
  baseline.eval();
  sync_stream(stream);

  enable_fusion();
  uint64_t before = counters().vk_compute_dispatches.load();
  array candidate = gate * sigmoid(gate) * up;
  candidate.eval();
  sync_stream(stream);
  CHECK_EQ(counters().vk_compute_dispatches.load() - before, 1);

  array baseline32 = astype(baseline, float32, stream);
  array candidate32 = astype(candidate, float32, stream);
  baseline32.eval();
  candidate32.eval();
  sync_stream(stream);
  for (size_t i = 0; i < baseline32.size(); ++i) {
    INFO("bf16 mismatch at ", i);
    CHECK_EQ(baseline32.data<float>()[i], candidate32.data<float>()[i]);
  }
}

TEST_CASE("eager bf16 swiglu matches native intermediate rounding") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  set_compile_mode(CompileMode::disabled);
  array gate = astype(
      array({-8.0f, -6.84375f, -2.0f, -0.5f, 0.0f, 0.5f,
             2.0f, 6.84375f, 8.0f, 0.0f, 2.0f, -2.0f}),
      bfloat16,
      stream);
  array up = astype(
      array({0.5f, -1.0f, 2.0f, -0.25f, 3.0f, -2.0f,
             0.75f, 1.5f, -0.5f, 4.0f, 0.0f, 0.0f}),
      bfloat16,
      stream);
  gate.eval();
  up.eval();
  sync_stream(stream);
  enable_fusion();
  uint64_t before = counters().vk_compute_dispatches.load();
  array output = gate * sigmoid(gate) * up;
  array output32 = astype(output, float32, stream);
  output32.eval();
  sync_stream(stream);
  CHECK_EQ(counters().vk_compute_dispatches.load() - before, 2);

  const std::array<float, 12> native = {
      -0.0013427734375f, 0.00726318359375f, -0.478515625f,
      0.047119140625f, 0.0f, -0.625f, 1.3203125f, 10.25f, -4.0f,
      0.0f, 0.0f, 0.0f};
  size_t mismatches = 0;
  for (size_t i = 0; i < native.size(); ++i) {
    mismatches += output32.data<float>()[i] != native[i];
  }
  CHECK_EQ(mismatches, 1);
  CHECK_EQ(output32.data<float>()[1], 0.00732421875f);
}
TEST_CASE("eager fusion materializes retained intermediate arrays") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  set_compile_mode(CompileMode::disabled);
  array gate = random::normal(Shape{8, 64}, float16, std::nullopt, stream);
  array up = random::normal(Shape{8, 64}, float16, std::nullopt, stream);
  gate.eval();
  up.eval();
  sync_stream(stream);

  setenv("MLX_OMARCHY_FUSED_CHAIN", "0", 1);
  array baseline_sigmoid = sigmoid(gate);
  array baseline_inner = gate * baseline_sigmoid;
  array baseline_out = baseline_inner * up;
  baseline_out.eval();
  sync_stream(stream);

  enable_fusion();
  array candidate_sigmoid = sigmoid(gate);
  array candidate_inner = gate * candidate_sigmoid;
  array candidate_out = candidate_inner * up;
  uint64_t before = counters().vk_compute_dispatches.load();
  candidate_out.eval();
  sync_stream(stream);
  CHECK_EQ(counters().vk_compute_dispatches.load() - before, 1);

  for (auto pair : std::array<std::pair<array, array>, 3>{
           std::pair{baseline_sigmoid, candidate_sigmoid},
           std::pair{baseline_inner, candidate_inner},
           std::pair{baseline_out, candidate_out}}) {
    array baseline32 = astype(pair.first, float32, stream);
    array candidate32 = astype(pair.second, float32, stream);
    baseline32.eval();
    candidate32.eval();
    sync_stream(stream);
    for (size_t i = 0; i < baseline32.size(); ++i) {
      CHECK_EQ(baseline32.data<float>()[i], candidate32.data<float>()[i]);
    }
  }
}

TEST_CASE("gate off keeps the per-node dispatch stream") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  setenv("MLX_OMARCHY_FUSED_CHAIN", "0", 1);
  std::vector<array> inputs{random::normal(Shape{32, 32}, float32),
                            random::normal(Shape{32, 32}, float32)};
  for (auto& in : inputs) {
    in.eval();
  }
  sync_stream(stream);
  std::function<std::vector<array>(const std::vector<array>&)> fn =
      [](const std::vector<array>& in) {
        array gate = in[0];
        array up = in[1];
        return std::vector<array>{gate * sigmoid(gate) * up};
      };

  set_compile_mode(CompileMode::disabled);
  uint64_t eager = counted_dispatches(
      [&] { return fn(inputs)[0]; }, 1, stream);

  set_compile_mode(CompileMode::enabled);
  auto compiled_fn = compile(fn);
  uint64_t per_node = counted_dispatches(
      [&] { return compiled_fn(inputs)[0]; }, 2, stream);
  set_compile_mode(CompileMode::disabled);

  // Default-off is byte-for-byte the pre-chain tape: same dispatch
  // count, same stream.
  CHECK_EQ(per_node, eager);
}

namespace {

std::string evaluation_error(array value) {
  try {
    value.eval();
  } catch (const std::exception& error) {
    return error.what();
  }
  return {};
}

} // namespace

TEST_CASE("bf16 compiled tape stays refused") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  enable_fusion();
  set_compile_mode(CompileMode::enabled);
  // A single-node graph never forms a Compiled tape; the swiglu chain
  // does, so the gate fires on the fragment itself. Compiled calls are
  // lazy: the refusal throws at eval time, not call time.
  auto fn = compile([](const std::vector<array>& in) {
    return std::vector<array>{in[0] * sigmoid(in[0]) * in[1]};
  });
  array a = astype(random::normal(Shape{8, 8}), bfloat16);
  array b = astype(random::normal(Shape{8, 8}), bfloat16);
  a.eval();
  b.eval();
  sync_stream(stream);
  std::string error = evaluation_error(fn({a, b})[0]);
  CHECK(error.find("[omarchy] Compiled tape bfloat16") != std::string::npos);
  set_compile_mode(CompileMode::disabled);
}

// The model fragment: mlx_lm compiles swiglu with shapeless=True, and a
// shapeless trace keeps the broadcast_arrays identity pairs (including the
// stop-gradient operand copies). Before identity-broadcast normalization
// that tape walked as three one-instruction chains; it must cost exactly
// one three-instruction dispatch like the non-shapeless tape does.
TEST_CASE("shapeless swiglu fragment collapses identity broadcast pairs (f32)") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  enable_fusion();
  std::vector<array> inputs{random::normal(Shape{4, 64}, float32),
                            random::normal(Shape{4, 64}, float32)};
  for (auto& in : inputs) {
    in.eval();
  }
  sync_stream(stream);
  std::function<std::vector<array>(const std::vector<array>&)> fn =
      [](const std::vector<array>& in) {
        return std::vector<array>{in[0] * sigmoid(in[0]) * in[1]};
      };

  set_compile_mode(CompileMode::disabled);
  uint64_t eager = counted_dispatches(
      [&] { return fn(inputs)[0]; }, 1, stream);
  set_compile_mode(CompileMode::enabled);
  auto compiled_fn = compile(fn, /*shapeless=*/true);
  uint64_t fused = counted_dispatches(
      [&] { return compiled_fn(inputs)[0]; }, 2, stream);
  set_compile_mode(CompileMode::disabled);
  CHECK_EQ(eager, 1);
  CHECK_EQ(fused, 1);
  check_compiled_matches_eager(fn, inputs, float32, stream, 0.0, true);
}

TEST_CASE("shapeless swiglu fragment collapses identity broadcast pairs (f16)") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  enable_fusion();
  std::vector<array> inputs{random::normal(Shape{4, 64}, float16),
                            random::normal(Shape{4, 64}, float16)};
  for (auto& in : inputs) {
    in.eval();
  }
  sync_stream(stream);
  std::function<std::vector<array>(const std::vector<array>&)> fn =
      [](const std::vector<array>& in) {
        return std::vector<array>{in[0] * sigmoid(in[0]) * in[1]};
      };

  set_compile_mode(CompileMode::disabled);
  uint64_t eager = counted_dispatches(
      [&] { return fn(inputs)[0]; }, 1, stream);
  set_compile_mode(CompileMode::enabled);
  auto compiled_fn = compile(fn, /*shapeless=*/true);
  uint64_t fused = counted_dispatches(
      [&] { return compiled_fn(inputs)[0]; }, 2, stream);
  set_compile_mode(CompileMode::disabled);
  CHECK_EQ(eager, 1);
  CHECK_EQ(fused, 1);
  check_compiled_matches_eager(fn, inputs, float16, stream, -1.0, true);
}

TEST_CASE("shapeless fragment serves new shapes with one dispatch each") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  enable_fusion();
  std::function<std::vector<array>(const std::vector<array>&)> fn =
      [](const std::vector<array>& in) {
        return std::vector<array>{in[0] * sigmoid(in[0]) * in[1]};
      };
  set_compile_mode(CompileMode::enabled);
  auto compiled_fn = compile(fn, /*shapeless=*/true);
  std::vector<array> warm{random::normal(Shape{4, 64}, float32),
                          random::normal(Shape{4, 64}, float32)};
  for (auto& in : warm) {
    in.eval();
  }
  array warm_out = compiled_fn(warm)[0];
  warm_out.eval();
  sync_stream(stream);
  for (Shape shape : {Shape{2, 64}, Shape{1, 64}}) {
    std::vector<array> in{random::normal(shape, float32),
                          random::normal(shape, float32)};
    for (auto& a : in) {
      a.eval();
    }
    set_compile_mode(CompileMode::disabled);
    array want = fn(in)[0];
    want.eval();
    sync_stream(stream);
    set_compile_mode(CompileMode::enabled);
    uint64_t fused = counted_dispatches(
        [&] { return compiled_fn(in)[0]; }, 0, stream);
    set_compile_mode(CompileMode::disabled);
    array got = compiled_fn(in)[0];
    got.eval();
    sync_stream(stream);
    CHECK_EQ(fused, 1);
    array got32 = as_float32(got, stream);
    array want32 = as_float32(want, stream);
    got32.eval();
    want32.eval();
    sync_stream(stream);
    const uint32_t* got_words =
        reinterpret_cast<const uint32_t*>(got32.data<float>());
    const uint32_t* want_words =
        reinterpret_cast<const uint32_t*>(want32.data<float>());
    for (size_t index = 0; index < want32.size(); ++index) {
      INFO("shape-reuse mismatch at ", index);
      CHECK_EQ(got_words[index], want_words[index]);
    }
  }
  set_compile_mode(CompileMode::disabled);
}

TEST_CASE("broadcast identity is reclassified per eval shape") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  enable_fusion();
  // Traced with row [1,64] against gate [1,64], every broadcast pair is an
  // identity at the trace shape. Serving gate [4,64] with row [1,64] makes
  // the row-side pair a REAL broadcast again: the walk must reclassify per
  // evaluation, fuse only the identity prefix, and drop the rest to the
  // per-node view path.
  std::function<std::vector<array>(const std::vector<array>&)> fn =
      [](const std::vector<array>& in) {
        return std::vector<array>{in[0] * sigmoid(in[0]) * in[1]};
      };
  set_compile_mode(CompileMode::enabled);
  std::vector<array> trace_in{random::normal(Shape{1, 64}, float32),
                              random::normal(Shape{1, 64}, float32)};
  for (auto& in : trace_in) {
    in.eval();
  }
  auto compiled_fn = compile(fn, /*shapeless=*/true);
  uint64_t identity = counted_dispatches(
      [&] { return compiled_fn(trace_in)[0]; }, 2, stream);
  CHECK_EQ(identity, 1);

  std::vector<array> grown{random::normal(Shape{4, 64}, float32),
                           random::normal(Shape{1, 64}, float32)};
  for (auto& in : grown) {
    in.eval();
  }
  set_compile_mode(CompileMode::disabled);
  array want = fn(grown)[0];
  want.eval();
  sync_stream(stream);
  set_compile_mode(CompileMode::enabled);
  uint64_t mixed = counted_dispatches(
      [&] { return compiled_fn(grown)[0]; }, 0, stream);
  set_compile_mode(CompileMode::disabled);
  CHECK_EQ(mixed, 2);
  array got = compiled_fn(grown)[0];
  got.eval();
  sync_stream(stream);
  array got32 = as_float32(got, stream);
  array want32 = as_float32(want, stream);
  got32.eval();
  want32.eval();
  sync_stream(stream);
  const uint32_t* got_words =
      reinterpret_cast<const uint32_t*>(got32.data<float>());
  const uint32_t* want_words =
      reinterpret_cast<const uint32_t*>(want32.data<float>());
  for (size_t index = 0; index < want32.size(); ++index) {
    INFO("reclassified mismatch at ", index);
    CHECK_EQ(got_words[index], want_words[index]);
  }
}

TEST_CASE("identity broadcast as tape output resolves through the alias") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  enable_fusion();
  std::vector<array> inputs{random::normal(Shape{4, 64}, float32),
                            random::normal(Shape{4, 64}, float32)};
  for (auto& in : inputs) {
    in.eval();
  }
  sync_stream(stream);
  std::function<std::vector<array>(const std::vector<array>&)> fn =
      [](const std::vector<array>& in) {
        array t = in[0] * sigmoid(in[0]);
        auto outs = broadcast_arrays({t, in[1]});
        return std::vector<array>{t, outs[0]};
      };
  set_compile_mode(CompileMode::disabled);
  uint64_t eager = counted_dispatches(
      [&] { return fn(inputs)[1]; }, 1, stream);
  set_compile_mode(CompileMode::enabled);
  auto compiled_fn = compile(fn, /*shapeless=*/true);
  uint64_t fused = counted_dispatches(
      [&] { return compiled_fn(inputs)[1]; }, 2, stream);
  set_compile_mode(CompileMode::disabled);
  // The second tape output is an identity broadcast: it resolves through
  // the alias to its source, dispatches nothing, and the fragment's one
  // chain carries the two compute nodes.
  CHECK_EQ(eager, 2);
  CHECK_EQ(fused, 1);
  check_compiled_matches_eager(fn, inputs, float32, stream, 0.0, true);
}

TEST_CASE("identity broadcast as tape output resolves through the alias") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  enable_fusion();
  std::vector<array> inputs{random::normal(Shape{4, 64}, float32),
                            random::normal(Shape{4, 64}, float32)};
  for (auto& in : inputs) {
    in.eval();
  }
  sync_stream(stream);
  std::function<std::vector<array>(const std::vector<array>&)> fn =
      [](const std::vector<array>& in) {
        auto outs = broadcast_arrays({in[0], in[1]});
        return std::vector<array>{outs[0]};
      };
  set_compile_mode(CompileMode::disabled);
  uint64_t eager = counted_dispatches(
      [&] { return fn(inputs)[0]; }, 1, stream);
  set_compile_mode(CompileMode::enabled);
  auto compiled_fn = compile(fn, /*shapeless=*/true);
  uint64_t fused = counted_dispatches(
      [&] { return compiled_fn(inputs)[0]; }, 2, stream);
  set_compile_mode(CompileMode::disabled);
  // The tape's only node is an identity broadcast and the tape output is
  // that node itself: nothing dispatches, and the output resolves through
  // the alias to the input's buffer.
  CHECK_EQ(eager, 0);
  CHECK_EQ(fused, 0);
  check_compiled_matches_eager(fn, inputs, float32, stream, 0.0, true);
}

TEST_CASE("nonidentity broadcast keeps the per-node fallback (shapeless)") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  enable_fusion();
  std::vector<array> inputs{random::normal(Shape{4, 64}, float32),
                            random::normal(Shape{64}, float32)};
  for (auto& in : inputs) {
    in.eval();
  }
  sync_stream(stream);
  std::function<std::vector<array>(const std::vector<array>&)> fn =
      [](const std::vector<array>& in) {
        return std::vector<array>{in[0] * sigmoid(in[0]) * in[1]};
      };
  set_compile_mode(CompileMode::disabled);
  uint64_t eager = counted_dispatches(
      [&] { return fn(inputs)[0]; }, 1, stream);
  set_compile_mode(CompileMode::enabled);
  auto compiled_fn = compile(fn, /*shapeless=*/true);
  uint64_t fused = counted_dispatches(
      [&] { return compiled_fn(inputs)[0]; }, 2, stream);
  set_compile_mode(CompileMode::disabled);
  // The scale-side broadcast is real ([64] to [4,64]): the identity prefix
  // still fuses, the scale side keeps the per-node view, and the tail mul
  // refuses the non-contiguous leaf.
  CHECK_EQ(eager, 1);
  CHECK_EQ(fused, 2);
  check_compiled_matches_eager(fn, inputs, float32, stream, 0.0, true);
}

TEST_CASE("incompatible call-time shapes refuse like eager (shapeless)") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  enable_fusion();
  std::function<std::vector<array>(const std::vector<array>&)> fn =
      [](const std::vector<array>& in) {
        array t = in[0] * sigmoid(in[0]);
        auto outs = broadcast_arrays({t, in[1]});
        return std::vector<array>{t, outs[0]};
      };
  std::vector<array> trace_in{random::normal(Shape{4, 64}, float32),
                              random::normal(Shape{4, 64}, float32)};
  for (auto& in : trace_in) {
    in.eval();
  }
  sync_stream(stream);
  set_compile_mode(CompileMode::enabled);
  auto compiled_fn = compile(fn, /*shapeless=*/true);
  for (auto& out : compiled_fn(trace_in)) {
    out.eval();
  }
  sync_stream(stream);
  set_compile_mode(CompileMode::disabled);

  std::vector<array> bad{random::normal(Shape{4, 64}, float32),
                         random::normal(Shape{8, 64}, float32)};
  for (auto& in : bad) {
    in.eval();
  }
  sync_stream(stream);
  std::string eager_error;
  try {
    auto outs = fn(bad);
    for (auto& out : outs) {
      out.eval();
    }
  } catch (const std::exception& e) {
    eager_error = e.what();
  }
  CHECK(eager_error.find("[broadcast_shapes]") != std::string::npos);

  set_compile_mode(CompileMode::enabled);
  std::string compiled_error;
  try {
    auto outs = compiled_fn(bad);
    for (auto& out : outs) {
      out.eval();
    }
    sync_stream(stream);
  } catch (const std::exception& e) {
    compiled_error = e.what();
  }
  set_compile_mode(CompileMode::disabled);
  CHECK(compiled_error.find("[broadcast_shapes]") != std::string::npos);
  INFO("eager: ", eager_error, " | compiled: ", compiled_error);
  CHECK_EQ(eager_error, compiled_error);
}

TEST_CASE("zero-sized inputs stay correct end to end (shapeless)") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  enable_fusion();
  std::function<std::vector<array>(const std::vector<array>&)> fn =
      [](const std::vector<array>& in) {
        return std::vector<array>{in[0] * sigmoid(in[0]) * in[1]};
      };
  std::vector<array> inputs{zeros(Shape{0, 64}, float32),
                            zeros(Shape{0, 64}, float32)};
  for (auto& in : inputs) {
    in.eval();
  }
  sync_stream(stream);
  set_compile_mode(CompileMode::enabled);
  auto compiled_fn = compile(fn, /*shapeless=*/true);
  auto outs = compiled_fn(inputs);
  CHECK_EQ(outs[0].shape(), Shape{0, 64});
  for (auto& out : outs) {
    out.eval();
  }
  sync_stream(stream);
  set_compile_mode(CompileMode::disabled);
  array want = fn(inputs)[0];
  want.eval();
  sync_stream(stream);
  CHECK_EQ(outs[0].shape(), want.shape());
}

// DecodeFusion: eager Q4 decode GEMV groups (fused_chain.h
// GemvFusionMember, primitives.cpp dispatch_quantized_gemv_group). The
// fused dispatch must be bit-exact against the per-node path for every
// member output and every folded Add, in every dtype the kernel ships.
namespace {

struct QuantizedLinear {
  array w = zeros({1});
  array scales = zeros({1});
  array biases = zeros({1});
  array bias = zeros({1});
};

QuantizedLinear make_linear(int n, int k, Dtype dtype, const Stream& stream) {
  QuantizedLinear linear;
  // Affine quantize runs in float32 here (the GPU quantizer refuses
  // bf16 input); the packed words are dtype-free and the parameters
  // cast to the leg's dtype like a converted checkpoint.
  array w = multiply(
      random::normal(Shape{n, k}, float32, std::nullopt, stream),
      array(0.05f), stream);
  auto parts = quantize(w, 64, 4, "affine", std::nullopt, stream);
  linear.w = parts[0];
  linear.scales = astype(parts[1], dtype, stream);
  linear.biases = astype(parts[2], dtype, stream);
  linear.bias = astype(
      random::normal(Shape{n}, float32, std::nullopt, stream), dtype, stream);
  for (array* a : {&linear.w, &linear.scales, &linear.biases, &linear.bias}) {
    a->eval();
  }
  sync_stream(stream);
  return linear;
}

array project(const array& x, const QuantizedLinear& l, const Stream& s) {
  return quantized_matmul(x, l.w, l.scales, l.biases, true, 64, 4, "affine", s);
}

void expect_bit_exact(const array& a, const array& b, const Stream& stream) {
  array a32 = astype(a, float32, stream);
  array b32 = astype(b, float32, stream);
  a32.eval();
  b32.eval();
  sync_stream(stream);
  REQUIRE_EQ(a32.size(), b32.size());
  for (size_t i = 0; i < a32.size(); ++i) {
    INFO("mismatch at ", i);
    CHECK_EQ(a32.data<float>()[i], b32.data<float>()[i]);
  }
}

} // namespace

TEST_CASE("eager q4 decode gemv group folds q/k/v and their biases") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  enable_fusion();
  set_compile_mode(CompileMode::disabled);
  for (Dtype dtype : {float16, bfloat16, float32}) {
    const int k = 896;
    auto q = make_linear(896, k, dtype, stream);
    auto kk = make_linear(128, k, dtype, stream);
    auto v = make_linear(128, k, dtype, stream);
    array x = astype(
        random::normal(Shape{1, 1, k}, float32, std::nullopt, stream), dtype,
        stream);
    x.eval();
    sync_stream(stream);
    auto forward = [&] {
      array q_raw = project(x, q, stream);
      array k_raw = project(x, kk, stream);
      array v_raw = project(x, v, stream);
      return std::vector<array>{
          q_raw, k_raw, v_raw, add(q_raw, q.bias, stream),
          add(k_raw, kk.bias, stream), add(v_raw, v.bias, stream)};
    };

    setenv("MLX_OMARCHY_FUSED_GEMV", "0", 1);
    auto baseline = forward();
    // Evaluate the three biased outputs; the raw ones ride along.
    eval({baseline[3], baseline[4], baseline[5]});
    sync_stream(stream);

    setenv("MLX_OMARCHY_FUSED_GEMV", "1", 1);
    auto candidate = forward();
    uint64_t before = counters().vk_compute_dispatches.load();
    eval({candidate[3], candidate[4], candidate[5]});
    sync_stream(stream);
    INFO("dtype ", dtype);
    CHECK_EQ(counters().vk_compute_dispatches.load() - before, 1);
    for (size_t i = 0; i < baseline.size(); ++i) {
      INFO("output ", i);
      expect_bit_exact(baseline[i], candidate[i], stream);
    }
  }
  unsetenv("MLX_OMARCHY_FUSED_GEMV");
}

TEST_CASE("eager q4 decode gemv group: gate/up pair and residual fold") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  enable_fusion();
  set_compile_mode(CompileMode::disabled);
  const int k = 896;
  auto gate = make_linear(4864, k, float16, stream);
  auto up = make_linear(4864, k, float16, stream);
  auto down = make_linear(k, 4864, float16, stream);
  array h = astype(
      random::normal(Shape{1, 1, k}, float32, std::nullopt, stream), float16,
      stream);
  h.eval();
  sync_stream(stream);
  // mlx-lm's block tail: gate/up share x, SwiGLU, down, residual add
  // against the tape-computed residual stream h2 (an ancestor of the
  // GEMV, so it is evaluated before the group dispatches). Per node: 2 gemv + 1 swiglu + 1 gemv +
  // 1 multiply + 1 add = 6; fused: gate/up (1) + swiglu (1) + down with
  // the residual folded (1) + multiply (1) = 4.
  auto forward = [&] {
    array h2 = multiply(h, array(2.0f, float16), stream);
    array x = h2;
    array g = project(x, gate, stream);
    array u = project(x, up, stream);
    array act = multiply(multiply(g, sigmoid(g, stream), stream), u, stream);
    array d = project(act, down, stream);
    return std::vector<array>{g, u, d, add(h2, d, stream)};
  };
  setenv("MLX_OMARCHY_FUSED_GEMV", "0", 1);
  auto baseline = forward();
  uint64_t before = counters().vk_compute_dispatches.load();
  eval({baseline[3]});
  sync_stream(stream);
  uint64_t per_node = counters().vk_compute_dispatches.load() - before;

  setenv("MLX_OMARCHY_FUSED_GEMV", "1", 1);
  auto candidate = forward();
  before = counters().vk_compute_dispatches.load();
  eval({candidate[3]});
  sync_stream(stream);
  uint64_t fused = counters().vk_compute_dispatches.load() - before;
  CHECK_EQ(per_node, 6);
  CHECK_EQ(fused, 4);
  for (size_t i = 0; i < baseline.size(); ++i) {
    INFO("output ", i);
    expect_bit_exact(baseline[i], candidate[i], stream);
  }
  unsetenv("MLX_OMARCHY_FUSED_GEMV");
}

TEST_CASE("eager q4 decode gemv group refuses an addend it has not computed") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  enable_fusion();
  set_compile_mode(CompileMode::disabled);
  const int k = 896;
  auto a = make_linear(256, k, float16, stream);
  auto b = make_linear(256, k, float16, stream);
  array x = astype(
      random::normal(Shape{1, 1, k}, float32, std::nullopt, stream), float16,
      stream);
  x.eval();
  sync_stream(stream);
  // y_b is y_a's only consumer's other operand and a member of the same
  // group: unscheduled when the group would dispatch, so the whole
  // group falls back to the per-node path (3 dispatches, same values).
  auto forward = [&] {
    array y_a = project(x, a, stream);
    array y_b = project(x, b, stream);
    return std::vector<array>{y_b, add(y_a, y_b, stream)};
  };
  setenv("MLX_OMARCHY_FUSED_GEMV", "0", 1);
  auto baseline = forward();
  eval({baseline[1]});
  sync_stream(stream);
  setenv("MLX_OMARCHY_FUSED_GEMV", "1", 1);
  auto candidate = forward();
  uint64_t before = counters().vk_compute_dispatches.load();
  eval({candidate[1]});
  sync_stream(stream);
  CHECK_EQ(counters().vk_compute_dispatches.load() - before, 3);
  expect_bit_exact(baseline[0], candidate[0], stream);
  expect_bit_exact(baseline[1], candidate[1], stream);
  unsetenv("MLX_OMARCHY_FUSED_GEMV");
}
