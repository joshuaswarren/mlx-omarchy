// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// FuseDecodeChains fused-chain coverage. Fusion is DEFAULT OFF
// (MLX_OMARCHY_FUSED_CHAIN); every equivalence case opts in, and two
// cases pin the gate behavior itself. The fused path must match the
// eager path BIT-EXACT for both float32 and float16: the chain shader
// rounds every intermediate to the storage dtype exactly like the
// per-node path materializes them, and every op formula mirrors
// shaders/elementwise.comp case for case (sigmoid, NaN-propagating
// max/min included). bf16 stays refused by the compiled-tape gate.

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
  set_compile_mode(CompileMode::disabled);
  std::vector<array> eager_outputs = fn(inputs);
  for (auto& out : eager_outputs) {
    out.eval();
  }
  sync_stream(stream);
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

TEST_CASE("gated-in fused chain collapses 3 swiglu dispatches to 1") {
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

  // The 3-op swiglu chain costs 3 eager dispatches and 1 fused one.
  CHECK_EQ(eager, 3);
  CHECK_EQ(fused, 1);
}

TEST_CASE("gate off keeps the per-node dispatch stream") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  unsetenv("MLX_OMARCHY_FUSED_CHAIN");
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
  CHECK_EQ(eager, 3);
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
  CHECK_EQ(eager, 3);
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
  CHECK_EQ(eager, 3);
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
