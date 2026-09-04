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

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
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
    double epsilon) {
  set_compile_mode(CompileMode::disabled);
  std::vector<array> eager_outputs = fn(inputs);
  for (auto& out : eager_outputs) {
    out.eval();
  }
  sync_stream(stream);

  set_compile_mode(CompileMode::enabled);
  auto compiled_fn = compile(fn);
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
