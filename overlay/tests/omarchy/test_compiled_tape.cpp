// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Wave 11 compiled-tape coverage: every upstream-fusable op class that the
// Omarchy tape interpreter admits must match its eager path. Equivalence
// tolerances are stated per case: float32 compares compiled against eager
// at 1e-6, float16 at 2e-3 after both sides are widened to float32, and
// integer, bitwise, boolean, and select classes compare exactly.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <complex>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/compile.h"
#include "mlx/device.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

using namespace mlx::core;

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
    skip(
        "no qualifying Vulkan device (set MLX_OMARCHY_ALLOW_NON_APPLE=1 on"
        " a development machine).");
    return false;
  }
  return true;
}

void sync_stream(const Stream& stream) {
  omarchy::get_command_encoder(stream).synchronize();
}

// Widens an array to float32 on the host path so non-f32 buffers can be
// read through one comparison loop.
array as_float32(const array& value, const Stream& stream) {
  if (value.dtype() == float32) {
    return value;
  }
  return astype(value, float32, stream);
}

// Compiled-versus-eager equivalence. Both sides run the same fn, once with
// compilation disabled and once fused, and the buffers must agree within
// the stated tolerance. epsilon == 0 means exact (integer and boolean).
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
    if (dtype == bool_ && eager_outputs[j].dtype() == bool_) {
      const bool* eager = eager_outputs[j].data<bool>();
      const bool* compiled = compiled_outputs[j].data<bool>();
      for (size_t index = 0; index < eager_outputs[j].size(); ++index) {
        INFO("bool mismatch at ", index, " eager=",
             static_cast<int>(eager[index]), " compiled=",
             static_cast<int>(compiled[index]));
        CHECK_EQ(static_cast<int>(eager[index]),
                 static_cast<int>(compiled[index]));
      }
      continue;
    }
    if (dtype == int32 && eager_outputs[j].dtype() == int32) {
      const int32_t* eager = eager_outputs[j].data<int32_t>();
      const int32_t* compiled = compiled_outputs[j].data<int32_t>();
      for (size_t index = 0; index < eager_outputs[j].size(); ++index) {
        INFO("int32 mismatch at ", index, " eager=", eager[index],
             " compiled=", compiled[index]);
        CHECK_EQ(eager[index], compiled[index]);
      }
      continue;
    }
    if (dtype == uint32 && eager_outputs[j].dtype() == uint32) {
      const uint32_t* eager = eager_outputs[j].data<uint32_t>();
      const uint32_t* compiled = compiled_outputs[j].data<uint32_t>();
      for (size_t index = 0; index < eager_outputs[j].size(); ++index) {
        INFO("uint32 mismatch at ", index, " eager=", eager[index],
             " compiled=", compiled[index]);
        CHECK_EQ(eager[index], compiled[index]);
      }
      continue;
    }
    if (dtype == float32) {
      const float* eager = eager_outputs[j].data<float>();
      const float* compiled = compiled_outputs[j].data<float>();
      for (size_t index = 0; index < eager_outputs[j].size(); ++index) {
        INFO("f32 mismatch at ", index, " eager=", eager[index],
             " compiled=", compiled[index]);
        CHECK(
            compiled[index] == doctest::Approx(eager[index]).epsilon(epsilon));
      }
    } else {
      array eager_wide = as_float32(eager_outputs[j], stream);
      array compiled_wide = as_float32(compiled_outputs[j], stream);
      eager_wide.eval();
      compiled_wide.eval();
      sync_stream(stream);
      const float* eager = eager_wide.data<float>();
      const float* compiled = compiled_wide.data<float>();
      for (size_t index = 0; index < eager_outputs[j].size(); ++index) {
        bool same =
            compiled[index] == doctest::Approx(eager[index]).epsilon(epsilon);
        if (!same) {
          std::printf(
              "WIDE MISMATCH idx=%zu eager=%.10g compiled=%.10g\n",
              index,
              (double)eager[index],
              (double)compiled[index]);
        }
        CHECK(same);
      }
    }
  }
}
std::string evaluation_error(array value) {
  try {
    value.eval();
  } catch (const std::exception& error) {
    return error.what();
  }
  return {};
}

using UnaryFn = std::function<array(const array&, const Stream&)>;
using BinaryFn =
    std::function<array(const array&, const array&, const Stream&)>;

} // namespace

TEST_CASE("compiled tape matches eager for every float unary class") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  struct UnaryCase {
    std::string name;
    std::vector<float> input;
    UnaryFn op;
  };

  std::vector<UnaryCase> cases = {
      {"abs",
       {0.2f, -0.5f, 1.7f, -3.4f},
       [](const array& a, const Stream& s) { return abs(a, s); }},
      {"arcsin",
       {-0.3f, 0.0f, 0.25f, 0.9f},
       [](const array& a, const Stream& s) { return arcsin(a, s); }},
      {"arccos",
       {-0.3f, 0.0f, 0.25f, 0.9f},
       [](const array& a, const Stream& s) { return arccos(a, s); }},
      {"arctan",
       {-1.5f, 0.4f, 2.2f, -0.8f},
       [](const array& a, const Stream& s) { return arctan(a, s); }},
      {"arcsinh",
       {-1.5f, 0.4f, 2.2f, -0.8f},
       [](const array& a, const Stream& s) { return arcsinh(a, s); }},
      {"arccosh",
       {1.05f, 1.5f, 3.0f, 7.25f},
       [](const array& a, const Stream& s) { return arccosh(a, s); }},
      {"arctanh",
       {-0.9f, 0.1f, 0.45f, -0.2f},
       [](const array& a, const Stream& s) { return arctanh(a, s); }},
      {"ceil",
       {-1.7f, -0.2f, 0.3f, 2.9f},
       [](const array& a, const Stream& s) { return ceil(a, s); }},
      {"cos",
       {0.1f, -0.4f, 1.1f, 2.5f},
       [](const array& a, const Stream& s) { return cos(a, s); }},
      {"cosh",
       {0.1f, -0.4f, 1.1f, 2.5f},
       [](const array& a, const Stream& s) { return cosh(a, s); }},
      {"erf",
       {-1.2f, 0.0f, 0.5f, 2.0f},
       [](const array& a, const Stream& s) { return erf(a, s); }},
      {"erfinv",
       {-0.9f, -0.1f, 0.1f, 0.9f},
       [](const array& a, const Stream& s) { return erfinv(a, s); }},
      {"expm1",
       {-1.2f, 0.0f, 0.5f, 2.0f},
       [](const array& a, const Stream& s) { return expm1(a, s); }},
      {"floor",
       {-1.7f, -0.2f, 0.3f, 2.9f},
       [](const array& a, const Stream& s) { return floor(a, s); }},
      {"log",
       {0.25f, 0.5f, 1.0f, 4.0f},
       [](const array& a, const Stream& s) { return log(a, s); }},
      {"log1p",
       {0.25f, 0.5f, 1.0f, 4.0f},
       [](const array& a, const Stream& s) { return log1p(a, s); }},
      {"round",
       {-1.6f, -0.5f, 0.4f, 2.5f},
       [](const array& a, const Stream& s) { return round(a, s); }},
      {"sigmoid",
       {-1.2f, 0.0f, 0.5f, 2.0f},
       [](const array& a, const Stream& s) { return sigmoid(a, s); }},
      {"sign",
       {-3.2f, -0.1f, 0.0f, 5.5f},
       [](const array& a, const Stream& s) { return sign(a, s); }},
      {"sin",
       {0.1f, -0.4f, 1.1f, 2.5f},
       [](const array& a, const Stream& s) { return sin(a, s); }},
      {"sinh",
       {0.1f, -0.4f, 1.1f, 2.5f},
       [](const array& a, const Stream& s) { return sinh(a, s); }},
      {"tan",
       {0.1f, -0.4f, 0.9f, -1.2f},
       [](const array& a, const Stream& s) { return tan(a, s); }},
      {"tanh",
       {0.1f, -0.4f, 1.1f, 2.5f},
       [](const array& a, const Stream& s) { return tanh(a, s); }},
      {"negative",
       {0.2f, -0.5f, 1.7f, -3.4f},
       [](const array& a, const Stream& s) { return negative(a, s); }},
      {"square",
       {0.2f, -0.5f, 1.7f, -3.4f},
       [](const array& a, const Stream& s) { return square(a, s); }},
      {"sqrt",
       {0.25f, 0.5f, 1.0f, 4.0f},
       [](const array& a, const Stream& s) { return sqrt(a, s); }},
      {"exp",
       {-1.2f, 0.0f, 0.5f, 2.0f},
       [](const array& a, const Stream& s) { return exp(a, s); }},
  };

  for (const auto& unary : cases) {
    INFO("unary class: ", unary.name);
    array x(unary.input.begin(), Shape{4}, float32);
    auto fn = [&unary, &stream](const std::vector<array>& inputs) {
      // Two fusable nodes per graph so the tracer builds a tape.
      return std::vector<array>{add(unary.op(inputs[0], stream), array(1.0f), stream)};
    };
    check_compiled_matches_eager(fn, {x}, float32, stream, 1e-6);
  }
}

TEST_CASE("compiled tape matches eager for every float binary class") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  struct BinaryCase {
    std::string name;
    std::vector<float> lhs;
    std::vector<float> rhs;
    BinaryFn op;
  };

  std::vector<BinaryCase> cases = {
      {"add",
       {0.2f, -0.5f, 1.7f, -3.4f},
       {1.0f, 0.25f, -2.0f, 0.5f},
       [](const array& a, const array& b, const Stream& s) {
         return add(a, b, s);
       }},
      {"subtract",
       {0.2f, -0.5f, 1.7f, -3.4f},
       {1.0f, 0.25f, -2.0f, 0.5f},
       [](const array& a, const array& b, const Stream& s) {
         return subtract(a, b, s);
       }},
      {"multiply",
       {0.2f, -0.5f, 1.7f, -3.4f},
       {1.0f, 0.25f, -2.0f, 0.5f},
       [](const array& a, const array& b, const Stream& s) {
         return multiply(a, b, s);
       }},
      {"divide",
       {0.2f, -0.5f, 1.7f, -3.4f},
       {1.0f, 0.25f, -2.0f, 0.5f},
       [](const array& a, const array& b, const Stream& s) {
         return divide(a, b, s);
       }},
      {"maximum",
       {0.2f, -0.5f, 1.7f, -3.4f},
       {1.0f, 0.25f, -2.0f, 0.5f},
       [](const array& a, const array& b, const Stream& s) {
         return maximum(a, b, s);
       }},
      {"minimum",
       {0.2f, -0.5f, 1.7f, -3.4f},
       {1.0f, 0.25f, -2.0f, 0.5f},
       [](const array& a, const array& b, const Stream& s) {
         return minimum(a, b, s);
       }},
      {"power",
       {0.5f, 1.5f, 2.0f, -0.5f},
       {2.0f, 3.0f, 0.5f, 3.0f},
       [](const array& a, const array& b, const Stream& s) {
         return power(a, b, s);
       }},
      {"logaddexp",
       {0.2f, -0.5f, 1.7f, -3.4f},
       {1.0f, 0.25f, -2.0f, 0.5f},
       [](const array& a, const array& b, const Stream& s) {
         return logaddexp(a, b, s);
       }},
      {"arctan2",
       {0.2f, -0.5f, 1.7f, -3.4f},
       {1.0f, 0.25f, -2.0f, 0.5f},
       [](const array& a, const array& b, const Stream& s) {
         return arctan2(a, b, s);
       }},
      {"remainder",
       {5.5f, -6.25f, 7.0f, -3.5f},
       {2.0f, 1.5f, 3.0f, 2.0f},
       [](const array& a, const array& b, const Stream& s) {
         return remainder(a, b, s);
       }},
  };

  for (const auto& binary : cases) {
    INFO("binary class: ", binary.name);
    array x(binary.lhs.begin(), Shape{4}, float32);
    array y(binary.rhs.begin(), Shape{4}, float32);
    auto fn = [&binary, &stream](const std::vector<array>& inputs) {
      return std::vector<array>{
          add(binary.op(inputs[0], inputs[1], stream), array(1.0f), stream)};
    };
    check_compiled_matches_eager(fn, {x, y}, float32, stream, 1e-6);
  }
}

TEST_CASE("compiled tape matches eager for integer and bitwise classes") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<int32_t> iv = {17, -5, 1024, -4096};
  std::vector<int32_t> jv = {3, 2, 7, 4095};
  std::vector<uint32_t> uv = {12, 345678, 7, 4294967290u};
  array x(iv.begin(), Shape{4}, int32);
  array y(jv.begin(), Shape{4}, int32);
  array u(uv.begin(), Shape{4}, uint32);

  // The backend implements int32 Add/Multiply only for float dtypes, so
  // every wrapper below composes int-capable classes: bitwise, remainder,
  // integer Power, integer Abs, and integer Sign. Two fusable nodes per
  // graph so the tracer builds a tape. All exact against eager.

  // BitwiseInvert, wrapped by a bitwise identity.
  auto invert_fn = [&stream](const std::vector<array>& inputs) {
    return std::vector<array>{
        bitwise_xor(bitwise_invert(inputs[0], stream), array(1), stream)};
  };
  check_compiled_matches_eager(invert_fn, {x}, int32, stream, 0);

  // BitwiseBinary and/or/xor through tapes.
  std::vector<BinaryFn> bitwise_cases = {
      [](const array& a, const array& b, const Stream& s) {
        return bitwise_xor(bitwise_and(a, b, s), a, s);
      },
      [](const array& a, const array& b, const Stream& s) {
        return bitwise_and(bitwise_or(a, b, s), a, s);
      },
      [](const array& a, const array& b, const Stream& s) {
        return bitwise_or(bitwise_xor(a, b, s), b, s);
      },
  };
  for (const auto& bitwise : bitwise_cases) {
    auto fn = [&bitwise, &stream](const std::vector<array>& inputs) {
      return std::vector<array>{bitwise(inputs[0], inputs[1], stream)};
    };
    check_compiled_matches_eager(fn, {x, y}, int32, stream, 0);
  }

  // Integer remainder wrapped by a bitwise identity.
  auto rem_fn = [&stream](const std::vector<array>& inputs) {
    return std::vector<array>{
        bitwise_xor(remainder(inputs[0], inputs[1], stream), array(0), stream)};
  };
  check_compiled_matches_eager(rem_fn, {x, y}, int32, stream, 0);

  // Integer Power into integer Abs.
  auto int_power_fn = [&stream](const std::vector<array>& inputs) {
    return std::vector<array>{
        abs(power(inputs[0], array(2), stream), stream)};
  };
  check_compiled_matches_eager(int_power_fn, {x}, int32, stream, 0);

  // Integer Sign wrapped by a bitwise identity.
  auto int_sign_fn = [&stream](const std::vector<array>& inputs) {
    return std::vector<array>{
        bitwise_xor(sign(inputs[0], stream), array(0), stream)};
  };
  check_compiled_matches_eager(int_sign_fn, {x}, int32, stream, 0);

  // uint32 bitwise through a tape.
  auto uint_or_fn = [&stream](const std::vector<array>& inputs) {
    return std::vector<array>{
        bitwise_and(bitwise_or(inputs[0], inputs[1], stream), inputs[0], stream)};
  };
  check_compiled_matches_eager(uint_or_fn, {u, u}, uint32, stream, 0);
}

TEST_CASE(
    "compiled tape matches eager for comparison, logical, select, and"
    " cast classes") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv = {0.0f, 0.5f, -1.0f, 2.0f};
  std::vector<float> yv = {0.5f, 0.5f, 1.0f, -2.0f};
  std::vector<bool> cv = {true, false, true, false};
  array x(xv.begin(), Shape{4}, float32);
  array y(yv.begin(), Shape{4}, float32);
  array cond(cv.begin(), Shape{4}, bool_);
  // Float comparisons (the backend gates GreaterEqual to int32, so it is
  // covered separately below).
  std::vector<BinaryFn> comparisons = {
      [](const array& a, const array& b, const Stream& s) {
        return equal(a, b, s);
      },
      [](const array& a, const array& b, const Stream& s) {
        return not_equal(a, b, s);
      },
      [](const array& a, const array& b, const Stream& s) {
        return greater(a, b, s);
      },
      [](const array& a, const array& b, const Stream& s) {
        return less(a, b, s);
      },
      [](const array& a, const array& b, const Stream& s) {
        return less_equal(a, b, s);
      },
  };
  for (const auto& compare : comparisons) {
    auto fn = [&compare, &stream](const std::vector<array>& inputs) {
      return std::vector<array>{
          compare(multiply(inputs[0], inputs[0], stream), inputs[1], stream)};
    };
    check_compiled_matches_eager(fn, {x, y}, bool_, stream, 0);
  }

  // GreaterEqual is int32-only in this backend (its causal-mask kernel):
  // feed it an int-capable node and compare exact against eager.
  {
    std::vector<int32_t> iv = {1, 2, 3, 4};
    std::vector<int32_t> jv = {2, 2, 2, 2};
    array xi(iv.begin(), Shape{4}, int32);
    array yi(jv.begin(), Shape{4}, int32);
    auto ge_fn = [&stream](const std::vector<array>& inputs) {
      return std::vector<array>{greater_equal(
          bitwise_xor(inputs[0], inputs[1], stream), inputs[1], stream)};
    };
    check_compiled_matches_eager(ge_fn, {xi, yi}, bool_, stream, 0);
  }

  // Logical classes on boolean tape inputs, exact against eager.
  auto logical_fn = [&stream](const std::vector<array>& inputs) {
    auto both = logical_and(inputs[0], inputs[1], stream);
    auto either = logical_or(both, inputs[1], stream);
    return std::vector<array>{logical_not(either, stream)};
  };
  check_compiled_matches_eager(logical_fn, {cond, cond}, bool_, stream, 0);

  // Select with a comparison feeding it inside one tape, plus an AsType
  // cast layered on top, exact against eager.
  auto select_fn = [&stream](const std::vector<array>& inputs) {
    auto pick = where(inputs[0], inputs[1], inputs[2], stream);
    return std::vector<array>{astype(pick, int32, stream)};
  };
  check_compiled_matches_eager(select_fn, {cond, x, y}, int32, stream, 0);

  // Casts across the float/int boundary through one tape.
  auto cast_fn = [&stream](const std::vector<array>& inputs) {
    auto scaled = multiply(inputs[0], array(3.0f), stream);
    return std::vector<array>{
        add(astype(astype(scaled, int32, stream), float32, stream),
            array(0.5f),
            stream)};
  };
  check_compiled_matches_eager(cast_fn, {x}, float32, stream, 1e-6);
}

TEST_CASE("compiled tape returns multiple outputs exactly") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv = {0.5f, -1.5f, 2.0f, 0.0f};
  std::vector<float> yv = {1.0f, 0.25f, -2.0f, 4.0f};
  array x(xv.begin(), Shape{4}, float32);
  array y(yv.begin(), Shape{4}, float32);

  auto fn = [&stream](const std::vector<array>& inputs) {
    auto sum = add(inputs[0], inputs[1], stream);
    auto prod = multiply(sum, inputs[0], stream);
    return std::vector<array>{sum, prod};
  };
  check_compiled_matches_eager(fn, {x, y}, float32, stream, 1e-6);
}

TEST_CASE("compiled tape interleaves proven and widened classes in f16") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> gate_v = {0.35f, -1.1f, 2.2f, -0.4f};
  std::vector<float> up_v = {1.5f, 0.25f, -0.75f, 2.0f};
  array gate(gate_v.begin(), Shape{4}, float16);
  array up(up_v.begin(), Shape{4}, float16);

  // The mlx-lm swiglu fragment shape (sigmoid and multiply, the proven
  // subset) with widened tanh and abs nodes interleaved.
  auto fn = [&stream](const std::vector<array>& inputs) {
    auto activated = multiply(sigmoid(inputs[0], stream), inputs[0], stream);
    auto blended = add(activated, inputs[1], stream);
    auto bounded =
        multiply(tanh(blended, stream), abs(blended, stream), stream);
    return std::vector<array>{bounded};
  };
  // f16 keeps the host-independent comparison within 2e-3.
  check_compiled_matches_eager(fn, {gate, up}, float16, stream, 2e-3);
}

TEST_CASE("bf16 tapes stay gated for the widened op set") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv = {0.0f, 0.1f, 0.2f, 0.3f};
  array x(xv.begin(), Shape{4}, bfloat16);

  // Widened classes must not open a bf16 hole: the dtype gate refuses
  // before any node dispatches.
  set_compile_mode(CompileMode::enabled);
  auto fn = [&stream](const std::vector<array>& inputs) {
    auto rounded = round(multiply(inputs[0], inputs[0], stream), stream);
    return std::vector<array>{abs(rounded, stream)};
  };
  auto fused = compile(fn);
  std::string fused_error = evaluation_error(fused({x})[0]);
  CHECK(
      fused_error.find("[omarchy] Compiled tape bfloat16") !=
      std::string::npos);
  set_compile_mode(CompileMode::disabled);
}

TEST_CASE("complex tape ops stay refused by name") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // The pinned upstream header cannot list-initialize complex arrays from
  // iterators, so build the scalar through the explicit complex overload.
  // Content never matters: the tape refuses the Real node before any
  // dispatch.
  array z = array(complex64_t{1.0f, 0.5f});

  // Real is upstream-fusable so the tracer places it in the tape, but the
  // backend does not implement it, so the named tape error must fire.
  set_compile_mode(CompileMode::enabled);
  auto fn = [&stream](const std::vector<array>& inputs) {
    return std::vector<array>{add(real(inputs[0], stream), array(1.0f), stream)};
  };
  auto fused = compile(fn);
  std::string real_error = evaluation_error(fused({z})[0]);
  CHECK(
      real_error.find("[omarchy] Compiled tape op Real") != std::string::npos);
  set_compile_mode(CompileMode::disabled);
}

// A tiny fusable tape used by the fail-closed cases below: two nodes, so
// the tracer builds a real tape and eval_compiled_tape runs.
auto square_plus_one_fn(const Stream& stream) {
  return [&stream](const std::vector<array>& inputs) {
    return std::vector<array>{
        add(multiply(inputs[0], inputs[0], stream), array(1.0f), stream)};
  };
}

TEST_CASE("compiled tapes refuse by default on real Apple GPUs") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  auto& dev = omarchy::device(stream.device.index);
  bool refused_default = omarchy::compiled_tapes_refused(dev);

  // The policy is keyed to the device: refusal on real Apple GPU targets
  // only, where the corruption is observed (docs/known-defects.md);
  // development devices accepted through MLX_OMARCHY_ALLOW_NON_APPLE
  // keep running compiled tapes for the batteries and the harness.
  CHECK_EQ(refused_default, !dev.non_apple_dev());

  std::vector<float> xv = {0.5f, -1.5f, 2.0f, 0.0f};
  array x(xv.begin(), Shape{4}, float32);
  set_compile_mode(CompileMode::enabled);
  auto fused = compile(square_plus_one_fn(stream));
  std::string error = evaluation_error(fused({x})[0]);
  if (refused_default) {
    CHECK(
        error.find("[omarchy] Compiled tapes are refused") !=
        std::string::npos);
    CHECK(error.find("MLX_DISABLE_COMPILE=1") != std::string::npos);
    CHECK(
        error.find("MLX_OMARCHY_ALLOW_UNSAFE_COMPILE=1") !=
        std::string::npos);
  } else {
    // Development device: the corruption class is not observed here, so
    // the default run must still execute the tape.
    CHECK(error.empty());
  }
  set_compile_mode(CompileMode::disabled);
}

TEST_CASE("MLX_OMARCHY_ALLOW_UNSAFE_COMPILE=1 re-enables compiled tapes") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  auto& dev = omarchy::device(stream.device.index);
  setenv("MLX_OMARCHY_ALLOW_UNSAFE_COMPILE", "1", 1);
  CHECK_FALSE(omarchy::compiled_tapes_refused(dev));

  std::vector<float> xv = {0.5f, -1.5f, 2.0f, 0.0f};
  array x(xv.begin(), Shape{4}, float32);
  set_compile_mode(CompileMode::enabled);
  auto fused = compile(square_plus_one_fn(stream));
  // The override must let the tape run. Output correctness is asserted
  // only where it holds: on the Apple target the override exists so the
  // differential harness can reproduce the defect, not to promise
  // correct values.
  std::string error = evaluation_error(fused({x})[0]);
  CHECK(error.empty());
  set_compile_mode(CompileMode::disabled);
  unsetenv("MLX_OMARCHY_ALLOW_UNSAFE_COMPILE");
}
