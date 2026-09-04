// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Wave 1 shape and layout primitives. Upstream resolves these through the
// shared backend/gpu primitives (zero-copy buffer views) or the omarchy copy
// engine (strided copies, casts, zero fill). Each TEST_CASE anchors one
// primitive with exact values so the compatibility matrix counts it as
// covered; the DynamicSlice cases pin today's named error, which is not
// coverage.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/primitives.h"
#include "mlx/device.h"
#include "mlx/ops.h"
#include "mlx/compile.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

using namespace mlx::core;

namespace {

void skip(const char* reason) {
  std::cout << "Skipping: " << reason << "\n";
}

bool compute_available() {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device (set MLX_OMARCHY_ALLOW_NON_APPLE=1 on"
         " a development machine).");
    return false;
  }
  return true;
}

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
}

template <typename T>
void check_exact(array value, const std::vector<T>& expected) {
  // Retained views share gapped buffers: read logical C order only.
  auto dense = contiguous(value);
  dense.eval();
  REQUIRE_EQ(dense.size(), expected.size());
  const T* values = dense.data<T>();
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK_EQ(values[index], expected[index]);
  }
}

void check_values(array value, const std::vector<float>& expected,
                  double epsilon = 1e-5) {
  auto dense = contiguous(value);
  dense.eval();
  REQUIRE_EQ(dense.size(), expected.size());
  const float* values = dense.data<float>();
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK(values[index] == doctest::Approx(expected[index]).epsilon(epsilon));
  }
}

std::string caught_message(array& value) {
  std::string message;
  try {
    value.eval();
  } catch (const std::exception& e) {
    message = e.what();
  }
  return message;
}

} // namespace

TEST_CASE("AsStrided shares a row-contiguous buffer with exact strides") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array flat = flatten(arange(0, 10, float32, stream), stream);

  // A 2x3 window of the flat buffer starting at element 2 with row strides.
  array window = as_strided(flat, {2, 3}, {3, 1}, 2, stream);
  check_values(window, {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f});

  // Overlapping strided rows read the same buffer elements twice; the
  // logical order goes through the strided General copy.
  array overlapped = as_strided(flat, {3, 2}, {1, 4}, 0, stream);
  check_values(contiguous(overlapped, false, stream),
               {0.0f, 4.0f, 1.0f, 5.0f, 2.0f, 6.0f});
}

TEST_CASE("AsType casts values exactly through the copy engine") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array f = array({-1.5f, 0.25f, 2.75f}, float32);
  check_exact<int32_t>(astype(f, int32, stream), {-1, 0, 2});
  check_values(
      astype(array({0, 1, 2}, int32), float32, stream),
      {0.0f, 1.0f, 2.0f});
  // bool -> f32 rides the CastBoolF32 word kernel.
  check_values(
      astype(array({true, false, true}, bool_), float32, stream),
      {1.0f, 0.0f, 1.0f});
  // A dtype-converting cast of a non-row-contiguous view must carry the
  // transposed logical order, not the storage order; copy_gpu
  // materializes the view first (same-dtype astype returns the view
  // itself, so only the dtype-changing form builds an AsType).
  array t = transpose(
      array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, float32),
      {1, 0}, stream);
  check_exact<int32_t>(astype(t, int32, stream), {1, 4, 2, 5, 3, 6});
  check_exact<uint16_t>(
      astype(t, float16, stream),
      {0x3c00, 0x4400, 0x4000, 0x4500, 0x4200, 0x4600});
}

TEST_CASE("Broadcast expands zero-stride views with exact values") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array row = array({1.0f, 2.0f}, float32);
  array grown = broadcast_to(row, {3, 2}, stream);
  CHECK_EQ(grown.ndim(), 2);
  CHECK_EQ(grown.shape(0), 3);
  // A broadcast view shares the source buffer, so read the logical
  // order back through the strided copy engine.
  check_values(contiguous(grown, false, stream),
               {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f});
  grown.eval();
  CHECK_EQ(grown.data_size(), 2);
}

TEST_CASE("BroadcastAxes aligns arrays through zero-stride views") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array a = array({1.0f, 2.0f}, {1, 2}, float32);        // shape 1x2
  array b = array({1.0f, 10.0f}, {2, 1}, float32);       // shape 2x1
  auto parts = broadcast_arrays({a, b}, stream);
  REQUIRE_EQ(parts.size(), 2);
  CHECK_EQ(parts[0].shape(), Shape({2, 2}));
  CHECK_EQ(parts[1].shape(), Shape({2, 2}));
  check_values(contiguous(parts[0], false, stream),
               {1.0f, 2.0f, 1.0f, 2.0f});
  check_values(contiguous(parts[1], false, stream),
               {1.0f, 1.0f, 10.0f, 10.0f});
}

TEST_CASE("Concatenate joins arrays through the copy engine") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array a = arange(0, 3, float32, stream);
  array b = arange(10, 13, float32, stream);

  // Axis 0 with row-contiguous inputs rides the vector vkCmdCopyBuffer path.
  check_values(
      concatenate({a, b}, 0, stream),
      {0.0f, 1.0f, 2.0f, 10.0f, 11.0f, 12.0f});

  // Axis 1 rides the strided CopyGeneral path.
  array m = array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, float32);
  array n = array({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2}, float32);
  check_values(
      concatenate({m, n}, 1, stream),
      {1.0f, 2.0f, 5.0f, 6.0f, 3.0f, 4.0f, 7.0f, 8.0f});
}

TEST_CASE("Contiguous materializes a strided view with exact values") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array m = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, float32);
  array t = transpose(m, {1, 0}, stream);
  t.eval();
  CHECK_FALSE(t.flags().row_contiguous);

  array packed = contiguous(t, false, stream);
  CHECK(packed.flags().row_contiguous);
  check_values(packed, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f});
}

TEST_CASE("Copy clones the buffer with exact values") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array a = array({3.5f, -1.25f, 9.0f}, float32);
  array c = copy(a, stream);
  check_values(c, {3.5f, -1.25f, 9.0f});

  // The copy reads back identical values after new work lands on device.
  array edited = add(c, array(1.0f), stream);
  check_values(c, {3.5f, -1.25f, 9.0f});
  check_values(edited, {4.5f, -0.25f, 10.0f});
}

TEST_CASE("CustomTransforms redefines the transform of a function") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // custom_function wraps fun; without explicit transforms the defaults
  // recompose vjp/jvp/vmap from core ops. Calling the wrapped function
  // exercises CustomTransforms eval, which forwards buffers.
  auto doubled = custom_function(
      [](const std::vector<array>& inputs) -> std::vector<array> {
        return {multiply(inputs[0], array(2.0f))};
      });
  auto out = doubled({array({1.0f, 2.0f, 3.0f}, float32)});
  REQUIRE_EQ(out.size(), 1);
  check_values(out[0], {2.0f, 4.0f, 6.0f});

  // A custom vjp replaces the backward rule; the forward still routes
  // through CustomTransforms.
  auto square_with_grad = custom_function(
      [](const std::vector<array>& inputs) -> std::vector<array> {
        return {multiply(inputs[0], inputs[0])};
      },
      [](const std::vector<array>& primals,
         const std::vector<array>& cotangents,
         const std::vector<array>&) -> std::vector<array> {
        return {multiply(cotangents[0], array(100.0f))};
      });
  auto fwd = square_with_grad({array({3.0f}, float32)});
  check_values(fwd[0], {9.0f});
  auto [value, grad] = vjp(square_with_grad,
                           {array({3.0f}, float32)},
                           {array({1.0f}, float32)});
  check_values(value[0], {9.0f});
  check_values(grad[0], {100.0f});
}

TEST_CASE("Depends forces evaluation of its dependencies") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array dep = multiply(arange(1, 4, float32, stream), array(10.0f), stream);
  array main = add(arange(1, 4, float32, stream), array(1.0f), stream);
  auto outs = depends({main}, {dep});
  REQUIRE_EQ(outs.size(), 1);
  check_values(outs[0], {2.0f, 3.0f, 4.0f});
  // Evaluating the output also evaluated the dependency.
  check_values(dep, {10.0f, 20.0f, 30.0f});
}

TEST_CASE("DynamicSlice raises the named dynamic-slice-offset error") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = arange(0, 8, float32, stream);
  array start = array({2}, int32);

  // No public op constructs DynamicSlice in C++, so build the primitive
  // directly the way the compiled dynamic-shape path would.
  array sliced(
      Shape{3},
      float32,
      std::make_shared<DynamicSlice>(stream, std::vector<int>{0}, Shape{3}),
      {src, start});
  auto message = caught_message(sliced);
  CHECK(message.find("dynamic slice offset") != std::string::npos);
}

TEST_CASE("DynamicSliceUpdate raises the named dynamic-slice-offset error") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array src = arange(0, 8, float32, stream);
  array upd = array({9.0f, 9.0f}, float32);
  array start = array({2}, int32);

  array updated(
      src.shape(),
      float32,
      std::make_shared<DynamicSliceUpdate>(stream, std::vector<int>{0}),
      {src, upd, start});
  auto message = caught_message(updated);
  CHECK(message.find("dynamic slice offset") != std::string::npos);
}

TEST_CASE("ExpandDims inserts length-one axes in place") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array a = arange(0, 6, float32, stream);
  array grown = expand_dims(a, 1, stream);
  CHECK_EQ(grown.shape(), Shape({6, 1}));
  check_values(grown, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

  array two = expand_dims(a, {0, 2}, stream);
  CHECK_EQ(two.shape(), Shape({1, 6, 1}));
  check_values(two, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
}

TEST_CASE("Flatten collapses axes with exact values") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array m = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, float32);
  array flat = flatten(m, 0, -1, stream);
  CHECK_EQ(flat.shape(), Shape({6}));
  check_values(flat, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  // A partial flatten of a transposed view rides the strided reshape copy.
  array t = transpose(m, {1, 0}, stream);
  array partial = flatten(t, 0, 1, stream);
  CHECK_EQ(partial.shape(), Shape({6}));
  check_values(partial, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f});
}

TEST_CASE("Full fills exact scalar values") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  check_values(
      full({2, 3}, 7.5f, float32, stream),
      {7.5f, 7.5f, 7.5f, 7.5f, 7.5f, 7.5f});
  // Zero fill of any dtype rides the byte-write fast path.
  check_exact<int32_t>(full({3}, 0, int32, stream), {0, 0, 0});
  // A non-zero integer fill keeps the named error; only float storage
  // has a fill kernel.
  array ints = full({2}, 7, int32, stream);
  auto message = caught_message(ints);
  CHECK(message.find("non-zero scalar fill") != std::string::npos);
}

TEST_CASE("NumberOfElements evaluates inside a shapeless compile") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Outside dynamic tracing mx.mean folds the element count at trace time;
  // a shapeless compile defers it to the NumberOfElements primitive.
  auto mean_fn = compile(
      [](const std::vector<array>& inputs) -> std::vector<array> {
        return {mean(inputs[0], 0)};
      },
      /*shapeless=*/true);
  check_values(mean_fn({array({1.0f, 2.0f, 3.0f, 4.0f}, float32)})[0], {2.5f});
  check_values(mean_fn({array({10.0f, 20.0f, 30.0f}, float32)})[0], {20.0f});
}

TEST_CASE("Pad zero-fills boundaries with exact values") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array a = array({1.0f, 2.0f}, float32);
  check_values(
      pad(a, std::pair<int, int>{1, 2}), {0.0f, 1.0f, 2.0f, 0.0f, 0.0f});
  array m = array({1.0f, 2.0f}, {1, 2}, float32);
  check_values(
      pad(m, std::vector<int>{1}, Shape{1}, Shape{1}),
      {0.0f, 1.0f, 2.0f, 0.0f});
}

TEST_CASE("Reshape shares buffers and copies strided views") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array m = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, float32);

  // A contiguous reshape shares the buffer: data_size is unchanged and the
  // values read back in the new shape.
  array shared = reshape(m, {3, 2}, stream);
  shared.eval();
  CHECK_EQ(shared.data_size(), 6);
  check_values(shared, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  // A reshape of a transposed view needs the strided-copy gather.
  array t = transpose(m, {1, 0}, stream);
  check_values(
      reshape(t, {2, 3}, stream), {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f});
}

TEST_CASE("Slice cuts exact windows with steps") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array a = arange(0, 10, float32, stream);
  check_values(slice(a, {2}, {7}, {2}, stream), {2.0f, 4.0f, 6.0f});

  array m = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, float32);
  check_values(slice(m, {0, 1}, {2, 3}, stream), {2.0f, 3.0f, 5.0f, 6.0f});

  // A window on a transposed view rides the strided copy path.
  array t = transpose(m, {1, 0}, stream);
  check_values(slice(t, {1, 0}, {3, 2}, stream), {2.0f, 5.0f, 3.0f, 6.0f});
}

TEST_CASE("Split returns exact per-part views") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array a = arange(0, 6, float32, stream);
  auto parts = split(a, 3, 0, stream);
  REQUIRE_EQ(parts.size(), 3);
  check_values(parts[0], {0.0f, 1.0f});
  check_values(parts[1], {2.0f, 3.0f});
  check_values(parts[2], {4.0f, 5.0f});

  array b = arange(0, 4, float32, stream);
  auto uneven = split(b, {1, 3}, 0, stream);
  REQUIRE_EQ(uneven.size(), 3);
  check_values(uneven[0], {0.0f});
  check_values(uneven[1], {1.0f, 2.0f});
  check_values(uneven[2], {3.0f});
}

TEST_CASE("Squeeze drops length-one axes with exact values") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array a = expand_dims(arange(0, 4, float32, stream), {0, 2}, stream);
  CHECK_EQ(a.shape(), Shape({1, 4, 1}));
  array squeezed = squeeze(a, {0, 2}, stream);
  CHECK_EQ(squeezed.shape(), Shape({4}));
  check_values(squeezed, {0.0f, 1.0f, 2.0f, 3.0f});
}

TEST_CASE("StopGradient passes values and detaches the tape") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array a = array({1.0f, 2.0f}, float32);
  array stopped = stop_gradient(a, stream);
  check_values(stopped, {1.0f, 2.0f});

  auto [value, grad] = vjp(
      [](const std::vector<array>& inputs) {
        return std::vector<array>{
            stop_gradient(multiply(inputs[0], array(3.0f)))};
      },
      {a},
      {array({1.0f, 1.0f}, float32)});
  check_values(value[0], {3.0f, 6.0f});
  check_values(grad[0], {0.0f, 0.0f});
}

TEST_CASE("Transpose permutes strides as a zero-copy view") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array m = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, float32);
  array t = transpose(m, {1, 0}, stream);
  CHECK_EQ(t.shape(), Shape({3, 2}));
  // The view shares the source buffer, so read the logical order back
  // through the strided copy engine after eval.
  t.eval();
  CHECK_EQ(t.data_size(), 6);
  CHECK_FALSE(t.flags().row_contiguous);
  check_values(contiguous(t, false, stream),
               {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f});
}

TEST_CASE("Unflatten splits one axis into exact values") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array m = array({1.0f, 2.0f, 3.0f, 4.0f}, {1, 4}, float32);
  array grown = unflatten(m, 1, {2, 2}, stream);
  CHECK_EQ(grown.shape(), Shape({1, 2, 2}));
  check_values(grown, {1.0f, 2.0f, 3.0f, 4.0f});
}

TEST_CASE("View reinterprets the buffer with exact bit patterns") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // float32 1.0 has bit pattern 0x3f800000, which is int32 1065353216.
  array f = array({1.0f, 2.0f}, float32);
  check_exact<int32_t>(view(f, int32, stream), {1065353216, 1073741824});

  // Two int32 words reinterpret to one int64 word.
  array i = array({5, 6}, int32);
  array wide = view(i, int64, stream);
  CHECK_EQ(wide.shape(), Shape({1}));
  check_exact<int64_t>(wide, {(static_cast<int64_t>(6) << 32) | 5});
}
