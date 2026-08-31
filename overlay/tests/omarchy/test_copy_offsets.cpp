// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Offset handling in the buffer-level copy path. An array's own buffer
// offset (slice views) and the explicit i_offset / o_offset items must both
// reach the vkCmdCopyBuffer / vkCmdFillBuffer offsets. Every expected-value
// check fails if either the source or the destination offset is ignored.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/device.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

using namespace mlx::core;

namespace {

void skip(const char* reason) {
  std::cout << "Skipping: " << reason << "\n";
}

// Compare an evaluated int32 array against expected values by reading the
// host-visible buffer (Omarchy buffers are mapped and coherent).
bool values_equal(const array& a, const std::vector<int32_t>& expected) {
  if (static_cast<size_t>(a.size()) != expected.size()) {
    return false;
  }
  return std::memcmp(
             a.data<int32_t>(),
             expected.data(),
             expected.size() * sizeof(int32_t)) == 0;
}

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
}

} // namespace

TEST_CASE("vector copy honors the source slice offset") {
  if (!gpu::is_available()) {
    skip(
        "no qualifying Vulkan device (set MLX_OMARCHY_ALLOW_NON_APPLE=1 on"
        " a development machine).");
    return;
  }
  Stream s = gpu_stream();

  array x({100, 101, 102, 103, 104, 105, 106, 107}, int32);
  array v = slice(x, {2}, {6}, {1}, s);
  v.eval();
  CHECK(v.offset() == 2 * static_cast<int64_t>(sizeof(int32_t)));

  // Full::eval_gpu routes a contiguous value array to a Vector copy.
  array y = full(v.shape(), v, s);
  y.eval();
  omarchy::get_command_encoder(s).synchronize();
  CHECK(values_equal(y, {102, 103, 104, 105}));

  // Ownership: the copy owns its bytes. Zero the source view through the
  // device fill path afterwards; y must keep its values while the parent
  // is zeroed exactly inside the view.
  array zero = array(0, int32);
  copy_gpu_inplace(
      zero, v, v.shape(), v.strides(), v.strides(), 0, 0, CopyType::Scalar, s);
  omarchy::get_command_encoder(s).synchronize();

  CHECK(values_equal(y, {102, 103, 104, 105}));
  CHECK(v.buffer().ptr() == x.buffer().ptr());
  const auto* xp = x.data<int32_t>();
  CHECK(xp[0] == 100);
  CHECK(xp[1] == 101);
  CHECK(xp[2] == 0);
  CHECK(xp[3] == 0);
  CHECK(xp[4] == 0);
  CHECK(xp[5] == 0);
  CHECK(xp[6] == 106);
  CHECK(xp[7] == 107);
}

TEST_CASE("vector copy honors explicit item offsets on both sides") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = gpu_stream();

  // Destinations are 3-item views over 4-item parents: an explicit
  // o_offset needs buffer slack beyond the logical view.
  array in({10, 11, 12, 13, 14, 15, 16, 17}, int32);
  array parent({9, 9, 9, 9}, int32);
  array out = slice(parent, {0}, {3}, {1}, s);
  out.eval();
  copy_gpu_inplace(
      in,
      out,
      in.shape(),
      in.strides(),
      out.strides(),
      /*i_offset=*/2,
      /*o_offset=*/1,
      CopyType::Vector,
      s);
  omarchy::get_command_encoder(s).synchronize();
  // 3 items are copied from in[2..] to parent items [1..).
  CHECK(values_equal(parent, {9, 12, 13, 14}));

  // The same explicit items combined with a slice view's buffer offset.
  array v = slice(in, {1}, {7}, {1}, s);
  v.eval();
  array parent2({9, 9, 9, 9}, int32);
  array out2 = slice(parent2, {0}, {3}, {1}, s);
  out2.eval();
  copy_gpu_inplace(
      v,
      out2,
      v.shape(),
      v.strides(),
      out2.strides(),
      /*i_offset=*/2,
      /*o_offset=*/1,
      CopyType::Vector,
      s);
  omarchy::get_command_encoder(s).synchronize();
  // The source starts at v's byte offset plus i_offset items.
  CHECK(values_equal(parent2, {9, 13, 14, 15}));
}

TEST_CASE("zero fill honors a slice view offset") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = gpu_stream();

  array parent({1, 2, 3, 4, 5, 6, 7, 8}, int32);
  array view = slice(parent, {2}, {6}, {1}, s);
  view.eval();
  array zero = array(0, int32);
  copy_gpu_inplace(
      zero,
      view,
      view.shape(),
      view.strides(),
      view.strides(),
      0,
      0,
      CopyType::Scalar,
      s);
  omarchy::get_command_encoder(s).synchronize();

  const auto* pp = parent.data<int32_t>();
  CHECK(pp[0] == 1);
  CHECK(pp[1] == 2);
  CHECK(pp[2] == 0);
  CHECK(pp[3] == 0);
  CHECK(pp[4] == 0);
  CHECK(pp[5] == 0);
  CHECK(pp[6] == 7);
  CHECK(pp[7] == 8);
}

TEST_CASE("zero fill honors an explicit destination item offset") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = gpu_stream();

  array parent({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, int32);
  // An 8-item window at buffer offset 0; the fill starts at item 2 of the
  // window and covers its full length.
  array head = slice(parent, {0}, {8}, {1}, s);
  head.eval();
  array zero = array(0, int32);
  copy_gpu_inplace(
      zero,
      head,
      head.shape(),
      head.strides(),
      head.strides(),
      0,
      /*o_offset=*/2,
      CopyType::Scalar,
      s);
  omarchy::get_command_encoder(s).synchronize();

  const auto* pp = parent.data<int32_t>();
  const std::vector<int32_t> expected = {1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 11, 12};
  for (size_t i = 0; i < expected.size(); ++i) {
    CHECK(pp[i] == expected[i]);
  }
}

TEST_CASE("zero fill handles a 2-byte-aligned destination") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = gpu_stream();

  // vkCmdFillBuffer needs 4-byte alignment; a uint16 view at byte offset 2
  // forces the misaligned lead onto the host path.
  array parent({10, 11, 12, 13, 14, 15, 16, 17, 18, 19}, uint16);
  array view = slice(parent, {1}, {7}, {1}, s);
  view.eval();
  array zero = array(0, uint16);
  copy_gpu_inplace(
      zero,
      view,
      view.shape(),
      view.strides(),
      view.strides(),
      0,
      0,
      CopyType::Scalar,
      s);
  omarchy::get_command_encoder(s).synchronize();

  const auto* pp = parent.data<uint16_t>();
  CHECK(pp[0] == 10);
  CHECK(pp[1] == 0);
  CHECK(pp[2] == 0);
  CHECK(pp[3] == 0);
  CHECK(pp[4] == 0);
  CHECK(pp[5] == 0);
  CHECK(pp[6] == 0);
  CHECK(pp[7] == 17);
  CHECK(pp[8] == 18);
  CHECK(pp[9] == 19);

  // The fresh-output public route (Full::eval_gpu Scalar path) stays exact.
  array z = zeros({5}, uint16, s);
  z.eval();
  omarchy::get_command_encoder(s).synchronize();
  const auto* zp = z.data<uint16_t>();
  for (int i = 0; i < 5; ++i) {
    CHECK(zp[i] == 0);
  }
}

TEST_CASE("zero fill covers sub-word outputs") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = gpu_stream();

  // Fewer than four bytes never form a device word; the fill is host-only
  // and the bytes must still be zeroed exactly.
  array zero = array(0, uint8);

  SUBCASE("one byte") {
    array parent({5, 6, 7, 8}, uint8);
    array view = slice(parent, {1}, {2}, {1}, s);
    view.eval();
    copy_gpu_inplace(
        zero,
        view,
        view.shape(),
        view.strides(),
        view.strides(),
        0,
        0,
        CopyType::Scalar,
        s);
    omarchy::get_command_encoder(s).synchronize();
    const auto* pp = parent.data<uint8_t>();
    CHECK(pp[0] == 5);
    CHECK(pp[1] == 0);
    CHECK(pp[2] == 7);
    CHECK(pp[3] == 8);
  }

  SUBCASE("two bytes") {
    array parent({5, 6, 7, 8}, uint8);
    array view = slice(parent, {1}, {3}, {1}, s);
    view.eval();
    copy_gpu_inplace(
        zero,
        view,
        view.shape(),
        view.strides(),
        view.strides(),
        0,
        0,
        CopyType::Scalar,
        s);
    omarchy::get_command_encoder(s).synchronize();
    const auto* pp = parent.data<uint8_t>();
    CHECK(pp[0] == 5);
    CHECK(pp[1] == 0);
    CHECK(pp[2] == 0);
    CHECK(pp[3] == 8);
  }

  SUBCASE("three bytes") {
    array parent({5, 6, 7, 8}, uint8);
    array view = slice(parent, {0}, {3}, {1}, s);
    view.eval();
    copy_gpu_inplace(
        zero,
        view,
        view.shape(),
        view.strides(),
        view.strides(),
        0,
        0,
        CopyType::Scalar,
        s);
    omarchy::get_command_encoder(s).synchronize();
    const auto* pp = parent.data<uint8_t>();
    CHECK(pp[0] == 0);
    CHECK(pp[1] == 0);
    CHECK(pp[2] == 0);
    CHECK(pp[3] == 8);
  }
}

TEST_CASE("reshape shares lazily and copies with the source offset") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = gpu_stream();

  array x({100, 101, 102, 103, 104, 105, 106}, int32);
  array v = slice(x, {1}, {7}, {1}, s);
  v.eval();

  // Row-contiguous reshape is a lazy view of the parent buffer.
  array shared = reshape(v, {2, 3}, s);
  shared.eval();
  CHECK(shared.buffer().ptr() == x.buffer().ptr());
  const auto* sp = shared.data<int32_t>();
  CHECK(sp[0] == 101);
  CHECK(sp[5] == 106);

  // A col-contiguous view cannot reshape lazily; the flat device copy must
  // start at the view's byte offset.
  array vt = transpose(reshape(v, {2, 3}, s), s);
  vt.eval();
  CHECK(vt.flags().contiguous);
  CHECK_FALSE(vt.flags().row_contiguous);
  array r = reshape(vt, {2, 3}, s);
  r.eval();
  omarchy::get_command_encoder(s).synchronize();
  CHECK(r.buffer().ptr() != x.buffer().ptr());
  CHECK(values_equal(r, {101, 102, 103, 104, 105, 106}));
}
