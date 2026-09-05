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

// Compare an evaluated uint32 array against expected words.
bool words_equal(const array& a, const std::vector<uint32_t>& expected) {
  if (static_cast<size_t>(a.size()) != expected.size()) {
    return false;
  }
  return std::memcmp(
             a.data<uint32_t>(),
             expected.data(),
             expected.size() * sizeof(uint32_t)) == 0;
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

TEST_CASE("nonzero scalar fill honors a slice view offset") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = gpu_stream();

  array parent({1, 2, 3, 4, 5, 6, 70, 8}, int32);
  array view = slice(parent, {2}, {6}, {1}, s);
  view.eval();
  array seven(7, int32);
  copy_gpu_inplace(
      seven,
      view,
      view.shape(),
      view.strides(),
      view.strides(),
      0,
      0,
      CopyType::Scalar,
      s);
  omarchy::get_command_encoder(s).synchronize();

  CHECK(view.buffer().ptr() == parent.buffer().ptr());
  const auto* pp = parent.data<int32_t>();
  CHECK(pp[0] == 1);
  CHECK(pp[1] == 2);
  CHECK(pp[2] == 7);
  CHECK(pp[3] == 7);
  CHECK(pp[4] == 7);
  CHECK(pp[5] == 7);
  CHECK(pp[6] == 70);
  CHECK(pp[7] == 8);
}

TEST_CASE("nonzero scalar fill keeps negative and large int32 exact") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = gpu_stream();

  // Destinations are 4-item views over 8-item parents: the explicit
  // o_offset starts the fill inside the parent and the edges must keep
  // their bytes.
  SUBCASE("negative value at a nonzero destination offset") {
    array parent({9, 9, 9, 9, 9, 9, 9, 9}, int32);
    array view = slice(parent, {1}, {5}, {1}, s);
    view.eval();
    array neg(-1, int32);
    copy_gpu_inplace(
        neg,
        view,
        view.shape(),
        view.strides(),
        view.strides(),
        0,
        /*o_offset=*/2,
        CopyType::Scalar,
        s);
    omarchy::get_command_encoder(s).synchronize();
    CHECK(words_equal(
        parent,
        {9u,
         9u,
         9u,
         0xFFFFFFFFu,
         0xFFFFFFFFu,
         0xFFFFFFFFu,
         0xFFFFFFFFu,
         9u}));
  }
  // float32 cannot represent 16777217 (it rounds to 16777216); an exact
  // word proves the scalar rode a raw-word transport, not a float.
  SUBCASE("integer beyond the float32 exact range") {
    array parent({9, 9, 9, 9}, int32);
    parent.eval();
    array big(16777217, int32);
    copy_gpu_inplace(
        big,
        parent,
        parent.shape(),
        parent.strides(),
        parent.strides(),
        0,
        0,
        CopyType::Scalar,
        s);
    omarchy::get_command_encoder(s).synchronize();
    CHECK(words_equal(
        parent,
        {16777217u, 16777217u, 16777217u, 16777217u}));
  }

  SUBCASE("large odd int32 stays exact") {
    array parent({9, 9, 9, 9, 9, 9}, int32);
    array view = slice(parent, {0}, {4}, {1}, s);
    view.eval();
    array odd(1000000007, int32);
    copy_gpu_inplace(
        odd,
        view,
        view.shape(),
        view.strides(),
        view.strides(),
        0,
        0,
        CopyType::Scalar,
        s);
    omarchy::get_command_encoder(s).synchronize();
    CHECK(words_equal(
        parent,
        {1000000007u, 1000000007u, 1000000007u, 1000000007u, 9u, 9u}));
  }
}

TEST_CASE("nonzero scalar fill writes uint32 max bit-exact") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = gpu_stream();

  array parent({3, 3, 3, 3, 3, 3}, uint32);
  array view = slice(parent, {1}, {5}, {1}, s);
  view.eval();
  array umax(4294967295u, uint32);
  copy_gpu_inplace(
      umax,
      view,
      view.shape(),
      view.strides(),
      view.strides(),
      0,
      0,
      CopyType::Scalar,
      s);
  omarchy::get_command_encoder(s).synchronize();
  CHECK(words_equal(parent, {3u, 4294967295u, 4294967295u, 4294967295u, 4294967295u, 3u}));

  // The all-ones word pattern also round-trips through a fresh output.
  array out = full({3}, 4294967295u, uint32, s);
  out.eval();
  omarchy::get_command_encoder(s).synchronize();
  CHECK(words_equal(out, {4294967295u, 4294967295u, 4294967295u}));
}

TEST_CASE("nonzero integer scalar fills keep named refusals elsewhere") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = gpu_stream();

  // A width the backend does not carry keeps the named refusal; it must
  // never crash.
  array wide = full({2}, 5, int64, s);
  bool caught = false;
  std::string message;
  try {
    wide.eval();
  } catch (const std::runtime_error& e) {
    caught = true;
    message = e.what();
  }
  REQUIRE(caught);
  CHECK(message.find("[omarchy]") != std::string::npos);
  CHECK(message.find("non-zero scalar fill") != std::string::npos);

  // A scalar dtype the destination does not share refuses by name too.
  array out = full({4}, 0.0f, float32, s);
  out.eval();
  array seven(7, int32);
  caught = false;
  try {
    copy_gpu_inplace(
        seven,
        out,
        out.shape(),
        seven.strides(),
        out.strides(),
        0,
        0,
        CopyType::Scalar,
        s);
  } catch (const std::runtime_error& e) {
    caught = true;
    message = e.what();
  }
  REQUIRE(caught);
  CHECK(message.find("non-zero scalar fill dtype") != std::string::npos);
}

TEST_CASE("nonzero scalar fill leaves an empty output untouched") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = gpu_stream();

  array out = full({0}, 16777217, int32, s);
  out.eval();
  omarchy::get_command_encoder(s).synchronize();
  CHECK_EQ(out.size(), 0);
  CHECK_EQ(out.nbytes(), 0);
}

TEST_CASE("full ones and full_like fill exact integers") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = gpu_stream();

  array f = full({5}, 16777217, int32, s);
  f.eval();
  omarchy::get_command_encoder(s).synchronize();
  CHECK(words_equal(
      f,
      {16777217u, 16777217u, 16777217u, 16777217u, 16777217u}));

  array o = ones({4}, uint32, s);
  o.eval();
  omarchy::get_command_encoder(s).synchronize();
  CHECK(words_equal(o, {1u, 1u, 1u, 1u}));

  array parent({1, 2, 3, 4, 5, 6}, int32);
  array v = slice(parent, {1}, {5}, {1}, s);
  v.eval();
  array like = full_like(v, -3, s);
  like.eval();
  omarchy::get_command_encoder(s).synchronize();
  CHECK(like.buffer().ptr() != parent.buffer().ptr());
  CHECK(words_equal(like, {0xFFFFFFFDu, 0xFFFFFFFDu, 0xFFFFFFFDu, 0xFFFFFFFDu}));
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

  // A col-contiguous view cannot reshape lazily; the general gather must
  // start at the view's byte offset and read through the view's strides,
  // so rows interleave exactly as upstream and the CPU backend do.
  array vt = transpose(reshape(v, {2, 3}, s), s);
  vt.eval();
  CHECK(vt.flags().contiguous);
  CHECK_FALSE(vt.flags().row_contiguous);
  array r = reshape(vt, {2, 3}, s);
  r.eval();
  omarchy::get_command_encoder(s).synchronize();
  CHECK(r.buffer().ptr() != x.buffer().ptr());
  CHECK(values_equal(r, {101, 104, 102, 105, 103, 106}));
}

namespace {

// Build an integer-family array from host values. Bool sources go
// through a byte vector, never std::vector<bool> (bit-packed proxy;
// see docs/known-defects.md).
array int_values_array(const std::vector<int64_t>& values, Dtype dtype) {
  Shape shape{static_cast<int>(values.size())};
  switch (dtype) {
    case bool_: {
      std::vector<uint8_t> v;
      for (int64_t x : values) {
        v.push_back(x != 0 ? 1 : 0);
      }
      return array(v.begin(), shape, bool_);
    }
    case uint8: {
      std::vector<uint8_t> v(values.begin(), values.end());
      return array(v.begin(), shape, uint8);
    }
    case int8: {
      std::vector<int8_t> v(values.begin(), values.end());
      return array(v.begin(), shape, int8);
    }
    case uint16: {
      std::vector<uint16_t> v(values.begin(), values.end());
      return array(v.begin(), shape, uint16);
    }
    case int16: {
      std::vector<int16_t> v(values.begin(), values.end());
      return array(v.begin(), shape, int16);
    }
    case uint32: {
      std::vector<uint32_t> v(values.begin(), values.end());
      return array(v.begin(), shape, uint32);
    }
    case int32: {
      std::vector<int32_t> v(values.begin(), values.end());
      return array(v.begin(), shape, int32);
    }
    case uint64: {
      std::vector<uint64_t> v(values.begin(), values.end());
      return array(v.begin(), shape, uint64);
    }
    case int64: {
      std::vector<int64_t> v(values.begin(), values.end());
      return array(v.begin(), shape, int64);
    }
    default:
      throw std::runtime_error("unsupported test dtype");
  }
}

// The exact bytes a static_cast chain from source value to destination
// must produce: canonicalize the source value (sign/zero extend to 64
// bits), then truncate to the destination width; a bool destination
// stores 0/1 by nonzero.
uint64_t source_bits(int64_t v, Dtype src) {
  switch (src) {
    case bool_:
      return v != 0 ? 1u : 0u;
    case uint8:
      return static_cast<uint8_t>(v);
    case int8:
      return static_cast<uint64_t>(
          static_cast<int64_t>(static_cast<int8_t>(v)));
    case uint16:
      return static_cast<uint16_t>(v);
    case int16:
      return static_cast<uint64_t>(
          static_cast<int64_t>(static_cast<int16_t>(v)));
    case uint32:
      return static_cast<uint32_t>(v);
    case int32:
      return static_cast<uint64_t>(
          static_cast<int64_t>(static_cast<int32_t>(v)));
    case uint64:
      return static_cast<uint64_t>(v);
    case int64:
      return static_cast<uint64_t>(v);
    default:
      return 0;
  }
}

std::vector<uint8_t> expected_cast_bytes(
    const std::vector<int64_t>& values,
    Dtype src,
    Dtype dst) {
  std::vector<uint8_t> out;
  for (int64_t v : values) {
    uint64_t canonical = source_bits(v, src);
    if (dst == bool_) {
      out.push_back(canonical != 0 ? 1 : 0);
      continue;
    }
    size_t n = static_cast<size_t>(size_of(dst));
    uint64_t bits = n == 8
        ? canonical
        : (canonical & ((1ull << (8 * n)) - 1));
    for (size_t i = 0; i < n; ++i) {
      out.push_back(static_cast<uint8_t>(bits >> (8 * i)));
    }
  }
  return out;
}

} // namespace

TEST_CASE("dtype converting copies cover the integer family and bool") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = gpu_stream();

  // Nine elements: a partial tail word for the packed-byte legs.
  const std::vector<int64_t> values = {
      0, 1, 3, 200, 255, 127, 128, -1, -128};
  const std::vector<Dtype> dtypes = {
      bool_, uint8, int8, uint16, int16, uint32, int32, uint64, int64};
  for (Dtype src : dtypes) {
    array in = int_values_array(values, src);
    in.eval();
    for (Dtype dst : dtypes) {
      std::vector<uint8_t> expected = expected_cast_bytes(values, src, dst);
      std::optional<array> out;
      bool caught = false;
      std::string message;
      try {
        out = astype(in, dst, s);
        out->eval();
        omarchy::get_command_encoder(s).synchronize();
      } catch (const std::runtime_error& e) {
        caught = true;
        message = e.what();
      }
      if (caught) {
        // Only a device without shaderInt64 may refuse a leg, and only
        // a 64-bit leg; the refusal must carry the exact name.
        CHECK(message.find("[omarchy]") != std::string::npos);
        CHECK(message.find("dtype converting copy") != std::string::npos);
        bool sixty_four_leg = src == int64 || src == uint64 || dst == int64 ||
            dst == uint64;
        CHECK(sixty_four_leg);
        continue;
      }
      CHECK(std::memcmp(
                out->data<uint8_t>(), expected.data(), expected.size()) == 0);
    }
  }
}

TEST_CASE("byte dtype casts write packed edge words exactly") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = gpu_stream();

  // A uint8 view at byte offset 1 inside its parent: the cast must not
  // clobber the untouched lead or tail bytes of the edge words.
  std::vector<int64_t> vals = {100, 200, 0, 1, 255, 3, 7};
  array parent = zeros({9}, uint8, s);
  parent.eval();
  array view = slice(parent, {1}, {8}, {1}, s);
  view.eval();
  array src = int_values_array(vals, int32);
  src.eval();
  copy_gpu_inplace(
      src,
      view,
      src.shape(),
      src.strides(),
      view.strides(),
      0,
      0,
      CopyType::Vector,
      s);
  omarchy::get_command_encoder(s).synchronize();
  const auto* pp = parent.data<uint8_t>();
  CHECK(pp[0] == 0);
  for (size_t i = 0; i < vals.size(); ++i) {
    CHECK(pp[i + 1] == static_cast<uint8_t>(vals[i]));
  }
  CHECK(pp[8] == 0);

  // The same shape with a bool destination keeps the 0/1 canonical.
  array bparent = zeros({5}, bool_, s);
  bparent.eval();
  array bview = slice(bparent, {1}, {4}, {1}, s);
  bview.eval();
  array bsrc = int_values_array({0, 5, -2}, int32);
  bsrc.eval();
  copy_gpu_inplace(
      bsrc,
      bview,
      bsrc.shape(),
      bsrc.strides(),
      bview.strides(),
      0,
      0,
      CopyType::Vector,
      s);
  omarchy::get_command_encoder(s).synchronize();
  const auto* bp = bparent.data<bool>();
  CHECK(bp[0] == false);
  CHECK(bp[1] == false);
  CHECK(bp[2] == true);
  CHECK(bp[3] == true);
  CHECK(bp[4] == false);
}

TEST_CASE("nonzero scalar fills cover bool and the 8-bit integers") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = gpu_stream();

  array tb = full({2}, true, bool_, s);
  tb.eval();
  omarchy::get_command_encoder(s).synchronize();
  CHECK(tb.nbytes() == 2);
  const auto* bp = tb.data<bool>();
  CHECK(bp[0] == true);
  CHECK(bp[1] == true);

  // Five int8 bytes of 0xFD straddle a word boundary at offset 0.
  array fi = full({5}, -3, int8, s);
  fi.eval();
  omarchy::get_command_encoder(s).synchronize();
  const auto* ip = fi.data<int8_t>();
  for (int i = 0; i < 5; ++i) {
    CHECK(ip[i] == -3);
  }

  array fu = full({7}, 200, uint8, s);
  fu.eval();
  omarchy::get_command_encoder(s).synchronize();
  const auto* up = fu.data<uint8_t>();
  for (int i = 0; i < 7; ++i) {
    CHECK(up[i] == 200);
  }

  // A bool fill into an odd-offset view leaves the parent's edge
  // bytes untouched.
  array parent = zeros({6}, bool_, s);
  parent.eval();
  array v = slice(parent, {1}, {5}, {1}, s);
  v.eval();
  array t(true, bool_);
  copy_gpu_inplace(
      t, v, v.shape(), v.strides(), v.strides(), 0, 0, CopyType::Scalar, s);
  omarchy::get_command_encoder(s).synchronize();
  const auto* pp = parent.data<bool>();
  CHECK(pp[0] == false);
  CHECK(pp[1] == true);
  CHECK(pp[2] == true);
  CHECK(pp[3] == true);
  CHECK(pp[4] == true);
  CHECK(pp[5] == false);
}

TEST_CASE("arange covers uint32") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = gpu_stream();

  array a = arange(0.0, 10.0, 2.0, uint32, s);
  a.eval();
  omarchy::get_command_encoder(s).synchronize();
  CHECK(a.dtype() == uint32);
  const auto* p = a.data<uint32_t>();
  const uint32_t expected[5] = {0, 2, 4, 6, 8};
  REQUIRE(a.size() == 5);
  for (int i = 0; i < 5; ++i) {
    CHECK(p[i] == expected[i]);
  }

  array b = arange(4.0, 0.0, -1.0, uint32, s);
  b.eval();
  omarchy::get_command_encoder(s).synchronize();
  const auto* q = b.data<uint32_t>();
  const uint32_t down[4] = {4, 3, 2, 1};
  REQUIRE(b.size() == 4);
  for (int i = 0; i < 4; ++i) {
    CHECK(q[i] == down[i]);
  }
}

TEST_CASE("negative strides flip through the strided copy engine") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = gpu_stream();

  array x({0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f}, float32);
  x.eval();
  array r = flip(x, 0, s);
  array rf = full(r.shape(), r, s);
  rf.eval();
  omarchy::get_command_encoder(s).synchronize();
  const auto* rp = rf.data<float>();
  for (int i = 0; i < 8; ++i) {
    CHECK(rp[i] == static_cast<float>(7 - i));
  }

  // Two-dimensional flip on a non-zero axis.
  array m({0.f, 1.f, 2.f, 3.f, 4.f, 5.f}, float32);
  m = reshape(m, {2, 3}, s);
  m.eval();
  array mf = full(m.shape(), flip(m, 1, s), s);
  mf.eval();
  omarchy::get_command_encoder(s).synchronize();
  const auto* mp = mf.data<float>();
  const float mexpected[6] = {2.f, 1.f, 0.f, 5.f, 4.f, 3.f};
  for (int i = 0; i < 6; ++i) {
    CHECK(mp[i] == mexpected[i]);
  }

  // A reversed destination view: the negative stride sits on the
  // output side of a GeneralGeneral copy.
  array parent({9.f, 9.f, 9.f, 9.f, 9.f, 9.f, 9.f, 9.f}, float32);
  parent.eval();
  array dst = slice(parent, {4}, {1}, {-1}, s);
  dst.eval();
  array src({1.f, 2.f, 3.f}, float32);
  src.eval();
  copy_gpu_inplace(
      src,
      dst,
      src.shape(),
      src.strides(),
      dst.strides(),
      0,
      0,
      CopyType::GeneralGeneral,
      s);
  omarchy::get_command_encoder(s).synchronize();
  const auto* gp = parent.data<float>();
  CHECK(gp[0] == 9.f);
  CHECK(gp[1] == 9.f);
  CHECK(gp[2] == 3.f);
  CHECK(gp[3] == 2.f);
  CHECK(gp[4] == 1.f);
  CHECK(gp[5] == 9.f);
  CHECK(gp[6] == 9.f);
  CHECK(gp[7] == 9.f);
}

TEST_CASE("strided copy covers bool") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = gpu_stream();

  array b({1, 0, 1, 1, 0, 1}, bool_);
  b = reshape(b, {2, 3}, s);
  b.eval();
  array t = transpose(b, {1, 0}, s);
  array tm = full(t.shape(), t, s);
  tm.eval();
  omarchy::get_command_encoder(s).synchronize();
  REQUIRE(tm.dtype() == bool_);
  REQUIRE(tm.nbytes() == 6);
  const auto* tp = tm.data<bool>();
  const bool texpected[6] = {true, true, false, false, true, true};
  for (int i = 0; i < 6; ++i) {
    CHECK(tp[i] == texpected[i]);
  }
}

TEST_CASE("nonzero scalar fills cover bool and the 8-bit integers") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = gpu_stream();

  array tb = full({2}, true, bool_, s);
  tb.eval();
  omarchy::get_command_encoder(s).synchronize();
  CHECK(tb.nbytes() == 2);
  const auto* bp = tb.data<bool>();
  CHECK(bp[0] == true);
  CHECK(bp[1] == true);

  // Five int8 bytes of 0xFD straddle a word boundary at offset 0.
  array fi = full({5}, -3, int8, s);
  fi.eval();
  omarchy::get_command_encoder(s).synchronize();
  const auto* ip = fi.data<int8_t>();
  for (int i = 0; i < 5; ++i) {
    CHECK(ip[i] == -3);
  }

  array fu = full({7}, 200, uint8, s);
  fu.eval();
  omarchy::get_command_encoder(s).synchronize();
  const auto* up = fu.data<uint8_t>();
  for (int i = 0; i < 7; ++i) {
    CHECK(up[i] == 200);
  }

  // A bool fill into an odd-offset view leaves the parent's edge
  // bytes untouched.
  array parent = zeros({6}, bool_, s);
  parent.eval();
  array v = slice(parent, {1}, {5}, {1}, s);
  v.eval();
  array t(true, bool_);
  copy_gpu_inplace(
      t, v, v.shape(), v.strides(), v.strides(), 0, 0, CopyType::Scalar, s);
  omarchy::get_command_encoder(s).synchronize();
  const auto* pp = parent.data<bool>();
  CHECK(pp[0] == false);
  CHECK(pp[1] == true);
  CHECK(pp[2] == true);
  CHECK(pp[3] == true);
  CHECK(pp[4] == true);
  CHECK(pp[5] == false);
}

TEST_CASE("arange covers uint32") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = gpu_stream();

  array a = arange(0.0, 10.0, 2.0, uint32, s);
  a.eval();
  omarchy::get_command_encoder(s).synchronize();
  CHECK(a.dtype() == uint32);
  const auto* p = a.data<uint32_t>();
  const uint32_t expected[5] = {0, 2, 4, 6, 8};
  REQUIRE(a.size() == 5);
  for (int i = 0; i < 5; ++i) {
    CHECK(p[i] == expected[i]);
  }

  array b = arange(4.0, 0.0, -1.0, uint32, s);
  b.eval();
  omarchy::get_command_encoder(s).synchronize();
  const auto* q = b.data<uint32_t>();
  const uint32_t down[4] = {4, 3, 2, 1};
  REQUIRE(b.size() == 4);
  for (int i = 0; i < 4; ++i) {
    CHECK(q[i] == down[i]);
  }
}

TEST_CASE("negative strides flip through the strided copy engine") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = gpu_stream();

  array x({0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f}, float32);
  x.eval();
  array r = flip(x, 0, s);
  array rf = full(r.shape(), r, s);
  rf.eval();
  omarchy::get_command_encoder(s).synchronize();
  const auto* rp = rf.data<float>();
  for (int i = 0; i < 8; ++i) {
    CHECK(rp[i] == static_cast<float>(7 - i));
  }

  // Two-dimensional flip on a non-zero axis.
  array m({0.f, 1.f, 2.f, 3.f, 4.f, 5.f}, float32);
  m = reshape(m, {2, 3}, s);
  m.eval();
  array mf = full(m.shape(), flip(m, 1, s), s);
  mf.eval();
  omarchy::get_command_encoder(s).synchronize();
  const auto* mp = mf.data<float>();
  const float mexpected[6] = {2.f, 1.f, 0.f, 5.f, 4.f, 3.f};
  for (int i = 0; i < 6; ++i) {
    CHECK(mp[i] == mexpected[i]);
  }

  // A reversed destination view: the negative stride sits on the
  // output side of a GeneralGeneral copy.
  array parent({9.f, 9.f, 9.f, 9.f, 9.f, 9.f, 9.f, 9.f}, float32);
  parent.eval();
  array dst = slice(parent, {4}, {1}, {-1}, s);
  dst.eval();
  array src({1.f, 2.f, 3.f}, float32);
  src.eval();
  copy_gpu_inplace(
      src,
      dst,
      src.shape(),
      src.strides(),
      dst.strides(),
      0,
      0,
      CopyType::GeneralGeneral,
      s);
  omarchy::get_command_encoder(s).synchronize();
  const auto* gp = parent.data<float>();
  CHECK(gp[0] == 9.f);
  CHECK(gp[1] == 9.f);
  CHECK(gp[2] == 3.f);
  CHECK(gp[3] == 2.f);
  CHECK(gp[4] == 1.f);
  CHECK(gp[5] == 9.f);
  CHECK(gp[6] == 9.f);
  CHECK(gp[7] == 9.f);
}

TEST_CASE("strided copy covers bool") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = gpu_stream();

  array b({1, 0, 1, 1, 0, 1}, bool_);
  b = reshape(b, {2, 3}, s);
  b.eval();
  array t = transpose(b, {1, 0}, s);
  array tm = full(t.shape(), t, s);
  tm.eval();
  omarchy::get_command_encoder(s).synchronize();
  REQUIRE(tm.dtype() == bool_);
  REQUIRE(tm.nbytes() == 6);
  const auto* tp = tm.data<bool>();
  const bool texpected[6] = {true, true, false, false, true, true};
  for (int i = 0; i < 6; ++i) {
    CHECK(tp[i] == texpected[i]);
  }
}

TEST_CASE("negative strides flip through the strided copy engine") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = gpu_stream();

  array x({0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f}, float32);
  x.eval();
  array r = flip(x, 0, s);
  array rf = full(r.shape(), r, s);
  rf.eval();
  omarchy::get_command_encoder(s).synchronize();
  const auto* rp = rf.data<float>();
  for (int i = 0; i < 8; ++i) {
    CHECK(rp[i] == static_cast<float>(7 - i));
}

}
TEST_CASE("nonzero scalar fills cover bool and the 8-bit integers") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = gpu_stream();

  array tb = full({2}, true, bool_, s);
  tb.eval();
  omarchy::get_command_encoder(s).synchronize();
  CHECK(tb.nbytes() == 2);
  const auto* bp = tb.data<bool>();
  CHECK(bp[0] == true);
  CHECK(bp[1] == true);

  // Five int8 bytes of 0xFD straddle a word boundary at offset 0.
  array fi = full({5}, -3, int8, s);
  fi.eval();
  omarchy::get_command_encoder(s).synchronize();
  const auto* ip = fi.data<int8_t>();
  for (int i = 0; i < 5; ++i) {
    CHECK(ip[i] == -3);
  }

  array fu = full({7}, 200, uint8, s);
  fu.eval();
  omarchy::get_command_encoder(s).synchronize();
  const auto* up = fu.data<uint8_t>();
  for (int i = 0; i < 7; ++i) {
    CHECK(up[i] == 200);
  }

  // A bool fill into an odd-offset view leaves the parent's edge
  // bytes untouched.
  array parent = zeros({6}, bool_, s);
  parent.eval();
  array v = slice(parent, {1}, {5}, {1}, s);
  v.eval();
  array t(true, bool_);
  copy_gpu_inplace(
      t, v, v.shape(), v.strides(), v.strides(), 0, 0, CopyType::Scalar, s);
  omarchy::get_command_encoder(s).synchronize();
  const auto* pp = parent.data<bool>();
  CHECK(pp[0] == false);
  CHECK(pp[1] == true);
  CHECK(pp[2] == true);
  CHECK(pp[3] == true);
  CHECK(pp[4] == true);
  CHECK(pp[5] == false);
}

TEST_CASE("arange covers uint32") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = gpu_stream();

  array a = arange(0.0, 10.0, 2.0, uint32, s);
  a.eval();
  omarchy::get_command_encoder(s).synchronize();
  CHECK(a.dtype() == uint32);
  const auto* p = a.data<uint32_t>();
  const uint32_t expected[5] = {0, 2, 4, 6, 8};
  REQUIRE(a.size() == 5);
  for (int i = 0; i < 5; ++i) {
    CHECK(p[i] == expected[i]);
  }

  array b = arange(4.0, 0.0, -1.0, uint32, s);
  b.eval();
  omarchy::get_command_encoder(s).synchronize();
  const auto* q = b.data<uint32_t>();
  const uint32_t down[4] = {4, 3, 2, 1};
  REQUIRE(b.size() == 4);
  for (int i = 0; i < 4; ++i) {
    CHECK(q[i] == down[i]);
  }
}

TEST_CASE("negative strides flip through the strided copy engine") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = gpu_stream();

  array x({0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f}, float32);
  x.eval();
  array r = flip(x, 0, s);
  array rf = full(r.shape(), r, s);
  rf.eval();
  omarchy::get_command_encoder(s).synchronize();
  const auto* rp = rf.data<float>();
  for (int i = 0; i < 8; ++i) {
    CHECK(rp[i] == static_cast<float>(7 - i));
  }

  // Two-dimensional flip on a non-zero axis.
  array m({0.f, 1.f, 2.f, 3.f, 4.f, 5.f}, float32);
  m = reshape(m, {2, 3}, s);
  m.eval();
  array mf = full(m.shape(), flip(m, 1, s), s);
  mf.eval();
  omarchy::get_command_encoder(s).synchronize();
  const auto* mp = mf.data<float>();
  const float mexpected[6] = {2.f, 1.f, 0.f, 5.f, 4.f, 3.f};
  for (int i = 0; i < 6; ++i) {
    CHECK(mp[i] == mexpected[i]);
  }

  // A reversed destination view: the negative stride sits on the
  // output side of a GeneralGeneral copy.
  array parent({9.f, 9.f, 9.f, 9.f, 9.f, 9.f, 9.f, 9.f}, float32);
  parent.eval();
  array dst = slice(parent, {4}, {1}, {-1}, s);
  dst.eval();
  array src({1.f, 2.f, 3.f}, float32);
  src.eval();
  copy_gpu_inplace(
      src,
      dst,
      src.shape(),
      src.strides(),
      dst.strides(),
      0,
      0,
      CopyType::GeneralGeneral,
      s);
  omarchy::get_command_encoder(s).synchronize();
  const auto* gp = parent.data<float>();
  CHECK(gp[0] == 9.f);
  CHECK(gp[1] == 9.f);
  CHECK(gp[2] == 3.f);
  CHECK(gp[3] == 2.f);
  CHECK(gp[4] == 1.f);
  CHECK(gp[5] == 9.f);
  CHECK(gp[6] == 9.f);
  CHECK(gp[7] == 9.f);
}

TEST_CASE("strided copy covers bool") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = gpu_stream();

  array b({1, 0, 1, 1, 0, 1}, bool_);
  b = reshape(b, {2, 3}, s);
  b.eval();
  array t = transpose(b, {1, 0}, s);
  array tm = full(t.shape(), t, s);
  tm.eval();
  omarchy::get_command_encoder(s).synchronize();
  REQUIRE(tm.dtype() == bool_);
  REQUIRE(tm.nbytes() == 6);
  const auto* tp = tm.data<bool>();
  const bool texpected[6] = {true, true, false, false, true, true};
  for (int i = 0; i < 6; ++i) {
    CHECK(tp[i] == texpected[i]);
  }
}
