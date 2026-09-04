// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Throwaway accuracy probe for the fused-vs-composed RoPE verdict on
// Honeykrisp. NOT part of the battery: it writes bit-level dumps of the
// three M1-failing configurations so a host float64 reference can score
// both paths element by element. Dump format: one "SECTION COUNT" header,
// then one "f32bits value" line per element (the f32 image of float16 and
// bfloat16 values is exact and injective, so hex f32 bits are the stored
// value).
//
// Sections per config, in order:
//   input_<name>        the input tensor as stored
//   fused_<name>        fast::rope output
//   composed_<name>     composed_rope output (the battery's reference)
//   fused_trig_<name>   unit-input probe: pairs (1,0) through the fused
//                       path emit its stored cos in the first half of the
//                       rotated axis and its stored sin in the second
//   composed_trig_<name>  the same through the composed path
// Plus shared intermediates of the composed graph:
//   positions_<name>, inv_freqs_<name>, theta_<name>, freqs (config A).
#include <cmath>
#include <variant>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

using namespace mlx::core;

namespace {

// Verbatim from test_fast_ops.cpp pattern(): LCG values in [-1, 1).
std::vector<float> pattern(size_t count, uint32_t seed) {
  std::vector<float> values;
  values.reserve(count);
  uint32_t state = seed;
  for (size_t index = 0; index < count; ++index) {
    state = state * 1664525u + 1013904223u;
    values.push_back(
        static_cast<float>(static_cast<double>(state % 20000u) / 10000.0) -
        1.0f);
  }
  return values;
}

array rope_input(const Shape& shape, uint32_t seed, Dtype dtype, Stream s) {
  size_t count = 1;
  for (auto dim : shape) {
    count *= static_cast<size_t>(dim);
  }
  auto values = pattern(count, seed);
  array flat_values(values.begin(), Shape{static_cast<int>(values.size())}, float32);
  return astype(reshape(flat_values, shape, s), dtype, s);
}

// Verbatim algebra from test_fast_ops.cpp composed_rope().
array composed_rope(
    const array& x_in,
    int dims,
    bool traditional,
    float base,
    float scale,
    const array& offset,
    const std::optional<array>& freqs,
    bool forward,
    Stream s) {
  array x = x_in;
  auto shape = x.shape();
  if (x.ndim() == 3) {
    x = expand_dims(x, 1, s);
  } else if (x.ndim() > 4) {
    x = flatten(x, 1, 1 + (x.ndim() - 4), s);
  }
  auto B = x.shape(0);
  auto N = x.shape(1);
  auto T = x.shape(2);
  auto t = x.dtype();
  auto half_dims = dims / 2;
  auto off = offset;
  if (off.size() > 1) {
    off = expand_dims(off, std::vector<int>{-1, -2}, s);
  }
  auto positions = multiply(
      add(arange(x.shape(2), float32, s), off, s), array(scale, float32), s);
  auto inv_freqs =
      freqs ? reciprocal(*freqs, s)
            : exp(
                  multiply(
                      arange(0, -half_dims, -1, float32, s),
                      array(std::log(base) / half_dims, float32),
                      s),
                  s);
  auto theta = multiply(expand_dims(positions, -1, s), inv_freqs, s);
  auto coss = astype(cos(theta, s), t, s);
  auto sins = astype(sin(theta, s), t, s);
  auto apply_rope = [&](const array& x1, const array& x2) {
    std::vector<array> outs;
    if (forward) {
      outs.push_back(
          subtract(multiply(x1, coss, s), multiply(x2, sins, s), s));
      outs.push_back(add(multiply(x1, sins, s), multiply(x2, coss, s), s));
    } else {
      outs.push_back(add(multiply(x2, sins, s), multiply(x1, coss, s), s));
      outs.push_back(
          subtract(multiply(x2, coss, s), multiply(x1, sins, s), s));
    }
    return outs;
  };
  if (traditional) {
    auto x1 = slice(x, {0, 0, 0, 0}, {B, N, T, dims}, {1, 1, 1, 2}, s);
    auto x2 = slice(x, {0, 0, 0, 1}, {B, N, T, dims}, {1, 1, 1, 2}, s);
    auto outs = apply_rope(x1, x2);
    for (auto& o : outs) {
      o = expand_dims(o, -1, s);
    }
    auto out = reshape(concatenate(outs, -1, s), {B, N, T, dims}, s);
    if (dims < x.shape(-1)) {
      out =
          concatenate({out, slice(x, {0, 0, 0, dims}, x.shape(), s)}, -1, s);
    }
    return reshape(out, shape, s);
  } else {
    auto out_s = x.shape();
    out_s.back() = half_dims;
    auto x1 = slice(x, {0, 0, 0, 0}, out_s, s);
    out_s.back() = dims;
    auto x2 = slice(x, {0, 0, 0, half_dims}, out_s, s);
    auto outs = apply_rope(x1, x2);
    if (dims < x.shape(-1)) {
      outs.push_back(slice(x, {0, 0, 0, dims}, x.shape(), s));
    }
    return reshape(concatenate(outs, -1, s), shape, s);
  }
}

std::vector<float> flat(const array& value, Stream stream) {
  array copy = astype(value, float32, stream);
  copy.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const float* data = copy.data<float>();
  return std::vector<float>(data, data + copy.size());
}

void dump(std::ostream& out, const std::string& label, const std::vector<float>& v) {
  out << label << " " << v.size() << "\n";
  for (float f : v) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    char buf[16];
    std::snprintf(buf, sizeof buf, "%08x", bits);
    out << buf << " " << f << "\n";
  }
}

// Unit-input probe: rotate a tensor whose pair values are (1, 0). The
// forward rotation out1 = 1*cos - 0*sin = stored cos and
// out2 = 1*sin + 0*cos = stored sin, exactly, in both paths (no product
// rounding can change a 1.0f multiply or a zero term, and contraction
// cannot touch it either).
array unit_pairs(const Shape& shape, bool traditional, Dtype dtype, Stream s) {
  int D = shape.back();
  size_t count = 1;
  for (auto dim : shape) {
    count *= static_cast<size_t>(dim);
  }
  std::vector<float> values(count, 0.0f);
  int half = D / 2;
  for (size_t row = 0; row < count / static_cast<size_t>(D); ++row) {
    for (int d = 0; d < D; ++d) {
      bool one = traditional ? ((d % 2) == 0 && d < 2 * half) : (d < half);
      values[row * static_cast<size_t>(D) + static_cast<size_t>(d)] =
          one ? 1.0f : 0.0f;
    }
  }
  array flat_values(values.begin(), Shape{static_cast<int>(values.size())}, float32);
  return astype(reshape(flat_values, shape, s), dtype, s);
}

struct Config {
  const char* name;
  Shape shape;
  int dims;
  bool traditional;
  bool with_freqs;
  float base;
  int offset_value;
  uint32_t seed;
  Dtype dtype;
};

void run_config(std::ostream& out, const Config& cfg, Stream stream) {
  array x = rope_input(cfg.shape, cfg.seed, cfg.dtype, stream);
  array offset = array(cfg.offset_value, int32);
  std::optional<array> freqs;
  if (cfg.with_freqs) {
    freqs = exp(
        multiply(
            arange(0, -cfg.dims / 2, -1, float32, stream),
            array(std::log(10000.0f) / (cfg.dims / 2), float32),
            stream),
        stream);
  }
  auto got = fast::rope(
      x,
      cfg.dims,
      cfg.traditional,
      cfg.with_freqs ? std::nullopt : std::optional<float>(cfg.base),
      1.0f,
      offset,
      freqs,
      stream);
  auto want = composed_rope(
      x,
      cfg.dims,
      cfg.traditional,
      cfg.base,
      1.0f,
      offset,
      freqs,
      true,
      stream);
  dump(out, std::string("input_") + cfg.name, flat(x, stream));
  dump(out, std::string("fused_") + cfg.name, flat(got, stream));
  dump(out, std::string("composed_") + cfg.name, flat(want, stream));

  // Trig probe on the same theta grid.
  array xu = unit_pairs(cfg.shape, cfg.traditional, cfg.dtype, stream);
  auto got_trig = fast::rope(
      xu,
      cfg.dims,
      cfg.traditional,
      cfg.with_freqs ? std::nullopt : std::optional<float>(cfg.base),
      1.0f,
      offset,
      freqs,
      stream);
  auto want_trig = composed_rope(
      xu,
      cfg.dims,
      cfg.traditional,
      cfg.base,
      1.0f,
      offset,
      freqs,
      true,
      stream);
  dump(out, std::string("fused_trig_") + cfg.name, flat(got_trig, stream));
  dump(out, std::string("composed_trig_") + cfg.name, flat(want_trig, stream));

  // Composed-graph intermediates (f32, before the astype to storage).
  int half_dims = cfg.dims / 2;
  int T = cfg.shape[cfg.shape.size() - 2];
  if (cfg.with_freqs) {
    dump(out, "freqs", flat(*freqs, stream));
  }
  auto positions = multiply(
      add(arange(T, float32, stream),
          array(static_cast<float>(cfg.offset_value), float32),
          stream),
      array(1.0f, float32),
      stream);
  auto inv_freqs =
      freqs ? reciprocal(*freqs, stream)
            : exp(
                  multiply(
                      arange(0, -half_dims, -1, float32, stream),
                      array(std::log(cfg.base) / half_dims, float32),
                      stream),
                  stream);
  auto theta =
      multiply(expand_dims(positions, -1, stream), inv_freqs, stream);
  dump(
      out,
      std::string("positions_") + cfg.name,
      flat(positions, stream));
  dump(
      out,
      std::string("inv_freqs_") + cfg.name,
      flat(inv_freqs, stream));
  dump(out, std::string("theta_") + cfg.name, flat(theta, stream));

  // In-probe summary of fused vs composed.
  auto got_v = flat(got, stream);
  auto want_v = flat(want, stream);
  double max_diff = 0.0;
  size_t diffs = 0;
  int D = cfg.shape.back();
  size_t tail_count = 0;
  double tail_max = 0.0;
  size_t tail_diffs = 0;
  for (size_t i = 0; i < got_v.size(); ++i) {
    double diff = std::abs(static_cast<double>(got_v[i]) - want_v[i]);
    bool tail = static_cast<int>(i % static_cast<size_t>(D)) >= cfg.dims;
    if (tail) {
      tail_count++;
      tail_max = std::max(tail_max, diff);
      if (got_v[i] != want_v[i]) {
        tail_diffs++;
      }
    } else {
      max_diff = std::max(max_diff, diff);
      if (got_v[i] != want_v[i]) {
        diffs++;
      }
    }
  }
  std::cout << "[" << cfg.name << "] rotated: " << diffs << " differing, max "
            << max_diff << "; tail: " << tail_diffs << " differing of "
            << tail_count << ", max " << tail_max << "\n";
}

} // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: rope_probe <dump-file>\n";
    return 2;
  }
  std::ofstream out(argv[1]);
  if (!out) {
    std::cerr << "cannot open " << argv[1] << "\n";
    return 2;
  }
  set_default_device(Device::gpu);
  Stream stream = new_stream(Device::gpu);
  for (const auto& [key, value] : gpu::device_info()) {
    const auto* name = std::get_if<std::string>(&value);
    if (name != nullptr) {
      std::cout << "device " << key << ": " << *name << "\n";
    }
  }

  // Variant 2 f32: half-split, freqs buffer, shape (2,3,7,16), seed 101.
  run_config(
      out,
      Config{
          "variant2_f32",
          Shape{2, 3, 7, 16},
          16,
          false,
          true,
          10000.0f,
          3,
          101,
          float32},
      stream);
  // Decode q f16: (1,14,1,128), dims 128, base 500000, offset 17, seed 103.
  run_config(
      out,
      Config{
          "decode_q_f16",
          Shape{1, 14, 1, 128},
          128,
          false,
          false,
          500000.0f,
          17,
          103,
          float16},
      stream);
  // Decode k f16: (1,2,1,64), dims 64, base 500000, offset 17, seed 107.
  run_config(
      out,
      Config{
          "decode_k_f16",
          Shape{1, 2, 1, 64},
          64,
          false,
          false,
          500000.0f,
          17,
          107,
          float16},
      stream);
  // Partial dims f32 then f16: (2,3,5,32), dims 16 traditional, offset 2.
  run_config(
      out,
      Config{
          "partial_f32",
          Shape{2, 3, 5, 32},
          16,
          true,
          false,
          10000.0f,
          2,
          109,
          float32},
      stream);
  run_config(
      out,
      Config{
          "partial_f16",
          Shape{2, 3, 5, 32},
          16,
          true,
          false,
          10000.0f,
          2,
          109,
          float16},
      stream);
  // Position sweep: RoPE's angle is position * inv_freq, so any range-
  // reduction defect must grow with position. Measure both paths across
  // the documented band (1e3..1e5; the failing variant2 config sits
  // inside it at theta <= 28460). The freqs variant amplifies inv_freq
  // to 3162, so positions must stay <= ~31 to avoid the 1e5 trig gate
  // (we probe position 32 below to demonstrate the gate, not the kernel,
  // refuses as documented in known-defects.md).
  struct Sweep {
    const char* tag;
    Dtype dtype;
    std::vector<int> positions;
    int dims;
    float base;
    bool with_freqs;
  };
  Sweep sweeps[] = {
      // Base variant, Qwen 0.5B decode shape: dims 128, base 500000.
      // Covers positions 1 (small angle) through 32000 (inside the
      // documented 5e-3 absolute envelope band).
      {"base_f16", float16, {1, 37, 1000, 8000, 32000}, 128, 500000.0f, false},
      {"base_f32", float32, {1, 37, 1000, 8000, 32000}, 128, 500000.0f, false},
  };
  for (const Sweep& sw : sweeps) {
    std::optional<array> sw_freqs;
    int half = sw.dims / 2;
    if (sw.with_freqs) {
      sw_freqs = exp(
          multiply(
              arange(0, -half, -1, float32, stream),
              array(std::log(10000.0f) / half, float32),
              stream),
          stream);
    }
    for (int pos : sw.positions) {
      Shape shape{1, 1, 1, sw.dims};
      array x = rope_input(shape, 999, sw.dtype, stream);
      array off = array(pos, int32);
      try {
        auto got = fast::rope(
            x,
            sw.dims,
            false,
            sw.with_freqs ? std::nullopt : std::optional<float>(sw.base),
            1.0f,
            off,
            sw_freqs,
            stream);
        auto want = composed_rope(
            x,
            sw.dims,
            false,
            sw.base,
            1.0f,
            off,
            sw_freqs,
            true,
            stream);
        auto gv = flat(got, stream);
        auto wv = flat(want, stream);
        double max_diff = 0.0;
        for (size_t i = 0; i < gv.size(); ++i) {
          max_diff = std::max(
              max_diff, std::abs(static_cast<double>(gv[i]) - wv[i]));
        }
        char sec[96];
        std::snprintf(
            sec,
            sizeof(sec),
            "sweep_%s_pos%d_d%d",
            sw.tag,
            pos,
            sw.dims);
        dump(out, std::string("fused_") + sec, gv);
        dump(out, std::string("composed_") + sec, wv);
        dump(out, std::string("input_") + sec, flat(x, stream));
        std::cout << "[" << sw.tag << "@pos" << pos
                  << "] max|fused-composed|=" << max_diff << "\n";
      } catch (const std::exception& e) {
        // The trig gate refuses above theta 1e5; record that here so
        // the analysis can distinguish gate-refusal from kernel-error.
        std::cout << "[" << sw.tag << "@pos" << pos
                  << "] refused: " << e.what() << "\n";
      }
    }
  }

  // Gate-boundary probe (freqs variant): dims 16, inv_freq max 3162;
  // position_max = floor(1e5 / 3162) = 31. Position 32 must refuse on
  // both paths; positions 1, 9, 25, 31 must run and the error-vs-truth
  // curve discriminates the mechanism.
  {
    array sw_freqs = exp(
        multiply(
            arange(0, -8, -1, float32, stream),
            array(std::log(10000.0f) / 8, float32),
            stream),
        stream);
    int positions[] = {1, 9, 25, 31, 32};
    for (int pos : positions) {
      Shape shape{1, 1, 1, 16};
      array x = rope_input(shape, 991, float32, stream);
      array off = array(pos, int32);
      try {
        auto got = fast::rope(
            x, 16, false, std::nullopt, 1.0f, off,
            std::optional<array>(sw_freqs), stream);
        auto want = composed_rope(
            x, 16, false, 10000.0f, 1.0f, off,
            std::optional<array>(sw_freqs), true, stream);
        auto gv = flat(got, stream);
        auto wv = flat(want, stream);
        double max_diff = 0.0;
        for (size_t i = 0; i < gv.size(); ++i) {
          max_diff = std::max(
              max_diff, std::abs(static_cast<double>(gv[i]) - wv[i]));
        }
        char sec[64];
        std::snprintf(sec, sizeof(sec), "sweep_freqs_f32_pos%d", pos);
        dump(out, std::string("fused_") + sec, gv);
        dump(out, std::string("composed_") + sec, wv);
        dump(out, std::string("input_") + sec, flat(x, stream));
        std::cout << "[freqs_f32@pos" << pos
                  << "] max|fused-composed|=" << max_diff << "\n";
      } catch (const std::exception& e) {
        std::cout << "[freqs_f32@pos" << pos
                  << "] refused: " << e.what() << "\n";
      }
    }
  }

   std::cout << "probe done\n";
   return 0;
 }
