// Throwaway: bit-compare the grid rope / copy paths against the 1-D paths.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "mlx/mlx.h"
#include "mlx/backend/omarchy/trace.h"

using namespace mlx::core;

static int failures = 0;

static std::vector<uint8_t> bytes(array a) {
  eval(a);
  std::vector<uint8_t> out(a.nbytes());
  std::memcpy(out.data(), a.data<uint8_t>(), a.nbytes());
  return out;
}

template <typename F>
static void compare(const char* label, F make) {
  setenv("MLX_OMARCHY_ROPE_GRID", "0", 1);
  setenv("MLX_OMARCHY_COPY_GRID", "0", 1);
  setenv("MLX_OMARCHY_RMS_NORM_SUBGROUP", "0", 1);
  auto base = bytes(make());
  setenv("MLX_OMARCHY_ROPE_GRID", "1", 1);
  setenv("MLX_OMARCHY_COPY_GRID", "1", 1);
  setenv("MLX_OMARCHY_RMS_NORM_SUBGROUP", "1", 1);
  auto grid = bytes(make());
  bool ok = base == grid;
  if (!ok) {
    failures++;
    size_t first = 0;
    while (first < base.size() && base[first] == grid[first]) first++;
    std::printf("FAIL %s: %zu bytes, first diff at %zu\n", label, base.size(), first);
  } else {
    std::printf("ok   %s (%zu bytes)\n", label, base.size());
  }
}

int main() {
  set_default_device(Device::gpu);
  random::seed(7);
  for (auto dtype : {float16, float32, bfloat16}) {
    const char* dn = dtype == float16 ? "f16" : dtype == float32 ? "f32" : "bf16";
    for (auto shape : std::vector<Shape>{{1, 14, 1, 64}, {1, 2, 1, 64}, {1, 14, 41, 64}, {2, 14, 5, 64}, {3, 4, 300, 128}, {1, 1, 1, 64}}) {
      for (bool traditional : {false, true}) {
        char label[128];
        std::snprintf(label, sizeof label, "rope %s %dx%dx%dx%d trad=%d", dn, shape[0], shape[1], shape[2], shape[3], traditional);
        array x = astype(random::normal(shape), dtype);
        eval(x);
        compare(label, [&]() { return fast::rope(x, shape[3], traditional, 10000.0f, 1.0f, 17); });
        std::snprintf(label, sizeof label, "rope %s %dx%dx%dx%d trad=%d scale", dn, shape[0], shape[1], shape[2], shape[3], traditional);
        compare(label, [&]() { return fast::rope(x, shape[3], traditional, 500000.0f, 0.5f, 3); });
        // freqs variant
        array freqs = astype(arange(1, shape[3] / 2 + 1), float32);
        eval(freqs);
        std::snprintf(label, sizeof label, "rope-freqs %s %dx%dx%dx%d trad=%d", dn, shape[0], shape[1], shape[2], shape[3], traditional);
        compare(label, [&]() { return fast::rope(x, shape[3], traditional, std::nullopt, 1.0f, 5, freqs); });
        // partial rotation (passthrough) must take the 1-D path either way
        std::snprintf(label, sizeof label, "rope-partial %s %dx%dx%dx%d trad=%d", dn, shape[0], shape[1], shape[2], shape[3], traditional);
        compare(label, [&]() { return fast::rope(x, shape[3] / 2, traditional, 10000.0f, 1.0f, 2); });
      }
      // head/seq transposed input (B, T, N, D) viewed as (B, N, T, D)
      if (shape[0] * shape[1] * shape[2] > 1) {
        array y = astype(random::normal(Shape{shape[0], shape[2], shape[1], shape[3]}), dtype);
        eval(y);
        array yt = transpose(y, {0, 2, 1, 3});
        char label[128];
        std::snprintf(label, sizeof label, "rope-transposed %s %dx%dx%dx%d", dn, shape[0], shape[1], shape[2], shape[3]);
        compare(label, [&]() { return fast::rope(yt, shape[3], false, 10000.0f, 1.0f, 9); });
      }
    }
    // 3-D input
    {
      array x = astype(random::normal(Shape{2, 7, 64}), dtype);
      eval(x);
      char label[128];
      std::snprintf(label, sizeof label, "rope-3d %s", dn);
      compare(label, [&]() { return fast::rope(x, 64, false, 10000.0f, 1.0f, 1); });
    }
  }
  // Copies: KV-cache row paste and other slice updates
  for (auto dtype : {float16, float32, bfloat16, int32, uint16}) {
    for (int B : {1, 2}) {
      array cache = zeros({B, 2, 256, 64}, dtype);
      array upd = astype(random::normal(Shape{B, 2, 1, 64}), dtype);
      array upd41 = astype(random::normal(Shape{B, 2, 41, 64}), dtype);
      eval(cache, upd, upd41);
      char label[128];
      std::snprintf(label, sizeof label, "kv-row-update dtype=%d B=%d", (int)dtype.val(), B);
      compare(label, [&]() { return slice_update(cache, upd, Shape{0, 0, 53, 0}, Shape{B, 2, 54, 64}); });
      std::snprintf(label, sizeof label, "kv-41-update dtype=%d B=%d", (int)dtype.val(), B);
      compare(label, [&]() { return slice_update(cache, upd41, Shape{0, 0, 0, 0}, Shape{B, 2, 41, 64}); });
      std::snprintf(label, sizeof label, "kv-41-update-strided dtype=%d B=%d", (int)dtype.val(), B);
      compare(label, [&]() { return slice_update(cache, upd41, Shape{0, 0, 3, 0}, Shape{B, 2, 85, 64}, Shape{1, 1, 2, 1}); });
      // rank-1: contiguous 1-D window
      array v = zeros({4096}, dtype);
      array w = astype(random::normal(Shape{100}), dtype);
      eval(v, w);
      std::snprintf(label, sizeof label, "1d-update dtype=%d", (int)dtype.val());
      compare(label, [&]() { return slice_update(v, w, Shape{17}, Shape{117}); });
      // rank-2 transposed gather
      array m = astype(random::normal(Shape{64, 37}), dtype);
      eval(m);
      std::snprintf(label, sizeof label, "transpose-copy dtype=%d", (int)dtype.val());
      compare(label, [&]() { return contiguous(transpose(m)); });
      // column slice update (inner stride > 1)
      array col = astype(random::normal(Shape{64, 3}), dtype);
      eval(col);
      std::snprintf(label, sizeof label, "col-update dtype=%d", (int)dtype.val());
      compare(label, [&]() { return slice_update(m, col, Shape{0, 5}, Shape{64, 8}); });
    }
  }
  // rms_norm: subgroup tree finish vs shared-memory tree
  for (auto dtype : {float16, float32, bfloat16}) {
    for (auto shape : std::vector<Shape>{{1, 896}, {1, 64}, {1, 1}, {3, 4096}, {2, 5000}, {7, 3, 896}, {1, 151936}}) {
      array x = astype(random::normal(shape) * 3.0f, dtype);
      array w = astype(random::normal(Shape{shape.back()}), dtype);
      array ws = astype(full(Shape{1}, 1.5f), dtype);
      eval(x, w, ws);
      char label[128];
      std::snprintf(label, sizeof label, "rms_norm dtype=%d rows=%d len=%d", (int)dtype.val(), (int)(x.size() / shape.back()), shape.back());
      compare(label, [&]() { return fast::rms_norm(x, w, 1e-5f); });
      std::snprintf(label, sizeof label, "rms_norm-scalar-w dtype=%d rows=%d len=%d", (int)dtype.val(), (int)(x.size() / shape.back()), shape.back());
      compare(label, [&]() { return fast::rms_norm(x, std::nullopt, 1e-6f); });
    }
  }
  // copy_gpu donation: a chained in-place KV-cache pattern must equal the
  // CPU result, and a still-referenced input must stay intact.
  for (auto dtype : {float16, float32, int32}) {
    array cache = zeros({1, 2, 64, 16}, dtype);
    array cache_cpu = zeros({1, 2, 64, 16}, dtype, Device::cpu);
    eval(cache);
    uint64_t copies0 = mlx::core::omarchy::trace::counters().vk_buffer_copies.load();
    for (int t = 0; t < 40; ++t) {
      array upd = astype(random::normal(Shape{1, 2, 1, 16}), dtype);
      eval(upd);
      array upd_cpu = astype(upd, dtype, Device::cpu);
      cache = slice_update(cache, upd, Shape{0, 0, t, 0}, Shape{1, 2, t + 1, 16});
      cache_cpu = slice_update(cache_cpu, upd_cpu, Shape{0, 0, t, 0}, Shape{1, 2, t + 1, 16}, Device::cpu);
    }
    eval(cache, cache_cpu);
    uint64_t copies = mlx::core::omarchy::trace::counters().vk_buffer_copies.load() - copies0;
    bool same = bytes(cache) == bytes(cache_cpu);
    std::printf("%s donation-chain dtype=%d vk_buffer_copies=%llu (40 slice_updates)\n", same ? "ok  " : "FAIL", (int)dtype.val(), (unsigned long long)copies);
    failures += !same;
    array a = zeros({2, 8, 16}, dtype);
    eval(a);
    array a_before = a;
    array u = astype(random::normal(Shape{2, 1, 16}), dtype);
    eval(u);
    array b = slice_update(a, u, Shape{0, 3, 0}, Shape{2, 4, 16});
    eval(b);
    bool intact = bytes(a) == bytes(zeros({2, 8, 16}, dtype, Device::cpu));
    bool pasted = bytes(contiguous(slice(b, Shape{0, 3, 0}, Shape{2, 4, 16}))) == bytes(u);
    std::printf("%s donation-refcount dtype=%d intact=%d pasted=%d\n", (intact && pasted) ? "ok  " : "FAIL", (int)dtype.val(), intact, pasted);
    failures += !(intact && pasted);
  }
  std::printf("failures=%d\n", failures);
  return failures ? 1 : 0;
}
