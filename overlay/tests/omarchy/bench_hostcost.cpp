// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Fixed-chain host-cost benchmark for the Omarchy encoder.
//
// Runs a dependent chain of N elementwise dispatches, R times, on the GPU
// stream. The chain shape reproduces eager decode's host structure: one
// dispatch per op, an eager finalize per op, one submission per op, a
// begin per submission (the ring never spans two chain steps), and a join
// when the chain materializes. Run it with MLX_OMARCHY_GPU_PROFILE=<path>
// so the NDJSON harness records the per-phase breakdown; the analysis
// script (scripts/profile_analyze.py) ranks phases from that stream.
//
// Host wall time here is a DEV-BOX number (llvmpipe): host STRUCTURE is
// comparable across drivers, wall times are not product numbers.

#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"

using namespace mlx::core;

namespace {

struct Args {
  int chain = 64;
  int reps = 30;
  int elems = 1024;
  bool bf16 = false;
  bool check = false;
  // dep: each add consumes the previous output (strictly dependent
  // chain). indep: N adds from the same input, evaluated together (an
  // independent fragment the evaluator can batch into one submission).
  bool independent = false;
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    auto need = [&](const char* name) -> int {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", name);
        std::exit(2);
      }
      return std::atoi(argv[++i]);
    };
    if (std::strcmp(argv[i], "--chain") == 0) {
      args.chain = need("--chain");
    } else if (std::strcmp(argv[i], "--reps") == 0) {
      args.reps = need("--reps");
    } else if (std::strcmp(argv[i], "--elems") == 0) {
      args.elems = need("--elems");
    } else if (std::strcmp(argv[i], "--dtype") == 0 && i + 1 < argc) {
      args.bf16 = std::strcmp(argv[++i], "bf16") == 0;
    } else if (std::strcmp(argv[i], "--shape") == 0 && i + 1 < argc) {
      std::string shape = argv[++i];
      if (shape == "dep") {
        args.independent = false;
      } else if (shape == "indep") {
        args.independent = true;
      } else {
        std::fprintf(stderr, "shape must be dep or indep\n");
        std::exit(2);
      }
    } else if (std::strcmp(argv[i], "--check") == 0) {
      args.check = true;
    } else {
      std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
      std::exit(2);
    }
  }
  if (args.chain <= 0 || args.reps <= 0 || args.elems <= 0) {
    std::fprintf(stderr, "chain, reps, elems must be positive\n");
    std::exit(2);
  }
  return args;
}

inline uint64_t ns_since(
    std::chrono::steady_clock::time_point t0) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - t0)
          .count());
}

} // namespace

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  if (!gpu::is_available()) {
    std::fprintf(
        stderr,
        "no qualifying Vulkan device (set MLX_OMARCHY_ALLOW_NON_APPLE=1).\n");
    return 1;
  }
  set_default_device(Device::gpu);
  Stream stream = new_stream(Device::gpu);

  Dtype dtype = args.bf16 ? bfloat16 : float32;
  array x = ones({args.elems}, dtype, stream);
  array y = zeros({args.elems}, dtype, stream);
  eval(x);

  // Warm every lazily-built path: kernels, pipelines, descriptor pool,
  // allocator. Two short chains, fully joined.
  for (int warm = 0; warm < 2; ++warm) {
    array w = y;
    for (int i = 0; i < 2; ++i) {
      w = add(w, x, stream);
    }
    eval(w);
    omarchy::get_command_encoder(stream).synchronize();
  }

  if (args.check) {
    // Correctness guard on a fresh one-element chain: after 4 adds of 1,
    // 0.5 becomes 4.5. Kept independent of --elems so the value is a
    // scalar either way.
    array c = full({1}, 1.0f, float32, stream);
    array z = full({1}, 0.5f, float32, stream);
    for (int i = 0; i < 4; ++i) {
      z = add(z, c, stream);
    }
    eval(z);
    omarchy::get_command_encoder(stream).synchronize();
    float got = z.item<float>();
    if (got != 4.5f) {
      std::fprintf(stderr, "CHECK FAILED: z=%f want 4.5\n", got);
      return 1;
    }
    std::printf("check ok: z=%f\n", got);
  }

  std::printf(
      "chain=%d reps=%d elems=%d dtype=%s shape=%s\n",
      args.chain,
      args.reps,
      args.elems,
      args.bf16 ? "bf16" : "f32",
      args.independent ? "indep" : "dep");
  std::printf("rep,total_ns,dispatches,submissions\n");

  for (int rep = 0; rep < args.reps; ++rep) {
    uint64_t dispatch_t0 = omarchy::trace::counters().vk_compute_dispatches;
    uint64_t submit_t0 = omarchy::trace::counters().vk_submissions;
    auto t0 = std::chrono::steady_clock::now();

    if (args.independent) {
      std::vector<array> outs;
      outs.reserve(args.chain);
      for (int i = 0; i < args.chain; ++i) {
        outs.push_back(add(x, x, stream));
      }
      eval(outs);
    } else {
      array z = y;
      for (int i = 0; i < args.chain; ++i) {
        z = add(z, x, stream);
      }
      eval(z);
    }
    omarchy::get_command_encoder(stream).synchronize();

    uint64_t wall = ns_since(t0);
    uint64_t dispatches =
        omarchy::trace::counters().vk_compute_dispatches - dispatch_t0;
    uint64_t submissions =
        omarchy::trace::counters().vk_submissions - submit_t0;
    std::printf(
        "%d,%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
        rep,
        wall,
        dispatches,
        submissions);
  }
  return 0;
}
