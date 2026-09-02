// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Wave 10 distributed primitives. Verdict: the five distributed primitives
// (AllReduce, AllGather, Send, Recv, ReduceScatter) are UNREACHABLE at this
// backend. Upstream mlx/distributed/ops.cpp short-circuits every one of them
// at group.size() == 1 BEFORE constructing the primitive (all_reduce line 29,
// all_gather line 69, send line 94, recv line 118, sum_scatter line 149), so
// no backend eval_gpu for these is ever called at one rank. Upstream's own
// Metal backend throws "no GPU implementation" for all five
// (mlx/backend/metal/distributed.cpp lines 17-35). This build compiles no
// communication backend (MLX_BUILD_CPU=OFF stubs mpi+ring; MLX_BUILD_CUDA=OFF
// stubs nccl; jaccl is stubbed), so distributed::is_available() is false and
// init() returns EmptyGroup of size 1. The honest deliverable is a pinned
// contract at the mlx.distributed ops level, not a backend implementation.
// The OMARCHY_UNSUPPORTED_MULTI entries in primitives.cpp stay as defensive
// parity with upstream Metal; they are dead code at one rank by upstream
// design.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <stdexcept>
#include <vector>

#include "mlx/transforms.h"
#include "mlx/distributed/distributed.h"
#include "mlx/distributed/ops.h"
#include "mlx/ops.h"

using namespace mlx::core;

TEST_CASE("distributed is unavailable in this build") {
  CHECK_UNARY_FALSE(distributed::is_available());
  CHECK_FALSE(distributed::is_available("any"));
  CHECK_FALSE(distributed::is_available("mpi"));
  CHECK_FALSE(distributed::is_available("ring"));
  CHECK_FALSE(distributed::is_available("nccl"));
  CHECK_FALSE(distributed::is_available("jaccl"));
  auto group = distributed::init();
  CHECK_EQ(group.size(), 1);
  CHECK_EQ(group.rank(), 0);
}

TEST_CASE("all_sum is identity at one rank") {
  std::vector<float> data = {1.0f, -2.0f, 3.5f, 0.25f};
  array x = array(data.data(), {2, 2}, float32);
  eval(x);
  array y = distributed::all_sum(x);
  eval(y);
  CHECK_EQ(y.shape(), x.shape());
  CHECK_EQ(y.dtype(), x.dtype());
  CHECK_EQ(std::vector<float>(y.data<float>(), y.data<float>() + 4), data);
}

TEST_CASE("all_max and all_min are identity at one rank") {
  std::vector<float> data = {4.0f, -1.0f, 7.5f};
  array x = array(data.data(), {3}, float32);
  eval(x);
  array ymax = distributed::all_max(x);
  array ymin = distributed::all_min(x);
  eval(ymax);
  eval(ymin);
  CHECK_EQ(ymax.shape(), x.shape());
  CHECK_EQ(std::vector<float>(ymax.data<float>(), ymax.data<float>() + 3),
           data);
  CHECK_EQ(ymin.shape(), x.shape());
  CHECK_EQ(std::vector<float>(ymin.data<float>(), ymin.data<float>() + 3),
           data);
}

TEST_CASE("all_gather is identity at one rank") {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  array x = array(data.data(), {4}, float32);
  eval(x);
  array y = distributed::all_gather(x);
  eval(y);
  // Upstream ops.cpp line 69 returns x unchanged; the shape does NOT get the
  // rank-multiplied first dim because the short-circuit precedes it.
  CHECK_EQ(y.shape(), x.shape());
  CHECK_EQ(y.dtype(), x.dtype());
  CHECK_EQ(std::vector<float>(y.data<float>(), y.data<float>() + 4), data);
}

TEST_CASE("sum_scatter is identity at one rank") {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  array x = array(data.data(), {2, 3}, float32);
  eval(x);
  array y = distributed::sum_scatter(x);
  eval(y);
  // Upstream ops.cpp line 149 returns x unchanged; the first dim is NOT
  // divided because the short-circuit precedes it.
  CHECK_EQ(y.shape(), x.shape());
  CHECK_EQ(y.dtype(), x.dtype());
  CHECK_EQ(std::vector<float>(y.data<float>(), y.data<float>() + 6), data);
}

TEST_CASE("send to a singleton group throws invalid_argument") {
  array x = array({1.0f, 2.0f}, {2}, float32);
  CHECK_THROWS_AS(distributed::send(x, 0), std::invalid_argument);
}

TEST_CASE("recv from a singleton group throws invalid_argument") {
  CHECK_THROWS_AS(distributed::recv({2}, float32, 0), std::invalid_argument);
}
