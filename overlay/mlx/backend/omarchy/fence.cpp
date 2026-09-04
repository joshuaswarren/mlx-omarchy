// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Fence on top of the event implementation. The events are real timeline
// semaphores on GPU streams, so fences carry real device ordering.

#include "mlx/fence.h"

#include "mlx/event.h"
#include "mlx/primitives.h"

#include <cstdio>
#include <cstdlib>

namespace mlx::core {

struct FenceImpl {
  uint32_t count;
  Event event;

  FenceImpl(uint32_t count, Stream s) : count(count), event(s) {}
};

Fence::Fence(Stream s) {
  fence_ = std::make_shared<FenceImpl>(0, s);
}

static bool trace_fence() {
  static bool on = std::getenv("MLX_OMARCHY_TRACE_FENCE") != nullptr;
  return on;
}

void Fence::wait(Stream s, const array& a) {
  if (trace_fence()) {
    std::fprintf(stderr, "[fence] wait  on stream %d for %s (produced on stream %d)\n",
        s.index, a.has_primitive() ? a.primitive().name() : "leaf",
        a.has_primitive() ? a.primitive().stream().index : -1);
  }
  cast<FenceImpl>().event.wait(s);
}

void Fence::update(Stream s, const array& a, bool) {
  if (trace_fence()) {
    std::fprintf(stderr, "[fence] update on stream %d after %s\n", s.index,
        a.has_primitive() ? a.primitive().name() : "leaf");
  }
  auto& f = cast<FenceImpl>();
  f.count++;
  f.event.set_value(f.count);
  f.event.signal(s);
}

} // namespace mlx::core
