// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "mlx/array.h"

namespace mlx::core::gpu {

// Fold a single-element operation to host arithmetic when every input's
// bytes are already final without running GPU work. Returns true when it
// wrote the output host-side; the caller must then skip eval_gpu and the
// dispatch it would record. Never synchronizes and never reads an input
// that still has GPU work behind it: an input that is not provably final
// keeps the dispatch.
bool host_fold_scalar(array& arr);

} // namespace mlx::core::gpu
