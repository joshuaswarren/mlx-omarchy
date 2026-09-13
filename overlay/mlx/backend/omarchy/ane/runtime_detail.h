// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "mlx/backend/omarchy/ane/manifest.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlx::core::omarchy::ane::detail {

constexpr int kWorkerControlFd = 3;
constexpr int kWorkerStagingFd = 4;
constexpr size_t kWorkerDetailBytes = 1024;
constexpr uint32_t kWorkerProtocolVersion = 1;

inline std::runtime_error runtime_error(const std::string& reason) {
  return std::runtime_error("[omarchy-ane] runtime: " + reason + ".");
}

inline void validate_deadline(std::chrono::milliseconds deadline) {
  if (deadline.count() <= 0) {
    throw std::invalid_argument("[omarchy-ane] runtime: deadline must be positive.");
  }
}

inline size_t checked_size(uint64_t value, const std::string& label) {
  if (value > std::numeric_limits<size_t>::max()) {
    throw runtime_error(label + " exceeds host size_t");
  }
  return static_cast<size_t>(value);
}

inline void pack_binding(
    const AneProgramBinding& binding,
    const std::vector<uint8_t>& dense,
    std::vector<uint8_t>& packed) {
  const size_t logical = checked_size(binding.logical_bytes, "logical byte count");
  const size_t allocation =
      checked_size(binding.allocation_bytes, "allocation byte count");
  const size_t begin = checked_size(binding.element_offset, "element offset") * 2;
  if (begin > dense.size() || logical > dense.size() - begin) {
    throw runtime_error(
        "tensor '" + binding.tensor + "' dense staging byte count is " +
        std::to_string(dense.size()) + ", expected at least " +
        std::to_string(begin + logical));
  }
  if (packed.size() != allocation) {
    throw runtime_error(
        "tensor '" + binding.tensor + "' packed byte count is " +
        std::to_string(packed.size()) + ", expected " +
        std::to_string(allocation));
  }

  const size_t width = checked_size(binding.nchw[3], "packed width");
  const size_t row = checked_size(binding.nchw[5], "packed row stride");
  std::fill(packed.begin(), packed.end(), 0);
  for (size_t element = 0; element < logical / 2; ++element) {
    const size_t destination = (element / width) * row + (element % width) * 2;
    if (destination > packed.size() - 2) {
      throw runtime_error("tensor '" + binding.tensor + "' packed layout exceeds allocation");
    }
    std::memcpy(packed.data() + destination, dense.data() + begin + element * 2, 2);
  }
}

inline std::vector<uint8_t> pack_binding(
    const AneProgramBinding& binding,
    const std::vector<uint8_t>& dense) {
  std::vector<uint8_t> packed(
      checked_size(binding.allocation_bytes, "allocation byte count"));
  pack_binding(binding, dense, packed);
  return packed;
}

inline void unpack_binding(
    const AneProgramBinding& binding,
    const std::vector<uint8_t>& packed,
    std::vector<uint8_t>& dense) {
  const size_t logical = checked_size(binding.logical_bytes, "logical byte count");
  const size_t allocation =
      checked_size(binding.allocation_bytes, "allocation byte count");
  const size_t begin = checked_size(binding.element_offset, "element offset") * 2;
  if (packed.size() != allocation) {
    throw runtime_error(
        "tensor '" + binding.tensor + "' packed byte count is " +
        std::to_string(packed.size()) + ", expected " + std::to_string(allocation));
  }
  if (begin > dense.size() || logical > dense.size() - begin) {
    throw runtime_error(
        "tensor '" + binding.tensor + "' dense staging byte count is " +
        std::to_string(dense.size()) + ", expected at least " +
        std::to_string(begin + logical));
  }

  const size_t width = checked_size(binding.nchw[3], "packed width");
  const size_t row = checked_size(binding.nchw[5], "packed row stride");
  for (size_t element = 0; element < logical / 2; ++element) {
    const size_t source = (element / width) * row + (element % width) * 2;
    if (source > packed.size() - 2) {
      throw runtime_error("tensor '" + binding.tensor + "' packed layout exceeds allocation");
    }
    std::memcpy(dense.data() + begin + element * 2, packed.data() + source, 2);
  }
}

enum class WorkerOperation : uint32_t {
  execute = 1,
  shutdown = 2,
};

enum class WorkerReplyKind : uint32_t {
  ready = 1,
  executed = 2,
  stopped = 3,
  failed = 4,
  uncertain = 5,
};

struct WorkerCommand {
  uint32_t version{kWorkerProtocolVersion};
  WorkerOperation operation{WorkerOperation::execute};
  uint64_t serial{0};
  int64_t deadline_monotonic_nanoseconds{0};
};

struct WorkerReply {
  uint32_t version{kWorkerProtocolVersion};
  WorkerReplyKind kind{WorkerReplyKind::failed};
  uint64_t serial{0};
  uint64_t released_programs{0};
  char detail[kWorkerDetailBytes]{};
};

int run_worker(
    int control_fd,
    int staging_fd,
    size_t staging_size,
    const std::filesystem::path& bundle_path);

} // namespace mlx::core::omarchy::ane::detail
