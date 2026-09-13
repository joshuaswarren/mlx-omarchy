// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/ane/runtime_detail.h"

#include <cerrno>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "[omarchy-ane] worker: expected BUNDLE STAGING_BYTES\n";
    return 2;
  }
  try {
    errno = 0;
    char* end = nullptr;
    unsigned long long size = std::strtoull(argv[2], &end, 10);
    if (errno != 0 || end == argv[2] || *end != '\0' || size == 0 ||
        size > std::numeric_limits<size_t>::max()) {
      throw std::invalid_argument("invalid staging byte count");
    }
    return mlx::core::omarchy::ane::detail::run_worker(
        mlx::core::omarchy::ane::detail::kWorkerControlFd,
        mlx::core::omarchy::ane::detail::kWorkerStagingFd,
        static_cast<size_t>(size),
        std::filesystem::path(argv[1]));
  } catch (const std::exception& error) {
    std::cerr << "[omarchy-ane] worker: " << error.what() << '\n';
    return 2;
  }
}
