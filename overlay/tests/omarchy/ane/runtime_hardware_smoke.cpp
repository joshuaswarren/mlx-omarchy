// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/ane/runtime.h"

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <signal.h>
#include <unistd.h>

using mlx::core::omarchy::ane::AneBuffer;
using mlx::core::omarchy::ane::AneBufferMap;
using mlx::core::omarchy::ane::AneRuntime;

namespace {

AneBuffer filled_fp16(size_t elements, uint16_t bits) {
  AneBuffer result(elements * 2);
  for (size_t i = 0; i < elements; ++i) {
    result[i * 2] = static_cast<uint8_t>(bits);
    result[i * 2 + 1] = static_cast<uint8_t>(bits >> 8);
  }
  return result;
}

void require_fp16(const AneBuffer& output, uint16_t expected) {
  if (output.size() != 128) {
    throw std::runtime_error("output is not 64 fp16 values");
  }
  for (size_t i = 0; i < output.size(); i += 2) {
    uint16_t actual = static_cast<uint16_t>(output[i]) |
        (static_cast<uint16_t>(output[i + 1]) << 8);
    if (actual != expected) {
      throw std::runtime_error(
          "wrong fp16 output at element " + std::to_string(i / 2) +
          ": expected 0x" + std::to_string(expected) +
          " found 0x" + std::to_string(actual));
    }
  }
}

long positive_long(const char* value) {
  errno = 0;
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed <= 0) {
    throw std::invalid_argument("deadline must be a positive integer number of milliseconds");
  }
  return parsed;
}

} // namespace

int main(int argc, char** argv) {
  if (argc != 5 ||
      (std::string(argv[2]) != "add" && std::string(argv[2]) != "add-mul")) {
    std::cerr << "usage: mlx-omarchy-ane-smoke BUNDLE add|add-mul "
                 "DEADLINE_MS DIAGNOSTIC_PATH\n";
    return 2;
  }

  try {
    const auto deadline = std::chrono::milliseconds(positive_long(argv[3]));
    auto runtime = AneRuntime::load(argv[1], deadline, argv[4]);
    const int worker = runtime->worker_pid();
    std::cout << "runtime_identity " << runtime->runtime_identity() << '\n';
    std::cout << "worker_executable "
              << std::filesystem::canonical(
                     "/proc/" + std::to_string(worker) + "/exe")
              << std::endl;
    std::cout << "worker_pid " << worker << '\n';

    AneBufferMap inputs{
        {"a", filled_fp16(64, 0x3c00)},
        {"b", filled_fp16(64, 0x4000)},
    };
    const uint16_t expected =
        std::string(argv[2]) == "add" ? uint16_t{0x4200} : uint16_t{0x4600};
    for (int iteration = 0; iteration < 2; ++iteration) {
      auto outputs = runtime->execute(inputs, deadline);
      if (outputs.size() != 1) {
        throw std::runtime_error("bundle did not return exactly one output");
      }
      require_fp16(outputs.begin()->second, expected);
      std::cout << "iteration " << iteration << " exact_fp16=PASS bytes="
                << outputs.begin()->second.size() << '\n';
    }

    auto receipt = runtime->shutdown(deadline);
    errno = 0;
    const bool process_released = ::kill(receipt.worker_pid, 0) == -1 && errno == ESRCH;
    if (!process_released) {
      throw std::runtime_error("worker process still exists after clean shutdown");
    }
    std::cout << "shutdown worker_pid=" << receipt.worker_pid
              << " released_programs=" << receipt.released_programs
              << " process_released=true\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ANE runtime smoke failed: " << error.what() << '\n';
    return 1;
  }
}
