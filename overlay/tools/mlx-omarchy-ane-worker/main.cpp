// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Bounded ANE worker CLI (plan sections 24-27). One invocation is one
// supervised execution of one bundle: load programs, run the dispatch
// plan for N iterations inside a wall-clock deadline, verify expected
// outputs byte-exactly when asked, release, exit. There is no server,
// no socket, and no state between invocations.

#include "mlx/backend/omarchy/ane/bundle.h"
#include "mlx/backend/omarchy/ane/worker.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace mlx::core::omarchy::ane;

#ifdef MLX_OMARCHY_ANE_DEVICE
namespace mlx::core::omarchy::ane {
std::unique_ptr<AneDevice> make_libane_device(const std::string& library);
}
#endif

namespace {

int usage() {
  std::fprintf(
      stderr,
      "usage: mlx-omarchy-ane-worker --bundle DIR --libane PATH\n"
      "       [--deadline-ms N] [--iterations N]\n"
      "       [--input NAME=FILE]... [--expect NAME=FILE]...\n");
  return 64;
}

// fp16 value equality: identical bits, except +0.0 and -0.0 compare
// equal. The H13 compiler models zero products as unsigned (aa688df
// "model unsigned zero products"), so a -0.0 expectation against a +0.0
// device result is a value match, not a mismatch.
bool fp16_values_equal(const uint8_t* lhs, const uint8_t* rhs, size_t size) {
  for (size_t i = 0; i + 1 < size; i += 2) {
    uint16_t left = static_cast<uint16_t>(lhs[i]) |
                    (static_cast<uint16_t>(lhs[i + 1]) << 8);
    uint16_t right = static_cast<uint16_t>(rhs[i]) |
                     (static_cast<uint16_t>(rhs[i + 1]) << 8);
    if (left == right) continue;
    if ((left & 0x7FFF) == 0 && (right & 0x7FFF) == 0) continue; // +/-0
    return false;
  }
  return true;
}

std::vector<uint8_t> read_file(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("cannot open " + path);
  }
  return std::vector<uint8_t>(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
}

} // namespace

int main(int argc, char** argv) {
  std::string bundle_dir;
  std::string libane_path;
  long deadline_ms = 2000;
  long iterations = 1;
  std::map<std::string, std::string> input_files;
  std::map<std::string, std::string> expect_files;
  std::map<std::string, std::string> save_files;

  for (int i = 1; i < argc; ++i) {
    std::string flag = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", flag.c_str());
        std::exit(64);
      }
      return argv[++i];
    };
    if (flag == "--bundle") {
      bundle_dir = value();
    } else if (flag == "--libane") {
      libane_path = value();
    } else if (flag == "--deadline-ms") {
      deadline_ms = std::stol(value());
    } else if (flag == "--iterations") {
      iterations = std::stol(value());
    } else if (flag == "--input") {
      auto assignment = value();
      auto sep = assignment.find('=');
      if (sep == std::string::npos) {
        std::fprintf(stderr, "--input expects NAME=FILE\n");
        return usage();
      }
      input_files[assignment.substr(0, sep)] = assignment.substr(sep + 1);
    } else if (flag == "--expect") {
      auto assignment = value();
      auto sep = assignment.find('=');
      if (sep == std::string::npos) {
        std::fprintf(stderr, "--expect expects NAME=FILE\n");
        return usage();
      }
      expect_files[assignment.substr(0, sep)] = assignment.substr(sep + 1);
    } else if (flag == "--save") {
      auto assignment = value();
      auto sep = assignment.find('=');
      if (sep == std::string::npos) {
        std::fprintf(stderr, "--save expects NAME=FILE\n");
        return usage();
      }
      save_files[assignment.substr(0, sep)] = assignment.substr(sep + 1);
    } else {
      return usage();
    }
  }
  if (bundle_dir.empty() || libane_path.empty()) {
    return usage();
  }

  try {
    AneBundle bundle = load_bundle(bundle_dir);
    std::printf(
        "bundle name=%s programs=%zu driver_abi=%llu graph=%s\n",
        bundle.manifest.name.c_str(), bundle.programs.size(),
        static_cast<unsigned long long>(bundle.manifest.driver_abi_major),
        bundle.manifest.graph_hash.c_str());

    std::map<std::string, AneWorker::Buffer> inputs;
    for (const auto& entry : input_files) {
      inputs[entry.first] = read_file(entry.second);
    }
    for (const auto& tensor : bundle.manifest.inputs) {
      if (!inputs.count(tensor.name)) {
        std::fprintf(
            stderr, "missing --input for manifest input '%s'\n",
            tensor.name.c_str());
        return 65;
      }
    }

    AneWorkerOptions options;
    options.deadline = std::chrono::milliseconds(deadline_ms);
    options.iterations = static_cast<int>(iterations);

#ifdef MLX_OMARCHY_ANE_DEVICE
    AneWorker worker(
        [libane_path] { return make_libane_device(libane_path); }, options);
#else
    std::fprintf(
        stderr,
        "this binary was built without MLX_OMARCHY_ANE_DEVICE; no device "
        "backend is linked\n");
    return 70;
#endif

    std::map<std::string, AneWorker::Buffer> outputs;
    AneWorkerReport report = worker.run(bundle, inputs, &outputs);
    std::printf(
        "worker status=%d iterations=%d released=%d elapsed_ms=%lld\n",
        static_cast<int>(report.status), report.iterations,
        report.released_programs,
        static_cast<long long>(report.elapsed.count()));
    std::printf("worker detail=%s\n", report.detail.c_str());
    if (worker.quarantined()) {
      std::printf("worker quarantined: %s\n",
                  worker.quarantine_reason().c_str());
    }

    if (report.status != AneWorkerStatus::Completed) {
      return 1;
    }

    for (const auto& entry : save_files) {
      auto found = outputs.find(entry.first);
      if (found == outputs.end()) {
        std::fprintf(
            stderr, "cannot save missing output '%s'\n",
            entry.first.c_str());
        return 1;
      }
      std::ofstream stream(entry.second, std::ios::binary);
      stream.write(
          reinterpret_cast<const char*>(found->second.data()),
          static_cast<std::streamsize>(found->second.size()));
      if (!stream) {
        std::fprintf(stderr, "cannot write %s\n", entry.second.c_str());
        return 1;
      }
      std::printf("saved output %s (%zu bytes)\n", entry.first.c_str(),
                  found->second.size());
    }
    for (const auto& entry : expect_files) {
      auto found = outputs.find(entry.first);
      if (found == outputs.end()) {
        std::fprintf(
            stderr, "expected output '%s' was not produced\n",
            entry.first.c_str());
        return 1;
      }
      auto expected = read_file(entry.second);
      if (found->second.size() != expected.size() ||
          !fp16_values_equal(
              found->second.data(), expected.data(), expected.size())) {
        std::fprintf(
            stderr, "output '%s' does not match the expected fp16 values\n",
            entry.first.c_str());
        return 1;
      }
      std::printf("verified output %s exact\n", entry.first.c_str());
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
}
