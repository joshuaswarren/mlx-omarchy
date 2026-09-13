// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "mlx/api.h"

namespace mlx::core::omarchy::ane {

constexpr uint64_t kAneTileAlignment = 0x4000;
constexpr int kAneManifestVersion = 3;
constexpr uint64_t kAneDriverAbiMajor = 1;

struct AneTensor {
  std::string name;
  uint64_t index{0};
  std::string dtype;
  std::vector<uint64_t> shape;
  uint64_t byte_size{0};
  uint64_t stride{0};
};

struct AneProgramBinding {
  std::string tensor;
  uint64_t channel{0};
  std::string dtype;
  std::vector<uint64_t> shape;
  std::array<uint64_t, 6> nchw{};
  uint64_t logical_bytes{0};
  uint64_t allocation_bytes{0};
  uint64_t element_offset{0};
  uint64_t element_count{0};
  uint64_t physical_elements{0};
};

struct AneProgram {
  std::string payload;
  std::string operation;
  std::string encoder;
  uint64_t task_descriptors{0};
  uint64_t scratch_bytes{0};
  std::vector<AneProgramBinding> inputs;
  std::vector<AneProgramBinding> outputs;
};

struct AnePayload {
  std::string role;
  std::string path;
  std::string sha256;
  uint64_t byte_size{0};
};

struct AneCompilerIdentity {
  std::string host_build;
  std::string toolchain;
  std::string target;
};

struct AneProvenance {
  std::string source_repo;
  std::string source_commit;
  std::string exported_at;
};

struct AneReleaseAsset {
  std::string model;
  std::string model_sha256;
};

struct AneManifest {
  int manifest_version{0};
  std::string name;
  std::string graph_hash;
  uint64_t task_descriptors{0};
  std::vector<AneTensor> inputs;
  std::vector<AneTensor> outputs;
  std::vector<AneTensor> state;
  std::vector<AneTensor> intermediates;
  std::vector<AneProgram> programs;
  std::vector<uint64_t> dispatch_plan;
  std::vector<AnePayload> payloads;
  AneCompilerIdentity compiler;
  uint64_t driver_abi_major{0};
  AneProvenance provenance;
  AneReleaseAsset release_asset;
};

// Strict structural parsing only. A valid manifest is not proof that the
// current machine is H13-qualified or eligible for model execution.
MLX_API AneManifest
parse_ane_manifest(const std::filesystem::path& manifest_path);

} // namespace mlx::core::omarchy::ane
