// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "mlx/api.h"

namespace mlx::core::omarchy::ane {

// ANE DMA rows use 0x4000 tiles. Grounded in ane-linux-experiments:
// tools/hwxv2-to-anec.py (TILE_SIZE = 0x4000) and the task-layout receipt
// (KDMA offsets 0x28000..0x3c000 step 0x4000).
constexpr uint64_t kAneTileAlignment = 0x4000;

// One tensor contract entry. Inputs, outputs, state, and workspace all use
// this shape. `byte_size` is the tight logical size. `stride` is the
// tile-aligned DMA stride; it must be a multiple of kAneTileAlignment and at
// least byte_size.
struct AneTensor {
  std::string name;
  uint64_t index{0};
  std::string dtype;
  std::vector<uint64_t> shape;
  uint64_t byte_size{0};
  uint64_t stride{0};
};

// One file inside the bundle directory. `sha256` is the lowercase hex digest
// of the file contents. Roles: "anec" (compiled program) and "weights".
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

struct AneFirmwareRange {
  std::string min;
  std::string max;
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

// The typed result of strict manifest parsing. Every field was validated
// before this struct is returned; no field is optional after parse.
struct AneManifest {
  int manifest_version{0};
  std::string name;
  std::string graph_hash;
  uint64_t task_descriptors{0};
  std::vector<AneTensor> inputs;
  std::vector<AneTensor> outputs;
  std::vector<AneTensor> state;
  std::vector<AneTensor> workspace;
  std::vector<AnePayload> payloads;
  AneCompilerIdentity compiler;
  AneFirmwareRange firmware;
  AneProvenance provenance;
  AneReleaseAsset release_asset;
};

// Parses and validates manifest.json. Every check fails closed with an
// exception whose message names the field and the reason. Manifest validation
// performs no payload access; U6 requires every field check to pass before
// Linux maps or submits a descriptor.
MLX_API AneManifest
parse_ane_manifest(const std::filesystem::path& manifest_path);

} // namespace mlx::core::omarchy::ane
