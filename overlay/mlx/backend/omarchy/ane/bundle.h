// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "mlx/backend/omarchy/ane/manifest.h"

#include <array>
#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>

#include "mlx/api.h"

namespace mlx::core::omarchy::ane {

constexpr size_t kAnecHeaderSize = 0x6a8;
constexpr size_t kAnecPayloadOffset = 0x1000;
constexpr size_t kAnecTileCount = 0x20;

struct AneAnecHeader {
  uint64_t payload_size{0};
  uint32_t task_descriptor_size{0};
  uint32_t task_descriptor_count{0};
  uint64_t task_size{0};
  uint64_t kernel_size{0};
  uint32_t source_count{0};
  uint32_t destination_count{0};
  uint64_t bootstrap_channel_size{0};
  std::array<uint32_t, kAnecTileCount> tiles{};
  std::array<std::array<uint64_t, 6>, kAnecTileCount> nchw{};
};

// Raised when the bundle directory does not exist. Callers treat this outcome
// as "region stays on Vulkan": a missing bundle is a normal runtime condition,
// not an error. Every other failure throws std::runtime_error with a named
// reason, and happens before any payload mapping or device access.
struct AneBundleNotFound : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// A validated bundle: the parsed manifest plus the verified payload paths.
// No device access, no submission. This is the pre-device validation layer.
struct AneBundle {
  AneManifest manifest;
  AneAnecHeader anec_header;
  std::filesystem::path anec;
  std::optional<std::filesystem::path> weights;
};

// Reads the libane `.anec` header and validates the file and channel envelope
// libane will use: a 0x6a8-byte ABI struct at file offset 0, executable payload
// at offset 0x1000, command and reserved kernel channels, task fields, and
// 16-bit source/destination tile geometry.
MLX_API AneAnecHeader parse_anec_header(const std::filesystem::path& path);

// Loads and fully verifies the bundle at `dir`:
//   1. The directory must exist, else AneBundleNotFound.
//   2. manifest.json is parsed and validated field by field (see
//      parse_ane_manifest). All field checks pass before payload access.
//   3. Every regular file in the directory must be manifest.json or a listed
//      payload; anything else is an unknown payload.
//   4. Each listed payload must exist with the manifest byte size and match
//      its sha256 digest.
//   5. The ANEC header, task envelope, channel bindings, and 16-bit tile/NCHW
//      geometry must match before any caller can hand the bundle to a worker.
MLX_API AneBundle load_bundle(const std::filesystem::path& dir);

// FIPS 180-4 SHA-256. Used for payload descriptor digests so the Linux
// validation gate and the fixture tests share one implementation.
MLX_API std::string sha256_hex(const uint8_t* data, size_t size);
MLX_API std::string sha256_file(const std::filesystem::path& path);

} // namespace mlx::core::omarchy::ane
