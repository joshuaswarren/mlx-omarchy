// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/ane/manifest.h"

#include <json.hpp>

#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlx::core::omarchy::ane {
namespace {

constexpr const char* kManifestName = "manifest.json";
constexpr int kSupportedManifestVersion = 1;

std::runtime_error manifest_error(const std::string& reason) {
  return std::runtime_error("[omarchy-ane] manifest: " + reason + ".");
}

const nlohmann::json& required_field(
    const nlohmann::json& object,
    const std::string& field) {
  auto found = object.find(field);
  if (found == object.end()) {
    throw manifest_error("missing field '" + field + "'");
  }
  return *found;
}

// Rejects any key not in `allowed`. Fail closed on unknown fields: a bundle
// from a newer exporter must not silently drop its new meaning.
void reject_unknown_fields(
    const nlohmann::json& object,
    const std::vector<const char*>& allowed) {
  std::set<std::string> allowed_set(allowed.begin(), allowed.end());
  for (auto it = object.begin(); it != object.end(); ++it) {
    if (allowed_set.count(it.key()) == 0) {
      throw manifest_error("unknown field '" + it.key() + "'");
    }
  }
}

std::string require_string(const nlohmann::json& object, const char* field) {
  const auto& value = required_field(object, field);
  if (!value.is_string()) {
    throw manifest_error(std::string("field '") + field + "' must be a string");
  }
  return value.get<std::string>();
}

std::string require_non_empty_string(
    const nlohmann::json& object,
    const char* field) {
  std::string value = require_string(object, field);
  if (value.empty()) {
    throw manifest_error(std::string("field '") + field + "' must not be empty");
  }
  return value;
}

uint64_t require_positive_integer(
    const nlohmann::json& object,
    const char* field) {
  const auto& value = required_field(object, field);
  if (!value.is_number_unsigned()) {
    throw manifest_error(
        std::string("field '") + field + "' must be a non-negative integer");
  }
  uint64_t parsed = value.get<uint64_t>();
  if (parsed == 0) {
    throw manifest_error(std::string("field '") + field + "' must be positive");
  }
  return parsed;
}

bool is_lower_hex(const std::string& value, size_t length) {
  if (value.size() != length) {
    return false;
  }
  for (char c : value) {
    bool digit = (c >= '0' && c <= '9');
    bool lower = (c >= 'a' && c <= 'f');
    if (!digit && !lower) {
      return false;
    }
  }
  return true;
}

std::string require_hex_field(
    const nlohmann::json& object,
    const char* field,
    size_t length,
    const char* what) {
  std::string value = require_string(object, field);
  if (!is_lower_hex(value, length)) {
    throw manifest_error(
        std::string("field '") + field + "' must be " + what + " (" +
        std::to_string(length) + " lowercase hex characters)");
  }
  return value;
}

std::vector<uint64_t> require_positive_shape(
    const nlohmann::json& object,
    const char* field) {
  const auto& value = required_field(object, field);
  if (!value.is_array() || value.empty()) {
    throw manifest_error(std::string("field '") + field + "' must be a non-empty array");
  }
  std::vector<uint64_t> shape;
  for (const auto& dim : value) {
    if (!dim.is_number_unsigned() || dim.get<uint64_t>() == 0) {
      throw manifest_error(
          std::string("field '") + field + "' must contain positive integers");
    }
    shape.push_back(dim.get<uint64_t>());
  }
  return shape;
}

uint64_t dtype_size(const std::string& dtype) {
  if (dtype == "float16") return 2;
  if (dtype == "float32") return 4;
  if (dtype == "bfloat16") return 2;
  if (dtype == "int32") return 4;
  if (dtype == "uint8") return 1;
  return 0;
}

// Validates one tensor entry and returns it. Checks, in order: object shape,
// name, dtype, positive shape, byte size against the dtype geometry, and the
// tile-aligned stride contract.
AneTensor parse_tensor(
    const nlohmann::json& value,
    const std::string& list_name,
    size_t position,
    bool require_aligned_stride) {
  std::string where = list_name + "[" + std::to_string(position) + "]";
  if (!value.is_object()) {
    throw manifest_error(where + " must be an object");
  }
  reject_unknown_fields(
      value, {"name", "index", "dtype", "shape", "byte_size", "stride"});

  AneTensor tensor;
  tensor.name = require_non_empty_string(value, "name");
  const auto& index = required_field(value, "index");
  if (!index.is_number_unsigned()) {
    throw manifest_error(where + " field 'index' must be a non-negative integer");
  }
  tensor.index = index.get<uint64_t>();
  tensor.dtype = require_non_empty_string(value, "dtype");
  tensor.shape = require_positive_shape(value, "shape");
  tensor.byte_size = require_positive_integer(value, "byte_size");
  tensor.stride = require_positive_integer(value, "stride");

  uint64_t element_size = dtype_size(tensor.dtype);
  if (element_size == 0) {
    throw manifest_error(
        where + " has unsupported dtype '" + tensor.dtype +
        "' (expected float16, float32, bfloat16, int32, or uint8)");
  }
  uint64_t logical_size = element_size;
  for (uint64_t dim : tensor.shape) {
    logical_size *= dim;
  }
  if (tensor.byte_size != logical_size) {
    std::ostringstream expected;
    expected << where << " byte_size " << tensor.byte_size
             << " does not match dtype geometry " << logical_size;
    throw manifest_error(expected.str());
  }
  // Descriptor DMAs address inputs, outputs, and state through tile-aligned
  // rows (KDMA offsets step 0x4000 in the task-layout receipt). The workspace
  // is a flat scratch buffer whose size the compiler chooses; the
  // terminal-task receipt records 0x66000, not a 0x4000 multiple.
  if (require_aligned_stride && tensor.stride % kAneTileAlignment != 0) {
    std::ostringstream aligned;
    aligned << where << " stride " << tensor.stride << " is not a multiple of 0x"
            << std::hex << kAneTileAlignment;
    throw manifest_error(aligned.str());
  }
  if (tensor.stride < tensor.byte_size) {
    throw manifest_error(where + " stride must be at least byte_size");
  }
  return tensor;
}

std::vector<AneTensor> parse_tensor_list(
    const nlohmann::json& object,
    const char* field,
    bool require_aligned_stride) {
  const auto& value = required_field(object, field);
  if (!value.is_array()) {
    throw manifest_error(std::string("field '") + field + "' must be an array");
  }
  std::vector<AneTensor> list;
  std::set<std::string> names;
  std::set<uint64_t> indices;
  for (size_t i = 0; i < value.size(); ++i) {
    AneTensor tensor = parse_tensor(value[i], field, i, require_aligned_stride);
    if (!names.insert(tensor.name).second) {
      throw manifest_error(
          std::string("field '") + field + "' has duplicate tensor name '" +
          tensor.name + "'");
    }
    if (!indices.insert(tensor.index).second) {
      throw manifest_error(
          std::string("field '") + field + "' has duplicate tensor index " +
          std::to_string(tensor.index));
    }
    list.push_back(std::move(tensor));
  }
  return list;
}

AnePayload parse_payload(
    const nlohmann::json& value,
    size_t position) {
  std::string where = "payloads[" + std::to_string(position) + "]";
  if (!value.is_object()) {
    throw manifest_error(where + " must be an object");
  }
  reject_unknown_fields(value, {"role", "path", "sha256", "byte_size"});

  AnePayload payload;
  payload.role = require_non_empty_string(value, "role");
  payload.path = require_non_empty_string(value, "path");
  payload.sha256 = require_hex_field(value, "sha256", 64, "a sha256 digest");
  payload.byte_size = require_positive_integer(value, "byte_size");

  std::filesystem::path path(payload.path);
  if (path.is_absolute() || payload.path.find("..") != std::string::npos) {
    throw manifest_error(
        where + " path '" + payload.path +
        "' must be relative and stay inside the bundle directory");
  }
  return payload;
}

// Firmware identifiers are dotted integer versions. The range check matches
// the inclusive firmware compatibility gate the runtime applies before any
// device access.
std::vector<uint64_t> parse_version(const std::string& text) {
  if (text.empty()) {
    throw manifest_error("field 'firmware' versions must not be empty");
  }
  std::vector<uint64_t> parts;
  size_t start = 0;
  while (true) {
    size_t dot = text.find('.', start);
    std::string part = text.substr(start, dot - start);
    if (part.empty()) {
      throw manifest_error(
          "field 'firmware' version '" + text + "' is not a dotted integer version");
    }
    for (char c : part) {
      if (c < '0' || c > '9') {
        throw manifest_error(
            "field 'firmware' version '" + text +
            "' is not a dotted integer version");
      }
    }
    parts.push_back(std::stoull(part));
    if (dot == std::string::npos) {
      break;
    }
    start = dot + 1;
  }
  return parts;
}

void parse_firmware_range(const nlohmann::json& object, AneManifest& manifest) {
  const auto& value = required_field(object, "firmware");
  if (!value.is_object()) {
    throw manifest_error("field 'firmware' must be an object");
  }
  reject_unknown_fields(value, {"min", "max"});
  manifest.firmware.min = require_non_empty_string(value, "min");
  manifest.firmware.max = require_non_empty_string(value, "max");
  if (parse_version(manifest.firmware.max) < parse_version(manifest.firmware.min)) {
    throw manifest_error("field 'firmware' range max must be at least min");
  }
}

} // namespace

AneManifest parse_ane_manifest(const std::filesystem::path& manifest_path) {
  std::ifstream input(manifest_path);
  if (!input) {
    throw manifest_error(
        std::string("cannot open ") + kManifestName + " at " +
        manifest_path.parent_path().string());
  }
  nlohmann::json root;
  try {
    input >> root;
  } catch (const nlohmann::json::exception& error) {
    throw manifest_error(std::string("invalid JSON (") + error.what() + ")");
  }
  if (!root.is_object()) {
    throw manifest_error("root must be a JSON object");
  }
  reject_unknown_fields(
      root,
      {"manifest_version",
       "name",
       "graph_hash",
       "task_descriptors",
       "inputs",
       "outputs",
       "state",
       "workspace",
       "payloads",
       "compiler",
       "firmware",
       "provenance",
       "release_asset"});

  AneManifest manifest;
  const auto& version = required_field(root, "manifest_version");
  if (!version.is_number_integer()) {
    throw manifest_error("field 'manifest_version' must be an integer");
  }
  if (version.get<int64_t>() != kSupportedManifestVersion) {
    throw manifest_error(
        "unsupported manifest_version " + version.dump() + " (expected " +
        std::to_string(kSupportedManifestVersion) + ")");
  }
  manifest.manifest_version = version.get<int>();
  manifest.name = require_non_empty_string(root, "name");
  manifest.graph_hash =
      require_hex_field(root, "graph_hash", 64, "a sha256 digest");
  manifest.task_descriptors = require_positive_integer(root, "task_descriptors");

  manifest.inputs = parse_tensor_list(root, "inputs", true);
  if (manifest.inputs.empty()) {
    throw manifest_error("field 'inputs' must list at least one tensor");
  }
  manifest.outputs = parse_tensor_list(root, "outputs", true);
  if (manifest.outputs.empty()) {
    throw manifest_error("field 'outputs' must list at least one tensor");
  }
  manifest.state = parse_tensor_list(root, "state", true);
  // The workspace is the scratch buffer bound at submit time. Grounded in the
  // terminal-task receipt: workspace binds as its own buffer (0x66000), and
  // submission requires it before enqueue.
  manifest.workspace = parse_tensor_list(root, "workspace", false);
  if (manifest.workspace.size() != 1) {
    throw manifest_error("field 'workspace' must list exactly one tensor");
  }

  const auto& payloads = required_field(root, "payloads");
  if (!payloads.is_array() || payloads.empty()) {
    throw manifest_error("field 'payloads' must be a non-empty array");
  }
  size_t anec_count = 0;
  size_t weights_count = 0;
  std::set<std::string> paths;
  for (size_t i = 0; i < payloads.size(); ++i) {
    AnePayload payload = parse_payload(payloads[i], i);
    if (payload.role == "anec") {
      ++anec_count;
    } else if (payload.role == "weights") {
      ++weights_count;
    } else {
      throw manifest_error(
          "payloads[" + std::to_string(i) + "] has unknown role '" + payload.role +
          "' (expected anec or weights)");
    }
    if (!paths.insert(payload.path).second) {
      throw manifest_error(
          "field 'payloads' has duplicate path '" + payload.path + "'");
    }
    manifest.payloads.push_back(std::move(payload));
  }
  if (anec_count != 1) {
    throw manifest_error("field 'payloads' must list exactly one 'anec' payload");
  }
  if (weights_count > 1) {
    throw manifest_error("field 'payloads' must list at most one 'weights' payload");
  }

  const auto& compiler = required_field(root, "compiler");
  if (!compiler.is_object()) {
    throw manifest_error("field 'compiler' must be an object");
  }
  reject_unknown_fields(compiler, {"macos_build", "anecompiler"});
  manifest.compiler.macos_build = require_non_empty_string(compiler, "macos_build");
  manifest.compiler.anecompiler = require_non_empty_string(compiler, "anecompiler");

  parse_firmware_range(root, manifest);

  const auto& provenance = required_field(root, "provenance");
  if (!provenance.is_object()) {
    throw manifest_error("field 'provenance' must be an object");
  }
  reject_unknown_fields(provenance, {"source_repo", "source_commit", "exported_at"});
  manifest.provenance.source_repo =
      require_non_empty_string(provenance, "source_repo");
  manifest.provenance.source_commit =
      require_hex_field(provenance, "source_commit", 40, "a git commit hash");
  manifest.provenance.exported_at = require_non_empty_string(provenance, "exported_at");
  if (manifest.provenance.exported_at.size() != 10 ||
      manifest.provenance.exported_at[4] != '-' ||
      manifest.provenance.exported_at[7] != '-') {
    throw manifest_error(
        "field 'provenance' exported_at must have the form YYYY-MM-DD");
  }

  const auto& release_asset = required_field(root, "release_asset");
  if (!release_asset.is_object()) {
    throw manifest_error("field 'release_asset' must be an object");
  }
  reject_unknown_fields(release_asset, {"model", "model_sha256"});
  manifest.release_asset.model = require_non_empty_string(release_asset, "model");
  manifest.release_asset.model_sha256 =
      require_hex_field(release_asset, "model_sha256", 64, "a sha256 digest");

  return manifest;
}

} // namespace mlx::core::omarchy::ane
