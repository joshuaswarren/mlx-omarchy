// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Linux-side ANE bundle validation tests (U6). These run on the host with no
// device access: they prove the manifest and libane ANEC header contract reject
// malformed fields before any payload mapping or device submit.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "mlx/backend/omarchy/ane/bundle.h"
#include "mlx/backend/omarchy/ane/manifest.h"

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <unistd.h>

using namespace mlx::core::omarchy::ane;

namespace {

constexpr uint64_t kStride0x4000 = 0x4000;
constexpr uint64_t kWorkspace0x4000 = 0x4000;
constexpr uint64_t kDefaultPayloadSize = 0x4000;
const std::string kWeightsContent(32, 'B');

std::string hex64(char seed) {
  return std::string(64, seed);
}

std::string hex40(char seed) {
  return std::string(40, seed);
}

template <typename T>
void write_le(std::string& bytes, size_t offset, T value) {
  static_assert(std::is_integral_v<T>);
  REQUIRE(offset + sizeof(T) <= bytes.size());
  std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

void write_nchw(std::string& bytes, uint32_t bdx, const std::array<uint64_t, 6>& dims) {
  size_t offset = 40 + kAnecTileCount * sizeof(uint32_t) + bdx * dims.size() * sizeof(uint64_t);
  for (uint64_t dim : dims) {
    write_le(bytes, offset, dim);
    offset += sizeof(uint64_t);
  }
}

class TempDir {
 public:
  TempDir() {
    namespace fs = std::filesystem;
    fs::path base = fs::temp_directory_path() / "mlx-omarchy-ane-bundle-test";
    fs::create_directories(base);
    for (int attempt = 0; attempt < 64; ++attempt) {
      fs::path candidate =
          base / ("case-" + std::to_string(attempt) + "-" + std::to_string(::getpid()));
      std::error_code error;
      if (fs::create_directory(candidate, error) && !error) {
        path_ = candidate;
        return;
      }
    }
    throw std::runtime_error("cannot create temp directory");
  }

  ~TempDir() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  const std::filesystem::path& path() const {
    return path_;
  }

 private:
  std::filesystem::path path_;
};

int bundle_dir_count = 0;

void write_file(const std::filesystem::path& path, const std::string& content) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out << content;
  REQUIRE(out.good());
}

struct FixtureManifest {
  int manifest_version = 1;
  bool omit_manifest_version = false;
  std::string name = "ane-add-fp16-1x512";
  std::string graph_hash = hex64('1');
  int task_descriptors = 1;
  std::string input_shape = "[1, 512]";
  int input_byte_size = 1024;
  int input_stride = static_cast<int>(kStride0x4000);
  std::string output_shape = "[1, 512]";
  std::string output_dtype = "float16";
  int output_byte_size = 1024;
  int output_stride = static_cast<int>(kStride0x4000);
  bool include_state = false;
  int state_byte_size = 131072;
  int state_stride = 131072;
  int workspace_byte_size = static_cast<int>(kWorkspace0x4000);
  int workspace_stride = static_cast<int>(kWorkspace0x4000);
  bool include_anec = true;
  std::string anec_path = "model.anec";
  std::string anec_sha256_override;
  bool include_weights = true;
  std::string weights_path = "weights.bin";
  std::string weights_sha256_override;
  bool omit_compiler = false;
  std::string macos_build = "20A2411";
  std::string anecompiler = "ANECompiler 5.5.0 (Monterey)";
  std::string firmware_min = "13.0";
  std::string firmware_max = "13.5";
  std::string source_repo = "joshuaswarren/mlx-omarchy";
  std::string source_commit = hex40('c');
  std::string exported_at = "2026-08-30";
  std::string model = "ane-add-fp16-1x512";
  std::string model_sha256 = hex64('4');
  uint64_t anec_payload_size = kDefaultPayloadSize;
  uint64_t anec_file_payload_bytes = kDefaultPayloadSize;
  uint32_t anec_td_size = 0x274;
  int anec_td_count_override = -1;
  uint64_t anec_task_size = 0x1f8;
  uint64_t anec_kernel_size = 0x400;
  uint32_t command_tiles = 1;
  uint32_t kernel_tiles = 0;
  int64_t anec_source_count_override = -1;
  int64_t anec_destination_count_override = -1;
  uint32_t input_tiles = 2;
  uint32_t output_tiles = 2;
  uint32_t state_tiles = 8;

  uint32_t source_count() const {
    return anec_source_count_override >= 0 ? static_cast<uint32_t>(anec_source_count_override)
                                           : static_cast<uint32_t>(1 + (include_state ? 1 : 0));
  }

  uint32_t destination_count() const {
    return anec_destination_count_override >= 0
        ? static_cast<uint32_t>(anec_destination_count_override)
        : static_cast<uint32_t>(1 + (include_state ? 1 : 0));
  }

  uint32_t task_descriptor_count() const {
    return anec_td_count_override >= 0 ? static_cast<uint32_t>(anec_td_count_override)
                                       : static_cast<uint32_t>(task_descriptors);
  }

  std::string anec_bytes() const {
    std::string bytes(kAnecPayloadOffset + anec_file_payload_bytes, '\0');
    for (uint64_t i = 0; i < anec_file_payload_bytes; ++i) {
      bytes[kAnecPayloadOffset + i] = static_cast<char>('A' + (i % 23));
    }

    write_le<uint64_t>(bytes, 0, anec_payload_size);
    write_le<uint32_t>(bytes, 8, anec_td_size);
    write_le<uint32_t>(bytes, 12, task_descriptor_count());
    write_le<uint64_t>(bytes, 16, anec_task_size);
    write_le<uint64_t>(bytes, 24, anec_kernel_size);
    write_le<uint32_t>(bytes, 32, source_count());
    write_le<uint32_t>(bytes, 36, destination_count());

    write_le<uint32_t>(bytes, 40, command_tiles);
    write_le<uint32_t>(bytes, 40 + sizeof(uint32_t), kernel_tiles);
    write_le<uint32_t>(bytes, 40 + output_bdx() * sizeof(uint32_t), output_tiles);
    write_le<uint32_t>(bytes, 40 + input_bdx(0) * sizeof(uint32_t), input_tiles);
    write_nchw(bytes, output_bdx(), {1, 512, 1, 1, 64, 64});
    write_nchw(bytes, input_bdx(0), {1, 512, 1, 1, 64, 64});
    if (include_state) {
      write_le<uint32_t>(bytes, 40 + state_dst_bdx() * sizeof(uint32_t), state_tiles);
      write_le<uint32_t>(bytes, 40 + state_src_bdx() * sizeof(uint32_t), state_tiles);
      write_nchw(bytes, state_dst_bdx(), {1, 2, 128, 256, 65536, 512});
      write_nchw(bytes, state_src_bdx(), {1, 2, 128, 256, 65536, 512});
    }
    return bytes;
  }

  uint32_t output_bdx() const {
    return 4;
  }

  uint32_t state_dst_bdx() const {
    return 5;
  }

  uint32_t input_bdx(uint32_t ordinal) const {
    return 4 + destination_count() + ordinal;
  }

  uint32_t state_src_bdx() const {
    return input_bdx(1);
  }

  std::string anec_sha256() const {
    if (!anec_sha256_override.empty()) {
      return anec_sha256_override;
    }
    const auto bytes = anec_bytes();
    return sha256_hex(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
  }

  std::string weights_sha256() const {
    if (!weights_sha256_override.empty()) {
      return weights_sha256_override;
    }
    return sha256_hex(
        reinterpret_cast<const uint8_t*>(kWeightsContent.data()),
        kWeightsContent.size());
  }

  std::string render() const {
    const auto anec = anec_bytes();
    std::string json = "{\n";
    if (!omit_manifest_version) {
      json += "  \"manifest_version\": " + std::to_string(manifest_version) + ",\n";
    }
    json += "  \"name\": \"" + name + "\",\n";
    json += "  \"graph_hash\": \"" + graph_hash + "\",\n";
    json += "  \"task_descriptors\": " + std::to_string(task_descriptors) + ",\n";
    json += "  \"inputs\": [\n";
    json += "    {\"name\": \"input\", \"index\": 0, \"dtype\": \"float16\",";
    json += " \"shape\": " + input_shape + ",";
    json += " \"byte_size\": " + std::to_string(input_byte_size) + ",";
    json += " \"stride\": " + std::to_string(input_stride) + "}\n";
    json += "  ],\n";
    json += "  \"outputs\": [\n";
    json += "    {\"name\": \"output\", \"index\": 1, \"dtype\": \"" +
        output_dtype + "\",";
    json += " \"shape\": " + output_shape + ",";
    json += " \"byte_size\": " + std::to_string(output_byte_size) + ",";
    json += " \"stride\": " + std::to_string(output_stride) + "}\n";
    json += "  ],\n";
    json += "  \"state\": ";
    if (include_state) {
      json += "[\n";
      json += "    {\"name\": \"kv_state\", \"index\": 2, \"dtype\": \"float16\",";
      json += " \"shape\": [1, 2, 128, 256],";
      json += " \"byte_size\": " + std::to_string(state_byte_size) + ",";
      json += " \"stride\": " + std::to_string(state_stride) + "}\n";
      json += "  ],\n";
    } else {
      json += "[],\n";
    }
    json += "  \"workspace\": [\n";
    json += "    {\"name\": \"workspace\", \"index\": 0, \"dtype\": \"uint8\",";
    json += " \"shape\": [" + std::to_string(workspace_byte_size) + "],";
    json += " \"byte_size\": " + std::to_string(workspace_byte_size) + ",";
    json += " \"stride\": " + std::to_string(workspace_stride) + "}\n";
    json += "  ],\n";
    json += "  \"payloads\": [\n";
    if (include_anec) {
      json += "    {\"role\": \"anec\", \"path\": \"" + anec_path + "\",";
      json += " \"sha256\": \"" + anec_sha256() + "\", \"byte_size\": " +
          std::to_string(anec.size()) + "}";
      if (include_weights) {
        json += ",\n";
      }
    }
    if (include_weights) {
      json += "    {\"role\": \"weights\", \"path\": \"" + weights_path + "\",";
      json += " \"sha256\": \"" + weights_sha256() + "\", \"byte_size\": " +
          std::to_string(kWeightsContent.size()) + "}";
    }
    json += "\n  ],\n";
    if (!omit_compiler) {
      json += "  \"compiler\": {\"macos_build\": \"" + macos_build + "\",";
      json += " \"anecompiler\": \"" + anecompiler + "\"},\n";
    }
    json += "  \"firmware\": {\"min\": \"" + firmware_min + "\",";
    json += " \"max\": \"" + firmware_max + "\"},\n";
    json += "  \"provenance\": {\"source_repo\": \"" + source_repo + "\",";
    json += " \"source_commit\": \"" + source_commit + "\",";
    json += " \"exported_at\": \"" + exported_at + "\"},\n";
    json += "  \"release_asset\": {\"model\": \"" + model + "\",";
    json += " \"model_sha256\": \"" + model_sha256 + "\"}\n";
    json += "}\n";
    return json;
  }
};

std::filesystem::path write_bundle(
    const TempDir& temp,
    const FixtureManifest& fixture,
    bool write_anec = true,
    bool write_weights = true,
    const std::string& extra_file = "") {
  namespace fs = std::filesystem;
  fs::path dir = temp.path() / ("bundle-" + std::to_string(bundle_dir_count++));
  fs::create_directories(dir);
  write_file(dir / "manifest.json", fixture.render());
  if (fixture.include_anec && write_anec) {
    write_file(dir / fixture.anec_path, fixture.anec_bytes());
  }
  if (fixture.include_weights && write_weights) {
    write_file(dir / fixture.weights_path, kWeightsContent);
  }
  if (!extra_file.empty()) {
    write_file(dir / extra_file, "unlisted");
  }
  return dir;
}

std::string load_error(const std::filesystem::path& dir) {
  try {
    load_bundle(dir);
  } catch (const AneBundleNotFound& error) {
    return std::string("not_found: ") + error.what();
  } catch (const std::exception& error) {
    return error.what();
  }
  return {};
}

} // namespace

TEST_CASE("valid bundle exposes its exact contracts") {
  TempDir temp;
  FixtureManifest fixture;
  auto dir = write_bundle(temp, fixture);

  AneBundle bundle = load_bundle(dir);
  CHECK(bundle.manifest.manifest_version == 1);
  CHECK(bundle.manifest.name == "ane-add-fp16-1x512");
  CHECK(bundle.manifest.graph_hash == hex64('1'));
  CHECK(bundle.manifest.task_descriptors == 1);

  REQUIRE_EQ(bundle.manifest.inputs.size(), size_t(1));
  CHECK(bundle.manifest.inputs[0].name == "input");
  CHECK(bundle.manifest.inputs[0].index == 0);
  CHECK(bundle.manifest.inputs[0].dtype == "float16");
  REQUIRE_EQ(bundle.manifest.inputs[0].shape.size(), size_t(2));
  CHECK(bundle.manifest.inputs[0].shape[0] == 1);
  CHECK(bundle.manifest.inputs[0].shape[1] == 512);
  CHECK(bundle.manifest.inputs[0].byte_size == 1024);
  CHECK(bundle.manifest.inputs[0].stride == kStride0x4000);

  REQUIRE_EQ(bundle.manifest.outputs.size(), size_t(1));
  CHECK(bundle.manifest.outputs[0].name == "output");
  CHECK(bundle.manifest.outputs[0].index == 1);
  CHECK(bundle.manifest.outputs[0].dtype == "float16");
  CHECK(bundle.manifest.outputs[0].byte_size == 1024);

  REQUIRE_EQ(bundle.manifest.workspace.size(), size_t(1));
  CHECK(bundle.manifest.workspace[0].dtype == "uint8");
  CHECK(bundle.manifest.workspace[0].byte_size == kWorkspace0x4000);
  CHECK(bundle.manifest.workspace[0].stride == kWorkspace0x4000);

  CHECK(bundle.anec == dir / "model.anec");
  REQUIRE(bundle.weights.has_value());
  CHECK(bundle.weights.value() == dir / "weights.bin");

  CHECK(bundle.anec_header.payload_size == kDefaultPayloadSize);
  CHECK(bundle.anec_header.task_descriptor_size == 0x274);
  CHECK(bundle.anec_header.task_descriptor_count == 1);
  CHECK(bundle.anec_header.source_count == 1);
  CHECK(bundle.anec_header.destination_count == 1);
  CHECK(bundle.anec_header.bootstrap_channel_size == kStride0x4000);
  CHECK(bundle.anec_header.tiles[fixture.output_bdx()] == 2);
  CHECK(bundle.anec_header.tiles[fixture.input_bdx(0)] == 2);
  CHECK(bundle.anec_header.nchw[fixture.output_bdx()][1] == 512);
  CHECK(bundle.anec_header.nchw[fixture.input_bdx(0)][4] == 64);

  CHECK(bundle.manifest.compiler.macos_build == "20A2411");
  CHECK(bundle.manifest.compiler.anecompiler == "ANECompiler 5.5.0 (Monterey)");
  CHECK(bundle.manifest.firmware.min == "13.0");
  CHECK(bundle.manifest.firmware.max == "13.5");
  CHECK(bundle.manifest.provenance.source_repo == "joshuaswarren/mlx-omarchy");
  CHECK(bundle.manifest.provenance.source_commit == hex40('c'));
  CHECK(bundle.manifest.provenance.exported_at == "2026-08-30");
  CHECK(bundle.manifest.release_asset.model == "ane-add-fp16-1x512");
  CHECK(bundle.manifest.release_asset.model_sha256 == hex64('4'));
}

TEST_CASE("state channels bind as both source and destination") {
  TempDir temp;
  FixtureManifest fixture;
  fixture.include_state = true;
  auto dir = write_bundle(temp, fixture);

  AneBundle bundle = load_bundle(dir);
  REQUIRE_EQ(bundle.manifest.state.size(), size_t(1));
  CHECK(bundle.anec_header.source_count == 2);
  CHECK(bundle.anec_header.destination_count == 2);
  CHECK(bundle.anec_header.tiles[fixture.output_bdx()] == 2);
  CHECK(bundle.anec_header.tiles[fixture.state_dst_bdx()] == 8);
  CHECK(bundle.anec_header.tiles[fixture.input_bdx(0)] == 2);
  CHECK(bundle.anec_header.tiles[fixture.state_src_bdx()] == 8);
}

TEST_CASE("missing bundle directory returns not_found") {
  TempDir temp;
  auto missing = temp.path() / "does-not-exist";
  try {
    load_bundle(missing);
    FAIL("expected AneBundleNotFound");
  } catch (const AneBundleNotFound& error) {
    CHECK(std::string(error.what()).find("not found") != std::string::npos);
    CHECK(std::string(error.what()).find("Vulkan") != std::string::npos);
  }
}

TEST_CASE("directory without manifest.json is a manifest error") {
  TempDir temp;
  auto dir = temp.path() / ("empty-" + std::to_string(bundle_dir_count++));
  std::filesystem::create_directories(dir);
  std::string message = load_error(dir);
  CHECK(message.find("cannot open") != std::string::npos);
}

TEST_CASE("repeated identical manifests yield identical graph identity") {
  TempDir temp;
  auto dir = write_bundle(temp, FixtureManifest{});
  AneManifest first = parse_ane_manifest(dir / "manifest.json");
  AneManifest second = parse_ane_manifest(dir / "manifest.json");
  CHECK(first.graph_hash == second.graph_hash);
  CHECK(first.name == second.name);
  CHECK(first.task_descriptors == second.task_descriptors);
  REQUIRE_EQ(first.inputs.size(), second.inputs.size());
  CHECK(first.inputs[0].shape == second.inputs[0].shape);
  CHECK(first.inputs[0].byte_size == second.inputs[0].byte_size);
  CHECK(first.inputs[0].stride == second.inputs[0].stride);
  CHECK(first.workspace[0].byte_size == second.workspace[0].byte_size);
  CHECK(first.compiler.anecompiler == second.compiler.anecompiler);
  CHECK(first.firmware.min == second.firmware.min);
  CHECK(first.release_asset.model_sha256 == second.release_asset.model_sha256);
}

TEST_CASE("sha256_file matches the manifest descriptor digest") {
  TempDir temp;
  FixtureManifest fixture;
  auto dir = write_bundle(temp, fixture);
  AneBundle bundle = load_bundle(dir);
  CHECK(sha256_file(dir / "model.anec") == bundle.manifest.payloads[0].sha256);
  CHECK(sha256_file(dir / "weights.bin") == bundle.manifest.payloads[1].sha256);
}

TEST_CASE("single-field mutations fail before payload access") {
  TempDir temp;

  SUBCASE("unknown manifest_version") {
    FixtureManifest fixture;
    fixture.manifest_version = 2;
    std::string message = load_error(write_bundle(temp, fixture, false, false));
    CHECK(message.find("unsupported manifest_version") != std::string::npos);
  }
  SUBCASE("missing manifest_version") {
    FixtureManifest fixture;
    fixture.omit_manifest_version = true;
    std::string message = load_error(write_bundle(temp, fixture, false, false));
    CHECK(message.find("missing field 'manifest_version'") != std::string::npos);
  }
  SUBCASE("short graph_hash") {
    FixtureManifest fixture;
    fixture.graph_hash = "42";
    std::string message = load_error(write_bundle(temp, fixture, false, false));
    CHECK(message.find("'graph_hash' must be a sha256 digest") != std::string::npos);
  }
  SUBCASE("uppercase graph_hash") {
    FixtureManifest fixture;
    fixture.graph_hash = std::string(64, 'G');
    std::string message = load_error(write_bundle(temp, fixture, false, false));
    CHECK(message.find("'graph_hash' must be a sha256 digest") != std::string::npos);
  }
  SUBCASE("missing compiler") {
    FixtureManifest fixture;
    fixture.omit_compiler = true;
    std::string message = load_error(write_bundle(temp, fixture, false, false));
    CHECK(message.find("missing field 'compiler'") != std::string::npos);
  }
  SUBCASE("empty anecompiler") {
    FixtureManifest fixture;
    fixture.anecompiler = "";
    std::string message = load_error(write_bundle(temp, fixture, false, false));
    CHECK(message.find("'anecompiler' must not be empty") != std::string::npos);
  }
  SUBCASE("changed shape") {
    FixtureManifest fixture;
    fixture.input_shape = "[1, 1024]";
    std::string message = load_error(write_bundle(temp, fixture, false, false));
    CHECK(message.find("does not match dtype geometry") != std::string::npos);
  }
  SUBCASE("non-positive shape") {
    FixtureManifest fixture;
    fixture.input_shape = "[1, 0]";
    std::string message = load_error(write_bundle(temp, fixture, false, false));
    CHECK(message.find("must contain positive integers") != std::string::npos);
  }
  SUBCASE("misaligned stride") {
    FixtureManifest fixture;
    fixture.input_stride = 4096;
    std::string message = load_error(write_bundle(temp, fixture, false, false));
    CHECK(message.find("not a multiple of 0x4000") != std::string::npos);
  }
  SUBCASE("stride below byte_size") {
    FixtureManifest fixture;
    fixture.include_state = true;
    fixture.state_stride = static_cast<int>(kStride0x4000);
    std::string message = load_error(write_bundle(temp, fixture, false, false));
    CHECK(message.find("stride must be at least byte_size") != std::string::npos);
  }
  SUBCASE("malformed firmware version") {
    FixtureManifest fixture;
    fixture.firmware_min = "thirteen";
    std::string message = load_error(write_bundle(temp, fixture, false, false));
    CHECK(message.find("not a dotted integer version") != std::string::npos);
  }
  SUBCASE("inverted firmware range") {
    FixtureManifest fixture;
    fixture.firmware_min = "13.5";
    fixture.firmware_max = "13.0";
    std::string message = load_error(write_bundle(temp, fixture, false, false));
    CHECK(message.find("max must be at least min") != std::string::npos);
  }
  SUBCASE("malformed descriptor hash format") {
    FixtureManifest fixture;
    fixture.anec_sha256_override = "short";
    std::string message = load_error(write_bundle(temp, fixture, false, false));
    CHECK(message.find("must be a sha256 digest") != std::string::npos);
  }
  SUBCASE("malformed provenance commit") {
    FixtureManifest fixture;
    fixture.source_commit = std::string(40, 'x');
    std::string message = load_error(write_bundle(temp, fixture, false, false));
    CHECK(message.find("git commit hash") != std::string::npos);
  }
  SUBCASE("malformed exported_at") {
    FixtureManifest fixture;
    fixture.exported_at = "yesterday";
    std::string message = load_error(write_bundle(temp, fixture, false, false));
    CHECK(message.find("YYYY-MM-DD") != std::string::npos);
  }
  SUBCASE("malformed release model hash") {
    FixtureManifest fixture;
    fixture.model_sha256 = "deadbeef";
    std::string message = load_error(write_bundle(temp, fixture, false, false));
    CHECK(message.find("must be a sha256 digest") != std::string::npos);
  }
  SUBCASE("unknown top-level field") {
    FixtureManifest fixture;
    std::string json = fixture.render();
    json.insert(1, "\"surprise\": true,\n");
    auto dir = write_bundle(temp, fixture, false, false);
    write_file(dir / "manifest.json", json);
    std::string message = load_error(dir);
    CHECK(message.find("unknown field 'surprise'") != std::string::npos);
  }
  SUBCASE("wrong task_descriptors type") {
    FixtureManifest fixture;
    std::string json = fixture.render();
    auto value = json.find("\"task_descriptors\": ");
    REQUIRE(value != std::string::npos);
    auto line_end = json.find(",\n", value);
    REQUIRE(line_end != std::string::npos);
    json.replace(value, line_end - value, "\"task_descriptors\": \"many\"");
    auto dir = write_bundle(temp, fixture, false, false);
    write_file(dir / "manifest.json", json);
    std::string message = load_error(dir);
    CHECK(message.find("'task_descriptors' must be a non-negative integer") !=
          std::string::npos);
  }
}

TEST_CASE("descriptor hash mismatch fails after field checks") {
  TempDir temp;
  FixtureManifest fixture;
  fixture.anec_sha256_override = hex64('f');
  auto dir = write_bundle(temp, fixture);
  std::string message = load_error(dir);
  CHECK(message.find("sha256 mismatch") != std::string::npos);
  CHECK(message.find("model.anec") != std::string::npos);
}

TEST_CASE("payload byte size mismatch fails before hashing") {
  TempDir temp;
  auto dir = write_bundle(temp, FixtureManifest{});
  write_file(dir / "model.anec", std::string(32, 'A'));
  std::string message = load_error(dir);
  CHECK(message.find("byte size") != std::string::npos);
}

TEST_CASE("ANEC task count mismatch fails after digest verification") {
  TempDir temp;
  FixtureManifest fixture;
  fixture.anec_td_count_override = 2;
  auto dir = write_bundle(temp, fixture);
  std::string message = load_error(dir);
  CHECK(message.find("ANEC task_descriptors") != std::string::npos);
}

TEST_CASE("ANEC source and destination counts must match manifest channels") {
  TempDir temp;
  FixtureManifest fixture;
  fixture.anec_source_count_override = 0;
  auto dir = write_bundle(temp, fixture);
  std::string message = load_error(dir);
  CHECK(message.find("ANEC source_count") != std::string::npos);
}
TEST_CASE("ANEC channel count arithmetic cannot wrap") {
  TempDir temp;
  FixtureManifest fixture;
  fixture.anec_source_count_override = 0xffffffffu;
  auto dir = write_bundle(temp, fixture);
  std::string message = load_error(dir);
  CHECK(message.find("exceeds libane channel table") != std::string::npos);
}
TEST_CASE("ANEC executable payload envelope is checked after digest verification") {
  TempDir temp;
  FixtureManifest fixture;
  fixture.anec_payload_size = 0x8000;
  fixture.anec_file_payload_bytes = 0x4000;
  auto dir = write_bundle(temp, fixture);
  std::string message = load_error(dir);
  CHECK(message.find("executable payload extends past file") != std::string::npos);
}

TEST_CASE("ANEC command channel must hold the payload copied by libane") {
  TempDir temp;
  FixtureManifest fixture;
  fixture.anec_payload_size = 0x8000;
  fixture.anec_file_payload_bytes = 0x8000;
  auto dir = write_bundle(temp, fixture);
  std::string message = load_error(dir);
  CHECK(message.find("payload exceeds command channel allocation") != std::string::npos);
}

TEST_CASE("ANEC kernel channel must stay unbound for the packed command buffer") {
  TempDir temp;
  FixtureManifest fixture;
  fixture.kernel_tiles = 1;
  auto dir = write_bundle(temp, fixture);
  std::string message = load_error(dir);
  CHECK(message.find("reserved kernel channel allocation must be zero") !=
        std::string::npos);
}

TEST_CASE("ANEC task descriptor bytes must fit the loaded payload") {
  TempDir temp;
  FixtureManifest fixture;
  fixture.anec_td_size = 0x4004;
  auto dir = write_bundle(temp, fixture);
  std::string message = load_error(dir);
  CHECK(message.find("task descriptor bytes exceed executable payload") !=
        std::string::npos);
}

TEST_CASE("ANEC task size must satisfy the driver command buffer contract") {
  TempDir temp;
  FixtureManifest fixture;
  fixture.anec_task_size = 0;
  auto dir = write_bundle(temp, fixture);
  std::string message = load_error(dir);
  CHECK(message.find("task size is zero") != std::string::npos);

  fixture.anec_task_size = kDefaultPayloadSize;
  dir = write_bundle(temp, fixture);
  message = load_error(dir);
  CHECK(message.find("task size reaches command channel end") != std::string::npos);
}

TEST_CASE("ANEC kernel bytes start at the aligned task boundary") {
  TempDir temp;
  FixtureManifest fixture;
  fixture.anec_task_size = 0x1ff;
  fixture.anec_kernel_size = kDefaultPayloadSize - fixture.anec_task_size;
  auto dir = write_bundle(temp, fixture);
  std::string message = load_error(dir);
  CHECK(message.find("task plus kernel bytes exceed executable payload") !=
        std::string::npos);
}

TEST_CASE("ANEC task descriptor size must match the driver's word unit") {
  TempDir temp;
  FixtureManifest fixture;
  fixture.anec_td_size = 2;
  auto dir = write_bundle(temp, fixture);
  std::string message = load_error(dir);
  CHECK(message.find("task descriptor size is not a multiple of 4") !=
        std::string::npos);
}

TEST_CASE("ANEC tiled channels reject non-16-bit manifest tensors") {
  TempDir temp;
  FixtureManifest fixture;
  fixture.output_dtype = "uint8";
  fixture.output_byte_size = 512;
  auto dir = write_bundle(temp, fixture);
  std::string message = load_error(dir);
  CHECK(message.find("tiled channel requires a 16-bit tensor dtype") !=
        std::string::npos);
}

TEST_CASE("manifest workspace is separate from libane's bootstrap allocation") {
  TempDir temp;
  FixtureManifest fixture;
  fixture.workspace_byte_size = 1024;
  fixture.workspace_stride = 1024;
  auto dir = write_bundle(temp, fixture);
  CHECK_NOTHROW(load_bundle(dir));
}

TEST_CASE("ANEC NCHW must match manifest shape") {
  TempDir temp;
  FixtureManifest fixture;
  fixture.output_shape = "[1, 256]";
  fixture.output_byte_size = 512;
  auto dir = write_bundle(temp, fixture);
  std::string message = load_error(dir);
  CHECK(message.find("ANEC NCHW does not match manifest shape") != std::string::npos);
}

TEST_CASE("missing payload file is named") {
  TempDir temp;
  auto dir = write_bundle(
      temp, FixtureManifest{}, /*write_anec=*/false, /*write_weights=*/true);
  std::string message = load_error(dir);
  CHECK(message.find("payload file missing") != std::string::npos);
  CHECK(message.find("model.anec") != std::string::npos);
}

TEST_CASE("unknown extra payload file is rejected") {
  TempDir temp;
  auto dir = write_bundle(temp, FixtureManifest{}, true, true, "extra.bin");
  std::string message = load_error(dir);
  CHECK(message.find("unknown payload file") != std::string::npos);
  CHECK(message.find("extra.bin") != std::string::npos);
}

TEST_CASE("payload path traversal is rejected") {
  TempDir temp;
  FixtureManifest fixture;
  fixture.anec_path = "../escape.anec";
  auto dir = write_bundle(temp, fixture, /*write_anec=*/false, /*write_weights=*/true);
  std::string message = load_error(dir);
  CHECK(message.find("must be relative") != std::string::npos);
}

TEST_CASE("bundle without an anec payload is rejected") {
  TempDir temp;
  FixtureManifest fixture;
  fixture.include_anec = false;
  auto dir = write_bundle(temp, fixture, /*write_anec=*/false, /*write_weights=*/true);
  std::string message = load_error(dir);
  CHECK(message.find("exactly one 'anec'") != std::string::npos);
}
