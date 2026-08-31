// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Linux-side ANE bundle validation tests (U6). These run on the host with no
// device access: they prove the manifest and bundle loader reject every
// malformed contract field with its named reason before any payload mapping.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "mlx/backend/omarchy/ane/bundle.h"
#include "mlx/backend/omarchy/ane/manifest.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>

using namespace mlx::core::omarchy::ane;

namespace {

// Tile-aligned DMA stride for the fp16 activation tensors.
constexpr uint64_t kStride0x4000 = 0x4000;
// Workspace size from the terminal-task receipt (13-layer graph).
constexpr uint64_t kWorkspace0x66000 = 0x66000;

// Payload byte contents written by write_bundle.
const std::string kAnecContent(64, 'A');
const std::string kWeightsContent(32, 'B');

// Deterministic digest-shaped strings. Not real hashes; only fixture values.
std::string hex64(char seed) {
  return std::string(64, seed);
}

std::string hex40(char seed) {
  return std::string(40, seed);
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

// The valid fixture manifest. Field values trace to ane-linux-experiments
// receipts; see docs/ane-bundles.md. The payload digests match the bytes
// write_bundle writes, so a default fixture is a valid bundle. Tests mutate
// exactly one member and re-render to prove each field check fails closed
// with its named reason.
struct FixtureManifest {
  int manifest_version = 1;
  bool omit_manifest_version = false;
  std::string name = "qwen3.8-chunk13";
  std::string graph_hash = hex64('1');
  int task_descriptors = 1403;
  std::string input_shape = "[1, 2048]";
  int input_byte_size = 4096;
  int input_stride = static_cast<int>(kStride0x4000);
  int state_byte_size = 131072;  // 2 kv heads x 128 context x 256 dim x fp16
  int state_stride = 131072;
  int workspace_byte_size = static_cast<int>(kWorkspace0x66000);
  bool include_anec = true;
  std::string anec_path = "model.anec";
  std::string anec_sha256;
  bool include_weights = true;
  std::string weights_path = "weights.bin";
  std::string weights_sha256;
  bool omit_compiler = false;
  std::string macos_build = "20A2411";
  std::string anecompiler = "ANECompiler 5.5.0 (Monterey)";
  std::string firmware_min = "13.0";
  std::string firmware_max = "13.5";
  std::string source_repo = "joshuaswarren/mlx-omarchy";
  std::string source_commit = hex40('c');
  std::string exported_at = "2026-08-30";
  std::string model = "Qwen3.8-2B-Q4_K_M";
  std::string model_sha256 =
      "4aa0fb13c431514262f259d420ecc95a8714df58ac2a2384514e20b93983f0ff";

  FixtureManifest() {
    anec_sha256 = sha256_hex(
        reinterpret_cast<const uint8_t*>(kAnecContent.data()),
        kAnecContent.size());
    weights_sha256 = sha256_hex(
        reinterpret_cast<const uint8_t*>(kWeightsContent.data()),
        kWeightsContent.size());
  }

  std::string render() const {
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
    json += "    {\"name\": \"output\", \"index\": 1, \"dtype\": \"float16\",";
    json += " \"shape\": [1, 2048], \"byte_size\": 4096,";
    json += " \"stride\": " + std::to_string(kStride0x4000) + "}\n";
    json += "  ],\n";
    json += "  \"state\": [\n";
    json += "    {\"name\": \"kv_state\", \"index\": 2, \"dtype\": \"float16\",";
    json += " \"shape\": [1, 2, 128, 256],";
    json += " \"byte_size\": " + std::to_string(state_byte_size) + ",";
    json += " \"stride\": " + std::to_string(state_stride) + "}\n";
    json += "  ],\n";
    json += "  \"workspace\": [\n";
    json += "    {\"name\": \"workspace\", \"index\": 0, \"dtype\": \"uint8\",";
    json += " \"shape\": [" + std::to_string(workspace_byte_size) + "],";
    json += " \"byte_size\": " + std::to_string(workspace_byte_size) + ",";
    json += " \"stride\": " + std::to_string(workspace_byte_size) + "}\n";
    json += "  ],\n";
    json += "  \"payloads\": [\n";
    if (include_anec) {
      json += "    {\"role\": \"anec\", \"path\": \"" + anec_path + "\",";
      json += " \"sha256\": \"" + anec_sha256 + "\", \"byte_size\": 64}";
      if (include_weights) {
        json += ",\n";
      }
    }
    if (include_weights) {
      json += "    {\"role\": \"weights\", \"path\": \"" + weights_path + "\",";
      json += " \"sha256\": \"" + weights_sha256 + "\", \"byte_size\": 32}";
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

// One fresh bundle directory. Payload contents are arbitrary bytes: the loader
// verifies identity and integrity only, and this layer never executes ANEC.
// write_anec / write_weights control whether the payload files exist.
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
    write_file(dir / fixture.anec_path, kAnecContent);
  }
  if (fixture.include_weights && write_weights) {
    write_file(dir / fixture.weights_path, kWeightsContent);
  }
  if (!extra_file.empty()) {
    write_file(dir / extra_file, "unlisted");
  }
  return dir;
}

// Loads the bundle and returns the thrown error message.
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
  auto dir = write_bundle(temp, FixtureManifest{});

  AneBundle bundle = load_bundle(dir);
  CHECK(bundle.manifest.manifest_version == 1);
  CHECK(bundle.manifest.name == "qwen3.8-chunk13");
  CHECK(bundle.manifest.graph_hash == hex64('1'));
  CHECK(bundle.manifest.task_descriptors == 1403);

  REQUIRE_EQ(bundle.manifest.inputs.size(), size_t(1));
  CHECK(bundle.manifest.inputs[0].name == "input");
  CHECK(bundle.manifest.inputs[0].index == 0);
  CHECK(bundle.manifest.inputs[0].dtype == "float16");
  REQUIRE_EQ(bundle.manifest.inputs[0].shape.size(), size_t(2));
  CHECK(bundle.manifest.inputs[0].shape[0] == 1);
  CHECK(bundle.manifest.inputs[0].shape[1] == 2048);
  CHECK(bundle.manifest.inputs[0].byte_size == 4096);
  CHECK(bundle.manifest.inputs[0].stride == kStride0x4000);

  REQUIRE_EQ(bundle.manifest.outputs.size(), size_t(1));
  CHECK(bundle.manifest.outputs[0].name == "output");
  CHECK(bundle.manifest.outputs[0].index == 1);
  CHECK(bundle.manifest.outputs[0].dtype == "float16");
  CHECK(bundle.manifest.outputs[0].byte_size == 4096);

  REQUIRE_EQ(bundle.manifest.state.size(), size_t(1));
  CHECK(bundle.manifest.state[0].name == "kv_state");
  CHECK(bundle.manifest.state[0].index == 2);
  CHECK(bundle.manifest.state[0].dtype == "float16");
  CHECK(bundle.manifest.state[0].byte_size == 131072);
  CHECK(bundle.manifest.state[0].stride == 131072);

  REQUIRE_EQ(bundle.manifest.workspace.size(), size_t(1));
  CHECK(bundle.manifest.workspace[0].dtype == "uint8");
  CHECK(bundle.manifest.workspace[0].byte_size == kWorkspace0x66000);
  CHECK(bundle.manifest.workspace[0].stride == kWorkspace0x66000);

  CHECK(bundle.anec == dir / "model.anec");
  REQUIRE(bundle.weights.has_value());
  CHECK(bundle.weights.value() == dir / "weights.bin");

  CHECK(bundle.manifest.compiler.macos_build == "20A2411");
  CHECK(bundle.manifest.compiler.anecompiler == "ANECompiler 5.5.0 (Monterey)");
  CHECK(bundle.manifest.firmware.min == "13.0");
  CHECK(bundle.manifest.firmware.max == "13.5");
  CHECK(bundle.manifest.provenance.source_repo == "joshuaswarren/mlx-omarchy");
  CHECK(bundle.manifest.provenance.source_commit == hex40('c'));
  CHECK(bundle.manifest.provenance.exported_at == "2026-08-30");
  CHECK(bundle.manifest.release_asset.model == "Qwen3.8-2B-Q4_K_M");
  CHECK(bundle.manifest.release_asset.model_sha256 ==
        "4aa0fb13c431514262f259d420ecc95a8714df58ac2a2384514e20b93983f0ff");
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

// Each mutation below changes exactly one field of an otherwise valid
// manifest. No payload files are written, so a "payload file missing" message
// would mean field validation ran after payload access. Every case must fail
// with its named field error instead: U6 requires validation before any
// payload mapping.
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
    fixture.graph_hash = "42";  // Still a JSON string, but too short for sha256.
    std::string message = load_error(write_bundle(temp, fixture, false, false));
    CHECK(message.find("'graph_hash' must be a sha256 digest") != std::string::npos);
  }
  SUBCASE("uppercase graph_hash") {
    FixtureManifest fixture;
    fixture.graph_hash = std::string(64, 'G');  // Uppercase is not hex.
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
    fixture.input_shape = "[1, 4096]";  // Geometry no longer matches byte_size.
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
    fixture.input_stride = 4096;  // 0x1000, not a 0x4000 multiple.
    std::string message = load_error(write_bundle(temp, fixture, false, false));
    CHECK(message.find("not a multiple of 0x4000") != std::string::npos);
  }
  SUBCASE("stride below byte_size") {
    FixtureManifest fixture;
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
    fixture.anec_sha256 = "short";
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
  fixture.anec_sha256 = hex64('f');  // Valid format, wrong digest.
  auto dir = write_bundle(temp, fixture);
  std::string message = load_error(dir);
  CHECK(message.find("sha256 mismatch") != std::string::npos);
  CHECK(message.find("model.anec") != std::string::npos);
}

TEST_CASE("payload byte size mismatch fails before hashing") {
  TempDir temp;
  auto dir = write_bundle(temp, FixtureManifest{});
  write_file(dir / "model.anec", std::string(32, 'A'));  // Manifest says 64.
  std::string message = load_error(dir);
  CHECK(message.find("byte size") != std::string::npos);
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
