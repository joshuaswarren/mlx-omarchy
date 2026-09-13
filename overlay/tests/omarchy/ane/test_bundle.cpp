// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "mlx/backend/omarchy/ane/bundle.h"
#include "json.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <unistd.h>

using namespace mlx::core::omarchy::ane;

namespace {

constexpr uint64_t kPayloadBytes = 0x4000;
constexpr uint64_t kAllocationBytes = 0x4000;
const std::array<uint64_t, 6> kNchw{1, 64, 1, 1, 64, 64};

std::string hex(size_t size, char value) {
  return std::string(size, value);
}

template <typename T>
void write_le(std::string& bytes, size_t offset, T value) {
  static_assert(std::is_integral_v<T>);
  REQUIRE(offset + sizeof(T) <= bytes.size());
  std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

void write_nchw(
    std::string& bytes,
    uint32_t channel,
    const std::array<uint64_t, 6>& nchw = kNchw) {
  size_t offset = 40 + kAnecTileCount * sizeof(uint32_t) +
      channel * nchw.size() * sizeof(uint64_t);
  for (uint64_t value : nchw) {
    write_le(bytes, offset, value);
    offset += sizeof(value);
  }
}

std::string anec_bytes(char seed) {
  std::string bytes(kAnecPayloadOffset + kPayloadBytes, '\0');
  for (uint64_t i = 0; i < kPayloadBytes; ++i) {
    bytes[kAnecPayloadOffset + i] = static_cast<char>(seed + (i % 7));
  }
  write_le<uint64_t>(bytes, 0, kPayloadBytes);
  write_le<uint32_t>(bytes, 8, 0x274);
  write_le<uint32_t>(bytes, 12, 1);
  write_le<uint64_t>(bytes, 16, 0x1f8);
  write_le<uint64_t>(bytes, 24, 0x400);
  write_le<uint32_t>(bytes, 32, 2);
  write_le<uint32_t>(bytes, 36, 1);
  write_le<uint32_t>(bytes, 40, 1);
  for (uint32_t channel : {4u, 5u, 6u}) {
    write_le<uint32_t>(bytes, 40 + channel * sizeof(uint32_t), 1);
    write_nchw(bytes, channel);
  }
  return bytes;
}

nlohmann::json tensor(const std::string& name, uint64_t index) {
  return {
      {"name", name},
      {"index", index},
      {"dtype", "float16"},
      {"shape", {1, 64, 1, 1}},
      {"byte_size", 128},
      {"stride", kAllocationBytes},
  };
}

nlohmann::json binding(const std::string& name, uint64_t channel) {
  return {
      {"tensor", name},
      {"channel", channel},
      {"dtype", "float16"},
      {"shape", {1, 64, 1, 1}},
      {"nchw", kNchw},
      {"logical_bytes", 128},
      {"allocation_bytes", kAllocationBytes},
      {"element_offset", 0},
      {"element_count", 64},
      {"physical_elements", 64},
  };
}

nlohmann::json program(
    const std::string& payload,
    const std::string& operation,
    const std::array<std::string, 2>& inputs,
    const std::string& output) {
  return {
      {"payload", payload},
      {"operation", operation},
      {"encoder", "h13-oracle-parity"},
      {"task_descriptors", 1},
      {"scratch_bytes", 0},
      {"inputs", {binding(inputs[0], 5), binding(inputs[1], 6)}},
      {"outputs", {binding(output, 4)}},
  };
}

class TempDir {
 public:
  TempDir() {
    auto base = std::filesystem::temp_directory_path() / "mlx-omarchy-ane-bundle-test";
    std::filesystem::create_directories(base);
    for (int attempt = 0; attempt < 64; ++attempt) {
      auto candidate = base /
          ("case-" + std::to_string(::getpid()) + "-" + std::to_string(attempt));
      std::error_code error;
      if (std::filesystem::create_directory(candidate, error) && !error) {
        path_ = std::move(candidate);
        return;
      }
    }
    throw std::runtime_error("cannot create test directory");
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

void write_file(const std::filesystem::path& path, const std::string& bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  REQUIRE(output.good());
}

std::string payload_collection_identity(const nlohmann::json& payloads) {
  nlohmann::json records = payloads;
  std::sort(records.begin(), records.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.at("path").template get<std::string>() <
        rhs.at("path").template get<std::string>();
  });
  const std::string encoded = records.dump(-1, ' ', true);
  return sha256_hex(
      reinterpret_cast<const uint8_t*>(encoded.data()), encoded.size());
}

struct Fixture {
  TempDir dir;
  std::array<std::string, 2> payload_bytes{anec_bytes('A'), anec_bytes('K')};
  nlohmann::json manifest;

  Fixture() {
    manifest = {
        {"manifest_version", 3},
        {"name", "h13-chain-add-mul"},
        {"graph_hash", hex(64, '1')},
        {"task_descriptors", 2},
        {"inputs", {tensor("a", 0), tensor("b", 1)}},
        {"outputs", {tensor("y", 0)}},
        {"state", nlohmann::json::array()},
        {"intermediates", {tensor("sum", 0)}},
        {"programs",
         {program("program-0.anec", "add", {"a", "b"}, "sum"),
          program("program-1.anec", "mul", {"sum", "b"}, "y")}},
        {"dispatch_plan", {0, 1}},
        {"payloads",
         {{{"role", "anec"},
           {"path", "program-0.anec"},
           {"sha256", digest(0)},
           {"byte_size", payload_bytes[0].size()}},
          {{"role", "anec"},
           {"path", "program-1.anec"},
           {"sha256", digest(1)},
           {"byte_size", payload_bytes[1].size()}}}},
        {"compiler",
         {{"host_build", "Linux test host"},
          {"toolchain", "mil-hwxc test source"},
          {"target", "h13"}}},
        {"driver_abi_major", 1},
        {"provenance",
         {{"source_repo", "joshuaswarren/mlx-omarchy"},
          {"source_commit", hex(40, 'c')},
          {"exported_at", "2026-09-12"}}},
        {"release_asset",
         {{"model", "h13-chain-add-mul"},
          {"model_sha256", ""}}},
    };
    manifest["release_asset"]["model_sha256"] =
        payload_collection_identity(manifest["payloads"]);
  }

  std::string digest(size_t index) const {
    const auto& bytes = payload_bytes.at(index);
    return sha256_hex(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
  }

  void refresh_payload(size_t index) {
    manifest["payloads"][index]["sha256"] = digest(index);
    manifest["payloads"][index]["byte_size"] = payload_bytes[index].size();
    manifest["release_asset"]["model_sha256"] =
        payload_collection_identity(manifest["payloads"]);
  }

  void write() const {
    write_file(dir.path() / "program-0.anec", payload_bytes[0]);
    write_file(dir.path() / "program-1.anec", payload_bytes[1]);
    write_file(dir.path() / "manifest.json", manifest.dump(2) + "\n");
  }
};

template <typename Function>
void check_error(Function&& function, const std::string& expected) {
  try {
    function();
    FAIL("expected exception");
  } catch (const std::exception& error) {
    CHECK(std::string(error.what()).find(expected) != std::string::npos);
  }
}

} // namespace

TEST_CASE("valid multi-program bundle preserves dispatch and bindings") {
  Fixture fixture;
  fixture.write();
  AneBundle bundle = load_bundle(fixture.dir.path());
  REQUIRE(bundle.programs.size() == 2);
  CHECK(bundle.manifest.manifest_version == 3);
  CHECK(bundle.manifest.driver_abi_major == 1);
  CHECK(bundle.manifest.dispatch_plan == std::vector<uint64_t>{0, 1});
  CHECK(bundle.manifest.programs[bundle.programs[0].manifest_index].inputs[0].tensor == "a");
  CHECK(bundle.manifest.programs[bundle.programs[1].manifest_index].inputs[0].tensor == "sum");
  CHECK(bundle.programs[1].anec_header.source_count == 2);
}

TEST_CASE("snapshot loader uses supplied immutable payload paths") {
  Fixture caller;
  caller.write();
  TempDir snapshot;
  write_file(snapshot.path() / "manifest.json", caller.manifest.dump(2) + "\n");
  write_file(snapshot.path() / "program-0.anec", caller.payload_bytes[0]);
  write_file(snapshot.path() / "program-1.anec", caller.payload_bytes[1]);
  std::map<std::string, std::filesystem::path> payloads{
      {"program-0.anec", snapshot.path() / "program-0.anec"},
      {"program-1.anec", snapshot.path() / "program-1.anec"}};

  AneBundle loaded =
      load_bundle_snapshot(snapshot.path() / "manifest.json", payloads);
  write_file(caller.dir.path() / "program-0.anec", anec_bytes('Z'));

  CHECK(sha256_file(loaded.programs[0].anec) == caller.digest(0));
  CHECK(loaded.programs[0].anec.parent_path() == snapshot.path());
}

TEST_CASE("release identity uses the producer's canonical payload byte domain") {
  SUBCASE("mismatch") {
    Fixture fixture;
    fixture.manifest["release_asset"]["model_sha256"] = hex(64, '4');
    fixture.write();
    check_error(
        [&] { load_bundle(fixture.dir.path()); },
        "model_sha256 does not match compiled payload collection");
  }

  SUBCASE("non-ASCII paths use escaped canonical JSON") {
    Fixture fixture;
    const std::string renamed = "prögram-0.anec";
    fixture.manifest["programs"][0]["payload"] = renamed;
    fixture.manifest["payloads"][0]["path"] = renamed;
    fixture.manifest["release_asset"]["model_sha256"] =
        "1f1a7bd31300c3578ebbe0c96a03e52a467d669cefbd23c5fc3993975fcddc90";
    fixture.write();
    std::filesystem::rename(fixture.dir.path() / "program-0.anec",
                            fixture.dir.path() / renamed);
    CHECK_NOTHROW(load_bundle(fixture.dir.path()));
  }
}

TEST_CASE("schema 2 is rejected without compatibility shim") {
  Fixture fixture;
  fixture.manifest["manifest_version"] = 2;
  fixture.write();
  check_error([&] { load_bundle(fixture.dir.path()); }, "unsupported manifest_version");
}

TEST_CASE("removed firmware field is rejected") {
  Fixture fixture;
  fixture.manifest["firmware"] = {{"min", "13.0"}, {"max", "13.5"}};
  fixture.write();
  check_error([&] { load_bundle(fixture.dir.path()); }, "unknown field 'firmware'");
}

TEST_CASE("driver ABI is required, unsigned, and exact") {
  SUBCASE("missing") {
    Fixture fixture;
    fixture.manifest.erase("driver_abi_major");
    fixture.write();
    check_error([&] { load_bundle(fixture.dir.path()); }, "missing field 'driver_abi_major'");
  }
  SUBCASE("signed") {
    Fixture fixture;
    fixture.manifest["driver_abi_major"] = -1;
    fixture.write();
    check_error([&] { load_bundle(fixture.dir.path()); }, "must be a non-negative integer");
  }
  SUBCASE("wrong major") {
    Fixture fixture;
    fixture.manifest["driver_abi_major"] = 2;
    fixture.write();
    check_error([&] { load_bundle(fixture.dir.path()); }, "unsupported driver_abi_major 2");
  }
}

TEST_CASE("dispatch plan must be a complete permutation") {
  Fixture fixture;
  fixture.manifest["dispatch_plan"] = {0, 0};
  fixture.write();
  check_error([&] { load_bundle(fixture.dir.path()); }, "must be a permutation");
}

TEST_CASE("dispatch plan rejects use-before-write") {
  Fixture fixture;
  fixture.manifest["dispatch_plan"] = {1, 0};
  fixture.write();
  check_error([&] { load_bundle(fixture.dir.path()); }, "reads tensor 'sum' before its range is written");
}
TEST_CASE("declared final outputs require complete dispatched range coverage") {
  const std::array<uint64_t, 6> physical_128{1, 128, 1, 1, 64, 64};

  SUBCASE("a gap is rejected") {
    Fixture fixture;
    fixture.manifest["outputs"][0]["shape"] = {1, 128, 1, 1};
    fixture.manifest["outputs"][0]["byte_size"] = 256;
    auto& output = fixture.manifest["programs"][1]["outputs"][0];
    output["nchw"] = physical_128;
    output["physical_elements"] = 128;
    write_nchw(fixture.payload_bytes[1], 4, physical_128);
    fixture.refresh_payload(1);
    fixture.write();
    check_error(
        [&] { load_bundle(fixture.dir.path()); },
        "output tensor 'y' is not fully written");
  }

  SUBCASE("disjoint program tiles can cover the output") {
    Fixture fixture;
    fixture.manifest["outputs"][0]["shape"] = {1, 128, 1, 1};
    fixture.manifest["outputs"][0]["byte_size"] = 256;
    fixture.manifest["intermediates"] = nlohmann::json::array();
    for (size_t p = 0; p < 2; ++p) {
      fixture.manifest["programs"][p]["inputs"][0]["tensor"] = "a";
      auto& output = fixture.manifest["programs"][p]["outputs"][0];
      output["tensor"] = "y";
      output["nchw"] = physical_128;
      output["physical_elements"] = 128;
      output["element_offset"] = p * 64;
      write_nchw(fixture.payload_bytes[p], 4, physical_128);
      fixture.refresh_payload(p);
    }
    fixture.write();
    CHECK_NOTHROW(load_bundle(fixture.dir.path()));
  }
}

TEST_CASE("each ANEC payload must map to one program") {
  Fixture fixture;
  fixture.manifest["programs"][1]["payload"] = "program-0.anec";
  fixture.write();
  check_error([&] { load_bundle(fixture.dir.path()); }, "referenced more than once");
}

TEST_CASE("bindings reject unknown tensors and invalid ranges") {
  SUBCASE("unknown tensor") {
    Fixture fixture;
    fixture.manifest["programs"][0]["inputs"][0]["tensor"] = "missing";
    fixture.write();
    check_error([&] { load_bundle(fixture.dir.path()); }, "references unknown tensor 'missing'");
  }
  SUBCASE("range") {
    Fixture fixture;
    fixture.manifest["programs"][0]["inputs"][0]["element_offset"] = 1;
    fixture.write();
    check_error([&] { load_bundle(fixture.dir.path()); }, "range or allocation exceeds tensor 'a'");
  }
}
TEST_CASE("binding physical geometry is exact while slices may be smaller") {
  SUBCASE("physical elements must equal NCHW elements") {
    for (uint64_t physical : {uint64_t{1024}, uint64_t{20000}}) {
      Fixture fixture;
      fixture.manifest["programs"][0]["inputs"][0]["physical_elements"] = physical;
      fixture.write();
      check_error(
          [&] { load_bundle(fixture.dir.path()); },
          "physical_elements does not match NCHW geometry");
    }
  }

  SUBCASE("a valid slice can be smaller than physical geometry") {
    Fixture fixture;
    const std::array<uint64_t, 6> physical_512{1, 512, 1, 1, 32, 32};
    fixture.manifest["inputs"][0]["shape"] = {1, 1024, 1, 1};
    fixture.manifest["inputs"][0]["byte_size"] = 2048;
    auto& input = fixture.manifest["programs"][0]["inputs"][0];
    input["shape"] = {1, 384, 1, 1};
    input["nchw"] = physical_512;
    input["logical_bytes"] = 768;
    input["element_offset"] = 512;
    input["element_count"] = 384;
    input["physical_elements"] = 512;
    write_nchw(fixture.payload_bytes[0], 5, physical_512);
    fixture.refresh_payload(0);
    fixture.write();
    CHECK_NOTHROW(load_bundle(fixture.dir.path()));
  }
}

TEST_CASE("ordinary tensors cannot use zero geometry") {
  Fixture fixture;
  fixture.manifest["inputs"][0]["shape"] = {0};
  fixture.manifest["inputs"][0]["byte_size"] = 0;
  fixture.write();
  check_error([&] { load_bundle(fixture.dir.path()); }, "must contain positive integers");
}

TEST_CASE("binding channel mapping must match ANEC order") {
  Fixture fixture;
  fixture.manifest["programs"][0]["inputs"][0]["channel"] = 6;
  fixture.manifest["programs"][0]["inputs"][1]["channel"] = 5;
  fixture.write();
  check_error([&] { load_bundle(fixture.dir.path()); }, "channel does not match ANEC binding order");
}

TEST_CASE("binding allocation must match ANEC channel") {
  Fixture fixture;
  fixture.manifest["programs"][0]["inputs"][0]["allocation_bytes"] = 0x8000;
  fixture.manifest["inputs"][0]["stride"] = 0x8000;
  fixture.write();
  check_error([&] { load_bundle(fixture.dir.path()); }, "allocation_bytes does not match ANEC");
}

TEST_CASE("program task count and scratch allocation match each ANEC") {
  SUBCASE("task descriptors") {
    Fixture fixture;
    fixture.manifest["programs"][0]["task_descriptors"] = 2;
    fixture.manifest["task_descriptors"] = 3;
    fixture.write();
    check_error([&] { load_bundle(fixture.dir.path()); }, "task_descriptors does not match ANEC");
  }
  SUBCASE("scratch") {
    Fixture fixture;
    fixture.manifest["programs"][0]["scratch_bytes"] = 1;
    fixture.write();
    check_error([&] { load_bundle(fixture.dir.path()); }, "scratch_bytes does not match ANEC");
  }
  SUBCASE("understated positive scratch") {
    Fixture fixture;
    write_le<uint32_t>(fixture.payload_bytes[0], 40 + 3 * sizeof(uint32_t), 1);
    fixture.refresh_payload(0);
    fixture.manifest["programs"][0]["scratch_bytes"] = 1;
    fixture.write();
    check_error([&] { load_bundle(fixture.dir.path()); }, "scratch_bytes does not match ANEC");
  }
  SUBCASE("exact positive scratch") {
    Fixture fixture;
    write_le<uint32_t>(fixture.payload_bytes[0], 40 + 3 * sizeof(uint32_t), 1);
    fixture.refresh_payload(0);
    fixture.manifest["programs"][0]["scratch_bytes"] = kAllocationBytes;
    fixture.write();
    CHECK_NOTHROW(load_bundle(fixture.dir.path()));
  }
}

TEST_CASE("all payload digests are checked before ANEC parsing") {
  Fixture fixture;
  fixture.manifest["payloads"][1]["sha256"] = hex(64, '0');
  fixture.payload_bytes[0].resize(8);
  fixture.manifest["payloads"][0]["byte_size"] = 8;
  fixture.manifest["payloads"][0]["sha256"] = fixture.digest(0);
  fixture.manifest["release_asset"]["model_sha256"] =
      payload_collection_identity(fixture.manifest["payloads"]);
  fixture.write();
  check_error([&] { load_bundle(fixture.dir.path()); }, "program-1.anec sha256 mismatch");
}

TEST_CASE("ANEC header declares the exact file size") {
  Fixture fixture;
  fixture.payload_bytes[0] += "TRAILER";
  fixture.refresh_payload(0);
  fixture.write();
  check_error(
      [&] { load_bundle(fixture.dir.path()); },
      "ANEC file size does not match payload_size");
}

TEST_CASE("unknown files and missing payloads fail closed") {
  SUBCASE("unknown") {
    Fixture fixture;
    fixture.write();
    write_file(fixture.dir.path() / "extra.bin", "unexpected");
    check_error([&] { load_bundle(fixture.dir.path()); }, "unknown payload file 'extra.bin'");
  }
  SUBCASE("missing") {
    Fixture fixture;
    fixture.write();
    std::filesystem::remove(fixture.dir.path() / "program-1.anec");
    check_error([&] { load_bundle(fixture.dir.path()); }, "payload file missing: program-1.anec");
  }
}

TEST_CASE("directory links fail closed") {
  SUBCASE("listed payload") {
    Fixture fixture;
    fixture.write();
    TempDir outside;
    const auto payload = fixture.dir.path() / "program-0.anec";
    const auto target = outside.path() / "program-0.anec";
    std::filesystem::rename(payload, target);
    std::filesystem::create_symlink(target, payload);
    check_error(
        [&] { load_bundle(fixture.dir.path()); },
        "unexpected link 'program-0.anec'");
  }
  SUBCASE("unlisted entry") {
    Fixture fixture;
    fixture.write();
    std::filesystem::create_symlink(
        fixture.dir.path() / "program-0.anec",
        fixture.dir.path() / "alias.anec");
    check_error(
        [&] { load_bundle(fixture.dir.path()); },
        "unexpected link 'alias.anec'");
  }
}

TEST_CASE("missing bundle is a normal not-found outcome") {
  TempDir parent;
  try {
    load_bundle(parent.path() / "absent");
    FAIL("expected AneBundleNotFound");
  } catch (const AneBundleNotFound& error) {
    CHECK(std::string(error.what()).find("stays on Vulkan") != std::string::npos);
  }
}
