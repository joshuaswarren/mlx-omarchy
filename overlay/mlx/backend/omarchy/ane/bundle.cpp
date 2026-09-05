// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/ane/bundle.h"

#include <array>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <vector>

namespace mlx::core::omarchy::ane {
namespace {

std::runtime_error bundle_error(const std::string& reason) {
  return std::runtime_error("[omarchy-ane] bundle: " + reason + ".");
}

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4). Descriptor digests ground in the receipts: the export
// receipt hashes model.hwx with `shasum -a 256`, and every validation JSON
// records hwx_sha256 / anec_sha256 digests.
// ---------------------------------------------------------------------------

struct Sha256Context {
  uint32_t state[8];
  uint8_t buffer[64];
  size_t buffered;
  uint64_t total;
};

uint32_t rotate_right(uint32_t value, uint32_t bits) {
  return (value >> bits) | (value << (32 - bits));
}

void sha256_compress(uint32_t state[8], const uint8_t block[64]) {
  static constexpr uint32_t kK[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
      0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
      0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
      0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
      0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
      0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
      0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
      0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
      0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
      0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
      0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
      0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
      0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
  uint32_t w[64];
  for (int i = 0; i < 16; ++i) {
    w[i] = (uint32_t(block[i * 4]) << 24) | (uint32_t(block[i * 4 + 1]) << 16) |
           (uint32_t(block[i * 4 + 2]) << 8) | uint32_t(block[i * 4 + 3]);
  }
  for (int i = 16; i < 64; ++i) {
    uint32_t s0 = rotate_right(w[i - 15], 7) ^ rotate_right(w[i - 15], 18) ^
                  (w[i - 15] >> 3);
    uint32_t s1 = rotate_right(w[i - 2], 17) ^ rotate_right(w[i - 2], 19) ^
                  (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
  uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
  for (int i = 0; i < 64; ++i) {
    uint32_t s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
    uint32_t ch = (e & f) ^ (~e & g);
    uint32_t t1 = h + s1 + ch + kK[i] + w[i];
    uint32_t s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
    uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    uint32_t t2 = s0 + maj;
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
  state[5] += f;
  state[6] += g;
  state[7] += h;
}

// Appends bytes and compresses full blocks. Does not touch the message
// length: padding bytes go through the same path without being counted.
void sha256_feed(Sha256Context& context, const uint8_t* data, size_t size) {
  while (size > 0) {
    size_t take = 64 - context.buffered;
    if (take > size) {
      take = size;
    }
    for (size_t i = 0; i < take; ++i) {
      context.buffer[context.buffered + i] = data[i];
    }
    context.buffered += take;
    data += take;
    size -= take;
    if (context.buffered == 64) {
      sha256_compress(context.state, context.buffer);
      context.buffered = 0;
    }
  }
}

std::string sha256_digest(const Sha256Context& context) {
  static const char* hex = "0123456789abcdef";
  std::string out;
  out.reserve(64);
  for (uint32_t word : context.state) {
    for (int shift = 28; shift >= 0; shift -= 4) {
      out.push_back(hex[(word >> shift) & 0xf]);
    }
  }
  return out;
}

std::string sha256_pad_and_digest(Sha256Context& context) {
  uint64_t bits = context.total * 8;
  const uint8_t pad = 0x80;
  const uint8_t zero = 0;
  sha256_feed(context, &pad, 1);
  while (context.buffered != 56) {
    sha256_feed(context, &zero, 1);
  }
  uint8_t length[8];
  for (int i = 0; i < 8; ++i) {
    length[i] = uint8_t(bits >> (56 - 8 * i));
  }
  sha256_feed(context, length, 8);
  return sha256_digest(context);
}

Sha256Context sha256_begin() {
  Sha256Context context{};
  context.state[0] = 0x6a09e667u;
  context.state[1] = 0xbb67ae85u;
  context.state[2] = 0x3c6ef372u;
  context.state[3] = 0xa54ff53au;
  context.state[4] = 0x510e527fu;
  context.state[5] = 0x9b05688cu;
  context.state[6] = 0x1f83d9abu;
  context.state[7] = 0x5be0cd19u;
  context.buffered = 0;
  context.total = 0;
  return context;
}

} // namespace

std::string sha256_hex(const uint8_t* data, size_t size) {
  Sha256Context context = sha256_begin();
  context.total += size;
  sha256_feed(context, data, size);
  return sha256_pad_and_digest(context);
}

std::string sha256_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw bundle_error("cannot read payload " + path.string());
  }
  Sha256Context context = sha256_begin();
  std::vector<char> chunk(64 * 1024);
  while (input) {
    input.read(chunk.data(), chunk.size());
    size_t got = size_t(input.gcount());
    if (got == 0) {
      break;
    }
    context.total += got;
    sha256_feed(context, reinterpret_cast<const uint8_t*>(chunk.data()), got);
  }
  if (input.bad()) {
    throw bundle_error("read failed on payload " + path.string());
  }
  return sha256_pad_and_digest(context);
}

namespace {

uint64_t file_size_checked(const std::filesystem::path& path) {
  std::error_code ec;
  const auto size = std::filesystem::file_size(path, ec);
  if (ec) {
    throw bundle_error("cannot stat file " + path.string());
  }
  return static_cast<uint64_t>(size);
}

uint64_t checked_add(uint64_t lhs, uint64_t rhs, const std::string& label) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
    throw bundle_error(label + " overflows uint64");
  }
  return lhs + rhs;
}

uint64_t checked_mul(uint64_t lhs, uint64_t rhs, const std::string& label) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
    throw bundle_error(label + " overflows uint64");
  }
  return lhs * rhs;
}

uint64_t align_up(uint64_t value, uint64_t alignment, const std::string& label) {
  const uint64_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  return checked_add(value, alignment - remainder, label);
}

template <typename T>
T read_le(const std::array<unsigned char, kAnecHeaderSize>& bytes, size_t offset) {
  static_assert(std::is_unsigned_v<T>);
  if (offset > bytes.size() - sizeof(T)) {
    throw bundle_error("ANEC header read escapes header");
  }
  T value = 0;
  for (size_t i = 0; i < sizeof(T); ++i) {
    value |= static_cast<T>(bytes[offset + i]) << (i * 8);
  }
  return value;
}

uint64_t shape_elements(const std::vector<uint64_t>& shape, const std::string& label) {
  uint64_t total = 1;
  for (auto dim : shape) {
    total = checked_mul(total, dim, label);
  }
  return total;
}

uint64_t channel_size_bytes(const AneAnecHeader& header, uint32_t bdx) {
  return checked_mul(header.tiles.at(bdx), kAneTileAlignment, "ANEC channel size");
}

uint32_t output_bdx(uint32_t ordinal) {
  return 4 + ordinal;
}

uint32_t input_bdx(const AneAnecHeader& header, uint32_t ordinal) {
  return 4 + header.destination_count + ordinal;
}

void validate_nchw_geometry(
    const std::string& label,
    const AneTensor& tensor,
    const AneAnecHeader& header,
    uint32_t bdx) {
  const auto& nchw = header.nchw.at(bdx);
  for (size_t i = 0; i < 4; ++i) {
    if (nchw[i] == 0) {
      throw bundle_error(label + " ANEC NCHW geometry is incomplete");
    }
  }
  if (nchw[4] == 0 || nchw[5] == 0) {
    throw bundle_error(label + " ANEC tile geometry is incomplete");
  }

  uint64_t header_elements = 1;
  for (size_t i = 0; i < 4; ++i) {
    header_elements = checked_mul(header_elements, nchw[i], label + " ANEC NCHW");
  }
  if (header_elements != shape_elements(tensor.shape, label)) {
    throw bundle_error(label + " ANEC NCHW does not match manifest shape");
  }
  if (tensor.dtype != "float16" && tensor.dtype != "bfloat16") {
    throw bundle_error(label + " ANEC tiled channel requires a 16-bit tensor dtype");
  }
  constexpr uint64_t element_bytes = sizeof(uint16_t);
  if (nchw[4] % nchw[5] != 0 || nchw[5] % element_bytes != 0) {
    throw bundle_error(label + " ANEC tile geometry is not 16-bit aligned");
  }
  if (nchw[4] / nchw[5] < nchw[2] || nchw[5] / element_bytes < nchw[3]) {
    throw bundle_error(label + " ANEC packed tile is smaller than logical shape");
  }

  const uint64_t physical_bytes = checked_mul(
      checked_mul(nchw[0], nchw[1], label + " ANEC physical bytes"),
      nchw[4],
      label + " ANEC physical bytes");
  if (physical_bytes > channel_size_bytes(header, bdx)) {
    throw bundle_error(label + " ANEC physical tile bytes exceed channel allocation");
  }
}

void validate_channel_contract(
    const std::string& label,
    const AneTensor& tensor,
    const AneAnecHeader& header,
    uint32_t bdx) {
  if (bdx >= kAnecTileCount) {
    throw bundle_error(label + " ANEC channel index is out of range");
  }
  const auto channel_bytes = channel_size_bytes(header, bdx);
  if (channel_bytes == 0) {
    throw bundle_error(label + " ANEC channel allocation is zero");
  }
  if (tensor.byte_size > channel_bytes) {
    throw bundle_error(label + " byte_size exceeds ANEC channel allocation");
  }
  if (tensor.stride > channel_bytes) {
    throw bundle_error(label + " stride exceeds ANEC channel allocation");
  }
  validate_nchw_geometry(label, tensor, header, bdx);
}

void validate_manifest_anec_contract(const AneManifest& manifest, const AneAnecHeader& header) {
  if (manifest.task_descriptors != header.task_descriptor_count) {
    throw bundle_error("ANEC task_descriptors does not match manifest");
  }

  const auto expected_sources = static_cast<uint32_t>(manifest.inputs.size() + manifest.state.size());
  const auto expected_destinations = static_cast<uint32_t>(manifest.outputs.size() + manifest.state.size());
  if (header.source_count != expected_sources) {
    throw bundle_error("ANEC source_count does not match manifest inputs plus state");
  }
  if (header.destination_count != expected_destinations) {
    throw bundle_error("ANEC destination_count does not match manifest outputs plus state");
  }

  for (uint32_t i = 0; i < manifest.outputs.size(); ++i) {
    validate_channel_contract(
        "output " + manifest.outputs[i].name, manifest.outputs[i], header, output_bdx(i));
  }
  for (uint32_t i = 0; i < manifest.state.size(); ++i) {
    validate_channel_contract(
        "state destination " + manifest.state[i].name,
        manifest.state[i],
        header,
        output_bdx(static_cast<uint32_t>(manifest.outputs.size() + i)));
  }
  for (uint32_t i = 0; i < manifest.inputs.size(); ++i) {
    validate_channel_contract(
        "input " + manifest.inputs[i].name, manifest.inputs[i], header, input_bdx(header, i));
  }
  for (uint32_t i = 0; i < manifest.state.size(); ++i) {
    validate_channel_contract(
        "state source " + manifest.state[i].name,
        manifest.state[i],
        header,
        input_bdx(header, static_cast<uint32_t>(manifest.inputs.size() + i)));
  }
}

} // namespace

AneAnecHeader parse_anec_header(const std::filesystem::path& path) {
  const auto file_size = file_size_checked(path);
  if (file_size < kAnecHeaderSize) {
    throw bundle_error("ANEC file is smaller than libane header");
  }

  std::array<unsigned char, kAnecHeaderSize> bytes{};
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw bundle_error("cannot read payload " + path.string());
  }
  in.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  if (in.gcount() != static_cast<std::streamsize>(bytes.size())) {
    throw bundle_error("cannot read ANEC header " + path.string());
  }

  AneAnecHeader header;
  header.payload_size = read_le<uint64_t>(bytes, 0);
  header.task_descriptor_size = read_le<uint32_t>(bytes, 8);
  header.task_descriptor_count = read_le<uint32_t>(bytes, 12);
  header.task_size = read_le<uint64_t>(bytes, 16);
  header.kernel_size = read_le<uint64_t>(bytes, 24);
  header.source_count = read_le<uint32_t>(bytes, 32);
  header.destination_count = read_le<uint32_t>(bytes, 36);

  size_t offset = 40;
  for (auto& tile : header.tiles) {
    tile = read_le<uint32_t>(bytes, offset);
    offset += sizeof(uint32_t);
  }
  for (auto& dims : header.nchw) {
    for (auto& dim : dims) {
      dim = read_le<uint64_t>(bytes, offset);
      offset += sizeof(uint64_t);
    }
  }

  if (header.payload_size == 0) {
    throw bundle_error("ANEC payload size is zero");
  }
  if (header.task_descriptor_size == 0 || header.task_descriptor_count == 0) {
    throw bundle_error("ANEC task descriptor table is empty");
  }
  if (header.task_descriptor_size % sizeof(uint32_t) != 0) {
    throw bundle_error("ANEC task descriptor size is not a multiple of 4");
  }
  if (header.task_size == 0) {
    throw bundle_error("ANEC task size is zero");
  }
  if (uint64_t{4} + header.destination_count + header.source_count >
      kAnecTileCount) {
    throw bundle_error("ANEC source/destination count exceeds libane channel table");
  }
  if (checked_add(kAnecPayloadOffset, header.payload_size, "ANEC payload end") > file_size) {
    throw bundle_error("ANEC executable payload extends past file");
  }
  if (header.tiles[0] == 0) {
    throw bundle_error("ANEC command channel allocation is zero");
  }
  if (header.tiles[1] != 0) {
    throw bundle_error("ANEC reserved kernel channel allocation must be zero");
  }

  const uint64_t command_channel_size = channel_size_bytes(header, 0);
  if (header.payload_size > command_channel_size) {
    throw bundle_error("ANEC executable payload exceeds command channel allocation");
  }
  if (header.task_descriptor_size > header.payload_size) {
    throw bundle_error("ANEC task descriptor bytes exceed executable payload");
  }
  if (header.task_size >= command_channel_size) {
    throw bundle_error("ANEC task size reaches command channel end");
  }

  const uint64_t kernel_offset = align_up(header.task_size, 16, "ANEC kernel offset");
  if (kernel_offset > header.payload_size ||
      header.kernel_size > header.payload_size - kernel_offset) {
    throw bundle_error("ANEC task plus kernel bytes exceed executable payload");
  }
  header.bootstrap_channel_size = align_up(
      header.task_descriptor_size, kAneTileAlignment, "ANEC bootstrap channel");
  return header;
}

AneBundle load_bundle(const std::filesystem::path& dir) {
  if (!std::filesystem::is_directory(dir)) {
    throw AneBundleNotFound(
        "[omarchy-ane] bundle directory not found: " + dir.string() +
        " (the affected region stays on Vulkan)");
  }

  // Manifest validation first: every field check passes before any payload
  // is opened or mapped. A changed graph, shape, compiler, or firmware field
  // fails here, before device access.
  AneManifest manifest = parse_ane_manifest(dir / "manifest.json");

  // Unknown extra files are rejected so a bundle cannot smuggle payloads the
  // manifest does not describe. The scan reads no payload bytes.
  std::vector<std::filesystem::path> actual_files;
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (entry.is_directory()) {
      throw bundle_error(
          "unexpected directory '" + entry.path().filename().string() +
          "' inside bundle");
    }
    if (!entry.is_regular_file()) {
      throw bundle_error(
          "unexpected non-regular file '" + entry.path().filename().string() +
          "' inside bundle");
    }
    actual_files.push_back(entry.path());
  }
  for (const auto& file : actual_files) {
    std::string name = file.filename().string();
    if (name == "manifest.json") {
      continue;
    }
    bool listed = false;
    for (const auto& payload : manifest.payloads) {
      if (payload.path == name) {
        listed = true;
        break;
      }
    }
    if (!listed) {
      throw bundle_error("unknown payload file '" + name + "' not listed in manifest");
    }
  }

  // Presence, then byte size, then digest: cheapest failure first.
  std::vector<std::filesystem::path> resolved;
  resolved.reserve(manifest.payloads.size());
  for (const auto& payload : manifest.payloads) {
    std::filesystem::path payload_path = dir / payload.path;
    if (!std::filesystem::is_regular_file(payload_path)) {
      throw bundle_error("payload file missing: " + payload.path);
    }
    resolved.push_back(payload_path);
  }
  for (size_t i = 0; i < manifest.payloads.size(); ++i) {
    const auto& payload = manifest.payloads[i];
    std::error_code size_error;
    uint64_t actual = uint64_t(std::filesystem::file_size(resolved[i], size_error));
    if (size_error) {
      throw bundle_error("cannot stat payload " + payload.path);
    }
    if (actual != payload.byte_size) {
      throw bundle_error(
          "payload " + payload.path + " byte size " + std::to_string(actual) +
          " does not match manifest " + std::to_string(payload.byte_size));
    }
    std::string digest = sha256_file(resolved[i]);
    if (digest != payload.sha256) {
      throw bundle_error(
          "payload " + payload.path + " sha256 mismatch: manifest " +
          payload.sha256 + " actual " + digest);
    }
  }

  AneBundle bundle;
  bundle.manifest = std::move(manifest);
  for (size_t i = 0; i < bundle.manifest.payloads.size(); ++i) {
    if (bundle.manifest.payloads[i].role == "anec") {
      bundle.anec = resolved[i];
    } else if (bundle.manifest.payloads[i].role == "weights") {
      bundle.weights = resolved[i];
    }
  }
  bundle.anec_header = parse_anec_header(bundle.anec);
  validate_manifest_anec_contract(bundle.manifest, bundle.anec_header);
  return bundle;
}

} // namespace mlx::core::omarchy::ane
