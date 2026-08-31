// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/ane/bundle.h"

#include <array>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
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
  std::error_code size_error;
  for (size_t i = 0; i < manifest.payloads.size(); ++i) {
    const auto& payload = manifest.payloads[i];
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
  return bundle;
}

} // namespace mlx::core::omarchy::ane
