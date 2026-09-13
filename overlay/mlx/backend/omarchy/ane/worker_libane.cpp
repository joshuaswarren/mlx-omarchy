// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// libane-backed device for the bounded ANE worker. Compiled only when
// the build is configured with MLX_OMARCHY_ANE_DEVICE=ON and points at
// an omarchy-ane checkout. Host builds compile nothing from this file:
// the mlx library never links libane, the symbols are dlopen'd from the
// library path the caller names, and no inference-server surface exists
// (plan section 25).

#ifdef MLX_OMARCHY_ANE_DEVICE

#include "mlx/backend/omarchy/ane/worker.h"

#include <dlfcn.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" {
#include "ane.h"
}

namespace mlx::core::omarchy::ane {

namespace {

struct LibaneApi {
  struct ane_nn* (*init)(const char* path, int dev_id);
  void (*free)(struct ane_nn* nn);
  int (*exec)(struct ane_nn* nn);
  uint64_t (*src_size)(struct ane_nn* nn, uint32_t idx);
  uint64_t (*dst_size)(struct ane_nn* nn, uint32_t idx);
  void (*send)(struct ane_nn* nn, void* from, uint32_t idx);
  void (*read)(struct ane_nn* nn, void* to, uint32_t idx);
  void (*tile)(
      void* data, void* tile, uint64_t n, uint64_t c, uint64_t h,
      uint64_t w, uint64_t p, uint64_t r);
  void (*untile)(
      void* data, void* tile, uint64_t n, uint64_t c, uint64_t h,
      uint64_t w, uint64_t p, uint64_t r);

  bool ok() const {
    return init && free && exec && src_size && dst_size && send && read &&
        tile && untile;
  }
};

LibaneApi load_api(void* handle) {
  LibaneApi api{};
  if (!handle) return api;
  api.init = reinterpret_cast<decltype(api.init)>(dlsym(handle, "__ane_init"));
  api.free = reinterpret_cast<decltype(api.free)>(dlsym(handle, "__ane_free"));
  api.exec = reinterpret_cast<decltype(api.exec)>(dlsym(handle, "ane_exec"));
  api.src_size = reinterpret_cast<decltype(api.src_size)>(
      dlsym(handle, "__ane_src_size"));
  api.dst_size = reinterpret_cast<decltype(api.dst_size)>(
      dlsym(handle, "__ane_dst_size"));
  api.send = reinterpret_cast<decltype(api.send)>(dlsym(handle, "__ane_send"));
  api.read = reinterpret_cast<decltype(api.read)>(dlsym(handle, "__ane_read"));
  api.tile = reinterpret_cast<decltype(api.tile)>(dlsym(handle, "ane_tile"));
  api.untile =
      reinterpret_cast<decltype(api.untile)>(dlsym(handle, "ane_untile"));
  return api;
}

class LibaneDevice : public AneDevice {
 public:
  explicit LibaneDevice(std::string library)
      : library_(std::move(library)) {
    handle_ = ::dlopen(library_.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle_) {
      const char* reason = ::dlerror();
      throw AneDeviceError(
          "cannot load libane from " + library_ + ": " +
          (reason ? reason : "unknown dlopen failure"));
    }
    api_ = load_api(handle_);
    if (!api_.ok()) {
      ::dlclose(handle_);
      handle_ = nullptr;
      throw AneDeviceError("libane is missing required symbols");
    }
  }

  ~LibaneDevice() override {
    release();
    if (handle_) {
      ::dlclose(handle_);
    }
  }

  std::string describe() const override {
    return "libane(" + library_ + ")";
  }

  void load(const AneValidatedProgram& program) override {
    if (networks_.count(program.manifest_index)) {
      throw AneDeviceError("program already loaded");
    }
    struct ane_nn* nn = api_.init(program.anec.c_str(), 0);
    if (!nn) {
      throw AneDeviceError(
          "ane_init failed for " + program.anec.string());
    }
    networks_.emplace(program.manifest_index, nn);
  }

  void send(
      size_t manifest_index,
      uint32_t channel,
      const AneProgramBinding& binding,
      const uint8_t* data,
      size_t size) override {
    struct ane_nn* nn = network(manifest_index, "send");
    // libane addresses sources by their sequential position among the
    // anec's inputs; the worker passes the manifest binding whose
    // channel order matches (ane.h: ane_send(nn, input, 0), (nn, input, 1)).
    (void)channel;
    std::vector<uint8_t> tile(binding.allocation_bytes, 0);
    stage(data, size, tile.data(), binding, true);
    api_.send(nn, tile.data(), channel);
  }

  void exec(size_t manifest_index) override {
    struct ane_nn* nn = network(manifest_index, "exec");
    if (api_.exec(nn) < 0) {
      throw AneDeviceError(
          "ane_exec failed for program " + std::to_string(manifest_index));
    }
  }

  void read(
      size_t manifest_index,
      uint32_t channel,
      const AneProgramBinding& binding,
      uint8_t* out,
      size_t size) override {
    struct ane_nn* nn = network(manifest_index, "read");
    (void)channel;
    std::vector<uint8_t> tile(binding.allocation_bytes, 0);
    api_.read(nn, tile.data(), channel);
    stage(tile.data(), tile.size(), out, binding, false);
    (void)size;
  }

  void release() override {
    for (auto& entry : networks_) {
      api_.free(entry.second);
    }
    networks_.clear();
  }

 private:
  struct ane_nn* network(size_t manifest_index, const char* what) const {
    auto found = networks_.find(manifest_index);
    if (found == networks_.end()) {
      throw AneDeviceError(
          std::string(what) + ": program " +
          std::to_string(manifest_index) + " is not loaded");
    }
    return found->second;
  }

  // Copies between logical bytes and the device tile using the binding's
  // NCHW[6] geometry (ane_tile/ane_untile in libane).
  void stage(
      const uint8_t* from,
      size_t logical_size,
      uint8_t* to,
      const AneProgramBinding& binding,
      bool to_tile) {
    if (binding.nchw[0] == 0) {
      std::memcpy(to, from, std::min(logical_size, binding.allocation_bytes));
      return;
    }
    if (to_tile) {
      api_.tile(
          const_cast<uint8_t*>(from), to, binding.nchw[0], binding.nchw[1],
          binding.nchw[2], binding.nchw[3], binding.nchw[4],
          binding.nchw[5]);
    } else {
      api_.untile(
          to, const_cast<uint8_t*>(from), binding.nchw[0], binding.nchw[1],
          binding.nchw[2], binding.nchw[3], binding.nchw[4],
          binding.nchw[5]);
    }
  }

  std::string library_;
  void* handle_{nullptr};
  LibaneApi api_{};
  std::unordered_map<size_t, struct ane_nn*> networks_;
};

} // namespace

std::unique_ptr<AneDevice> make_libane_device(const std::string& library) {
  return std::make_unique<LibaneDevice>(library);
}

} // namespace mlx::core::omarchy::ane

#endif // MLX_OMARCHY_ANE_DEVICE
