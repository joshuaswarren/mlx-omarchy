// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "mlx/api.h"

namespace mlx::core::omarchy {

// Vendor ID for Apple Silicon GPUs on Omarchy Linux. Honeykrisp is the Mesa
// Vulkan driver that serves these GPUs.
constexpr uint32_t kAppleVendorId = 0x106b;

// VK_DRIVER_ID_MESA_HONEYKRISP spelled out for Vulkan 1.3 headers, which
// predate the enum entry (value verified against the 1.4 registry). Keep in
// sync with vk.xml; the runtime Vulkan >= 1.3 requirement is unchanged.
constexpr int32_t kMesaHoneykrispDriverId = 26;

// Result of the discovery policy applied to one physical device. The logic is
// pure so tests can exercise it without a Vulkan loader or GPU.
struct DeviceSupport {
  bool supported{false};
  // Set when the device was accepted only through the explicit
  // MLX_OMARCHY_ALLOW_NON_APPLE development override.
  bool non_apple_dev{false};
  // Human-readable refusal reason. Empty when supported.
  std::string reason;
  std::string device_name;
  uint32_t vendor_id{0};
  uint32_t device_id{0};
  uint32_t api_version{0};
  int32_t driver_id{-1};
};

// Classify one physical device against the Omarchy support policy:
//   1. Vulkan 1.3 is required (timeline semaphores, sync2).
//   2. Driver identity is authoritative: a Mesa Honeykrisp driverID accepts
//      the device (the M1 target reports vendor 0x10005, name "Apple M1").
//   3. A "honeykrisp" device name or Apple vendor 0x106b are alternate
//      signals.
//   4. Anything else is refused unless allow_non_apple is set, which accepts
//      any Vulkan 1.3 device for software development on non-Omarchy machines.
DeviceSupport classify_physical_device(
    const VkPhysicalDeviceProperties& props,
    int32_t driver_id,
    bool allow_non_apple);

// Physical-device capability facts collected at discovery time. Reported by
// mlx-omarchy-info and stored in hardware receipts.
struct CapabilityReport {
  std::string device_name;
  std::string driver_name;
  uint32_t vendor_id{0};
  uint32_t device_id{0};
  uint32_t driver_version{0};
  uint32_t api_version{0};
  int32_t driver_id{-1};
  std::array<uint8_t, VK_UUID_SIZE> pipeline_cache_uuid{};
  uint32_t queue_family_index{0};
  uint32_t queue_count{0};
  bool unified_memory{false};
  bool timeline_semaphore{false};
  bool shader_float16{false};
  bool shader_int16{false};
  bool storage_buffer_16bit_access{false};
  // True when the device exposes VK_EXT_shader_atomic_float with the
  // shaderBufferFloat32AtomicAdd feature (measured true on llvmpipe and
  // on the M1 G13G B1 Honeykrisp target). Gates the float scatter
  // Sum/Prod kernels, which need OpAtomicFAddEXT on storage buffers.
  bool shader_atomic_float_add{false};
  size_t total_memory{0};
  VkDeviceSize max_allocation_size{0};
  VkDeviceSize max_buffer_size{0};
  VkDeviceSize max_storage_buffer_range{0};
  bool host_visible_coherent{false};
  // Storage-buffer descriptor limits reported by the physical device. These
  // bound the compute binding budget (compute.h kComputeBindingBudget).
  uint32_t max_per_stage_descriptor_storage_buffers{0};
  uint32_t max_descriptor_set_storage_buffers{0};
  uint32_t max_compute_work_group_invocations{0};
  uint32_t max_compute_shared_memory_size{0};
  std::array<uint32_t, 3> max_compute_work_group_size{0, 0, 0};
  float timestamp_period{0.0f};
};

// Bounded wait for any submission or completion (plan R16): a hung
// submission returns control with a typed error instead of blocking
// forever. No CPU fallback exists.
inline constexpr uint64_t kSubmitTimeoutNs = 10ull * 1000 * 1000 * 1000;

class CommandEncoder;
class ComputeRuntime;

// Completion tracking for async submissions on the single Honeykrisp queue.
// Every encoder submission signals one strictly increasing value on a
// device-wide timeline semaphore; a dispatcher thread waits that timeline
// and runs each submission's completion handlers. Buffer temporaries and
// queued-semaphore ownership release one completion generation later,
// because Mesa signals a submission's semaphores before its submit-final
// cleanup retires the submission's timeline points. Submitters never
// block on the queue.
class CompletionDispatcher {
 public:
  explicit CompletionDispatcher(VkDevice device);
  ~CompletionDispatcher();

  CompletionDispatcher(const CompletionDispatcher&) = delete;
  CompletionDispatcher& operator=(const CompletionDispatcher&) = delete;

  VkSemaphore semaphore() const {
    return semaphore_;
  }

  struct Completion {
    uint64_t value;
    std::vector<std::shared_ptr<void>> temporaries;
    std::vector<std::function<void()>> handlers;
  };

  uint64_t reserve();
  void enqueue(
      uint64_t value,
      std::vector<std::shared_ptr<void>> temporaries,
      std::vector<std::function<void()>> handlers);
  void wait(uint64_t value);
  uint64_t drained_value();
  void shutdown();

 private:
  void run();
  void drain_through(uint64_t max_value);

  VkDevice device_;
  VkSemaphore semaphore_{VK_NULL_HANDLE};
  std::deque<Completion> pending_;
  // Payloads of already-drained completions, released one completion
  // later. Mesa signals a submission's semaphores before its submit-final
  // cleanup releases timeline points, so a completion value on this
  // timeline does not prove the driver finished that submission.
  std::vector<std::shared_ptr<void>> retired_temporaries_;
  uint64_t next_value_{0};
  uint64_t drained_value_{0};
  std::mutex mutex_;
  std::mutex drain_mutex_;
  std::condition_variable cv_;
  std::thread thread_;
  bool stop_{false};
};

// A live Vulkan device. Created lazily per supported physical device index.
class Device {
 public:
  explicit Device(uint32_t physical_device_index);
  ~Device();

  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  VkDevice handle() const {
    return device_;
  }

  VkQueue queue() const {
    return queue_;
  }

  uint32_t queue_family() const {
    return caps_.queue_family_index;
  }

  const VkPhysicalDeviceMemoryProperties& memory_properties() const {
    return mem_props_;
  }

  const CapabilityReport& capabilities() const {
    return caps_;
  }

  std::mutex& queue_mutex() {
    return queue_mutex_;
  }

  void join_completed_handlers();

  CompletionDispatcher& completions() {
    return *completions_;
  }

  ComputeRuntime& compute() {
    return *compute_;
  }

  void signal_timeline(VkSemaphore semaphore, uint64_t value);

 private:
  CapabilityReport caps_;
  VkPhysicalDeviceMemoryProperties mem_props_{};
  VkDevice device_{VK_NULL_HANDLE};
  VkQueue queue_{VK_NULL_HANDLE};
  std::mutex queue_mutex_;
  std::unique_ptr<ComputeRuntime> compute_;
  std::unique_ptr<CompletionDispatcher> completions_;
};

// --- Process-wide runtime -------------------------------------------------

// Discover devices and prepare the runtime. Idempotent; never throws. On
// failure the reason is recorded and available through init_error().
bool init();

MLX_API bool is_available();
MLX_API const std::string& init_error();

// Number of devices that pass the support policy.
int device_count();

// Access the live device for a supported index. Initializes on first use and
// throws std::runtime_error with the discovery reason when unavailable.
MLX_API Device& device(uint32_t index = 0);

// Capability facts for a supported index without creating a VkDevice.
// Throws std::runtime_error when the index is out of range or discovery
// failed.
MLX_API const CapabilityReport& capability_report(uint32_t index);

} // namespace mlx::core::omarchy
