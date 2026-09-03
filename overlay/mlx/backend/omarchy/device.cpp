// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/device.h"

#include "mlx/backend/omarchy/compute.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <memory>
#include <mutex>
#include <stdexcept>

#include "mlx/backend/omarchy/vulkan.h"
#include "mlx/compile.h"

#include "mlx/backend/omarchy/allocator.h"

namespace mlx::core::omarchy {

namespace {

constexpr uint32_t kVulkan13 = VK_API_VERSION_1_3;

int32_t to_lower_ascii(char c) {
  return static_cast<int32_t>(std::tolower(static_cast<unsigned char>(c)));
}

bool contains_case_insensitive(const char* haystack, const char* needle) {
  const size_t nlen = std::strlen(needle);
  const size_t hlen = std::strlen(haystack);
  if (nlen == 0 || nlen > hlen) {
    return false;
  }
  for (size_t i = 0; i + nlen <= hlen; ++i) {
    size_t j = 0;
    while (j < nlen &&
           to_lower_ascii(haystack[i + j]) == to_lower_ascii(needle[j])) {
      ++j;
    }
    if (j == nlen) {
      return true;
    }
  }
  return false;
}

const char* driver_id_name(int32_t id) {
  // Driver-id constants are plain enum values. kMesaHoneykrispDriverId is
  // spelled out in device.h so Vulkan 1.3 headers (which predate the enum
  // entry) compile; unknown ids fall through to the default.
  switch (id) {
    case kMesaHoneykrispDriverId:
      return "Mesa Honeykrisp";
    case VK_DRIVER_ID_MESA_RADV:
      return "Mesa RADV";
    case VK_DRIVER_ID_MESA_LLVMPIPE:
      return "Mesa llvmpipe";
    case VK_DRIVER_ID_NVIDIA_PROPRIETARY:
      return "NVIDIA proprietary";
    case VK_DRIVER_ID_AMD_OPEN_SOURCE:
      return "AMD open source";
    default:
      return nullptr;
  }
}

int env_index(const char* name) {
  const char* v = std::getenv(name);
  if (!v) {
    return -1;
  }
  char* end = nullptr;
  long parsed = std::strtol(v, &end, 10);
  if (end == v || parsed < 0 || parsed > 1024) {
    return -1;
  }
  return static_cast<int>(parsed);
}

struct PhysicalDeviceInfo {
  VkPhysicalDevice handle{VK_NULL_HANDLE};
  DeviceSupport support;
  CapabilityReport caps;
};

// Process-wide Vulkan state. The VkInstance lives as long as the process;
// VkDevices are created lazily per used index.
struct Runtime {
  std::mutex mutex;
  VkInstance instance{VK_NULL_HANDLE};
  std::vector<PhysicalDeviceInfo> supported;
  std::vector<std::unique_ptr<Device>> devices;
  std::string error{"backend not initialized"};
  bool ready{false};
  bool probed{false};
  bool allow_non_apple{false};
  int preferred_device_index{-1};

  // Discover devices once per process. Never throws; failures are recorded
  // in |error| so callers can surface exact reasons.
  bool init() {
    std::lock_guard<std::mutex> lk(mutex);
    if (probed) {
      return ready;
    }
    probed = true;
    try {
      return init_impl();
    } catch (const std::exception& ex) {
      error = ex.what();
      if (instance != VK_NULL_HANDLE) {
        if (auto& it = vk::instance_table(); it.DestroyInstance) {
          it.DestroyInstance(instance, nullptr);
        }
        instance = VK_NULL_HANDLE;
      }
      return false;
    }
  }

  bool init_impl();

  ~Runtime() {
    devices.clear();
    if (instance != VK_NULL_HANDLE) {
      if (auto& it = vk::instance_table(); it.DestroyInstance) {
        it.DestroyInstance(instance, nullptr);
      }
    }
  }
};

Runtime& runtime() {
  static Runtime rt;
  return rt;
}

CapabilityReport collect_capabilities(
    vk::InstanceTable& it,
    VkPhysicalDevice pd,
    const DeviceSupport& support,
    const VkPhysicalDeviceProperties2& props2,
    const VkPhysicalDeviceMemoryProperties2& mem2,
    const VkPhysicalDeviceFeatures2& feats2,
    const VkPhysicalDeviceVulkan12Features& f12,
    const VkPhysicalDeviceVulkan13Features& f13,
    const VkPhysicalDeviceShaderAtomicFloatFeaturesEXT& fa,
    const VkPhysicalDevice16BitStorageFeatures& f16,
    const VkPhysicalDeviceMaintenance3Properties& m3,
    const VkPhysicalDeviceMaintenance4Properties& m4) {
  CapabilityReport caps;
  const auto& props = props2.properties;
  const auto& limits = props.limits;
  caps.device_name = props.deviceName;
  caps.vendor_id = props.vendorID;
  caps.device_id = props.deviceID;
  caps.driver_version = props.driverVersion;
  caps.api_version = props.apiVersion;
  caps.driver_id = support.driver_id;
  if (const char* name = driver_id_name(support.driver_id)) {
    caps.driver_name = name;
  } else {
    caps.driver_name = "driver id " + std::to_string(support.driver_id);
  }
  std::memcpy(
      caps.pipeline_cache_uuid.data(), props.pipelineCacheUUID, VK_UUID_SIZE);

  caps.unified_memory = false;
  size_t device_local = 0;
  for (uint32_t i = 0; i < mem2.memoryProperties.memoryHeapCount; ++i) {
    const auto& heap = mem2.memoryProperties.memoryHeaps[i];
    if (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      device_local = std::max(device_local, static_cast<size_t>(heap.size));
    }
  }
  for (uint32_t i = 0; i < mem2.memoryProperties.memoryTypeCount; ++i) {
    const auto& type = mem2.memoryProperties.memoryTypes[i];
    bool host_visible_coherent =
        (type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
        (type.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (host_visible_coherent) {
      caps.host_visible_coherent = true;
      if (mem2.memoryProperties.memoryHeaps[type.heapIndex].flags &
          VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
        caps.unified_memory = true;
        break;
      }
    }
  }
  caps.total_memory = device_local;

  caps.timeline_semaphore = f12.timelineSemaphore == VK_TRUE;
  caps.shader_float16 = f12.shaderFloat16 == VK_TRUE;
  caps.shader_int16 = feats2.features.shaderInt16 == VK_TRUE;
  caps.storage_buffer_16bit_access = f16.storageBuffer16BitAccess == VK_TRUE;

  caps.max_allocation_size = m3.maxMemoryAllocationSize;
  caps.max_buffer_size = m4.maxBufferSize;
  caps.max_storage_buffer_range = limits.maxStorageBufferRange;
  caps.max_per_stage_descriptor_storage_buffers =
      limits.maxPerStageDescriptorStorageBuffers;
  caps.max_descriptor_set_storage_buffers =
      limits.maxDescriptorSetStorageBuffers;
  caps.max_compute_work_group_invocations =
      limits.maxComputeWorkGroupInvocations;
  caps.max_compute_shared_memory_size = limits.maxComputeSharedMemorySize;
  for (int i = 0; i < 3; ++i) {
    caps.max_compute_work_group_size[i] = limits.maxComputeWorkGroupSize[i];
  }
  caps.timestamp_period = limits.timestampPeriod;

  uint32_t family_count = 0;
  it.GetPhysicalDeviceQueueFamilyProperties(pd, &family_count, nullptr);
  std::vector<VkQueueFamilyProperties> families(family_count);
  it.GetPhysicalDeviceQueueFamilyProperties(pd, &family_count, families.data());
  for (uint32_t f = 0; f < family_count; ++f) {
    if (families[f].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      caps.queue_family_index = f;
      caps.queue_count = families[f].queueCount;
      caps.queue_timestamp_valid_bits = families[f].timestampValidBits;
      caps.queue_timestamp_valid_bits = families[f].timestampValidBits;
      break;
    }
  }
  // The extension must be present AND the buffer float32 atomic-add
  // feature must be enabled; llvmpipe advertises the extension name and
  // clears the feature bit only when the device truly supports it.
  caps.shader_atomic_float_add = false;
  {
    uint32_t ext_count = 0;
    if (it.EnumerateDeviceExtensionProperties &&
        it.EnumerateDeviceExtensionProperties(pd, nullptr, &ext_count,
                                              nullptr) == VK_SUCCESS &&
        ext_count > 0) {
      std::vector<VkExtensionProperties> exts(ext_count);
      if (it.EnumerateDeviceExtensionProperties(
              pd, nullptr, &ext_count, exts.data()) == VK_SUCCESS) {
        for (const auto& e : exts) {
          if (std::strcmp(e.extensionName, "VK_EXT_shader_atomic_float") ==
              0) {
            caps.shader_atomic_float_add =
                fa.shaderBufferFloat32AtomicAdd == VK_TRUE;
            break;
          }
        }
      }
    }
  }
  return caps;
}

bool Runtime::init_impl() {
  allow_non_apple = env_flag("MLX_OMARCHY_ALLOW_NON_APPLE");
  preferred_device_index = env_index("MLX_OMARCHY_DEVICE_INDEX");

  if (!vk::load_loader()) {
    error =
        "[omarchy] Vulkan loader not found (libvulkan.so.1)."
        " Install vulkan-icd-loader or the distribution equivalent.";
    return false;
  }
  auto& it = vk::instance_table();
  it.CreateInstance = reinterpret_cast<PFN_vkCreateInstance>(
      vk::GetInstanceProcAddr(nullptr, "vkCreateInstance"));
  it.EnumerateInstanceVersion =
      reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
          vk::GetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion"));
  if (!it.CreateInstance) {
    error = "[omarchy] Vulkan loader has no vkCreateInstance.";
    return false;
  }
  if (!it.EnumerateInstanceVersion) {
    error = "[omarchy] Vulkan loader predates Vulkan 1.1.";
    return false;
  }

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "mlx-omarchy";
  app.pEngineName = "mlx-omarchy";
  app.apiVersion = kVulkan13;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  VKX_CHECK(it.CreateInstance(&ici, nullptr, &instance));

  it.EnumeratePhysicalDevices =
      reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
          vk::GetInstanceProcAddr(instance, "vkEnumeratePhysicalDevices"));
  it.GetPhysicalDeviceProperties2 =
      reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
          vk::GetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2"));
  it.EnumerateDeviceExtensionProperties =
      reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
          vk::GetInstanceProcAddr(
              instance, "vkEnumerateDeviceExtensionProperties"));
  it.GetPhysicalDeviceFeatures2 =
      reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
          vk::GetInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2"));
  it.GetPhysicalDeviceMemoryProperties2 =
      reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties2>(
          vk::GetInstanceProcAddr(
              instance, "vkGetPhysicalDeviceMemoryProperties2"));
  it.GetPhysicalDeviceQueueFamilyProperties =
      reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
          vk::GetInstanceProcAddr(
              instance, "vkGetPhysicalDeviceQueueFamilyProperties"));
  it.CreateDevice = reinterpret_cast<PFN_vkCreateDevice>(
      vk::GetInstanceProcAddr(instance, "vkCreateDevice"));
  it.DestroyInstance = reinterpret_cast<PFN_vkDestroyInstance>(
      vk::GetInstanceProcAddr(instance, "vkDestroyInstance"));
  if (!it.EnumeratePhysicalDevices || !it.GetPhysicalDeviceProperties2 ||
      !it.GetPhysicalDeviceFeatures2 ||
      !it.GetPhysicalDeviceMemoryProperties2 ||
      !it.GetPhysicalDeviceQueueFamilyProperties || !it.CreateDevice ||
      !it.DestroyInstance || !it.EnumerateDeviceExtensionProperties) {
    error = "[omarchy] Vulkan instance does not expose required 1.3 functions.";
    return false;
  }

  uint32_t count = 0;
  if (it.EnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS ||
      count == 0) {
    error =
        "[omarchy] no Vulkan physical devices found."
        " Is the Mesa Honeykrisp driver installed?";
    return false;
  }
  std::vector<VkPhysicalDevice> pds(count);
  VKX_CHECK(it.EnumeratePhysicalDevices(instance, &count, pds.data()));

  std::string first_refusal;

  for (VkPhysicalDevice pd : pds) {
    VkPhysicalDeviceDriverProperties driver{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceMaintenance3Properties m3{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
    VkPhysicalDeviceMaintenance4Properties m4{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES};
    driver.pNext = &m3;
    m3.pNext = &m4;
    VkPhysicalDeviceProperties2 props2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &driver;
    it.GetPhysicalDeviceProperties2(pd, &props2);

    DeviceSupport support = classify_physical_device(
        props2.properties,
        static_cast<int32_t>(driver.driverID),
        allow_non_apple);

    VkPhysicalDeviceMemoryProperties2 mem2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    it.GetPhysicalDeviceMemoryProperties2(pd, &mem2);

    VkPhysicalDevice16BitStorageFeatures f16{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features f13{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT fa{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
    VkPhysicalDeviceFeatures2 feats2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f16.pNext = &f12;
    f12.pNext = &f13;
    f13.pNext = &fa;
    feats2.pNext = &f16;
    it.GetPhysicalDeviceFeatures2(pd, &feats2);

    if (support.supported && f12.timelineSemaphore != VK_TRUE) {
      support.supported = false;
      support.reason = "[omarchy] device '" +
          std::string(props2.properties.deviceName) +
          "' does not expose timeline semaphores;"
          " the backend requires Vulkan 1.2+ synchronization.";
    }

    if (!support.supported) {
      if (first_refusal.empty()) {
        first_refusal = support.reason;
      }
      continue;
    }

    PhysicalDeviceInfo info;
    info.handle = pd;
    info.support = std::move(support);
    info.caps = collect_capabilities(
        it,
        pd,
        info.support,
        props2,
        mem2,
        feats2,
        f12,
        f13,
        fa,
        f16,
        m3,
        m4);
    if (info.caps.queue_count == 0) {
      if (first_refusal.empty()) {
        first_refusal = "[omarchy] device '" + info.caps.device_name +
            "' exposes no compute queue family.";
      }
      continue;
    }
    supported.push_back(std::move(info));
  }

  if (supported.empty()) {
    error = first_refusal.empty()
        ? "[omarchy] no qualifying Vulkan 1.3 device found."
        : first_refusal;
    it.DestroyInstance(instance, nullptr);
    instance = VK_NULL_HANDLE;
    return false;
  }

  error.clear();
  // Auto-eager on the real Apple GPU target. The tape interpreter has
  // produced silently wrong values on Honeykrisp and the defect is
  // unpinned, so compilation is switched off for the process and the
  // user is told once. Eager computes the same values, only slower, so
  // this trades speed for correctness instead of refusing. The
  // eval_compiled_tape refusal stays as a backstop for any tape that
  // still reaches the interpreter (calling mx.enable_compile() after
  // this point re-arms it), and the override keeps compiled tapes for
  // deliberate investigation. This hook runs at runtime discovery,
  // which precedes every array creation and every compiled call, since
  // both resolve the default device first.
  bool apple_target = false;
  for (const auto& info : supported) {
    apple_target = apple_target || !info.support.non_apple_dev;
  }
  if (apple_target && !env_flag("MLX_OMARCHY_ALLOW_UNSAFE_COMPILE")) {
    disable_compile();
    std::fprintf(
        stderr,
        "[omarchy] Compiled tapes are disabled on this Apple GPU: the tape"
        " interpreter has produced silently wrong values on Honeykrisp and"
        " the defect is unpinned (docs/known-defects.md;"
        " receipts/2026-09-03-dispatcher-compile-and-column-replace.md)."
        " Running eager instead - same values, slower. Set"
        " MLX_OMARCHY_ALLOW_UNSAFE_COMPILE=1 to re-enable compiled tapes"
        " for deliberate investigation.\n");
  }
  // One lazily-filled slot per supported device; device() indexes this
  // vector directly, so it must match |supported| before ready flips.
  devices.resize(supported.size());
  ready = true;
  return true;
}

} // namespace

// Compiled-tape debug switch plumbing (device.h). At omarchy scope, not
// in the anonymous namespace above: the encoder and allocator call these
// through the header declarations.
bool env_flag(const char* name) {
  const char* v = std::getenv(name);
  if (!v) {
    return false;
  }
  std::string s = v;
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s == "1" || s == "on" || s == "true" || s == "yes";
}
// Scoped compiled-tape diagnostic state (device.h). Plain atomics: the
// encoder and the allocator poll these per dispatch or per allocation,
// so the unset path must stay cheaper than an environment lookup.
namespace {
std::atomic<bool> g_tape_full_barriers{false};
std::atomic<bool> g_tape_no_reuse{false};
} // namespace

TapeDebugScope::TapeDebugScope(bool full_barriers, bool no_reuse)
    : full_barriers_(full_barriers), no_reuse_(no_reuse) {
  g_tape_full_barriers.store(full_barriers, std::memory_order_relaxed);
  g_tape_no_reuse.store(no_reuse, std::memory_order_relaxed);
}

TapeDebugScope::~TapeDebugScope() {
  g_tape_full_barriers.store(false, std::memory_order_relaxed);
  g_tape_no_reuse.store(false, std::memory_order_relaxed);
}

bool tape_full_barriers() {
  return g_tape_full_barriers.load(std::memory_order_relaxed);
}

bool tape_no_reuse() {
  return g_tape_no_reuse.load(std::memory_order_relaxed);
}

DeviceSupport classify_physical_device(
    const VkPhysicalDeviceProperties& props,
    int32_t driver_id,
    bool allow_non_apple) {
  DeviceSupport s;
  s.device_name = props.deviceName;
  s.vendor_id = props.vendorID;
  s.device_id = props.deviceID;
  s.api_version = props.apiVersion;
  s.driver_id = driver_id;

  if (props.apiVersion < kVulkan13) {
    s.reason = "[omarchy] device '" + std::string(props.deviceName) +
        "' reports Vulkan " +
        std::to_string(VK_API_VERSION_MAJOR(props.apiVersion)) + "." +
        std::to_string(VK_API_VERSION_MINOR(props.apiVersion)) +
        "; the backend requires Vulkan 1.3.";
    return s;
  }

  // Driver identity is authoritative: the M1 target reports vendor 0x10005
  // and deviceName "Apple M1" under Mesa Honeykrisp, so the driver id is
  // the signal that accepts it. Driver-id constants are C enum values, not
  // macros; never guard them with #ifdef. Apple vendor 0x106b stays as an
  // alternate signal for Apple GPUs on other driver builds.
  bool apple = props.vendorID == kAppleVendorId;
  bool honeykrisp_name =
      contains_case_insensitive(props.deviceName, "honeykrisp");
  bool honeykrisp_driver = driver_id == kMesaHoneykrispDriverId;

  if (apple || honeykrisp_name || honeykrisp_driver) {
    s.supported = true;
    return s;
  }

  if (allow_non_apple) {
    s.supported = true;
    s.non_apple_dev = true;
    return s;
  }

  s.reason = "[omarchy] device '" + std::string(props.deviceName) +
      "' is not an Apple GPU running Omarchy Honeykrisp."
      " Set MLX_OMARCHY_ALLOW_NON_APPLE=1 to use it for development only.";
  return s;
}

// --- Process-wide API -----------------------------------------------------

bool init() {
  return runtime().init();
}

bool is_available() {
  return runtime().init();
}

const std::string& init_error() {
  runtime().init();
  return runtime().error;
}

int device_count() {
  if (!runtime().init()) {
    return 0;
  }
  return static_cast<int>(runtime().supported.size());
}

Device& device(uint32_t index) {
  auto& rt = runtime();
  if (!rt.init()) {
    throw std::runtime_error(rt.error);
  }
  uint32_t effective = index;
  if (effective == 0 && rt.preferred_device_index >= 0) {
    effective = static_cast<uint32_t>(rt.preferred_device_index);
  }
  if (effective >= rt.supported.size()) {
    throw std::invalid_argument(
        "[omarchy] device index " + std::to_string(effective) +
        " is out of range; " + std::to_string(rt.supported.size()) +
        " supported device(s) found.");
  }
  std::lock_guard<std::mutex> lk(rt.mutex);
  auto& slot = rt.devices[effective];
  if (!slot) {
    slot = std::make_unique<Device>(effective);
  }
  return *slot;
}

const CapabilityReport& capability_report(uint32_t index) {
  auto& rt = runtime();
  if (!rt.init()) {
    throw std::runtime_error(rt.error);
  }
  if (index >= rt.supported.size()) {
    throw std::invalid_argument(
        "[omarchy] device index " + std::to_string(index) +
        " is out of range.");
  }
  return rt.supported[index].caps;
}

bool compiled_tapes_refused(const Device& device) {
  if (env_flag("MLX_OMARCHY_ALLOW_UNSAFE_COMPILE")) {
    return false;
  }
  // Real Apple GPU targets corrupt compiled tape values; development
  // devices accepted through MLX_OMARCHY_ALLOW_NON_APPLE do not.
  return !device.non_apple_dev();
}

// --- Device ---------------------------------------------------------------

Device::Device(uint32_t physical_device_index) {
  auto& rt = runtime();
  auto& info = rt.supported.at(physical_device_index);
  caps_ = info.caps;
  non_apple_dev_ = info.support.non_apple_dev;
  VkPhysicalDevice pd = info.handle;
  auto& it = vk::instance_table();

  VkPhysicalDeviceShaderAtomicFloatFeaturesEXT enabled_fa{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
  enabled_fa.shaderBufferFloat32AtomicAdd =
      caps_.shader_atomic_float_add ? VK_TRUE : VK_FALSE;
  VkPhysicalDevice16BitStorageFeatures enabled16{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
  enabled16.storageBuffer16BitAccess =
      caps_.storage_buffer_16bit_access ? VK_TRUE : VK_FALSE;
  VkPhysicalDeviceVulkan12Features enabled12{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  enabled12.timelineSemaphore = VK_TRUE;
  enabled12.shaderFloat16 = caps_.shader_float16 ? VK_TRUE : VK_FALSE;
  VkPhysicalDeviceVulkan13Features enabled13{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  VkPhysicalDeviceFeatures2 enabled2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  enabled2.features.shaderInt16 = caps_.shader_int16 ? VK_TRUE : VK_FALSE;
  enabled16.pNext = &enabled12;
  enabled12.pNext = &enabled13;
  enabled13.pNext = &enabled_fa;
  enabled2.pNext = &enabled16;

  // M1 receipt: Mesa Honeykrisp exposes one compute queue (family 0,
  // queueCount 1). Request exactly that; never invent additional queues.
  float priority = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = caps_.queue_family_index;
  qci.queueCount = 1;
  qci.pQueuePriorities = &priority;

  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.pNext = &enabled2;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  // The float-atomic extension is enabled only when the capability
  // query saw the extension and the buffer-add feature bit; the
  // scatter float Sum/Prod kernels are dispatched only behind that
  // same flag.
  const char* atomic_float_ext = "VK_EXT_shader_atomic_float";
  if (caps_.shader_atomic_float_add) {
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = &atomic_float_ext;
  }
  VKX_CHECK(it.CreateDevice(pd, &dci, nullptr, &device_));

  auto& dt = vk::device_table();
  dt.GetDeviceProcAddr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
      vk::GetInstanceProcAddr(rt.instance, "vkGetDeviceProcAddr"));
  if (!dt.GetDeviceProcAddr) {
    throw std::runtime_error(
        "[omarchy] Vulkan device is missing vkGetDeviceProcAddr.");
  }

  VkPhysicalDeviceMemoryProperties2 mem2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
  it.GetPhysicalDeviceMemoryProperties2(pd, &mem2);
  mem_props_ = mem2.memoryProperties;

#define VKX_LOAD_DEVICE_FN(member, symbol)                                    \
  dt.member =                                                                 \
      reinterpret_cast<PFN_##symbol>(dt.GetDeviceProcAddr(device_, #symbol)); \
  if (dt.member == nullptr) {                                                 \
    throw std::runtime_error(                                                 \
        std::string("[omarchy] Vulkan device is missing ") + #symbol);        \
  }

  VKX_LOAD_DEVICE_FN(DestroyDevice, vkDestroyDevice)
  VKX_LOAD_DEVICE_FN(GetDeviceQueue, vkGetDeviceQueue)
  VKX_LOAD_DEVICE_FN(DeviceWaitIdle, vkDeviceWaitIdle)
  VKX_LOAD_DEVICE_FN(AllocateMemory, vkAllocateMemory)
  VKX_LOAD_DEVICE_FN(FreeMemory, vkFreeMemory)
  VKX_LOAD_DEVICE_FN(MapMemory, vkMapMemory)
  VKX_LOAD_DEVICE_FN(UnmapMemory, vkUnmapMemory)
  VKX_LOAD_DEVICE_FN(CreateBuffer, vkCreateBuffer)
  VKX_LOAD_DEVICE_FN(DestroyBuffer, vkDestroyBuffer)
  VKX_LOAD_DEVICE_FN(GetBufferMemoryRequirements, vkGetBufferMemoryRequirements)
  VKX_LOAD_DEVICE_FN(BindBufferMemory, vkBindBufferMemory)
  VKX_LOAD_DEVICE_FN(CreateCommandPool, vkCreateCommandPool)
  VKX_LOAD_DEVICE_FN(DestroyCommandPool, vkDestroyCommandPool)
  VKX_LOAD_DEVICE_FN(AllocateCommandBuffers, vkAllocateCommandBuffers)
  VKX_LOAD_DEVICE_FN(FreeCommandBuffers, vkFreeCommandBuffers)
  VKX_LOAD_DEVICE_FN(ResetCommandPool, vkResetCommandPool)
  VKX_LOAD_DEVICE_FN(BeginCommandBuffer, vkBeginCommandBuffer)
  VKX_LOAD_DEVICE_FN(EndCommandBuffer, vkEndCommandBuffer)
  VKX_LOAD_DEVICE_FN(ResetCommandBuffer, vkResetCommandBuffer)
  VKX_LOAD_DEVICE_FN(CmdCopyBuffer, vkCmdCopyBuffer)
  VKX_LOAD_DEVICE_FN(CmdFillBuffer, vkCmdFillBuffer)
  VKX_LOAD_DEVICE_FN(CreateShaderModule, vkCreateShaderModule)
  VKX_LOAD_DEVICE_FN(DestroyShaderModule, vkDestroyShaderModule)
  VKX_LOAD_DEVICE_FN(CreateDescriptorSetLayout, vkCreateDescriptorSetLayout)
  VKX_LOAD_DEVICE_FN(DestroyDescriptorSetLayout, vkDestroyDescriptorSetLayout)
  VKX_LOAD_DEVICE_FN(CreateDescriptorPool, vkCreateDescriptorPool)
  VKX_LOAD_DEVICE_FN(DestroyDescriptorPool, vkDestroyDescriptorPool)
  VKX_LOAD_DEVICE_FN(AllocateDescriptorSets, vkAllocateDescriptorSets)
  VKX_LOAD_DEVICE_FN(UpdateDescriptorSets, vkUpdateDescriptorSets)
  VKX_LOAD_DEVICE_FN(CreatePipelineLayout, vkCreatePipelineLayout)
  VKX_LOAD_DEVICE_FN(DestroyPipelineLayout, vkDestroyPipelineLayout)
  VKX_LOAD_DEVICE_FN(CreateComputePipelines, vkCreateComputePipelines)
  VKX_LOAD_DEVICE_FN(DestroyPipeline, vkDestroyPipeline)
  VKX_LOAD_DEVICE_FN(CmdBindPipeline, vkCmdBindPipeline)
  VKX_LOAD_DEVICE_FN(CmdBindDescriptorSets, vkCmdBindDescriptorSets)
  VKX_LOAD_DEVICE_FN(CmdPushConstants, vkCmdPushConstants)
  VKX_LOAD_DEVICE_FN(CmdDispatch, vkCmdDispatch)
  VKX_LOAD_DEVICE_FN(CmdPipelineBarrier, vkCmdPipelineBarrier)
  VKX_LOAD_DEVICE_FN(CreateQueryPool, vkCreateQueryPool)
  VKX_LOAD_DEVICE_FN(GetQueryPoolResults, vkGetQueryPoolResults)
  VKX_LOAD_DEVICE_FN(CmdResetQueryPool, vkCmdResetQueryPool)
  VKX_LOAD_DEVICE_FN(CmdWriteTimestamp, vkCmdWriteTimestamp)
  VKX_LOAD_DEVICE_FN(QueueSubmit, vkQueueSubmit)
  VKX_LOAD_DEVICE_FN(QueueWaitIdle, vkQueueWaitIdle)
  VKX_LOAD_DEVICE_FN(CreateFence, vkCreateFence)
  VKX_LOAD_DEVICE_FN(DestroyFence, vkDestroyFence)
  VKX_LOAD_DEVICE_FN(ResetFences, vkResetFences)
  VKX_LOAD_DEVICE_FN(WaitForFences, vkWaitForFences)
  VKX_LOAD_DEVICE_FN(CreateSemaphore, vkCreateSemaphore)
  VKX_LOAD_DEVICE_FN(DestroySemaphore, vkDestroySemaphore)
  VKX_LOAD_DEVICE_FN(GetSemaphoreCounterValue, vkGetSemaphoreCounterValue)
  VKX_LOAD_DEVICE_FN(WaitSemaphores, vkWaitSemaphores)
  VKX_LOAD_DEVICE_FN(FlushMappedMemoryRanges, vkFlushMappedMemoryRanges)
  VKX_LOAD_DEVICE_FN(
      InvalidateMappedMemoryRanges, vkInvalidateMappedMemoryRanges)

#undef VKX_LOAD_DEVICE_FN

  dt.GetDeviceQueue(device_, caps_.queue_family_index, 0, &queue_);
  // The live binding budget: what the backend wants, clamped by what this
  // physical device actually reports. The spec floor (4) always passes
  // through; larger kernels check compute().binding_limit() at eval time.
  uint32_t binding_limit = std::min(
      kComputeBindingBudget,
      std::min(
          caps_.max_per_stage_descriptor_storage_buffers,
          caps_.max_descriptor_set_storage_buffers));
  compute_ = std::make_unique<ComputeRuntime>(device_, binding_limit);
  completions_ = std::make_unique<CompletionDispatcher>(device_);
}

Device::~Device() {
  if (device_ != VK_NULL_HANDLE) {
    auto& dt = vk::device_table();
    // Order is load-bearing: the queue finishes every submission first
    // (DeviceWaitIdle), then the dispatcher is shut down and drained while
    // the VkDevice and its semaphores are still valid, and only then is
    // the VkDevice destroyed.
    if (dt.DeviceWaitIdle) {
      dt.DeviceWaitIdle(device_);
    }
    if (completions_) {
      completions_->shutdown();
    }
    compute_.reset();
    if (dt.DestroyDevice) {
      dt.DestroyDevice(device_, nullptr);
    }
  }
}

void Device::signal_timeline(VkSemaphore semaphore, uint64_t value) {
  VkTimelineSemaphoreSubmitInfo timeline{
      VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
  timeline.signalSemaphoreValueCount = 1;
  timeline.pSignalSemaphoreValues = &value;
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.pNext = &timeline;
  si.signalSemaphoreCount = 1;
  si.pSignalSemaphores = &semaphore;
  std::lock_guard<std::mutex> lk(queue_mutex_);
  VKX_CHECK(vk::device_table().QueueSubmit(queue_, 1, &si, VK_NULL_HANDLE));
}

void Device::join_completed_handlers() {
  if (!completions_) {
    return;
  }
  uint64_t current = 0;
  if (vk::device_table().GetSemaphoreCounterValue(
          device_, completions_->semaphore(), &current) != VK_SUCCESS ||
      current == 0) {
    return;
  }
  completions_->wait(current);
}

// --- CompletionDispatcher -------------------------------------------------

namespace {

// Bounded dispatcher wait granularity (plan R16): a wedged queue delays
// shutdown and handler dispatch by at most one interval, never forever.
constexpr uint64_t kCompletionPollNs = 100ull * 1000 * 1000;

} // namespace

CompletionDispatcher::CompletionDispatcher(VkDevice device) : device_(device) {
  VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  type.initialValue = 0;
  VkSemaphoreCreateInfo ci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  ci.pNext = &type;
  VKX_CHECK(
      vk::device_table().CreateSemaphore(device_, &ci, nullptr, &semaphore_));
  thread_ = std::thread([this] { run(); });
}

CompletionDispatcher::~CompletionDispatcher() {
  shutdown();
}

void CompletionDispatcher::shutdown() {
  {
    std::lock_guard<std::mutex> lk(mutex_);
    stop_ = true;
  }
  cv_.notify_all();
  if (thread_.joinable()) {
    thread_.join();
  }
  // Static-destruction order can run this drain after the GPU allocator
  // registry is gone. Handlers still run (the scheduler is leaked by
  // design), but remaining buffer temporaries are intentionally leaked:
  // freeing arrays here would call into already-destroyed statics.
  static auto* leaked_temporaries = new std::vector<std::shared_ptr<void>>();
  while (!pending_.empty()) {
    auto completion = std::move(pending_.front());
    pending_.pop_front();
    for (auto& handler : completion.handlers) {
      handler();
    }
    leaked_temporaries->insert(
        leaked_temporaries->end(),
        std::make_move_iterator(completion.temporaries.begin()),
        std::make_move_iterator(completion.temporaries.end()));
  }
  leaked_temporaries->insert(
      leaked_temporaries->end(),
      std::make_move_iterator(retired_temporaries_.begin()),
      std::make_move_iterator(retired_temporaries_.end()));
  retired_temporaries_.clear();
  if (semaphore_ != VK_NULL_HANDLE) {
    vk::device_table().DestroySemaphore(device_, semaphore_, nullptr);
    semaphore_ = VK_NULL_HANDLE;
  }
}

uint64_t CompletionDispatcher::reserve() {
  // Called with the queue mutex held: the value order matches queue
  // execution order, as required for timeline signals.
  std::lock_guard<std::mutex> lk(mutex_);
  return ++next_value_;
}

void CompletionDispatcher::enqueue(
    uint64_t value,
    std::vector<std::shared_ptr<void>> temporaries,
    std::vector<std::function<void()>> handlers) {
  std::lock_guard<std::mutex> lk(mutex_);
  pending_.push_back(
      Completion{value, std::move(temporaries), std::move(handlers)});
  cv_.notify_all();
}

void CompletionDispatcher::wait(uint64_t value) {
  VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  info.semaphoreCount = 1;
  info.pSemaphores = &semaphore_;
  info.pValues = &value;
  VkResult res =
      vk::device_table().WaitSemaphores(device_, &info, kSubmitTimeoutNs);
  if (res != VK_SUCCESS) {
    throw std::runtime_error(
        std::string("[omarchy] Vulkan submission did not complete within ") +
        std::to_string(kSubmitTimeoutNs / 1000000ull) + " ms (" +
        vk::result_string(res) +
        "). The device may be hung; no CPU fallback is available.");
  }
  // Inline fast path: drain and run every ready completion whose value
  // is <= |value| on this thread, serialized end-to-end through
  // drain_mutex_ with the background thread so handlers cannot interleave
  // across separate completion values. With this serialization in place,
  // drained_value_ already covers |value| when drain_through returns and
  // the cv_.wait_until below acts as a defensive join only.
  drain_through(value);
  auto deadline = std::chrono::steady_clock::now() +
      std::chrono::nanoseconds(kSubmitTimeoutNs);
  std::unique_lock<std::mutex> lk(mutex_);
  cv_.wait_until(
      lk, deadline, [this, value] { return stop_ || drained_value_ >= value; });
  if (drained_value_ < value) {
    throw std::runtime_error(
        std::string("[omarchy] Vulkan submission did not complete within ") +
        std::to_string(kSubmitTimeoutNs / 1000000ull) +
        " ms (Timeout). The device may be hung; no CPU fallback is"
        " available.");
  }
}

void CompletionDispatcher::drain_through(uint64_t max_value) {
  std::lock_guard<std::mutex> drain_lk(drain_mutex_);
  std::vector<Completion> ready;
  {
    std::lock_guard<std::mutex> lk(mutex_);
    // reserve() and enqueue() preserve increasing value order.
    while (!pending_.empty() && pending_.front().value <= max_value) {
      ready.push_back(std::move(pending_.front()));
      pending_.pop_front();
    }
  }
  if (ready.empty()) {
    return;
  }
  // Completion boundary: make noncoherent host-visible writes from the
  // GPU visible before handlers and later host reads observe results.
  omarchy::allocator().invalidate_noncoherent(device_);
  uint64_t ready_value = ready.back().value;
  for (auto& completion : ready) {
    for (auto& handler : completion.handlers) {
      handler();
    }
  }
  // Mesa's queue thread signals a submission's semaphores (including the
  // completion timeline read above) BEFORE the submit-final cleanup
  // releases that submission's timeline points, so observing this
  // timeline does not prove the driver finished the submission. Retire
  // payloads one generation late: submits execute serially in value
  // order, so once completion V+1 is observable, cleanup for V has run.
  std::vector<std::shared_ptr<void>> retired;
  for (auto& completion : ready) {
    retired.insert(
        retired.end(),
        std::make_move_iterator(completion.temporaries.begin()),
        std::make_move_iterator(completion.temporaries.end()));
  }
  ready.clear();
  std::vector<std::shared_ptr<void>> release = std::move(retired_temporaries_);
  retired_temporaries_ = std::move(retired);
  std::lock_guard<std::mutex> lk(mutex_);
  drained_value_ = std::max(drained_value_, ready_value);
  cv_.notify_all();
  // |release| frees when this function returns: by then drained_value_
  // names a later generation, so the previous one is provably finished.
}

uint64_t CompletionDispatcher::drained_value() {
  std::lock_guard<std::mutex> lk(mutex_);
  return drained_value_;
}

void CompletionDispatcher::run() {
  // Polling, not vkWaitSemaphores: a host wait issued from this thread
  // while the queue executes the signal stalls some drivers (observed on
  // Mesa 22 lavapipe). GetSemaphoreCounterValue is a plain host query.
  // Explicit wait() drains inline. This thread releases payloads for
  // asynchronous commits that have no later wait.
  for (;;) {
    {
      std::unique_lock<std::mutex> lk(mutex_);
      if (pending_.empty()) {
        if (stop_) {
          return;
        }
        cv_.wait_for(lk, std::chrono::nanoseconds(kCompletionPollNs), [this] {
          return stop_ || !pending_.empty();
        });
        continue;
      }
    }
    uint64_t current = 0;
    if (vk::device_table().GetSemaphoreCounterValue(
            device_, semaphore_, &current) != VK_SUCCESS) {
      if (stop_) {
        return;
      }
      std::this_thread::sleep_for(std::chrono::nanoseconds(kCompletionPollNs));
      continue;
    }
    drain_through(current);

    // Do not busy-spin while the oldest pending submission is still on the
    // GPU. A caller-side drain or shutdown wakes this wait immediately.
    std::unique_lock<std::mutex> lk(mutex_);
    if (stop_) {
      return;
    }
    cv_.wait_for(lk, std::chrono::nanoseconds(kCompletionPollNs), [this] {
      return stop_ || pending_.empty();
    });
  }
}

} // namespace mlx::core::omarchy
