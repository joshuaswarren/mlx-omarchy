// Dense BF16 prefill matmul bench for the M1 (Honeykrisp fork) and
// correctness screening elsewhere. It times the REAL production shader
// (shaders/mm_bf16_base.comp, a frozen copy of
// overlay/mlx/backend/omarchy/shaders/matmul_coopmat_bf16.comp), a
// candidate restructure (shaders/mm_bf16_cand.comp), and the 16x16
// scalar tile the m < 32 gate falls back to (shaders/mm_tile_base.comp,
// a frozen copy of shaders/matmul.comp compiled with -DUSE_BF16=1) on
// the per-layer projection shapes of the Qwen2.5-0.5B BF16 prefill:
//   q/o    k=896  n=896
//   k/v    k=896  n=128
//   gate/up k=896  n=4864
//   down   k=4864 n=896
// at matrix_m in {30, 262, 1053}, flags=1 (transposed weights), the
// push-constant and binding contract of dispatch_matmul
// (primitives.cpp): four bindings a/b/c/out, alpha=1, beta=0, no bias.
//
// Measurements per (m, shape):
//   * bit-exactness: identical integer-valued bf16 inputs (exact in
//     every accumulation order) through every pair of kernels into
//     distinct outputs, compared as raw storage bits - must be 0
//     mismatches.
//   * the same with fractional bf16 inputs, plus a float64 host
//     reference at m=30: both kernels must sit within the documented
//     anchor bound of truth.
//   * wall-clock timing of R alternating base/cand/tile dispatches in
//     one command buffer, three rounds, per-kernel medians. Wall clock
//     over R=50 dispatches of 50-500 us each keeps submission overhead
//     in the noise and works on devices without timestamps.
//
// Devices without VK_KHR_cooperative_matrix (llvmpipe, stock Mesa) run
// the tile legs and the capability probe, then stop: the coopmat
// kernels cannot dispatch there, which is exactly why their validation
// is a hardware leg.
//
// Build: g++ -std=c++17 -O2 -o /tmp/bf16-prefill-bench tools/bf16-prefill-bench/bench.cpp
// Run (repo root): /tmp/bf16-prefill-bench [--quick]

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

#define LIBVK "libvulkan.so.1"
// KHR cooperative matrix landed in Vulkan headers after 1.3.239; the
// struct is layout-stable and reuses the NV extension's type value, so
// declare it when the installed header lacks it.
#ifndef VK_KHR_cooperative_matrix
#define VK_KHR_cooperative_matrix 1
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR \
  ((VkStructureType)1000249000)
typedef struct VkPhysicalDeviceCooperativeMatrixFeaturesKHR {
  VkStructureType sType;
  void* pNext;
  VkBool32 cooperativeMatrix;
} VkPhysicalDeviceCooperativeMatrixFeaturesKHR;
#endif


struct VkTable {
  void* handle{nullptr};
  VkInstance inst{VK_NULL_HANDLE};
  VkDevice dev{VK_NULL_HANDLE};
#define VK_FN(name) PFN_vk##name name{nullptr}
  VK_FN(GetInstanceProcAddr);
  VK_FN(GetDeviceProcAddr);
  VK_FN(CreateInstance);
  VK_FN(EnumerateDeviceExtensionProperties);
  VK_FN(EnumeratePhysicalDevices);
  VK_FN(GetPhysicalDeviceProperties);
  VK_FN(GetPhysicalDeviceProperties2);
  VK_FN(GetPhysicalDeviceFeatures2);
  VK_FN(GetPhysicalDeviceMemoryProperties);
  VK_FN(GetPhysicalDeviceQueueFamilyProperties);
  VK_FN(CreateDevice);
  VK_FN(GetDeviceQueue);
  VK_FN(CreateBuffer);
  VK_FN(GetBufferMemoryRequirements);
  VK_FN(BindBufferMemory);
  VK_FN(MapMemory);
  VK_FN(UnmapMemory);
  VK_FN(AllocateMemory);
  VK_FN(FreeMemory);
  VK_FN(CreateShaderModule);
  VK_FN(CreateComputePipelines);
  VK_FN(CreatePipelineLayout);
  VK_FN(CreateDescriptorSetLayout);
  VK_FN(CreateDescriptorPool);
  VK_FN(AllocateDescriptorSets);
  VK_FN(UpdateDescriptorSets);
  VK_FN(CreateCommandPool);
  VK_FN(AllocateCommandBuffers);
  VK_FN(BeginCommandBuffer);
  VK_FN(EndCommandBuffer);
  VK_FN(CmdBindPipeline);
  VK_FN(CmdBindDescriptorSets);
  VK_FN(CmdDispatch);
  VK_FN(CmdPushConstants);
  VK_FN(CreateFence);
  VK_FN(WaitForFences);
  VK_FN(ResetFences);
  VK_FN(QueueSubmit);
  VK_FN(DestroyShaderModule);
  VK_FN(DestroyPipeline);
  VK_FN(DestroyPipelineLayout);
  VK_FN(DestroyDescriptorSetLayout);
  VK_FN(DestroyDescriptorPool);
  VK_FN(DestroyCommandPool);
  VK_FN(DestroyBuffer);
  VK_FN(DestroyDevice);
  VK_FN(DestroyInstance);
#undef VK_FN
};

static VkTable g_vk;

static PFN_vkVoidFunction vk_load(VkInstance h, const char* name) {
  return ((PFN_vkGetInstanceProcAddr)g_vk.GetInstanceProcAddr)(h, name);
}
static PFN_vkVoidFunction vk_dev_load(VkDevice d, const char* name) {
  return ((PFN_vkGetDeviceProcAddr)g_vk.GetDeviceProcAddr)(d, name);
}

#define LOAD(name) g_vk.name = (decltype(g_vk.name))vk_load(g_vk.inst, "vk" #name)
#define LOAD_DEV(name) \
  g_vk.name = (decltype(g_vk.name))vk_dev_load(g_vk.dev, "vk" #name)

[[noreturn]] static void die(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::vfprintf(stderr, fmt, ap);
  va_end(ap);
  std::fputc('\n', stderr);
  std::exit(1);
}

static std::string read_file(const char* path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) die("open %s", path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

static int compile_shader(const char* src, const char* defines,
    const char* out_spv) {
  const char* fronts[] = {
      "glslangValidator -V --target-env vulkan1.3 ",
      "glslc -fshader-stage=compute --target-env=vulkan1.3 ",
  };
  for (const char* front : fronts) {
    std::string cmd = front;
    cmd += defines;
    cmd += " ";
    cmd += src;
    cmd += " -o ";
    cmd += out_spv;
    cmd += " 2>&1";
    if (system(cmd.c_str()) == 0) {
      struct stat st;
      if (stat(out_spv, &st) == 0 && st.st_size != 0) return 0;
    }
    std::fprintf(stderr, "shader compile failed: %s\n", cmd.c_str());
  }
  return -1;
}

struct Buf {
  VkBuffer buf{VK_NULL_HANDLE};
  VkDeviceMemory mem{VK_NULL_HANDLE};
  VkDeviceSize size{0};
  void* mapped{nullptr};
};

static uint32_t find_memtype(uint32_t bits, VkMemoryPropertyFlags want,
    const VkPhysicalDeviceMemoryProperties& mp) {
  for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
    if ((bits & (1u << i)) &&
        (mp.memoryTypes[i].propertyFlags & want) == want) {
      return i;
    }
  }
  return UINT32_MAX;
}

static Buf make_buf(VkDevice dev, const VkPhysicalDeviceMemoryProperties& mp,
    VkDeviceSize size) {
  Buf b;
  b.size = size;
  VkBufferCreateInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bi.size = size;
  bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (g_vk.CreateBuffer(dev, &bi, nullptr, &b.buf) != VK_SUCCESS)
    die("CreateBuffer");
  VkMemoryRequirements req;
  g_vk.GetBufferMemoryRequirements(dev, b.buf, &req);
  uint32_t mt = find_memtype(req.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      mp);
  if (mt == UINT32_MAX) die("no host-visible memtype");
  VkMemoryAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = mt;
  if (g_vk.AllocateMemory(dev, &ai, nullptr, &b.mem) != VK_SUCCESS)
    die("AllocateMemory");
  if (g_vk.BindBufferMemory(dev, b.buf, b.mem, 0) != VK_SUCCESS)
    die("BindBufferMemory");
  if (g_vk.MapMemory(dev, b.mem, 0, size, 0, &b.mapped) != VK_SUCCESS)
    die("MapMemory");
  return b;
}

// Push constants: must byte-match the matmul shader Params block
// (32 x 4-byte words, scalar alignment throughout).
struct Params {
  uint32_t count;
  uint32_t operation;
  uint32_t lhs_size;
  uint32_t rhs_size;
  uint32_t reduce_size;
  uint32_t output_size;
  uint32_t lhs_offset;
  uint32_t rhs_offset;
  uint32_t output_offset;
  uint32_t aux_size;
  uint32_t aux_offset;
  uint32_t matrix_m;
  uint32_t matrix_n;
  uint32_t matrix_k;
  uint32_t flags;
  float alpha;
  float beta;
  uint32_t dims;
  uint32_t shape[4];
  uint32_t in_strides[4];
  uint32_t out_strides[4];
  uint32_t lhs_gap;
  uint32_t rhs_gap;
};
static_assert(sizeof(Params) == 128, "push constant block must be 128 bytes");

static constexpr uint32_t kBindings = 4;

struct Shape {
  const char* name;
  uint32_t k;
  uint32_t n;
};

static const Shape kShapes[] = {
    {"q_o_896x896", 896, 896},
    {"kv_896x128", 896, 128},
    {"gate_up_896x4864", 896, 4864},
    {"down_4864x896", 4864, 896},
};
static const uint32_t kMs[] = {30, 262, 1053};

struct DeviceCtx {
  VkQueue queue{VK_NULL_HANDLE};
  uint32_t qfi{0};
  VkPhysicalDeviceMemoryProperties mp{};
  std::string name;
  uint32_t subgroupSize{0};
  bool coopmat{false};
};

static DeviceCtx setup_device() {
  VkApplicationInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  ai.pApplicationName = "bf16-prefill-bench";
  ai.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo ici{};
  ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ici.pApplicationInfo = &ai;
  if (g_vk.CreateInstance(&ici, nullptr, &g_vk.inst) != VK_SUCCESS)
    die("CreateInstance");
  LOAD(DestroyInstance);
  LOAD(EnumeratePhysicalDevices);
  LOAD(EnumerateDeviceExtensionProperties);
  LOAD(GetPhysicalDeviceFeatures2);
  LOAD(GetPhysicalDeviceProperties);
  LOAD(GetPhysicalDeviceProperties2);
  LOAD(GetPhysicalDeviceMemoryProperties);
  LOAD(GetPhysicalDeviceQueueFamilyProperties);
  LOAD(CreateDevice);

  uint32_t n = 0;
  g_vk.EnumeratePhysicalDevices(g_vk.inst, &n, nullptr);
  if (n == 0) die("no Vulkan physical devices");
  std::vector<VkPhysicalDevice> pds(n);
  g_vk.EnumeratePhysicalDevices(g_vk.inst, &n, pds.data());
  DeviceCtx c;
  // Pre-pass: when any device exposes cooperative matrix (the fork
  // driver), prefer it over a software device that enumerates first.
  bool any_cm = false;
  for (uint32_t i = 0; i < n && !any_cm; ++i) {
    uint32_t extn = 0;
    g_vk.EnumerateDeviceExtensionProperties(pds[i], nullptr, &extn, nullptr);
    std::vector<VkExtensionProperties> exts(extn);
    if (extn) {
      g_vk.EnumerateDeviceExtensionProperties(
          pds[i], nullptr, &extn, exts.data());
    }
    for (const auto& e : exts) {
      if (std::strcmp(e.extensionName, "VK_KHR_cooperative_matrix") == 0) {
        any_cm = true;
        break;
      }
    }
  }
  for (uint32_t i = 0; i < n; ++i) {
    VkPhysicalDeviceProperties props{};
    g_vk.GetPhysicalDeviceProperties(pds[i], &props);
    if (props.apiVersion < VK_API_VERSION_1_1) continue;
    std::fprintf(stderr, "probing %s api=%u.%u\n", props.deviceName,
        VK_API_VERSION_MAJOR(props.apiVersion),
        VK_API_VERSION_MINOR(props.apiVersion));
    uint32_t extn = 0;
    g_vk.EnumerateDeviceExtensionProperties(pds[i], nullptr, &extn, nullptr);
    std::vector<VkExtensionProperties> exts(extn);
    if (extn) {
      g_vk.EnumerateDeviceExtensionProperties(
          pds[i], nullptr, &extn, exts.data());
    }
    bool has_cm = false;
    for (const auto& e : exts) {
      if (std::strcmp(e.extensionName, "VK_KHR_cooperative_matrix") == 0) {
        has_cm = true;
      }
    }

    VkPhysicalDeviceSubgroupProperties sub{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmf{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    std::memset(&cmf, 0, sizeof(cmf));
    cmf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    VkPhysicalDeviceProperties2 props2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &sub;
    // Only chain the cooperative-matrix struct when the driver knows
    // the extension: old llvmpipe crashed on unknown property types.
    sub.pNext = nullptr;
    g_vk.GetPhysicalDeviceProperties2(pds[i], &props2);
    // Feature flags are queried through Features2, not Properties2:
    // the driver leaves unknown-to-properties structs untouched.
    if (has_cm) {
      VkPhysicalDeviceFeatures2 f2{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
      f2.pNext = &cmf;
      g_vk.GetPhysicalDeviceFeatures2(pds[i], &f2);
    }
    uint32_t qfn = 0;
    g_vk.GetPhysicalDeviceQueueFamilyProperties(pds[i], &qfn, nullptr);
    std::vector<VkQueueFamilyProperties> qfpv(qfn);
    g_vk.GetPhysicalDeviceQueueFamilyProperties(pds[i], &qfn, qfpv.data());
    for (uint32_t q = 0; q < qfn; ++q) {
      if ((qfpv[q].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
      c.qfi = q;
      c.subgroupSize = sub.subgroupSize;
      c.coopmat = has_cm && cmf.cooperativeMatrix == VK_TRUE;
      c.name = props.deviceName;
      g_vk.GetPhysicalDeviceMemoryProperties(pds[i], &c.mp);
      float prio = 1.0f;
      VkDeviceQueueCreateInfo qci{};
      qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      qci.queueFamilyIndex = q;
      qci.queueCount = 1;
      qci.pQueuePriorities = &prio;
      VkDeviceCreateInfo dci{};
      dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
      dci.queueCreateInfoCount = 1;
      dci.pQueueCreateInfos = &qci;
      const char* exts_on[1] = {"VK_KHR_cooperative_matrix"};
      if (c.coopmat) {
        dci.enabledExtensionCount = 1;
        dci.ppEnabledExtensionNames = exts_on;
        dci.pNext = &cmf;
      }
      if (g_vk.CreateDevice(pds[i], &dci, nullptr, &g_vk.dev) != VK_SUCCESS)
        die("CreateDevice");
      LOAD_DEV(GetDeviceQueue);
      LOAD_DEV(CreateBuffer);
      LOAD_DEV(GetBufferMemoryRequirements);
      LOAD_DEV(BindBufferMemory);
      LOAD_DEV(MapMemory);
      LOAD_DEV(UnmapMemory);
      LOAD_DEV(AllocateMemory);
      LOAD_DEV(FreeMemory);
      LOAD_DEV(CreateShaderModule);
      LOAD_DEV(CreateComputePipelines);
      LOAD_DEV(CreatePipelineLayout);
      LOAD_DEV(CreateDescriptorSetLayout);
      LOAD_DEV(CreateDescriptorPool);
      LOAD_DEV(AllocateDescriptorSets);
      LOAD_DEV(UpdateDescriptorSets);
      LOAD_DEV(CreateCommandPool);
      LOAD_DEV(AllocateCommandBuffers);
      LOAD_DEV(BeginCommandBuffer);
      LOAD_DEV(EndCommandBuffer);
      LOAD_DEV(CmdBindPipeline);
      LOAD_DEV(CmdBindDescriptorSets);
      LOAD_DEV(CmdDispatch);
      LOAD_DEV(CmdPushConstants);
      LOAD_DEV(CreateFence);
      LOAD_DEV(WaitForFences);
      LOAD_DEV(ResetFences);
      LOAD_DEV(QueueSubmit);
      LOAD_DEV(DestroyShaderModule);
      LOAD_DEV(DestroyPipeline);
      LOAD_DEV(DestroyPipelineLayout);
      LOAD_DEV(DestroyDescriptorSetLayout);
      LOAD_DEV(DestroyDescriptorPool);
      LOAD_DEV(DestroyCommandPool);
      LOAD_DEV(DestroyBuffer);
      LOAD_DEV(FreeMemory);
      LOAD_DEV(DestroyDevice);
      g_vk.GetDeviceQueue(g_vk.dev, q, 0, &c.queue);
      std::printf(
          "{\"k\":\"dev\",\"name\":\"%s\",\"subgroupSize\":%u,"
          "\"coopmat\":%s}\n",
          c.name.c_str(), c.subgroupSize, c.coopmat ? "true" : "false");
      return c;
    }
  }
  die("no Vulkan 1.2 compute device");
}

struct Side {
  const char* tag;
  VkShaderModule mod{VK_NULL_HANDLE};
  VkDescriptorSetLayout dsl{VK_NULL_HANDLE};
  VkPipelineLayout layout{VK_NULL_HANDLE};
  VkPipeline pipe{VK_NULL_HANDLE};
};

static void make_pipeline(DeviceCtx& c, Side& side) {
  VkDescriptorSetLayoutBinding b[kBindings]{};
  for (uint32_t i = 0; i < kBindings; ++i) {
    b[i].binding = i;
    b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[i].descriptorCount = 1;
    b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dslci{};
  dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslci.bindingCount = kBindings;
  dslci.pBindings = b;
  if (g_vk.CreateDescriptorSetLayout(g_vk.dev, &dslci, nullptr, &side.dsl) !=
      VK_SUCCESS)
    die("CreateDescriptorSetLayout");

  VkPipelineLayoutCreateInfo plci{};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &side.dsl;
  VkPushConstantRange pc{};
  pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pc.offset = 0;
  pc.size = sizeof(Params);
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pc;
  if (g_vk.CreatePipelineLayout(g_vk.dev, &plci, nullptr, &side.layout) !=
      VK_SUCCESS)
    die("CreatePipelineLayout");

  VkComputePipelineCreateInfo cpci{};
  cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = side.mod;
  cpci.stage.pName = "main";
  cpci.layout = side.layout;
  if (g_vk.CreateComputePipelines(g_vk.dev, VK_NULL_HANDLE, 1, &cpci, nullptr,
          &side.pipe) != VK_SUCCESS)
    die("CreateComputePipelines");
}

struct SetBufs {
  Buf a;
  Buf b;
  Buf c;
  Buf out;
};

static VkDescriptorSet make_set(DeviceCtx& c, VkDescriptorPool pool,
    VkDescriptorSetLayout dsl, const SetBufs& bufs) {
  VkDescriptorSetAllocateInfo dsai{};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &dsl;
  VkDescriptorSet set;
  if (g_vk.AllocateDescriptorSets(g_vk.dev, &dsai, &set) != VK_SUCCESS)
    die("AllocateDescriptorSets");
  VkDescriptorBufferInfo dbi[kBindings]{};
  VkWriteDescriptorSet w[kBindings]{};
  Buf const* bufs_arr[kBindings] = {&bufs.a, &bufs.b, &bufs.c, &bufs.out};
  for (uint32_t i = 0; i < kBindings; ++i) {
    dbi[i].buffer = bufs_arr[i]->buf;
    dbi[i].offset = 0;
    dbi[i].range = VK_WHOLE_SIZE;
    w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[i].dstSet = set;
    w[i].dstBinding = i;
    w[i].descriptorCount = 1;
    w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[i].pBufferInfo = &dbi[i];
  }
  g_vk.UpdateDescriptorSets(g_vk.dev, kBindings, w, 0, nullptr);
  return set;
}

static uint16_t bf16_of(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, 4);
  return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

// bf16 fill: mode 0 small integers -4..4 (exact on the bf16 grid, every
// accumulation order identical); mode 1 fractional values in
// [1.0, 2.0) on the bf16 grid (0x3F80 | 7 mantissa bits); mode 2
// positional ramp value(i) = (row % 4) + 1 with k elements per row,
// so A x B^T = k * ((r%4)+1) * ((c%4)+1) exactly in every order.
static void fill_bf16(void* p, size_t bytes, uint32_t mode, uint32_t seed,
    uint32_t k = 0) {
  uint16_t* h = (uint16_t*)p;
  std::mt19937 rng(seed);
  size_t n16 = bytes / sizeof(uint16_t);
  if (mode == 2) {
    for (size_t i = 0; i < n16; ++i) {
      h[i] = bf16_of((float)((i / k) % 4 + 1));
    }
    return;
  }
  if (mode == 0) {
    for (size_t i = 0; i < n16; ++i) {
      // bf16 of small integers -4..4: sign | exponent 129 (2^(2..-1))
      // | mantissa. Easiest correct route: build from float.
      float v = (float)(rng() % 9) - 4.0f;
      uint32_t bits;
      std::memcpy(&bits, &v, 4);
      h[i] = (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
    }
  } else if (mode == 1) {
    for (size_t i = 0; i < n16; ++i) {
      h[i] = (uint16_t)(0x3F80u | (rng() & 0x7Fu));
    }
  }
}

static double host_u16_to_f32_bits(uint16_t h) {
  uint32_t bits = (uint32_t)h << 16;
  float v;
  std::memcpy(&v, &bits, 4);
  return (double)v;
}

static void host_reference(const std::vector<uint16_t>& a,
    const std::vector<uint16_t>& b, uint32_t m, uint32_t k, uint32_t n,
    std::vector<double>& out) {
  out.assign((size_t)m * n, 0.0);
  unsigned threads = std::thread::hardware_concurrency();
  if (threads == 0u) threads = 4u;
  threads = std::min<unsigned>(threads, m);
  auto worker = [&](unsigned t) {
    size_t row_begin = (size_t)m * t / threads;
    size_t row_end = (size_t)m * (t + 1u) / threads;
    for (size_t r = row_begin; r < row_end; ++r) {
      for (uint32_t col = 0; col < n; ++col) {
        double acc = 0.0;
        const uint16_t* a_row = &a[r * k];
        const uint16_t* b_row = &b[(size_t)col * k];
        for (uint32_t inner = 0; inner < k; ++inner) {
          acc += host_u16_to_f32_bits(a_row[inner]) *
              host_u16_to_f32_bits(b_row[inner]);
        }
        out[r * n + col] = acc;
      }
    }
  };
  std::vector<std::thread> pool;
  for (unsigned t = 1u; t < threads; ++t) pool.emplace_back(worker, t);
  worker(0u);
  for (auto& th : pool) th.join();
}

// bf16 round-to-nearest-even of an f64 value, folded onto the bf16 grid
// the way the kernel drain folds its f32 tile.
static double host_bf16_round_f64(double value) {
  float f = (float)value;
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  if (std::isnan(f)) {
    bits = (bits >> 16) | 0x40u;
  } else {
    bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
  }
  bits <<= 16;
  std::memcpy(&f, &bits, 4);
  return (double)f;
}

static uint32_t group_count(uint32_t extent, uint32_t tile) {
  return (extent + tile - 1u) / tile;
}

struct RunCase {
  uint32_t m;
  const Shape* shape;
};

int main(int argc, char** argv) {
  bool quick = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--quick") == 0) quick = true;
  }
  g_vk.handle = dlopen(LIBVK, RTLD_NOW | RTLD_LOCAL);
  if (!g_vk.handle) die("dlopen %s: %s", LIBVK, dlerror());
  g_vk.GetInstanceProcAddr =
      (PFN_vkGetInstanceProcAddr)dlsym(g_vk.handle, "vkGetInstanceProcAddr");
  g_vk.GetDeviceProcAddr =
      (PFN_vkGetDeviceProcAddr)dlsym(g_vk.handle, "vkGetDeviceProcAddr");
  if (!g_vk.GetInstanceProcAddr || !g_vk.GetDeviceProcAddr)
    die("vulkan entry points missing");
  g_vk.CreateInstance =
      (PFN_vkCreateInstance)g_vk.GetInstanceProcAddr(
          nullptr, "vkCreateInstance");
  if (!g_vk.CreateInstance) die("vkCreateInstance missing");

  DeviceCtx c = setup_device();

  const bool tiny = argc > 1 && std::strcmp(argv[1], "--tiny") == 0;

  // Compile the three shaders.
  const char* dir = "tools/bf16-prefill-bench/shaders";
  static std::string spv_base, spv_cand, spv_tile;
  std::string base_spv_path = std::string("/tmp/mm_bf16_base.") +
      std::to_string(getpid()) + ".spv";
  std::string cand_spv_path = std::string("/tmp/mm_bf16_cand.") +
      std::to_string(getpid()) + ".spv";
  std::string tile_spv_path = std::string("/tmp/mm_tile_base.") +
      std::to_string(getpid()) + ".spv";
  if (compile_shader(
          (std::string(dir) + "/mm_bf16_base.comp").c_str(), "",
          base_spv_path.c_str()) != 0)
    die("base shader compile");
  if (compile_shader(
          (std::string(dir) + "/mm_bf16_cand.comp").c_str(), "",
          cand_spv_path.c_str()) != 0)
    die("cand shader compile");
  if (compile_shader(
          (std::string(dir) + "/mm_tile_base.comp").c_str(), "-DUSE_BF16=1",
          tile_spv_path.c_str()) != 0)
    die("tile shader compile");
  spv_base = read_file(base_spv_path.c_str());
  spv_cand = read_file(cand_spv_path.c_str());
  spv_tile = read_file(tile_spv_path.c_str());

  auto make_module = [&](const std::string& spv) {
    VkShaderModuleCreateInfo mi{};
    mi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    mi.codeSize = spv.size();
    mi.pCode = (const uint32_t*)spv.data();
    VkShaderModule m;
    if (g_vk.CreateShaderModule(g_vk.dev, &mi, nullptr, &m) != VK_SUCCESS)
      die("CreateShaderModule");
    return m;
  };
  Side base{"base"};
  Side cand{"cand"};
  Side tile{"tile"};
  tile.mod = make_module(spv_tile);
  make_pipeline(c, tile);
  if (c.coopmat) {
    base.mod = make_module(spv_base);
    cand.mod = make_module(spv_cand);
    make_pipeline(c, base);
    make_pipeline(c, cand);
  }

  VkCommandPoolCreateInfo cpi{};
  cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cpi.queueFamilyIndex = c.qfi;
  VkCommandPool pool;
  if (g_vk.CreateCommandPool(g_vk.dev, &cpi, nullptr, &pool) != VK_SUCCESS)
    die("CreateCommandPool");
  VkCommandBufferAllocateInfo cbai{};
  cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbai.commandPool = pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd;
  if (g_vk.AllocateCommandBuffers(g_vk.dev, &cbai, &cmd) != VK_SUCCESS)
    die("AllocateCommandBuffers");
  VkFenceCreateInfo fci{};
  fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence;
  if (g_vk.CreateFence(g_vk.dev, &fci, nullptr, &fence) != VK_SUCCESS)
    die("CreateFence");

  uint32_t total_sets = 96;
  VkDescriptorPoolSize ps{
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kBindings * total_sets};
  VkDescriptorPoolCreateInfo dpci{};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpci.maxSets = total_sets;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &ps;
  VkDescriptorPool dpool;
  if (g_vk.CreateDescriptorPool(g_vk.dev, &dpci, nullptr, &dpool) != VK_SUCCESS)
    die("CreateDescriptorPool");

  const int reps = quick ? 12 : 50;
  const int rounds = quick ? 1 : 3;

  if (tiny) {
    if (!c.coopmat) {
      std::printf("{\"k\":\"tiny\",\"note\":\"no coopmat device\"}\n");
      return 0;
    }
    for (uint32_t k : {32u}) {
    const uint32_t m = 8, n = 8;
    uint32_t mode = 0;
    size_t a_bytes = (size_t)m * k * 2, b_bytes = (size_t)k * n * 2;
    size_t out_bytes = (size_t)m * n * 2;
    SetBufs bufs{make_buf(g_vk.dev, c.mp, a_bytes), make_buf(g_vk.dev, c.mp, b_bytes),
        make_buf(g_vk.dev, c.mp, 16), make_buf(g_vk.dev, c.mp, out_bytes)};
    fill_bf16(bufs.a.mapped, a_bytes, mode, 0xBEEF, k);
    fill_bf16(bufs.b.mapped, b_bytes, mode, 0xFACE, k);
    VkDescriptorSet base_set = make_set(c, dpool, base.dsl, bufs);
    VkDescriptorSet cand_set = make_set(c, dpool, cand.dsl, bufs);
    Params p{};
    p.count = m * n;
    p.output_size = m * n;
    p.lhs_size = m * k;
    p.rhs_size = k * n;
    p.reduce_size = k;
    p.matrix_m = m;
    p.matrix_n = n;
    p.matrix_k = k;
    p.flags = 1;
    p.alpha = 1.0f;
    p.beta = 0.0f;
    p.lhs_gap = k;
    p.rhs_gap = k;
    std::vector<uint16_t> base_out((size_t)m * n), cand_out((size_t)m * n);
    auto run1 = [&](Side& side, VkDescriptorSet set, std::vector<uint16_t>& o) {
      VkCommandBufferBeginInfo bi{};
      bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      g_vk.BeginCommandBuffer(cmd, &bi);
      g_vk.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, side.pipe);
      g_vk.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          side.layout, 0, 1, &set, 0, nullptr);
      g_vk.CmdPushConstants(cmd, side.layout, VK_SHADER_STAGE_COMPUTE_BIT,
          0, sizeof(Params), &p);
      g_vk.CmdDispatch(cmd, 1, 1, 1);
      g_vk.EndCommandBuffer(cmd);
      g_vk.ResetFences(g_vk.dev, 1, &fence);
      VkSubmitInfo si{};
      si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      si.commandBufferCount = 1;
      si.pCommandBuffers = &cmd;
      g_vk.QueueSubmit(c.queue, 1, &si, fence);
      g_vk.WaitForFences(g_vk.dev, 1, &fence, VK_TRUE, 30'000'000'000ull);
      o.resize((size_t)m * n);
      std::memcpy(o.data(), bufs.out.mapped, out_bytes);
    };
    std::printf("== A (%u x %u) ==\n", m, k);
    for (uint32_t r = 0; r < m; ++r) {
      for (uint32_t i = 0; i < k; ++i)
        std::printf(" %g",
            host_u16_to_f32_bits(((uint16_t*)bufs.a.mapped)[r * k + i]));
      std::printf("\n");
    }
    std::printf("== B (%u x %u stored n-major) ==\n", n, k);
    for (uint32_t cc = 0; cc < n; ++cc) {
      for (uint32_t i = 0; i < k; ++i)
        std::printf(" %g",
            host_u16_to_f32_bits(((uint16_t*)bufs.b.mapped)[cc * k + i]));
      std::printf("\n");
    }
    run1(base, base_set, base_out);
    run1(cand, cand_set, cand_out);
    uint32_t bad = 0;
    size_t firstbad = SIZE_MAX;
    for (uint32_t i = 0; i < (uint32_t)m * n; ++i) {
      if (base_out[i] != cand_out[i]) {
        ++bad;
        if (firstbad == SIZE_MAX) firstbad = i;
      }
    }
    std::printf(
        "{\"k\":\"tiny\",\"kk\":%u,\"mismatch\":%u,\"first_bad\":%zu",
        k, bad, firstbad == SIZE_MAX ? (size_t)-1 : firstbad);
    if (bad) {
      std::printf(",\"r\":%u,\"c\":%u,\"base\":%g,\"cand\":%g",
          (uint32_t)(firstbad / n), (uint32_t)(firstbad % n),
          host_u16_to_f32_bits(base_out[firstbad]),
          host_u16_to_f32_bits(cand_out[firstbad]));
    }
    std::printf("}\n");
    // Full diff dump for the failing case.
    std::printf("== A row 0: ==");
    for (uint32_t i = 0; i < k; ++i)
      std::printf(" %g", host_u16_to_f32_bits(((uint16_t*)bufs.a.mapped)[i]));
    std::printf("\n== B col 0 (n-major row 0): ==");
    for (uint32_t i = 0; i < k; ++i)
      std::printf(" %g", host_u16_to_f32_bits(((uint16_t*)bufs.b.mapped)[i]));
    std::printf("\n== per-element base vs cand vs exp ==\n");
    std::vector<double> ref;
    std::vector<uint16_t> a_vals((size_t)m * k), b_vals((size_t)k * n);
    std::memcpy(a_vals.data(), bufs.a.mapped, a_bytes);
    std::memcpy(b_vals.data(), bufs.b.mapped, b_bytes);
    host_reference(a_vals, b_vals, m, k, n, ref);
    for (uint32_t r = 0; r < m; ++r)
      for (uint32_t cc = 0; cc < n; ++cc) {
        double b_ = host_u16_to_f32_bits(base_out[r * n + cc]);
        double c_ = host_u16_to_f32_bits(cand_out[r * n + cc]);
        if (b_ != c_)
          std::printf("r%u c%u base=%g cand=%g exp=%g\n", r, cc, b_, c_,
              ref[r * n + cc]);
      }
    }
    return 0;
  }

  for (uint32_t m : kMs) {
    if (!c.coopmat && m != 30) {
      std::printf(
          "{\"k\":\"skip\",\"m\":%u,\"note\":\"software device - "
          "jumbo tile legs are a hardware leg\"}\n", m);
      continue;
    }
    for (const auto& shape : kShapes) {
      uint32_t k = shape.k, n = shape.n;
      size_t a_bytes = (size_t)m * k * 2, b_bytes = (size_t)k * n * 2;
      size_t out_bytes = (size_t)m * n * 2;
      if (out_bytes % 4) die("odd output bytes");

      // Correctness and timing run at flags=1 only: the production
      // word-x-word layout this harness fills (A row-major, B n-major,
      // gaps = k). Non-production flag combos would read out of bounds
      // under these fills; multi-orientation staging coverage lives in
      // the family suite (valid layouts) and the digest gates.
      for (uint32_t mode = 0; mode < 3; ++mode) {
        SetBufs bufs{make_buf(g_vk.dev, c.mp, a_bytes),
            make_buf(g_vk.dev, c.mp, b_bytes), make_buf(g_vk.dev, c.mp, 16),
            make_buf(g_vk.dev, c.mp, out_bytes)};
        fill_bf16(bufs.a.mapped, a_bytes, mode, 0xBEEF + m + k, k);
        fill_bf16(bufs.b.mapped, b_bytes, mode, 0xFACE + k + n, k);

        // Tile descriptor set (always runs).
        VkDescriptorSet tile_set = make_set(c, dpool, tile.dsl, bufs);
        VkDescriptorSet base_set = VK_NULL_HANDLE;
        VkDescriptorSet cand_set = VK_NULL_HANDLE;
        if (c.coopmat) {
          base_set = make_set(c, dpool, base.dsl, bufs);
          cand_set = make_set(c, dpool, cand.dsl, bufs);
        }

        Params p{};
        p.count = m * n;
        p.output_size = m * n;
        p.lhs_size = m * k;
        p.rhs_size = k * n;
        p.reduce_size = k;
        p.matrix_m = m;
        p.matrix_n = n;
        p.matrix_k = k;
        p.flags = 1;  // production word-x-word path
        p.alpha = 1.0f;
        p.lhs_gap = k;
        p.rhs_gap = k;

        // Correctness: run all three into distinct outputs. The tile
        // runs into a scratch buffer first so base/cand keep bufs.out.
        std::vector<uint16_t> tile_out((size_t)m * n);
        auto dispatch_one = [&](Side& side, VkDescriptorSet set,
                                Buf& out, uint32_t tile_size) {
          VkCommandBufferBeginInfo bi{};
          bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
          bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
          g_vk.BeginCommandBuffer(cmd, &bi);
          g_vk.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              side.pipe);
          VkDescriptorBufferInfo oi{out.buf, 0, VK_WHOLE_SIZE};
          VkWriteDescriptorSet w{};
          w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
          w.dstSet = set;
          w.dstBinding = 3;
          w.descriptorCount = 1;
          w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
          w.pBufferInfo = &oi;
          g_vk.UpdateDescriptorSets(g_vk.dev, 1, &w, 0, nullptr);
          g_vk.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              side.layout, 0, 1, &set, 0, nullptr);
          g_vk.CmdPushConstants(cmd, side.layout,
              VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Params), &p);
          g_vk.CmdDispatch(cmd, group_count(n, tile_size),
              group_count(m, tile_size), 1);
          g_vk.EndCommandBuffer(cmd);
          g_vk.ResetFences(g_vk.dev, 1, &fence);
          VkSubmitInfo si{};
          si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
          si.commandBufferCount = 1;
          si.pCommandBuffers = &cmd;
          if (g_vk.QueueSubmit(c.queue, 1, &si, fence) != VK_SUCCESS)
            die("QueueSubmit");
          g_vk.WaitForFences(g_vk.dev, 1, &fence, VK_TRUE, 30'000'000'000ull);
        };
        dispatch_one(tile, tile_set, bufs.out, 16);
        std::memcpy(tile_out.data(), bufs.out.mapped, out_bytes);
        if (c.coopmat) {
          dispatch_one(base, base_set, bufs.out, 32);
          std::vector<uint16_t> base_out((size_t)m * n);
          std::memcpy(base_out.data(), bufs.out.mapped, out_bytes);
          dispatch_one(cand, cand_set, bufs.out, 32);
          std::vector<uint16_t> cand_out((size_t)m * n);
          std::memcpy(cand_out.data(), bufs.out.mapped, out_bytes);

          uint64_t mismatches_bc = 0, mismatches_bt = 0;
          size_t first_bad = SIZE_MAX;
          for (size_t i = 0; i < base_out.size(); ++i) {
            if (cand_out[i] != base_out[i]) {
              ++mismatches_bc;
              if (first_bad == SIZE_MAX) first_bad = i;
            }
            if (tile_out[i] != base_out[i]) ++mismatches_bt;
          }
          // Dead-row guard: every expected row must carry a nonzero
          // element, so an unwritten drain cannot pass silently.
          uint32_t dead_rows = 0;
          for (uint32_t r = 0; r < m; ++r) {
            bool nonzero = false;
            for (uint32_t col = 0; col < n; ++col) {
              if (base_out[(size_t)r * n + col] != 0) nonzero = true;
            }
            if (!nonzero) ++dead_rows;
          }

          double worst_rel = 0.0;
          if (mode == 1 && m == 30) {
            std::vector<uint16_t> a_vals((size_t)m * k), b_vals((size_t)k * n);
            std::memcpy(a_vals.data(), bufs.a.mapped, a_bytes);
            std::memcpy(b_vals.data(), bufs.b.mapped, b_bytes);
            std::vector<double> ref;
            host_reference(a_vals, b_vals, m, k, n, ref);
            double bound = (double)(k / 8u) * 4.5 * 0x1p-23;
            for (size_t i = 0; i < ref.size(); ++i) {
              double truth = host_bf16_round_f64(ref[i]);
              double got = host_u16_to_f32_bits(cand_out[i]);
              double denom = std::max(1e-30, std::fabs(truth));
              worst_rel = std::max(worst_rel,
                  std::fabs(got - truth) / denom /
                      (bound + 0x1p-8));
            }
          }

          std::printf(
              "{\"k\":\"correct\",\"m\":%u,\"shape\":\"%s\",\"mode\":%u,"
              "\"cand_vs_base_mismatch\":%" PRIu64
              ",\"base_vs_tile_mismatch\":%" PRIu64
              ",\"dead_rows\":%u,\"first_bad\":%zu",
              m, shape.name, mode, mismatches_bc, mismatches_bt,
              dead_rows, first_bad == SIZE_MAX ? (size_t)-1 : first_bad);
          if (mode != 1 && mismatches_bc && first_bad != SIZE_MAX) {
            size_t i = first_bad;
            uint32_t r = (uint32_t)(i / n), c = (uint32_t)(i % n);
            double got = host_u16_to_f32_bits(cand_out[i]);
            double bas = host_u16_to_f32_bits(base_out[i]);
            double exp = -1.0;
            if (mode == 0) {
              std::vector<uint16_t> a_vals((size_t)m * k),
                  b_vals((size_t)k * n);
              std::memcpy(a_vals.data(), bufs.a.mapped, a_bytes);
              std::memcpy(b_vals.data(), bufs.b.mapped, b_bytes);
              std::vector<double> ref;
              host_reference(a_vals, b_vals, m, k, n, ref);
              exp = ref[i];
            } else if (mode == 2) {
              exp = (double)k * (double)((r % 4) + 1) * (double)((c % 4) + 1);
            }
            std::printf(
                ",\"diag_r\":%u,\"diag_c\":%u,\"got\":%.4f,"
                "\"base\":%.4f,\"exp\":%.4f",
                r, c, got, bas, exp);
          }
          if (mode == 1 && m == 30) {
            std::printf(",\"f64_worst_rel_of_bound\":%.4f", worst_rel);
          }
          if (getenv("BF16_DUMP") && mismatches_bc) {
            std::printf("== A (%u x %u) ==", m, k);
            for (uint32_t r = 0; r < m; ++r)
              for (uint32_t i = 0; i < k; ++i)
                std::printf(" %g",
                    host_u16_to_f32_bits(
                        ((uint16_t*)bufs.a.mapped)[r * k + i]));
            std::printf("\n== B (%u x %u, stored n-major) ==", n, k);
            for (uint32_t c = 0; c < n; ++c)
              for (uint32_t i = 0; i < k; ++i)
                std::printf(" %g",
                    host_u16_to_f32_bits(
                        ((uint16_t*)bufs.b.mapped)[c * k + i]));
            std::printf("\n== cand ==\n");
            for (uint32_t r = 0; r < m; ++r) {
              for (uint32_t c = 0; c < n; ++c)
                std::printf(" %g", host_u16_to_f32_bits(cand_out[r * n + c]));
              std::printf("\n");
            }
            std::printf("== base ==\n");
            for (uint32_t r = 0; r < m; ++r) {
              for (uint32_t c = 0; c < n; ++c)
                std::printf(" %g", host_u16_to_f32_bits(base_out[r * n + c]));
              std::printf("\n");
            }
          }
          std::printf("}\n");
        } else {
          uint32_t dead_rows = 0;
          for (uint32_t r = 0; r < m; ++r) {
            bool nonzero = false;
            for (uint32_t col = 0; col < n; ++col) {
              if (tile_out[(size_t)r * n + col] != 0) nonzero = true;
            }
            if (!nonzero) ++dead_rows;
          }
          std::printf(
              "{\"k\":\"correct\",\"m\":%u,\"shape\":\"%s\",\"mode\":%u,"
              "\"note\":\"no coopmat device - tile leg only\","
              "\"dead_rows\":%u}\n",
              m, shape.name, mode, dead_rows);
        }

        // Timing: R alternating dispatches in one command buffer,
        // wall-clocked around the submission; three rounds; medians.
        auto time_kernel = [&](Side& side, VkDescriptorSet set,
                               uint32_t tile_size) {
          std::vector<double> samples;
          for (int round = 0; round < rounds; ++round) {
            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            g_vk.BeginCommandBuffer(cmd, &bi);
            g_vk.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                side.pipe);
            g_vk.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                side.layout, 0, 1, &set, 0, nullptr);
            g_vk.CmdPushConstants(cmd, side.layout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Params), &p);
            for (int i = 0; i < reps; ++i) {
              g_vk.CmdDispatch(cmd, group_count(n, tile_size),
                  group_count(m, tile_size), 1);
            }
            g_vk.EndCommandBuffer(cmd);
            g_vk.ResetFences(g_vk.dev, 1, &fence);
            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cmd;
            auto t0 = std::chrono::steady_clock::now();
            g_vk.QueueSubmit(c.queue, 1, &si, fence);
            g_vk.WaitForFences(g_vk.dev, 1, &fence, VK_TRUE,
                30'000'000'000ull);
            auto t1 = std::chrono::steady_clock::now();
            samples.push_back(
                std::chrono::duration<double, std::micro>(t1 - t0).count() /
                reps);
          }
          std::sort(samples.begin(), samples.end());
          return samples[samples.size() / 2];
        };
        if (c.coopmat && mode == 1) {
          double base_us = time_kernel(base, base_set, 32);
          double cand_us = time_kernel(cand, cand_set, 32);
          double tile_us = time_kernel(tile, tile_set, 16);
          double flops = 2.0 * m * k * n;
          std::printf(
              "{\"k\":\"time\",\"m\":%u,\"shape\":\"%s\","
              "\"base_us\":%.1f,\"cand_us\":%.1f,\"tile_us\":%.1f,"
              "\"base_tflops\":%.3f,\"cand_tflops\":%.3f,"
              "\"tile_tflops\":%.3f}\n",
              m, shape.name, base_us, cand_us, tile_us,
              flops / (base_us * 1e6), flops / (cand_us * 1e6),
              flops / (tile_us * 1e6));
        } else if (mode == 1) {
          double tile_us = time_kernel(tile, tile_set, 16);
          double flops = 2.0 * m * k * n;
          std::printf(
              "{\"k\":\"time\",\"m\":%u,\"shape\":\"%s\",\"tile_us\":%.1f,"
              "\"tile_tflops\":%.3f}\n",
              m, shape.name, tile_us, flops / (tile_us * 1e6));
        }

        g_vk.FreeMemory(g_vk.dev, bufs.a.mem, nullptr);
        g_vk.FreeMemory(g_vk.dev, bufs.b.mem, nullptr);
        g_vk.FreeMemory(g_vk.dev, bufs.c.mem, nullptr);
        g_vk.FreeMemory(g_vk.dev, bufs.out.mem, nullptr);
        g_vk.DestroyBuffer(g_vk.dev, bufs.a.buf, nullptr);
        g_vk.DestroyBuffer(g_vk.dev, bufs.b.buf, nullptr);
        g_vk.DestroyBuffer(g_vk.dev, bufs.c.buf, nullptr);
        g_vk.DestroyBuffer(g_vk.dev, bufs.out.buf, nullptr);
      }
    }
  }
  std::printf("{\"k\":\"done\"}\n");
  return 0;
}
