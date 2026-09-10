// Q4 decode GEMV bandwidth bench for the M1 (Honeykrisp) and lavapipe
// correctness screening. It times the REAL production shader
// (shaders/qmm_vec_base.comp, a frozen copy of
// overlay/mlx/backend/omarchy/shaders/qmm_vec.comp) and a candidate
// edit (shaders/qmm_vec_cand.comp) on the four per-layer dispatch
// shapes the Qwen2.5-0.5B decode chain issues through
// QmmVecQ4MultiSubgroupF16:
//   qkv     dims=3  n=(896,128,128)  k=896  grid=144
//   o       dims=1  n=(896,)         k=896  grid=112
//   gate_up dims=2  n=(4864,4864)    k=896  grid=1216
//   down    dims=1  n=(896,)         k=4864 grid=112
// with the push-constant and binding contract of
// dispatch_quantized_gemv_group (primitives.cpp): operation=4,
// reduce_size=64, matrix_m=1, per-weight shape[i], grid =
// sum(ceil(n_i/8)), bindings 0 = x then 6 per weight slot.
//
// Measurements:
//   * streaming copy and read probes over 256 MB: the device's
//     achievable bandwidth ceiling.
//   * isolated per-shape dispatch, timestamped per submit (base and
//     candidate alternate within each repetition to cancel drift).
//   * the four-dispatch layer chain in ONE command buffer with a
//     timestamp after each dispatch (what one decode token issues per
//     layer), two interleaved rounds.
//   * bit-exactness: identical input buffers through base and
//     candidate into distinct outputs, compared as raw storage bits -
//     must be 0 mismatches per shape.
//
// Modes: default compiles the production subgroup variant
// (-DUSE_SUBGROUP=1, requires subgroupSize == 32 - the M1); "--tree"
// compiles the tree-reduction variants for lavapipe screening
// (lavapipe subgroupSize != 32 would make the subgroup variant wrong
// by design; llvmpipe dispatches the tree variant).
//
// Build: g++ -std=c++17 -O2 -o /tmp/q4-bw-bench tools/q4-bw-bench/bench.cpp
// Run (repo root): /tmp/q4-bw-bench [--tree] [--quick]
// Output: NDJSON on stdout.

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
#include <unistd.h>
#include <vector>

#define LIBVK "libvulkan.so.1"

struct VkTable {
  void* handle{nullptr};
  VkInstance inst{VK_NULL_HANDLE};
  VkDevice dev{VK_NULL_HANDLE};
#define VK_FN(name) PFN_vk##name name{nullptr}
  VK_FN(GetInstanceProcAddr);
  VK_FN(GetDeviceProcAddr);
  VK_FN(CreateInstance);
  VK_FN(EnumerateInstanceExtensionProperties);
  VK_FN(EnumerateInstanceLayerProperties);
  VK_FN(DestroyInstance);
  VK_FN(EnumeratePhysicalDevices);
  VK_FN(GetPhysicalDeviceProperties);
  VK_FN(GetPhysicalDeviceProperties2);
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
  VK_FN(DestroyFence);
  VK_FN(WaitForFences);
  VK_FN(CreateQueryPool);
  VK_FN(CmdResetQueryPool);
  VK_FN(CmdWriteTimestamp);
  VK_FN(GetQueryPoolResults);
  VK_FN(QueueSubmit);
  VK_FN(DestroyShaderModule);
  VK_FN(DestroyPipeline);
  VK_FN(DestroyPipelineLayout);
  VK_FN(DestroyDescriptorSetLayout);
  VK_FN(DestroyDescriptorPool);
  VK_FN(DestroyCommandPool);
  VK_FN(DestroyBuffer);
  VK_FN(DestroyDevice);
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

static int vk_init() {
  g_vk.handle = dlopen(LIBVK, RTLD_NOW | RTLD_LOCAL);
  if (!g_vk.handle) {
    std::fprintf(stderr, "dlopen %s: %s\n", LIBVK, dlerror());
    return -1;
  }
  g_vk.GetInstanceProcAddr =
      (PFN_vkGetInstanceProcAddr)dlsym(g_vk.handle, "vkGetInstanceProcAddr");
  if (!g_vk.GetInstanceProcAddr) {
    std::fprintf(stderr, "vkGetInstanceProcAddr missing\n");
    return -1;
  }
  g_vk.CreateInstance =
      (PFN_vkCreateInstance)vk_load(nullptr, "vkCreateInstance");
  g_vk.EnumerateInstanceExtensionProperties =
      (PFN_vkEnumerateInstanceExtensionProperties)vk_load(
          nullptr, "vkEnumerateInstanceExtensionProperties");
  if (!g_vk.CreateInstance) {
    std::fprintf(stderr, "vkCreateInstance missing\n");
    return -1;
  }
  g_vk.GetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)dlsym(
      g_vk.handle, "vkGetDeviceProcAddr");
  if (!g_vk.GetDeviceProcAddr) {
    std::fprintf(stderr, "vkGetDeviceProcAddr missing\n");
    return -1;
  }
  return 0;
}

static void vk_load_instance() {
  LOAD(DestroyInstance);
  LOAD(EnumeratePhysicalDevices);
  LOAD(GetPhysicalDeviceProperties);
  LOAD(GetPhysicalDeviceProperties2);
  LOAD(GetPhysicalDeviceMemoryProperties);
  LOAD(GetPhysicalDeviceQueueFamilyProperties);
  LOAD(CreateDevice);
}

static void vk_load_device() {
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
  LOAD_DEV(DestroyFence);
  LOAD_DEV(WaitForFences);
  LOAD_DEV(CreateQueryPool);
  LOAD_DEV(CmdResetQueryPool);
  LOAD_DEV(CmdWriteTimestamp);
  LOAD_DEV(GetQueryPoolResults);
  LOAD_DEV(QueueSubmit);
  LOAD_DEV(DestroyShaderModule);
  LOAD_DEV(DestroyPipeline);
  LOAD_DEV(DestroyPipelineLayout);
  LOAD_DEV(DestroyDescriptorSetLayout);
  LOAD_DEV(DestroyDescriptorPool);
  LOAD_DEV(DestroyCommandPool);
  LOAD_DEV(DestroyBuffer);
  LOAD_DEV(DestroyDevice);
}

#undef LOAD
#undef LOAD_DEV

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

static bool have_tool(const char* tool) {
  std::string cmd = std::string("which ") + tool + " >/dev/null 2>&1";
  return system(cmd.c_str()) == 0;
}

// Compile src with the given defines into out_spv, matching the repo's
// omarchy_shader (glslc preferred; vulkan1.3 target required for the
// subgroup extensions).
static int compile_shader(const char* src, const char* defines,
    const char* out_spv) {
  std::string cmd;
  if (have_tool("glslc")) {
    cmd = "glslc -fshader-stage=compute --target-env=vulkan1.3 ";
  } else if (have_tool("glslangValidator")) {
    cmd = "glslangValidator -V --target-env vulkan1.3 ";
  } else {
    die("neither glslc nor glslangValidator found in PATH");
  }
  cmd += defines;
  cmd += " ";
  cmd += src;
  cmd += " -o ";
  cmd += out_spv;
  cmd += " 2>&1";
  if (system(cmd.c_str()) != 0) {
    std::fprintf(stderr, "shader compile failed: %s\n", cmd.c_str());
    return -1;
  }
  struct stat st;
  if (stat(out_spv, &st) != 0 || st.st_size == 0) {
    std::fprintf(stderr, "shader compile produced no SPIR-V: %s\n", out_spv);
    return -1;
  }
  return 0;
}

struct Buf {
  VkBuffer buf{VK_NULL_HANDLE};
  VkDeviceMemory mem{VK_NULL_HANDLE};
  VkDeviceSize size{0};
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
    VkDeviceSize size, bool zero) {
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
  if (zero) return b;
  // Deterministic fill. As f16 halves the pattern lands in
  // [2^-4, 2^-1): never NaN/Inf, so every dot chain stays finite; as
  // packed uint32 weight words the nibbles are plain 0..15 values.
  void* p;
  g_vk.MapMemory(dev, b.mem, 0, size, 0, &p);
  std::mt19937 rng(0xC0FFEEu ^ (uint32_t)(size * 2654435761ull));
  uint16_t* h = (uint16_t*)p;
  for (VkDeviceSize i = 0; i < size / sizeof(uint16_t); ++i) {
    h[i] = (uint16_t)(0x3000u | (rng() & 0x3FFu));
  }
  g_vk.UnmapMemory(dev, b.mem);
  return b;
}

// Push constants: must byte-match the shader Params block (30 x
// 4-byte words, scalar alignment throughout).
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
};
static_assert(sizeof(Params) == 120, "push constant block must be 120 bytes");

static constexpr uint32_t kBindings = 19; // x + 3 weight slots * 6
static constexpr uint32_t kMaxDims = 3;

struct Shape {
  const char* name;
  uint32_t k;
  uint32_t dims;
  uint32_t n[kMaxDims];
};

static const Shape kShapes[] = {
    {"qkv", 896, 3, {896, 128, 128}},
    {"o", 896, 1, {896, 0, 0}},
    {"gate_up", 896, 2, {4864, 4864, 0}},
    {"down", 4864, 1, {896, 0, 0}},
};
static constexpr uint32_t kNumShapes = 4;

static uint32_t shape_groups(const Shape& s) {
  uint32_t total = 0;
  for (uint32_t i = 0; i < s.dims; ++i) {
    total += (s.n[i] + 7u) / 8u;
  }
  return total;
}

// Bytes one dispatch touches: weight words + scales + biases + x row +
// outputs (flags are 0 in the bench, so no addend reads / sum writes).
static uint64_t shape_bytes(const Shape& s) {
  uint64_t words = 0, params = 0, outs = 0;
  for (uint32_t i = 0; i < s.dims; ++i) {
    words += (uint64_t)s.n[i] * (s.k / 8u) * 4u;
    params += 2ull * s.n[i] * (s.k / 64u) * 2u;
    outs += s.n[i] * 2u;
  }
  return words + params + (uint64_t)s.k * 2u + outs;
}

struct DeviceCtx {
  VkQueue queue{VK_NULL_HANDLE};
  uint32_t qfi{0};
  VkPhysicalDeviceMemoryProperties mp{};
  uint32_t timestampPeriod{0};
  uint32_t subgroupSize{0};
  bool subgroupArith{false};
  std::string name;
};

static DeviceCtx setup_device() {
  VkApplicationInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  ai.pApplicationName = "q4-bw-bench";
  ai.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo ici{};
  ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ici.pApplicationInfo = &ai;
  if (g_vk.CreateInstance(&ici, nullptr, &g_vk.inst) != VK_SUCCESS)
    die("CreateInstance");
  vk_load_instance();

  uint32_t n = 0;
  g_vk.EnumeratePhysicalDevices(g_vk.inst, &n, nullptr);
  if (n == 0) die("no Vulkan physical devices");
  std::vector<VkPhysicalDevice> pds(n);
  g_vk.EnumeratePhysicalDevices(g_vk.inst, &n, pds.data());
  DeviceCtx c;
  for (uint32_t i = 0; i < n; ++i) {
    VkPhysicalDeviceProperties props{};
    g_vk.GetPhysicalDeviceProperties(pds[i], &props);
    if (props.apiVersion < VK_API_VERSION_1_2) continue;
    VkPhysicalDeviceSubgroupProperties sub{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 props2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &sub;
    g_vk.GetPhysicalDeviceProperties2(pds[i], &props2);
    uint32_t qfn = 0;
    g_vk.GetPhysicalDeviceQueueFamilyProperties(pds[i], &qfn, nullptr);
    std::vector<VkQueueFamilyProperties> qfpv(qfn);
    g_vk.GetPhysicalDeviceQueueFamilyProperties(pds[i], &qfn, qfpv.data());
    for (uint32_t q = 0; q < qfn; ++q) {
      if ((qfpv[q].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
      c.qfi = q;
      c.subgroupSize = sub.subgroupSize;
      c.subgroupArith =
          (sub.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
      c.timestampPeriod = props.limits.timestampPeriod;
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
      VkPhysicalDeviceVulkan12Features f12{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
      f12.shaderFloat16 = VK_TRUE;
      dci.pNext = &f12;
      if (g_vk.CreateDevice(pds[i], &dci, nullptr, &g_vk.dev) != VK_SUCCESS)
        die("CreateDevice");
      vk_load_device();
      g_vk.GetDeviceQueue(g_vk.dev, q, 0, &c.queue);
      std::printf(
          "{\"k\":\"dev\",\"name\":\"%s\",\"subgroupSize\":%u,"
          "\"arith\":%s,\"ts_period_ns\":%.3f}\n",
          props.deviceName, c.subgroupSize, c.subgroupArith ? "true" : "false",
          c.timestampPeriod / 1.0);
      return c;
    }
  }
  die("no Vulkan 1.2 compute device");
}

static VkShaderModule make_module(const std::string& spv) {
  VkShaderModuleCreateInfo mi{};
  mi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  mi.codeSize = spv.size();
  mi.pCode = (const uint32_t*)spv.data();
  VkShaderModule m;
  if (g_vk.CreateShaderModule(g_vk.dev, &mi, nullptr, &m) != VK_SUCCESS)
    die("CreateShaderModule");
  return m;
}

struct Side {
  std::string tag;
  VkShaderModule mod{VK_NULL_HANDLE};
  VkDescriptorSetLayout dsl{VK_NULL_HANDLE};
  VkPipelineLayout layout{VK_NULL_HANDLE};
  VkPipeline pipe{VK_NULL_HANDLE};
};

static void make_pipeline(Side& side) {
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
  if (g_vk.CreateComputePipelines(g_vk.dev, VK_NULL_HANDLE, 1, &cpci,
          nullptr, &side.pipe) != VK_SUCCESS)
    die("CreateComputePipelines");
}

struct SetBufs {
  Buf x;
  Buf w[kMaxDims];
  Buf scales[kMaxDims];
  Buf biases[kMaxDims];
  Buf out[kMaxDims];
};

static VkDescriptorSet make_set(VkDescriptorPool pool, VkDescriptorSetLayout dsl,
    const SetBufs& bufs, uint32_t dims) {
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
  auto info = [&](uint32_t i, Buf& buf) {
    dbi[i].buffer = buf.buf;
    dbi[i].offset = 0;
    dbi[i].range = VK_WHOLE_SIZE;
  };
  info(0, const_cast<SetBufs&>(bufs).x);
  for (uint32_t i = 0; i < kMaxDims; ++i) {
    uint32_t base = 1 + i * 6;
    if (i < dims) {
      info(base, const_cast<SetBufs&>(bufs).w[i]);
      info(base + 1, const_cast<SetBufs&>(bufs).scales[i]);
      info(base + 2, const_cast<SetBufs&>(bufs).biases[i]);
      info(base + 3, const_cast<SetBufs&>(bufs).out[i]);
      info(base + 4, const_cast<SetBufs&>(bufs).out[i]); // addend filler
      info(base + 5, const_cast<SetBufs&>(bufs).out[i]); // sum filler
    } else {
      for (uint32_t j = 0; j < 6; ++j) {
        info(base + j, const_cast<SetBufs&>(bufs).out[0]);
      }
    }
  }
  for (uint32_t i = 0; i < kBindings; ++i) {
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

static void fill_params(Params& p, const Shape& s, uint32_t groups) {
  std::memset(&p, 0, sizeof(p));
  p.count = groups;
  p.operation = 4u;
  p.reduce_size = 64u;
  p.matrix_m = 1u;
  p.matrix_k = s.k;
  p.dims = s.dims;
  for (uint32_t i = 0; i < s.dims; ++i) p.shape[i] = s.n[i];
}

struct CmdRes {
  VkCommandPool pool{VK_NULL_HANDLE};
  VkCommandBuffer cmd{VK_NULL_HANDLE};
  VkQueryPool qpool{VK_NULL_HANDLE};
  uint32_t queries{0};
};

static CmdRes make_cmd(const DeviceCtx& ctx, uint32_t queries) {
  CmdRes r;
  r.queries = queries;
  VkCommandPoolCreateInfo cpci{};
  cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cpci.queueFamilyIndex = ctx.qfi;
  if (g_vk.CreateCommandPool(g_vk.dev, &cpci, nullptr, &r.pool) !=
      VK_SUCCESS)
    die("CreateCommandPool");
  VkCommandBufferAllocateInfo cbai{};
  cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbai.commandPool = r.pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  if (g_vk.AllocateCommandBuffers(g_vk.dev, &cbai, &r.cmd) != VK_SUCCESS)
    die("AllocateCommandBuffers");
  VkQueryPoolCreateInfo qpci{};
  qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
  qpci.queryCount = queries;
  if (g_vk.CreateQueryPool(g_vk.dev, &qpci, nullptr, &r.qpool) !=
      VK_SUCCESS)
    die("CreateQueryPool");
  return r;
}

static VkFence submit_and_wait(const DeviceCtx& ctx, VkCommandBuffer cmd) {
  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  VkFence fence;
  VkFenceCreateInfo fci{};
  fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  if (g_vk.CreateFence(g_vk.dev, &fci, nullptr, &fence) != VK_SUCCESS)
    die("CreateFence");
  if (g_vk.QueueSubmit(ctx.queue, 1, &si, fence) != VK_SUCCESS)
    die("QueueSubmit");
  if (g_vk.WaitForFences(g_vk.dev, 1, &fence, VK_TRUE, UINT64_MAX) !=
      VK_SUCCESS)
    die("WaitForFences");
  g_vk.DestroyFence(g_vk.dev, fence, nullptr);
  return VK_NULL_HANDLE;
}

static void begin(CmdRes& r) {
  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  g_vk.BeginCommandBuffer(r.cmd, &bi);
  g_vk.CmdResetQueryPool(r.cmd, r.qpool, 0, r.queries);
}

static uint64_t read_ticks(const DeviceCtx& ctx, CmdRes& r, uint32_t count,
    uint64_t* ticks) {
  if (g_vk.GetQueryPoolResults(g_vk.dev, r.qpool, 0, count,
          count * sizeof(uint64_t), ticks, sizeof(uint64_t),
          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS)
    die("GetQueryPoolResults");
  return ticks[0];
}

// One isolated dispatch per submit; returns gpu ns across the dispatch.
static uint64_t dispatch_isolated(const DeviceCtx& ctx, const Side& side,
    CmdRes& r, VkDescriptorSet set, const Params& params, uint32_t groups) {
  begin(r);
  g_vk.CmdWriteTimestamp(r.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, r.qpool, 0);
  g_vk.CmdBindPipeline(r.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, side.pipe);
  g_vk.CmdBindDescriptorSets(r.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
      side.layout, 0, 1, &set, 0, nullptr);
  g_vk.CmdPushConstants(r.cmd, side.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
      sizeof(Params), &params);
  g_vk.CmdDispatch(r.cmd, groups, 1, 1);
  g_vk.CmdWriteTimestamp(r.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, r.qpool, 1);
  g_vk.EndCommandBuffer(r.cmd);
  submit_and_wait(ctx, r.cmd);
  uint64_t ticks[2] = {0, 0};
  read_ticks(ctx, r, 2, ticks);
  return (uint64_t)((double)(ticks[1] - ticks[0]) * ctx.timestampPeriod);
}

// The four-dispatch layer chain in ONE command buffer, timestamped
// before the first dispatch and after each dispatch.
static uint64_t dispatch_chain(const DeviceCtx& ctx, const Side& side,
    CmdRes& r, VkDescriptorSet sets[kNumShapes], const Params params[kNumShapes],
    const uint32_t groups[kNumShapes], uint64_t per_dispatch_ns[kNumShapes]) {
  begin(r);
  g_vk.CmdWriteTimestamp(r.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, r.qpool, 0);
  for (uint32_t i = 0; i < kNumShapes; ++i) {
    g_vk.CmdBindPipeline(r.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, side.pipe);
    g_vk.CmdBindDescriptorSets(r.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        side.layout, 0, 1, &sets[i], 0, nullptr);
    g_vk.CmdPushConstants(r.cmd, side.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
        sizeof(Params), &params[i]);
    g_vk.CmdDispatch(r.cmd, groups[i], 1, 1);
    g_vk.CmdWriteTimestamp(
        r.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, r.qpool, 1 + i);
  }
  g_vk.EndCommandBuffer(r.cmd);
  submit_and_wait(ctx, r.cmd);
  uint64_t ticks[8] = {};
  read_ticks(ctx, r, 5, ticks);
  double period = ctx.timestampPeriod;
  // Timestamps: q[0] before the first dispatch, q[1+i] after dispatch
  // i; dispatch i's in-chain time is q[1+i] - q[i].
  for (uint32_t i = 0; i < kNumShapes; ++i) {
    per_dispatch_ns[i] =
        (uint64_t)((double)(ticks[1 + i] - ticks[i]) * period);
  }
  return (uint64_t)((double)(ticks[4] - ticks[0]) * period);
}

static uint64_t median(std::vector<uint64_t> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

static uint64_t min_of(const std::vector<uint64_t>& v) {
  return *std::min_element(v.begin(), v.end());
}

struct PeakPipe {
  VkShaderModule mod{VK_NULL_HANDLE};
  VkDescriptorSetLayout dsl{VK_NULL_HANDLE};
  VkPipelineLayout layout{VK_NULL_HANDLE};
  VkPipeline pipe{VK_NULL_HANDLE};
};

// Streaming probe: 2 bindings, a push-constant-free pipeline, one
// dispatch over uvec4_count elements.
static void run_peak(const DeviceCtx& ctx, const char* src_path,
    const char* spv, const char* tag, uint32_t uvec4_count, bool skip) {
  if (compile_shader(src_path, "", spv) != 0) die("compile %s", tag);
  PeakPipe p;
  p.mod = make_module(read_file(spv));
  VkDescriptorSetLayoutBinding b[2]{};
  for (uint32_t i = 0; i < 2; ++i) {
    b[i].binding = i;
    b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[i].descriptorCount = 1;
    b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dslci{};
  dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslci.bindingCount = 2;
  dslci.pBindings = b;
  if (g_vk.CreateDescriptorSetLayout(g_vk.dev, &dslci, nullptr, &p.dsl) !=
      VK_SUCCESS)
    die("CreateDescriptorSetLayout peak");
  VkPipelineLayoutCreateInfo plci{};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &p.dsl;
  if (g_vk.CreatePipelineLayout(g_vk.dev, &plci, nullptr, &p.layout) !=
      VK_SUCCESS)
    die("CreatePipelineLayout peak");
  VkComputePipelineCreateInfo cpci{};
  cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = p.mod;
  cpci.stage.pName = "main";
  cpci.layout = p.layout;
  if (g_vk.CreateComputePipelines(g_vk.dev, VK_NULL_HANDLE, 1, &cpci,
          nullptr, &p.pipe) != VK_SUCCESS)
    die("CreateComputePipelines peak");

  Buf a = make_buf(g_vk.dev, ctx.mp, (uint64_t)uvec4_count * 16u, false);
  Buf dst = make_buf(g_vk.dev, ctx.mp, (uint64_t)uvec4_count * 16u, true);
  VkDescriptorPoolSize ps{};
  ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  ps.descriptorCount = 2;
  VkDescriptorPoolCreateInfo dpci{};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpci.maxSets = 1;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &ps;
  VkDescriptorPool pool;
  if (g_vk.CreateDescriptorPool(g_vk.dev, &dpci, nullptr, &pool) !=
      VK_SUCCESS)
    die("CreateDescriptorPool peak");
  VkDescriptorSetAllocateInfo dsai{};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &p.dsl;
  VkDescriptorSet set;
  if (g_vk.AllocateDescriptorSets(g_vk.dev, &dsai, &set) != VK_SUCCESS)
    die("AllocateDescriptorSets peak");
  VkDescriptorBufferInfo dbi[2]{};
  dbi[0].buffer = a.buf;
  dbi[0].range = VK_WHOLE_SIZE;
  dbi[1].buffer = dst.buf;
  dbi[1].range = VK_WHOLE_SIZE;
  VkWriteDescriptorSet w[2]{};
  for (int i = 0; i < 2; ++i) {
    w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[i].dstSet = set;
    w[i].dstBinding = i;
    w[i].descriptorCount = 1;
    w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[i].pBufferInfo = &dbi[i];
  }
  g_vk.UpdateDescriptorSets(g_vk.dev, 2, w, 0, nullptr);

  CmdRes cmd = make_cmd(ctx, 2);
  uint32_t groups = uvec4_count / 256u;
  std::vector<uint64_t> samples;
  if (!skip) {
    for (int rep = 0; rep < 3; ++rep) {
      begin(cmd);
      g_vk.CmdWriteTimestamp(
          cmd.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, cmd.qpool, 0);
      g_vk.CmdBindPipeline(
          cmd.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
      g_vk.CmdBindDescriptorSets(cmd.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          p.layout, 0, 1, &set, 0, nullptr);
      g_vk.CmdDispatch(cmd.cmd, groups, 1, 1);
      g_vk.CmdWriteTimestamp(
          cmd.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, cmd.qpool, 1);
      g_vk.EndCommandBuffer(cmd.cmd);
      submit_and_wait(ctx, cmd.cmd);
      uint64_t ticks[2] = {0, 0};
      read_ticks(ctx, cmd, 2, ticks);
      samples.push_back(
          (uint64_t)((double)(ticks[1] - ticks[0]) * ctx.timestampPeriod));
    }
  }
  double bytes = (double)uvec4_count * 16.0 *
      (std::string(tag) == "copy" ? 2.0 : 1.0);
  if (samples.empty()) {
    std::printf("{\"k\":\"peak\",\"tag\":\"%s\",\"skipped\":true}\n", tag);
  } else {
    uint64_t med = median(samples);
    std::printf(
        "{\"k\":\"peak\",\"tag\":\"%s\",\"bytes\":%llu,\"med_gpu_ns\":%llu,"
        "\"min_gpu_ns\":%llu,\"med_gb_s\":%.2f,\"min_gb_s\":%.2f}\n",
        tag, (unsigned long long)bytes, (unsigned long long)med,
        (unsigned long long)min_of(samples), bytes / (double)med,
        bytes / (double)min_of(samples));
  }
  g_vk.DestroyDescriptorPool(g_vk.dev, pool, nullptr);
  g_vk.DestroyBuffer(g_vk.dev, a.buf, nullptr);
  g_vk.FreeMemory(g_vk.dev, a.mem, nullptr);
  g_vk.DestroyBuffer(g_vk.dev, dst.buf, nullptr);
  g_vk.FreeMemory(g_vk.dev, dst.mem, nullptr);
}

int main(int argc, char** argv) {
  bool tree_mode = false;
  bool quick = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--tree") tree_mode = true;
    if (std::string(argv[i]) == "--quick") quick = true;
  }
  const int reps = quick ? 7 : 21;

  const char* q4_defines =
      "-DUSE_FP16=1 -DQMM_VEC_Q4_WORD=1 -DQMM_VEC_MULTI=1";
  const char* variant_defines = tree_mode
      ? q4_defines
      : "-DUSE_FP16=1 -DUSE_SUBGROUP=1 -DQMM_VEC_Q4_WORD=1 "
        "-DQMM_VEC_MULTI=1";
  if (compile_shader("tools/q4-bw-bench/shaders/qmm_vec_base.comp",
          variant_defines, "/tmp/q4base.spv") != 0)
    die("compile base");
  if (compile_shader("tools/q4-bw-bench/shaders/qmm_vec_cand.comp",
          variant_defines, "/tmp/q4cand.spv") != 0)
    die("compile cand");

  if (vk_init() != 0) return 1;
  DeviceCtx ctx = setup_device();
  std::printf(
      "{\"k\":\"mode\",\"variant\":\"%s\",\"reps\":%d}\n",
      tree_mode ? "tree" : "subgroup", reps);

  Side base, cand;
  base.tag = "base";
  cand.tag = "cand";
  base.mod = make_module(read_file("/tmp/q4base.spv"));
  cand.mod = make_module(read_file("/tmp/q4cand.spv"));
  make_pipeline(base);
  make_pipeline(cand);

  // Input buffers shared by both sides; output buffers distinct so the
  // bit-exactness compare sees each side's own writes.
  SetBufs inputs[kNumShapes];
  SetBufs out_base[kNumShapes], out_cand[kNumShapes];
  Params shape_params[kNumShapes];
  uint32_t shape_groups_v[kNumShapes];
  for (uint32_t s = 0; s < kNumShapes; ++s) {
    const Shape& sh = kShapes[s];
    inputs[s].x = make_buf(g_vk.dev, ctx.mp, (uint64_t)sh.k * 2u, false);
    for (uint32_t d = 0; d < sh.dims; ++d) {
      inputs[s].w[d] = make_buf(g_vk.dev, ctx.mp,
          (uint64_t)sh.n[d] * (sh.k / 8u) * 4u, false);
      inputs[s].scales[d] = make_buf(g_vk.dev, ctx.mp,
          (uint64_t)sh.n[d] * (sh.k / 64u) * 2u, false);
      inputs[s].biases[d] = make_buf(g_vk.dev, ctx.mp,
          (uint64_t)sh.n[d] * (sh.k / 64u) * 2u, false);
    }
    for (uint32_t d = 0; d < kMaxDims; ++d) {
      uint32_t n = d < sh.dims ? sh.n[d] : sh.n[0];
      out_base[s].out[d] =
          make_buf(g_vk.dev, ctx.mp, (uint64_t)n * 2u, true);
      out_cand[s].out[d] =
          make_buf(g_vk.dev, ctx.mp, (uint64_t)n * 2u, true);
    }
    shape_groups_v[s] = shape_groups(sh);
    fill_params(shape_params[s], sh, shape_groups_v[s]);
  }

  VkDescriptorPoolSize ps{};
  ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  ps.descriptorCount = kBindings * kNumShapes * 2;
  VkDescriptorPoolCreateInfo dpci{};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpci.maxSets = kNumShapes * 2;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &ps;
  VkDescriptorPool pool;
  if (g_vk.CreateDescriptorPool(g_vk.dev, &dpci, nullptr, &pool) !=
      VK_SUCCESS)
    die("CreateDescriptorPool");
  VkDescriptorSet sets_base[kNumShapes], sets_cand[kNumShapes];
  for (uint32_t s = 0; s < kNumShapes; ++s) {
    SetBufs combined_base = inputs[s], combined_cand = inputs[s];
    for (uint32_t d = 0; d < kMaxDims; ++d) {
      combined_base.out[d] = out_base[s].out[d];
      combined_cand.out[d] = out_cand[s].out[d];
    }
    sets_base[s] =
        make_set(pool, base.dsl, combined_base, kShapes[s].dims);
    sets_cand[s] =
        make_set(pool, cand.dsl, combined_cand, kShapes[s].dims);
  }

  CmdRes iso_cmd = make_cmd(ctx, 2);
  CmdRes chain_cmd = make_cmd(ctx, 8);

  run_peak(ctx, "tools/q4-bw-bench/shaders/peak_copy.comp",
      "/tmp/peak_copy.spv", "copy", 1u << 24u, quick);
  run_peak(ctx, "tools/q4-bw-bench/shaders/peak_read.comp",
      "/tmp/peak_read.spv", "read", 1u << 24u, quick);

  // ---- Bit-exactness: identical inputs through base and cand ----
  for (uint32_t s = 0; s < kNumShapes; ++s) {
    const Shape& sh = kShapes[s];
    dispatch_isolated(ctx, base, iso_cmd, sets_base[s], shape_params[s],
        shape_groups_v[s]);
    for (uint32_t d = 0; d < sh.dims; ++d) {
      void* p;
      g_vk.MapMemory(g_vk.dev, out_base[s].out[d].mem, 0,
          out_base[s].out[d].size, 0, &p);
      // keep mapped until cand run done; snapshot instead
      std::vector<uint16_t> snap(out_base[s].out[d].size / 2);
      std::memcpy(snap.data(), p, out_base[s].out[d].size);
      g_vk.UnmapMemory(g_vk.dev, out_base[s].out[d].mem);

      dispatch_isolated(ctx, cand, iso_cmd, sets_cand[s], shape_params[s],
          shape_groups_v[s]);
      g_vk.MapMemory(g_vk.dev, out_cand[s].out[d].mem, 0,
          out_cand[s].out[d].size, 0, &p);
      uint32_t mismatches = 0;
      uint16_t* got = (uint16_t*)p;
      for (size_t i = 0; i < snap.size(); ++i) {
        if (got[i] != snap[i]) ++mismatches;
      }
      g_vk.UnmapMemory(g_vk.dev, out_cand[s].out[d].mem);
      std::printf(
          "{\"k\":\"eq\",\"shape\":\"%s\",\"dim\":%u,\"elements\":%zu,"
          "\"bit_mismatches\":%u}\n",
          sh.name, d, snap.size(), mismatches);
    }
  }

  // ---- Isolated per-shape timings, base and cand alternating ----
  for (uint32_t s = 0; s < kNumShapes; ++s) {
    const Shape& sh = kShapes[s];
    std::vector<uint64_t> bs, cs;
    for (int rep = 0; rep < reps; ++rep) {
      bs.push_back(dispatch_isolated(ctx, base, iso_cmd, sets_base[s],
          shape_params[s], shape_groups_v[s]));
      cs.push_back(dispatch_isolated(ctx, cand, iso_cmd, sets_cand[s],
          shape_params[s], shape_groups_v[s]));
    }
    uint64_t mb = median(bs), mc = median(cs);
    uint64_t bytes = shape_bytes(sh);
    std::printf(
        "{\"k\":\"shape\",\"name\":\"%s\",\"grid\":%u,\"bytes\":%llu,"
        "\"base_med_ns\":%llu,\"cand_med_ns\":%llu,"
        "\"base_min_ns\":%llu,\"cand_min_ns\":%llu,"
        "\"base_gb_s\":%.2f,\"cand_gb_s\":%.2f,\"ratio\":%.4f}\n",
        sh.name, shape_groups_v[s], (unsigned long long)bytes,
        (unsigned long long)mb, (unsigned long long)mc,
        (unsigned long long)min_of(bs), (unsigned long long)min_of(cs),
        (double)bytes / (double)mb, (double)bytes / (double)mc,
        (double)mc / (double)mb);
  }

  // ---- Chain timings, two interleaved rounds ----
  for (int round = 0; round < 2; ++round) {
    std::vector<uint64_t> bs, cs;
    std::vector<std::array<uint64_t, kNumShapes>> bd, cd;
    for (int rep = 0; rep < reps; ++rep) {
      uint64_t pd[kNumShapes];
      bs.push_back(dispatch_chain(ctx, base, chain_cmd, sets_base,
          shape_params, shape_groups_v, pd));
      bd.push_back({pd[0], pd[1], pd[2], pd[3]});
      cs.push_back(dispatch_chain(ctx, cand, chain_cmd, sets_cand,
          shape_params, shape_groups_v, pd));
      cd.push_back({pd[0], pd[1], pd[2], pd[3]});
    }
    uint64_t mb = median(bs), mc = median(cs);
    uint64_t bytes = 0;
    for (uint32_t s = 0; s < kNumShapes; ++s) bytes += shape_bytes(kShapes[s]);
    std::printf(
        "{\"k\":\"chain\",\"round\":%d,\"bytes\":%llu,"
        "\"base_med_ns\":%llu,\"cand_med_ns\":%llu,"
        "\"base_min_ns\":%llu,\"cand_min_ns\":%llu,"
        "\"base_gb_s\":%.2f,\"cand_gb_s\":%.2f,\"ratio\":%.4f}\n",
        round, (unsigned long long)bytes, (unsigned long long)mb,
        (unsigned long long)mc, (unsigned long long)min_of(bs),
        (unsigned long long)min_of(cs), (double)bytes / (double)mb,
        (double)bytes / (double)mc, (double)mc / (double)mb);
    if (round == 0) {
      std::printf("{\"k\":\"chain_dispatch\",\"base\":[");
      for (uint32_t s = 0; s < kNumShapes; ++s) {
        std::vector<uint64_t> col;
        for (auto& row : bd) col.push_back(row[s]);
        std::printf("%s%llu", s ? "," : "", (unsigned long long)median(col));
      }
      std::printf("],\"cand\":[");
      for (uint32_t s = 0; s < kNumShapes; ++s) {
        std::vector<uint64_t> col;
        for (auto& row : cd) col.push_back(row[s]);
        std::printf("%s%llu", s ? "," : "", (unsigned long long)median(col));
      }
      std::printf("]}\n");
    }
  }

  std::printf("{\"k\":\"done\"}\n");
  return 0;
}
