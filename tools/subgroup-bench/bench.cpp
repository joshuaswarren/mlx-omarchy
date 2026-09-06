// Microbenchmark: subgroupAdd vs five-round shared-memory float tree on
// 32-element reductions, with the equivalence check both produce the same
// 32-input sum that a CPU reference does. The benchmark exists to settle
// the float-versus-integer subgroup question the qmm_vec.comp and
// matmul_vec.comp source comments raise: the comments claim Honeykrisp
// lowers float subgroup arithmetic to software, but no receipt
// established that with measurement. This program gives one M1 leg an
// answer.
//
// Run:
//   ./bench                # equivalence check + timed legs, prints NDJSON
//   ./bench --quick        # equivalence + one short timed leg, no warmup
//
// The harness dlopens libvulkan at runtime (no link dependency). It
// compiles both compute shaders at startup via glslc if present, else
// glslangValidator. The two shaders are byte-identical except for the
// reduction body, so any timing difference is the subgroup-vs-tree
// difference, not dispatch or memory overhead.

#include <vulkan/vulkan.h>

#include <algorithm>
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
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#define LIBVK "libvulkan.so.1"

// ---------------------------------------------------------------------------
// dlopen table. Only functions guaranteed available pre-instance (per
// Vulkan loader spec) live in vk_init(); the rest load after the
// instance or device handle exists. This matches the omarchy backend's
// no-direct-link pattern and avoids the trap where some loaders (older
// Mesa Lavapipe, certain Android stacks) return NULL for instance-level
// functions queried with NULL instance.
// ---------------------------------------------------------------------------
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
  VK_FN(GetPhysicalDeviceFeatures2);
  VK_FN(GetPhysicalDeviceMemoryProperties);
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
  VK_FN(CmdCopyBuffer);
  VK_FN(CreateFence);
  VK_FN(DestroyFence);
  VK_FN(WaitForFences);
  VK_FN(CreateQueryPool);
  VK_FN(CmdResetQueryPool);
  VK_FN(CmdWriteTimestamp);
  VK_FN(GetQueryPoolResults);
  VK_FN(DestroyQueryPool);
  VK_FN(QueueSubmit);
  VK_FN(QueueWaitIdle);
  VK_FN(DestroyShaderModule);
  VK_FN(DestroyPipeline);
  VK_FN(DestroyPipelineLayout);
  VK_FN(DestroyDescriptorSetLayout);
  VK_FN(DestroyDescriptorPool);
  VK_FN(DestroyCommandPool);
  VK_FN(FreeCommandBuffers);
  VK_FN(DestroyBuffer);
  VK_FN(DestroyDevice);
#undef VK_FN
};

static VkTable g_vk;

// Loader dispatch: use vkGetDeviceProcAddr for device-level functions
// and vkGetInstanceProcAddr for instance-level functions. Both are
// guaranteed available post-device for the device-level set.
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
  // Pre-instance loader; only the spec-required names here.
  g_vk.CreateInstance =
      (PFN_vkCreateInstance)vk_load(nullptr, "vkCreateInstance");
  g_vk.EnumerateInstanceExtensionProperties =
      (PFN_vkEnumerateInstanceExtensionProperties)vk_load(
          nullptr, "vkEnumerateInstanceExtensionProperties");
  g_vk.EnumerateInstanceLayerProperties =
      (PFN_vkEnumerateInstanceLayerProperties)vk_load(
          nullptr, "vkEnumerateInstanceLayerProperties");
  if (!g_vk.CreateInstance) {
    std::fprintf(stderr, "vkCreateInstance missing\n");
    return -1;
  }
  // Some loaders also expose vkGetDeviceProcAddr pre-device, others do
  // not. We can dlsym it directly: the loader always exports it.
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
  LOAD(GetPhysicalDeviceFeatures2);
  LOAD(GetPhysicalDeviceMemoryProperties);
  LOAD(CreateDevice);  // instance-level; called pre-device
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
  LOAD_DEV(CmdCopyBuffer);
  LOAD_DEV(CreateFence);
  LOAD_DEV(DestroyFence);
  LOAD_DEV(WaitForFences);
  LOAD_DEV(CreateQueryPool);
  LOAD_DEV(CmdResetQueryPool);
  LOAD_DEV(CmdWriteTimestamp);
  LOAD_DEV(GetQueryPoolResults);
  LOAD_DEV(DestroyQueryPool);
  LOAD_DEV(QueueSubmit);
  LOAD_DEV(QueueWaitIdle);
  LOAD_DEV(DestroyShaderModule);
  LOAD_DEV(DestroyPipeline);
  LOAD_DEV(DestroyPipelineLayout);
  LOAD_DEV(DestroyDescriptorSetLayout);
  LOAD_DEV(DestroyDescriptorPool);
  LOAD_DEV(DestroyCommandPool);
  LOAD_DEV(FreeCommandBuffers);
  LOAD_DEV(DestroyBuffer);
  LOAD_DEV(DestroyDevice);
}

#undef LOAD
#undef LOAD_DEV

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static void die(const char* fmt, ...) {
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

// Compile one shader source file via glslc if present, else
// glslangValidator. The --target-env=vulkan1.3 flag is REQUIRED for
// the subgroup extension in reduce_subgroup.comp: subgroup operations
// need SPIR-V 1.3, and glslc without the flag rejects them with
// "'subgroup op' : requires SPIR-V 1.3" (this bit the M1 leg once -
// the harness used to print BENCH_DONE on this failure because the
// exit path was wrong). The repo's own omarchy_shader uses
// vulkan1.3; the bench matches.
//
// Returns 0 on success (and the SPIR-V exists and is non-empty), -1
// on any failure. The caller dies on -1.
static int compile_shader(
    const char* src,
    const char* out_spv,
    const std::string& defines = std::string()) {
  const char* cmd_template = nullptr;
  std::string cmd;
  if (system("which glslc >/dev/null 2>&1") == 0) {
    // -O / -Os match the production omarchy_shader compile flags; the
    // unoptimized SPIR-V of the bigger qmm_vec variants trips an old
    // lavapipe pipeline-creation crash, and matching production is
    // the point of the bench either way. 1>&2 keeps glslang's echoed
    // input filename off the NDJSON stdout stream.
    cmd = std::string("glslc -O -fshader-stage=compute "
                      "--target-env=vulkan1.3 ") +
        defines + " " + src + " -o " + out_spv + " 2>&1 1>&2";
  } else if (system("which glslangValidator >/dev/null 2>&1") == 0) {
    cmd = std::string("glslangValidator -V -Os --target-env vulkan1.3 ") +
        defines + " " + src + " -o " + out_spv + " 2>&1 1>&2";
  } else {
    die("neither glslc nor glslangValidator found in PATH");
  }
  int rc = system(cmd.c_str());
  if (rc != 0) {
    std::fprintf(stderr, "shader compile failed (rc=%d): %s\n", rc,
        cmd.c_str());
    return -1;
  }
  // Verify the SPIR-V output exists and is non-empty. A glslc that
  // exited 0 but wrote nothing is rare but possible; the bench
  // would otherwise dereference a zero-byte file.
  struct stat st;
  if (stat(out_spv, &st) != 0 || st.st_size == 0) {
    std::fprintf(stderr,
        "shader compile produced no SPIR-V: %s (size=%ld)\n",
        out_spv, (long)(stat(out_spv, &st) == 0 ? st.st_size : -1));
    return -1;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Buffer helpers
// ---------------------------------------------------------------------------
struct Buf {
  VkBuffer buf{VK_NULL_HANDLE};
  VkDeviceMemory mem{VK_NULL_HANDLE};
  uint32_t size{0};
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
    VkDeviceSize size, VkBufferUsageFlags usage, bool fill_random) {
  Buf b;
  b.size = (uint32_t)size;
  VkBufferCreateInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bi.size = size;
  bi.usage = usage;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (g_vk.CreateBuffer(dev, &bi, nullptr, &b.buf) != VK_SUCCESS)
    die("CreateBuffer");

  VkMemoryRequirements req;
  g_vk.GetBufferMemoryRequirements(dev, b.buf, &req);
  uint32_t mt = find_memtype(req.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
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

  if (fill_random) {
    void* p;
    g_vk.MapMemory(dev, b.mem, 0, size, 0, &p);
    std::mt19937 rng(0xC0FFEEu);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    float* f = (float*)p;
    for (uint32_t i = 0; i < size / sizeof(float); ++i) f[i] = dist(rng);
    g_vk.UnmapMemory(dev, b.mem);
  }
  return b;
}

// ---------------------------------------------------------------------------
// Pipeline construction (compute)
// ---------------------------------------------------------------------------
struct Ctx {
  VkPhysicalDevice pd{VK_NULL_HANDLE};
  VkQueue queue{VK_NULL_HANDLE};
  uint32_t qfi{0};
  VkPhysicalDeviceMemoryProperties mp{};
  VkPhysicalDeviceSubgroupProperties sub{};
  uint32_t subgroupSize{0};
  bool subgroupArithFloat{false};
  uint32_t timestampPeriod{0};
  bool f16_ready{false};
  uint32_t max_group_count{0};
};

static void setup_device(Ctx& c) {
  VkApplicationInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  ai.pApplicationName = "subgroup-bench";
  ai.applicationVersion = 1;
  ai.pEngineName = "subgroup-bench";
  ai.engineVersion = 1;
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
    PFN_vkGetPhysicalDeviceQueueFamilyProperties qfp =
        (PFN_vkGetPhysicalDeviceQueueFamilyProperties)vk_load(
            g_vk.inst, "vkGetPhysicalDeviceQueueFamilyProperties");
    if (!qfp) continue;
    qfp(pds[i], &qfn, nullptr);
    std::vector<VkQueueFamilyProperties> qfpv(qfn);
    qfp(pds[i], &qfn, qfpv.data());

    for (uint32_t q = 0; q < qfn; ++q) {
      if ((qfpv[q].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
      c.pd = pds[i];
      c.qfi = q;
      c.sub = sub;
      c.subgroupSize = sub.subgroupSize;
      c.subgroupArithFloat =
          (sub.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
      c.timestampPeriod = props.limits.timestampPeriod;
      c.max_group_count = props.limits.maxComputeWorkGroupCount[0];
      g_vk.GetPhysicalDeviceMemoryProperties(pds[i], &c.mp);

      // Query and enable features through the Features2 chain exactly
      // the way the production backend does (device.cpp). A bare
      // Vulkan12Features pNext with 16-bit storage hangs off it
      // corrupts this Mesa's lavapipe: device creation succeeds but
      // the first f16 pipeline compile segfaults. Same enables,
      // different chain - the chain is load-bearing.
      VkPhysicalDeviceVulkan13Features q13{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
      VkPhysicalDeviceVulkan12Features q12{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
      VkPhysicalDevice16BitStorageFeatures q16{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
      VkPhysicalDeviceFeatures2 q2{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
      q16.pNext = &q12;
      q12.pNext = &q13;
      q2.pNext = &q16;
      g_vk.GetPhysicalDeviceFeatures2(pds[i], &q2);
      c.f16_ready = q16.storageBuffer16BitAccess == VK_TRUE &&
          q12.shaderFloat16 == VK_TRUE;

      float prio = 1.0f;
      VkDeviceQueueCreateInfo qci{};
      qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      qci.queueFamilyIndex = q;
      qci.queueCount = 1;
      qci.pQueuePriorities = &prio;
      VkPhysicalDeviceVulkan13Features e13{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
      VkPhysicalDeviceVulkan12Features e12{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
      e12.shaderFloat16 = q12.shaderFloat16;
      VkPhysicalDevice16BitStorageFeatures e16{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
      e16.storageBuffer16BitAccess = q16.storageBuffer16BitAccess;
      e16.pNext = &e12;
      e12.pNext = &e13;
      VkDeviceCreateInfo dci{};
      dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
      dci.queueCreateInfoCount = 1;
      dci.pQueueCreateInfos = &qci;
      dci.pNext = &e16;
      if (g_vk.CreateDevice(pds[i], &dci, nullptr, &g_vk.dev) != VK_SUCCESS)
        die("CreateDevice");
      vk_load_device();
      g_vk.GetDeviceQueue(g_vk.dev, q, 0, &c.queue);
      std::printf(
          "{\"k\":\"dev\",\"name\":\"%s\",\"api\":%u.%u.%u,"
          "\"subgroupSize\":%u,\"arith\":%s,\"ts_period_ns\":%.3f}\n",
          props.deviceName, VK_VERSION_MAJOR(props.apiVersion),
          VK_VERSION_MINOR(props.apiVersion),
          VK_VERSION_PATCH(props.apiVersion), c.subgroupSize,
          c.subgroupArithFloat ? "true" : "false",
          c.timestampPeriod / 1.0);
      return;
    }
  }
  die("no Vulkan 1.2 compute-capable device");
}

static VkShaderModule make_module(VkDevice dev, const std::string& spv) {
  VkShaderModuleCreateInfo mi{};
  mi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  mi.codeSize = spv.size();
  mi.pCode = (const uint32_t*)spv.data();
  VkShaderModule m;
  if (g_vk.CreateShaderModule(dev, &mi, nullptr, &m) != VK_SUCCESS)
    die("CreateShaderModule");
  return m;
}

struct Pipeline {
  VkPipelineLayout layout{VK_NULL_HANDLE};
  VkPipeline pipe{VK_NULL_HANDLE};
  VkDescriptorSetLayout dsl{VK_NULL_HANDLE};
  VkDescriptorPool pool{VK_NULL_HANDLE};
  VkDescriptorSet set{VK_NULL_HANDLE};
};

static Pipeline make_pipeline(VkShaderModule mod, Buf in, Buf out) {
  Pipeline p;
  VkDescriptorSetLayoutBinding b[2]{};
  b[0].binding = 0;
  b[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  b[0].descriptorCount = 1;
  b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  b[1].binding = 1;
  b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  b[1].descriptorCount = 1;
  b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  VkDescriptorSetLayoutCreateInfo dslci{};
  dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslci.bindingCount = 2;
  dslci.pBindings = b;
  if (g_vk.CreateDescriptorSetLayout(g_vk.dev, &dslci, nullptr, &p.dsl) !=
      VK_SUCCESS) die("CreateDescriptorSetLayout");

  VkPipelineLayoutCreateInfo plci{};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &p.dsl;
  VkPushConstantRange pc{};
  pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pc.offset = 0;
  pc.size = sizeof(uint32_t);
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pc;
  if (g_vk.CreatePipelineLayout(g_vk.dev, &plci, nullptr, &p.layout) !=
      VK_SUCCESS) die("CreatePipelineLayout");

  VkComputePipelineCreateInfo cpci{};
  cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = mod;
  cpci.stage.pName = "main";
  cpci.layout = p.layout;
  if (g_vk.CreateComputePipelines(g_vk.dev, VK_NULL_HANDLE, 1, &cpci, nullptr,
      &p.pipe) != VK_SUCCESS) die("CreateComputePipelines");

  VkDescriptorPoolSize ps{};
  ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  ps.descriptorCount = 2;
  VkDescriptorPoolCreateInfo dpci{};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpci.maxSets = 1;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &ps;
  if (g_vk.CreateDescriptorPool(g_vk.dev, &dpci, nullptr, &p.pool) !=
      VK_SUCCESS) die("CreateDescriptorPool");

  VkDescriptorSetAllocateInfo dsai{};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = p.pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &p.dsl;
  if (g_vk.AllocateDescriptorSets(g_vk.dev, &dsai, &p.set) != VK_SUCCESS)
    die("AllocateDescriptorSets");

  VkDescriptorBufferInfo dbi[2]{};
  dbi[0].buffer = in.buf;
  dbi[0].offset = 0;
  dbi[0].range = in.size;
  dbi[1].buffer = out.buf;
  dbi[1].offset = 0;
  dbi[1].range = out.size;
  VkWriteDescriptorSet w[2]{};
  for (int i = 0; i < 2; ++i) {
    w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[i].dstSet = p.set;
    w[i].dstBinding = i;
    w[i].descriptorCount = 1;
    w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[i].pBufferInfo = &dbi[i];
  }
  g_vk.UpdateDescriptorSets(g_vk.dev, 2, w, 0, nullptr);
  return p;
}

// ---------------------------------------------------------------------------
// Dispatch + timing
// ---------------------------------------------------------------------------
struct CmdRes {
  VkCommandPool pool{VK_NULL_HANDLE};
  VkCommandBuffer cmd{VK_NULL_HANDLE};
  VkQueryPool qpool{VK_NULL_HANDLE};
};

static CmdRes make_cmd(uint32_t qfi) {
  CmdRes r;
  VkCommandPoolCreateInfo cpci{};
  cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cpci.queueFamilyIndex = qfi;
  if (g_vk.CreateCommandPool(g_vk.dev, &cpci, nullptr, &r.pool) != VK_SUCCESS)
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
  qpci.queryCount = 2;
  if (g_vk.CreateQueryPool(g_vk.dev, &qpci, nullptr, &r.qpool) != VK_SUCCESS)
    die("CreateQueryPool");
  return r;
}

struct TimedRun {
  uint64_t host_ns;
  uint64_t gpu_ns;
  int rc;
};

// Real dispatch — keeps the queue handle on the Ctx and uses it.
struct TimedRun2 {
  uint64_t host_ns;
  uint64_t gpu_ns;
  int rc;
};

static TimedRun dispatch(Ctx& c, Pipeline& p, CmdRes& r, uint32_t n_groups) {
  TimedRun t{};
  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  g_vk.BeginCommandBuffer(r.cmd, &bi);
  g_vk.CmdResetQueryPool(r.cmd, r.qpool, 0, 2);
  g_vk.CmdWriteTimestamp(r.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, r.qpool, 0);
  g_vk.CmdBindPipeline(r.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
  g_vk.CmdBindDescriptorSets(r.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0,
      1, &p.set, 0, nullptr);
  g_vk.CmdPushConstants(r.cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
      sizeof(uint32_t), &n_groups);
  g_vk.CmdDispatch(r.cmd, n_groups, 1, 1);
  g_vk.CmdWriteTimestamp(r.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, r.qpool, 1);
  if (g_vk.EndCommandBuffer(r.cmd) != VK_SUCCESS) die("EndCommandBuffer");

  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &r.cmd;

  VkFence fence;
  VkFenceCreateInfo fci{};
  fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  if (g_vk.CreateFence(g_vk.dev, &fci, nullptr, &fence) != VK_SUCCESS)
    die("CreateFence");

  auto t0 = std::chrono::steady_clock::now();
  if (g_vk.QueueSubmit(c.queue, 1, &si, fence) != VK_SUCCESS)
    die("QueueSubmit");
  if (g_vk.WaitForFences(g_vk.dev, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
    die("WaitForFences");
  auto t1 = std::chrono::steady_clock::now();
  t.host_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
      t1 - t0).count();

  uint64_t ticks[2] = {0, 0};
  VkResult qr = g_vk.GetQueryPoolResults(g_vk.dev, r.qpool, 0, 2, sizeof(ticks),
      ticks, sizeof(uint64_t),
      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
  if (qr != VK_SUCCESS) die("GetQueryPoolResults rc=%d", qr);
  // ticks[0] is the TOP timestamp, ticks[1] is the BOTTOM.
  double period_ns = c.timestampPeriod;
  t.gpu_ns = (uint64_t)((double)(ticks[1] - ticks[0]) * period_ns);
  t.rc = 0;
  g_vk.DestroyFence(g_vk.dev, fence, nullptr);
  return t;
}

// ---------------------------------------------------------------------------
// qmm_vec variant mode (--qmm-vec)
//
// Times the production decode GEMV kernel
// (overlay/mlx/backend/omarchy/shaders/qmm_vec.comp) compiled per
// MLX_OMARCHY_QMM_VEC_VARIANT define set, on the real Qwen2.5-0.5B
// decode shapes (K=896, N in {896, 4864, 151936}, bits 4, group 64,
// transposed layout), with GPU timestamps around the dispatch. This
// is the TOP-2 (qmm_vec kernel quality) pre-measurement: the M1 run
// is one script, and llvmpipe runs prove the harness end to end
// (timing meaningless there, per the repo hardware split).
//
// Weights are host-affine-quantized from U(-1,1) with the model's
// quantization (groups along k per output column), so the timing sees
// real quantized value distributions and the CPU reference (f64 dot
// over the exact stored f16/f32 operands) is the honest correctness
// gate. Variant 1 must be bit-identical to variant 0 (same per-lane
// arithmetic order); variants 2/3 get the derived reduction-order
// bound. Subgroup flavors are exact only where subgroupSize == 32
// (the M1); elsewhere their eq check is expected to fail and is
// reported as such, never silently passed.
// ---------------------------------------------------------------------------

// Mirror of the push-constant Params block in qmm_vec.comp (30 words).
struct QmmParams {
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

static float f16_to_f32(uint16_t h) {
  bool neg = (h & 0x8000u) != 0;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t man = h & 0x3FFu;
  float v;
  if (exp == 0) {
    v = (man == 0) ? 0.0f : std::ldexp(static_cast<float>(man), -24);
  } else if (exp == 31) {
    v = man ? std::nanf("") : INFINITY;
  } else {
    v = std::ldexp(static_cast<float>(man | 0x400u),
        static_cast<int>(exp) - 25);
  }
  return neg ? -v : v;
}

static uint16_t f32_to_f16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t biased = static_cast<int32_t>((x >> 23) & 0xFFu);
  uint32_t man = x & 0x7FFFFFu;
  if (biased == 0xFF) {
    return static_cast<uint16_t>(sign | 0x7C00u | (man ? 0x200u : 0u));
  }
  int32_t exp = biased - 127 + 15;
  if (exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00u);
  }
  if (exp <= 0) {
    if (exp < -10) {
      return static_cast<uint16_t>(sign);
    }
    man |= 0x800000u;
    int shift = 14 - exp;
    uint32_t half = man >> shift;
    uint32_t rem = man & ((1u << shift) - 1u);
    uint32_t mid = 1u << (shift - 1);
    if (rem > mid || (rem == mid && (half & 1u) != 0u)) {
      half++;
    }
    return static_cast<uint16_t>(sign | half);
  }
  uint32_t half = (static_cast<uint32_t>(exp) << 10) | (man >> 13);
  uint32_t rem = man & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (half & 1u) != 0u)) {
    half++;
  }
  return static_cast<uint16_t>(sign | half);
}

struct QmmLeg {
  const char* name;
  int variant;
  bool subgroup;
  std::string defines;
};

struct QmmPipeline {
  VkPipelineLayout layout{VK_NULL_HANDLE};
  VkPipeline pipe{VK_NULL_HANDLE};
  VkDescriptorSetLayout dsl{VK_NULL_HANDLE};
  VkDescriptorPool pool{VK_NULL_HANDLE};
  VkDescriptorSet set{VK_NULL_HANDLE};
};

static QmmPipeline make_qmm_pipeline(
    VkShaderModule mod,
    const Buf bufs[4]) {
  QmmPipeline p;
  VkDescriptorSetLayoutBinding b[4]{};
  for (uint32_t i = 0; i < 4; ++i) {
    b[i].binding = i;
    b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[i].descriptorCount = 1;
    b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dslci{};
  dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslci.bindingCount = 4;
  dslci.pBindings = b;
  if (g_vk.CreateDescriptorSetLayout(g_vk.dev, &dslci, nullptr, &p.dsl) !=
      VK_SUCCESS) {
    die("CreateDescriptorSetLayout(qmm)");
  }
  VkPipelineLayoutCreateInfo plci{};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &p.dsl;
  VkPushConstantRange pc{};
  pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pc.offset = 0;
  pc.size = sizeof(QmmParams);
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pc;
  if (g_vk.CreatePipelineLayout(g_vk.dev, &plci, nullptr, &p.layout) !=
      VK_SUCCESS) {
    die("CreatePipelineLayout(qmm)");
  }
  VkComputePipelineCreateInfo cpci{};
  cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.module = mod;
  cpci.stage.pName = "main";
  cpci.layout = p.layout;
  if (g_vk.CreateComputePipelines(g_vk.dev, VK_NULL_HANDLE, 1, &cpci,
          nullptr, &p.pipe) != VK_SUCCESS) {
    die("CreateComputePipelines(qmm)");
  }
  VkDescriptorPoolSize ps{};
  ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  ps.descriptorCount = 4;
  VkDescriptorPoolCreateInfo dpci{};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpci.maxSets = 1;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &ps;
  if (g_vk.CreateDescriptorPool(g_vk.dev, &dpci, nullptr, &p.pool) !=
      VK_SUCCESS) {
    die("CreateDescriptorPool(qmm)");
  }
  VkDescriptorSetAllocateInfo dsai{};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = p.pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &p.dsl;
  if (g_vk.AllocateDescriptorSets(g_vk.dev, &dsai, &p.set) != VK_SUCCESS) {
    die("AllocateDescriptorSets(qmm)");
  }
  VkDescriptorBufferInfo dbi[4]{};
  VkWriteDescriptorSet w[4]{};
  for (uint32_t i = 0; i < 4; ++i) {
    dbi[i].buffer = bufs[i].buf;
    dbi[i].offset = 0;
    dbi[i].range = bufs[i].size;
    w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[i].dstSet = p.set;
    w[i].dstBinding = i;
    w[i].descriptorCount = 1;
    w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[i].pBufferInfo = &dbi[i];
  }
  g_vk.UpdateDescriptorSets(g_vk.dev, 4, w, 0, nullptr);
  return p;
}

static void qmm_fill_params(QmmParams& params, uint32_t n, uint32_t k,
    uint32_t group_size, uint32_t bits) {
  std::memset(&params, 0, sizeof(params));
  params.operation = bits;
  params.reduce_size = group_size;
  params.output_size = n;
  params.matrix_m = 1;
  params.matrix_n = n;
  params.matrix_k = k;
  params.flags = 0; // transposed (nn.Linear) layout
  params.alpha = 1.0f;
  params.shape[0] = 1;        // batch
  params.shape[3] = n * k / group_size; // scale_total
}

// GPU-timestamped, fence-bracketed dispatch of `groups` workgroups,
// chunked by the device's maxComputeWorkGroupCount[0] (the K-split
// variant needs 151936 groups for the lm_head shape).
struct QmmTimed {
  uint64_t host_ns;
  uint64_t gpu_ns;
};

static QmmTimed qmm_dispatch(Ctx& c, QmmPipeline& p, CmdRes& r,
    const QmmParams& params, uint32_t groups) {
  QmmTimed t{};
  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  g_vk.BeginCommandBuffer(r.cmd, &bi);
  g_vk.CmdResetQueryPool(r.cmd, r.qpool, 0, 2);
  g_vk.CmdWriteTimestamp(
      r.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, r.qpool, 0);
  g_vk.CmdBindPipeline(r.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
  g_vk.CmdBindDescriptorSets(
      r.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &p.set, 0,
      nullptr);
  g_vk.CmdPushConstants(r.cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
      sizeof(QmmParams), &params);
  uint32_t done = 0;
  while (done < groups) {
    uint32_t chunk = groups - done;
    if (chunk > c.max_group_count) {
      chunk = c.max_group_count;
    }
    g_vk.CmdDispatch(r.cmd, chunk, 1, 1);
    done += chunk;
  }
  g_vk.CmdWriteTimestamp(
      r.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, r.qpool, 1);
  if (g_vk.EndCommandBuffer(r.cmd) != VK_SUCCESS) die("EndCommandBuffer");
  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &r.cmd;
  VkFence fence;
  VkFenceCreateInfo fci{};
  fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  if (g_vk.CreateFence(g_vk.dev, &fci, nullptr, &fence) != VK_SUCCESS) {
    die("CreateFence");
  }
  auto t0 = std::chrono::steady_clock::now();
  if (g_vk.QueueSubmit(c.queue, 1, &si, fence) != VK_SUCCESS) {
    die("QueueSubmit");
  }
  if (g_vk.WaitForFences(g_vk.dev, 1, &fence, VK_TRUE, UINT64_MAX) !=
      VK_SUCCESS) {
    die("WaitForFences");
  }
  auto t1 = std::chrono::steady_clock::now();
  t.host_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
      t1 - t0).count();
  uint64_t ticks[2] = {0, 0};
  if (g_vk.GetQueryPoolResults(g_vk.dev, r.qpool, 0, 2, sizeof(ticks), ticks,
          sizeof(uint64_t),
          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) !=
      VK_SUCCESS) {
    die("GetQueryPoolResults");
  }
  t.gpu_ns = (uint64_t)((double)(ticks[1] - ticks[0]) * c.timestampPeriod);
  g_vk.DestroyFence(g_vk.dev, fence, nullptr);
  return t;
}


// Storage-width helpers shared by the qmm child legs. The element
// width is the storage dtype the compiled legs read: f16 where the
// device supports 16-bit storage, f32 otherwise.
static void qmm_store_val(
    uint8_t* dst,
    size_t idx,
    float v,
    size_t width) {
  if (width == 2) {
    uint16_t h = f32_to_f16(v);
    std::memcpy(dst + idx * 2, &h, 2);
  } else {
    std::memcpy(dst + idx * 4, &v, 4);
  }
}

static float qmm_load_ref(
    const std::vector<uint8_t>& src,
    size_t idx,
    size_t width) {
  if (width == 2) {
    uint16_t h;
    std::memcpy(&h, src.data() + idx * 2, 2);
    return f16_to_f32(h);
  }
  float f;
  std::memcpy(&f, src.data() + idx * 4, 4);
  return f;
}

// One (shape, leg) measurement in a FORKED CHILD with a fresh Vulkan
// device. Rationale: this Mesa's lavapipe crashes in
// CreateComputePipelines for the bigger qmm_vec SPIR-V only when it
// runs late in a process that has already compiled other pipelines -
// process (verified with a standalone probe). Each (shape, leg) runs
// as a freshly EXEC'd process (re-exec via /proc/self/exe): a plain
// fork() child of this multithreaded Vulkan process inherits corrupt
// driver state and faults. A driver fault costs one leg, which the
// parent records by name instead of silently dropping.
static int qmm_leg_main(uint32_t N,
    bool quick,
    const char* spv_path,
    const char* v0_spv_path,
    const char* leg_name,
    int variant,
    bool subgroup,
    size_t W) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  if (vk_init() != 0) _exit(2);
  Ctx ctx;
  setup_device(ctx);

  const uint32_t K = 896, GS = 64, BITS = 4;
  const uint32_t PACK = 32u / BITS;
  const uint32_t WORDS_PER_ROW = K / PACK;
  const uint32_t SCALE_TOTAL = K / GS;
  const int kRepeats = quick ? 3 : 9;
  const int kWarmups = quick ? 1 : 3;

  // x is shared across shapes (same K); deterministic seeds, so every
  // child of the same shape sees identical inputs.
  Buf x_buf = make_buf(g_vk.dev, ctx.mp, (VkDeviceSize)K * W,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
  std::vector<float> x_f32(K);
  std::vector<uint8_t> x_bits((size_t)K * W);
  {
    std::mt19937 rng(0x51E5u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (uint32_t i = 0; i < K; ++i) {
      x_f32[i] = dist(rng);
      qmm_store_val(x_bits.data(), i, x_f32[i], W);
    }
    void* p;
    g_vk.MapMemory(g_vk.dev, x_buf.mem, 0, x_buf.size, 0, &p);
    std::memcpy(p, x_bits.data(), x_bits.size());
    g_vk.UnmapMemory(g_vk.dev, x_buf.mem);
  }

  // Quantize one [N, K] weight matrix the way mlx does (affine,
  // groups along k per output column, low nibble first).
  std::mt19937 rng(0xA11CEu + N);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> matrix((size_t)N * K);
  for (auto& v : matrix) v = dist(rng);
  std::vector<uint32_t> words((size_t)N * WORDS_PER_ROW, 0);
  std::vector<uint8_t> scales_bits((size_t)N * SCALE_TOTAL * W),
      biases_bits((size_t)N * SCALE_TOTAL * W);
  float n_bins = 15.0f;
  for (uint32_t n_col = 0; n_col < N; ++n_col) {
    for (uint32_t g = 0; g < SCALE_TOTAL; ++g) {
      float w_max = -INFINITY, w_min = INFINITY;
      for (uint32_t i = 0; i < GS; ++i) {
        float v = matrix[(size_t)n_col * K + g * GS + i];
        w_max = std::max(w_max, v);
        w_min = std::min(w_min, v);
      }
      bool min_dominant = std::fabs(w_min) > std::fabs(w_max);
      float scale = std::max((w_max - w_min) / n_bins, 1e-7f);
      if (!min_dominant) scale = -scale;
      float edge = min_dominant ? w_min : w_max;
      float q0 = std::round(edge / scale);
      if (q0 != 0.0f) scale = edge / q0;
      float bias = (q0 == 0.0f) ? 0.0f : edge;
      qmm_store_val(scales_bits.data(), (size_t)n_col * SCALE_TOTAL + g,
          scale, W);
      qmm_store_val(biases_bits.data(), (size_t)n_col * SCALE_TOTAL + g,
          bias, W);
      float s_use = qmm_load_ref(scales_bits,
          (size_t)n_col * SCALE_TOTAL + g, W);
      float b_use = qmm_load_ref(biases_bits,
          (size_t)n_col * SCALE_TOTAL + g, W);
      for (uint32_t i = 0; i < GS; ++i) {
        uint32_t kk = g * GS + i;
        float v = matrix[(size_t)n_col * K + kk];
        float q =
            std::clamp(std::round((v - b_use) / s_use), 0.0f, n_bins);
        words[(size_t)n_col * WORDS_PER_ROW + kk / PACK] |=
            ((uint32_t)q) << ((kk % PACK) * BITS);
      }
    }
  }
  // CPU f64 reference over the exact stored operands.
  std::vector<double> ref_d(N, 0.0);
  for (uint32_t n_col = 0; n_col < N; ++n_col) {
    double acc = 0.0;
    for (uint32_t kk = 0; kk < K; ++kk) {
      uint32_t code =
          (words[(size_t)n_col * WORDS_PER_ROW + kk / PACK] >>
              ((kk % PACK) * BITS)) &
          0xFu;
      double dequant =
          (double)code * qmm_load_ref(scales_bits,
              (size_t)n_col * SCALE_TOTAL + kk / GS, W) +
          qmm_load_ref(biases_bits,
              (size_t)n_col * SCALE_TOTAL + kk / GS, W);
      acc += (double)qmm_load_ref(x_bits, kk, W) * dequant;
    }
    ref_d[n_col] = acc;
  }
  double m_max = 0.0;
  for (uint32_t n_col = 0; n_col < N; ++n_col) {
    m_max = std::max(m_max, std::fabs(ref_d[n_col]));
  }

  Buf w_buf = make_buf(g_vk.dev, ctx.mp, (VkDeviceSize)words.size() * 4,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
  Buf sb_buf = make_buf(g_vk.dev, ctx.mp,
      (VkDeviceSize)2 * N * SCALE_TOTAL * W,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
  Buf out_buf = make_buf(g_vk.dev, ctx.mp, (VkDeviceSize)N * W,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
  {
    void* p;
    g_vk.MapMemory(g_vk.dev, w_buf.mem, 0, w_buf.size, 0, &p);
    std::memcpy(p, words.data(), words.size() * 4);
    g_vk.UnmapMemory(g_vk.dev, w_buf.mem);
    g_vk.MapMemory(g_vk.dev, sb_buf.mem, 0, sb_buf.size, 0, &p);
    std::memcpy(p, scales_bits.data(), scales_bits.size());
    std::memcpy((char*)p + scales_bits.size(), biases_bits.data(),
        biases_bits.size());
    g_vk.UnmapMemory(g_vk.dev, sb_buf.mem);
  }

  auto run_leg = [&](const char* path) {
    VkShaderModule mod = make_module(g_vk.dev, read_file(path));
    Buf bufs[4] = {x_buf, w_buf, sb_buf, out_buf};
    QmmPipeline pipe = make_qmm_pipeline(mod, bufs);
    CmdRes cmd = make_cmd(ctx.qfi);
    QmmParams params;
    qmm_fill_params(params, N, K, GS, BITS);
    uint32_t groups = variant == 3 ? N : (N + 7u) / 8u;
    qmm_dispatch(ctx, pipe, cmd, params, groups);
    std::vector<uint8_t> out_bits((size_t)N * W);
    {
      void* p;
      g_vk.MapMemory(g_vk.dev, out_buf.mem, 0, out_buf.size, 0, &p);
      std::memcpy(out_bits.data(), p, out_bits.size());
      g_vk.UnmapMemory(g_vk.dev, out_buf.mem);
    }
    g_vk.DestroyShaderModule(g_vk.dev, mod, nullptr);
    g_vk.DestroyPipeline(g_vk.dev, pipe.pipe, nullptr);
    g_vk.DestroyPipelineLayout(g_vk.dev, pipe.layout, nullptr);
    g_vk.DestroyDescriptorPool(g_vk.dev, pipe.pool, nullptr);
    g_vk.DestroyDescriptorSetLayout(g_vk.dev, pipe.dsl, nullptr);
    g_vk.DestroyCommandPool(g_vk.dev, cmd.pool, nullptr);
    g_vk.DestroyQueryPool(g_vk.dev, cmd.qpool, nullptr);
    return out_bits;
  };

  // v0 tree baseline on the same buffers (bit-equality anchor). For
  // leg 0 this is the same module as the target; still harmless.
  std::vector<uint8_t> v0_bits = run_leg(v0_spv_path);
  std::vector<uint8_t> target_bits = run_leg(spv_path);

  std::vector<float> v0_out(N), dev_out(N);
  for (uint32_t i = 0; i < N; ++i) {
    v0_out[i] = qmm_load_ref(v0_bits, i, W);
    dev_out[i] = qmm_load_ref(target_bits, i, W);
  }
  bool finite = true;
  double diff_ref = 0.0;
  for (uint32_t i = 0; i < N; ++i) {
    if (!std::isfinite(dev_out[i])) finite = false;
    diff_ref = std::max(diff_ref,
        std::fabs((double)dev_out[i] - ref_d[i]));
  }
  bool bit_eq_v0 = true;
  double diff_v0 = 0.0;
  for (uint32_t i = 0; i < N; ++i) {
    if (dev_out[i] != v0_out[i]) bit_eq_v0 = false;
    diff_v0 = std::max(diff_v0,
        std::fabs((double)dev_out[i] - (double)v0_out[i]));
  }
  bool sub_real = ctx.subgroupArithFloat && ctx.subgroupSize == 32;
  bool eq_meaningful = !subgroup || sub_real;
  double ops = 3.0 * K / 32.0 + 32.0 + (variant == 3 ? 8.0 : 0.0);
  double bound = std::max(
      ops * std::max(2.0 * m_max, 50.0) * 0.00000011920929, 1e-6);
  bool eq_ok = finite && diff_ref <= bound &&
      (variant == 1 ? bit_eq_v0 : true);
  std::printf(
      "{\"k\":\"qmm_eq\",\"N\":%u,\"leg\":\"%s\",\"finite\":%s,"
      "\"max_diff_ref\":%.6e,\"bound\":%.6e,\"bit_eq_v0\":%s,"
      "\"max_diff_v0\":%.6e,\"eq_meaningful\":%s,\"eq_ok\":%s}\n",
      N, leg_name, finite ? "true" : "false", diff_ref, bound,
      bit_eq_v0 ? "true" : "false", diff_v0,
      eq_meaningful ? "true" : "false", eq_ok ? "true" : "false");

  // Timed leg: the target pipeline again (rebuild it - run_leg tears
  // modules down), warmups then median of kRepeats GPU-timestamped
  // dispatches.
  {
    VkShaderModule mod = make_module(g_vk.dev, read_file(spv_path));
    Buf bufs[4] = {x_buf, w_buf, sb_buf, out_buf};
    QmmPipeline pipe = make_qmm_pipeline(mod, bufs);
    CmdRes cmd = make_cmd(ctx.qfi);
    QmmParams params;
    qmm_fill_params(params, N, K, GS, BITS);
    uint32_t groups = variant == 3 ? N : (N + 7u) / 8u;
    for (int i = 0; i < kWarmups; ++i) {
      qmm_dispatch(ctx, pipe, cmd, params, groups);
    }
    std::vector<uint64_t> gpu_samples, host_samples;
    for (int i = 0; i < kRepeats; ++i) {
      QmmTimed t = qmm_dispatch(ctx, pipe, cmd, params, groups);
      gpu_samples.push_back(t.gpu_ns);
      host_samples.push_back(t.host_ns);
    }
    std::sort(gpu_samples.begin(), gpu_samples.end());
    std::sort(host_samples.begin(), host_samples.end());
    std::printf(
        "{\"k\":\"qmm\",\"N\":%u,\"leg\":\"%s\",\"variant\":%d,"
        "\"sub\":%s,\"groups\":%u,\"gpu_ns_med\":%" PRIu64
        ",\"host_ns_med\":%" PRIu64 ",\"gpu_ns_us\":%.1f}\n",
        N, leg_name, variant, subgroup ? "true" : "false", groups,
        gpu_samples[gpu_samples.size() / 2],
        host_samples[host_samples.size() / 2],
        gpu_samples[gpu_samples.size() / 2] / 1000.0);
  }
  std::fflush(stdout);
  return 0;
}

struct QmmLegInfo {
  const char* name;
  int variant;
  bool subgroup;
  std::string defines;
  std::string spv;
};

static std::vector<QmmLegInfo> qmm_build_legs(
    const std::string& /*storage_def*/) {
  std::vector<QmmLegInfo> legs = {
      {"v0_tree", 0, false, "", ""},
      {"v1_lowpress_tree", 1, false, "-DQMM_VEC_STAGED_X=1", ""},
      {"v2_wordpack_tree", 2, false, "-DQMM_VEC_WORDPACK=1", ""},
      {"v3_ksplit_tree", 3, false, "-DQMM_VEC_KSPLIT=8", ""},
      {"v0_sub", 0, true, "-DUSE_SUBGROUP=1", ""},
      {"v1_lowpress_sub", 1, true, "-DUSE_SUBGROUP=1 -DQMM_VEC_STAGED_X=1", ""},
      {"v2_wordpack_sub", 2, true, "-DUSE_SUBGROUP=1 -DQMM_VEC_WORDPACK=1", ""},
      {"v3_ksplit_sub", 3, true, "-DUSE_SUBGROUP=1 -DQMM_VEC_KSPLIT=8", ""},
  };
  for (auto& leg : legs) {
    leg.spv = std::string("/tmp/qmm_") + leg.name + ".spv";
  }
  return legs;
}

// The probe process: prints the device line and capability notes,
// leaves the storage width and subgroup flag for the parent.
static int qmm_probe_main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  if (vk_init() != 0) return 2;
  Ctx ctx;
  setup_device(ctx);
  bool sub_real = ctx.subgroupArithFloat && ctx.subgroupSize == 32;
  if (!ctx.f16_ready) {
    std::printf(
        "{\"k\":\"qmm_note\",\"msg\":\"device lacks 16-bit storage "
        "or shaderFloat16; running f32 legs\"}\n");
  }
  if (!sub_real) {
    std::printf(
        "{\"k\":\"qmm_note\",\"msg\":\"subgroupSize != 32 or no "
        "ARITHMETIC: subgroup-flavor outputs are structurally "
        "partial sums on this device; eq flags are expected false "
        "and timing is structural only\"}\n");
  }
  FILE* info = std::fopen("/tmp/qmm_devinfo.txt", "w");
  if (info) {
    std::fprintf(info, "%d %d\n", ctx.f16_ready ? 2 : 4,
        sub_real ? 1 : 0);
    std::fclose(info);
  }
  std::fflush(stdout);
  return 0;
}

static int qmm_vec_run(bool quick) {
  const char* qmm_src = "overlay/mlx/backend/omarchy/shaders/qmm_vec.comp";

  // One probe exec owns the device-capability lines; the parent
  // stays device-free so a later driver fault cannot take the whole
  // run down.
  std::fflush(stdout);
  pid_t probe = fork();
  if (probe == 0) {
    execl("/proc/self/exe", "subgroup-bench", "--qmm-probe",
        static_cast<char*>(nullptr));
    _exit(127);
  }
  int probe_status = 0;
  waitpid(probe, &probe_status, 0);
  if (!WIFEXITED(probe_status) || WEXITSTATUS(probe_status) != 0) {
    die("device probe child failed (rc=%d)", probe_status);
  }
  size_t W = 4;
  int sub_real = 0;
  {
    std::ifstream f("/tmp/qmm_devinfo.txt");
    if (!(f >> W >> sub_real)) die("device info file missing");
  }
  if (!sub_real) {
    std::printf(
        "{\"k\":\"qmm_note\",\"msg\":\"timed subgroup legs are "
        "structural only on this device\"}\n");
    std::fflush(stdout);
  }

  std::vector<QmmLegInfo> legs = qmm_build_legs(
      W == 2 ? "-DUSE_FP16=1 " : "");
  const std::string storage_def = W == 2 ? "-DUSE_FP16=1 " : "";
  for (auto& leg : legs) {
    if (compile_shader(qmm_src, leg.spv.c_str(),
            storage_def + leg.defines) != 0) {
      die("compile %s", leg.name);
    }
  }

  const uint32_t shapes[] = {896, 4864, 151936};
  const std::string v0_spv = legs[0].spv;
  for (uint32_t N : shapes) {
    std::printf(
        "{\"k\":\"qmm_shape\",\"N\":%u,\"K\":896,\"group\":64,"
        "\"bits\":4,\"layout\":\"transposed\"}\n",
        N);
    std::fflush(stdout);
    for (auto& leg : legs) {
      char n_buf[16], l_buf[16], q_buf[4];
      std::snprintf(n_buf, sizeof(n_buf), "%u", N);
      std::snprintf(l_buf, sizeof(l_buf), "%zu",
          static_cast<size_t>(&leg - legs.data()));
      std::snprintf(q_buf, sizeof(q_buf), "%d", quick ? 1 : 0);
      std::fflush(stdout);
      pid_t pid = fork();
      if (pid == 0) {
        execl("/proc/self/exe", "subgroup-bench", "--qmm-child", n_buf,
            l_buf, q_buf, static_cast<char*>(nullptr));
        _exit(127);
      }
      int status = 0;
      waitpid(pid, &status, 0);
      if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        std::printf(
            "{\"k\":\"qmm_leg_fault\",\"N\":%u,\"leg\":\"%s\","
            "\"status\":%d,\"note\":\"driver fault in child; leg "
            "recorded, not silently dropped\"}\n",
            N, leg.name, status);
        std::fflush(stdout);
      }
    }
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  bool quick = false;
  bool qmm = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--quick") quick = true;
    if (arg == "--qmm-vec") qmm = true;
  }
  if (qmm) {
    return qmm_vec_run(quick);
  }
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--qmm-probe") {
      return qmm_probe_main();
    }
    if (arg == "--qmm-child" && i + 3 < argc) {
      uint32_t n = static_cast<uint32_t>(std::atoi(argv[i + 1]));
      size_t leg_idx = static_cast<size_t>(std::atoi(argv[i + 2]));
      bool child_quick = std::atoi(argv[i + 3]) != 0;
      size_t W = 4;
      {
        std::ifstream f("/tmp/qmm_devinfo.txt");
        int w_int = 4, sub = 0;
        if (f >> w_int >> sub) W = static_cast<size_t>(w_int);
      }
      auto legs = qmm_build_legs(W == 2 ? "-DUSE_FP16=1 " : "");
      if (leg_idx >= legs.size()) return 3;
      auto& leg = legs[leg_idx];
      return qmm_leg_main(n, child_quick, leg.spv.c_str(),
          legs[0].spv.c_str(), leg.name, leg.variant, leg.subgroup, W);
    }
  }

  std::string sub_spv_path = "/tmp/sub.spv";
  std::string tree_spv_path = "/tmp/tree.spv";
  if (compile_shader("tools/subgroup-bench/shaders/reduce_subgroup.comp",
        sub_spv_path.c_str()) != 0) die("compile subgroup");
  if (compile_shader("tools/subgroup-bench/shaders/reduce_tree.comp",
        tree_spv_path.c_str()) != 0) die("compile tree");

  if (vk_init() != 0) return 1;

  Ctx ctx;
  // Add queue and patch setup_device to store it on ctx.
  setup_device(ctx);

  auto sub_spv = read_file(sub_spv_path.c_str());
  auto tree_spv = read_file(tree_spv_path.c_str());

  VkShaderModule sub_mod = make_module(g_vk.dev, sub_spv);
  VkShaderModule tree_mod = make_module(g_vk.dev, tree_spv);

  // ---- Equivalence check ----
  const uint32_t kEqGroups = 256;
  Buf eq_in = make_buf(g_vk.dev, ctx.mp, kEqGroups * 32 * sizeof(float),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
  Buf eq_sub = make_buf(g_vk.dev, ctx.mp, kEqGroups * sizeof(float),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
  Buf eq_tree = make_buf(g_vk.dev, ctx.mp, kEqGroups * sizeof(float),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);

  Pipeline sub_pipe = make_pipeline(sub_mod, eq_in, eq_sub);
  Pipeline tree_pipe = make_pipeline(tree_mod, eq_in, eq_tree);
  CmdRes eq_cmd = make_cmd(ctx.qfi);

  dispatch(ctx, sub_pipe, eq_cmd, kEqGroups);
  dispatch(ctx, tree_pipe, eq_cmd, kEqGroups);

  float host_in[kEqGroups * 32];
  {
    void* p;
    g_vk.MapMemory(g_vk.dev, eq_in.mem, 0, eq_in.size, 0, &p);
    std::memcpy(host_in, p, eq_in.size);
    g_vk.UnmapMemory(g_vk.dev, eq_in.mem);
  }
  float sub_out[kEqGroups], tree_out[kEqGroups];
  {
    void* p;
    g_vk.MapMemory(g_vk.dev, eq_sub.mem, 0, eq_sub.size, 0, &p);
    std::memcpy(sub_out, p, eq_sub.size);
    g_vk.UnmapMemory(g_vk.dev, eq_sub.mem);
  }
  {
    void* p;
    g_vk.MapMemory(g_vk.dev, eq_tree.mem, 0, eq_tree.size, 0, &p);
    std::memcpy(tree_out, p, eq_tree.size);
    g_vk.UnmapMemory(g_vk.dev, eq_tree.mem);
  }

  int mismatches = 0;
  float max_diff = 0.0f;
  for (uint32_t g = 0; g < kEqGroups; ++g) {
    float ref = 0.0f;
    for (uint32_t i = 0; i < 32; ++i) ref += host_in[g * 32 + i];
    float d_sub = std::fabs(sub_out[g] - ref);
    float d_tree = std::fabs(tree_out[g] - ref);
    max_diff = std::max(max_diff, std::max(d_sub, d_tree));
    if (sub_out[g] != tree_out[g]) {
      if (mismatches < 5) {
        std::fprintf(stderr,
            "MISMATCH group=%u ref=%.9g sub=%.9g tree=%.9g "
            "(dsub=%.3g dtree=%.3g)\n",
            g, ref, sub_out[g], tree_out[g], d_sub, d_tree);
      }
      ++mismatches;
    }
  }
  std::printf(
      "{\"k\":\"eq\",\"groups\":%u,\"max_diff\":%.6e,\"mismatches\":%d}\n",
      kEqGroups, max_diff, mismatches);
  // Equivalence failure is recorded but non-fatal: it is the
  // expected outcome on a device where subgroupSize != 32, which the
  // leg below also gates. On the M1 (subgroupSize=32) it must be 0
  // mismatches; if not, that is the receipt's answer about subgroup
  // correctness on this hardware. The timed leg still runs.

  // ---- Timed legs ----
  const uint32_t kBenchGroups = quick ? 4096u : 65536u;
  Buf big_in = make_buf(g_vk.dev, ctx.mp, kBenchGroups * 32 * sizeof(float),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
  Buf big_sub = make_buf(g_vk.dev, ctx.mp, kBenchGroups * sizeof(float),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
  Buf big_tree = make_buf(g_vk.dev, ctx.mp, kBenchGroups * sizeof(float),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
  Pipeline big_sub_pipe = make_pipeline(sub_mod, big_in, big_sub);
  Pipeline big_tree_pipe = make_pipeline(tree_mod, big_in, big_tree);
  CmdRes big_cmd = make_cmd(ctx.qfi);

  bool subgroup_leg = ctx.subgroupArithFloat && ctx.subgroupSize == 32;
  if (!ctx.subgroupArithFloat) {
    std::printf(
        "{\"k\":\"subgroup_skip\",\"reason\":\"no ARITHMETIC in "
        "supportedOperations\"}\n");
  } else if (ctx.subgroupSize != 32) {
    std::printf(
        "{\"k\":\"subgroup_skip\",\"reason\":\"subgroupSize=%u not 32; "
        "subgroup variant assumes one slot equals one hardware subgroup\","
        "\"subgroupSize\":%u}\n",
        ctx.subgroupSize, ctx.subgroupSize);
  } else if (subgroup_leg) {
    dispatch(ctx, big_sub_pipe, big_cmd, kBenchGroups);
    dispatch(ctx, big_tree_pipe, big_cmd, kBenchGroups);
    if (!quick) {
      dispatch(ctx, big_sub_pipe, big_cmd, kBenchGroups);
      dispatch(ctx, big_tree_pipe, big_cmd, kBenchGroups);
    }

    int kRepeats = quick ? 3 : 7;
    std::vector<uint64_t> sub_host(kRepeats), sub_gpu(kRepeats);
    std::vector<uint64_t> tree_host(kRepeats), tree_gpu(kRepeats);
    for (int i = 0; i < kRepeats; ++i) {
      auto r = dispatch(ctx, big_sub_pipe, big_cmd, kBenchGroups);
      sub_host[i] = r.host_ns;
      sub_gpu[i] = r.gpu_ns;
      r = dispatch(ctx, big_tree_pipe, big_cmd, kBenchGroups);
      tree_host[i] = r.host_ns;
      tree_gpu[i] = r.gpu_ns;
    }
    auto med = [](const std::vector<uint64_t>& v) {
      std::vector<uint64_t> s = v;
      std::sort(s.begin(), s.end());
      return s[s.size() / 2];
    };
    std::printf(
        "{\"k\":\"leg\",\"groups\":%u,\"sub_host_ns\":%" PRIu64
        ",\"sub_gpu_ns\":%" PRIu64 ",\"tree_host_ns\":%" PRIu64
        ",\"tree_gpu_ns\":%" PRIu64 ",\"ratio_gpu\":%.3f}\n",
        kBenchGroups, med(sub_host), med(sub_gpu), med(tree_host),
        med(tree_gpu),
        (double)med(sub_gpu) / (double)med(tree_gpu));
  }

  g_vk.DestroyShaderModule(g_vk.dev, sub_mod, nullptr);
  g_vk.DestroyShaderModule(g_vk.dev, tree_mod, nullptr);
  return 0;
}
