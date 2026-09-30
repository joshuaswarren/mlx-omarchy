// Level-batch projection microbench for jw16 (Apple M1 Max, Honeykrisp).
//
// Discriminates how much of the per-barrier drain is EXPOSED wall-clock when
// kernels have decode-like duration, by timing the same 226 RMW kernels under
// three dependency encodings (mirrors the Jw16BarrierElide census shapes):
//
//   chain   226 dispatches on ONE slice (true RAW), full MEMORY/ALL_COMMANDS
//           barrier between every consecutive pair — the recorded-order
//           analogue (226.1 emitted barriers/token, 99.8% immediate pred).
//   levels  the same 226 dispatches on distinct slices as 42 levels
//           (16x6 + 26x5), one full barrier between levels plus a leading
//           one — the level-batched analogue (41.8 levels/token).
//   unord   the same 226 dispatches, distinct slices, NO barriers — the
//           in-queue floor reference.
//
// Per-kernel duration is calibrated at runtime (default targets ~3 us and
// ~31 us, the decode stream mean) by scaling the ALU loop so the unord
// wall/226 hits the target; the achieved value is reported, not the target.
// Wall = CLOCK_MONOTONIC around QueueSubmit..QueueWaitIdle. Cases interleave
// c/l/u within every rep so thermal drift is common-mode.
//
// NDJSON on stdout. Run inside a GPU window on jw16:
//   bash /var/tmp/appbar/gpuwin.sh 'bash /var/tmp/lb1/run.sh'
// Build: g++ -std=c++17 -O2 -o level-batch-bench bench.cpp -ldl

#include <dlfcn.h>
#include <vulkan/vulkan.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static void* vk_lib = nullptr;
static PFN_vkGetInstanceProcAddr g_gipa = nullptr;

struct Vk {
  VkInstance inst{VK_NULL_HANDLE};
  VkDevice dev{VK_NULL_HANDLE};
  VkQueue queue{VK_NULL_HANDLE};
  PFN_vkCreateInstance CreateInstance{nullptr};
  PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices{nullptr};
  PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties{nullptr};
  PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties{
      nullptr};
  PFN_vkGetPhysicalDeviceQueueFamilyProperties
      GetPhysicalDeviceQueueFamilyProperties{nullptr};
  PFN_vkCreateDevice CreateDevice{nullptr};
  PFN_vkGetDeviceProcAddr GetDeviceProcAddr{nullptr};
  PFN_vkGetDeviceQueue GetDeviceQueue{nullptr};
  PFN_vkCreateBuffer CreateBuffer{nullptr};
  PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements{nullptr};
  PFN_vkBindBufferMemory BindBufferMemory{nullptr};
  PFN_vkAllocateMemory AllocateMemory{nullptr};
  PFN_vkMapMemory MapMemory{nullptr};
  PFN_vkCreateShaderModule CreateShaderModule{nullptr};
  PFN_vkCreateComputePipelines CreateComputePipelines{nullptr};
  PFN_vkCreatePipelineLayout CreatePipelineLayout{nullptr};
  PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout{nullptr};
  PFN_vkCreateDescriptorPool CreateDescriptorPool{nullptr};
  PFN_vkAllocateDescriptorSets AllocateDescriptorSets{nullptr};
  PFN_vkUpdateDescriptorSets UpdateDescriptorSets{nullptr};
  PFN_vkCreateCommandPool CreateCommandPool{nullptr};
  PFN_vkAllocateCommandBuffers AllocateCommandBuffers{nullptr};
  PFN_vkBeginCommandBuffer BeginCommandBuffer{nullptr};
  PFN_vkEndCommandBuffer EndCommandBuffer{nullptr};
  PFN_vkCmdBindPipeline CmdBindPipeline{nullptr};
  PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets{nullptr};
  PFN_vkCmdDispatch CmdDispatch{nullptr};
  PFN_vkCmdPushConstants CmdPushConstants{nullptr};
  PFN_vkCmdPipelineBarrier CmdPipelineBarrier{nullptr};
  PFN_vkQueueSubmit QueueSubmit{nullptr};
  PFN_vkQueueWaitIdle QueueWaitIdle{nullptr};
};
static Vk g_vk;

struct Ctx {
  uint32_t qfi{0};
  const char* name{""};
  VkPhysicalDeviceMemoryProperties mp{};
};
static Ctx ctx;

[[noreturn]] static void die(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "level-batch-bench: ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
  exit(1);
}

static double now_us() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static uint32_t find_memtype(uint32_t bits, VkMemoryPropertyFlags want) {
  for (uint32_t i = 0; i < ctx.mp.memoryTypeCount; ++i) {
    if ((bits & (1u << i)) &&
        (ctx.mp.memoryTypes[i].propertyFlags & want) == want)
      return i;
  }
  return UINT32_MAX;
}

static void setup_device() {
  vk_lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!vk_lib) die("dlopen libvulkan.so.1: %s", dlerror());
  g_gipa = (PFN_vkGetInstanceProcAddr)dlsym(vk_lib, "vkGetInstanceProcAddr");
  if (!g_gipa) die("no vkGetInstanceProcAddr");
  g_vk.CreateInstance =
      (PFN_vkCreateInstance)g_gipa(nullptr, "vkCreateInstance");

  VkApplicationInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  ai.pApplicationName = "level-batch-bench";
  ai.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo ici{};
  ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ici.pApplicationInfo = &ai;
  if (g_vk.CreateInstance(&ici, nullptr, &g_vk.inst) != VK_SUCCESS)
    die("CreateInstance");
  g_vk.GetDeviceProcAddr =
      (PFN_vkGetDeviceProcAddr)g_gipa(g_vk.inst, "vkGetDeviceProcAddr");
  g_vk.EnumeratePhysicalDevices =
      (PFN_vkEnumeratePhysicalDevices)g_gipa(
          g_vk.inst, "vkEnumeratePhysicalDevices");
  g_vk.GetPhysicalDeviceProperties =
      (PFN_vkGetPhysicalDeviceProperties)g_gipa(
          g_vk.inst, "vkGetPhysicalDeviceProperties");
  g_vk.GetPhysicalDeviceMemoryProperties =
      (PFN_vkGetPhysicalDeviceMemoryProperties)g_gipa(
          g_vk.inst, "vkGetPhysicalDeviceMemoryProperties");
  g_vk.GetPhysicalDeviceQueueFamilyProperties =
      (PFN_vkGetPhysicalDeviceQueueFamilyProperties)g_gipa(
          g_vk.inst, "vkGetPhysicalDeviceQueueFamilyProperties");
  g_vk.CreateDevice = (PFN_vkCreateDevice)g_gipa(g_vk.inst, "vkCreateDevice");

  uint32_t n = 0;
  g_vk.EnumeratePhysicalDevices(g_vk.inst, &n, nullptr);
  if (n == 0) die("no physical devices");
  std::vector<VkPhysicalDevice> pds(n);
  g_vk.EnumeratePhysicalDevices(g_vk.inst, &n, pds.data());
  for (uint32_t i = 0; i < n; ++i) {
    VkPhysicalDeviceProperties props{};
    g_vk.GetPhysicalDeviceProperties(pds[i], &props);
    if (props.apiVersion < VK_API_VERSION_1_2) continue;
    if (!strstr(props.deviceName, "Apple")) continue;
    uint32_t qfn = 0;
    g_vk.GetPhysicalDeviceQueueFamilyProperties(pds[i], &qfn, nullptr);
    std::vector<VkQueueFamilyProperties> qfpv(qfn);
    g_vk.GetPhysicalDeviceQueueFamilyProperties(pds[i], &qfn, qfpv.data());
    for (uint32_t q = 0; q < qfn; ++q) {
      if ((qfpv[q].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
      ctx.qfi = q;
      ctx.name = props.deviceName;
      g_vk.GetPhysicalDeviceMemoryProperties(pds[i], &ctx.mp);
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
      if (g_vk.CreateDevice(pds[i], &dci, nullptr, &g_vk.dev) != VK_SUCCESS)
        die("CreateDevice");
      break;
    }
    if (g_vk.dev != VK_NULL_HANDLE) break;
  }
  if (g_vk.dev == VK_NULL_HANDLE) die("no Apple Vulkan 1.2 compute device");

#define DEV(fn) \
  g_vk.fn = (PFN_vk##fn)g_vk.GetDeviceProcAddr(g_vk.dev, "vk" #fn)
  DEV(GetDeviceQueue);
  DEV(CreateBuffer);
  DEV(GetBufferMemoryRequirements);
  DEV(BindBufferMemory);
  DEV(AllocateMemory);
  DEV(MapMemory);
  DEV(CreateShaderModule);
  DEV(CreateComputePipelines);
  DEV(CreatePipelineLayout);
  DEV(CreateDescriptorSetLayout);
  DEV(CreateDescriptorPool);
  DEV(AllocateDescriptorSets);
  DEV(UpdateDescriptorSets);
  DEV(CreateCommandPool);
  DEV(AllocateCommandBuffers);
  DEV(BeginCommandBuffer);
  DEV(EndCommandBuffer);
  DEV(CmdBindPipeline);
  DEV(CmdBindDescriptorSets);
  DEV(CmdDispatch);
  DEV(CmdPushConstants);
  DEV(CmdPipelineBarrier);
  DEV(QueueSubmit);
  DEV(QueueWaitIdle);
#undef DEV
  g_vk.GetDeviceQueue(g_vk.dev, ctx.qfi, 0, &g_vk.queue);
  printf("{\"k\":\"meta\",\"dev\":\"%s\",\"vk_driver_files\":\"%s\"}\n",
         ctx.name,
         getenv("VK_DRIVER_FILES") ? getenv("VK_DRIVER_FILES") : "");
  fflush(stdout);
}

static const uint32_t kDispatches = 226;  // emitted-barrier count per token
static const uint32_t kLevels = 42;       // greedy level count per token
static const uint32_t kGroups = 4;        // 4 x 64 = 256 threads / dispatch
static const uint32_t kSliceWords = 256;  // distinct-slice case stride
static const uint32_t kBufWords = kDispatches * kSliceWords;  // 57856

// Duration-parametrized RMW: reads its output word (true RAW when the next
// dispatch reuses the slice), burns `iters` dependent ALU rounds, writes back.
static const char* kShader =
    "#version 450\n"
    "layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;\n"
    "layout(set = 0, binding = 0) buffer Out { uint out_buf[]; };\n"
    "layout(push_constant) uniform PC { uint iters; uint base; } pc;\n"
    "void main() {\n"
    "  uint g = gl_GlobalInvocationID.x;\n"
    "  uint x = out_buf[pc.base + g] + g;\n"
    "  for (uint i = 0u; i < pc.iters; ++i) {\n"
    "    x = x * 1664525u + 1013904223u;\n"
    "  }\n"
    "  out_buf[pc.base + g] = x;\n"
    "}\n";

struct Buf {
  VkBuffer buf{VK_NULL_HANDLE};
  VkDeviceMemory mem{VK_NULL_HANDLE};
  void* map{nullptr};
};

struct Bench {
  Buf out;
  VkDescriptorSetLayout dsl{VK_NULL_HANDLE};
  VkPipelineLayout layout{VK_NULL_HANDLE};
  VkDescriptorPool pool{VK_NULL_HANDLE};
  VkDescriptorSet set{VK_NULL_HANDLE};
  VkShaderModule mod{VK_NULL_HANDLE};
  VkPipeline pipe{VK_NULL_HANDLE};
  VkCommandPool cpool{VK_NULL_HANDLE};
  VkCommandBuffer cmd{VK_NULL_HANDLE};
};

static std::string compile_shader() {
  char path_src[] = "/tmp/lbb-src-XXXXXX.comp";
  char path_spv[] = "/tmp/lbb-out-XXXXXX.spv";
  int fd = mkstemps(path_src, 5);
  if (fd < 0) die("mkstemps src");
  if (write(fd, kShader, strlen(kShader)) != (ssize_t)strlen(kShader))
    die("write src");
  close(fd);
  fd = mkstemps(path_spv, 4);
  if (fd < 0) die("mkstemps spv");
  close(fd);
  std::string cmd = std::string("glslc -fshader-stage=compute "
      "--target-env=vulkan1.3 ") + path_src + " -o " + path_spv + " 2>&1";
  if (system(cmd.c_str()) != 0) die("shader compile failed");
  std::ifstream f(path_spv, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  unlink(path_src);
  unlink(path_spv);
  return ss.str();
}

static void setup_bench(Bench& b) {
  // Prefer host-visible device memory so the execution proof can read back.
  VkBufferCreateInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bi.size = kBufWords * 4;
  bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  if (g_vk.CreateBuffer(g_vk.dev, &bi, nullptr, &b.out.buf) != VK_SUCCESS)
    die("CreateBuffer");
  VkMemoryRequirements req;
  g_vk.GetBufferMemoryRequirements(g_vk.dev, b.out.buf, &req);
  uint32_t mt = find_memtype(req.memoryTypeBits,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  VkMemoryPropertyFlags got = 0;
  if (mt == UINT32_MAX) {
    mt = find_memtype(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    got = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  } else {
    got = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  }
  if (mt == UINT32_MAX)
    mt = find_memtype(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  if (mt == UINT32_MAX) die("no memtype");
  VkMemoryAllocateInfo mai{};
  mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mai.allocationSize = req.size;
  mai.memoryTypeIndex = mt;
  if (g_vk.AllocateMemory(g_vk.dev, &mai, nullptr, &b.out.mem) != VK_SUCCESS)
    die("AllocateMemory");
  if (g_vk.BindBufferMemory(g_vk.dev, b.out.buf, b.out.mem, 0) != VK_SUCCESS)
    die("BindBufferMemory");
  if (got & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
    if (g_vk.MapMemory(g_vk.dev, b.out.mem, 0, VK_WHOLE_SIZE, 0, &b.out.map) !=
        VK_SUCCESS)
      die("MapMemory");
  }

  VkDescriptorSetLayoutBinding bind{};
  bind.binding = 0;
  bind.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bind.descriptorCount = 1;
  bind.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  VkDescriptorSetLayoutCreateInfo dlci{};
  dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dlci.bindingCount = 1;
  dlci.pBindings = &bind;
  if (g_vk.CreateDescriptorSetLayout(g_vk.dev, &dlci, nullptr, &b.dsl) !=
      VK_SUCCESS)
    die("CreateDescriptorSetLayout");
  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset = 0;
  pcr.size = 128;
  VkPipelineLayoutCreateInfo plci{};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &b.dsl;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;
  if (g_vk.CreatePipelineLayout(g_vk.dev, &plci, nullptr, &b.layout) !=
      VK_SUCCESS)
    die("CreatePipelineLayout");

  VkDescriptorPoolSize ps{};
  ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  ps.descriptorCount = 1;
  VkDescriptorPoolCreateInfo pci{};
  pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pci.maxSets = 1;
  pci.poolSizeCount = 1;
  pci.pPoolSizes = &ps;
  if (g_vk.CreateDescriptorPool(g_vk.dev, &pci, nullptr, &b.pool) !=
      VK_SUCCESS)
    die("CreateDescriptorPool");
  VkDescriptorSetAllocateInfo dsai{};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = b.pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &b.dsl;
  if (g_vk.AllocateDescriptorSets(g_vk.dev, &dsai, &b.set) != VK_SUCCESS)
    die("AllocateDescriptorSets");
  VkDescriptorBufferInfo dbi{};
  dbi.buffer = b.out.buf;
  dbi.offset = 0;
  dbi.range = VK_WHOLE_SIZE;
  VkWriteDescriptorSet wr{};
  wr.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  wr.dstSet = b.set;
  wr.dstBinding = 0;
  wr.descriptorCount = 1;
  wr.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  wr.pBufferInfo = &dbi;
  g_vk.UpdateDescriptorSets(g_vk.dev, 1, &wr, 0, nullptr);

  std::string spv = compile_shader();
  VkShaderModuleCreateInfo smci{};
  smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  smci.codeSize = spv.size();
  smci.pCode = (const uint32_t*)spv.data();
  if (g_vk.CreateShaderModule(g_vk.dev, &smci, nullptr, &b.mod) != VK_SUCCESS)
    die("CreateShaderModule");
  VkComputePipelineCreateInfo cpci{};
  cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = b.mod;
  cpci.stage.pName = "main";
  cpci.layout = b.layout;
  if (g_vk.CreateComputePipelines(
          g_vk.dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &b.pipe) != VK_SUCCESS)
    die("CreateComputePipelines");

  VkCommandPoolCreateInfo cpoi{
      VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpoi.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
      VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  cpoi.queueFamilyIndex = ctx.qfi;
  if (g_vk.CreateCommandPool(g_vk.dev, &cpoi, nullptr, &b.cpool) != VK_SUCCESS)
    die("CreateCommandPool");
  VkCommandBufferAllocateInfo cbai{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = b.cpool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  if (g_vk.AllocateCommandBuffers(g_vk.dev, &cbai, &b.cmd) != VK_SUCCESS)
    die("AllocateCommandBuffers");
}

static void full_barrier(VkCommandBuffer c) {
  VkMemoryBarrier m{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  m.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  m.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  g_vk.CmdPipelineBarrier(
      c,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      0,
      1,
      &m,
      0,
      nullptr,
      0,
      nullptr);
}

static void record_dispatch(Bench& b, uint32_t iters, uint32_t base) {
  g_vk.CmdBindPipeline(b.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, b.pipe);
  g_vk.CmdBindDescriptorSets(
      b.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, b.layout, 0, 1, &b.set, 0,
      nullptr);
  uint32_t pc[2] = {iters, base};
  g_vk.CmdPushConstants(
      b.cmd, b.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
  g_vk.CmdDispatch(b.cmd, kGroups, 1, 1);
}

enum class Case { Chain, Levels, Unord };

// One measured rep: record the case into a fresh begin, submit, wait, wall.
static double run_rep(Bench& b, Case which, uint32_t iters) {
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (g_vk.BeginCommandBuffer(b.cmd, &bi) != VK_SUCCESS)
    die("BeginCommandBuffer");
  if (which == Case::Chain) {
    for (uint32_t i = 0; i < kDispatches; ++i) {
      record_dispatch(b, iters, 0);  // same slice: true RAW chain
      if (i + 1 < kDispatches) full_barrier(b.cmd);
    }
  } else if (which == Case::Levels) {
    full_barrier(b.cmd);  // leading head barrier, like the encoder path
    // 226 over 42 levels: 16 levels of 6 then 26 levels of 5.
    uint32_t emitted = 0;
    for (uint32_t lev = 0; lev < kLevels; ++lev) {
      uint32_t width = lev < 16 ? 6 : 5;
      for (uint32_t j = 0; j < width; ++j) {
        record_dispatch(b, iters, emitted * kSliceWords);
        ++emitted;
      }
      if (lev + 1 < kLevels) full_barrier(b.cmd);
    }
  } else {
    for (uint32_t i = 0; i < kDispatches; ++i) {
      record_dispatch(b, iters, i * kSliceWords);
    }
  }
  if (g_vk.EndCommandBuffer(b.cmd) != VK_SUCCESS) die("EndCommandBuffer");
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &b.cmd;
  double t0 = now_us();
  if (g_vk.QueueSubmit(g_vk.queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS)
    die("QueueSubmit");
  if (g_vk.QueueWaitIdle(g_vk.queue) != VK_SUCCESS) die("QueueWaitIdle");
  double t1 = now_us();
  return t1 - t0;
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
static double pct(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  size_t i = (size_t)(p * (v.size() - 1));
  return v[i];
}

static uint32_t g_iters = 1024;

// Calibrate `iters` so a no-barrier 226-dispatch batch averages ~target us
// per kernel, then report the achieved value.
static double calibrate(Bench& b, double target_us) {
  auto batch_wall = [&](uint32_t it) {
    for (uint32_t i = 0; i < 8; ++i) run_rep(b, Case::Unord, it);
    double best = 1e30;
    for (int r = 0; r < 5; ++r) best = std::min(best, run_rep(b, Case::Unord, it));
    return best;
  };
  double t1 = batch_wall(1024);
  double t2 = batch_wall(8192);
  printf(
      "{\"k\":\"cal\",\"iters\":1024,\"wall_us\":%.1f,\"per_kernel_us\":%.3f}\n"
      "{\"k\":\"cal\",\"iters\":8192,\"wall_us\":%.1f,\"per_kernel_us\":%.3f}\n",
      t1, t1 / kDispatches, t2, t2 / kDispatches);
  fflush(stdout);
  double slope = (t2 - t1) / (double)(8192 - 1024);  // us per iter (whole batch)
  double iters = 1024.0 + (target_us * kDispatches - t1) / slope;
  if (iters < 1.0) iters = 1.0;
  g_iters = (uint32_t)iters;
  double achieved = batch_wall(g_iters) / kDispatches;
  printf(
      "{\"k\":\"cal_done\",\"target_us\":%.1f,\"iters\":%u,"
      "\"achieved_per_kernel_us\":%.3f}\n",
      target_us, g_iters, achieved);
  fflush(stdout);
  return achieved;
}

int main() {
  setup_device();
  Bench b;
  setup_bench(b);

  // Execution proof: zero slice 0 via one chain rep, read back a stable
  // nonzero mixed value (host-visible allocation only).
  if (b.out.map) {
    memset(b.out.map, 0, 4096);
    run_rep(b, Case::Chain, 64);
    uint32_t w0 = ((uint32_t*)b.out.map)[0];
    uint32_t w1 = ((uint32_t*)b.out.map)[1];
    printf("{\"k\":\"exec_proof\",\"word0\":%u,\"word1\":%u,\"ok\":%d}\n",
           w0, w1, (w0 != 0 && w1 != 0) ? 1 : 0);
    fflush(stdout);
    if (w0 == 0 || w1 == 0) die("kernels did not execute (zero readback)");
  } else {
    printf("{\"k\":\"exec_proof\",\"skipped\":\"memory not host visible\"}\n");
  }

  uint32_t warmup = 30;
  uint32_t reps = 200;
  if (const char* e = getenv("LB_WARMUP")) warmup = (uint32_t)atoi(e);
  if (const char* e = getenv("LB_REPS")) reps = (uint32_t)atoi(e);

  const char* targets_env = getenv("LB_TARGET_US");
  std::string targets(targets_env ? targets_env : "3,31");
  std::vector<double> targets_v;
  {
    size_t pos = 0;
    while (pos < targets.size()) {
      size_t comma = targets.find(',', pos);
      std::string tok = targets.substr(
          pos, comma == std::string::npos ? std::string::npos : comma - pos);
      targets_v.push_back(atof(tok.c_str()));
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
  }

  for (double target : targets_v) {
    double achieved = calibrate(b, target);
    std::vector<double> walls[3];
    for (uint32_t r = 0; r < warmup; ++r) {
      run_rep(b, Case::Chain, g_iters);
      run_rep(b, Case::Levels, g_iters);
      run_rep(b, Case::Unord, g_iters);
    }
    for (uint32_t r = 0; r < reps; ++r) {
      walls[0].push_back(run_rep(b, Case::Chain, g_iters));
      walls[1].push_back(run_rep(b, Case::Levels, g_iters));
      walls[2].push_back(run_rep(b, Case::Unord, g_iters));
    }
    const char* names[3] = {"chain", "levels", "unord"};
    for (int c = 0; c < 3; ++c) {
      printf(
          "{\"k\":\"case\",\"target_us\":%.1f,\"iters\":%u,"
          "\"achieved_per_kernel_us\":%.3f,\"case\":\"%s\",\"reps\":%u,"
          "\"wall_us_median\":%.1f,\"p10\":%.1f,\"p90\":%.1f}\n",
          target,
          g_iters,
          achieved,
          names[c],
          reps,
          median(walls[c]),
          pct(walls[c], 0.10),
          pct(walls[c], 0.90));
      fflush(stdout);
    }
  }
  printf("LEVEL_BATCH_BENCH_DONE\n");
  return 0;
}
