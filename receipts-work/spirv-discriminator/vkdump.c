/* Pipeline-compile-only Vulkan harness: instance -> device -> vkCreateComputePipelines.
 * Zero queue submissions, zero dispatches. Used with AGX_MESA_DEBUG=shaders to
 * capture the Honeykrisp NIR + AGX disassembly of a shader module on stdout. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

static unsigned char *read_file(const char *path, size_t *len) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  unsigned char *buf = malloc((size_t)n);
  if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(buf); return NULL; }
  fclose(f);
  *len = (size_t)n;
  return buf;
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: vkdump <shader.spv>\n"); return 2; }
  size_t spv_len = 0;
  unsigned char *spv = read_file(argv[1], &spv_len);
  if (!spv) { perror("read spv"); return 1; }

  VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  VkInstance inst;
  VkResult r = vkCreateInstance(&ici, NULL, &inst);
  if (r != VK_SUCCESS) { fprintf(stderr, "vkCreateInstance: %d\n", r); return 1; }

  uint32_t n = 0;
  vkEnumeratePhysicalDevices(inst, &n, NULL);
  if (!n) { fprintf(stderr, "no physical devices\n"); return 1; }
  VkPhysicalDevice *pds = calloc(n, sizeof(VkPhysicalDevice));
  vkEnumeratePhysicalDevices(inst, &n, pds);
  VkPhysicalDevice pd = VK_NULL_HANDLE;
  for (uint32_t i = 0; i < n; ++i) {
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(pds[i], &p);
    fprintf(stderr, "device %u: %s vendor 0x%x api %u.%u\n", i, p.deviceName,
            p.vendorID, VK_API_VERSION_MAJOR(p.apiVersion),
            VK_API_VERSION_MINOR(p.apiVersion));
    if (strncmp(p.deviceName, "Apple", 5) == 0) pd = pds[i];
  }
  if (!pd) { fprintf(stderr, "no Apple GPU found\n"); return 1; }

  uint32_t qf = 0, qfn = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfn, NULL);
  VkQueueFamilyProperties *qfp = calloc(qfn, sizeof(VkQueueFamilyProperties));
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfn, qfp);
  for (uint32_t i = 0; i < qfn; ++i)
    if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }

  float prio = 1.0f;
  VkDeviceQueueCreateInfo q = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  q.queueFamilyIndex = qf;
  q.queueCount = 1;
  q.pQueuePriorities = &prio;

  VkPhysicalDevice16BitStorageFeatures f16 =
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
  f16.storageBuffer16BitAccess = VK_TRUE;
  f16.uniformAndStorageBuffer16BitAccess = VK_TRUE;
  VkPhysicalDeviceVulkan12Features f12 =
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  f12.pNext = &f16;
  f12.timelineSemaphore = VK_TRUE;
  VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.pNext = &f12;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &q;
  VkDevice dev;
  r = vkCreateDevice(pd, &dci, NULL, &dev);
  if (r != VK_SUCCESS) { fprintf(stderr, "vkCreateDevice: %d\n", r); return 1; }

  VkDescriptorSetLayoutBinding b[21];
  for (uint32_t i = 0; i < 21; ++i) {
    b[i].binding = i;
    b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[i].descriptorCount = 1;
    b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dlci =
      {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dlci.bindingCount = 21;
  dlci.pBindings = b;
  VkDescriptorSetLayout dsl;
  r = vkCreateDescriptorSetLayout(dev, &dlci, NULL, &dsl);
  if (r != VK_SUCCESS) { fprintf(stderr, "layout: %d\n", r); return 1; }

  VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 120};
  VkPipelineLayoutCreateInfo plci =
      {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &dsl;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;
  VkPipelineLayout pl;
  r = vkCreatePipelineLayout(dev, &plci, NULL, &pl);
  if (r != VK_SUCCESS) { fprintf(stderr, "pipeline layout: %d\n", r); return 1; }

  VkShaderModuleCreateInfo smci =
      {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = spv_len;
  smci.pCode = (const uint32_t *)spv;
  VkShaderModule sm;
  r = vkCreateShaderModule(dev, &smci, NULL, &sm);
  if (r != VK_SUCCESS) { fprintf(stderr, "shader module: %d\n", r); return 1; }

  VkPipelineShaderStageCreateInfo st =
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  st.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  st.module = sm;
  st.pName = "main";
  VkComputePipelineCreateInfo cpi =
      {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpi.stage = st;
  cpi.layout = pl;
  VkPipeline pipe;
  r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe);
  fprintf(stderr, "vkCreateComputePipelines: %d\n", r);

  vkDestroyPipeline(dev, pipe, NULL);
  vkDestroyShaderModule(dev, sm, NULL);
  vkDestroyPipelineLayout(dev, pl, NULL);
  vkDestroyDescriptorSetLayout(dev, dsl, NULL);
  vkDestroyDevice(dev, NULL);
  vkDestroyInstance(inst, NULL);
  return r == VK_SUCCESS ? 0 : 1;
}
