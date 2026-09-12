/* Pipeline-creation probe: does the installed Honeykrisp driver accept a
 * cooperative-matrix object loaded from Private storage?
 *
 * Usage: coopmat_probe <module.spv> <expected: success|failure>
 *
 * Control module (StorageBuffer load) must create a pipeline; the
 * Private-load module must be rejected. Together they name, at the
 * driver level, whether a cooperative-matrix B operand can be built
 * from per-invocation values without a memory round trip.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

static const char *result_name(VkResult r) {
  switch (r) {
#define C(x)                                                                   \
  case x:                                                                      \
    return #x
    C(VK_SUCCESS);
    C(VK_NOT_READY);
    C(VK_ERROR_INVALID_SHADER_NV);
    C(VK_ERROR_OUT_OF_HOST_MEMORY);
    C(VK_ERROR_OUT_OF_DEVICE_MEMORY);
    C(VK_ERROR_INITIALIZATION_FAILED);
    C(VK_ERROR_DEVICE_LOST);
    C(VK_ERROR_EXTENSION_NOT_PRESENT);
    C(VK_ERROR_FEATURE_NOT_PRESENT);
    C(VK_ERROR_INVALID_EXTENSION);
    C(VK_ERROR_LAYER_NOT_PRESENT);
    C(VK_ERROR_INCOMPATIBLE_DRIVER);
    C(VK_ERROR_TOO_MANY_OBJECTS);
    C(VK_ERROR_FORMAT_NOT_SUPPORTED);
    C(VK_ERROR_FRAGMENTED_POOL);
    C(VK_ERROR_INVALID_EXTERNAL_HANDLE);
#undef C
    default: {
      static char buf[32];
      snprintf(buf, sizeof(buf), "VkResult(%d)", (int)r);
      return buf;
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <module.spv> <success|failure>\n", argv[0]);
    return 2;
  }
  const char *expect = argv[2];

  FILE *f = fopen(argv[1], "rb");
  if (!f) {
    perror("open");
    return 2;
  }
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (size <= 0 || size % 4 != 0) {
    fprintf(stderr, "bad module size %ld\n", size);
    return 2;
  }
  unsigned *code = malloc((size_t)size);
  if (fread(code, 1, (size_t)size, f) != (size_t)size) {
    perror("read");
    return 2;
  }
  fclose(f);

  VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  VkInstance instance;
  VkResult r = vkCreateInstance(&ici, NULL, &instance);
  if (r != VK_SUCCESS) {
    printf("create-instance %s\n", result_name(r));
    return 1;
  }
  uint32_t ndev = 0;
  vkEnumeratePhysicalDevices(instance, &ndev, NULL);
  if (ndev == 0) {
    printf("no-physical-device\n");
    return 1;
  }
  VkPhysicalDevice pdev;
  vkEnumeratePhysicalDevices(instance, &ndev, &pdev);

  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = 0;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;

  const char *exts[] = {"VK_KHR_cooperative_matrix",
                        "VK_KHR_shader_subgroup_basic",
                        "VK_KHR_shader_subgroup_arithmetic",
                        "VK_KHR_shader_subgroup_vote",
                        "VK_KHR_shader_subgroup_ballot",
                        "VK_KHR_shader_subgroup_shuffle",
                        "VK_KHR_shader_subgroup_shuffle_relative",
                        "VK_KHR_shader_subgroup_clustered",
                        "VK_KHR_16bit_storage",
                        "VK_KHR_8bit_storage",
                        "VK_KHR_shader_float16_int8"};
  uint32_t next = 0;
  vkEnumerateDeviceExtensionProperties(pdev, NULL, &next, NULL);
  VkExtensionProperties *avail = malloc(sizeof(*avail) * next);
  vkEnumerateDeviceExtensionProperties(pdev, NULL, &next, avail);
  const char *want[16];
  uint32_t nwant = 0;
  for (uint32_t i = 0; i < sizeof(exts) / sizeof(exts[0]); ++i) {
    for (uint32_t j = 0; j < next; ++j) {
      if (strcmp(avail[j].extensionName, exts[i]) == 0) {
        want[nwant++] = exts[i];
        break;
      }
    }
  }
  free(avail);

  VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = nwant;
  dci.ppEnabledExtensionNames = want;
  VkDevice device;
  r = vkCreateDevice(pdev, &dci, NULL, &device);
  if (r != VK_SUCCESS) {
    printf("create-device %s (enabled %u exts)\n", result_name(r), nwant);
    return 1;
  }

  VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = (size_t)size;
  smci.pCode = code;
  VkShaderModule module;
  r = vkCreateShaderModule(device, &smci, NULL, &module);
  if (r != VK_SUCCESS) {
    printf("create-shader-module %s\n", result_name(r));
    return 1;
  }

  /* One storage buffer binding, matching both probe modules. */
  VkDescriptorSetLayoutBinding binding = {0};
  binding.binding = 0;
  binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  binding.descriptorCount = 1;
  binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  VkDescriptorSetLayoutCreateInfo dlci = {
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dlci.bindingCount = 1;
  dlci.pBindings = &binding;
  VkDescriptorSetLayout layout;
  r = vkCreateDescriptorSetLayout(device, &dlci, NULL, &layout);
  if (r != VK_SUCCESS) {
    printf("create-layout %s\n", result_name(r));
    return 1;
  }
  VkPipelineLayoutCreateInfo plci = {
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &layout;
  VkPipelineLayout pipeline_layout;
  r = vkCreatePipelineLayout(device, &plci, NULL, &pipeline_layout);
  if (r != VK_SUCCESS) {
    printf("create-pipeline-layout %s\n", result_name(r));
    return 1;
  }

  VkPipelineShaderStageCreateInfo stage = {
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage.module = module;
  stage.pName = "main";
  VkComputePipelineCreateInfo pci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pci.stage = stage;
  pci.layout = pipeline_layout;

  VkPipeline pipeline;
  r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pci, NULL, &pipeline);
  printf("pipeline-creation %s\n", result_name(r));

  int failed = (strcmp(expect, "success") == 0) ? (r != VK_SUCCESS)
                                                : (r == VK_SUCCESS);
  printf("expect=%s observed=%s -> %s\n", expect, result_name(r),
         failed ? "UNEXPECTED" : "EXPECTED");
  return failed;
}
