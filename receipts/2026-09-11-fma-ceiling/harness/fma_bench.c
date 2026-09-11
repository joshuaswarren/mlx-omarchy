/* fma_bench.c — Vulkan compute FMA ceiling benchmark for Apple GPU (honeykrisp).
 *
 * Usage: fma_bench <shader.spv> <total_threads> <iters_per_thread> <flops_total_g> <groups>
 *   - total_threads = SSBO elements (one uint out per invocation); groups * local_size_x
 *   - groups = workgroup count to dispatch
 *   - flops_total_g = total GFLOP (2 per FMA) computed by the sweep script
 * Prints one JSON line: {"spv":..., "gflops_med":..., "gflops_min":..., "ms_med":..., "checksum":...}
 */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define WARMUP 3
#define REPS 9

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    if (argc != 6) { fprintf(stderr, "usage: %s spv total_threads iters flops_total_g groups\n", argv[0]); return 2; }
    const char *spv_path = argv[1];
    uint32_t total_threads = (uint32_t)strtoul(argv[2], NULL, 10);
    uint32_t iters = (uint32_t)strtoul(argv[3], NULL, 10);
    double flops_g = atof(argv[4]);
    uint32_t groups = (uint32_t)strtoul(argv[5], NULL, 10);

    FILE *f = fopen(spv_path, "rb");
    if (!f) { perror("open spv"); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *spv = malloc((size_t)sz);
    if (fread(spv, 1, (size_t)sz, f) != (size_t)sz) { perror("read spv"); return 1; }
    fclose(f);

    VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3 };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
    VkInstance inst;
    if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) { fprintf(stderr, "vkCreateInstance failed\n"); return 1; }

    uint32_t nd = 0; vkEnumeratePhysicalDevices(inst, &nd, NULL);
    VkPhysicalDevice devs[8]; vkEnumeratePhysicalDevices(inst, &nd, devs);
    VkPhysicalDevice pd = NULL; char dname[256] = "?";
    for (uint32_t i = 0; i < nd; i++) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(devs[i], &p);
        if (strstr(p.deviceName, "llvmpipe") || strstr(p.deviceName, "lavapipe") || strstr(p.deviceName, "softpipe")) continue;
        pd = devs[i]; snprintf(dname, sizeof dname, "%s", p.deviceName); break;
    }
    if (!pd) { fprintf(stderr, "no hardware device\n"); return 1; }

    uint32_t qf = 0xFFFFFFFFu, nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, NULL);
    VkQueueFamilyProperties qp[16]; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qp);
    for (uint32_t i = 0; i < nq; i++) if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }
    if (qf == 0xFFFFFFFFu) { fprintf(stderr, "no compute queue\n"); return 1; }

    const char *exts[] = { "VK_KHR_shader_float16_int8", "VK_KHR_cooperative_matrix" };
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio };
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmfeat = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR, .cooperativeMatrix = VK_TRUE };
    VkPhysicalDeviceFloat16Int8FeaturesKHR f16feat = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR, .pNext = &cmfeat, .shaderFloat16 = VK_TRUE };
    VkPhysicalDeviceFeatures2 feat2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f16feat };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &feat2,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
        .enabledExtensionCount = 2, .ppEnabledExtensionNames = exts };
    VkDevice dev;
    if (vkCreateDevice(pd, &dci, NULL, &dev) != VK_SUCCESS) {
        /* retry without fp16 extension for plain fp32/int arms */
        feat2.pNext = NULL; dci.enabledExtensionCount = 0;
        if (vkCreateDevice(pd, &dci, NULL, &dev) != VK_SUCCESS) { fprintf(stderr, "vkCreateDevice failed\n"); return 1; }
    }
    VkQueue queue; vkGetDeviceQueue(dev, qf, 0, &queue);

    VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = (VkDeviceSize)total_threads * 4, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT };
    VkBuffer buf; VkDeviceMemory mem;
    if (vkCreateBuffer(dev, &bci, NULL, &buf) != VK_SUCCESS) { fprintf(stderr, "buffer failed\n"); return 1; }
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, buf, &mr);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t mti = 0; while (!(mr.memoryTypeBits & (1u << mti)) ||
        !(mp.memoryTypes[mti].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) mti++;
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = mti };
    if (vkAllocateMemory(dev, &mai, NULL, &mem) != VK_SUCCESS) { fprintf(stderr, "alloc failed\n"); return 1; }
    vkBindBufferMemory(dev, buf, mem, 0);

    VkDescriptorSetLayoutBinding bind = { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
    VkDescriptorSetLayoutCreateInfo dlci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1, .pBindings = &bind };
    VkDescriptorSetLayout dsl;
    if (vkCreateDescriptorSetLayout(dev, &dlci, NULL, &dsl) != VK_SUCCESS) { fprintf(stderr, "dsl failed\n"); return 1; }
    VkPushConstantRange pcr = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = 4 };
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
    VkPipelineLayout pl;
    if (vkCreatePipelineLayout(dev, &plci, NULL, &pl) != VK_SUCCESS) { fprintf(stderr, "pl failed\n"); return 1; }
    VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = (size_t)sz, .pCode = spv };
    VkShaderModule sm;
    if (vkCreateShaderModule(dev, &smci, NULL, &sm) != VK_SUCCESS) { fprintf(stderr, "shader module failed\n"); return 1; }
    VkPipelineShaderStageCreateInfo ss = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main" };
    VkComputePipelineCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .stage = ss, .layout = pl };
    VkPipeline pipe;
    if (vkCreateComputePipelines(dev, NULL, 1, &cpci, NULL, &pipe) != VK_SUCCESS) { fprintf(stderr, "pipeline failed (spv: %s)\n", spv_path); return 1; }
    VkDescriptorPoolSize psize = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1 };
    VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &psize };
    VkDescriptorPool pool; vkCreateDescriptorPool(dev, &dpci, NULL, &pool);
    VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &dsl };
    VkDescriptorSet dset; vkAllocateDescriptorSets(dev, &dsai, &dset);
    VkDescriptorBufferInfo dbi = { .buffer = buf, .offset = 0, .range = VK_WHOLE_SIZE };
    VkWriteDescriptorSet wr = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = dset, .dstBinding = 0,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi };
    vkUpdateDescriptorSets(dev, 1, &wr, 0, NULL);

    VkCommandPoolCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = qf };
    VkCommandPool cpool; vkCreateCommandPool(dev, &cpi, NULL, &cpool);
    VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
    VkCommandBuffer cmd; vkAllocateCommandBuffers(dev, &cbai, &cmd);

    double times[REPS];
    for (int r = 0; r < WARMUP + REPS; r++) {
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(cmd, &bi);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &dset, 0, NULL);
        vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &iters);
        vkCmdDispatch(cmd, groups, 1, 1);
        vkEndCommandBuffer(cmd);
        VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd };
        double t0 = now_s();
        vkQueueSubmit(queue, 1, &si, NULL);
        vkQueueWaitIdle(queue);
        double t1 = now_s();
        if (r >= WARMUP) times[r - WARMUP] = t1 - t0;
    }
    qsort(times, REPS, sizeof(double), cmp_d);
    double med = times[REPS / 2], mn = times[0];

    uint32_t *host;
    if (vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, (void **)&host) != VK_SUCCESS) { fprintf(stderr, "map failed\n"); return 1; }
    uint64_t sum = 0; for (uint32_t i = 0; i < total_threads; i++) sum += host[i];
    vkUnmapMemory(dev, mem);

    double gflops_med = flops_g / med, gflops_min = flops_g / mn;
    printf("{\"spv\":\"%s\",\"device\":\"%s\",\"ms_med\":%.3f,\"ms_min\":%.3f,\"gflops_med\":%.1f,\"gflops_min\":%.1f,\"checksum\":%llu}\n",
           spv_path, dname, med * 1e3, mn * 1e3, gflops_med, gflops_min, (unsigned long long)sum);
    return 0;
}
