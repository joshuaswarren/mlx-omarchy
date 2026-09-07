// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Per-dispatch fixed cost of the omarchy dispatch path, uninstrumented.
// Every row is N trivial dispatches (ElementwiseF16 add on 896 f16
// elements, the decode bias-add shape) and reports us/dispatch, best of
// three. Three levels separate the floor:
//
//   raw      N dispatches recorded straight into one command buffer
//            through the loaded device table, each mechanism switched
//            independently: barriers, descriptor sets, push-constant
//            size, pipeline rebinds, dispatches per submit.
//   encoder  N CommandEncoder::dispatch_compute calls, one commit and
//            synchronize: the production recording path.
//   ops      N add() ops through eval(): host graph plus dispatch.
//
//   ./omarchy_dispatch_floor_bench [raw|encoder|ops|all] [N]
//
// Not registered with ctest: it is a benchmark, and its raw level
// records outside the encoder's dependency contract on purpose.

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/backend/omarchy/vulkan.h"
#include "mlx/mlx.h"

using namespace mlx::core;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int kCount = 896;
constexpr int kTrials = 3;

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Host-timed levels spin the CPU first: a ~5 ms recording burst after a
// GPU wait otherwise runs at whatever frequency the governor left the
// core at (measured 0.9-13 us/dispatch for the same code on the M1).
void warm_cpu() {
  // DISPATCH_FLOOR_NO_WARM=1 skips the spin (A/B of the spin itself).
  static const bool skip = omarchy::env_flag("DISPATCH_FLOOR_NO_WARM");
  if (skip) {
    return;
  }
  volatile uint64_t sink = 0;
  auto t0 = Clock::now();
  while (seconds_since(t0) < 0.3) {
    for (int i = 0; i < 100000; ++i) {
      sink = sink * 6364136223846793005ull + 1442695040888963407ull;
    }
  }
}

omarchy::ComputeBinding binding(const array& value) {
  auto* buf = static_cast<const omarchy::VulkanBuffer*>(value.buffer().ptr());
  return {buf->buffer, 0, buf->size, buf};
}

omarchy::ComputeParams rope_params() {
  omarchy::ComputeParams p;
  p.count = 14 * 32; // heads x half_dims, one time step
  p.flags = 1u; // forward, half-split, no transpose, no passthrough
  p.alpha = 1.0f;
  p.beta = 0.28782f; // log(10000)/32
  p.dims = 32;
  p.shape[0] = 14;
  p.shape[1] = 1;
  p.matrix_m = 0;
  p.in_strides[0] = 64;
  p.in_strides[1] = 896;
  p.in_strides[2] = 1;
  p.out_strides[0] = 64;
  p.out_strides[1] = 896;
  p.out_strides[2] = 1;
  return p;
}

// The same query rope on the three-axis grid (x = pairs, y = head, z =
// batch): RoPE::eval_gpu's default since the HALF_DIMS specialization.
omarchy::ComputeParams rope_grid_params() {
  omarchy::ComputeParams p = rope_params();
  p.flags |= omarchy::kRopeGridFlag;
  return p;
}

// The decode KV-cache row paste: SliceUpdate of a [2 heads, 64] f16 row
// into the (2, 256, 64) cache, collapsed to rank 2 by copy_gpu_inplace.
omarchy::ComputeParams copy_params() {
  omarchy::ComputeParams p;
  p.count = 128;
  p.dims = 2;
  p.shape[0] = 2;
  p.shape[1] = 64;
  p.in_strides[0] = 64;
  p.in_strides[1] = 1;
  p.out_strides[0] = 256 * 64;
  p.out_strides[1] = 1;
  return p;
}

omarchy::ComputeParams copy_grid_params() {
  omarchy::ComputeParams p = copy_params();
  p.flags |= omarchy::kCopyGridFlag;
  return p;
}

// mx.fast.rms_norm of one 896-wide decode row (RMSNorm::eval_gpu shape).
omarchy::ComputeParams rms_norm_params() {
  omarchy::ComputeParams p;
  p.count = kCount;
  p.lhs_size = kCount;
  p.reduce_size = kCount;
  p.output_size = 1;
  p.alpha = 1e-6f;
  return p;
}

omarchy::ComputeParams add_params() {
  omarchy::ComputeParams p;
  p.count = kCount;
  p.operation = 0;
  p.lhs_size = kCount;
  p.rhs_size = kCount;
  p.output_size = kCount;
  return p;
}

struct Fixture {
  Stream stream;
  array a;
  array b;
  array out;
  // A (2, 256, 64) f16 KV-cache buffer for the copy rows.
  array cache;
  Fixture()
      : stream(new_stream(Device::gpu)),
        a(ones(Shape{kCount}, float16, stream)),
        b(ones(Shape{kCount}, float16, stream)),
        out(zeros(Shape{kCount}, float16, stream)),
        cache(zeros(Shape{2 * 256 * 64}, float16, stream)) {
    eval(a, b, out, cache);
    omarchy::get_command_encoder(stream).synchronize();
  }
  std::array<omarchy::ComputeBinding, 4> bindings() const {
    return {binding(a), binding(b), binding(out), binding(out)};
  }
  bool check() {
    array host = astype(out, float32, stream);
    eval(host);
    omarchy::get_command_encoder(stream).synchronize();
    for (int i = 0; i < kCount; ++i) {
      if (host.data<float>()[i] != 2.0f) {
        return false;
      }
    }
    return true;
  }
};

// ---------------------------------------------------------------- raw --

enum class Barrier { None, PrePost, Pre, Full, Narrow };
enum class Desc { Alloc, Bind, Reuse };

enum class Node { Dispatch, Copy, Fill };

// Which production kernel a dispatch row measures, each on its decode
// shape: Add = ElementwiseF16 add (896), Fill = FillF16 (a 14-instruction
// shader isolating the launch cost), Rope = FastRopeF16 query rope on the
// 1-D unravel, RopeGrid = the same on the three-axis grid, CopyRows =
// CopyGeneralF16 KV-cache row paste on the runtime unravel, CopyGrid =
// the same on the two-axis grid, RmsNorm = FastRmsNormF16 on one row.
enum class Kernel { Add, Fill, Rope, RopeGrid, CopyRows, CopyGrid, RmsNorm, RmsNormSg };

struct RawConfig {
  Barrier barrier{Barrier::PrePost};
  Desc desc{Desc::Alloc};
  uint32_t push_bytes{sizeof(omarchy::ComputeParams)};
  bool rebind{true};
  int submits{1};
  bool serial{false};
  const char* label;
  // Workgroups per dispatch (0 = the 896-element default) or the node
  // kind when the row measures a transfer command instead.
  uint32_t groups{0};
  Node node{Node::Dispatch};
  Kernel kernel{Kernel::Add};
  // params.count = 0 (shape[1] = 0 for the grid rows): every invocation
  // exits at once, so the row measures launch plus preamble with no
  // main-body work.
  bool count0{false};
  // Cycle through kAlternate (a decode-like mix of pipelines, bindings
  // and grids) instead of repeating one kernel: production decode never
  // dispatches the same pipeline twice in a row.
  bool alternate{false};
};

constexpr Kernel kAlternate[] = {
    Kernel::Add, Kernel::RmsNorm, Kernel::RopeGrid, Kernel::CopyGrid,
    Kernel::Fill};
constexpr int kAlternateCount = sizeof(kAlternate) / sizeof(*kAlternate);

const char* kernel_name(Kernel k) {
  switch (k) {
    case Kernel::Add:
      return "ElementwiseF16";
    case Kernel::Fill:
      return "FillF16";
    case Kernel::Rope:
      return "FastRopeF16";
    case Kernel::RopeGrid:
      return "FastRopeF16/grid";
    case Kernel::CopyRows:
      return "CopyGeneralF16";
    case Kernel::CopyGrid:
      return "CopyGeneralF16/grid";
    case Kernel::RmsNorm:
      return "FastRmsNormF16";
    case Kernel::RmsNormSg:
      return "FastRmsNormSubgroupF16";
  }
  return "?";
}

omarchy::ComputeParams kernel_params(Kernel k) {
  switch (k) {
    case Kernel::Add:
      return add_params();
    case Kernel::Fill: {
      auto p = add_params();
      p.alpha = 0.0f; // halfword 0x0000 in the low 16 bits
      return p;
    }
    case Kernel::Rope:
      return rope_params();
    case Kernel::RopeGrid:
      return rope_grid_params();
    case Kernel::CopyRows:
      return copy_params();
    case Kernel::CopyGrid:
      return copy_grid_params();
    case Kernel::RmsNorm:
    case Kernel::RmsNormSg:
      return rms_norm_params();
  }
  return add_params();
}

omarchy::ComputeKernel kernel_enum(Kernel k) {
  switch (k) {
    case Kernel::Add:
      return omarchy::ComputeKernel::ElementwiseF16;
    case Kernel::Fill:
      return omarchy::ComputeKernel::FillF16;
    case Kernel::Rope:
    case Kernel::RopeGrid:
      return omarchy::ComputeKernel::FastRopeF16;
    case Kernel::CopyRows:
    case Kernel::CopyGrid:
      return omarchy::ComputeKernel::CopyGeneralF16;
    case Kernel::RmsNorm:
      return omarchy::ComputeKernel::FastRmsNormF16;
    case Kernel::RmsNormSg:
      return omarchy::ComputeKernel::FastRmsNormSubgroupF16;
  }
  return omarchy::ComputeKernel::ElementwiseF16;
}

// Workgroup grid of one dispatch row.
std::array<uint32_t, 3> kernel_groups(Kernel k, const omarchy::ComputeParams& p) {
  switch (k) {
    case Kernel::RopeGrid:
      return {omarchy::compute_dispatch_group_count(p.dims * p.shape[1]),
              p.shape[0], 1u};
    case Kernel::CopyGrid:
      return {omarchy::compute_dispatch_group_count(p.shape[1]), p.shape[0],
              1u};
    case Kernel::RmsNorm:
    case Kernel::RmsNormSg:
      return {1u, 1u, 1u};
    default:
      return {omarchy::compute_dispatch_group_count(p.count), 1u, 1u};
  }
}

const char* barrier_name(Barrier b) {
  switch (b) {
    case Barrier::None:
      return "none";
    case Barrier::PrePost:
      return "prepost";
    case Barrier::Pre:
      return "pre";
    case Barrier::Full:
      return "full";
    case Barrier::Narrow:
      return "narrow";
  }
  return "?";
}

const char* desc_name(Desc d) {
  switch (d) {
    case Desc::Alloc:
      return "alloc";
    case Desc::Bind:
      return "bind";
    case Desc::Reuse:
      return "reuse";
  }
  return "?";
}

class RawBench {
 public:
  RawBench(Fixture& fx, int n)
      : fx_(fx), n_(n), dev_(omarchy::device()), cache_(binding(fx.cache)) {
    auto& dt = omarchy::vk::device_table();
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = dev_.queue_family();
    VKX_CHECK(dt.CreateCommandPool(dev_.handle(), &pci, nullptr, &pool_));
    cmds_.resize(n);
    VkCommandBufferAllocateInfo ai{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = static_cast<uint32_t>(n);
    VKX_CHECK(dt.AllocateCommandBuffers(dev_.handle(), &ai, cmds_.data()));
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VKX_CHECK(dt.CreateFence(dev_.handle(), &fi, nullptr, &fence_));
    push_ = dev_.push_descriptors();
    if (!push_) {
      VkDescriptorPoolSize ps{};
      ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      ps.descriptorCount = 4u * static_cast<uint32_t>(n + 1);
      VkDescriptorPoolCreateInfo dpi{
          VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
      dpi.maxSets = static_cast<uint32_t>(n + 1);
      dpi.poolSizeCount = 1;
      dpi.pPoolSizes = &ps;
      VKX_CHECK(dt.CreateDescriptorPool(dev_.handle(), &dpi, nullptr, &dpool_));
    }
    layout_ = dev_.compute().pipeline_layout();
    set_layout_ = dev_.compute().descriptor_layout();
    bindings_ = fx.bindings();
    if (!push_) {
      shared_set_ = alloc_set();
      write_set(shared_set_);
    }
  }

  ~RawBench() {
    auto& dt = omarchy::vk::device_table();
    dt.DeviceWaitIdle(dev_.handle());
    if (!push_) {
      dt.DestroyDescriptorPool(dev_.handle(), dpool_, nullptr);
    }
    dt.DestroyFence(dev_.handle(), fence_, nullptr);
    dt.DestroyCommandPool(dev_.handle(), pool_, nullptr);
  }

  // Returns {record_us_per_dispatch, gpu_us_per_dispatch}; gpu time is
  // first submit to last fence signal.
  std::pair<double, double> run(const RawConfig& cfg) {
    auto& dt = omarchy::vk::device_table();
    if (!push_) {
      VKX_CHECK(dt.ResetDescriptorPool(dev_.handle(), dpool_, 0));
      shared_set_ = alloc_set();
    }
    const std::array<omarchy::ComputeBinding, 4> saved = bindings_;
    std::vector<Slot> slots;
    if (cfg.alternate) {
      for (Kernel k : kAlternate) {
        slots.push_back(prepare(k, cfg.count0, saved));
      }
    } else {
      slots.push_back(prepare(cfg.kernel, cfg.count0, saved));
    }
    bindings_ = slots[0].bindings;
    if (!push_) {
      write_set(shared_set_);
    }
    const int per_submit = n_ / cfg.submits;

    auto t0 = Clock::now();
    for (int s = 0; s < cfg.submits; ++s) {
      VkCommandBuffer cmd = cmds_[s];
      VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      VKX_CHECK(dt.BeginCommandBuffer(cmd, &bi));
      dt.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, slots[0].pipeline);
      bind_shared(cmd);
      dt.CmdPushConstants(
          cmd, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(slots[0].params),
          &slots[0].params);
      for (int i = 0; i < per_submit; ++i) {
        const Slot& slot = slots[cfg.alternate ? i % kAlternateCount : 0];
        const VkPipeline pipeline = slot.pipeline;
        const omarchy::ComputeParams& params = slot.params;
        const std::array<uint32_t, 3>& grid = slot.grid;
        bindings_ = slot.bindings;
        if (cfg.barrier == Barrier::PrePost || cfg.barrier == Barrier::Pre) {
          pre_barrier(cmd);
        } else if (cfg.barrier == Barrier::Full) {
          full_barrier(cmd);
        } else if (cfg.barrier == Barrier::Narrow) {
          narrow_barrier(cmd);
        }
        if (cfg.rebind) {
          dt.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        }
        if (cfg.desc == Desc::Alloc && !push_) {
          VkDescriptorSet set = alloc_set();
          write_set(set);
          dt.CmdBindDescriptorSets(
              cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &set, 0,
              nullptr);
        } else if (cfg.desc != Desc::Reuse) {
          // Pooled: rebind the shared set. Push: push the four writes
          // (the alloc and bind rows are the same command in push mode).
          bind_shared(cmd);
        }
        if (cfg.push_bytes > 0) {
          dt.CmdPushConstants(
              cmd, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, cfg.push_bytes,
              &params);
        }
        if (cfg.node == Node::Copy) {
          VkBufferCopy region{0, 0, kCount * sizeof(uint16_t)};
          dt.CmdCopyBuffer(cmd, bindings_[0].buffer, bindings_[2].buffer, 1, &region);
        } else if (cfg.node == Node::Fill) {
          dt.CmdFillBuffer(cmd, bindings_[2].buffer, 0, kCount * sizeof(uint16_t), 0x3c003c00u);
        } else {
          dt.CmdDispatch(
              cmd, cfg.groups ? cfg.groups : grid[0], grid[1], grid[2]);
        }
        if (cfg.barrier == Barrier::PrePost) {
          post_barrier(cmd);
        }
      }
      VKX_CHECK(dt.EndCommandBuffer(cmd));
    }
    double record = seconds_since(t0);

    auto t1 = Clock::now();
    if (cfg.serial) {
      for (int s = 0; s < cfg.submits; ++s) {
        submit_and_wait(&cmds_[s], 1);
      }
    } else {
      submit_and_wait(cmds_.data(), cfg.submits);
    }
    double gpu = seconds_since(t1);
    bindings_ = saved;
    return {record / n_ * 1e6, gpu / n_ * 1e6};
  }

 private:
  // One kernel's dispatch recipe on its decode shape.
  struct Slot {
    std::array<omarchy::ComputeBinding, 4> bindings;
    omarchy::ComputeParams params;
    VkPipeline pipeline;
    std::array<uint32_t, 3> grid;
  };

  Slot prepare(
      Kernel kernel,
      bool count0,
      const std::array<omarchy::ComputeBinding, 4>& saved) {
    Slot slot;
    switch (kernel) {
      case Kernel::Fill:
        // FillF16 writes binding 0: bind `out` in every slot.
        slot.bindings = {saved[2], saved[2], saved[2], saved[2]};
        break;
      case Kernel::Rope:
      case Kernel::RopeGrid:
        // FastRopeF16: input, output, int32 offsets (b's bytes), freqs unused.
        slot.bindings = {saved[0], saved[2], saved[1], saved[1]};
        break;
      case Kernel::CopyRows:
      case Kernel::CopyGrid:
        // CopyGeneralF16: input, input, cache output, axis metadata unused.
        slot.bindings = {saved[0], saved[0], cache_, cache_};
        break;
      case Kernel::RmsNorm:
      case Kernel::RmsNormSg:
        // FastRmsNormF16: input, weight, (bias unused), output.
        slot.bindings = {saved[0], saved[1], saved[1], saved[2]};
        break;
      default:
        slot.bindings = saved;
        break;
    }
    slot.params = kernel_params(kernel);
    // Pairs the driver's shaderdb line (AGX_MESA_DEBUG=shaderdb) with the
    // pipeline it describes; a cached pipeline prints no shaderdb line.
    std::fprintf(stderr, "PIPELINE %s\n", kernel_name(kernel));
    std::fflush(stderr);
    slot.pipeline = dev_.compute().pipeline(kernel_enum(kernel), slot.params);
    // The grid keeps its full size: count0 measures launch plus preamble
    // with every invocation exiting at once, not an empty dispatch.
    slot.grid = kernel_groups(kernel, slot.params);
    if (count0) {
      slot.params.count = 0;
      slot.params.shape[1] = 0;
      slot.params.output_size = 0;
    }
    return slot;
  }

  VkDescriptorSet alloc_set() {
    VkDescriptorSetAllocateInfo ai{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = dpool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &set_layout_;
    VkDescriptorSet set{VK_NULL_HANDLE};
    VKX_CHECK(omarchy::vk::device_table().AllocateDescriptorSets(
        dev_.handle(), &ai, &set));
    return set;
  }

  void fill_writes(
      VkDescriptorSet set,
      std::array<VkDescriptorBufferInfo, 4>& info,
      std::array<VkWriteDescriptorSet, 4>& writes) {
    for (uint32_t i = 0; i < 4; ++i) {
      info[i] = {bindings_[i].buffer, bindings_[i].offset, bindings_[i].range};
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = set;
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo = &info[i];
    }
  }

  void write_set(VkDescriptorSet set) {
    std::array<VkDescriptorBufferInfo, 4> info{};
    std::array<VkWriteDescriptorSet, 4> writes{};
    fill_writes(set, info, writes);
    omarchy::vk::device_table().UpdateDescriptorSets(
        dev_.handle(), 4, writes.data(), 0, nullptr);
  }

  // Bind the current bindings_: push them (push-descriptor device) or
  // bind the shared pooled set.
  void bind_shared(VkCommandBuffer cmd) {
    auto& dt = omarchy::vk::device_table();
    if (push_) {
      std::array<VkDescriptorBufferInfo, 4> info{};
      std::array<VkWriteDescriptorSet, 4> writes{};
      fill_writes(VK_NULL_HANDLE, info, writes);
      dt.CmdPushDescriptorSetKHR(
          cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 4, writes.data());
    } else {
      dt.CmdBindDescriptorSets(
          cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &shared_set_,
          0, nullptr);
    }
  }

  // The production blanket pair (encoder.cpp, gated barriers off).
  void pre_barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    before.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT |
        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    omarchy::vk::device_table().CmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before, 0, nullptr, 0,
        nullptr);
  }
  void post_barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    after.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
        VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
    omarchy::vk::device_table().CmdPipelineBarrier(
        cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
            VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &after, 0, nullptr, 0, nullptr);
  }
  // The gated-mode dependency barrier (record_dependency_barrier).
  void full_barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier full{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    full.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    full.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    omarchy::vk::device_table().CmdPipelineBarrier(
        cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &full, 0, nullptr, 0,
        nullptr);
  }
  // Compute-to-compute only.
  void narrow_barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    omarchy::vk::device_table().CmdPipelineBarrier(
        cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0,
        nullptr);
  }

  void submit_and_wait(const VkCommandBuffer* cmds, int count) {
    auto& dt = omarchy::vk::device_table();
    std::vector<VkSubmitInfo> infos(count);
    for (int i = 0; i < count; ++i) {
      infos[i] = VkSubmitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      infos[i].commandBufferCount = 1;
      infos[i].pCommandBuffers = &cmds[i];
    }
    VKX_CHECK(dt.ResetFences(dev_.handle(), 1, &fence_));
    {
      std::lock_guard<std::mutex> lk(dev_.queue_mutex());
      VKX_CHECK(dt.QueueSubmit(
          dev_.queue(), static_cast<uint32_t>(count), infos.data(), fence_));
    }
    VKX_CHECK(dt.WaitForFences(dev_.handle(), 1, &fence_, VK_TRUE, UINT64_MAX));
  }

  Fixture& fx_;
  int n_;
  omarchy::Device& dev_;
  VkCommandPool pool_{VK_NULL_HANDLE};
  std::vector<VkCommandBuffer> cmds_;
  VkFence fence_{VK_NULL_HANDLE};
  VkDescriptorPool dpool_{VK_NULL_HANDLE};
  VkPipelineLayout layout_{VK_NULL_HANDLE};
  omarchy::ComputeBinding cache_{};
  VkDescriptorSetLayout set_layout_{VK_NULL_HANDLE};
  VkDescriptorSet shared_set_{VK_NULL_HANDLE};
  std::array<omarchy::ComputeBinding, 4> bindings_{};
  bool push_{false};
};

void print_raw(const RawConfig& cfg, int n, double record, double gpu) {
  std::printf(
      "{\"level\": \"raw\", \"label\": \"%s\", \"n\": %d, \"barrier\": \"%s\", "
      "\"desc\": \"%s\", \"push_bytes\": %u, \"rebind\": %s, \"submits\": %d, "
      "\"serial\": %s, \"groups\": %u, \"node\": \"%s\", "
      "\"kernel\": \"%s\", \"push_descriptors\": %s, "
      "\"record_us\": %.2f, \"gpu_us\": %.2f}\n",
      cfg.label, n, barrier_name(cfg.barrier), desc_name(cfg.desc),
      cfg.push_bytes, cfg.rebind ? "true" : "false", cfg.submits,
      cfg.serial ? "true" : "false",
      cfg.groups ? cfg.groups : omarchy::compute_dispatch_group_count(kCount),
      cfg.node == Node::Copy ? "copy" : cfg.node == Node::Fill ? "fill" : "dispatch",
      (std::string(cfg.alternate ? "alternate" : kernel_name(cfg.kernel)) +
       (cfg.count0 ? "/count0" : ""))
          .c_str(),
      omarchy::device().push_descriptors() ? "true" : "false", record, gpu);
  std::fflush(stdout);
}

int run_raw(Fixture& fx, int n) {
  RawBench bench(fx, n);
  const uint32_t full = sizeof(omarchy::ComputeParams);
  std::vector<RawConfig> configs = {
      {Barrier::PrePost, Desc::Alloc, full, true, 1, false, "production"},
      {Barrier::None, Desc::Reuse, 0, false, 1, false, "floor"},
      {Barrier::None, Desc::Alloc, full, true, 1, false, "no-barrier"},
      {Barrier::Pre, Desc::Alloc, full, true, 1, false, "pre-only"},
      {Barrier::Full, Desc::Alloc, full, true, 1, false, "full-per-dispatch"},
      {Barrier::Narrow, Desc::Alloc, full, true, 1, false, "narrow-per-dispatch"},
      {Barrier::PrePost, Desc::Bind, full, true, 1, false, "desc-bind-only"},
      {Barrier::PrePost, Desc::Reuse, full, true, 1, false, "desc-reuse"},
      {Barrier::PrePost, Desc::Alloc, 16, true, 1, false, "push16"},
      {Barrier::PrePost, Desc::Alloc, 0, true, 1, false, "push0"},
      {Barrier::PrePost, Desc::Alloc, full, false, 1, false, "no-rebind"},
      {Barrier::None, Desc::Reuse, 0, false, 10, false, "floor-10-submits"},
      {Barrier::None, Desc::Reuse, 0, false, 100, false, "floor-100-submits"},
      {Barrier::None, Desc::Reuse, 0, false, n, false, "floor-1-per-submit"},
      {Barrier::None, Desc::Reuse, 0, false, 100, true, "floor-100-serial"},
      {Barrier::None, Desc::Reuse, 0, false, n, true, "floor-1-per-submit-serial"},
      {Barrier::PrePost, Desc::Alloc, full, true, 10, false, "production-10-submits"},
      {Barrier::PrePost, Desc::Alloc, full, true, 100, true, "production-100-serial"},
      {Barrier::None, Desc::Reuse, 0, false, 1, false, "floor-1-group", 1},
      {Barrier::None, Desc::Reuse, 0, false, 1, false, "floor-64-groups", 64},
      {Barrier::None, Desc::Reuse, 0, false, 1, false, "floor-1024-groups", 1024},
      {Barrier::PrePost, Desc::Alloc, full, true, 1, false, "production-1-group", 1},
      {Barrier::PrePost, Desc::Alloc, full, true, 1, false, "production-64-groups", 64},
      {Barrier::None, Desc::Reuse, 0, false, 1, false, "copy-no-barrier", 0, Node::Copy},
      {Barrier::PrePost, Desc::Reuse, 0, false, 1, false, "copy-prepost", 0, Node::Copy},
      {Barrier::None, Desc::Reuse, 0, false, 1, false, "fill-no-barrier", 0, Node::Fill},
      {Barrier::PrePost, Desc::Reuse, 0, false, 1, false, "fill-prepost", 0, Node::Fill},
      {Barrier::None, Desc::Reuse, full, false, 1, false, "fillkernel-floor", 0, Node::Dispatch, Kernel::Fill},
      {Barrier::PrePost, Desc::Alloc, full, true, 1, false, "fillkernel-production", 0, Node::Dispatch, Kernel::Fill},
      {Barrier::PrePost, Desc::Alloc, full, true, 1, false, "fillkernel-production-64-groups", 64, Node::Dispatch, Kernel::Fill},
      {Barrier::PrePost, Desc::Alloc, full, true, 1, false, "production-count0", 0, Node::Dispatch, Kernel::Add, true},
      {Barrier::PrePost, Desc::Alloc, full, true, 1, false, "fillkernel-production-count0", 0, Node::Dispatch, Kernel::Fill, true},
      {Barrier::PrePost, Desc::Alloc, full, true, 1, false, "rope-production", 0, Node::Dispatch, Kernel::Rope},
      {Barrier::PrePost, Desc::Alloc, full, true, 1, false, "rope-production-count0", 0, Node::Dispatch, Kernel::Rope, true},
      {Barrier::PrePost, Desc::Alloc, full, true, 1, false, "rope-grid-production", 0, Node::Dispatch, Kernel::RopeGrid},
      {Barrier::PrePost, Desc::Alloc, full, true, 1, false, "rope-grid-production-count0", 0, Node::Dispatch, Kernel::RopeGrid, true},
      {Barrier::PrePost, Desc::Alloc, full, true, 1, false, "copy-rows-production", 0, Node::Dispatch, Kernel::CopyRows},
      {Barrier::PrePost, Desc::Alloc, full, true, 1, false, "copy-rows-production-count0", 0, Node::Dispatch, Kernel::CopyRows, true},
      {Barrier::PrePost, Desc::Alloc, full, true, 1, false, "copy-grid-production", 0, Node::Dispatch, Kernel::CopyGrid},
      {Barrier::PrePost, Desc::Alloc, full, true, 1, false, "copy-grid-production-count0", 0, Node::Dispatch, Kernel::CopyGrid, true},
      {Barrier::PrePost, Desc::Alloc, full, true, 1, false, "rmsnorm-production", 0, Node::Dispatch, Kernel::RmsNorm},
      {Barrier::PrePost, Desc::Alloc, full, true, 1, false, "rmsnorm-production-count0", 0, Node::Dispatch, Kernel::RmsNorm, true},
      {Barrier::PrePost, Desc::Alloc, full, true, 1, false, "rmsnorm-sg-production", 0, Node::Dispatch, Kernel::RmsNormSg},
      {Barrier::PrePost, Desc::Alloc, full, true, 1, false, "rmsnorm-sg-production-count0", 0, Node::Dispatch, Kernel::RmsNormSg, true},
      {Barrier::PrePost, Desc::Alloc, full, true, 1, false, "alternate-production", 0, Node::Dispatch, Kernel::Add, false, true},
      {Barrier::PrePost, Desc::Alloc, full, true, 1, false, "alternate-production-count0", 0, Node::Dispatch, Kernel::Add, true, true},
      {Barrier::None, Desc::Alloc, full, true, 1, false, "alternate-no-barrier", 0, Node::Dispatch, Kernel::Add, false, true},
  };
  bench.run(configs[0]);
  const auto& caps = omarchy::device().capabilities();
  const bool subgroup32 = caps.subgroup_size == 32u &&
      (caps.subgroup_operations & VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT) != 0;
  for (const auto& cfg : configs) {
    if (n % cfg.submits != 0) {
      continue;
    }
    if (cfg.kernel == Kernel::RmsNormSg && !subgroup32) {
      continue;
    }
    double best_record = 1e30;
    double best_gpu = 1e30;
    for (int t = 0; t < kTrials; ++t) {
      auto [record, gpu] = bench.run(cfg);
      best_record = std::min(best_record, record);
      best_gpu = std::min(best_gpu, gpu);
    }
    print_raw(cfg, n, best_record, best_gpu);
  }
  // The transfer rows overwrite `out`; the production row restores it
  // before the value check.
  bench.run(configs[0]);
  return fx.check() ? 0 : 2;
}

// ------------------------------------------------------------ encoder --

int run_encoder(Fixture& fx, int n) {
  auto& encoder = omarchy::get_command_encoder(fx.stream);
  const auto params = add_params();
  auto bindings = fx.bindings();
  auto run = [&]() {
    warm_cpu();
    auto t0 = Clock::now();
    for (int i = 0; i < n; ++i) {
      encoder.dispatch_compute(
          omarchy::ComputeKernel::ElementwiseF16, bindings, params,
          omarchy::compute_dispatch_group_count(kCount));
    }
    double record = seconds_since(t0);
    encoder.commit();
    encoder.synchronize();
    return std::pair<double, double>{record, seconds_since(t0)};
  };
  run();
  double best_record = 1e30;
  double best_total = 1e30;
  for (int t = 0; t < kTrials; ++t) {
    auto [record, total] = run();
    best_record = std::min(best_record, record);
    best_total = std::min(best_total, total);
  }
  std::printf(
      "{\"level\": \"encoder\", \"n\": %d, \"gated_barriers\": %s, "
      "\"push_descriptors\": %s, \"record_us\": %.2f, \"total_us\": %.2f}\n",
      n, omarchy::env_flag("MLX_OMARCHY_GATED_BARRIERS") ? "true" : "false",
      omarchy::device().push_descriptors() ? "true" : "false",
      best_record / n * 1e6, best_total / n * 1e6);
  std::fflush(stdout);
  return fx.check() ? 0 : 2;
}

// ----------------------------------------------------------- shaderdb --

// Compile every kernel in enum order and print its name first, so an
// AGX_MESA_DEBUG=shaderdb MESA_SHADER_CACHE_DISABLE=1 run pairs each
// driver "CS shader:" line with the kernel that produced it.
static const char* const kKernelNames[] = {
    "ElementwiseF32",
    "ElementwiseF16",
    "ElementwiseBF16",
    "CastF16F32",
    "CastBoolF32",
    "CastBoolI32",
    "CastBoolF16",
    "CastBoolBF16",
    "CastF32F16",
    "CastBF16F32",
    "CastF32BF16",
    "CastBF16F16",
    "CastF16BF16",
    "CastI32F32",
    "CastU32F32",
    "CastF32I32",
    "CastI32F16",
    "CastF16I32",
    "CastI32BF16",
    "CastBF16I32",
    "ReduceF32",
    "ReduceF16",
    "ReduceBF16",
    "MatmulF32",
    "MatmulF16",
    "MatmulBF16",
    "FillF32",
    "FillF16",
    "FillBF16",
    "SoftmaxF32",
    "SoftmaxF16",
    "SoftmaxBF16",
    "LogSumExpF32",
    "LogSumExpF16",
    "LogSumExpBF16",
    "SelectF32",
    "SelectF16",
    "SelectBF16",
    "SelectI32",
    "SelectBool",
    "SelectComplex64",
    "CompareF32",
    "CompareF16",
    "CompareBF16",
    "CompareI32",
    "CompareU32",
    "CompareI64",
    "CompareComplex",
    "LogicalOrBool",
    "CompareBool",
    "CopyGeneralF32",
    "CopyGeneralF16",
    "CopyGeneralBF16",
    "CopyGeneralU32",
    "CopyGeneralBool",
    "CopyGeneralU8",
    "CopyGeneralU16",
    "CopyGeneralU64",
    "ArgReduceF32",
    "ArgReduceF16",
    "ArgReduceBF16",
    "ArgReduceI8",
    "ArgReduceU8",
    "ArgReduceI16",
    "ArgReduceU16",
    "ArgReduceI32",
    "ArgReduceU32",
    "ArgReduceI64",
    "ArgReduceU64",
    "ArangeF32",
    "ArangeF16",
    "ArangeBF16",
    "ArangeI32",
    "ArangeU32",
    "ArangeI64",
    "ArangeU64",
    "SortF32",
    "SortF16",
    "SortBF16",
    "SortC64",
    "ArgSortF32",
    "ArgSortF16",
    "ArgSortBF16",
    "ArgSortC64",
    "SortI32",
    "SortU32",
    "SortI8",
    "SortU8",
    "SortI16",
    "SortU16",
    "ArgSortI32",
    "ArgSortU32",
    "ArgSortI8",
    "ArgSortU8",
    "ArgSortI16",
    "ArgSortU16",
    "RandomBitsU32",
    "ElementwiseI32",
    "ElementwiseU32",
    "ScanF32",
    "ScanF16",
    "ScanBF16",
    "SearchSortedF32",
    "SearchSortedF16",
    "SearchSortedBF16",
    "SearchSortedI32",
    "SearchSortedU32",
    "ReduceGeneralF32",
    "ReduceGeneralF16",
    "ReduceGeneralBF16",
    "ReduceGeneralI32",
    "ReduceGeneralU32",
    "ReduceGeneralI8",
    "ReduceGeneralU8",
    "ReduceGeneralI16",
    "ReduceGeneralU16",
    "ReduceGeneralI64",
    "ReduceGeneralU64",
    "ReduceGeneralComplex",
    "AnyAllF32",
    "AnyAllF16",
    "AnyAllBF16",
    "AnyAllI32",
    "AnyAllU32",
    "AnyAllBool",
    "ScanGeneralF32",
    "ScanGeneralF16",
    "ScanGeneralBF16",
    "ScanGeneralI32",
    "ScanGeneralU32",
    "ScanGeneralBool",
    "ScanGeneralI8",
    "ScanGeneralU8",
    "ScanGeneralI16",
    "ScanGeneralU16",
    "ScanGeneralI64",
    "ScanGeneralU64",
    "ScanGeneralComplex",
    "HadamardF32",
    "HadamardF16",
    "HadamardBF16",
    "QmmF32",
    "QmmF16",
    "QmmBF16",
    "DequantF32",
    "DequantF16",
    "ConvF32",
    "ConvF16",
    "ConvBF16",
    "GatherAxisU32",
    "GatherAxisI64",
    "GatherAxisF16",
    "GatherAxisBF16",
    "GatherAxisComplex64",
    "ScatterU32",
    "ScatterF16",
    "ScatterBF16",
    "ScatterComplex64",
    "ScatterAxisU32",
    "ScatterAxisF16",
    "ScatterAxisBF16",
    "ScatterAxisComplex64",
    "MaskedScatterU32",
    "MaskedScatterF16",
    "MaskedScatterBF16",
    "ScatterMultiU32",
    "ScatterMultiF16",
    "ScatterMultiBF16",
    "ScatterTripleU32",
    "ScatterTripleF16",
    "ScatterTripleBF16",
    "ScatterBoolTriple",
    "ScatterGeneralU32",
    "ScatterGeneralF16",
    "ScatterGeneralBF16",
    "ScatterGeneralBool",
    "ScatterGeneralU8",
    "ScatterGeneralI8",
    "ScatterGeneralU16",
    "ScatterGeneralI16",
    "SliceUpdateReduceF32",
    "SliceUpdateReduceF16",
    "SliceUpdateReduceBF16",
    "SliceUpdateReduceU32",
    "TakeF32",
    "TakeF16",
    "TakeBF16",
    "TakeU32",
    "TakeU16",
    "TakeI64",
    "TakeComplex64",
    "TakeMultiF32",
    "TakeMultiF16",
    "TakeMultiBF16",
    "TakeMultiU32",
    "TakeMultiU16",
    "TakeMultiI64",
    "TakeMultiComplex64",
    "ClearU32",
    "BlockMaskF32",
    "GatherMmF32",
    "GatherMmF16",
    "GatherMmBF16",
    "SegmentedMmF32",
    "SegmentedMmF16",
    "SegmentedMmBF16",
    "GatherQmmF32",
    "GatherQmmF16",
    "GatherQmmBF16",
    "GatherQmmNbF32",
    "GatherQmmNbF16",
    "GatherQmmNbBF16",
    "FftF32",
    "FftRealF32",
    "FftStageF32",
    "FastRmsNormF32",
    "FastRmsNormF16",
    "FastRmsNormBF16",
    "FastLayerNormF32",
    "FastLayerNormF16",
    "FastLayerNormBF16",
    "FastRmsNormVjpDxF32",
    "FastRmsNormVjpDxF16",
    "FastRmsNormVjpDxBF16",
    "FastLayerNormVjpDxF32",
    "FastLayerNormVjpDxF16",
    "FastLayerNormVjpDxBF16",
    "FastRmsNormVjpDwF32",
    "FastRmsNormVjpDwF16",
    "FastRmsNormVjpDwBF16",
    "FastLayerNormVjpDwF32",
    "FastLayerNormVjpDwF16",
    "FastLayerNormVjpDwBF16",
    "CrossEntropyVjpF32",
    "CrossEntropyVjpF16",
    "CrossEntropyVjpBF16",
    "CrossEntropyF32",
    "CrossEntropyF16",
    "CrossEntropyBF16",
    "FastRopeF32",
    "FastRopeF16",
    "FastRopeBF16",
    "FastRopeFreqsF32",
    "FastRopeFreqsF16",
    "FastRopeFreqsBF16",
    "Fp8ToF32",
    "Fp8ToF16",
    "Fp8ToBF16",
    "Fp8FromF32",
    "Fp8FromF16",
    "Fp8FromBF16",
    "LinalgCholeskyF32",
    "LinalgInverseF32",
    "LinalgLuF32",
    "LinalgQrF32",
    "LinalgEighF32",
    "LinalgEigF32",
    "LinalgSvdF32",
    "LinalgSvdFinalizeF32",
    "FastRmsNormVjpDwReduceF32",
    "FastRmsNormVjpDwReduceF16",
    "FastRmsNormVjpDwReduceBF16",
    "ComplexElementwise",
    "ComplexReal",
    "ComplexImag",
    "CastF32Complex64",
    "CastI32Complex64",
    "CastU32Complex64",
    "CastBoolComplex64",
    "CastF16Complex64",
    "CastBF16Complex64",
    "CastComplex64F32",
    "FillComplex64",
    "CopyGeneralComplex64",
    "ComplexAbs",
    "ComplexAbsAsComplex",
    "ScatterFAddF32",
    "ScatterFAddF16",
    "ScatterFAddBF16",
    "ScatterFAddMultiF32",
    "ScatterFAddTripleF32",
    "ScatterFAddGeneralF32",
    "ScatterFCasF32",
    "ScatterFCasF16",
    "ScatterFCasBF16",
    "ScatterFCasMultiF32",
    "ScatterFCasTripleF32",
    "ScatterFCasGeneralF32",
    "ScatterBool",
    "ScatterBoolMulti",
    "ScatterAxisFAddF32",
    "ScatterAxisFAddF16",
    "ScatterAxisFAddBF16",
    "ScatterAxisFCasF32",
    "ScatterAxisFCasF16",
    "ScatterAxisFCasBF16",
    "ScatterAxisBool",
    "QmmVecF32",
    "QmmVecF16",
    "QmmVecBF16",
    "QmmVecSubgroupF32",
    "QmmVecSubgroupF16",
    "QmmVecSubgroupBF16",
    "QmmTileF32",
    "QmmTileF16",
    "QmmTileBF16",
    "FusedChainF32",
    "FusedChainF16",
    "QuantizeF32",
    "QuantizeF16",
    "ReduceGeneralBool",
    "CastIntW1W1",
    "CastIntW1W2",
    "CastIntW1W4",
    "CastIntW1W8",
    "CastIntW2W1",
    "CastIntW2W2",
    "CastIntW2W4",
    "CastIntW2W8",
    "CastIntW4W1",
    "CastIntW4W2",
    "CastIntW4W4",
    "CastIntW4W8",
    "CastIntW8W1",
    "CastIntW8W2",
    "CastIntW8W4",
    "CastIntW8W8",
    "CompareU64",
    "ElementwiseI8",
    "ElementwiseU8",
    "ElementwiseI16",
    "ElementwiseU16",
    "ElementwiseI64",
    "ElementwiseU64",
    "CompareI8",
    "CompareU8",
    "CompareI16",
    "CompareU16",
    "ScatterU8",
    "ScatterI8",
    "ScatterU16",
    "ScatterI16",
    "ScatterMultiU8",
    "ScatterMultiI8",
    "ScatterMultiU16",
    "ScatterMultiI16",
    "ScatterTripleU8",
    "ScatterTripleI8",
    "ScatterTripleU16",
    "ScatterTripleI16",
    "FillU64",
    "FillU16",
    "SortMergeF32",
    "SortMergeF16",
    "SortMergeBF16",
    "SortMergeI32",
    "SortMergeU32",
    "ArgSortMergeF32",
    "ArgSortMergeF16",
    "ArgSortMergeBF16",
    "ArgSortMergeC64",
    "ArgSortMergeI32",
    "ArgSortMergeU32",
    "QuantizeFpF32",
    "QuantizeFpF16",
    "QuantizeFpBF16",
    "DequantFpF32",
    "DequantFpF16",
    "DequantFpBF16",
    "QmmFpF32",
    "QmmFpF16",
    "QmmFpBF16",
    "QmmVecFpF32",
    "QmmVecFpF16",
    "QmmVecFpBF16",
    "QmmVecSubgroupFpF32",
    "QmmVecSubgroupFpF16",
    "QmmVecSubgroupFpBF16",
    "QmmTileFpF32",
    "QmmTileFpF16",
    "QmmTileFpBF16",
    "GatherQmmNbFpF32",
    "GatherQmmNbFpF16",
    "GatherQmmNbFpBF16",
    "GatherQmmNbFpHgsF32",
    "GatherQmmNbFpHgsF16",
    "GatherQmmNbFpHgsBF16",
    "MatmulComplex64",
    "QmmVecQ4WordF32",
    "QmmVecQ4WordF16",
    "QmmVecQ4WordBF16",
    "QmmVecQ4WordSubgroupF32",
    "QmmVecQ4WordSubgroupF16",
    "QmmVecQ4WordSubgroupBF16",
    "QmmTileRbF16",
    "QuantizeBF16",
    "DequantBF16",
    "SortI64",
    "SortU64",
    "ArgSortI64",
    "ArgSortU64",
    "SortMergeI64",
    "SortMergeU64",
    "ArgSortMergeI64",
    "ArgSortMergeU64",
    "SdpaDecodeF16",
    "SdpaDecodePartialF16",
    "SdpaDecodeCombineF16",
    "SdpaDecodeSubgroupF16",
    "SdpaDecodePartialSubgroupF16",
    "QmmVecQ4V2F32",
    "QmmVecQ4V2F16",
    "QmmVecQ4V2BF16",
    "QmmVecQ4V2SubgroupF32",
    "QmmVecQ4V2SubgroupF16",
    "QmmVecQ4V2SubgroupBF16",
    "QmmGemm64F16",
    "QmmGemm32K4F16",
    "MatmulGemmF32",
    "MatmulGemmF16",
    "MatmulGemmBF16",
    "QmmGemm64T84F16",
    "QmmGemm128x64T84F16",
    "QmmGemm64T88F16",
    "QmmGemm128T88F16",
    "QmmGemm64Bk16F16",
    "FastRmsNormSubgroupF32",
    "FastRmsNormSubgroupF16",
    "FastRmsNormSubgroupBF16",
};

int run_shaderdb(int first) {
  auto& compute = omarchy::device().compute();
  const int count = static_cast<int>(omarchy::ComputeKernel::Count);
  const int named = static_cast<int>(sizeof(kKernelNames) / sizeof(*kKernelNames));
  for (int i = first; i < count; ++i) {
    std::fprintf(stderr, "KERNEL %s\n", i < named ? kKernelNames[i] : "?");
    std::fflush(stderr);
    try {
      compute.pipeline(static_cast<omarchy::ComputeKernel>(i));
    } catch (const std::exception& e) {
      std::fprintf(stderr, "KERNEL %s failed: %s\n", i < named ? kKernelNames[i] : "?", e.what());
    }
    std::fflush(stderr);
  }
  return 0;
}

// ---------------------------------------------------------------- ops --

int run_ops(Fixture& fx, int n) {
  auto& encoder = omarchy::get_command_encoder(fx.stream);
  auto& counters = omarchy::trace::counters();
  auto measure = [&](const char* label, auto&& build) {
    build();
    double best = 1e30;
    uint64_t submits = 0;
    for (int t = 0; t < kTrials; ++t) {
      uint64_t s0 = counters.vk_submissions.load();
      warm_cpu();
      auto t0 = Clock::now();
      build();
      double total = seconds_since(t0);
      submits = counters.vk_submissions.load() - s0;
      best = std::min(best, total);
    }
    std::printf(
        "{\"level\": \"ops\", \"label\": \"%s\", \"n\": %d, \"submits\": %llu, "
        "\"total_us\": %.2f}\n",
        label, n, static_cast<unsigned long long>(submits), best / n * 1e6);
    std::fflush(stdout);
  };
  measure("chain", [&]() {
    array x = fx.a;
    for (int i = 0; i < n; ++i) {
      x = add(x, fx.b, fx.stream);
    }
    eval(x);
    encoder.synchronize();
  });
  measure("independent", [&]() {
    std::vector<array> ys;
    ys.reserve(n);
    for (int i = 0; i < n; ++i) {
      ys.push_back(add(fx.a, fx.b, fx.stream));
    }
    eval(ys);
    encoder.synchronize();
  });
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  if (!gpu::is_available()) {
    std::fprintf(stderr, "no qualifying Vulkan device\n");
    return 1;
  }
  set_default_device(Device::gpu);
  const std::string level = argc > 1 ? argv[1] : "all";
  const int n = argc > 2 ? std::atoi(argv[2]) : 1000;
  Fixture fx;
  int rc = 0;
  if (level == "raw" || level == "all") {
    rc |= run_raw(fx, n);
  }
  if (level == "encoder" || level == "all") {
    rc |= run_encoder(fx, n);
  }
  if (level == "ops" || level == "all") {
    rc |= run_ops(fx, n);
  }
  if (level == "shaderdb") {
    rc |= run_shaderdb(argc > 2 ? n : 0);
  }
  return rc;
}
