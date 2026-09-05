// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// mlx-omarchy-info: capability report and runtime smoke tool for the Omarchy
// Vulkan backend of MLX.
//
//   mlx-omarchy-info                 human-readable capability report
//   mlx-omarchy-info --json          the same report as JSON
//   mlx-omarchy-info --trace-smoke   execute a real Vulkan buffer round trip
//                                  and print the backend trace counters
//   mlx-omarchy-info --device N      report device N instead of the default
//   mlx-omarchy-info --check-bundle D
//                                  validate the ANE bundle at D and print its
//                                  parsed contract as [receipt] lines
//
// Exit codes: 0 success, 1 the backend, smoke, or bundle check failed,
// 2 usage error or bundle directory not found.

#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/ane/bundle.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/device.h"
#include "mlx/stream.h"

namespace omarchy = mlx::core::omarchy;

namespace {
std::string json_escape(const std::string& in) {
  std::string out;
  for (char c : in) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      default:
        out += c;
    }
  }
  return out;
}

std::string version_string(uint32_t version) {
  return std::to_string(VK_API_VERSION_MAJOR(version)) + "." +
      std::to_string(VK_API_VERSION_MINOR(version)) + "." +
      std::to_string(VK_API_VERSION_PATCH(version));
}

bool env_flag(const char* name) {
  const char* v = std::getenv(name);
  if (!v) {
    return false;
  }
  std::string s = v;
  for (auto& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s == "1" || s == "on" || s == "true" || s == "yes";
}

void print_json(uint32_t index) {
  const auto& caps = omarchy::capability_report(index);
  const auto& trace = omarchy::trace::counters();
  bool non_apple = env_flag("MLX_OMARCHY_ALLOW_NON_APPLE");
  std::cout << "{\n";
  auto str_field = [&](const char* k, const std::string& v, bool comma) {
    std::cout << "  \"" << k << "\": \"" << json_escape(v) << "\""
              << (comma ? ",\n" : "\n");
  };
  auto num_field = [&](const char* k, unsigned long long v, bool comma) {
    std::cout << "  \"" << k << "\": " << v << (comma ? ",\n" : "\n");
  };
  str_field("tool", "mlx-omarchy-info", true);
  str_field("device_name", caps.device_name, true);
  str_field("driver_name", caps.driver_name, true);
  str_field("api_version", version_string(caps.api_version), true);
  str_field("driver_version_raw", std::to_string(caps.driver_version), true);
  num_field("vendor_id", caps.vendor_id, true);
  num_field("device_id", caps.device_id, true);
  num_field("queue_family_index", caps.queue_family_index, true);
  num_field("queue_count", caps.queue_count, true);
  num_field(
      "total_memory_bytes",
      static_cast<unsigned long long>(caps.total_memory),
      true);
  num_field(
      "max_allocation_bytes",
      static_cast<unsigned long long>(caps.max_allocation_size),
      true);
  num_field(
      "max_buffer_bytes",
      static_cast<unsigned long long>(caps.max_buffer_size),
      true);
  num_field(
      "max_storage_buffer_range_bytes",
      static_cast<unsigned long long>(caps.max_storage_buffer_range),
      true);
  num_field(
      "max_compute_shared_memory_bytes",
      caps.max_compute_shared_memory_size,
      true);
  num_field(
      "max_compute_work_group_invocations",
      caps.max_compute_work_group_invocations,
      true);
  num_field(
      "max_compute_work_group_size_x",
      caps.max_compute_work_group_size[0],
      true);
  num_field("unified_memory", caps.unified_memory ? 1 : 0, true);
  num_field("host_visible_coherent", caps.host_visible_coherent ? 1 : 0, true);
  num_field("timeline_semaphore", caps.timeline_semaphore ? 1 : 0, true);
  num_field("shader_float16", caps.shader_float16 ? 1 : 0, true);
  num_field("shader_int16", caps.shader_int16 ? 1 : 0, true);
  num_field(
      "storage_buffer_16bit_access",
      caps.storage_buffer_16bit_access ? 1 : 0,
      true);
  num_field(
      "max_per_stage_descriptor_storage_buffers",
      caps.max_per_stage_descriptor_storage_buffers,
      true);
  num_field(
      "max_descriptor_set_storage_buffers",
      caps.max_descriptor_set_storage_buffers,
      true);
  num_field("non_apple_dev_override", non_apple ? 1 : 0, true);
  std::cout << "  \"trace\": {\n";
  num_field(
      "gpu_primitive_dispatches", trace.gpu_primitive_dispatches.load(), true);
  num_field("vk_submissions", trace.vk_submissions.load(), true);
  num_field("vk_buffer_copies", trace.vk_buffer_copies.load(), true);
  num_field("vk_buffer_fills", trace.vk_buffer_fills.load(), false);
  num_field("vk_compute_dispatches", trace.vk_compute_dispatches.load(),
            false);
  num_field(
      "compiled_tape_dispatches", trace.compiled_tape_dispatches.load(), false);
  num_field(
      "compiled_tape_node_evaluations",
      trace.compiled_tape_node_evaluations.load(),
      false);
  std::cout << "  }\n";
  std::cout << "}\n";
}

void print_text(uint32_t index) {
  const auto& caps = omarchy::capability_report(index);
  std::cout << "mlx-omarchy-info\n";
  std::cout << "  device:            " << caps.device_name << "\n";
  std::cout << "  driver:            " << caps.driver_name << "\n";
  std::cout << "  api version:       " << version_string(caps.api_version)
            << "\n";
  std::cout << "  vendor:device id:  0x" << std::hex << caps.vendor_id << ":0x"
            << caps.device_id << std::dec << "\n";
  std::cout << "  queue family:      " << caps.queue_family_index << " ("
            << caps.queue_count << " queues)\n";
  std::cout << "  total memory:      " << (caps.total_memory >> 20)
            << " MiB (device-local heap)\n";
  std::cout << "  unified memory:    " << (caps.unified_memory ? "yes" : "no")
            << "\n";
  std::cout << "  host coherent:     "
            << (caps.host_visible_coherent ? "yes" : "no")
            << (caps.host_visible_coherent
                    ? ""
                    : " (explicit flush/invalidate required)")
            << "\n";
  std::cout << "  timeline semaphore:"
            << (caps.timeline_semaphore ? "yes" : "no") << "\n";
  std::cout << "  shader float16:    " << (caps.shader_float16 ? "yes" : "no")
            << "\n";
  std::cout << "  shader int16:      " << (caps.shader_int16 ? "yes" : "no")
            << "\n";
  std::cout << "  16-bit storage:    "
            << (caps.storage_buffer_16bit_access ? "yes" : "no") << "\n";
  std::cout << "  max compute shm:   " << caps.max_compute_shared_memory_size
            << " B\n";
  std::cout << "  max wg invocations:"
            << caps.max_compute_work_group_invocations << "\n";
  std::cout << "  storage buffer bindings (per stage / per set): "
            << caps.max_per_stage_descriptor_storage_buffers << " / "
            << caps.max_descriptor_set_storage_buffers << "\n";
  if (env_flag("MLX_OMARCHY_ALLOW_NON_APPLE")) {
    std::cout << "  NOTE: MLX_OMARCHY_ALLOW_NON_APPLE=1; this is a"
                 " development-only device, not Omarchy Honeykrisp.\n";
  }
}

// Execute one real buffer round trip: host write -> vkCmdCopyBuffer ->
// fence -> host read back.
int trace_smoke(uint32_t index) {
  auto& dev = omarchy::device(index);
  mlx::core::Stream s = mlx::core::new_stream(mlx::core::Device::gpu);
  auto& encoder = omarchy::get_command_encoder(s);

  constexpr size_t kBytes = 1 << 16;
  auto src = omarchy::allocator().malloc(kBytes);
  auto dst = omarchy::allocator().malloc(kBytes);
  auto* src_ptr = static_cast<uint8_t*>(src.raw_ptr());
  auto* dst_ptr = static_cast<uint8_t*>(dst.raw_ptr());
  if (!src_ptr || !dst_ptr) {
    std::cerr << "[mlx-omarchy-info] smoke failed: unmapped buffer\n";
    return 1;
  }
  for (size_t i = 0; i < kBytes; ++i) {
    src_ptr[i] = static_cast<uint8_t>(i % 251);
  }
  std::memset(dst_ptr, 0, kBytes);

  auto* src_buf = static_cast<omarchy::VulkanBuffer*>(src.ptr());
  auto* dst_buf = static_cast<omarchy::VulkanBuffer*>(dst.ptr());
  encoder.copy_buffer(src_buf->buffer, dst_buf->buffer, kBytes);
  encoder.synchronize();

  int mismatches = 0;
  for (size_t i = 0; i < kBytes; ++i) {
    if (src_ptr[i] != dst_ptr[i]) {
      mismatches++;
    }
  }
  auto& trace = omarchy::trace::counters();
  if (mismatches != 0) {
    std::cerr << "[mlx-omarchy-info] smoke FAILED: " << mismatches
              << " mismatching bytes after device copy\n";
    return 1;
  }
  std::cout << "buffer round trip: OK (" << kBytes << " bytes, device "
            << dev.capabilities().device_name << ")\n";
  std::cout << "trace: vk_submissions=" << trace.vk_submissions.load()
            << " vk_buffer_copies=" << trace.vk_buffer_copies.load()
            << " vk_buffer_fills=" << trace.vk_buffer_fills.load()
            << " vk_compute_dispatches="
            << trace.vk_compute_dispatches.load()
            << " gpu_primitive_dispatches="
            << trace.gpu_primitive_dispatches.load() << "\n";
  return 0;
}

// Prints one tensor list of the parsed bundle contract as [receipt] lines.
void print_tensor_list(
    const char* kind,
    const std::vector<omarchy::ane::AneTensor>& tensors) {
  if (tensors.empty()) {
    std::cout << "[receipt] " << kind << ": none\n";
    return;
  }
  for (const auto& tensor : tensors) {
    std::cout << "[receipt] " << kind << " " << tensor.name << ": index="
              << tensor.index << " dtype=" << tensor.dtype << " shape=[";
    for (size_t i = 0; i < tensor.shape.size(); ++i) {
      if (i > 0) {
        std::cout << ",";
      }
      std::cout << tensor.shape[i];
    }
    std::cout << "] byte_size=" << tensor.byte_size
              << " stride=" << tensor.stride << "\n";
  }
}

uint64_t anec_channel_bytes(
    const omarchy::ane::AneAnecHeader& header,
    uint32_t channel) {
  return uint64_t(header.tiles[channel]) * omarchy::ane::kAneTileAlignment;
}

void print_anec_nchw(
    const omarchy::ane::AneAnecHeader& header,
    uint32_t channel) {
  std::cout << " nchw=[";
  for (size_t i = 0; i < header.nchw[channel].size(); ++i) {
    if (i > 0) {
      std::cout << ",";
    }
    std::cout << header.nchw[channel][i];
  }
  std::cout << "]";
}

void print_anec_channel(
    const char* kind,
    const omarchy::ane::AneTensor& tensor,
    const omarchy::ane::AneAnecHeader& header,
    uint32_t channel) {
  std::cout << "[receipt] anec " << kind << " " << tensor.name
            << ": descriptor_index=" << tensor.index
            << " channel=" << channel
            << " channel_bytes=" << anec_channel_bytes(header, channel);
  print_anec_nchw(header, channel);
  std::cout << "\n";
}

void print_anec_contract(
    const omarchy::ane::AneManifest& manifest,
    const omarchy::ane::AneAnecHeader& header) {
  std::cout << "[receipt] anec: payload_size=" << header.payload_size
            << " td_size=" << header.task_descriptor_size
            << " td_count=" << header.task_descriptor_count
            << " task_size=" << header.task_size
            << " kernel_size=" << header.kernel_size
            << " sources=" << header.source_count
            << " destinations=" << header.destination_count
            << " bootstrap_channel_size=" << header.bootstrap_channel_size
            << "\n";
  for (uint32_t i = 0; i < manifest.outputs.size(); ++i) {
    print_anec_channel("output", manifest.outputs[i], header, 4 + i);
  }
  for (uint32_t i = 0; i < manifest.state.size(); ++i) {
    print_anec_channel(
        "state-destination",
        manifest.state[i],
        header,
        4 + static_cast<uint32_t>(manifest.outputs.size() + i));
  }
  for (uint32_t i = 0; i < manifest.inputs.size(); ++i) {
    print_anec_channel("input", manifest.inputs[i], header, 4 + header.destination_count + i);
  }
  for (uint32_t i = 0; i < manifest.state.size(); ++i) {
    print_anec_channel(
        "state-source",
        manifest.state[i],
        header,
        4 + header.destination_count + static_cast<uint32_t>(manifest.inputs.size() + i));
  }
}

// Validates one bundle directory and prints its parsed contract. This runs
// load_bundle only (see mlx/backend/omarchy/ane/bundle.h): no Vulkan device
// is opened and nothing reaches a descriptor submission.
// Exit codes: 0 valid, 1 named loader error, 2 directory not found.
int check_bundle(const std::string& dir_arg) {
  std::filesystem::path dir(dir_arg);
  try {
    omarchy::ane::AneBundle bundle = omarchy::ane::load_bundle(dir);
    const omarchy::ane::AneManifest& m = bundle.manifest;
    std::cout << "[receipt] bundle: " << dir.string() << "\n";
    std::cout << "[receipt] graph: " << m.name << "\n";
    std::cout << "[receipt] graph_hash: " << m.graph_hash << "\n";
    std::cout << "[receipt] task_descriptors: " << m.task_descriptors << "\n";
    print_tensor_list("input", m.inputs);
    print_tensor_list("output", m.outputs);
    print_tensor_list("state", m.state);
    print_tensor_list("workspace", m.workspace);
    print_anec_contract(m, bundle.anec_header);
    for (const auto& payload : m.payloads) {
      std::cout << "[receipt] payload " << payload.role << ": " << payload.path
                << " sha256=" << payload.sha256
                << " byte_size=" << payload.byte_size << "\n";
    }
    std::cout << "[receipt] compiler: macos_build=" << m.compiler.macos_build
              << " anecompiler=" << m.compiler.anecompiler << "\n";
    std::cout << "[receipt] firmware: min=" << m.firmware.min
              << " max=" << m.firmware.max << "\n";
    std::cout << "[receipt] provenance: repo=" << m.provenance.source_repo
              << " commit=" << m.provenance.source_commit << "\n";
    std::cout << "[receipt] OK: bundle valid\n";
    return 0;
  } catch (const omarchy::ane::AneBundleNotFound&) {
    std::cerr << "[mlx-omarchy-info] bundle not found (region stays on "
                 "Vulkan): "
              << dir.string() << "\n";
    return 2;
  } catch (const std::exception& ex) {
    std::cerr << "[mlx-omarchy-info] check-bundle failed: " << ex.what()
              << "\n";
    return 1;
  }
}

void usage() {
  std::cerr
      << "usage: mlx-omarchy-info [--json] [--trace-smoke] [--device N]\n"
         "       mlx-omarchy-info --check-bundle <dir>\n";
}

} // namespace

int main(int argc, char** argv) {
  uint32_t index = 0;
  bool json = false;
  bool smoke = false;
  std::string check_bundle_dir;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--json") {
      json = true;
    } else if (arg == "--trace-smoke") {
      smoke = true;
    } else if (arg == "--check-bundle" && i + 1 < argc) {
      check_bundle_dir = argv[++i];
    } else if (arg == "--check-bundle") {
      usage();
      return 2;
    } else if (arg == "--device" && i + 1 < argc) {
      index = static_cast<uint32_t>(std::atoi(argv[++i]));
    } else if (arg == "--help" || arg == "-h") {
      usage();
      return 0;
    } else {
      usage();
      return 2;
    }
  }

  if (!check_bundle_dir.empty()) {
    return check_bundle(check_bundle_dir);
  }

  try {
    if (smoke) {
      return trace_smoke(index);
    }
    if (!omarchy::is_available()) {
      std::cerr << "[mlx-omarchy-info] backend unavailable:\n  "
                << omarchy::init_error() << "\n";
      return 1;
    }
    json ? print_json(index) : print_text(index);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "[mlx-omarchy-info] error: " << ex.what() << "\n";
    return 1;
  }
}
