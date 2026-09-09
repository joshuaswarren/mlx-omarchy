## mlx-omarchy hardware report

Machine: Apple MacBook Air (13-inch, [host], 2024) (aarch64, kernel 7.1.13-401.asahi.fc44.aarch64+16k)
Vulkan: llvmpipe (LLVM 22.1.8, 128 bits) / llvmpipe, API 1.4.354
mlx-omarchy: 0.32.2.dev202609071529+f5ba1c8, device Device(cpu, 0)
Source commit: 5516b28cc1cc0600396762d4633c75d537585028
Correctness: 0/6 probe ops pass
Not available on this machine: benchmark, profile
Redaction applied before writing: {"home_path": 24} (names, paths, IPs, MACs, serials, credentials; no upload code)

Members with SHA-256 (see attached archive for full contents):

```
a438cbe729d86c76c46c59d96a5abbaad13bde0cbed56026995c69ff5ae1bc0b  benchmark.json (4406 B)
99e327597b7b68703a3a0adb8d1f129b6fb3e9fb8605bd5973c907dd52855acd  correctness.json (8068 B)
6341fcb021da4bab7236af9854e870b3e6852488089f9bde4052cf20f960554e  environment.json (340 B)
c54d0db04844dd245a261e5f96f9e1e827cac3c0e659b4229b19430ce62a40b4  profile.json (1408 B)
61c9557250a842562bc5b8516beb4d5198d7b5a7ec14691a1d9d74b3d545ba09  quick.json (9180 B)
f8b16ac673505f3b3f858a2e074097de0ff4790a69206652ac4cabf331a52f84  thermal.json (264 B)
```

<details><summary>quick report JSON</summary>

```json
{
  "_redaction": {},
  "ane": {
    "available": false,
    "device_node": false,
    "devicetree": {
      "compatible": null,
      "node": false
    },
    "libane": null,
    "libane_probe": {
      "argv": [
        "sh",
        "-c",
        "ldconfig -p 2>/dev/null | grep -i libane"
      ],
      "available": true,
      "error": null,
      "exit_code": 1,
      "label": "ldconfig libane",
      "stderr": "",
      "stdout": ""
    },
    "libane_version": null,
    "pkgconfig": {
      "argv": [
        "pkg-config",
        "--modversion",
        "libane"
      ],
      "available": true,
      "error": null,
      "exit_code": 1,
      "label": "pkg-config libane",
      "stderr": "Package libane was not found in the pkg-config search path.\nPerhaps you should add the directory containing `libane.pc'\nto the PKG_CONFIG_PATH environment variable\nPackage 'libane' not found",
      "stdout": ""
    }
  },
  "host": {
    "arch": "aarch64",
    "available": true,
    "boot": {
      "iboot1": "mBoot-18000.161.10",
      "iboot2": "iBoot-10151.140.19.700.2",
      "m1n1_stage1": "v1.6.1",
      "m1n1_stage2": "v1.6.1",
      "os_fw": "14.7",
      "system_fw": "unknown"
    },
    "cmdline": "BOOT_IMAGE=/EFI/omarchy/[redacted-uuid]/test-fedora-7.1.13-sHKKiA/vmlinuz root=UUID=[redacted-uuid] rw rootflags=subvol=@ cryptdevice=UUID=[redacted-uuid]:root:allow-discards modprobe.blacklist=appledrm module_blacklist=appledrm loglevel=3 quiet splash",
    "core_shortfall": null,
    "cpu": {
      "hotplug_control": false,
      "offline": 0,
      "offline_list": null,
      "online": 8,
      "online_list": "0-7",
      "possible": 8,
      "possible_list": "0-7",
      "present": 8,
      "present_list": "0-7"
    },
    "cpu_online": 8,
    "devicetree": {
      "compatible": [
        "apple,j613",
        "apple,t8122",
        "apple,arm-platform"
      ],
      "model": "Apple MacBook Air (13-inch, [host], 2024)"
    },
    "kernel_release": "7.1.13-401.asahi.fc44.aarch64+16k",
    "memory_total_mib": 15628,
    "page_size_bytes": 16384
  },
  "mesa": {
    "available": true,
    "device_count": 1,
    "devices": [
      {
        "apiVersion": "1.4.354",
        "conformanceVersion": "1.3.1.1",
        "deviceID": "0x0000",
        "deviceName": "llvmpipe (LLVM 22.1.8, 128 bits)",
        "deviceType": "PHYSICAL_DEVICE_TYPE_CPU",
        "deviceUUID": "[redacted-uuid]",
        "driverID": "DRIVER_ID_MESA_LLVMPIPE",
        "driverInfo": "Mesa 26.1.8 (LLVM 22.1.8)",
        "driverName": "llvmpipe",
        "driverUUID": "[redacted-uuid]",
        "driverVersion": "26.1.8",
        "vendorID": "0x10005"
      }
    ],
    "gpu": {
      "apiVersion": "1.4.354",
      "conformanceVersion": "1.3.1.1",
      "deviceID": "0x0000",
      "deviceName": "llvmpipe (LLVM 22.1.8, 128 bits)",
      "deviceType": "PHYSICAL_DEVICE_TYPE_CPU",
      "driverID": "DRIVER_ID_MESA_LLVMPIPE",
      "driverInfo": "Mesa 26.1.8 (LLVM 22.1.8)",
      "driverName": "llvmpipe",
      "driverVersion": "26.1.8",
      "vendorID": "0x10005"
    },
    "properties": {
      "apiVersion": "1.4.354",
      "conformanceVersion": "1.3.1.1",
      "deviceID": "0x0000",
      "deviceName": "llvmpipe (LLVM 22.1.8, 128 bits)",
      "deviceType": "PHYSICAL_DEVICE_TYPE_CPU",
      "deviceUUID": "[redacted-uuid]",
      "driverID": "DRIVER_ID_MESA_LLVMPIPE",
      "driverInfo": "Mesa 26.1.8 (LLVM 22.1.8)",
      "driverName": "llvmpipe",
      "driverUUID": "[redacted-uuid]",
      "driverVersion": "26.1.8",
      "vendorID": "0x10005"
    },
    "summary": null,
    "vulkaninfo": {
      "argv": [
        "vulkaninfo",
        "--summary"
      ],
      "available": true,
      "error": null,
      "exit_code": 0,
      "label": "vulkaninfo --summary",
      "stderr": "'DISPLAY' environment variable not set... skipping surface info\nWARNING: [Loader Message] Code 0 : ICD for selected physical device does not export vkGetPhysicalDeviceDisplayPropertiesKHR!\nWARNING: [Loader Message] Code 0 : ICD for selected physical device does not export vkGetPhysicalDeviceDisplayPlanePropertiesKHR!\nWARNING: [Loader Message] Code 0 : ICD for selected physical device does not export vkGetPhysicalDeviceDisplayPropertiesKHR!",
      "stdout": "==========\nVULKANINFO\n==========\n\nVulkan Instance Version: 1.4.357\n\n\nInstance Extensions: count = 26\n-------------------------------\nVK_EXT_acquire_drm_display             : extension revision 1\nVK_EXT_acquire_xlib_display            : extension revision 1\nVK_EXT_debug_report                    : extension revision 10\nVK_EXT_debug_utils                     : extension revision 2\nVK_EXT_direct_mode_display             : extension revision 1\nVK_EXT_display_surface_counter         : extension revision 1\nVK_EXT_headless_surface                : extension revision 1\nVK_EXT_layer_settings                  : extension revision 2\nVK_EXT_surface_maintenance1            : extension revision 1\nVK_EXT_swapchain_colorspace            : extension revision 5\nVK_KHR_device_group_creation           : extension revision 1\nVK_KHR_display                         : extension revision 23\nVK_KHR_external_fence_capabilities     : extension revision 1\nVK_KHR_external_memory_capabilities    : extension revision 1\nVK_KHR_external_semaphore_capabilities : extension revision 1\nVK_KHR_get_display_properties2         : extension revision 1\nVK_KHR_get_physical_device_properties2 : extension revision 2\nVK_KHR_get_surface_capabilities2       : extension revision 1\nVK_KHR_portability_enumeration         : extension revision 1\nVK_KHR_surface                         : extension revision 25\nVK_KHR_surface_maintenance1            : extension revision 1\nVK_KHR_surface_protected_capabilities  : extension revision 1\nVK_KHR_wayland_surface                 : extension revision 6\nVK_KHR_xcb_surface                     : extension revision 6\nVK_KHR_xlib_surface                    : extension revision 6\nVK_LUNARG_direct_driver_loading        : extension revision 1\n\nInstance Layers: count = 1\n--------------------------\nVK_LAYER_MESA_device_select Linux device selection layer 1.4.303  version 1\n\nDevices:\n========\nGPU0:\n\tapiVersion         = 1.4.354\n\tdriverVersion      = 26.1.8\n\tvendorID           = 0x10005\n\tdeviceID           = 0x0000\n\tdeviceType         = PHYSICAL_DEVICE_TYPE_CPU\n\tdeviceName         = llvmpipe (LLVM 22.1.8, 128 bits)\n\tdriverID           = DRIVER_ID_MESA_LLVMPIPE\n\tdriverName         = llvmpipe\n\tdriverInfo         = Mesa 26.1.8 (LLVM 22.1.8)\n\tconformanceVersion = 1.3.1.1\n\tdeviceUUID         = [redacted-uuid]\n\tdriverUUID         = [redacted-uuid]"
    }
  },
  "mesa_package": {
    "dpkg": {
      "argv": [
        "dpkg-query",
        "-W",
        "-f=${Package} ${Version}\\n",
        "mesa"
      ],
      "available": false,
      "error": "not-found",
      "exit_code": null,
      "label": "dpkg-query mesa",
      "stderr": "",
      "stdout": ""
    },
    "pacman": {
      "argv": [
        "pacman",
        "-Q",
        "mesa"
      ],
      "available": true,
      "error": null,
      "exit_code": 0,
      "label": "pacman -Q mesa",
      "stderr": "",
      "stdout": "mesa 26.1.8-1"
    }
  },
  "mlx": {
    "available": true,
    "capabilities": null,
    "default_device": "Device(cpu, 0)",
    "distributions": {
      "mlx-omarchy": "0.32.2.dev202609071529+f5ba1c8"
    },
    "import_error": "TypeError: argument should be a str or an os.PathLike object where __fspath__ returns a str, not 'NoneType'",
    "info": null,
    "info_tool": null,
    "mlx_version": "0.32.2.dev202609071529+f5ba1c8",
    "probe": {
      "argv": [
        "[home]/.venvs/mlx-collect/bin/python",
        "-c",
        "\nimport json\nout = {\"distributions\": {}, \"import_ok\": False, \"import_error\": None,\n       \"info_tool\": None, \"default_device\": None, \"mlx_version\": None}\nimport importlib.metadata\nfor dist in (\"mlx-omarchy\", \"mlx\"):\n    try:\n        out[\"distributions\"][dist] = importlib.metadata.version(dist)\n    except Exception:\n        pass\nimport pathlib\ntry:\n    import mlx.core as mx\n    out[\"import_ok\"] = True\n    out[\"default_device\"] = str(mx.default_device())\n    out[\"mlx_version\"] = getattr(mx, \"__version__\", None)\n    import mlx\n    cand = pathlib.Path(mlx.__file__).resolve().parent / \"bin\" / \"mlx-omarchy-info\"\n    if cand.exists():\n        out[\"info_tool\"] = str(cand)\nexcept Exception as exc:\n    out[\"import_error\"] = f\"{type(exc).__name__}: {exc}\"\nprint(json.dumps(out))\n"
      ],
      "available": true,
      "error": null,
      "exit_code": 0,
      "label": "mlx import probe",
      "stderr": "",
      "stdout": "{\"distributions\": {\"mlx-omarchy\": \"0.32.2.dev202609071529+f5ba1c8\"}, \"import_ok\": true, \"import_error\": \"TypeError: argument should be a str or an os.PathLike object where __fspath__ returns a str, not 'NoneType'\", \"info_tool\": null, \"default_device\": \"Device(cpu, 0)\", \"mlx_version\": \"0.32.2.dev202609071529+f5ba1c8\"}"
    }
  },
  "report": "mlx-omarchy-quick",
  "schema_version": 1
}
```

</details>
