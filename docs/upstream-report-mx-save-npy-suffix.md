# Upstream report draft: `mx.save` silently appends `.npy` to non-`.npy` filenames

Status: draft, not filed. Written 2026-09-01 against `mlx-omarchy`
(`mlx` pinned at `1f8e74e3f12f31365464a6867c6579f0e9b29d85`, MLX 0.32.2).
Verified also against current `ml-explore/mlx` `main`: `python/src/load.cpp`
`mlx_save_helper` and `mlx/io/load.cpp` `save(std::string, array)` are
unchanged.

## Summary

`mx.save(file, arr)` documents and implements only the `.npy` format, but it
accepts any filename and silently rewrites one: when `file` does not end in
`.npy`, it appends `.npy`. The caller asked for `x.safetensors`; the array
lands in `x.safetensors.npy`. Nothing warns, and the requested path never
appears, so a naive `mx.save(p, a); mx.load(p)` round-trip fails on the load
side with `[load_safetensors] Failed to open file`.

The size of the array does not matter. The report that surfaced this looked
like a zero-size quirk, but zero-size and non-zero-size arrays behave
identically:

```python
import mlx.core as mx
mx.save("empty.safetensors", mx.zeros((0,)))
mx.save("full.safetensors", mx.ones((2, 2)))
# Files on disk: empty.safetensors.npy, full.safetensors.npy
# Neither x.safetensors exists.
# mx.load("full.safetensors") raises the same
# [load_safetensors] Failed to open file as the zero-size case.
```

## Where it comes from

`mlx/io/load.cpp`, `void save(std::string file, array a)`:

```cpp
// Add .npy to file name if it is not there
if (file.length() < 4 || file.substr(file.length() - 4, 4) != ".npy")
  file += ".npy";
```

The Python binding `mlx_save_helper` (`python/src/load.cpp`) routes every
string or path straight to this function. There is no extension-based format
dispatch anywhere in `mx.save`; the safetensors writer is reachable only
through the separate `mx.save_safetensors(file, dict)` API, whose C++ side
(`mlx/io/safetensors.cpp`) appends `.safetensors` the same way.

## Why it is worth reporting

The blind suffix is not a bug by itself; writing a differently named file
without telling the caller is. Two low-cost options:

1. Warn (or raise) when the requested name does not end in `.npy`, so the
   caller learns the real output path instead of debugging a missing file
   later. `save_safetensors` has the same habit with `.safetensors`.
2. Alternatively, teach `mx.load` to try the `.npy` sibling when the named
   file is absent. This is a larger contract change and probably the wrong
   direction; the warning is the smaller fix.

## Workaround

Use the format-specific API:

```python
mx.save_safetensors("x.safetensors", {"w": a})   # not mx.save
w = mx.load("x.safetensors")["w"]
```

`mx.save_safetensors` round-trips zero-size arrays correctly, so no local
change was made in `mlx-omarchy`; the behavior is documented in
`docs/compatibility.md` instead.
