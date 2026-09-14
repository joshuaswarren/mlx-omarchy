import hashlib
import pathlib
import sys

import mlx.core as mx

core = pathlib.Path(mx.__file__).resolve()
lib = core.parent / "lib" / "libmlx.so"
data = lib.read_bytes()
print("mx", mx.__version__)
print("core", core)
print("core sha256", hashlib.sha256(core.read_bytes()).hexdigest())
print("libmlx", lib)
print("libmlx.so sha256", hashlib.sha256(data).hexdigest())
print("profiler literal", b"MLX_OMARCHY_GPU_PROFILE" in data)
print("device", mx.default_device())
if b"MLX_OMARCHY_GPU_PROFILE" not in data:
    sys.exit("installed libmlx.so lacks the profiler literal")
