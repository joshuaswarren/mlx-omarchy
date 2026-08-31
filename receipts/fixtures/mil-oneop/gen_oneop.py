"""Generate a hand-authored one-op MIL program (fp16 elementwise add) + weights.bin.

Dialect replicated from the retained coremlc-3520.5.1 capture at
/private/tmp/aneforge-add-cache/.../model.mil (runtime input + fp16 const,
single add). Shape [1,512] differs from the captured [1,2048] on purpose.
"""

import struct
import sys

SHAPE = int(sys.argv[1]) if len(sys.argv) > 1 else 512
VALUE = 0.25

def fp16_bits(values):
    import numpy as np

    return np.asarray(values, dtype=np.float16).tobytes()


def main():
    weights_payload = fp16_bits([VALUE] * SHAPE)

    file_header = struct.pack("<II56x", 1, 2)
    blob_record = struct.pack(
        "<IIQQQ32x",
        0xDEADBEEF,      # blob magic
        1,               # blob version
        len(weights_payload),
        0,
        64 + 64,         # absolute offset of raw data (0x80)
    )
    with open("weights.bin", "wb") as handle:
        handle.write(file_header)
        handle.write(blob_record)
        handle.write(weights_payload)

    shape = f"[1, {SHAPE}]"
    mil = (
        "program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3520.4.1\"},"
        " {\"coremlc-version\", \"3520.5.1\"}})]\n"
        "{\n"
        f"    func main<ios18>(tensor<fp16, {shape}> t1) {{\n"
        f"        tensor<fp16, {shape}> t0 = const()[name = string(\"t0\"),"
        f" val = tensor<fp16, {shape}>(BLOBFILE(path = string(\"@model_path/weights.bin\"),"
        " offset = uint64(64)))];\n"
        f"        tensor<fp16, {shape}> t2 = add(x = t1, y = t0)[name = string(\"t2\")];\n"
        "    } -> (t2);\n"
        "}\n"
    )
    with open("model.mil", "w") as handle:
        handle.write(mil)

    print(f"weights.bin={64 + 64 + len(weights_payload)} bytes, "
          f"model.mil={len(mil)} bytes")


if __name__ == "__main__":
    main()
