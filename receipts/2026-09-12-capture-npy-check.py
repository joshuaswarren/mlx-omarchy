"""Run on macOS: python3 THIS_FILE path/to/capture/main.swift."""

import ast
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

source = Path(sys.argv[1]).read_text()
helpers = source.split("// MARK: - args", 1)[0].replace("import ParakeetTDT\n", "")
check = r'''
var values: [Float] = [1, 2, -99, 3, 4, -99]
try values.withUnsafeMutableBufferPointer { buffer in
    let array = try MLMultiArray(dataPointer: buffer.baseAddress!, shape: [2, 2],
                                dataType: .float32, strides: [3, 1])
    try writeNpy(URL(fileURLWithPath: CommandLine.arguments.last!), mlArray: array)
}
'''
with tempfile.TemporaryDirectory(prefix="capture-npy-") as temporary:
    root = Path(temporary)
    script = root / "check.swift"
    output = root / "strided.npy"
    script.write_text(helpers + check)
    subprocess.run(["swift", str(script), str(output)], check=True, timeout=60)
    data = output.read_bytes()
    assert data[:8] == b"\x93NUMPY\x01\x00"
    end = 10 + int.from_bytes(data[8:10], "little")
    assert end % 64 == 0 and data[end - 1] == 10
    header = ast.literal_eval(data[10:end].decode("ascii"))
    assert header == {"descr": "<f4", "fortran_order": False, "shape": (2, 2)}
    assert struct.unpack("<4f", data[end:]) == (1.0, 2.0, 3.0, 4.0)
print("PASS: strided tensor values, NPY shape/dtype, newline and alignment")
