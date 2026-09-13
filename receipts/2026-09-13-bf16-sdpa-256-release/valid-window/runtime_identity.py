#!/usr/bin/env python3
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import pathlib
import sys
import zipfile

wheel = pathlib.Path(sys.argv[1]).resolve()
expected_python = pathlib.Path(sys.argv[2]).absolute()
expected_prefix = pathlib.Path(sys.argv[3]).resolve()
expected_stamp = sys.argv[4]
assert pathlib.Path(sys.executable).absolute() == expected_python
assert pathlib.Path(sys.prefix).resolve() == expected_prefix
assert "PYTHONPATH" not in os.environ
assert "LD_LIBRARY_PATH" not in os.environ

with zipfile.ZipFile(wheel) as archive:
    names = archive.namelist()
    record_name = next(name for name in names if name.endswith(".dist-info/RECORD"))
    records = {row[0]: row[1] for row in csv.reader(io.TextIOWrapper(archive.open(record_name)))}
    members = [name for name in names if name.endswith(".so")]
    wheel_hashes = {}
    for name in members:
        data = archive.read(name)
        digest = hashlib.sha256(data).digest()
        encoded = base64.urlsafe_b64encode(digest).decode().rstrip("=")
        assert records[name] == "sha256=" + encoded, (name, records.get(name))
        wheel_hashes[name] = hashlib.sha256(data).hexdigest()

import mlx.core as mx
mapped = sorted({line.split()[-1] for line in pathlib.Path("/proc/self/maps").read_text().splitlines()
                 if line.split()[-1].endswith(".so") and
                 ("/mlx/core." in line.split()[-1] or line.split()[-1].endswith("/mlx/lib/libmlx.so"))})
assert len(mapped) == 2, mapped
mapped_hashes = {}
for value in mapped:
    path = pathlib.Path(value).resolve()
    assert path.is_relative_to(expected_prefix), (path, expected_prefix)
    member = next(name for name in members if name.endswith("/" + path.name))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == wheel_hashes[member], (path, digest, wheel_hashes[member])
    mapped_hashes[str(path)] = digest
version = importlib.metadata.version("mlx-omarchy")
assert expected_stamp in version, version
device = mx.device_info()
assert device["device_name"] == "Apple M1 (G13G B1)", device
print(json.dumps({
    "python": str(expected_python),
    "prefix": str(expected_prefix),
    "version": version,
    "wheel": str(wheel),
    "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
    "record_name": record_name,
    "record_so_members": wheel_hashes,
    "loaded_mappings": mapped_hashes,
    "pythonpath_cleared": True,
    "ld_library_path_cleared": True,
    "device": device,
}, indent=2, sort_keys=True))
