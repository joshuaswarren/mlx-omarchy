#!/usr/bin/env python3
"""Execute the minted ac-head bundle through the resident worker transport.

Differential-window driver (EncoderSubmitRepair × EncoderCompilerCoverage):
loads island-attn-ac-head-L00 via ResidentAneWorker, runs each staged arm
directory (arm inputs: q/k/cond/relpos/a_fill .bin + .json shape files),
dumps each arm's smax output + per-arm timing. CPU/host prep only; the
actual execution needs the ANE device (run inside a flock hold).

Usage:
  python3 ac_head_differential.py --bundles DIR --arms DIR --out DIR \
      [--runner-dump DIR] [--gpu-ref DIR]
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ane_resident import ResidentAneWorker  # noqa: E402

BUNDLE = "island-attn-ac-head-L00"
INPUTS = ("a_fill", "q", "k", "cond", "relpos")


def load_tensor(directory: Path, name: str):
    meta = json.loads((directory / f"{name}.json").read_text())
    import numpy as np
    buf = (directory / f"{name}.bin").read_bytes()
    arr = np.frombuffer(buf, dtype=np.dtype(meta["dtype"]))
    return arr.reshape(meta["shape"])


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", required=True,
                        help="bundle root containing " + BUNDLE)
    parser.add_argument("--bundle-name", default=BUNDLE,
                        help="bundle identifier (default: " + BUNDLE + ")")
    parser.add_argument("--worker", required=True)
    parser.add_argument("--libane", required=True)
    parser.add_argument("--arms", required=True,
                        help="directory of arm input sets")
    parser.add_argument("--out", required=True)
    parser.add_argument("--deadline-ms", type=int, default=20000)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bundle_name = args.bundle_name
    # --bundles accepts either the bundles PARENT or the bundle directory
    # itself; resolve once and refuse anything ambiguous before the worker
    # ever launches (a nested copy made the C++ loader fail mid-window).
    candidates = [Path(args.bundles) / bundle_name, Path(args.bundles)]
    bundle_dir = next(
        (c for c in candidates if (c / "manifest.json").is_file()), None)
    if bundle_dir is None:
        raise SystemExit(
            f"cannot resolve {bundle_name}: {args.bundles} must be either "
            "the bundle directory itself (manifest.json directly inside) or "
            "its parent, without nested copies")
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    stray = sorted(d.name for d in bundle_dir.iterdir() if d.is_dir())
    if stray:
        raise SystemExit(
            f"{bundle_dir} contains unexpected subdirectories {stray}; the "
            "bundle layout is manifest.json + program-N.anec files only")
    print(f"bundle {bundle_name} graph {manifest.get('graph_hash', '?')[:16]} "
          f"programs {len(manifest.get('programs', []))}")
    actual_name = manifest.get("name")
    if actual_name != bundle_name:
        raise SystemExit(
            f"FAIL: manifest.name={actual_name!r} does not match "
            f"--bundle-name={bundle_name!r}")

    arms = sorted(p for p in Path(args.arms).iterdir() if p.is_dir())
    if not arms:
        raise SystemExit(f"no arm directories under {args.arms}")

    # Use the context manager (ResidentAneWorker.__exit__) so a worker
    # crash mid-submit terminates the process exactly once instead of
    # re-raising on close() of an already-dead session.
    with ResidentAneWorker(
        worker=Path(args.worker),
        libane=Path(args.libane),
        bundles={bundle_name: bundle_dir},
        scratch=out / "scratch",
        deadline_ms=args.deadline_ms,
    ) as session:
        for arm in arms:
            inputs = {}
            for name in INPUTS:
                meta = json.loads((arm / f"{name}.json").read_text())
                raw = (arm / f"{name}.bin").read_bytes()
                import numpy as np
                expected = int(np.prod(meta["shape"]) *
                               np.dtype(meta["dtype"]).itemsize)
                if len(raw) != expected:
                    raise SystemExit(
                        f"FAIL: {arm.name}/{name} on-disk bytes={len(raw)} "
                        f"!= fixture-metadata byte count={expected}")
                # Frame the raw bytes (len(bytes) == byte count). A
                # multidim memoryview would make len() the dimension
                # count, corrupting the wire "in <name> <len>" header.
                inputs[name] = raw
            started = time.monotonic_ns()
            results = session.submit(bundle_name, arm.name, inputs, ["smax"])
            elapsed = time.monotonic_ns() - started
            payload = bytes(results["smax"])
            digest = hashlib.sha256(payload).hexdigest()
            (out / f"{arm.name}.smax.bin").write_bytes(payload)
            record = {
                "arm": arm.name, "elapsed_ms": round(elapsed / 1e6, 2),
                "smax_sha256": digest, "smax_bytes": len(payload),
            }
            print(json.dumps(record))
            (out / f"{arm.name}.json").write_text(json.dumps(record))
    print(f"ARMS={len(arms)} smax dumps in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
