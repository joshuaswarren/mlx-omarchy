#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Add the resident-worker arm to the encoder runner.

The runner (``vulkan_encoder.py``, EncoderParityAne's, living in
/var/tmp/EncoderParityAne on jwm1) is not a repo file, so this script is
the record of what was changed in it: it inserts one class and one pair
of flags by anchored replacement and refuses if any anchor is missing,
then writes the result and the unified diff beside it.

    python3 add_resident_arm.py SOURCE DESTINATION [DIFF]

ResidentIsland keeps AneIsland's submit contract exactly -- same tensor
staging, same dtypes, same ordering -- and only changes who owns the
worker process and the loaded programs.
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

RESIDENT_CLASS = '''

class ResidentIsland:
    """Same submit contract as AneIsland, one resident worker for the pass.

    Plan section 24 lists "reuse model resources" as a worker duty; this is
    that duty applied to the encoder: each island bundle is parsed once, its
    programs are loaded on the device once, and all 24 layers submit against
    the resident programs. Every submit is still bounded by the same
    wall-clock deadline, and a failure ends the run without a retry.
    """

    def __init__(self, worker: Path, libane: Path, bundles: Path, scratch: Path,
                 names, deadline_ms: int = 20000):
        from coreml.ane_resident import ResidentAneWorker

        self.session = ResidentAneWorker(
            worker=worker,
            libane=libane,
            bundles={name: bundles / name for name in names},
            scratch=scratch,
            deadline_ms=deadline_ms,
        )
        self.session.start()
        self.submissions = 0
        self.worker_starts = self.session.worker_starts
        self.input_bytes = 0
        self.output_bytes = 0
        self.exec_ns = 0
        self.timeouts = 0
        self.log: list[dict] = []

    def submit(self, bundle: str, tag: str, inputs: dict, outputs: dict) -> dict:
        """inputs: name -> mx array. outputs: name -> (shape, dtype name)."""
        payloads = {}
        in_bytes = 0
        for name, value in inputs.items():
            mx.eval(value)
            raw = np.ascontiguousarray(np.asarray(value))
            payloads[name] = raw.tobytes()
            in_bytes += raw.nbytes

        started = time.monotonic_ns()
        produced = self.session.submit(
            bundle=bundle, tag=tag, inputs=payloads, outputs=tuple(outputs)
        )
        elapsed = time.monotonic_ns() - started

        self.submissions += 1
        self.exec_ns += elapsed
        self.input_bytes += in_bytes
        record = {
            "tag": tag,
            "bundle": bundle,
            "exit": 0,
            "elapsed_ns": elapsed,
            "stdout": [self.session.log[-1]["report"]],
            "stderr": [],
            "input_bytes": in_bytes,
        }
        self.log.append(record)

        results = {}
        out_bytes = 0
        for name, (shape, dtype_name) in outputs.items():
            raw = produced[name]
            count = 1
            for dim in shape:
                count *= dim
            expect = count * np.dtype(NP_DTYPES[dtype_name]).itemsize
            if len(raw) != expect:
                raise EncoderRunError(
                    f"ANE output {name} for {tag} is {len(raw)} bytes, want {expect}"
                )
            out_bytes += len(raw)
            host = np.frombuffer(raw, dtype=NP_DTYPES[dtype_name], count=count)
            results[name] = mx.array(host).reshape(shape)
        self.output_bytes += out_bytes
        record["output_bytes"] = out_bytes
        return results

    def counters(self) -> dict:
        return self.session.counters()

    def close(self) -> None:
        self.timeouts = self.session.timeouts
        self.session.close()

'''

FLAGS_ANCHOR = '''    parser.add_argument(
        "--no-ane", action="store_true",
        help="Vulkan-only control run: the attention matmuls stay on the GPU.",
    )
'''

FLAGS_REPLACEMENT = FLAGS_ANCHOR + '''    parser.add_argument(
        "--resident", action="store_true",
        help="One resident worker for the pass instead of one process per submit.",
    )
    parser.add_argument(
        "--resident-bundle", action="append", default=[],
        help="Bundle directory name to keep resident; repeatable. "
             "Defaults to island-attn-a-kt and island-pv.",
    )
'''

CONSTRUCT_ANCHOR = '''    island = None
    if not args.no_ane:
        island = AneIsland(
            args.worker, args.libane, args.bundles, args.scratch, args.deadline_ms
        )
'''

CONSTRUCT_REPLACEMENT = '''    island = None
    if not args.no_ane:
        if args.resident:
            island = ResidentIsland(
                args.worker, args.libane, args.bundles, args.scratch,
                args.resident_bundle or ["island-attn-a-kt", "island-pv"],
                args.deadline_ms,
            )
        else:
            island = AneIsland(
                args.worker, args.libane, args.bundles, args.scratch,
                args.deadline_ms,
            )
'''

REPORT_ANCHOR = '''            "exec_ns": island.exec_ns,
            "log": island.log,
        }
'''

REPORT_REPLACEMENT = REPORT_ANCHOR + '''        if isinstance(island, ResidentIsland):
            island.close()
            report["ane"]["resident"] = island.counters()
            report["ane"]["timeouts"] = island.timeouts
'''

CLASS_ANCHOR = "class EncoderRunner:"


def rewrite(source: str) -> str:
    replacements = (
        ("ResidentIsland class", CLASS_ANCHOR, RESIDENT_CLASS.lstrip("\n") + "\n" + CLASS_ANCHOR),
        ("resident flags", FLAGS_ANCHOR, FLAGS_REPLACEMENT),
        ("island construction", CONSTRUCT_ANCHOR, CONSTRUCT_REPLACEMENT),
        ("ane report", REPORT_ANCHOR, REPORT_REPLACEMENT),
    )
    for what, anchor, replacement in replacements:
        count = source.count(anchor)
        if count != 1:
            raise SystemExit(
                f"anchor for {what} appears {count} times, expected exactly one"
            )
        source = source.replace(anchor, replacement)
    return source


def main(argv: list[str]) -> int:
    if len(argv) not in (3, 4):
        raise SystemExit(__doc__)
    source_path = Path(argv[1])
    destination = Path(argv[2])
    original = source_path.read_text()
    rewritten = rewrite(original)
    destination.write_text(rewritten)
    destination.chmod(0o755)
    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            rewritten.splitlines(keepends=True),
            fromfile=f"a/{source_path.name}",
            tofile=f"b/{destination.name}",
        )
    )
    if len(argv) == 4:
        Path(argv[3]).write_text(diff)
    else:
        sys.stdout.write(diff)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
