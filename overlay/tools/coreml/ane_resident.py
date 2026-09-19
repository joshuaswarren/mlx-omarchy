# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Client for the resident ANE worker (plan sections 24-27, phase 8).

One supervised ``mlx-omarchy-ane-worker --serve`` process serves every
submit of a pass: each named bundle is parsed once, its programs are
loaded onto the device once, and they stay resident, so N island
submits cost one process launch, one bundle load, and one device load
instead of N of each.

This is not an inference server (section 25). There is no socket, no
listener, and no user-managed daemon: the worker is a private child of
this client, it only ever reads jobs from its own stdin, every submit is
bounded by the worker's wall-clock deadline, and a failure is reported
and ends the session -- never retried.

Payloads travel inline on the worker stdin and stdout. They do not
go through host files.
"""

from __future__ import annotations

import fcntl
import os
import select
import subprocess
import time
from pathlib import Path
from typing import Mapping, Sequence


class ResidentWorkerError(RuntimeError):
    """The resident worker refused or failed; the reason is named."""


# The client's own guard on top of the worker's per-submit deadline: if
# the supervising CLI itself stops answering, the client must not wait
# forever either.
_CLIENT_GRACE_MS = 5000


class ResidentAneWorker:
    """A resident worker session over a fixed set of named bundles.

    ``submit`` takes and returns raw bytes; tensor packing, dtypes and
    array conversion stay with the caller, exactly as with the
    one-process-per-submit path.
    """

    def __init__(
        self,
        worker: Path,
        libane: Path,
        bundles: Mapping[str, Path],
        scratch: Path,
        deadline_ms: int = 20000,
        iterations: int = 1,
    ):
        if not bundles:
            raise ResidentWorkerError("a resident session needs at least one bundle")
        if deadline_ms <= 0 or iterations <= 0:
            raise ResidentWorkerError(
                "a resident session needs a positive deadline and iteration count"
            )
        self.worker = Path(worker)
        self.libane = Path(libane)
        self.bundles = {name: Path(path) for name, path in bundles.items()}
        self.scratch = Path(scratch)
        self.deadline_ms = deadline_ms
        self.iterations = iterations

        # Counters in the shape the parity harness reports (section 42).
        self.submissions = 0
        self.batch_opens = 0
        self.batch_rounds = 0
        self.worker_starts = 0
        self.bundle_loads = 0
        self.device_program_loads = 0
        self.timeouts = 0
        self.input_bytes = 0
        self.output_bytes = 0
        self.exec_ns = 0
        self.start_ns = 0
        self.close_ns = 0
        self.log: list[dict] = []

        self._process: subprocess.Popen | None = None
        self._stderr_path = self.scratch / "resident-worker.stderr"
        self._stderr = None
        self._banner: list[str] = []
        self._inbox = bytearray()
        self._batch_until: float | None = None

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._process is not None:
            raise ResidentWorkerError("resident session is already started")
        self.scratch.mkdir(parents=True, exist_ok=True)
        argv = [
            str(self.worker),
            "--serve",
            "--libane", str(self.libane),
            "--deadline-ms", str(self.deadline_ms),
            "--iterations", str(self.iterations),
        ]
        for name, path in self.bundles.items():
            argv += ["--bundle", f"{name}={path}"]

        self._stderr = self._stderr_path.open("wb")
        started = time.monotonic_ns()
        self._process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
        )
        # The default 64 KiB pipe buffer makes every multi-MB payload
        # round-trip a wake-up storm between this client and the worker
        # (hundreds of context switches per submit); 1 MiB is the max
        # without privileges. Best-effort: an old kernel refusing the
        # ioctl only keeps the old latency.
        set_pipe_sz = getattr(fcntl, "F_SETPIPE_SZ", 1031)
        for stream in (self._process.stdin, self._process.stdout):
            try:
                fcntl.fcntl(stream.fileno(), set_pipe_sz, 1 << 20)
            except OSError:
                pass
        self.worker_starts += 1
        # One "resident bundle=..." line per bundle, then the load report.
        for _ in self.bundles:
            line = self._readline("bundle report")
            if not line.startswith("resident bundle="):
                self._die(f"expected a resident bundle report, got {line!r}")
            self._banner.append(line)
            self.bundle_loads += 1
        line = self._readline("load report")
        if not line.startswith("resident loaded "):
            self._die(f"expected the resident load report, got {line!r}")
        self._banner.append(line)
        self.device_program_loads = _loaded_programs(line)
        self.start_ns = time.monotonic_ns() - started

    def close(self) -> dict:
        if self._process is None:
            raise ResidentWorkerError("no resident session is open")
        started = time.monotonic_ns()
        self._write("quit")
        line = self._readline("release report")
        if not line.startswith("resident released "):
            self._die(f"expected the resident release report, got {line!r}")
        code = self._process.wait()
        self.close_ns = time.monotonic_ns() - started
        self._finish()
        if code != 0:
            raise ResidentWorkerError(
                f"resident worker exited {code} after releasing: {self._stderr_tail()}"
            )
        return {"released": line, "exit": code}

    def __enter__(self) -> "ResidentAneWorker":
        self.start()
        return self

    def __exit__(self, kind, value, traceback) -> None:
        if self._process is None:
            return
        if kind is None:
            self.close()
            return
        self._terminate()

    # -------------------------------------------------------------- submits
    def submit(
        self,
        bundle: str,
        tag: str,
        inputs: Mapping[str, bytes],
        outputs: Sequence[str],
) -> dict[str, bytearray]:
        """One bounded submit against an already-resident bundle.

        Payloads travel inline on the worker's stdin and stdout. They
        used to be staged through files in ``scratch``; measured on
        jwm1, the worker's own read of a 3 MB island input cost 14-19 ms
        per submit -- more than the process launch residency saves -- so
        the file round trip is gone and only the bytes cross.
        """
        if self._process is None:
            raise ResidentWorkerError("no resident session is open")
        if bundle not in self.bundles:
            raise ResidentWorkerError(f"unknown resident bundle {bundle!r}")

        job = ["submit", bundle]
        in_bytes = 0
        ordered = [(name, memoryview(payload))
                   for name, payload in inputs.items()]
        for name, view in ordered:
            # nbytes, not len(): the runner passes contiguous numpy
            # buffers, whose len() is a shape dimension, not a byte count.
            job += ["--inline", f"{name}={view.nbytes}"]
            in_bytes += view.nbytes
        for name in outputs:
            job += ["--emit", name]

        request = " ".join(job).encode() + b"\n"
        started = time.monotonic_ns()
        # Payloads go straight from the caller's buffers: no bytes() or
        # bytearray assembly copies on the send side.
        self._writev([request] + [view for _, view in ordered])
        write_ns = time.monotonic_ns() - started
        results: dict[str, bytearray] = {}
        out_bytes = 0
        while True:
            line = self._readline(f"job report for {tag}")
            if line.startswith("out "):
                _, name, length = line.split(" ", 2)
                payload = self._read_exact(int(length), f"output {name}")
                results[name] = payload
                out_bytes += len(payload)
                continue
            break
        elapsed = time.monotonic_ns() - started
        read_ns = elapsed - write_ns

        self.submissions += 1
        self.exec_ns += elapsed
        self.input_bytes += in_bytes
        # The worker's own job report carries its internal split: elapsed_ms
        # covers recv+stage+exec+read+send-back inside the child; stage_ms is
        # its input staging share and save_ms its output retrieval share.
        report_fields = {}
        for token in line.replace("\n", " ").split():
            key, sep, value = token.partition("=")
            if sep and key in ("status", "elapsed_ms", "stage_ms", "save_ms"):
                report_fields[key] = value
        record = {
            "tag": tag,
            "bundle": bundle,
            "elapsed_ns": elapsed,
            "write_ns": write_ns,
            "read_ns": read_ns,
            "report": line,
            "input_bytes": in_bytes,
            "output_bytes": out_bytes,
            **report_fields,
        }
        self.log.append(record)

        if not line.startswith("job status=0"):
            if "deadline" in line:
                self.timeouts += 1
            self._die(f"resident submit {tag} ({bundle}) failed: {line}")
        missing = [name for name in outputs if name not in results]
        if missing:
            self._die(
                f"resident submit {tag} ({bundle}) produced no bytes for "
                f"{missing}"
            )
        self.output_bytes += out_bytes
        return results

    def begin_batch(self, deadline_ms: int) -> None:
        """Open a batch scope: one absolute deadline bounds N submits.

        The bounded unit is the batch, not each submit inside it, so a
        caller can turn a whole island pass of per-layer submits into
        one deadline-bounded unit. Everything else about the safety
        model is unchanged: a missed deadline or one failed submit ends
        the session and is reported, never retried.
        """
        if self._process is None:
            raise ResidentWorkerError("no resident session is open")
        if deadline_ms <= 0:
            raise ResidentWorkerError("a batch scope needs a positive deadline")
        self._write(f"batch {deadline_ms}")
        line = self._readline("batch open report")
        if not line.startswith("batch opened "):
            self._die(f"expected a batch open report, got {line!r}")
        self._batch_until = time.monotonic() + deadline_ms / 1000
        self.batch_opens += 1

    def end_batch(self) -> int:
        """Close the batch scope; returns the rounds the batch served."""
        if self._process is None:
            raise ResidentWorkerError("no resident session is open")
        if self._batch_until is None:
            raise ResidentWorkerError("no batch scope is open")
        self._write("batch-end")
        line = self._readline("batch close report")
        if not line.startswith("batch closed "):
            self._die(f"expected a batch close report, got {line!r}")
        self._batch_until = None
        try:
            rounds = int(line.rsplit("=", 1)[-1])
        except ValueError:
            rounds = 0
        self.batch_rounds += rounds
        return rounds

    def counters(self) -> dict:
        """The submit-boundary counters this session measured."""
        return {
            "worker_starts": self.worker_starts,
            "bundle_loads": self.bundle_loads,
            "device_program_loads": self.device_program_loads,
            "submissions": self.submissions,
            "timeouts": self.timeouts,
            "batch_opens": self.batch_opens,
            "batch_rounds": self.batch_rounds,
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
            "exec_ns": self.exec_ns,
            "start_ns": self.start_ns,
            "close_ns": self.close_ns,
        }

    # --------------------------------------------------------------- plumbing
    def _write(self, line: str) -> None:
        self._write_bytes((line + "\n").encode())

    def _write_bytes(self, payload: bytes) -> None:
        self._writev([payload])

    def _writev(self, parts) -> None:
        assert self._process is not None and self._process.stdin is not None
        try:
            stream = self._process.stdin
            for part in parts:
                stream.write(part)
            stream.flush()
        except (BrokenPipeError, ValueError):
            self._die("resident worker closed its input before the job was sent")

    def _fill(self, what: str) -> None:
        """Read whatever the worker has ready, inside the client guard."""
        assert self._process is not None and self._process.stdout is not None
        stream = self._process.stdout
        deadline = time.monotonic() + (self.deadline_ms + _CLIENT_GRACE_MS) / 1000
        if self._batch_until is not None:
            deadline = self._batch_until + _CLIENT_GRACE_MS / 1000
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.timeouts += 1
                self._die(
                    f"resident worker did not produce the {what} within "
                    f"{self.deadline_ms + _CLIENT_GRACE_MS} ms"
                )
            ready, _, _ = select.select([stream], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(stream.fileno(), 1 << 20)
            if not chunk:
                self._die(
                    f"resident worker closed its output before the {what}: "
                    f"{self._stderr_tail()}"
                )
            self._inbox += chunk
            return

    def _readline(self, what: str) -> str:
        while True:
            end = self._inbox.find(b"\n")
            if end >= 0:
                line = bytes(self._inbox[:end]).decode(errors="replace")
                del self._inbox[: end + 1]
                return line
            self._fill(what)

    def _read_exact(self, count: int, what: str) -> bytearray:
        # Bulk payloads land directly in the final buffer: the old
        # path copied every byte twice more (inbox append + bytes()
        # slice), which is real time at ~180 MB of outputs per pass.
        out = bytearray(count)
        got = 0
        if self._inbox:
            take = min(len(self._inbox), count)
            out[:take] = self._inbox[:take]
            del self._inbox[:take]
            got = take
        if got < count:
            assert self._process is not None and self._process.stdout is not None
            stream = self._process.stdout
            fd = stream.fileno()
            deadline = time.monotonic() + (
                self.deadline_ms + _CLIENT_GRACE_MS) / 1000
            if self._batch_until is not None:
                deadline = self._batch_until + _CLIENT_GRACE_MS / 1000
            view = memoryview(out)
            while got < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.timeouts += 1
                    self._die(
                        f"resident worker did not produce the {what} within "
                        f"{self.deadline_ms + _CLIENT_GRACE_MS} ms"
                    )
                ready, _, _ = select.select([stream], [], [], remaining)
                if not ready:
                    continue
                n = os.readv(fd, [view[got:]])
                if not n:
                    self._die(
                        f"resident worker closed its output before the {what}: "
                        f"{self._stderr_tail()}"
                    )
                got += n
        return out

    def _stderr_tail(self, limit: int = 400) -> str:
        if self._stderr is not None:
            self._stderr.flush()
        try:
            return self._stderr_path.read_text(errors="replace").strip()[-limit:]
        except OSError:
            return ""

    def _terminate(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.kill()
            process.wait()
        self._finish()

    def _finish(self) -> None:
        process = self._process
        if process is not None:
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None
        self._process = None

    def _die(self, message: str) -> None:
        tail = self._stderr_tail()
        self._terminate()
        raise ResidentWorkerError(
            message if not tail else f"{message} [worker stderr: {tail}]"
        )


def _loaded_programs(line: str) -> int:
    """Programs the worker reported resident, from its load report."""
    marker = "loaded "
    detail = line.split("detail=", 1)[-1]
    position = detail.find(marker)
    if position < 0:
        return 0
    digits = detail[position + len(marker):].split(" ", 1)[0]
    try:
        return int(digits)
    except ValueError:
        return 0
