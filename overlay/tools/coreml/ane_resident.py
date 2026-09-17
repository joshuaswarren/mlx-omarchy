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

import mmap
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

# Worker-side phase timers reported on the job line (microseconds).
# serve-side: recv_us = reading inline payloads from stdin, submit_us =
# the whole AneWorker::submit call (frames + device loop + output
# frames), emit_us = writing output payloads to stdout. Device-side
# (inside submit_us): pack_us = copying inputs into the device,
# exec_us = execute ioctls, read_us = copying results back, crecv_us =
# reading the request frames themselves. Everything the elapsed wall
# does not explain at a layer is waiting/serialization at that layer.
_PHASE_KEYS = (
    "recv_us",
    "submit_us",
    "emit_us",
    "pack_us",
    "exec_us",
    "read_us",
    "crecv_us",
)


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
        self.phase_us = {key: 0 for key in _PHASE_KEYS}
        self.log: list[dict] = []

        self._process: subprocess.Popen | None = None
        self._stderr_path = self.scratch / "resident-worker.stderr"
        self._stderr = None
        self._banner: list[str] = []
        self._inbox = bytearray()
        self._batch_until: float | None = None
        # Shared-memory payload transport. The client creates two memfd
        # regions and passes them to the worker at start; the worker's
        # "shm ok" acknowledgement switches the session onto the shm
        # path. Anything that fails (old kernel, old worker, oversize
        # payload) falls back to the inline pipe protocol.
        self._shm_in = None
        self._shm_out = None
        self._shm_in_fd = None
        self._shm_out_fd = None
        self._shm_bytes = 0
        self._shm_ok = False
        self.transport = "inline"

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._process is not None:
            raise ResidentWorkerError("resident session is already started")
        self.scratch.mkdir(parents=True, exist_ok=True)
        shm_bytes = int(os.environ.get("ANE_WORKER_SHM_BYTES", 64 * 1024 * 1024))
        pass_fds: tuple[int, ...] = ()
        if os.environ.get("ANE_WORKER_SHM", "1") != "0":
            try:
                in_fd = os.memfd_create("ane-worker-shm-in")
                out_fd = os.memfd_create("ane-worker-shm-out")
                os.ftruncate(in_fd, shm_bytes)
                os.ftruncate(out_fd, shm_bytes)
                self._shm_in = mmap.mmap(in_fd, shm_bytes)
                self._shm_out = mmap.mmap(out_fd, shm_bytes)
                self._shm_in_fd = in_fd
                self._shm_out_fd = out_fd
                self._shm_bytes = shm_bytes
                pass_fds = (in_fd, out_fd)
            except (OSError, ValueError):
                for fd in (self._shm_in_fd, self._shm_out_fd):
                    if fd is not None:
                        os.close(fd)
                self._shm_in = self._shm_out = None
                self._shm_in_fd = self._shm_out_fd = None
                self._shm_bytes = 0
                pass_fds = ()
        argv = [
            str(self.worker),
            "--serve",
            "--libane", str(self.libane),
            "--deadline-ms", str(self.deadline_ms),
            "--iterations", str(self.iterations),
        ]
        if self._shm_in is not None:
            argv += [
                "--shm-in", f"{self._shm_in_fd}={self._shm_bytes}",
                "--shm-out", f"{self._shm_out_fd}={self._shm_bytes}",
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
            pass_fds=pass_fds,
        )
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
        # A worker with mapped regions answers right after the load
        # report. An old worker that ignored the flags sends nothing:
        # a short bounded wait decides the transport, then never again.
        if self._shm_in is not None:
            deadline = time.monotonic() + 2.0
            stream = self._process.stdout
            while time.monotonic() < deadline:
                ready, _, _ = select.select(
                    [stream], [], [], deadline - time.monotonic()
                )
                if not ready:
                    break
                chunk = os.read(stream.fileno(), 1 << 16)
                if not chunk:
                    self._die(
                        "resident worker closed its output during the "
                        "shm handshake"
                    )
                self._inbox += chunk
                if b"\n" in self._inbox:
                    line = self._readline("shm handshake")
                    if line.startswith("shm ok "):
                        self._shm_ok = True
                        self.transport = "shm"
                    break
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
    ) -> dict[str, bytes]:
        """One bounded submit against an already-resident bundle.

        Payloads travel inline on the worker's stdin and stdout. They
        used to be staged through files in ``scratch``; measured on
        m1-test-host, the worker's own read of a 3 MB island input cost 14-19 ms
        per submit -- more than the process launch residency saves -- so
        the file round trip is gone and only the bytes cross.
        """
        if self._process is None:
            raise ResidentWorkerError("no resident session is open")
        if bundle not in self.bundles:
            raise ResidentWorkerError(f"unknown resident bundle {bundle!r}")

        job = ["submit", bundle]
        in_bytes = 0
        ordered = list(inputs.items())
        # ponytail: single-slot in-ring, offset restarts per submit. Safe
        # because the protocol is strictly synchronous (the runner writes
        # round N+1 only after reading round N's outputs). Double-buffer
        # the ring if a caller ever pipelines submits.
        shm_fits = (
            self._shm_ok
            and self._shm_in is not None
            and sum(len(payload) for _, payload in ordered) <= self._shm_bytes
        )
        if shm_fits:
            offset = 0
            for name, payload in ordered:
                self._shm_in[offset : offset + len(payload)] = payload
                job += ["--shmin", f"{name}={offset}:{len(payload)}"]
                offset += len(payload)
                in_bytes += len(payload)
            for name in outputs:
                job += ["--shout", name]
        else:
            for name, payload in ordered:
                job += ["--inline", f"{name}={len(payload)}"]
                in_bytes += len(payload)
            for name in outputs:
                job += ["--emit", name]

        request = bytearray(" ".join(job).encode() + b"\n")
        if not shm_fits:
            for _, payload in ordered:
                request += payload

        started = time.monotonic_ns()
        self._write_bytes(bytes(request))
        write_ns = time.monotonic_ns() - started
        results: dict[str, bytes] = {}
        out_bytes = 0
        while True:
            line = self._readline(f"job report for {tag}")
            if line.startswith("out "):
                _, name, length = line.split(" ", 2)
                payload = self._read_exact(int(length), f"output {name}")
                results[name] = payload
                out_bytes += len(payload)
                continue
            if line.startswith("shmout "):
                _, name, offset, length = line.split(" ", 3)
                span = self._shm_out[int(offset) : int(offset) + int(length)]
                results[name] = bytes(span)
                out_bytes += int(length)
                continue
            break
        elapsed = time.monotonic_ns() - started
        read_ns = elapsed - write_ns

        self.submissions += 1
        self.exec_ns += elapsed
        self.input_bytes += in_bytes
        # The worker's own job report carries its internal split: elapsed_ms
        # covers recv+stage+exec+read+send-back inside the child; stage_ms is
        # its input staging share and save_ms its output retrieval share;
        # the _PHASE_KEYS are the fine-grained serve/device split.
        report_fields = {}
        for token in line.replace("\n", " ").split():
            key, sep, value = token.partition("=")
            if sep and key in ("status", "elapsed_ms", "stage_ms", "save_ms", *_PHASE_KEYS):
                report_fields[key] = value
        for key in _PHASE_KEYS:
            if key in report_fields:
                try:
                    self.phase_us[key] += int(report_fields[key])
                except ValueError:
                    pass
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
            "transport": self.transport,
            "phase_us": dict(self.phase_us),
        }

    # --------------------------------------------------------------- plumbing
    def _write(self, line: str) -> None:
        self._write_bytes((line + "\n").encode())

    def _write_bytes(self, payload: bytes) -> None:
        assert self._process is not None and self._process.stdin is not None
        try:
            self._process.stdin.write(payload)
            self._process.stdin.flush()
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

    def _read_exact(self, count: int, what: str) -> bytes:
        while len(self._inbox) < count:
            self._fill(what)
        payload = bytes(self._inbox[:count])
        del self._inbox[:count]
        return payload

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
        for region in (self._shm_in, self._shm_out):
            if region is not None:
                try:
                    region.close()
                except OSError:
                    pass
        for fd in (self._shm_in_fd, self._shm_out_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self._shm_in = self._shm_out = None
        self._shm_in_fd = self._shm_out_fd = None
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
