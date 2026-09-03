#!/usr/bin/env python3
"""Shared helpers for the mlx-omarchy contributor collectors.

`collect_quick.py` and `collect_deep.py` import this module. It owns the
three things both collectors must agree on:

- PII redaction (`Redactor`): usernames, hostnames, home paths, IP
  addresses, MAC addresses, serial numbers, UUIDs, and credential-shaped
  strings never reach an output. Every embedded command output passes
  through it, and its per-kind counts go into the manifest so a reader can
  see what was removed.
- bounded external commands (`run_tool`, `run_python_probe`): a missing or
  hanging tool is recorded data, never a crash, never an unbounded wait.
- deterministic packaging (`build_manifest`, `archive_bytes`): sorted
  members, fixed mtime and owner, gzip mtime 0, so the previewed manifest
  and the written archive agree byte for byte.

Nothing in this module talks to the network. The collectors never upload.
"""

import gzip
import hashlib
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import getpass

SCHEMA_VERSION = 1

MAX_STREAM_LINES = 400
MAX_STREAM_CHARS = 200_000


class Redactor:
    """Replace personally identifying strings with typed placeholders."""

    def __init__(self, hostname=None, username=None, home=None):
        self.hostname = hostname or socket.gethostname()
        try:
            self.username = username or getpass.getuser()
        except Exception:
            self.username = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
        self.home = home or os.path.expanduser("~")
        self.counts = {}
        self._rules = self._build_rules()

    def _note(self, kind):
        self.counts[kind] = self.counts.get(kind, 0) + 1

    def _replace(self, kind, repl):
        def fn(_match):
            self._note(kind)
            return repl
        return fn

    # A dotted quad is not always an address: Vulkan reports
    # `conformanceVersion = 1.4.0.0`, and redacting that destroys real
    # hardware data. Keep the quad when the text right before it is a
    # version assignment; redact every other dotted quad.
    _VERSION_CONTEXT = re.compile(r"(?i)version\s*[:=]\s*$")

    def _ipv4_sub(self, match):
        line_start = match.string.rfind("\n", 0, match.start()) + 1
        prefix = match.string[line_start:match.start()]
        if self._VERSION_CONTEXT.search(prefix):
            return match.group(0)
        self._note("ipv4")
        return "[redacted-ip4]"

    def _build_rules(self):
        rules = []

        def rx(pattern, kind, repl, flags=0):
            rules.append((re.compile(pattern, flags), self._replace(kind, repl)))

        # Cred-shaped assignments first, so a value that also matches a
        # later rule is already gone. Name=NAME VALUE=VALUE keeps the name.
        rx(
            r"(?i)\b([A-Za-z0-9_]*(?:token|secret|passwd|password|api_?key|"
            r"private_?key)[A-Za-z0-9_]*)\s*([:=])\s*(\"[^\"]*\"|\S+)",
            "credential",
            r"\1\2 [redacted]",
        )
        # Known credential shapes without a key name.
        rx(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b", "credential", "[redacted]")
        rx(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b", "credential", "[redacted]")
        rx(r"\bAKIA[0-9A-Z]{16}\b", "credential", "[redacted]")
        rx(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b", "credential", "[redacted]")
        rx(r"\bsk-[A-Za-z0-9_-]{20,}\b", "credential", "[redacted]")
        rx(r"\bBearer\s+\S+", "credential", "Bearer [redacted]")
        rx(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}\b",
           "credential", "[redacted]")
        # Hardware identity.
        rx(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b", "mac", "[redacted-mac]")
        rx(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
           r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b", "uuid", "[redacted-uuid]")
        rx(r"(?i)(\"serial(?:[-_]?number)?\"\s*:\s*\")[^\"]*(\")",
           "serial", r"\1[redacted]\2")
        rx(r"(?i)\b(serial[-_]?number)\s*([=:])\s*(\S+)",
           "serial", r"\1\2 [redacted]")
        # Network identity. IPv6 before IPv4 so embedded v4-in-v6 is gone.
        rx(r"\b(?:fe80|fd[0-9a-f]{2}|fc[0-9a-f]{2})(?::[0-9a-fA-F]{0,4}){1,7}"
           r"(?:%\w+)?\b", "ipv6", "[redacted-ip6]", re.IGNORECASE)
        rx(r"\b(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}\b",
           "ipv6", "[redacted-ip6]")
        rules.append((
            re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?:/\d{1,3})?\b"),
            self._ipv4_sub,
        ))
        # Home paths: this user's home first, then any user's.
        if self.home and self.home != "/" and self.home != "":
            rules.append((
                re.compile(re.escape(self.home)),
                self._replace("home_path", "[home]"),
            ))
        rx(r"(?<![\w.-])/(?:home|Users)/[^/\s:\"'@]+", "home_path", "[home]")
        # Live host and user names, last, so path placeholders above win.
        if self.hostname and len(self.hostname) >= 2:
            rules.append((
                re.compile(r"(?<![\w.-])" + re.escape(self.hostname) +
                           r"(?![\w.-])", re.IGNORECASE),
                self._replace("hostname", "[host]"),
            ))
        if self.username and len(self.username) >= 2:
            rules.append((
                re.compile(r"(?<![\w.-])" + re.escape(self.username) +
                           r"(?![\w.-])", re.IGNORECASE),
                self._replace("username", "[user]"),
            ))
        return rules

    def apply(self, text):
        if not isinstance(text, str):
            text = str(text)
        for pattern, repl in self._rules:
            text = pattern.sub(repl, text)
        return text


def cap_stream(text):
    """Cap one captured output stream so archives stay small."""
    if text is None:
        return ""
    lines = text.splitlines()
    total = len(lines)
    kept = lines[:MAX_STREAM_LINES]
    out = "\n".join(kept)[:MAX_STREAM_CHARS]
    if total > MAX_STREAM_LINES or len(text) > MAX_STREAM_CHARS:
        out += f"\n[truncated: {total} lines, {len(text)} chars captured]"
    return out


def redact_argv(argv, redactor):
    """Record the command shape without values that could carry secrets."""
    return [redactor.apply(str(a)) for a in argv]


def run_tool(argv, redactor, label=None, timeout=30, cwd=None, env=None):
    """Run one external command; record absence, timeout, and output.

    Returns a dict. Never raises for a missing binary or a timeout.
    """
    record = {
        "label": label or argv[0],
        "argv": redact_argv(argv, redactor),
        "available": True,
        "exit_code": None,
        "error": None,
        "duration_ms": None,
        "stdout": "",
        "stderr": "",
    }
    if shutil.which(argv[0]) is None:
        record["available"] = False
        record["error"] = "not-found"
        return record
    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            cwd=cwd,
            env=env,
        )
        record["exit_code"] = proc.returncode
        record["stdout"] = redactor.apply(cap_stream(proc.stdout or ""))
        record["stderr"] = redactor.apply(cap_stream(proc.stderr or ""))
    except subprocess.TimeoutExpired:
        record["error"] = f"timeout after {timeout}s"
    except OSError as exc:
        record["available"] = False
        record["error"] = f"os-error: {exc}"
    record["duration_ms"] = int((time.monotonic() - started) * 1000)
    return record


def run_python_probe(code, redactor, label, timeout=120, env=None):
    """Run a python snippet in a child interpreter, bounded."""
    return run_tool(
        [sys.executable, "-c", code],
        redactor,
        label=label,
        timeout=timeout,
        env=env,
    )


def dump_json(obj):
    """Deterministic JSON text: sorted keys, fixed indent, trailing newline."""
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def json_bytes(obj):
    return dump_json(obj).encode("utf-8")


def archive_bytes(files):
    """Deterministic gzip tarball from {member name: bytes}.

    Sorted members, mtime 0, root owner, empty uname/gname, gzip mtime 0.
    Same files in, same bytes out.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tf:
        for name in sorted(files):
            data = files[name]
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.type = tarfile.REGTYPE
            tf.addfile(info, io.BytesIO(data))
    return gzip.compress(buf.getvalue(), compresslevel=9, mtime=0)


def build_manifest(archive_name, files, extra=None):
    """Manifest describing every member except itself.

    `files` must not contain "manifest.json"; the caller adds it to the
    archive after hashing the manifest bytes themselves.
    """
    entries = []
    for name in sorted(files):
        data = files[name]
        entries.append({
            "path": name,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "manifest_version": 1,
        "tool": "mlx-omarchy collect-deep",
        "archive": archive_name,
        "files": entries,
        "no_network": True,
        "upload": "never; this archive is shared by a human, by hand",
    }
    if extra:
        manifest.update(extra)
    return manifest


def build_payload(kind, quick, manifest, generated_at=None, benchmark=None):
    """Build the strict-schema JSON summary sent with the upload.

    Must match schema/payload-v1.schema.json in services/community-data:
    fixed key set, schema_version pinned, every identity field nullable.
    All values come from already-redacted data.
    """
    host = quick.get("host") or {}
    dt = host.get("devicetree") or {}
    gpu = (quick.get("mesa") or {}).get("gpu") or {}
    mlx = quick.get("mlx") or {}
    distributions = mlx.get("distributions") or {}
    compatible = dt.get("compatible") or []
    # Group by SoC, not by board: an M1 MacBook Pro reports
    # ["apple,j293", "apple,t8103", "apple,arm-platform"], and only
    # apple,t8103 identifies the chip that every other M1 machine shares.
    soc = next((c for c in compatible
                if re.match(r"^apple,t\d{4}", c)), None)
    host_cpu = host.get("cpu_online")
    # The benchmark numbers must ride in the summary, not only inside the
    # archive: the read API serves summaries, so cross-machine comparison
    # is impossible unless the numbers travel with them.
    rows = []
    for row in (benchmark or [])[:16]:
        if not isinstance(row, dict) or not isinstance(row.get("n"), int):
            continue
        rows.append({
            "n": row["n"],
            "tflops": row.get("tflops"),
            "median_ms": row.get("median_ms"),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "generated_at": generated_at or time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arch": host.get("arch"),
        "model": dt.get("model"),
        "chip": soc or (compatible[0] if compatible else None),
        "kernel": host.get("kernel_release"),
        "mesa_driver": gpu.get("driverName"),
        "mesa_device": gpu.get("deviceName"),
        "mlx_version": distributions.get("mlx-omarchy")
            or mlx.get("mlx_version"),
        "mlx_device": mlx.get("default_device"),
        "source_commit": manifest.get("source_commit"),
        "repo_dirty": manifest.get("repo_dirty"),
        "cpu_online": host_cpu if isinstance(host_cpu, int) else None,
        "benchmark": rows,
        "redaction_summary": dict(
            manifest.get("redaction_summary") or {}),
        "files": [
            {key: entry[key] for key in ("path", "bytes", "sha256")}
            for entry in manifest.get("files", [])
        ],
    }


def read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None
