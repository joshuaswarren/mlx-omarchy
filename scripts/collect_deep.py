#!/usr/bin/env python3
"""Collect deep mlx-omarchy diagnostics into one reviewable archive.

For remote performance and correctness work. Self-contained: it needs no
repository build and no model download. Every section detects its own
preconditions, and an unavailable wheel, benchmark binary, or profiling
harness is recorded as data instead of failing the run. Section results
are written as they complete, so a timeout keeps everything that finished.

Sections (schema_version 1):
  quick         the fast capability report (collect_quick.collect)
  environment   python, git commit of the repo, allowlisted env vars
  correctness   fixed-shape mlx ops checked against pure-python references
  benchmark     small matmul timing sweep, plus the kernel spike binary
                when a build is present
  profile       trace smoke, an MLX_OMARCHY_GPU_PROFILE event stream with
                a kernel histogram, and scripts/profile_analyze.py output
  thermal       thermal zone readings before and after the sections

Privacy: every captured value passes through the shared Redactor; raw
logs are capped, never included unredacted. The default run only
previews and uploads nothing. Review the printed manifest, then rerun
with `--out FILE` to write the deterministic archive (sorted members,
fixed mtime, gzip mtime 0: same workspace, same bytes) plus a paste-ready
`FILE.submission.md`. Uploading is always explicit: pass `--submit URL`,
or set MLX_OMARCHY_SUBMIT_URL and type SUBMIT at the prompt. The redacted
summary and archive are sent; the endpoint answers with a public receipt
URL, identical content is deduplicated by its SHA-256, and a failed
upload keeps the local files. The JSON schema does not change with the
sharing path.

"""
import argparse
import glob
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import collect_quick
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
from collect_common import (
    SCHEMA_VERSION,
    Redactor,
    archive_bytes,
    build_manifest,
    build_payload,
    json_bytes,
    read_text,
    run_tool,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SECTION_TIMEOUTS = {
    "quick": 120,
    "environment": 60,
    "correctness": 300,
    "benchmark": 420,
    "profile": 300,
}
SECTION_ORDER = ("quick", "environment", "correctness", "benchmark", "profile")

MAX_STREAM_LINES = 2000

CORRECTNESS_PROBE = r"""
import json, math, time
res = {"available": False, "device": None, "mlx_version": None,
       "import_error": None, "ops": []}
try:
    import mlx.core as mx
except Exception as exc:
    res["import_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(res))
    raise SystemExit(0)
res["available"] = True
res["device"] = str(mx.default_device())
res["mlx_version"] = getattr(mx, "__version__", None)

import sys as _sys
_sys.path.insert(0, __SCRIPTS_DIR__)
from mlx_provenance import installed_provenance

res["provenance"] = installed_provenance()
if res["provenance"].get("verified") == "mismatch":
    res["available"] = False
    res["error"] = ("refusing to emit correctness numbers: "
                    + res["provenance"]["mismatch"])
    print(json.dumps(res))
    raise SystemExit(0)

def close(a, b, tol=1e-5):
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, list):
        return False
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))

def record(name, fn, expect):
    entry = {"op": name, "pass": False, "detail": None, "ms": None}
    start = time.perf_counter()
    try:
        got = fn()
        got = got.tolist() if hasattr(got, "tolist") else got
        entry["pass"] = bool(close(got, expect))
        if not entry["pass"]:
            entry["detail"] = f"expected {expect}, got {got}"
    except Exception as exc:
        entry["detail"] = f"{type(exc).__name__}: {exc}"
    entry["ms"] = round((time.perf_counter() - start) * 1000, 3)
    res["ops"].append(entry)

record("add", lambda: mx.array([1.0, 2.0]) + mx.array([3.0, 4.0]), [4.0, 6.0])
record("matmul", lambda: mx.array([[1.0, 2.0], [3.0, 4.0]])
       @ mx.array([[5.0, 6.0], [7.0, 8.0]]),
       [[19.0, 22.0], [43.0, 50.0]])
record("softmax", lambda: mx.softmax(mx.array([1.0, 2.0, 3.0])),
       [0.09003057, 0.24472847, 0.66524096])
record("argmax", lambda: mx.argmax(mx.array([-1.0, 5.0, 3.0])), 1)
record("cumsum", lambda: mx.cumsum(mx.array([1.0, 2.0, 3.0])),
       [1.0, 3.0, 6.0])

xv, wv = [0.5, -0.25], [[1.0, 2.0], [3.0, 4.0]]
xw = [xv[0] * wv[0][j] + xv[1] * wv[1][j] for j in range(2)]
grad_expect = [[2.0 * xw[j] * xv[i] for j in range(2)] for i in range(2)]

def _grad():
    x = mx.array(xv)
    def f(w):
        return mx.sum((x @ w) ** 2)
    return mx.value_and_grad(f)(mx.array(wv))[1]

record("value_and_grad", _grad, grad_expect)
print(json.dumps(res))
"""

BENCH_PROBE = r"""
import json, statistics, time
res = {"available": False, "device": None, "error": None, "matmul": []}
try:
    import mlx.core as mx
    res["available"] = True
    res["device"] = str(mx.default_device())

    import sys as _sys
    _sys.path.insert(0, __SCRIPTS_DIR__)
    from mlx_provenance import installed_provenance

    res["provenance"] = installed_provenance()
    if res["provenance"].get("verified") == "mismatch":
        res["available"] = False
        res["error"] = ("refusing to emit timing numbers: "
                        + res["provenance"]["mismatch"])
        print(json.dumps(res))
        raise SystemExit(0)
    sync = getattr(mx, "synchronize", None)
    for n in (256, 512, 1024):
        a = mx.random.normal((n, n))
        b = mx.random.normal((n, n))
        for _ in range(2):
            c = a @ b
            mx.eval(c)
            if sync:
                sync()
        times = []
        for _ in range(8):
            start = time.perf_counter()
            c = a @ b
            mx.eval(c)
            if sync:
                sync()
            times.append((time.perf_counter() - start) * 1000)
        med = statistics.median(times)
        res["matmul"].append({
            "n": n, "reps": len(times),
            "median_ms": round(med, 3),
            "min_ms": round(min(times), 3),
            "max_ms": round(max(times), 3),
            "tflops": round(2 * n ** 3 / (med / 1000) / 1e12, 4),
        })
except Exception as exc:
    res["available"] = False
    res["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(res))
"""

PROFILE_PROBE = r"""
import json, os
out = {"available": False, "error": None}
try:
    import mlx.core as mx
    a = mx.random.normal((256, 256))
    b = mx.random.normal((256, 256))
    for _ in range(3):
        c = a @ b
        s = mx.softmax(a)
        mx.eval(c, s)
    sync = getattr(mx, "synchronize", None)
    if sync:
        sync()
    path = os.environ["MLX_OMARCHY_GPU_PROFILE"]
    out["available"] = os.path.exists(path) and os.path.getsize(path) > 0
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(out))
"""


def section_environment(redactor, repo):
    out = {"available": True, "python": platform.python_version(),
           "implementation": platform.python_implementation(),
           "machine": platform.machine(),
           "executable": redactor.apply(sys.executable)}
    out["env"] = [
        {"name": k, "value": redactor.apply(v)[:300]}
        for k, v in sorted(os.environ.items())
        if k.startswith(("MLX_", "MESA_", "VK_"))
    ]
    head = run_tool(["git", "-C", repo, "rev-parse", "HEAD"], redactor,
                    label="git rev-parse HEAD", timeout=15)
    out["source_commit"] = head["stdout"].strip() \
        if head["exit_code"] == 0 else None
    dirty = run_tool(["git", "-C", repo, "status", "--porcelain"], redactor,
                     label="git status count", timeout=15)
    out["repo_dirty"] = dirty["exit_code"] == 0 and bool(dirty["stdout"].strip())
    out["wheels"] = sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(repo, "dist", "*.whl")))
    out["prepared_tree"] = os.path.isdir(os.path.join(repo, ".work", "mlx"))
    return out


def section_correctness(redactor):
    rec = run_tool([sys.executable, "-c", CORRECTNESS_PROBE.replace(
        "__SCRIPTS_DIR__", json.dumps(SCRIPTS_DIR))], redactor,
                   label="correctness probe", timeout=240)
    out = {"available": False, "probe": rec, "ops": []}
    if not rec["available"] or rec["exit_code"] != 0:
        return out
    try:
        found = json.loads(rec["stdout"].strip().splitlines()[-1])
    except (ValueError, IndexError):
        out["probe"]["error"] = "unparseable probe output"
        return out
    out["available"] = bool(found.get("available"))
    out["device"] = found.get("device")
    out["mlx_version"] = found.get("mlx_version")
    out["import_error"] = found.get("import_error")
    out["provenance"] = found.get("provenance")
    out["error"] = found.get("error")
    out["ops"] = found.get("ops", [])
    out["ops_passed"] = sum(1 for op in out["ops"] if op.get("pass"))
    return out


def find_spike_binary(repo):
    patterns = (
        ".work/build*/spike/omarchy_matmul_attention",
        ".work/build*/benchmarks/omarchy/omarchy_matmul_attention",
        ".work/build-omarchy/spike/omarchy_matmul_attention",
    )
    for pat in patterns:
        for path in sorted(glob.glob(os.path.join(repo, pat))):
            if os.path.isfile(path):
                return path
    return None


def section_benchmark(redactor, repo):
    out = {"available": False, "python": None, "kernel_spike": None}
    rec = run_tool([sys.executable, "-c", BENCH_PROBE.replace(
        "__SCRIPTS_DIR__", json.dumps(SCRIPTS_DIR))], redactor,
                   label="python microbench", timeout=300)
    out["python_probe"] = rec
    if rec["available"] and rec["exit_code"] == 0:
        try:
            found = json.loads(rec["stdout"].strip().splitlines()[-1])
            out["python"] = found
            out["available"] = bool(found.get("available"))
        except (ValueError, IndexError):
            pass
    spike = find_spike_binary(repo)
    if spike is None:
        out["kernel_spike"] = {"available": False, "error": "binary not built "
                               "(overlay/benchmarks/omarchy, see its CMakeLists.txt)"}
        return out
    output = os.path.join(tempfile.gettempdir(), "omarchy-spike-result.json")
    argv = [spike, "--output", output, "--warmup", "3", "--reps", "10",
            "--rounds", "1"]
    srec = run_tool(argv, redactor, label="kernel spike", timeout=300)
    out["kernel_spike"] = srec
    out["available"] = out["available"] or srec["exit_code"] == 0
    result = read_text(output)
    if result:
        try:
            out["kernel_spike_result"] = json.loads(result)
        except ValueError:
            out["kernel_spike_result_text"] = result[:20_000]
    return out


def find_info_tool(redactor, repo):
    rec = run_tool([sys.executable, "-c", collect_quick.MLX_PROBE_CODE],
                   redactor, label="info tool locate", timeout=90)
    if rec["exit_code"] == 0:
        try:
            tool = json.loads(rec["stdout"].strip().splitlines()[-1]).get("info_tool")
            if tool:
                return tool
        except (ValueError, IndexError):
            pass
    for pat in (".work/build*/mlx-omarchy-info",
                ".work/build*/tools/mlx-omarchy-info/mlx-omarchy-info"):
        for path in sorted(glob.glob(os.path.join(repo, pat))):
            if os.path.isfile(path):
                return path
    return None


def section_profile(redactor, repo, ws):
    out = {"available": False, "trace_smoke": None, "stream": None,
           "analysis": None}
    tool = find_info_tool(redactor, repo)
    if tool:
        out["trace_smoke"] = run_tool([tool, "--trace-smoke"], redactor,
                                      label="mlx-omarchy-info --trace-smoke",
                                      timeout=60)
        out["available"] = out["trace_smoke"]["exit_code"] == 0
    else:
        out["trace_smoke"] = {"available": False, "error": "not-found",
                              "label": "mlx-omarchy-info --trace-smoke"}
    raw = os.path.join(ws, "raw-stream.jsonl")
    env = dict(os.environ)
    env["MLX_OMARCHY_GPU_PROFILE"] = raw
    probe = run_tool([sys.executable, "-c", PROFILE_PROBE], redactor,
                     label="gpu profile probe", timeout=120, env=env)
    out["profile_probe"] = probe
    if os.path.exists(raw):
        with open(raw, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
        hist = {}
        for line in lines:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            name = event.get("kernel") or event.get("name") or event.get("k")
            if name:
                hist[name] = hist.get(name, 0) + 1
        kept = lines[:MAX_STREAM_LINES]
        with open(os.path.join(ws, "profile-stream.jsonl"), "w",
                  encoding="utf-8") as fh:
            for line in kept:
                fh.write(redactor.apply(line) + "\n")
        out["stream"] = {
            "available": bool(lines),
            "lines_total": len(lines),
            "lines_included": len(kept),
            "kernel_histogram": sorted(
                hist.items(), key=lambda kv: (-kv[1], kv[0]))[:20],
        }
        out["available"] = out["available"] or bool(lines)
    else:
        out["stream"] = {"available": False,
                         "error": "no event stream; the installed wheel may "
                                  "lack MLX_OMARCHY_GPU_PROFILE tracing"}
    analyze = os.path.join(repo, "scripts", "profile_analyze.py")
    if os.path.exists(analyze) and os.path.exists(raw):
        out["analysis"] = run_tool(
            [sys.executable, analyze, raw], redactor,
            label="profile_analyze.py", timeout=60)
    return out


def read_thermal(redactor):
    zones = []
    base = "/sys/class/thermal"
    try:
        entries = sorted(os.listdir(base))
    except OSError:
        return zones
    for zone in entries:
        if not zone.startswith("thermal_zone"):
            continue
        temp = read_text(os.path.join(base, zone, "temp"))
        if not temp:
            continue
        kind = read_text(os.path.join(base, zone, "type")) or ""
        try:
            value = int(temp.strip())
        except ValueError:
            continue
        zones.append({"zone": zone, "type": redactor.apply(kind.strip()),
                      "temp_mc": value})
    return zones


def run_sections(ws, repo, redactor, timeout_override, skip):
    """Run each section as a bounded child; keep whatever completed."""
    for name in SECTION_ORDER:
        if name in skip:
            continue
        timeout = timeout_override or SECTION_TIMEOUTS[name]
        rec = run_tool(
            [sys.executable, os.path.abspath(__file__), "--_section", name,
             "--_workspace", ws, "--_repo", repo],
            redactor, label=f"section:{name}", timeout=timeout)
        if not os.path.exists(os.path.join(ws, f"{name}.json")):
            with open(os.path.join(ws, f"{name}.json"), "wb") as fh:
                fh.write(json_bytes({
                    "available": False,
                    "error": rec["error"] or f"exit {rec['exit_code']}",
                    "stderr": rec["stderr"][-2000:],
                }))


def assemble_files(ws, repo, thermal):
    """Member dict from whatever the sections produced; partial is fine."""
    files = {}
    unavailable = []
    redaction = {}
    for name in SECTION_ORDER:
        path = os.path.join(ws, f"{name}.json")
        data = read_text(path)
        if data is None:
            data = json_bytes({"available": False,
                               "error": "section produced no result"})
            unavailable.append(name)
        else:
            try:
                parsed = json.loads(data)
                if parsed.get("available") is False:
                    unavailable.append(name)
                for kind, count in parsed.get("_redaction", {}).items():
                    redaction[kind] = redaction.get(kind, 0) + count
            except ValueError:
                pass
        files[f"{name}.json"] = data if isinstance(data, bytes) \
            else data.encode("utf-8")
    stream = os.path.join(ws, "profile-stream.jsonl")
    if os.path.exists(stream):
        with open(stream, "rb") as fh:
            files["profile-stream.jsonl"] = fh.read()
    files["thermal.json"] = json_bytes({"zones": thermal})
    return files, unavailable, redaction


SUBMISSION_MEMBERS = ("manifest.json", "submission.md")


def _member(files, name):
    try:
        return json.loads(files[name].decode("utf-8"))
    except (KeyError, ValueError):
        return {}


def build_submission(manifest, files, archive_name):
    """Paste-ready cover text, derived from the manifest only."""
    quick = _member(files, "quick.json")
    host = quick.get("host", {})
    mesa = quick.get("mesa", {}).get("gpu", {})
    mlx = quick.get("mlx", {})
    correctness = _member(files, "correctness.json")
    ops = correctness.get("ops", [])
    lines = [
        "## mlx-omarchy hardware report",
        "",
        f"Machine: {host.get('devicetree', {}).get('model') or 'unknown model'}"
        f" ({host.get('arch', 'unknown arch')}, kernel "
        f"{host.get('kernel_release', 'unknown')})",
        f"Vulkan: {mesa.get('deviceName') or 'unavailable'} / "
        f"{mesa.get('driverName') or 'unavailable'}, API "
        f"{mesa.get('apiVersion') or 'unavailable'}",
        f"mlx-omarchy: {mlx.get('distributions', {}).get('mlx-omarchy') or 'not installed'}"
        f", device {mlx.get('default_device') or 'unavailable'}",
        f"Source commit: {manifest.get('source_commit') or 'unknown'}",
        f"Correctness: {sum(1 for op in ops if op.get('pass'))}/{len(ops)}"
        f" probe ops pass"
        + (" (mlx not importable; probes did not run)"
           if correctness.get("available") is False else ""),
        f"Not available on this machine: "
        f"{', '.join(manifest.get('sections_unavailable', [])) or 'nothing'}",
        f"Redaction applied before writing: "
        f"{json.dumps(manifest.get('redaction_summary', {}), sort_keys=True)}"
        " (names, paths, IPs, MACs, serials, credentials; no upload code)",
        "",
        "Members with SHA-256 (see attached archive for full contents):",
        "",
        "```",
    ]
    for entry in manifest.get("files", []):
        lines.append(f"{entry['sha256']}  {entry['path']} ({entry['bytes']} B)")
    lines += ["```", "",
              "<details><summary>quick report JSON</summary>", "",
              "```json",
              files.get("quick.json", b"{}").decode("utf-8").strip(),
              "```", "", "</details>", ""]
    return "\n".join(lines)


def finalize(files, unavailable, redaction, archive_name, repo):
    commit = None
    dirty = None
    head = run_tool(["git", "-C", repo, "rev-parse", "HEAD"], Redactor(),
                    label="git rev-parse HEAD", timeout=15)
    if head["exit_code"] == 0:
        commit = head["stdout"].strip()
        dirty_rec = run_tool(["git", "-C", repo, "status", "--porcelain"],
                             Redactor(), label="git status count", timeout=15)
        dirty = bool(dirty_rec["stdout"].strip())
    listed = {name: data for name, data in files.items()
              if name not in SUBMISSION_MEMBERS}
    manifest = build_manifest(archive_name, listed, extra={
        "source_commit": commit,
        "repo_dirty": dirty,
        "sections_unavailable": unavailable,
        "redaction_summary": dict(sorted(redaction.items())),
        "schema_note": "one file per section; probe records carry argv, "
                       "exit code, capped redacted output; the schema does "
                       "not change with the sharing path",
    })
    quick = json.loads(files.get("quick.json", b"{}").decode("utf-8")
                       or "{}")
    bench = json.loads(files.get("benchmark.json", b"{}").decode("utf-8")
                       or "{}")
    matmul = ((bench.get("python") or {}).get("matmul") or []) \
        if isinstance(bench, dict) else []
    payload = build_payload("deep", quick, manifest, benchmark=matmul)
    files["manifest.json"] = json_bytes(manifest)
    files["submission.md"] = build_submission(
        manifest, files, archive_name).encode("utf-8")
    return manifest, archive_bytes(files), payload


def print_preview(manifest, data, archive_name):
    print(dump_preview(manifest), end="")
    print(f"[preview] archive: {archive_name} bytes={len(data)} "
          f"sha256={hashlib.sha256(data).hexdigest()}")
    print("[preview] nothing written, nothing uploaded; "
          "rerun with --out FILE to write these exact bytes")


def dump_preview(manifest):
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def section_child(name, ws, repo):
    redactor = Redactor()
    try:
        if name == "quick":
            data = collect_quick.collect()
        elif name == "environment":
            data = section_environment(redactor, repo)
        elif name == "correctness":
            data = section_correctness(redactor)
        elif name == "benchmark":
            data = section_benchmark(redactor, repo)
        elif name == "profile":
            data = section_profile(redactor, repo, ws)
        else:
            data = {"available": False, "error": f"unknown section {name}"}
    except Exception as exc:
        data = {"available": False,
                "error": redactor.apply(f"{type(exc).__name__}: {exc}")}
    if isinstance(data, dict):
        data["_redaction"] = redactor.counts
    with open(os.path.join(ws, f"{name}.json"), "wb") as fh:
        fh.write(json_bytes(data))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", metavar="FILE",
                    help="write the archive after the preview (default: "
                         "preview only)")
    ap.add_argument("--submit", metavar="URL", default=None,
                    help="upload the redacted archive to this endpoint "
                         "after the preview")
    ap.add_argument("--repo", default=REPO,
                    help="repository root (default: parent of scripts/)")
    ap.add_argument("--workspace", metavar="DIR",
                    help="keep the section files in DIR instead of a temp dir")
    ap.add_argument("--skip", default="",
                    help="comma-separated sections to skip")
    ap.add_argument("--timeout", type=int, default=None,
                    help="override the per-section timeout in seconds")
    ap.add_argument("--_section", help=argparse.SUPPRESS)
    ap.add_argument("--_workspace", help=argparse.SUPPRESS)
    ap.add_argument("--_repo", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._section:
        section_child(args._section, args._workspace, args._repo)
        return

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    keep = False
    if args.workspace:
        os.makedirs(args.workspace, exist_ok=True)
        ws = args.workspace
        keep = True
    else:
        ws = tempfile.mkdtemp(prefix="mlx-omarchy-deep-")

    pre_redactor = Redactor()
    thermal_start = read_thermal(pre_redactor)
    run_sections(ws, os.path.abspath(args.repo), pre_redactor,
                 args.timeout, skip)
    thermal_end = read_thermal(pre_redactor)
    thermal = [{"phase": "start", **z} for z in thermal_start] + \
              [{"phase": "end", **z} for z in thermal_end]

    files, unavailable, redaction = assemble_files(
        ws, os.path.abspath(args.repo), thermal)
    for kind, count in pre_redactor.counts.items():
        redaction[kind] = redaction.get(kind, 0) + count
    archive_name = os.path.basename(args.out) if args.out \
        else "mlx-omarchy-deep.tar.gz"
    manifest, data, payload = finalize(files, unavailable, redaction,
                                       archive_name, os.path.abspath(args.repo))
    print_preview(manifest, data, archive_name)
    if args.out:
        with open(args.out, "wb") as fh:
            fh.write(data)
        base = args.out.removesuffix(".tar.gz") \
            if args.out.endswith(".tar.gz") \
            else os.path.splitext(args.out)[0]
        submission = base + ".submission.md"
        with open(submission, "wb") as fh:
            fh.write(files["submission.md"])
        print(f"[receipt] wrote {args.out} ({len(data)} bytes, "
              f"sha256={hashlib.sha256(data).hexdigest()})")
        print(f"[receipt] wrote {submission} (paste-ready cover text)")
    exit_code = maybe_submit(args, data, archive_name, args.out, payload)
    if not keep:
        shutil.rmtree(ws, ignore_errors=True)
    raise SystemExit(exit_code)


def maybe_submit(args, data, archive_name, out_path, payload):
    """Upload only after explicit consent; never without a local copy."""
    import collect_submit
    endpoint = collect_submit.endpoint_from_args(args)
    if endpoint is None:
        return 0
    if not out_path:
        print("[submit] no --out file; refusing to upload without keeping "
              "a local copy", file=sys.stderr)
        return 4
    digest = hashlib.sha256(data).hexdigest()
    if not args.submit and not collect_submit.confirm_interactive(
            endpoint, archive_name, digest):
        print(f"[submit] declined; {out_path} stays local")
        return 0
    try:
        receipt = collect_submit.submit(endpoint, data, payload,
                                        token=collect_submit.token_from_env())
    except collect_submit.SubmitError as exc:
        print(f"[submit] FAILED: {exc}", file=sys.stderr)
        print(f"[submit] local output preserved: {out_path}", file=sys.stderr)
        return 4
    print(f"[receipt] public URL: {receipt['url']} "
          f"(deduplicated={receipt['deduplicated']})")
    return 0


if __name__ == "__main__":
    main()
