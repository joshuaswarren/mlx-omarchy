#!/usr/bin/env python3
"""Query the mlx-omarchy community hardware dataset.

Stdlib-only CLI for coding agents and humans. It reads the mirrored
snapshot on the `community-data` branch, or the live public endpoint, and
answers: which machines reported, what ran there, and how benchmark
numbers compare across devices.

Usage:
  query_community_data.py [filters] list
  query_community_data.py show SHA256_PREFIX
  query_community_data.py [filters] compare [--metric tflops] [--size N]

Sources (--source):
  auto    local snapshot when present, else the live endpoint (default)
  local   mirrored snapshot only
  remote  live public endpoint only

Local snapshot ladder, first hit wins:
  --snapshot DIR           directory holding latest.jsonl
  $MLX_OMARCHY_DATA_DIR    same, from the environment
  <repo>/community-data/snapshot/latest.jsonl
  origin/community-data:snapshot/latest.jsonl via `git show`

Exit codes: 0 ok, 1 no dataset or endpoint unreachable. Partial records
and unknown schema_version values never crash the tool; malformed JSONL
lines are skipped and counted.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "https://mlx-omarchy-community-data.joshua-s-warren.workers.dev"
# Cloudflare bot protection 403s default Python user agents on workers.dev.
USER_AGENT = ("mlx-omarchy-community-data/1.0 "
              "(+https://github.com/joshuaswarren/mlx-omarchy)")
SNAPSHOT_RELPATH = Path("community-data") / "snapshot"

METRIC_UNITS = {"tflops": "TFLOP/s", "median_ms": "ms", "min_ms": "ms",
                "max_ms": "ms"}


class DatasetError(RuntimeError):
    """User-facing failure: no dataset, bad source, endpoint down."""


# ---------------------------------------------------------------------------
# record accessors - tolerant across partial payloads and schema drift
# ---------------------------------------------------------------------------

def _payload(record):
    payload = record.get("payload")
    return payload if isinstance(payload, dict) else {}


def _walk(mapping, *path):
    """Final value at path, whatever its type; None on any miss."""
    current = mapping
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _dig(mapping, *path):
    """Scalar at path; None when missing or a container."""
    value = _walk(mapping, *path)
    if value is None or isinstance(value, (dict, list)):
        return None
    return value


def _dig_dict(mapping, *path):
    """Dict at path; {} when missing or not a dict."""
    value = _walk(mapping, *path)
    return value if isinstance(value, dict) else {}


def _first(record, paths):
    """First scalar hit among paths, checking the record then payload."""
    payload = _payload(record)
    for path in paths:
        value = _dig(record, *path)
        if value is None:
            value = _dig(payload, *path)
        if value not in (None, ""):
            return value
    return None


def record_sha(record):
    for key in ("content_sha256", "sha256"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def record_kind(record):
    for value in (record.get("kind"), _payload(record).get("kind"),
                  _payload(record).get("report")):
        if isinstance(value, str) and value:
            return value
    return "unknown"


def record_chip(record):
    value = _first(record, [
        ("host", "chip"), ("host", "devicetree", "model"),
        ("host", "model"), ("chip",), ("soc",), ("soc_name",),
    ])
    if value:
        return str(value)
    device = _dig_dict(_payload(record), "mesa", "gpu").get("deviceName")
    return str(device) if device else None


def record_kernel(record):
    value = _first(record, [
        ("host", "kernel_release"), ("host", "kernel"),
        ("kernel",), ("kernel_version",),
    ])
    return str(value) if value else None


def _mesa_package_version(record):
    packages = _dig_dict(_payload(record), "mesa_package")
    for channel in ("pacman", "dpkg"):
        stdout = _dig(_dig_dict(packages, channel), "stdout")
        if not isinstance(stdout, str):
            continue
        for line in stdout.splitlines():
            parts = line.split()
            if parts and parts[0] == "mesa" and len(parts) > 1:
                return parts[1]
    return None


def record_mesa(record):
    """Display string: driver name plus the freshest version we can find."""
    gpu = _dig_dict(_payload(record), "mesa", "gpu")
    driver = gpu.get("driverName") or _first(record, [("mesa", "driver")])
    version = _mesa_package_version(record) or gpu.get("driverVersion")
    if driver and version:
        return f"{driver} {version}"
    return str(driver or version) if (driver or version) else None


def record_mesa_blob(record):
    """Lowercased searchable text for --mesa: driver, API and package."""
    gpu = _dig_dict(_payload(record), "mesa", "gpu")
    parts = [str(gpu.get(k)) for k in ("driverName", "apiVersion")
             if gpu.get(k)]
    package = _mesa_package_version(record)
    if package:
        parts.append(package)
    return " ".join(parts).lower()


def record_mlx_version(record):
    value = _first(_payload(record), [
        ("mlx", "distributions", "mlx-omarchy"), ("mlx", "version"),
        ("mlx_omarchy_version",), ("version",),
    ])
    return str(value) if value else None


def record_benchmarks(record):
    """[(label, metric, size, value)] from known benchmark locations."""
    benchmark = _dig_dict(_payload(record), "benchmark")
    out = []
    entries = benchmark.get("matmul") if isinstance(benchmark, dict) else None
    if not isinstance(entries, list):
        return out
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        size = entry.get("n")
        for metric in ("tflops", "median_ms", "min_ms", "max_ms"):
            value = entry.get(metric)
            if isinstance(value, (int, float)):
                out.append((f"matmul@{size}", metric, size, float(value)))
    return out


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def parse_jsonl(text):
    """(records, skipped) from JSONL text; bad lines never raise."""
    records, skipped = [], 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            skipped += 1
            continue
        if isinstance(obj, dict):
            records.append(obj)
        else:
            skipped += 1
    return records, skipped


def snapshot_file(snapshot_dir):
    return Path(snapshot_dir) / "latest.jsonl"


def load_local(args, repo_root):
    """Return (records, skipped, source_label) from the local ladder."""
    tried = []
    candidates = []
    if args.snapshot:
        candidates.append(Path(args.snapshot))
    env_dir = os.environ.get("MLX_OMARCHY_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(repo_root / SNAPSHOT_RELPATH)
    for directory in candidates:
        path = snapshot_file(directory)
        tried.append(str(path))
        if path.is_file():
            records, skipped = parse_jsonl(
                path.read_text(encoding="utf-8", errors="replace"))
            return records, skipped, str(path)
    for ref in ("origin/community-data", "community-data"):
        spec = f"{ref}:snapshot/latest.jsonl"
        tried.append(f"git {spec}")
        try:
            proc = subprocess.run(["git", "show", spec], cwd=repo_root,
                                  capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            break
        if proc.returncode == 0:
            records, skipped = parse_jsonl(proc.stdout.decode("utf-8",
                                                             "replace"))
            return records, skipped, f"git {spec}"
    raise DatasetError(
        "no local dataset found. Looked at: " + ", ".join(tried) +
        ". Fetch the community-data branch (git fetch origin "
        "community-data), pass --snapshot DIR, or use --source remote.")


def fetch_url(url, timeout=60):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise DatasetError(f"HTTP {exc.code} from {url}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise DatasetError(f"endpoint unreachable: {exc}") from exc


def load_remote(args):
    """(records, skipped, source_label) from the live JSONL endpoint."""
    base = args.base_url.rstrip("/")
    text = fetch_url(f"{base}/v1/dataset/latest.jsonl").decode(
        "utf-8", "replace")
    records, skipped = parse_jsonl(text)
    return records, skipped, f"{base}/v1/dataset/latest.jsonl"


def load_dataset(args, repo_root):
    if args.source == "local":
        return load_local(args, repo_root)
    if args.source == "remote":
        return load_remote(args)
    try:
        return load_local(args, repo_root)
    except DatasetError:
        return load_remote(args)


# ---------------------------------------------------------------------------
# filtering and commands
# ---------------------------------------------------------------------------

def record_matches(record, filters):
    chip_blob = " ".join(
        x for x in (record_chip(record),
                    str(_dig(_payload(record), "mesa", "gpu", "deviceName")
                        or "")) if x).lower()
    checks = (
        ("kind", record_kind(record)),
        ("chip", chip_blob),
        ("kernel", (record_kernel(record) or "").lower()),
        ("mesa", record_mesa_blob(record)),
        ("mlx_version", (record_mlx_version(record) or "").lower()),
    )
    for attr, haystack in checks:
        needle = getattr(filters, attr)
        if needle and needle.lower() not in haystack:
            return False
    return True


def filter_records(records, filters):
    return [r for r in records if record_matches(r, filters)]


def one_line(record):
    benches = record_benchmarks(record)
    return "  ".join(filter(None, (
        (record_sha(record) or "?")[:12],
        f"kind={record_kind(record)}",
        f"chip={record_chip(record) or 'unknown'}",
        f"kernel={record_kernel(record) or 'unknown'}",
        f"mesa={record_mesa(record) or 'unknown'}",
        f"mlx-omarchy={record_mlx_version(record) or 'unknown'}",
        f"bench={len(benches)}" if benches else "",
    )))


def find_record(records, prefix):
    matches = [r for r in records
               if (record_sha(r) or "").startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise DatasetError(f"no record with hash prefix {prefix!r}")
    raise DatasetError(f"hash prefix {prefix!r} is ambiguous "
                       f"({len(matches)} records)")


def cmd_list(records, skipped, source, as_json):
    if as_json:
        print(json.dumps({"source": source, "count": len(records),
                          "skipped_malformed": skipped,
                          "results": records}, indent=2, sort_keys=True))
        return 0
    if not records:
        print(f"no records match ({source}).")
        return 0
    for record in records:
        print(one_line(record))
    if skipped:
        print(f"({skipped} malformed line(s) skipped)", file=sys.stderr)
    return 0


def cmd_show(records, prefix, args):
    if args.source == "remote":
        base = args.base_url.rstrip("/")
        body = fetch_url(f"{base}/v1/results/{prefix}")
        print(json.dumps(json.loads(body.decode("utf-8")),
                         indent=2, sort_keys=True))
        return 0
    record = find_record(records, prefix)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def compare_groups(records, metric, size):
    """[{device, count, min, median, max}] sorted by device name."""
    groups = {}
    for record in records:
        device = record_chip(record) or "unknown"
        for _, bench_metric, bench_size, value in record_benchmarks(record):
            if bench_metric != metric:
                continue
            if size is not None and bench_size != size:
                continue
            groups.setdefault(device, []).append(value)
    out = []
    for device in sorted(groups):
        values = groups[device]
        out.append({"device": device, "count": len(values),
                    "min": min(values),
                    "median": statistics.median(values),
                    "max": max(values)})
    return out


def cmd_compare(records, metric, size, as_json):
    groups = compare_groups(records, metric, size)
    if as_json:
        print(json.dumps({"metric": metric,
                          "unit": METRIC_UNITS.get(metric, ""),
                          "size": size, "groups": groups},
                         indent=2, sort_keys=True))
        return 0
    if not groups:
        print(f"no {metric} benchmark data in the selected records.")
        return 0
    print(f"metric {metric} ({METRIC_UNITS.get(metric, '?')}), "
          f"{len(groups)} device(s):")
    print(f"{'device':<28}{'n':>4}{'min':>12}{'median':>12}{'max':>12}")
    for group in groups:
        print(f"{group['device'][:27]:<28}{group['count']:>4}"
              f"{group['min']:>12.6g}{group['median']:>12.6g}"
              f"{group['max']:>12.6g}")
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="query_community_data.py",
        description="Query the mlx-omarchy community hardware dataset.")
    parser.add_argument("--source", choices=("auto", "local", "remote"),
                        default="auto")
    parser.add_argument("--snapshot", default=None,
                        help="directory holding latest.jsonl")
    parser.add_argument("--base-url", default=os.environ.get(
        "MLX_OMARCHY_COMMUNITY_URL", DEFAULT_BASE_URL))
    parser.add_argument("--json", action="store_true",
                        help="machine-readable output")
    parser.add_argument("--kind", default=None,
                        help="substring: quick, deep, ...")
    parser.add_argument("--chip", default=None,
                        help="substring: Apple M1, M2, ...")
    parser.add_argument("--kernel", default=None,
                        help="substring of the kernel release")
    parser.add_argument("--mesa", default=None,
                        help="substring of driver name, API or package")
    parser.add_argument("--mlx-version", default=None,
                        help="substring of the mlx-omarchy version")
    parser.add_argument("--metric", default="tflops",
                        help="compare: tflops, median_ms, min_ms, max_ms")
    parser.add_argument("--size", type=int, default=None,
                        help="compare: matmul matrix size n")
    parser.add_argument("command", nargs="?", default="list",
                        choices=("list", "show", "compare"))
    parser.add_argument("target", nargs="?", default=None,
                        help="show: sha256 prefix")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    try:
        if args.command == "show" and not args.target:
            print("show needs a sha256 prefix.", file=sys.stderr)
            return 1
        records, skipped, source = load_dataset(args, repo_root)
        records = filter_records(records, args)
        if args.command == "list":
            return cmd_list(records, skipped, source, args.json)
        if args.command == "show":
            return cmd_show(records, args.target, args)
        return cmd_compare(records, args.metric, args.size, args.json)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
