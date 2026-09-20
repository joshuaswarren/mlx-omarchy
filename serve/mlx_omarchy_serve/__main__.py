"""mlx-omarchy serve CLI: recommend, plan, approve, download, launch.

Exit codes: 0 ok, 1 refused (does not fit / fetch failed / user said no),
2 bad input, 3 missing environment (e.g. backend not installed).

Design invariants (do not weaken):
- Approval before any download; noninteractive runs fail closed unless
  --yes is given AND the target was named explicitly.
- The memory budget counts the full checkpoint (MoE total parameters),
  the KV at the explicit context limit, a labeled workspace margin, the
  aggregate active reservations, and a safety reserve. No fit -> refuse.
- Catalog refresh touches ONLY the public GitHub raw URL; no telemetry,
  no background timers, no automatic code or model downloads.
- trust_remote_code is never enabled anywhere.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from . import budget, catalog

GiB = 1024**3
DISK_SLACK_FRACTION = 0.05  # headroom over the download size for partial files

REPO_RE = re.compile(r"^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)+$")


def warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


def fail(msg: str, code: int) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return code


def machine_soc() -> str | None:
    try:
        raw = Path("/proc/device-tree/compatible").read_bytes()
    except OSError:
        return None
    for token in raw.split(b"\0"):
        match = re.fullmatch(rb"apple,t(\d+)", token)
        if match:
            return "t" + match.group(1).decode()
    return None


# ---------------------------------------------------------------- resolution


class Resolved:
    def __init__(self, target: str, entry: dict | None, local_path: Path | None,
                 repo: str | None, revision: str | None):
        self.target = target
        self.entry = entry
        self.local_path = local_path
        self.repo = repo
        self.revision = revision

    @property
    def named_explicitly(self) -> bool:
        return self.target is not None


def pick_recommended(cat: dict, kind: str, context_tokens: int, home: Path | None):
    """Curated priority among compatible fits: recommended first, then
    priority, then file order — first entry that fits memory wins."""
    soc = machine_soc()
    ranked = sorted(
        (e for e in cat["models"] if e["kind"] == kind),
        key=lambda e: (not e["recommended"], e["priority"]),
    )
    fits = []
    for entry in ranked:
        if entry["capability"]["arch"] is not None and soc is not None \
                and soc not in entry["capability"]["arch"]:
            continue
        if entry["capability"]["min_mem_gib"] is not None:
            try:
                if budget.mem_available() < entry["capability"]["min_mem_gib"] * GiB:
                    continue
            except budget.BudgetError:
                pass
        est = budget.estimate_required(entry["memory"], context_tokens)
        if budget.admit(est.total, home).fits:
            fits.append(entry)
    return fits, ranked, soc


def resolve_target(target: str | None, cat: dict) -> Resolved:
    if target is None:
        return Resolved(None, None, None, None, None)
    for entry in cat["models"]:
        if entry["id"] == target:
            return Resolved(target, entry, None, entry["repo"], entry["revision"])
    path = Path(target).expanduser()
    if path.is_dir():
        return Resolved(target, None, path, None, None)
    if REPO_RE.match(target) and "/" in target and not target.startswith("."):
        return Resolved(target, None, None, target, None)
    raise budget.BudgetError(
        f"{target!r} is not a catalog id, an existing local directory, or an HF repo id"
    )


def quant_label(entry: dict) -> str:
    mode = entry["quant"]["mode"]
    if mode == "affine":
        return f"{entry['quant']['bits']}bit/g{entry['quant']['group_size']}"
    return mode


def backend_for(resolved: Resolved, server_flag: str | None) -> tuple[str, str | None]:
    entry = resolved.entry
    if server_flag is not None:
        return server_flag, (entry["serve"]["module"] if entry and entry["serve"] else None)
    if entry is not None and entry["serve"] is not None:
        backend = entry["serve"]["backend"]
        if backend == "module":
            return "module", entry["serve"]["module"]
        return backend, None
    return "mlx-lm", None


# ------------------------------------------------------------------- planning


def hf_cache_dir() -> Path:
    home = os.environ.get("HF_HOME")
    base = Path(home) if home else Path.home() / ".cache/huggingface"
    return base / "hub"


def disk_need_bytes(resolved: Resolved) -> int | None:
    if resolved.local_path is not None:
        return None
    if resolved.entry is not None:
        size = resolved.entry["availability"]["size_bytes"]
        weights = resolved.entry["memory"]["weights_bytes"]
        base = size if size is not None else weights
    else:
        base = None
    if base is None:
        return None
    return int(base * (1 + DISK_SLACK_FRACTION))


def plan_lines(resolved: Resolved, context_tokens: int, backend: str,
               module: str | None, host: str, port: int,
               weights_override: int | None, home: Path | None) -> tuple[list[str], budget.Admission | None]:
    lines: list[str] = []
    est = None
    admission = None
    if resolved.local_path is not None:
        if weights_override is None:
            raise budget.BudgetError(
                f"cannot budget local model {resolved.local_path} without --weights-gib "
                "(the full checkpoint size on disk; MoE counts total parameters)"
            )
        memory = {"weights_bytes": weights_override, "kv_bytes_per_token": None, "peak_estimate_bytes": None}
    elif resolved.entry is not None:
        memory = resolved.entry["memory"]
    else:
        if weights_override is None:
            raise budget.BudgetError(
                f"{resolved.repo} is not in the catalog; pass --weights-gib so the "
                "memory budget stays honest (or pick a catalog id)"
            )
        memory = {"weights_bytes": weights_override, "kv_bytes_per_token": None, "peak_estimate_bytes": None}

    est = budget.estimate_required(memory, context_tokens)
    if not est.kv_known and context_tokens > budget.DEFAULT_CONTEXT_TOKENS:
        raise budget.BudgetError(
            f"KV-per-token unknown for this model, so the budget cannot honestly "
            f"cover {context_tokens} tokens; use --context <= "
            f"{budget.DEFAULT_CONTEXT_TOKENS} or a catalog entry with KV facts"
        )
    lines.append(f"model:        {resolved.repo or resolved.local_path}")
    if resolved.revision:
        lines.append(f"revision:     {resolved.revision} (pinned)")
    lines.append(f"backend:      {backend}" + (f" module={module}" if module else ""))
    lines.append(f"bind:         {host}:{port}")
    lines.append(f"context:      {context_tokens} tokens")
    lines.append(f"weights:      {est.peak_override and '(peak override) ' or ''}{est.weights / GiB:.2f} GiB")
    if est.kv_known:
        lines.append(f"kv cache:     {est.kv / GiB:.2f} GiB at {context_tokens} tokens")
    else:
        lines.append(f"kv cache:     UNKNOWN -> flat {budget.UNKNOWN_KV_MARGIN / GiB:.2f} GiB margin included")
    label = "measured peak" if est.peak_override else "workspace margin, unmeasured estimate"
    lines.append(f"margin:       {est.workspace / GiB:.2f} GiB ({label})")
    need = disk_need_bytes(resolved) if resolved.local_path is None else None
    if need is not None:
        ok, free, where = budget.disk_check(need, hf_cache_dir())
        lines.append(f"disk:         need {need / GiB:.2f} GiB, {free / GiB:.2f} GiB free at {where}"
                     + ("" if ok else "  -> INSUFFICIENT"))
    admission = budget.admit(est.total, home)
    lines.extend("mem:          " + stripped for stripped in (ln.strip() for ln in admission.lines))
    lines.append("verdict:      " + ("FITS" if admission.fits else "DOES NOT FIT — refusing (no swap/OOM gambles)"))
    return lines, admission


def needs_download(resolved: Resolved) -> bool:
    if resolved.local_path is not None:
        return False
    hub = _import_huggingface_hub()
    if hub is None:
        return True  # cannot prove presence; the approval flow will say so
    try:
        hub.snapshot_download(
            repo_id=resolved.repo,
            revision=resolved.revision,
            local_files_only=True,
        )
        return False
    except Exception:  # huggingface_hub raises several types for a missing snapshot
        return True


def _import_huggingface_hub():
    spec = importlib.util.find_spec("huggingface_hub")
    if spec is None:
        return None
    import huggingface_hub

    return huggingface_hub


def download_snapshot(resolved: Resolved) -> Path:
    hub = _import_huggingface_hub()
    if hub is None:
        raise budget.BudgetError("huggingface_hub is not installed in this environment")
    return Path(
        hub.snapshot_download(repo_id=resolved.repo, revision=resolved.revision)
    )


def check_custom_code(model_dir: Path) -> None:
    config = model_dir / "config.json"
    try:
        obj = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if isinstance(obj, dict) and "auto_map" in obj:
        warn(
            f"{config} declares auto_map (custom modeling code). mlx-omarchy never "
            "enables trust_remote_code; loading this checkpoint will likely fail."
        )


def server_argv(backend: str, module: str | None, model_dir: Path, host: str, port: int) -> list[str]:
    if backend == "mlx-lm":
        return [sys.executable, "-m", "mlx_lm.server", "--model", str(model_dir),
                "--host", host, "--port", str(port)]
    if backend == "omlx":
        return [sys.executable, "-m", "omlx.server", "--model-dir", str(model_dir),
                "--host", host, "--port", str(port)]
    if backend == "module":
        if not module:
            raise budget.BudgetError("module backend needs a module name (catalog serve.module)")
        code = f"import sys; from {module} import serve_main; serve_main(sys.argv[1:])"
        return [sys.executable, "-c", code, "--model", str(model_dir),
                "--host", host, "--port", str(port)]
    raise budget.BudgetError(f"unknown backend {backend!r}")


def omlx_available() -> bool:
    return importlib.util.find_spec("omlx") is not None


# --------------------------------------------------------------------- flows


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mlx-omarchy-serve",
        description="Serve a vetted local model on the Apple GPU (admission-checked, approve-first).",
    )
    sub = parser.add_subparsers(dest="command")

    def common(p, with_target=True):
        p.add_argument("--offline", action="store_true",
                       help="never touch the network (catalog stays cached/bundled)")
        if with_target:
            p.add_argument("target", nargs="?",
                           help="catalog id, HF repo id (org/name), or local model directory; "
                                "default: curated recommendation")
        p.add_argument("--context", type=int, default=None,
                       help="context tokens (KV is budgeted at this); default = model limit or 4096")
        p.add_argument("--server", choices=("mlx-lm", "omlx", "module"), default=None,
                       help="serve backend; default from catalog, else mlx-lm")
        p.add_argument("--weights-gib", type=float, default=None, dest="weights_gib",
                       help="explicit checkpoint size for off-catalog/local models")

    p_recommend = sub.add_parser("recommend", help="show curated fits and the pick")
    common(p_recommend, with_target=False)
    p_recommend.add_argument("--kind", default="chat", choices=catalog.KINDS)

    p_plan = sub.add_parser("plan", help="admission + disk check only; downloads nothing")
    common(p_plan)
    p_plan.add_argument("--host", default="127.0.0.1")
    p_plan.add_argument("--port", type=int, default=8080)

    p_serve = sub.add_parser("serve", help="plan, require approval, download, launch foreground")
    common(p_serve)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--yes", action="store_true",
                         help="noninteractive explicit opt-in; requires an explicit target")

    p_cat = sub.add_parser("catalog", help="catalog list/status/refresh")
    csub = p_cat.add_subparsers(dest="catalog_command", required=True)
    for name, help_text in (
        ("list", "print the catalog table with memory verdicts"),
        ("status", "which catalog is live (cache age, source)"),
        ("refresh", "force a GitHub refresh now"),
    ):
        p = csub.add_parser(name, help=help_text)
        p.add_argument("--offline", action="store_true",
                       help="never touch the network")

    p_res = sub.add_parser("reserve", help="declare resident memory another local service owns")
    p_res.add_argument("name")
    p_res.add_argument("gib", type=float)
    p_res.add_argument("--note", default="")
    p_unres = sub.add_parser("unreserve", help="remove a reservation")
    p_unres.add_argument("name")

    return parser.parse_args(argv)


def cmd_recommend(args, cat, home) -> int:
    context = None
    fits, ranked, soc = pick_recommended(cat, args.kind, context or budget.DEFAULT_CONTEXT_TOKENS, home)
    print(f"machine SoC: {soc or 'unknown'}; kind: {args.kind}; curated order:")
    for entry in ranked:
        est = budget.estimate_required(entry["memory"], context or budget.DEFAULT_CONTEXT_TOKENS)
        fits_now = entry in fits
        print(f"  {'*' if fits_now else ' '} {entry['id']:<28} {quant_label(entry):<10}"
              f" p{entry['priority']}"
              f" {'recommended' if entry['recommended'] else '           '}"
              f" {est.total / GiB:6.2f} GiB"
              f" gen:{entry['qualification']['generation']['status']}"
              f" http:{entry['qualification']['http']['status']}")
    if fits:
        pick = fits[0]
        print(f"pick: {pick['id']} (curated priority within {args.kind}, memory fits; "
              "other fitting variants above are servable by id)")
        return 0
    print("pick: none — no compatible entry fits this machine's memory right now")
    return 0


def build_resolved(args, cat) -> Resolved:
    resolved = resolve_target(args.target, cat)
    if resolved.target is None and args.command == "serve":
        fits, ranked, _soc = pick_recommended(
            cat, "chat", args.context or budget.DEFAULT_CONTEXT_TOKENS, None
        )
        if not fits:
            raise budget.BudgetError("no recommended chat model fits this machine; name a target explicitly")
        entry = fits[0]
        if sys.stdin.isatty() and len(fits) > 1:
            # Manual variant selection: every fitting quant variant is offered,
            # the curated pick is just the default. Never all of them.
            for i, candidate in enumerate(fits, 1):
                est = budget.estimate_required(
                    candidate["memory"], args.context or budget.DEFAULT_CONTEXT_TOKENS)
                print(f"  {i}) {candidate['id']:<28} {quant_label(candidate):<10}"
                      f" {est.total / GiB:.2f} GiB")
            answer = input(f"Select variant [1-{len(fits)}] (default 1): ").strip()
            choice = int(answer) if answer.isdigit() and 1 <= int(answer) <= len(fits) else 1
            entry = fits[choice - 1]
        print(f"serving variant: {entry['id']}")
        resolved = Resolved(None, entry, None, entry["repo"], entry["revision"])
    return resolved


def do_plan(args, cat, home) -> tuple[Resolved, list[str], budget.Admission | None, str, str | None]:
    resolved = build_resolved(args, cat)
    context = budget.resolve_context(
        resolved.entry["context"] if resolved.entry else {"max_tokens": None},
        args.context,
    )
    backend, module = backend_for(resolved, args.server)
    if backend == "module" and module is None:
        raise budget.BudgetError("module backend selected but no serve.module is set")
    lines, admission = plan_lines(
        resolved, context, backend, module, args.host, args.port,
        int(args.weights_gib * GiB) if args.weights_gib else None, home,
    )
    return resolved, lines, admission, backend, module


def cmd_plan(args, cat, home) -> int:
    _resolved, lines, _adm, _backend, _module = do_plan(args, cat, home)
    print("\n".join(lines))
    return 0


def approve_download(lines: list[str], *, interactive: bool, assume_yes: bool,
                     explicit_target: bool, input_fn=None) -> bool:
    print("\n".join(lines))
    print()
    if not interactive:
        if not assume_yes:
            print("refusing: noninteractive without --yes; nothing was downloaded", file=sys.stderr)
            return False
        if not explicit_target:
            print("refusing: --yes requires an explicitly named target; nothing was downloaded",
                  file=sys.stderr)
            return False
        return True
    if assume_yes:
        return True
    answer = (input_fn or input)("Download this model now? Type 'yes' to proceed: ")
    return answer.strip().lower() == "yes"


def cmd_serve(args, cat, home) -> int:
    resolved, lines, admission, backend, module = do_plan(args, cat, home)
    if admission is not None and not admission.fits:
        print("\n".join(lines), file=sys.stderr)
        print("\nrefusing: does not fit; nothing was downloaded or started", file=sys.stderr)
        return 1

    interactive = sys.stdin.isatty()
    download = needs_download(resolved)
    if download and args.offline:
        return fail("--offline set and the model is not on disk; refusing to download", 1)

    if download:
        if not approve_download(lines, interactive=interactive, assume_yes=args.yes,
                                explicit_target=resolved.named_explicitly):
            return 1
        model_dir = download_snapshot(resolved)
        print(f"downloaded to {model_dir}")
    else:
        print("\n".join(lines))
        model_dir = resolved.local_path or Path(
            download_snapshot(resolved)  # resolve the on-disk snapshot path
        )
    check_custom_code(model_dir)

    if backend == "omlx" and not omlx_available():
        return fail("omlx is not installed (source install only; see docs/serve.md); "
                    "use --server mlx-lm", 3)
    if host_nonlocal(args.host):
        warn(f"{args.host} is not loopback: this development server has no authentication")

    argv = server_argv(backend, module, model_dir, args.host, args.port)
    print(f"launching: {' '.join(argv)}")
    try:
        completed = subprocess.run(argv)
    except FileNotFoundError as exc:
        return fail(f"server launch failed: {exc}", 3)
    return completed.returncode


def host_nonlocal(host: str) -> bool:
    return host not in ("127.0.0.1", "localhost", "::1")


def cmd_catalog(args, _cat, home) -> int:
    if args.catalog_command == "list":
        cat = catalog.load_catalog(home)
        print(f"{'id':<28} {'kind':<9} {'p':<3} {'rec':<3} {'gen':<9} {'http':<9} {'GiB':>7}")
        for e in cat["models"]:
            print(f"{e['id']:<28} {e['kind']:<9} {e['priority']:<3}"
                  f" {'yes' if e['recommended'] else '-':<3}"
                  f" {e['qualification']['generation']['status']:<9}"
                  f" {e['qualification']['http']['status']:<9}"
                  f" {e['memory']['weights_bytes'] / GiB:>7.2f}")
        return 0
    if args.catalog_command == "status":
        path = catalog.cache_path(home)
        if path.is_file():
            age_h = 0.0
            import time

            age_h = (time.time() - path.stat().st_mtime) / 3600.0
            print(f"cache:  {path} ({age_h:.1f}h old)")
        else:
            print(f"cache:  none at {path}")
        print(f"bundled: {catalog.bundled_path()}")
        print(f"url:    {os.environ.get('MLX_OMARCHY_CATALOG_URL') or catalog.DEFAULT_CATALOG_URL}")
        print(f"live:   {'cache' if path.is_file() else 'bundled fallback'}")
        return 0
    result = catalog.refresh(home=home, offline=args.offline)
    print(f"catalog refresh: {result['status']} ({result['detail']})")
    return 0 if result["status"] in ("fresh", "updated", "not-modified", "skipped") else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    home = budget.default_home()
    try:
        catalog.refresh(home=home, offline=getattr(args, "offline", False))
        cat = catalog.load_catalog(home)
        if args.command == "recommend":
            return cmd_recommend(args, cat, home)
        if args.command == "plan":
            return cmd_plan(args, cat, home)
        if args.command == "serve":
            return cmd_serve(args, cat, home)
        if args.command == "catalog":
            return cmd_catalog(args, cat, home)
        if args.command == "reserve":
            budget.set_reservation(args.name, int(args.gib * GiB), args.note, home)
            print(f"reserved {args.gib:g} GiB for {args.name}")
            return 0
        if args.command == "unreserve":
            if not budget.clear_reservation(args.name, home):
                return fail(f"no reservation named {args.name!r}", 2)
            print(f"removed reservation {args.name}")
            return 0
    except budget.BudgetError as exc:
        return fail(str(exc), 2)
    except catalog.CatalogError as exc:
        return fail(f"catalog invalid: {exc}", 3)
    return fail("unreachable", 2)


if __name__ == "__main__":
    sys.exit(main())
