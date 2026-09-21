"""Memory and disk admission for serving.

Everything here is CONSERVATIVE and labeled. The goal is to refuse a model
that does not fit before any download or load, never to trade an OOM/swap
death for a false "fits".

Estimate model (per catalog entry, at an explicit context limit):

    required = weights + kv + workspace
      weights  = memory.weights_bytes  — the FULL checkpoint. MoE total
                 parameters count (35B-A3B reserves for 35B, not 3B active);
                 "active" is compute, not resident memory.
      kv       = memory.kv_bytes_per_token * context_tokens (0 when unknown)
      workspace= memory.peak_estimate_bytes overrides everything when set
                 (measured); otherwise max(WORKSPACE_FRACTION * weights,
                 WORKSPACE_MIN_BYTES) — an UNMEASURED margin for activations,
                 compile buffers, and allocator slack.
    kv unknown -> flat UNKNOWN_KV_MARGIN added, labeled in every table.

Admission: required + sum(reservation headroom) + SAFETY_RESERVE must fit
in MemAvailable.

Reservation states (two-phase): "pending" = declared, nothing materialized
(admission subtracts the full bytes); "resident" = weights materialized and
already visible as reduced MemAvailable, BUT the service's future KV and
workspace are NOT in MemAvailable until requests arrive — so a resident
reservation still subtracts its unmaterialized peak headroom:
bytes − resident_floor_bytes, where resident_floor_bytes is the verified
materialized baseline the service reports at relabel time (conservative
floor: its weights; never an RSS/GPU undercount). Without an explicit
floor the FULL bytes stay subtracted — over-reservation can only cause a
conservative false refusal, never an unsafe admission under concurrent load.

This module is a conservative budget gate, not a live admission
coordinator: it sees MemAvailable snapshots and explicit declarations, not
process tables.

Co-serving race safety: admit_and_reserve() is the shared transaction —
the fit check and the reservation write happen under ONE exclusive file
lock, so two processes cannot both admit against the same MemAvailable and
then overcommit. Every transaction-served reservation carries an owner
token (generate_owner()); a reservation held by one owner can never be
overwritten or cleared by a different one, so two processes serving the
same catalog id refuse instead of clobbering each other.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

GiB = 1024**3

SAFETY_RESERVE_BYTES = 2 * GiB          # keep for the OS/desktop; never budgeted to models
WORKSPACE_FRACTION = 0.25               # unmeasured; catalog peak_estimate_bytes overrides
WORKSPACE_MIN_BYTES = 1 * GiB
UNKNOWN_KV_MARGIN = 2 * GiB             # flat margin when kv_bytes_per_token is null
DEFAULT_CONTEXT_TOKENS = 4096           # when a catalog entry has no max_tokens

RESERVATIONS_FILE = "reservations.json"
RESERVATION_STATES = ("pending", "resident")


class BudgetError(ValueError):
    """A budget question cannot be answered honestly (bad input/limits)."""


def default_home() -> Path:
    return Path(os.environ.get("MLX_OMARCHY_HOME", Path.home() / ".local/share/mlx-omarchy"))


def parse_meminfo(text: str) -> int:
    """Extract MemAvailable in bytes. Raises BudgetError if absent."""
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key.strip() == "MemAvailable":
            try:
                kb = int(rest.split()[0])
            except (ValueError, IndexError) as exc:
                raise BudgetError(f"MemAvailable unparsable: {line!r}") from exc
            if kb <= 0:
                raise BudgetError(f"MemAvailable non-positive ({kb} kB); refusing to budget")
            return kb * 1024
    raise BudgetError("/proc/meminfo has no MemAvailable; cannot budget memory honestly")


def mem_available() -> int:
    return parse_meminfo(Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace"))


def disk_free(path: Path) -> int:
    return shutil.disk_usage(path).free


@dataclass
class Estimate:
    weights: int
    kv: int
    kv_known: bool
    workspace: int
    peak_override: bool
    total: int


def estimate_required(memory: dict, context_tokens: int) -> Estimate:
    weights = memory["weights_bytes"]
    kv_per_tok = memory.get("kv_bytes_per_token")
    peak = memory.get("peak_estimate_bytes")
    kv_known = kv_per_tok is not None
    kv = (kv_per_tok * context_tokens) if kv_known else 0
    if peak is not None:
        # The measurement was taken at some (unknown) context; a larger
        # requested context must not hide behind it, and the unmeasured
        # workspace floor still applies — only a measured context-specific
        # peak could justify less, and the schema does not carry one.
        workspace_floor = max(int(WORKSPACE_FRACTION * weights), WORKSPACE_MIN_BYTES)
        total = max(peak, weights + kv + workspace_floor)
        return Estimate(weights, kv, kv_known, total - weights - kv, True, total)
    workspace = max(int(WORKSPACE_FRACTION * weights), WORKSPACE_MIN_BYTES)
    if not kv_known:
        workspace += UNKNOWN_KV_MARGIN
    return Estimate(weights, kv, kv_known, workspace, False, weights + kv + workspace)


def resolve_context(context: dict, requested: int | None) -> int:
    """Explicit per-model context admission. Over the model limit is an
    error, not a silent clamp; the limit is the admission contract."""
    limit = context.get("max_tokens")
    if requested is not None:
        if limit is not None and requested > limit:
            raise BudgetError(
                f"requested context {requested} exceeds this model's limit {limit}; "
                "lower --context"
            )
        return requested
    return limit if limit is not None else DEFAULT_CONTEXT_TOKENS


def reservations_path(home: Path | None = None) -> Path:
    return (home or default_home()) / RESERVATIONS_FILE


def _load_reservations_file(path: Path) -> dict[str, dict]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        raise BudgetError(f"reservations file unreadable ({exc}); refusing to guess") from exc
    if not isinstance(obj, dict):
        raise BudgetError("reservations file must be an object of name -> reservation")
    out: dict[str, dict] = {}
    for name, val in obj.items():
        if not isinstance(val, dict) or isinstance(val.get("bytes"), bool) \
                or not isinstance(val.get("bytes"), int) or val["bytes"] <= 0:
            raise BudgetError(f"reservation {name!r}: expected positive {{bytes: int, note, state}}")
        state = val.get("state", "pending")  # legacy files were pending by definition
        if state not in RESERVATION_STATES:
            raise BudgetError(f"reservation {name!r}: unknown state {state!r}")
        floor = val.get("resident_floor_bytes")
        if floor is not None and (isinstance(floor, bool) or not isinstance(floor, int) or floor < 0):
            raise BudgetError(f"reservation {name!r}: resident_floor_bytes must be a non-negative int")
        if floor is not None and floor > val["bytes"]:
            raise BudgetError(f"reservation {name!r}: resident_floor_bytes exceeds total bytes")
        owner = val.get("owner")
        if owner is not None and not isinstance(owner, str):
            raise BudgetError(f"reservation {name!r}: owner must be a string")
        out[str(name)] = {"bytes": val["bytes"], "note": str(val.get("note", "")),
                          "state": state, "resident_floor_bytes": floor, "owner": owner}
    return out


def load_reservations(home: Path | None = None) -> dict[str, dict]:
    return _load_reservations_file(reservations_path(home))


def _contribution(r: dict) -> int:
    """Bytes a reservation holds against future admission: pending entries
    hold everything; resident entries hold only the unmaterialized peak
    headroom (weights are already inside MemAvailable)."""
    if r["state"] == "pending":
        return r["bytes"]
    floor = r.get("resident_floor_bytes")
    return r["bytes"] if floor is None else max(r["bytes"] - floor, 0)


def _transaction(mutate, home: Path | None, available_bytes: int | None = None):
    """One exclusive lock spans the read, the fit check, and the write:
    the shared atomic budget transaction. No coordinator beyond this.
    mutate receives (data, available_bytes)."""
    path = reservations_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        data = _load_reservations_file(path)
        available = available_bytes if available_bytes is not None else mem_available()
        result = mutate(data, available)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return result


def _update_reservations(mutate, home: Path | None):
    """Read-modify-write under an exclusive lock, atomic replace: no torn
    files, no lost update between concurrent launcher processes."""
    return _transaction(mutate, home)


def set_reservation(name: str, byte_count: int, note: str = "",
                    home: Path | None = None, state: str = "pending",
                    resident_floor_bytes: int | None = None,
                    owner: str | None = None) -> None:
    if not name or byte_count <= 0:
        raise BudgetError("reservation needs a name and positive bytes")
    if state not in RESERVATION_STATES:
        raise BudgetError(f"reservation state must be one of {RESERVATION_STATES}")
    _validate_floor(byte_count, resident_floor_bytes)

    def mutate(data, _available):
        existing = data.get(name)
        if existing is not None and existing.get("owner") and existing.get("owner") != owner:
            raise BudgetError(
                f"reservation {name!r} is owned by another holder "
                f"({existing['owner']}); refusing to overwrite"
            )
        data[name] = {"bytes": int(byte_count), "note": note, "state": state,
                      "resident_floor_bytes": resident_floor_bytes, "owner": owner}

    _update_reservations(mutate, home)


def _validate_floor(total: int, floor: int | None) -> None:
    if floor is None:
        return
    if isinstance(floor, bool) or not isinstance(floor, int) or floor < 0:
        raise BudgetError("resident_floor_bytes must be a non-negative int")
    if floor > total:
        raise BudgetError("resident_floor_bytes cannot exceed the reservation total")


def set_reservation_state(name: str, state: str, home: Path | None = None,
                          resident_floor_bytes: int | None = None,
                          owner: str | None = None) -> None:
    """Two-phase: pending -> resident once the weights are materialized.
    Pass resident_floor_bytes = the verified materialized baseline (at
    least the weights); the reservation then holds only the unmaterialized
    peak headroom. Without a floor the full bytes keep counting."""
    if state not in RESERVATION_STATES:
        raise BudgetError(f"reservation state must be one of {RESERVATION_STATES}")

    def mutate(data, _available):
        if name not in data:
            raise BudgetError(f"no reservation named {name!r}")
        entry = data[name]
        if entry.get("owner") and entry["owner"] != owner:
            raise BudgetError(
                f"reservation {name!r} is owned by another holder "
                f"({entry['owner']}); refusing to relabel"
            )
        total = entry["bytes"]
        floor = resident_floor_bytes
        if floor is None:
            floor = entry.get("resident_floor_bytes")
        else:
            _validate_floor(total, floor)
        entry["state"] = state
        entry["resident_floor_bytes"] = floor

    _update_reservations(mutate, home)


def clear_reservation(name: str, home: Path | None = None,
                      owner: str | None = None) -> bool:
    def mutate(data, _available):
        entry = data.get(name)
        if entry is None:
            return False
        if entry.get("owner") and entry["owner"] != owner:
            raise BudgetError(
                f"reservation {name!r} is owned by another holder "
                f"({entry['owner']}); refusing to clear"
            )
        del data[name]
        return True

    return bool(_update_reservations(mutate, home))


def generate_owner() -> str:
    """A unique per-process owner token. Deliberately excludes hostnames:
    this repository is public."""
    return f"pid{os.getpid()}-{uuid.uuid4().hex[:12]}"


@dataclass
class Admission:
    fits: bool
    required: int
    reserved: int
    available: int
    safety: int
    headroom: int  # available - safety - reserved - required
    lines: list[str]
    owner: str | None = None


def admit(required: int, home: Path | None = None) -> Admission:
    """Fit check against MemAvailable with aggregate reservations."""
    available = mem_available()
    reserved = sum(_contribution(r) for r in load_reservations(home).values())
    headroom = available - SAFETY_RESERVE_BYTES - reserved - required
    lines = [
        f"MemAvailable:      {available / GiB:.2f} GiB",
        f"safety reserve:    {SAFETY_RESERVE_BYTES / GiB:.2f} GiB",
        f"reservations:      {reserved / GiB:.2f} GiB",
        f"model requirement: {required / GiB:.2f} GiB",
        f"headroom:          {headroom / GiB:.2f} GiB",
    ]
    return Admission(headroom >= 0, required, reserved, available, SAFETY_RESERVE_BYTES, headroom, lines)


def disk_check(need: int, path: Path) -> tuple[bool, int, str]:
    free = disk_free(path)
    return free >= need, free, str(path)


def admit_and_reserve(
    name: str,
    byte_count: int,
    *,
    note: str = "",
    owner: str,
    home: Path | None = None,
    state: str = "pending",
    resident_floor_bytes: int | None = None,
    available_bytes: int | None = None,
) -> Admission:
    """The shared atomic budget transaction: the fit check and the
    reservation write happen under one exclusive lock, so concurrent
    adapters cannot both admit against the same MemAvailable and
    overcommit. The reservation is owned: a reservation held by one owner
    refuses overwrite/clear/relabel by another, so two processes serving
    the same catalog id fail loudly instead of clobbering each other.

    available_bytes exists for deterministic tests; production omits it
    and the transaction reads live MemAvailable inside the lock.
    """
    if not name or byte_count <= 0:
        raise BudgetError("admit_and_reserve needs a name and positive bytes")
    if not owner:
        raise BudgetError("admit_and_reserve needs an owner token (generate_owner())")
    _validate_floor(byte_count, resident_floor_bytes)

    def mutate(data, available):
        existing = data.get(name)
        if existing is not None and existing.get("owner") != owner:
            holder = existing.get("owner") or "an unowned manual reservation"
            raise BudgetError(
                f"reservation {name!r} already held ({holder}); two processes "
                "may not serve the same catalog id"
            )
        others = sum(_contribution(r) for key, r in data.items() if key != name)
        headroom = available - SAFETY_RESERVE_BYTES - others - byte_count
        lines = [
            f"MemAvailable:      {available / GiB:.2f} GiB",
            f"safety reserve:    {SAFETY_RESERVE_BYTES / GiB:.2f} GiB",
            f"other reservations:{others / GiB:7.2f} GiB",
            f"this reservation:  {byte_count / GiB:.2f} GiB",
            f"headroom:          {headroom / GiB:.2f} GiB",
        ]
        if headroom < 0:
            raise BudgetError(
                "does not fit: "
                + "; ".join(lines)
                + " — refusing concurrent overcommit"
            )
        data[name] = {"bytes": int(byte_count), "note": note, "state": state,
                      "resident_floor_bytes": resident_floor_bytes, "owner": owner}
        return Admission(True, byte_count, others, available,
                         SAFETY_RESERVE_BYTES, headroom, lines, owner)

    return _transaction(mutate, home, available_bytes)
