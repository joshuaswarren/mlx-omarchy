# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Build the fused ios18.lstm unary tables for vulkan_decoder.

Inputs (all measured, see receipts/2026-09-14-lstm-unary-mac.md):
  - sigma_grid.npz / sigma_args.npz  fused sigmoid exposures. Each fp16
    exposure pins the internal sigmoid to one fp16 bucket at that argument.
  - tanh_extracted.npz               fused tanh exposures (soft ±2 ulp bands).
  - capture trace-0 layer-0 golden next_cell / next_hidden (the reduction-free
    transition: token 8192 embedding row is zero and the entry state is zero,
    so layer-0 gate preacts equal the fp16 bias bits exactly).

Method:
  1. Base tables: piecewise-linear sigmoid/tanh over [-8, 8] at h = 2^-9,
     fitted by relaxed projection into the exposure buckets (sigmoid) and the
     golden-product windows (tanh).
  2. Proof refinement per stage: one sigma value per unique gate-bias
     argument, one shared tanh value per unique tanh argument, all verified
     by exact fp16 rounding; stubborn groups trigger a g-first band scan.
  3. Decollide: a cell tanh value that lands exactly on some hidden lane's
     internal product removes that lane's sigma freedom when the output-gate
     argument is shared with the cell stage; such values are nudged inside
     their own cell windows until both products verify.
  4. Splice + hill-climb: solved points override the base tables, then the
     merged tables are hill-climbed against the exact evaluator (single and
     paired moves inside each solved point's allowed window) until the
     golden 640/640 holds or no improving move remains.

Output: unary_tables.npz with args/values arrays for both unaries, consumed
by overlay/tools/coreml/vulkan_decoder.py.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import platform
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
MAC = HERE.parent / "2026-09-14-lstm-unary-mac"
CAPTURE = (
    pathlib.Path.home()
    / ".cache/mlx-omarchy/parakeet-reference/captures/b650695c-75aec2a/"
    "20260913T105550Z-librispeech-tdt-tensors/ane"
)
XLO, XHI = -8.0, 8.0
H = 2.0**-9
SCAN = (0.25, 0.75, 0.3, 0.7, 0.4, 0.6, 0.2, 0.8, 0.5, 1e-6, 1.0 - 1e-6, 0.0, 1.0)


def r16(values):
    return np.asarray(values, np.float64).astype(np.float16).astype(np.float64)


def halfq(values):
    y16 = np.asarray(values, np.float16)
    up = np.nextafter(y16, np.float16(np.inf)).astype(np.float64)
    dn = np.nextafter(y16, np.float16(-np.inf)).astype(np.float64)
    return np.minimum(np.abs(up - values), np.abs(values - dn)) / 2.0


def bucket_edges(values, eps=1e-13):
    h = halfq(values)
    return values - h + eps, values + h - eps


def load_sigma_lanes():
    sg = np.load(MAC / "sigma_grid.npz")
    sa = np.load(MAC / "sigma_args.npz")
    table = {}
    for args, sigma in ((sg["args"], sg["sigma"]), (sa["args"], sa["sigma"])):
        for x, y in zip(args.astype(np.float64), sigma.astype(np.float64)):
            table[float(x)] = float(y)
    x = np.array(sorted(table))
    return x, np.array([table[v] for v in x])


def project(knots, x, lo, hi, init, iters=40000, lam=0.95):
    """Relaxed projection of a piecewise-linear knot vector into lane windows."""
    K = knots.size
    F = np.interp(knots, x, init)
    seg = np.clip(((x - knots[0]) / H).astype(int), 0, K - 2)
    t = (x - knots[seg]) / H
    w0, w1 = 1.0 - t, t
    norm = np.zeros(K)
    np.add.at(norm, seg, w0 * w0)
    np.add.at(norm, seg + 1, w1 * w1)
    norm = np.maximum(norm, 1e-9)
    for _ in range(iters):
        f = F[seg] * w0 + F[seg + 1] * w1
        d = np.clip(f, lo, hi) - f
        np.add.at(F, seg, lam * d * w0 / norm[seg])
        np.add.at(F, seg + 1, lam * d * w1 / norm[seg + 1])
        if not np.isfinite(F).all():
            raise RuntimeError("projection diverged")
    return F


def prod_window(p, target, lo, hi):
    """Real g-window with round16(p*g) == target, clipped to [lo, hi]."""
    h = halfq(target)
    a, b = (target - h) / p, (target + h) / p
    return max(min(a, b), lo), min(max(a, b), hi)


def p_candidates(exp, n=65):
    slo, shi = bucket_edges(exp)
    return np.linspace(slo, shi, n)


def solve_stage(args, exps, targets, wins_lo, wins_hi, tan_args, fixed=None,
                fixed_tan=None):
    """One sigma value per unique sigma argument, one shared tanh value per
    unique tanh argument, both verified by exact fp16 rounding."""
    frozen = [
        fixed_tan.get(float(x)) if fixed_tan is not None else None
        for x in tan_args
    ]

    def lane_ok(p, j):
        if frozen[j] is not None:
            return r16(p * frozen[j]) == float(np.float16(targets[j]))
        w = prod_window(p, targets[j], wins_lo[j], wins_hi[j])
        return w[0] < w[1]

    def frozen_interval(j):
        g = frozen[j]
        if g is None:
            return None
        h = halfq(targets[j])
        a, b = (targets[j] - h) / g, (targets[j] + h) / g
        slo2, shi2 = bucket_edges(exps[j])
        return max(min(a, b), slo2), min(max(a, b), shi2)

    groups = {}
    for j, x in enumerate(args):
        groups.setdefault(float(x), []).append(j)
    pmap = {}
    unresolved_sigma = 0
    for arg, idx in groups.items():
        chosen = None
        if fixed is not None and arg in fixed:
            if all(lane_ok(fixed[arg], j) for j in idx):
                chosen = fixed[arg]
        else:
            extra = [frozen_interval(j) for j in idx if frozen[j] is not None]
            cands = list(p_candidates(exps[idx[0]]))
            if extra:
                lo_i = max(v[0] for v in extra)
                hi_i = min(v[1] for v in extra)
                if lo_i < hi_i:
                    cands = (list(np.linspace(lo_i, hi_i, 65))
                             + [lo_i, hi_i, 0.5 * (lo_i + hi_i)] + cands)
            for p in cands:
                if all(lane_ok(p, j) for j in idx):
                    chosen = p
                    break
        if chosen is None:
            unresolved_sigma += len(idx)
            chosen = p_candidates(exps[idx[0]])[0]
        pmap[arg] = chosen

    tan_groups = {}
    for j, x in enumerate(tan_args):
        tan_groups.setdefault(float(x), []).append(j)
    tan_map = {}
    repaired = 0
    unresolved_tan = 0
    for arg, idx in tan_groups.items():
        if frozen[idx[0]] is not None:
            tan_map[arg] = frozen[idx[0]]
            continue
        common_ok = all(lane_ok(pmap[float(args[j])], j) for j in idx)
        if common_ok:
            wins = [prod_window(pmap[float(args[j])], targets[j], wins_lo[j], wins_hi[j])
                    for j in idx]
            lo, hi = max(w[0] for w in wins), min(w[1] for w in wins)
            span = hi - lo
            g = None
            for f in SCAN:
                cand = lo + span * f
                if all(r16(pmap[float(args[j])] * cand)
                       == float(np.float16(targets[j])) for j in idx):
                    g = cand
                    break
            if g is not None:
                tan_map[arg] = g
                continue
        band_lo = max(wins_lo[j] for j in idx)
        band_hi = min(wins_hi[j] for j in idx)
        g = None
        if band_lo < band_hi:
            sub = {}
            for j in idx:
                sub.setdefault(float(args[j]), []).append(j)
            for gv in np.linspace(band_lo, band_hi, 8193):
                assign = {}
                ok = True
                for m, js in sub.items():
                    slo2, shi2 = bucket_edges(exps[js[0]])
                    lo2, hi2 = -np.inf, np.inf
                    for j in js:
                        h = halfq(targets[j])
                        a, b = (targets[j] - h) / gv, (targets[j] + h) / gv
                        lo2 = max(lo2, min(a, b), slo2)
                        hi2 = min(hi2, max(a, b), shi2)
                    if lo2 >= hi2:
                        ok = False
                        break
                    assign[m] = min(max(pmap.get(m, lo2), lo2), hi2)
                if not ok:
                    continue
                if all(r16(assign[float(args[j])] * gv)
                       == float(np.float16(targets[j])) for j in idx):
                    g = float(gv)
                    pmap.update(assign)
                    repaired += 1
                    break
        if g is None:
            unresolved_tan += len(idx)
            wins = [prod_window(pmap[float(args[j])], targets[j], wins_lo[j], wins_hi[j])
                    for j in idx]
            lo, hi = max(w[0] for w in wins), min(w[1] for w in wins)
            g = 0.5 * (lo + hi)
        tan_map[arg] = g
    gval = np.array([tan_map[float(x)] for x in tan_args])
    return pmap, tan_map, gval, unresolved_sigma, unresolved_tan, repaired


def splice(knots, F, points, report):
    """Base knot table with (x, y) proof points taking precedence."""
    conflicts = 0
    table = {}
    for x, y in points:
        key = float(x)
        if key in table:
            if abs(table[key] - y) > 0.51 * max(halfq(np.array([y]))[0], 1e-12):
                conflicts += 1
            y = 0.5 * (table[key] + y)
        table[key] = y
    base = {float(x): float(y) for x, y in zip(knots, F)}
    base.update(table)
    mx = np.array(sorted(base))
    report["splice_conflicts"] = conflicts
    return mx, np.array([base[v] for v in mx])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(HERE / "unary_tables.npz"))
    args = parser.parse_args()
    report = {
        "schema": "mlx-omarchy.lstm-fused-unary-tables/1",
        "host": platform.node(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "knot_h": H,
    }

    gold_c = np.load(CAPTURE / "tdt_trace_0000_decoder_next_cell.npy")[0].ravel().astype(np.float64)
    gold_h = np.load(CAPTURE / "tdt_trace_0000_decoder_next_hidden.npy")[0].ravel().astype(np.float64)
    sa = np.load(MAC / "sigma_args.npz")
    te = np.load(MAC / "tanh_extracted.npz")
    b_i = sa["args"][:640].astype(np.float64)
    b_o = sa["args"][640:].astype(np.float64)
    b_g = te["args"][:640].astype(np.float64)
    sig_i_exp = sa["sigma"][:640].astype(np.float64)
    sig_o_exp = sa["sigma"][640:].astype(np.float64)

    # ---- 1. base sigmoid table -------------------------------------------------
    sx, sy = load_sigma_lanes()
    slo, shi = bucket_edges(sy)
    mid = (sx >= XLO) & (sx <= XHI)
    knots = np.arange(XLO, XHI + H / 2, H)
    F = project(knots, sx[mid], slo[mid], shi[mid], 1.0 / (1.0 + np.exp(-sx[mid])))
    sig_exact = int((r16(np.interp(sx[mid], knots, F)) == sy[mid]).sum())
    report["sigma_base"] = {"lanes": int(mid.sum()), "exposure_exact": sig_exact}
    print("sigma base:", json.dumps(report["sigma_base"]), flush=True)

    # ---- 2. base tanh table ----------------------------------------------------
    h_c = halfq(gold_c)
    h_h = halfq(gold_h)
    plo, phi = bucket_edges(sig_i_exp)
    a, b = (gold_c - h_c) / plo, (gold_c + h_c) / phi
    lo_g, hi_g = np.minimum(a, b) + 1e-13, np.maximum(a, b) - 1e-13
    plo, phi = bucket_edges(sig_o_exp)
    a, b = (gold_h - h_h) / plo, (gold_h + h_h) / phi
    lo_h = np.minimum(a, b) - h_c + 1e-13
    hi_h = np.maximum(a, b) + h_c - 1e-13
    anchor_x = np.array([0.0, 2.0, 5.0, 8.0])
    anchor_lo = np.array([0.0, 0.96484375, 1.0, 1.0]) - 1e-9
    anchor_hi = np.array([0.0, 0.96484375, 1.0, 1.0]) + 1e-9
    t_x = np.concatenate([b_g, gold_c, anchor_x])
    t_lo = np.concatenate([lo_g, lo_h, anchor_lo])
    t_hi = np.concatenate([hi_g, hi_h, anchor_hi])
    keep = t_lo < t_hi
    G = project(knots, t_x[keep], t_lo[keep], t_hi[keep], np.tanh(t_x[keep]))
    report["tanh_base"] = {"lanes": int(keep.sum()), "dropped": int((~keep).sum())}
    print("tanh base:", json.dumps(report["tanh_base"]), flush=True)

    # ---- 3. cell stage ---------------------------------------------------------
    tg = te["tanh_cell_gate"].astype(np.float64)
    ulf = halfq(np.where(np.isfinite(tg), tg, 1.0)) * 2
    tg_lo = np.where(np.isfinite(tg), tg - 2 * ulf, -1.0)
    tg_hi = np.where(np.isfinite(tg), tg + 2 * ulf, 1.0)

    p_i, tan_g_map, solved_tg, us, ut, rep = solve_stage(
        b_i, sig_i_exp, gold_c, tg_lo, tg_hi, b_g,
    )
    solved_si = np.array([p_i[float(x)] for x in b_i])
    cell_exact = int((r16(solved_si * solved_tg) == gold_c).sum())
    report["cell_refine"] = {"exact": cell_exact, "of": 640, "unresolved_sigma": us,
                             "unresolved_tanh": ut, "repaired_groups": rep}
    print("cell refine:", json.dumps(report["cell_refine"]), flush=True)

    c1 = solved_si * solved_tg  # internal cell state (sub-fp16)

    # ---- 3a. decollide origin: a cell product that lands exactly on another
    # lane's gate-bias value becomes a frozen table key and can
    # over-constrain the hidden stage; re-solve such a lane's own cell pair
    # so its product lands off the grid, keeping every group mate verifying.
    bg_key_set = set(map(float, b_g))
    for j in range(640):
        k = float(c1[j])
        if k not in bg_key_set:
            continue
        sa = float(b_i[j])
        ta = float(b_g[j])
        sig_mates = [m for m in range(640) if float(b_i[m]) == sa and m != j]
        tan_mates = [m for m in range(640) if float(b_g[m]) == ta and m != j]
        done_pair = False
        lo, hi = prod_window(1.0, gold_c[j], tg_lo[j], tg_hi[j])
        for p_new in p_candidates(sig_i_exp[j]):
            if any(float(np.float16(gold_c[m])) != r16(p_new * tan_g_map[float(b_g[m])])
                   for m in sig_mates):
                continue
            lo, hi = prod_window(1.0, gold_c[j], tg_lo[j], tg_hi[j])
            for m in tan_mates:
                pm = p_new if float(b_i[m]) == sa else p_i[float(b_i[m])]
                w = prod_window(pm, gold_c[m], tg_lo[m], tg_hi[m])
                lo, hi = max(lo, w[0]), min(hi, w[1])
            if lo >= hi:
                continue
            for gv in np.linspace(lo, hi, 1025):
                if float(p_new * gv) in bg_key_set:
                    continue
                bad = False
                for m in tan_mates:
                    pm = p_new if float(b_i[m]) == sa else p_i[float(b_i[m])]
                    if float(np.float16(gold_c[m])) != r16(pm * gv):
                        bad = True
                        break
                if bad:
                    continue
                p_i[sa] = p_new
                tan_g_map[ta] = gv
                solved_si[j] = p_new
                solved_tg[j] = gv
                c1[j] = p_new * gv
                for m in sig_mates:
                    solved_si[m] = p_new
                for m in tan_mates:
                    solved_tg[m] = gv
                    c1[m] = (p_new if float(b_i[m]) == sa
                             else p_i[float(b_i[m])]) * gv
                done_pair = True
                break
            if done_pair:
                break

    # ---- 3b. decollide ---------------------------------------------------------
    nudges = 0
    for j in range(640):
        k = float(c1[j])
        if k not in tan_g_map or float(b_o[j]) not in p_i:
            continue
        p = p_i[float(b_o[j])]
        if r16(p * tan_g_map[k]) == float(np.float16(gold_h[j])):
            continue
        members = [m for m in range(640) if float(b_g[m]) == k]
        lo, hi = -1.0, 1.0
        for m in members:
            w = prod_window(p_i[float(b_i[m])], gold_c[m], tg_lo[m], tg_hi[m])
            lo, hi = max(lo, w[0]), min(hi, w[1])
        hh = halfq(gold_h[j])
        a, b = (gold_h[j] - hh) / p, (gold_h[j] + hh) / p
        lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
        if lo >= hi:
            continue
        for f in SCAN:
            v = lo + (hi - lo) * f
            if all(r16(p_i[float(b_i[m])] * v) == float(np.float16(gold_c[m]))
                   for m in members) and \
               r16(p * v) == float(np.float16(gold_h[j])):
                tan_g_map[k] = v
                for m in members:
                    c1[m] = p_i[float(b_i[m])] * v
                    solved_tg[m] = v
                nudges += 1
                break
    report["decollide"] = {"nudges": nudges}
    print("decollide:", json.dumps(report["decollide"]), flush=True)

    # ---- 4. hidden stage against frozen cell values ---------------------------
    p_o, tan_c_map, solved_tc, us, unt, rep = solve_stage(
        b_o, sig_o_exp, gold_h, np.full(640, -1.0), np.ones(640), c1,
        fixed=p_i, fixed_tan=tan_g_map,
    )
    solved_so = np.array([p_o[float(x)] for x in b_o])
    hid_exact = int((r16(solved_so * solved_tc) == gold_h).sum())
    report["hidden_refine"] = {"exact": hid_exact, "of": 640, "unresolved_sigma": us,
                               "unresolved_tanh": unt, "repaired_groups": rep}
    print("hidden refine:", json.dumps(report["hidden_refine"]), flush=True)

    # ---- 5. splice exact knots -------------------------------------------------
    sig_pts = dict(p_i)
    sig_pts.update(p_o)
    sig_x, sig_y = splice(knots, F, sorted(sig_pts.items()), report)
    tan_pts = dict(tan_g_map)
    for k, v in tan_c_map.items():
        tan_pts[k] = tan_g_map.get(k, v)
    tan_x, tan_y = splice(knots, G, sorted(tan_pts.items()), report)

    # ---- 5b. hill-climb the merged tables against the exact evaluator ----------
    sig_exp_map = {}
    for j in range(640):
        sig_exp_map[float(b_i[j])] = sig_i_exp[j]
        sig_exp_map[float(b_o[j])] = sig_o_exp[j]
    sig_pos = {}
    for i, x in enumerate(sig_x):
        sig_pos.setdefault(float(x), i)
    tan_pos = {}
    for i, x in enumerate(tan_x):
        tan_pos.setdefault(float(x), i)
    moves = []
    for x, e in sig_exp_map.items():
        i = sig_pos.get(x)
        if i is None:
            continue
        lo, hi = bucket_edges(np.array([float(e)]))
        moves.append(("s", i, np.linspace(float(lo[0]), float(hi[0]), 17)))
    for x, v in list(tan_pts.items()):
        i = tan_pos.get(x)
        if i is None:
            continue
        u = float(max(halfq(np.array([float(v)]))[0], 1e-9))
        moves.append(("t", i, np.linspace(float(v) - u, float(v) + u, 17)))

    def total_fails(sy_, ty_):
        se = lambda x: np.interp(x, sig_x, sy_)
        te = lambda x: np.interp(x, tan_x, ty_)
        cc1 = se(b_i) * te(b_g)
        fc = int((r16(se(b_i) * te(b_g)) != gold_c).sum())
        fh = int((r16(se(b_o) * te(cc1)) != gold_h).sum())
        return fc + fh

    fails_now = total_fails(sig_y, tan_y)
    rounds = 0
    while fails_now > 0 and rounds < 60:
        rounds += 1
        improved = False
        for kind, i, grid in moves:
            orig = sig_y[i] if kind == "s" else tan_y[i]
            for v in grid:
                v = float(v)
                if v == orig:
                    continue
                if kind == "s":
                    sig_y[i] = v
                else:
                    tan_y[i] = v
                f = total_fails(sig_y, tan_y)
                if f < fails_now:
                    fails_now = f
                    improved = True
                    break
                if kind == "s":
                    sig_y[i] = orig
                else:
                    tan_y[i] = orig
            if fails_now == 0:
                break
        if improved:
            continue
        # joint pair moves on a failing lane's own two table entries
        se = lambda x: np.interp(x, sig_x, sig_y)
        te = lambda x: np.interp(x, tan_x, tan_y)
        cc1 = se(b_i) * te(b_g)
        cell_bad = np.flatnonzero(r16(se(b_i) * te(b_g)) != gold_c)
        hid_bad = np.flatnonzero(r16(se(b_o) * te(cc1)) != gold_h)
        grids = {(k, i): g for k, i, g in moves}
        done = False
        for j in cell_bad:
            si = sig_pos.get(float(b_i[j]))
            ti = tan_pos.get(float(b_g[j]))
            gs = grids.get(("s", si)) if si is not None else None
            gt = grids.get(("t", ti)) if ti is not None else None
            if gs is None or gt is None:
                continue
            os_, ot = sig_y[si], tan_y[ti]
            for vs in gs:
                for vt in gt:
                    sig_y[si], tan_y[ti] = float(vs), float(vt)
                    f = total_fails(sig_y, tan_y)
                    if f < fails_now:
                        fails_now, improved, done = f, True, True
                        break
                    sig_y[si], tan_y[ti] = os_, ot
                if done:
                    break
            if done:
                break
        for j in hid_bad:
            if done:
                break
            si = sig_pos.get(float(b_o[j]))
            ti = tan_pos.get(float(cc1[j]))
            gs = grids.get(("s", si)) if si is not None else None
            gt = grids.get(("t", ti)) if ti is not None else None
            if gs is None or gt is None:
                continue
            os_, ot = sig_y[si], tan_y[ti]
            for vs in gs:
                for vt in gt:
                    sig_y[si], tan_y[ti] = float(vs), float(vt)
                    f = total_fails(sig_y, tan_y)
                    if f < fails_now:
                        fails_now, improved, done = f, True, True
                        break
                    sig_y[si], tan_y[ti] = os_, ot
                if done:
                    break
            if done:
                break
        if not improved:
            break
    report["hill_climb"] = {"rounds": rounds, "fails_left": fails_now}
    print("hill climb:", json.dumps(report["hill_climb"]), flush=True)

    # ---- 6. verify with the merged tables (the shipped evaluator) --------------
    sig_eval = lambda x: np.interp(x, sig_x, sig_y)
    tan_eval = lambda x: np.interp(x, tan_x, tan_y)
    c1_eval = sig_eval(b_i) * tan_eval(b_g)
    cell = r16(sig_eval(b_i) * tan_eval(b_g))
    hid = r16(sig_eval(b_o) * tan_eval(c1_eval))
    nc = int((cell == gold_c).sum())
    nh = int((hid == gold_h).sum())
    report["golden"] = {"next_cell": nc, "next_hidden": nh}
    print("golden verification: next_cell", nc, "/640  next_hidden", nh, "/640", flush=True)

    np.savez_compressed(
        args.output,
        sigmoid_x=sig_x, sigmoid_y=sig_y, tanh_x=tan_x, tanh_y=tan_y,
    )
    report["output"] = str(args.output)
    report["table_sizes"] = {"sigmoid": int(sig_x.size), "tanh": int(tan_x.size)}
    (HERE / "fit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("wrote", args.output, json.dumps(report["table_sizes"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
