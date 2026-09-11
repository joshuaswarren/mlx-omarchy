#!/usr/bin/env python3
"""Root-cause clustering + delta analysis for receipts/2026-09-11-upstream-suite.

Inputs:
  py/*.xml      junit XML per upstream test file (from tools/run-upstream-suite.sh)
  py/summary.tsv
  ../upstream-suite-2026-09-06-py4/case-classification.csv   (baseline)

Outputs (written next to this script):
  case-classification.csv          same schema as the py4 receipt (file,case,kind,detail)
  root-cause-clusters.csv          rank-ordered clusters: kind,primitive,signature,n_cases,example
  delta-vs-2026-09-06.csv          per-file per-kind case-level delta, including regressions
  per-file-wall-times.csv          per-file junit wall seconds + near-ceiling flag

Kinds mirror tools/analyze-py-suite.py: named / assert / other (cpucpu, ncpuimpl kept).
"""
import collections
import glob
import os
import re
import sys
import xml.etree.ElementTree as ET

RECEIPT = os.path.dirname(os.path.abspath(__file__))
PY_DIR = os.path.join(RECEIPT, "py")
PY4_CSV = os.path.join(RECEIPT, "..", "upstream-suite-2026-09-06-py4", "case-classification.csv")

NAMED = re.compile(r"RuntimeError: \[omarchy\] (.+?) is not implemented")
FIRST_LINE = lambda s: (s or "").strip().split("\n")[0]


def kind_of(msg):
    first = FIRST_LINE(msg)
    m = NAMED.match(first)
    if m:
        return "named", m.group(1)
    if "vector::_M_range_check" in first:
        return "cpucpu", "IndexError cpu stream table"
    if "has no CPU implementation" in first:
        return "ncpuimpl", FIRST_LINE(first.replace("RuntimeError: ", ""))[:90]
    if first.startswith("AssertionError"):
        return "assert", ""
    if first.startswith("UnboundLocalError"):
        return "other", "UnboundLocalError custom_kernel (upstream harness artifact)"
    return "other", first[:90]


def blur(msg):
    """Normalize a failure line into a cluster signature: blur numbers."""
    s = re.sub(r"\d+\.\d+(e[+-]?\d+)?", "N", msg)
    s = re.sub(r"\d+", "N", s)
    s = re.sub(r"0x[0-9a-fA-F]+", "HEX", s)
    return s[:220]


PRIM = re.compile(r"\b(mx|nn)\.([A-Za-z_][A-Za-z0-9_.]*)")


def primitive_of(text):
    """Primitive under test: last mx./nn. call on the test-body frame lines."""
    calls = PRIM.findall(text or "")
    return calls[-1][1] if calls else ""


def main():
    rows = []          # (file, case, kind, detail, msg, text, wall)
    walls = {}
    for xf in sorted(glob.glob(os.path.join(PY_DIR, "*.xml"))):
        stem = os.path.basename(xf)[:-4]
        root = ET.parse(xf).getroot()
        suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
        walls[stem] = sum(float(s.get("time", 0)) for s in suites)
        for suite in suites:
            for tc in suite.iter("testcase"):
                for fail in list(tc.findall("failure")) + list(tc.findall("error")):
                    msg = fail.get("message") or (fail.text or "")
                    text = (fail.text or "")[:4000]
                    kind, detail = kind_of(msg)
                    rows.append((stem, tc.get("name"), kind, detail, msg, text))
    with open(os.path.join(RECEIPT, "case-classification.csv"), "w") as out:
        out.write("file,case,kind,detail\n")
        for stem, case, kind, detail, _, _ in rows:
            out.write(",".join(x.replace(",", ";") for x in (stem, case, kind, detail)) + "\n")

    # 2. clusters: named by gap name; assert/other by blurred first line
    clusters = collections.defaultdict(list)
    for stem, case, kind, detail, msg, text in rows:
        if kind == "named":
            sig = detail
            prim = detail.split()[-1] if detail else ""
        elif kind == "assert":
            sig = blur(FIRST_LINE(msg))
            prim = primitive_of(text)
        else:
            sig = blur(detail if detail else FIRST_LINE(msg))
            prim = primitive_of(text) if kind == "other" else ""
        clusters[(kind, prim, sig)].append(f"{stem}::{case}")

    with open(os.path.join(RECEIPT, "root-cause-clusters.csv"), "w") as out:
        out.write("kind,primitive,signature,n_cases,examples\n")
        ranked = sorted(clusters.items(), key=lambda kv: (-len(kv[1]), kv[0][0]))
        for (kind, prim, sig), cases in ranked:
            out.write(",".join([
                kind, prim or "", '"' + sig.replace('"', "'") + '"',
                str(len(cases)), '"' + ";".join(cases[:3]) + '"']) + "\n")

    # 3. delta vs py4 at case level
    py4 = collections.defaultdict(set)
    kinds_py4 = {}
    with open(PY4_CSV) as fh:
        next(fh)
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 3:
                continue
            key = (parts[0], parts[1])
            py4[key].add(parts[2])
            kinds_py4[key] = parts[2]
    now = collections.defaultdict(set)
    for stem, case, kind, _, _, _ in rows:
        now[(stem, case)].add(kind)

    with open(os.path.join(RECEIPT, "delta-vs-2026-09-06.csv"), "w") as out:
        out.write("file,case,baseline_kind,current_kind,delta\n")
        keys = sorted(set(py4) | set(now))
        for key in keys:
            b = ",".join(sorted(py4.get(key, {"PASS"})))
            c = ",".join(sorted(now.get(key, {"PASS"})))
            if b == c:
                continue
            delta = "regression" if b == "PASS" else (
                "fixed" if c == "PASS" else "changed")
            out.write(f'{key[0]},{key[1]},"{b}","{c}",{delta}\n')

    # 4. per-file wall times
    with open(os.path.join(RECEIPT, "per-file-wall-times.csv"), "w") as out:
        out.write("file,junit_seconds,near_900s_ceiling\n")
        for stem, wall in sorted(walls.items(), key=lambda kv: -kv[1]):
            out.write(f"{stem},{wall:.1f},{'YES' if wall >= 600 else ''}\n")

    # console summary
    kinds = collections.Counter(r[2] for r in rows)
    print("failure kinds:", dict(kinds))
    print("total failing cases:", len(rows))
    print("top clusters:")
    for (kind, prim, sig), cases in ranked[:15]:
        print(f"  {len(cases):4d} {kind:7s} {prim:28s} {sig[:90]}")
    reg = sum(1 for k in keys if py4.get(k) and not now.get(k))
    print("fixed (fail->pass):", reg)


if __name__ == "__main__":
    main()
