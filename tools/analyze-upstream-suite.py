#!/usr/bin/env python3
# Classify doctest XML failure reports from tools/run-upstream-suite.sh
# (phase 1, C++) into the four coverage buckets used by
# receipts/2026-09-01-upstream-suite-coverage.md:
#   a  genuine omarchy gap, named [omarchy] error
#      (flagged "masked" when every failed assert is the test-suite's own
#       array_equal machinery hitting the bool And/Equal gap)
#   b  CPU-backend-absence artifact (explicit Device::cpu use in the test)
#   c  out-of-scope module (fft, linalg/LAPACK, export/import)
#   d  wrong values without a named error
#
# Usage: tools/analyze-upstream-suite.py <cpp-xml-dir> [--csv OUT.csv]
import collections
import glob
import os.path
import re
import sys
import xml.etree.ElementTree as ET

MACHINERY = re.compile(r'^"\[omarchy\] (And|Equal) dtype is not implemented.*dtype=bool')
RANGE_CHECK = "vector::_M_range_check"


def classify_file(stem, first_exc, any_exc):
    # Roadmap-excluded modules: the whole upstream file is out of scope.
    if stem in ("fft_tests", "export_import_tests", "linalg_tests"):
        return "c"
    if any("FFT" in e for e in any_exc):
        return "c"
    if first_exc and RANGE_CHECK in first_exc:
        return "b"
    return "a"


def main():
    xml_dir = sys.argv[1]
    csv_path = None
    if "--csv" in sys.argv:
        csv_path = sys.argv[sys.argv.index("--csv") + 1]

    metadata = ("flags().", "data_size()", "siblings()")
    rows = []
    for xf in sorted(glob.glob(f"{xml_dir}/*.xml")):
        stem = os.path.basename(xf)[:-4]
        tree = ET.parse(xf)
        for c in tree.findall(".//TestCase"):
            if c.get("skipped") == "true":
                continue
            a = c.find("OverallResultsAsserts")
            if a is None or a.get("test_case_success") != "false":
                continue
            name = c.get("name")
            excs, noexc = [], []
            for e in c.findall(".//Expression[@success='false']"):
                exc = e.find("Exception")
                if exc is not None:
                    excs.append((exc.text or "").strip().split("\n")[0])
                else:
                    orig = (e.findtext("Original") or "").strip()
                    exp = (e.findtext("Expanded") or "").strip()
                    noexc.append((e.get("line"), orig, exp))
            case_level = [(x.text or "").strip() for x in c.findall("Exception")]
            excs += case_level
            first_exc = excs[0] if excs else ""
            cat = classify_file(stem, first_exc, excs)
            if any(not any(p in o for p in metadata) for _, o, _ in noexc):
                cat = "d"
            masked = (
                cat == "a"
                and bool(excs)
                and all(MACHINERY.match(x) for x in excs if x.startswith('"'))
                and not case_level
            )
            rows.append((stem, name, cat, "masked" if masked else "",
                         first_exc[:130]))

    out = open(csv_path, "w") if csv_path else sys.stdout
    if csv_path:
        out.write("file,case,category,flag,first_error\n")
    for r in rows:
        line = f"{r[0]},{r[1]},{r[2]},{r[3]},\"{r[4]}\"\n"
        if csv_path:
            out.write(line)
        else:
            print(line.replace('"', ""))
    if csv_path:
        out.close()

    per = collections.defaultdict(collections.Counter)
    for r in rows:
        per[r[0]][r[2]] += 1
    print("per-file failing-case categories:", file=sys.stderr)
    for f in sorted(per):
        c = per[f]
        print(f"  {f:26s} a={c['a']:3d} b={c['b']:2d} c={c['c']:2d} d={c['d']:2d}",
              file=sys.stderr)
    tot = collections.Counter(r[2] for r in rows)
    masked_n = sum(1 for r in rows if r[3] == "masked")
    print(f"TOTAL failing cases: {len(rows)}; "
          + " ".join(f"{k}={tot[k]}" for k in "abcd")
          + f"; a-masked={masked_n}", file=sys.stderr)


if __name__ == "__main__":
    main()
