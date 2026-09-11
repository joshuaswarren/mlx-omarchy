import json, glob, statistics
from collections import defaultdict
seen = defaultdict(list)
for f in glob.glob("r*.json"):
    stem = f[:-5]
    rep, arm, workload = stem.split("-", 2)
    try:
        d = json.load(open(f))
    except Exception:
        continue
    for l in d.get("legs", []):
        if l["workload_id"] == workload and l["status"] == "measured":
            seen[(arm, workload)].append(
                (l["metrics"]["decode_tok_s"],
                 l["metrics"]["generated_ids_sha256_16"]))
for (arm, wl), v in sorted(seen.items()):
    tok = statistics.median([x[0] for x in v])
    digs = sorted({x[1] for x in v})
    print("%-9s %-26s n=%2d tok/s=%7.2f digests=%s" % (arm, wl, len(v), tok, digs))
