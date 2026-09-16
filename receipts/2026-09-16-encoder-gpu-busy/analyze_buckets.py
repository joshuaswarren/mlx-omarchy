import json, sys, collections
names = [l.strip() for l in open('/tmp/kernel_names.txt') if l.strip()]
path = sys.argv[1]
tot = collections.Counter(); cnt = collections.Counter()
shapes = collections.defaultdict(list)
busy = 0.0; nd = 0
for line in open(path):
    r = json.loads(line)
    if r.get('k') != 'd': continue
    if 't0' not in r: continue
    k = names[r['e']] if r['e'] < len(names) else f'UNK{r["e"]}'
    us = (r['t1'] - r['t0'])
    tot[k] += us; cnt[k] += 1; busy += us; nd += 1
    if k in ('MatmulF32Coopmat','MatmulF32','SliceUpdatePairF16','CastF16F32','CastF32F16','CopyGeneralF16','CopyGeneralF32'):
        ranges = [b[2] for b in r['b']]
        shapes[k].append((r['n'], r['gx'], r['gy'], r['gz'], ranges))
print(f"dispatches={nd} gpu_busy={busy/1e6:.1f} ms")
print(f"{'kernel':<32}{'count':>7}{'ms':>10}")
for k, v in tot.most_common(28):
    print(f"{k:<32}{cnt[k]:>7}{v/1e6:>10.1f}")
import numpy as np
for k, rows in shapes.items():
    if not rows: continue
    arr = collections.Counter((r[0], r[1], r[2], r[3], tuple(r[4])) for r in rows)
    print(f"\n== {k}: {len(rows)} disp, {len(arr)} distinct (n,gx,gy,gz,ranges) ==")
    for key, c in arr.most_common(14):
        print(f"  x{c:<4} n={key[0]:>9} g=({key[1]},{key[2]},{key[3]}) ranges={key[4]}")
