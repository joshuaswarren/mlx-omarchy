import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
rows = [json.loads(s) for s in (root / 'dense-wall-split.jsonl').read_text().splitlines()]
head = [r for r in rows if r['k'] == 'd' and r['e'] == 26 and r['n'] == 151936]
assert len(head) == 51
elapsed = statistics.median((r['t1'] - r['t0']) / 1e6 for r in head[1:])
wall = next(r['median_ms'] for r in json.loads((root / 'dense-wall-split.json').read_text())['results'] if r['n'] == 151936)
assert 0.9 < elapsed / wall < 1.05, (elapsed, wall)
for model in ('4bit', 'bf16'):
    markers = [json.loads(s) for s in (root / f'{model}-markers.jsonl').read_text().splitlines()]
    assert sum(r['p'] == 'tok' for r in markers) == 32
    assert markers[-1]['p'] == 'decode_done'
print(json.dumps({'head_timestamp_median_ms': elapsed, 'head_wall_median_ms': wall, 'timestamp_over_wall': elapsed / wall, 'generation_profiles': '32 tokens each; not numerical acceptance'}))
