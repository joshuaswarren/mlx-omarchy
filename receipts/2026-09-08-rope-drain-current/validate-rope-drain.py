import hashlib
import json
import math
from pathlib import Path

root = Path('/home/joshuawarren/.config/superpowers/worktrees/mlx-omarchy/wave-integrate')
folder = root / 'receipts/2026-09-08-rope-drain-current'
expected = [json.loads(s)['ids'] for s in (root / 'receipts/2026-09-08-dense-final/pair-1-candidate/rep1.ids.jsonl').read_text().splitlines()]
count = 0
for p in sorted(folder.glob('**/rep1.json')):
    data = json.loads(p.read_text())
    assert data['clean_check']['status'] == 'clean', p
    assert data['power']['source'] == 'AC', p
    assert data['binary_provenance']['omarchy']['verified'] == 'match', p
    legs = [x for x in data['legs'] if x['status'] == 'measured']
    records = [json.loads(s) for s in p.with_suffix('.ids.jsonl').read_text().splitlines()]
    assert len(legs) == len(records) == 6
    assert [r['ids'] for r in records] == expected, p
    for leg, record in zip(legs, records):
        assert len(record['ids']) == leg['tokens'] == record['requested']
        assert record['prompt_tokens'] == leg['metrics']['prompt_tokens']
        digest = hashlib.sha256(",".join(map(str, record["ids"])).encode("ascii")).hexdigest()[:16]
        assert digest == leg["metrics"]["generated_ids_sha256_16"], p
        for metric in ('decode_tok_s', 'prefill_tok_s'):
            value = leg['metrics'][metric]
            assert math.isfinite(value) and value > 0
        count += 1
assert count == 66, count
print('Verified 66 full-ID-matched legs, clean AC power and wheel provenance; all timings finite and positive.')
