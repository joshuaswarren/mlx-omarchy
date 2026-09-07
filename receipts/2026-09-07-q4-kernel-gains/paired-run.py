import json
import os
from pathlib import Path
import statistics
import subprocess

out = Path('/home/joshuawarren/benchq/morning-20260907/five-pairs-417c06e6-defaults-verified')
out.mkdir()
manifest = json.loads(Path('/tmp/mlx-morning-q4-eager.json').read_text())
manifest['generation']['engine_script'] = '/tmp/mlx-paired-capture.py'
manifest_path = out / 'manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2))
roots = {
    'baseline': Path('/home/joshuawarren/src/mlx-perf-morning-20260907'),
    'candidate': Path('/home/joshuawarren/src/mlx-perf-candidates-20260907'),
}
wheels = {
    'baseline': Path('/home/joshuawarren/benchq/morning-20260907/wheels/mlx_omarchy-0.32.2.dev202609071152+348919c-cp314-cp314-linux_aarch64.whl'),
    'candidate': next(roots['candidate'].glob('dist/*.whl')),
}
expected_commits = {'baseline': '348919c', 'candidate': '417c06e'}
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=roots['candidate'], text=True).strip() == '417c06e6df9637c5d2dd583979598adc6926ab58'
pairs = []
for pair in range(5):
    runs = {}
    token_records = {}
    order = ['baseline', 'candidate'] if pair % 2 == 0 else ['candidate', 'baseline']
    for label in order:
        root = roots[label]
        prefix = out / f'pair-{pair + 1}-{label}'
        env = {k: v for k, v in os.environ.items() if not k.startswith('MLX_')}
        env.update(HF_HUB_OFFLINE='1', MLX_DISABLE_COMPILE='1', MLX_OMARCHY_QMM_TILE='1', MLX_OMARCHY_QMM_TILE_RB=str(int(label == 'candidate')), MLX_OMARCHY_QMM_VEC_Q4_WORD=str(int(label == 'candidate')), MLX_OMARCHY_GATED_BARRIERS='0', MLX_PAIR_IDS=str(prefix.with_suffix('.ids.jsonl')))
        if label == 'candidate':
            env.pop('MLX_OMARCHY_QMM_TILE_RB')
            env.pop('MLX_OMARCHY_QMM_VEC_Q4_WORD')
        command = [str(root / '.venv-benchmark/bin/python'), 'scripts/bench_matrix.py', '--mode', 'run', '--manifest', str(manifest_path), '--python', str(root / '.venv-benchmark/bin/python'), '--wheel', str(wheels[label]), '--expect-pins', 'qwen25-0.5b-4bit=a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3', '--out', str(prefix.with_suffix('.json')), '--timeout', '180']
        with prefix.with_suffix('.log').open('w') as log:
            subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        data = json.loads(prefix.with_suffix('.json').read_text())
        assert data['clean_check']['status'] == 'clean', (pair, label, 'contention')
        assert data['power']['source'] == 'AC', (pair, label, 'power')
        assert data['binary_provenance']['omarchy']['verified'] == 'match', (pair, label, 'binary mismatch')
        assert data['source']['harness_commit'].startswith(expected_commits[label]), (pair, label, 'source mismatch')
        assert len(data['legs']) == 3 and all(leg['status'] == 'measured' and not leg['contended'] for leg in data['legs']), (pair, label, 'incomplete run')
        ids = [json.loads(line) for line in prefix.with_suffix('.ids.jsonl').read_text().splitlines()]
        assert len(ids) == len(data['legs']), (pair, label, 'missing IDs')
        for leg, record in zip(data['legs'], ids):
            assert len(record['ids']) == leg['tokens'] == record['requested'], (pair, label, 'token count')
            assert record['prompt_tokens'] == leg['metrics']['prompt_tokens'], (pair, label, 'prompt mismatch')
        runs[label] = data
        token_records[label] = ids
    assert token_records['baseline'] == token_records['candidate'], (pair, 'exact generated IDs differ')
    paired = {'pair': pair + 1, 'order': order, 'exact_ids_equal': True, 'legs': []}
    for baseline, candidate in zip(runs['baseline']['legs'], runs['candidate']['legs']):
        assert baseline['leg_id'] == candidate['leg_id']
        row = {'workload': baseline['workload_id']}
        for metric in ['decode_tok_s', 'prefill_tok_s']:
            b, c = baseline['metrics'][metric], candidate['metrics'][metric]
            row[metric] = {'baseline': b, 'candidate': c, 'gain_pct': (c / b - 1) * 100}
        paired['legs'].append(row)
    pairs.append(paired)
    (out / 'pairs.json').write_text(json.dumps(pairs, indent=2))
    print(json.dumps(paired), flush=True)
summary = {'source': expected_commits, 'pairs': len(pairs), 'exact_ids_equal_all_pairs': True, 'workloads': []}
for index in range(3):
    row = {'workload': pairs[0]['legs'][index]['workload']}
    for metric in ['decode_tok_s', 'prefill_tok_s']:
        measurements = [pair['legs'][index][metric] for pair in pairs]
        gains = [m['gain_pct'] for m in measurements]
        median = statistics.median(gains)
        mad = statistics.median(abs(gain - median) for gain in gains)
        row[metric] = {'baseline_median': statistics.median(m['baseline'] for m in measurements), 'candidate_median': statistics.median(m['candidate'] for m in measurements), 'paired_gain_median_pct': median, 'paired_gain_mad_pct': mad, 'positive_pairs': sum(gain > 0 for gain in gains)}
    summary['workloads'].append(row)
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
