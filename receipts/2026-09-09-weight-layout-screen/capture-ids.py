import mlx_lm.utils as _utils
_original_load = _utils.load

def _layout_load(*args, **kwargs):
    model, tokenizer = _original_load(*args, **kwargs)
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    updates = []
    for key, value in tree_flatten(model.parameters()):
        if value.ndim == 2 and value.dtype == mx.bfloat16:
            packed = mx.contiguous(value.T).T
            mx.eval(packed)
            assert mx.array_equal(value, packed).item(), key
            updates.append((key, packed))
    model.update(tree_unflatten(updates))
    print("EXACT_WEIGHT_LAYOUT_CHECK", len(updates), flush=True)
    return model, tokenizer

_utils.load = _layout_load
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / 'scripts'))
import bench_decode

original_report = bench_decode.report


def capture_report(prefill_ns, token_times, requested, ids=None, prompt_tokens=None, device=None):
    if ids is None:
        raise RuntimeError('Benchmark did not provide exact generated token IDs')
    result = original_report(prefill_ns, token_times, requested, ids, prompt_tokens=prompt_tokens, device=device)
    record = {'requested': requested, 'prompt_tokens': prompt_tokens, 'ids': [int(i) for i in ids]}
    with open(os.environ['MLX_PAIR_IDS'], 'a') as output:
        output.write(json.dumps(record) + '\n')
    return result


bench_decode.report = capture_report
raise SystemExit(bench_decode.main())
