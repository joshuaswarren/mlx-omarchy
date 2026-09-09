import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

out = Path(sys.argv[1])
out.mkdir()
x = np.zeros((1, 1, 261, 64), dtype=np.float32)
x[..., :32] = 1
for dtype, label in ((mx.float32, 'f32'), (mx.float16, 'f16')):
    result = mx.fast.rope(mx.array(x).astype(dtype), 64, traditional=False, base=1000000.0, scale=1.0, offset=0)
    mx.eval(result)
    np.save(out / f'basis-{label}.npy', np.array(result.astype(mx.float32)))
(out / 'version.json').write_text(json.dumps({'version': mx.__version__, 'device': str(mx.default_device())}))
print(out, mx.__version__)
