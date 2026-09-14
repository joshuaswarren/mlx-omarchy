import json

import mlx.core as mx

info = mx.device_info()
print(json.dumps({k: info[k] for k in sorted(info)}, indent=1, default=str))
