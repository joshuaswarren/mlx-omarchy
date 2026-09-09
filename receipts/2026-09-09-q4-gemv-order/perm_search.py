import itertools, os, sys, json
import numpy as np
sys.argv = ['x', sys.argv[1]]
import qmv_model as M
from safetensors.numpy import load_file
DATA = M.DATA
EXP = M.Path(os.environ.get('EXPECTED', str(DATA)))
tensors = load_file(str(DATA / 'model.safetensors'))
projs = {'q_proj': ('self_attn','attn'),'k_proj': ('self_attn','attn'),'v_proj': ('self_attn','attn'),'o_proj': ('self_attn','attn'),'gate_proj': ('mlp','mlp'),'up_proj': ('mlp','mlp'),'down_proj': ('mlp','mlp')}
combine = os.environ.get('COMBINE', 'fma_scale')
res = {}
for name,(mod,label) in projs.items():
    pre=f'model.layers.0.{mod}.{name}'
    w=tensors[pre+'.weight']; sc=tensors[pre+'.scales'].astype(np.float32); bi=tensors[pre+'.biases'].astype(np.float32); lb=tensors.get(pre+'.bias')
    x=np.load(DATA/f'call35-{label}-{name}-input.npy').reshape(-1)
    exp=np.load(EXP/f'call35-{label}-{name}-output.npy').reshape(-1)
    part=M.lane_partials(x,w,sc,bi,'nofma',combine)
    res[name]={}
    for perm in itertools.permutations([1,2,4,8,16]):
        v=part.copy()
        for o in perm: v=v+v[:,np.arange(32)^o]
        y=M.h(v[:,0])
        if lb is not None: y=(y.astype(np.float64)+lb.astype(np.float64)).astype(np.float16).astype(np.float32)
        res[name][perm]=int(np.count_nonzero(y!=exp))
    # sequential variants too
    for tag,v in (('seq',None),):
        pass
keys=list(res['q_proj'])
exact=[k for k in keys if all(res[n][k]==0 for n in res)]
print('combine',combine,'exact perms for all:',exact)
for n in res:
    print(n, 'min', min(res[n].values()), 'perms attaining min', [k for k,v in res[n].items() if v==min(res[n].values())][:8])
