import itertools, sys
import numpy as np
_a=sys.argv[:]; sys.argv=['x','/tmp/pgp/data']
import qmv_model as M
f32=np.float32
d=np.load(_a[1] if len(_a)>1 else '/tmp/pgp/dump/dotonly-896-float32-0.npz')
x,w,sc,bi,dev=d['x'],d['words'],d['scales'],d['biases'],d['device']
K=x.shape[0]; N=w.shape[0]; SIMD=32; BLOCK=256
nb=(K+BLOCK-1)//BLOCK
kk=(np.arange(nb)[:,None,None]*BLOCK+np.arange(SIMD)[None,:,None]*8+np.arange(8)[None,None,:])
valid=kk<K
xs=np.where(valid,x[np.minimum(kk,K-1)],f32(0))
xt=xs/np.array([1,16,256,4096]*2,f32)
widx=np.arange(nb)[:,None]*SIMD+np.arange(SIMD)[None,:]
words=np.where((widx<K//8)[None],w[:,np.minimum(widx,K//8-1)],np.uint32(0))
ws=np.stack([words&0xFFFF,words>>16],-1)
m=(ws[...,None]&np.array([0xF,0xF0,0xF00,0xF000],np.uint32)).astype(f32)  # (N,B,32,2,4)
p=(xt.reshape(nb,SIMD,2,4)[None]*m)  # exact products (N,B,32,2,4)
pp=p.reshape(N,nb,SIMD,8)
gidx=np.arange(nb)[:,None]*4+np.arange(SIMD)[None,:]//8
scb=np.where((gidx<K//64)[None],sc[:,np.minimum(gidx,K//64-1)],f32(0)).astype(f32)
def finish(dot):
    qd=M.fma(scb,dot,f32(0))
    r=np.zeros((N,SIMD),f32)
    for b in range(nb): r=r+qd[:,b,:]
    v=r.copy()
    for o in [1,2,4,8,16]: v=v+v[:,np.arange(SIMD)^o]
    return v[:,0]
def assoc_seq(order):
    acc=np.zeros(pp.shape[:-1],f32)
    for i in order: acc=acc+pp[...,i]
    return acc
cands={}
cands['A quads ((0+1)+2)+3 then q0+q1']=(((pp[...,0]+pp[...,1])+pp[...,2])+pp[...,3])+(((pp[...,4]+pp[...,5])+pp[...,6])+pp[...,7])
cands['C sequential 0..7']=assoc_seq(range(8))
cands['C rev 7..0']=assoc_seq(range(7,-1,-1))
cands['E pairwise']=((pp[...,0]+pp[...,1])+(pp[...,2]+pp[...,3]))+((pp[...,4]+pp[...,5])+(pp[...,6]+pp[...,7]))
cands['F right-assoc per quad']=(pp[...,0]+(pp[...,1]+(pp[...,2]+pp[...,3])))+(pp[...,4]+(pp[...,5]+(pp[...,6]+pp[...,7])))
cands['G interleave 0,4,1,5,2,6,3,7']=assoc_seq([0,4,1,5,2,6,3,7])
cands['H nibble-major: sum over quads per position then']=((pp[...,0]+pp[...,4])+(pp[...,1]+pp[...,5]))+((pp[...,2]+pp[...,6])+(pp[...,3]+pp[...,7]))
cands['I quad1 first']=(((pp[...,4]+pp[...,5])+pp[...,6])+pp[...,7])+(((pp[...,0]+pp[...,1])+pp[...,2])+pp[...,3])
for name,dot in cands.items():
    print(f'{np.count_nonzero(finish(dot)!=dev):4d}/{N}  {name}')
# exhaustive: all sequential permutations of 8 (40320) is feasible? 40320 * cost... try a subset: all orders within-quad + quad combos
best=None
for perm in itertools.permutations(range(8)):
    mm=np.count_nonzero(finish(assoc_seq(perm))!=dev)
    if best is None or mm<best[0]: best=(mm,perm); print('best seq perm so far', best, flush=True)
    if mm==0: break
