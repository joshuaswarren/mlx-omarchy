## short-decode-32
| arm | n | tok/s median | token ms |
|---|---|---|---|
| baseline | 12 | 112.705 | 8.8727 |
| gemv | 12 | 313.37 | 3.1911 |
| rope | 12 | 114.325 | 8.747 |
| rms | 12 | 116.495 | 8.5841 |
| kvwrite | 12 | 112.855 | 8.861 |
| attn | 12 | 114.02 | 8.7704 |
| swiglu | 12 | 113.155 | 8.8374 |
| sampler | 12 | 116.48 | 8.5852 |
| all | 12 | 395.895 | 2.5264 |

marginals (baseline - ablated), ms/token:
- gemv: 5.6816
- rope: 0.1257
- rms: 0.2886
- kvwrite: 0.0117
- attn: 0.1023
- swiglu: 0.0353
- sampler: 0.2875
- skeleton (all ablated): 2.5264
- sum(marginals)+skeleton: 9.0591 vs token 8.8727 -> residual -0.1864 ms
- skeleton minus 273x4.5us cadence: 1.2979 ms (host pacing + tail kernels + bodies)

## long-decode-128
| arm | n | tok/s median | token ms |
|---|---|---|---|
| baseline | 12 | 109.2 | 9.1575 |
| kvwrite | 8 | 109.53 | 9.1299 |
| attn | 8 | 111.195 | 8.9932 |
| all | 8 | 178.79 | 5.5933 |

marginals (baseline - ablated), ms/token:
- kvwrite: 0.0276
- attn: 0.1643
- skeleton (all ablated): 5.5933
- sum(marginals)+skeleton: 5.7852 vs token 9.1575 -> residual 3.3723 ms
- skeleton minus 273x4.5us cadence: 4.3648 ms (host pacing + tail kernels + bodies)

## longctx-1024-decode-32
| arm | n | tok/s median | token ms |
|---|---|---|---|
| baseline | 12 | 97.945 | 10.2098 |
| gemv | 8 | 151.425 | 6.6053 |
| rope | 8 | 98.86 | 10.1153 |
| rms | 8 | 99.33 | 10.0675 |
| kvwrite | 8 | 97.21 | 10.287 |
| attn | 8 | 101.77 | 9.8261 |
| swiglu | 8 | 97.18 | 10.2903 |
| sampler | 8 | 99.495 | 10.0514 |
| all | 8 | 155.655 | 6.4247 |

marginals (baseline - ablated), ms/token:
- gemv: 3.6045
- rope: 0.0945
- rms: 0.1423
- kvwrite: -0.0772
- attn: 0.3837
- swiglu: -0.0805
- sampler: 0.1584
- skeleton (all ablated): 6.4247
- sum(marginals)+skeleton: 10.6504 vs token 10.2098 -> residual -0.4406 ms
- skeleton minus 273x4.5us cadence: 5.1962 ms (host pacing + tail kernels + bodies)

## microbench cross-check (per token, ms)
| chain | dispatches | per-dispatch us (median) | minus floor | gpu chain ms est |
|---|---|---|---|---|
| gemv_layer | 48 | 72.658 | 68.158 | 3.2716 |
| harness | 24 | 40.221 | 35.721 | 0.8573 |
| kvwrite | 48 | 322.479 | 317.979 | 15.263 |
| lm_head | 24 | 1642.645 | 1638.145 | 39.3155 |
| rms | 49 | 23.796 | 19.296 | 0.9455 |
| rope | 48 | 36.272 | 31.772 | 1.5251 |
| sampler | 48 | 181.424 | 176.924 | 8.4923 |
| sdpa@seq1024 | 24 | 88.162 | 83.662 | 2.0079 |
| sdpa@seq16 | 24 | 30.838 | 26.338 | 0.6321 |
| sdpa@seq48 | 24 | 190.289 | 185.789 | 4.4589 |
| swiglu | 24 | 66.882 | 62.382 | 1.4972 |
