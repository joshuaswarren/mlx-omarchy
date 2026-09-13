# Frontend pad elimination (2026-09-13)

## Evidence

Apple's own ANE compiler (`ane-compile-hwx`, oracle witness recorded in
mil-hwx-compiler `receipts/2026-09-13-batched-oracle-mint-round2/`)
REJECTS the MIL `pad` op in every tested form — including the identity
view — with `callback_status=1`. macOS therefore eliminates pads from
the graph before compilation; the mlx-omarchy frontend now does the
same, and the compiler keeps its exact rejection citing this evidence.

## The encoder's pads

All 24 pads are one identical pattern (one per attention layer):

```
pad(x = matmul scores [1,8,375,749],
    amounts = [0,0,0,0, 0,0,1,0],       # (begin,end) per dim → +1 at BEGIN of W
    constant_val = +0.0, mode = "constant")
  → [1,8,375,750]
```

feeding the skew chain reshape[1,8,750,375] → slice (drop first row) →
reshape[1,8,375,749] → band slice. The pads are semantically load
bearing (they create the diagonal skew), so they cannot be deleted —
they must be rewritten.

## The rewrite

Each pad becomes a depthwise identity convolution carrying the amounts
as native conv padding (op-for-op, output name and shape unchanged):

```
conv(x,
     weight = ones[C,1,1,1] fp16, groups = C,
     strides = [1,1], dilations = [1,1],
     pad = [h1,h2,w1,w2], pad_type = "custom")
```

## Bit-exactness proof (per lane)

- **Padded lanes**: conv pads with zeros and multiplies by 1.0:
  `1.0 * 0.0 = +0.0` — identical bits (`0x0000`) to the pad op's
  constant.
- **Data lanes**: depthwise (groups=C) computes exactly `1.0 * v`.
  IEEE 754 multiplication by 1.0 is exact for every fp16 value: ±0
  keeps its sign bit (`0x8000` stays `0x8000`), subnormals are
  unchanged, ±inf pass through, quiet-NaN payloads are preserved.
- **No cross-channel terms**: depthwise means no `0 * x` accumulation —
  the failure mode that would flip a −0 lane to +0 in a full
  [C,C,1,1] identity kernel.

So the rewrite is BIT-EXACT unconditionally, not merely value-exact.

## Verification (host tests, 4/4 OK)

`overlay/tests/omarchy/coreml/test_pad_elimination.py`:

1. Synthetic pad (H-end 1, W-begin 1) on adversarial fp16 payload
   (±0, subnormals, ±max, ±inf, NaN): MIL-pad semantics vs the
   identity-conv rewrite — **bitwise (uint16) equality**; rewritten op
   carries weight = ones [C,1,1,1], groups = C, pad = [h1,h2,w1,w2],
   pad_type = custom.
2. The real encoder shape case [1,8,375,749]→[1,8,375,750] with
   planted ±0/subnormal lanes: bitwise equality.
3. Named rejections: non-zero pad constant ("only +0.0 is expressible
   as conv zero padding"), batch/channel-dim amounts.
4. Real encoder: all 24 pads eliminated (pad 24 → 0, conv 77 → 101),
   every rewritten op keeps its output name/shape, consumers unchanged.

## Eligibility delta

`mlx-omarchy-coreml inspect` (section 39) now routes pad through the
elimination planner:

- LOWERABLE-VIA-FRONTEND: 221 → **245** (+24 pads)
- NEEDS-COMPILER-OP: 253 → **229** (pad gone from the blocking list)
- post-frontend histogram: pad 0, conv 101 (folds sum-neutral)

Remaining blocking classes: transpose 146, slice_by_index 48, select 24
(−inf fill), less 4, floor 3, floor_div 3, tile 1 — stream 2/3
compiler-side work.

## Cost note

24 identity convs (1×1, depthwise, one per attention layer) replace 24
pads; each is a trivial single-task program — the encoder already
carries 77 real convs. Envelope verification (W=749/750 conv shapes)
happens at Phase 5/6 compile time with named refusals.
