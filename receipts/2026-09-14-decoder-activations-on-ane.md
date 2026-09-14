# 2026-09-14 Parakeet decoder activations: the H13 hardware, asked directly

Status: the hypothesis is dead on device evidence, and the hardware interpolator
is identified as a by-product. The native decoder's `sigmoid` and `tanh` are not
the H13 unary lookup table plus its interpolator: the T8103 ANE reproduces 0 of 11
pinned native `sigmoid` values and 1 of 8 pinned `tanh` values, earns 50 of 1280
reduction-free lanes against the fp16 chain's 986, and is refuted on 171 of 1198
`tanh` arguments and 194 of 1172 `sigmoid` arguments. Those are the same counts
the host-side table model predicted, so the table model was right about the
hardware and the hardware is not what evaluated the decoder.

- Still open: `parakeet.tdt.decoder-lstm.fp16-elementwise-activation-vs-native-coreml`.
- Newly closed: the open question in mil-hwx-compiler `docs/ane/numerics.md`, "how
  does the hardware interpolate between the entries of a unary lookup table". The
  rule below reproduces every one of 1536 device `tanh` lanes and 1533 of 1536
  device `sigmoid` lanes, with the three exceptions named.
- Compute-unit finding: the capture records only `compute_units=ane` as *requested*,
  and its own receipt disclaims per-operation placement. No compute plan exists in
  the capture. The device result settles what the manifest cannot: whatever ran the
  decoder's `lstm`, it was not this table.

No shipped code changed. `_lstm` is unchanged per the change gate.

## Execution

Device: jwm1 (`jwm1-linux`, `apple,t8103-ane`, kernel `7.1.6-1-1-ARCH`, module
`f2a3e5e+lifecycle6`), ANE only. Ownership: Main confirmed, AneResidentCache
released the ANE to me at 11:17 CDT and I handed it back at 11:18:56 CDT.

- Six one-shot submits, 11:18:46 CDT, one `mlx-omarchy-ane-worker` process per
  submit, `--deadline-ms 5000 --iterations 1`. Every submit: `worker status=0
  iterations=1 released=1 elapsed_ms=1..2`. Worker exit 1 on each is the
  `--expect` mismatch against a correctly-rounded oracle, by design; the output
  was saved before the compare.
- Pre-state and post-state identical: `boot_id 95872130-fd28-4d67-8247-03a0e8b61202`,
  `ane refcnt 0`, zero worker processes (`/proc` cmdline scan), zero `accel0`
  holders (`fuser`), 0 `tm-execution-failed`, 0 `errno -110` in `dmesg`.
- Not done: no GPU lock, no `/tmp/m1-gpu.lock` open, no 375-wide surface, no
  `1x896`, no SET write, no module unload, no reboot, no retry, no T6001 host.
  `{512,1,1}` selects no channel-3 scratch slot, so this says nothing about the
  select arena path and vice versa.
- Worker `762dd1de868b52e4d6ffa30e18d842d9341db463b06aef2355f18fb1eb248929` and
  libane `1ab9d95debcc8b5fee3b6653dfce0b50412bc7efef43c2d2167dc83ce270ca49`, hashed
  on the device immediately before each exec; the same pair as the tiny-sigmoid and
  silu-512 receipts.
- Host: `omp-studio-local`, python 3.11.2, numpy 1.26.4. No file under
  `ane-linux-experiments` was edited or executed.
- Model: `anthropic/claude-fable-5-1`, no routing fallback observed.

## Sources

| file | sha256 |
| --- | --- |
| capture `tdt_tensors.json` (`20260913T105550Z-librispeech-tdt-tensors/ane`) | `6eaf2bfe801584af570cd66f42e8dc17efd67776fb181f691585d2bcab590e0a` |
| capture `receipt.json` | `5ee6328fa622a9440ecd1070ef02de9705294d4e73d4a989c88aeb4885175985` |
| `receipts/2026-09-14-decoder-activations/probe.py` (loader) | `617cf0e291cd980bea25586933c45d66d2d80b5317f4ccfc2605b82934e4cb0d` |
| `receipts/2026-09-14-decoder-activations/result.json` (pins) | `89baf47e12c3460de30c053f2ef9e65330376f4759a508b252feef2080bef0b1` |
| `receipts/2026-09-14-decoder-activation-stages/stages.py` (solver) | `f89d0184b04f828821812c1b7eb21d0d480b49cb42aa4e252e61042fd655b01a` |
| `mil-hwxc` binary, `build/mil-hwxc` | `d86bebb25b2b5023b3071b612654c42c79a1b93f51c471e07cfc1c1df82520c2` |
| bundle adapter `h13_package_to_bundle.py` | `83934107188b77c5e5c80d0ca542f96805d5ea14d7f648b26854a30375c0a4b2` |

Decoder package revision `b650695c2322ee5281dff48d7345b2f3a58ff018`, overlay from
worktree `CoremlTdtStream` at `1f8804cd087e60b7ab34a2a730510977f5aca0c3`.

Compiler: mil-hwx-compiler `feature/h13-concat`, HEAD `417554c3e22de9124623765717acee25e9860f19`.
The binary was built 10:47 CDT, between `83d486b` and `7ab3eb5`; every commit after
`37fb29e` touches select or receipts, not the unary path. What matters is checked
directly: the 128-byte table block at offset 4736 of each emitted `program-0.anec`
hashes to the oracle prefix the stages receipt decoded, `73f5680a…` for `sigmoid`
and `f4a38468…` for `tanh`, so the device ran Apple's own tables.

Programs: `sigmoid(x)` and `tanh(x)` on `tensor<fp16, [1, 512, 1, 1]>`, one
`h13-oracle-parity` program each, 1 task descriptor, 2048-byte LUT constant at
offset 640, schema-4 wrapped. Bundle names `schema4-sigmoid-512` (graph
`4a62eede…`) and `schema4-tanh-512` (graph `de23b14c…`); the graph hash is the
sha256 of the `.mil` source, which is in the artifact directory.

Everything under `receipts/2026-09-14-decoder-activations-on-ane/` is hashed in its
`SHA256SUMS`: both bundles, the six input and six output `.bin` files exactly as sent
to and read back from the device, `layout.json`, `build_bundles.sh`, `run_remote.sh`,
`analyze.py`, `result.json`.

## Command

```text
# host: compile + wrap (receipt dir has the script)
receipts/2026-09-14-decoder-activations-on-ane/build_bundles.sh
# jwm1: six one-shots
bash /var/tmp/jwm1-decoder-activations/run_remote.sh
# host: score the device outputs
python3 receipts/2026-09-14-decoder-activations-on-ane/analyze.py \
  --capture-dir ~/.cache/mlx-omarchy/parakeet-reference/captures/\
b650695c-75aec2a/20260913T105550Z-librispeech-tdt-tensors/ane \
  --package ~/.cache/mlx-omarchy/parakeet-reference/mweinbach1/\
parakeet-tdt-0.6b-v3-coreml/b650695c2322ee5281dff48d7345b2f3a58ff018/decoder.mlpackage \
  --overlay <CoremlTdtStream>/overlay \
  --output receipts/2026-09-14-decoder-activations-on-ane/result.json
```

`analyze.py` wall 1.6 s; `result.json` is byte-identical across two runs.

## The fixture

Each activation got 1536 fp16 lanes across three 512-lane submits (`layout.json`):

| lanes | content |
| --- | --- |
| 0–1279 | the decoder's own arguments for transition 0: `sigmoid` gets `b_i` then `b_o` of the pinned layer-0 bias; `tanh` gets `b_c` then the captured `next_cell` |
| 1280–1312 | the 33 table knots, `-8 + 0.5k` and `0.125k` |
| 1313–1344 | the 32 segment midpoints |
| 1345–1350 | the header clamps and one fp16 step either side: 8.3203125 and -9.9375; ±4.0 |
| 1351–1535 | a sweep on a fine fp16 grid through the pin range: `k/128` on [-0.71875, 0.71875] and `k/256` on [-0.359375, 0.359375] |

The script asserts the argument lanes are the decoder arguments bit for bit, and
that the device returned one value per distinct argument (1396 and 1424 distinct
arguments, no conflicts, no non-finite output).

## Compute unit

`environment.json` records `"compute_units": "ane"`. `receipt.json` says, in its own
words: "The environment records requested compute_units=ane; this receipt does not
claim every CoreML operation executed exclusively on ANE." There is no
`MLComputePlan` output anywhere in the capture, in the reference lock, or in any
receipt (searched). So the manifest cannot say whether the `lstm` op ran on the ANE.

That question did not need the manifest. The hypothesis was that the native
activations are the H13 LUT plus its interpolator. The H13 LUT plus its interpolator
is now measured, on the exact arguments, and it is not the native function.

## Device versus native

| measurement | device H13 LUT | host table model, stages receipt | fp16 chain, best known |
| --- | --- | --- | --- |
| `sigmoid` pins reproduced | 0 of 11 | 0 of 11 | 11 of 11 |
| `tanh` pins reproduced | 1 of 8 | 1 of 8 | 7 of 8 (correctly rounded) |
| `next_cell` lanes, device pair | 19 of 640 | | 500 |
| `next_hidden` lanes, device pair | 31 of 640 | | 486 |
| `sigmoid` refuted on `tanh` arguments | 171 of 1198 | 172 of 1198 | 15 |
| `tanh` refuted on `sigmoid` arguments | 194 of 1172 | 196 of 1172 | 49 (correctly rounded) |
| device `tanh` equal to the 949 conditional native `tanh` values | 59 of 949 | | 703 (correctly rounded) |

The refutation column is the partner-free instrument from the stages receipt: with
the device `sigmoid` fixed, 171 `tanh` arguments have no fp16 value at all that
multiplies and rounds to the captured lane, and vice versa. It assumes only that the
partner is a deterministic fp16 function and the lane is one fp16 multiply.

Mixing does not rescue it. Device `sigmoid` with correctly rounded `tanh`: 54 and 78
lanes. Chain `sigmoid` with device `tanh`: 78 and 32.

The reason is the size of the interpolation error on these arguments. Against
correct rounding over the 1280 decoder arguments the device `sigmoid` is bit-exact
on 116 and off by up to 25 fp16 ulps; the device `tanh` is bit-exact on 99 and off by
up to 11. The stages receipt measured the native functions within about two ulps of
correct rounding everywhere. A 0.5-wide `sigmoid` knot spacing cannot produce that,
and the device confirms it does not.

The one `tanh` pin the device reproduces (x = 0.6220703125) is the same coincidence
the host model reported. The host table model and the device agree on 1197 of 1280
`sigmoid` and 1194 of 1280 `tanh` decoder arguments; the 83 and 86 differences are
one-ulp rounding differences inside the interpolator, which is what the next section
identifies, and they move the refutation counts by one and two.

## Where this leaves the hypothesis

Two readings survive, and they are the same as the stages receipt's:

1. The macOS decoder's `lstm` did not execute on the ANE. Core ML placed it
   elsewhere despite `compute_units=ane`, and the capture cannot show placement.
2. It did execute on the ANE, and the `lstm` lowering evaluates its gates through a
   different path than the standalone unary table: the same compiler emits the
   table for an elementwise `sigmoid` op and something else inside a recurrent op.

This run does not separate them; nothing in the capture can. It rules out the third
reading the assignment was built on, that the unary LUT plus an unknown interpolator
rule is the native function, because the interpolator rule is no longer unknown and
the LUT is measured on the exact arguments.

Named stop for the decoder hole: `parakeet.tdt.decoder-lstm.fp16-elementwise-activation-vs-native-coreml`
stays open. The two inputs that close it are unchanged: an authenticated
single-operator capture of fp16 `sigmoid` and `tanh` over an argument sweep on the
same macOS stack, or the compiled compute plan naming the `lstm` placement and
lowering. Running the LSTM activations on the Linux ANE is not a path to bit
exactness; it reproduces 50 of 1280 lanes.

## The interpolator, identified

The device samples do support a rule, and it reproduces every interior lane of both
tables.

Let `a0` be the knot nearer `x = 0` in the interval containing `x`, `a1` the other
knot, and `s ∈ [0, 1)` the fraction of the interval from `a0` outward. Then

```text
p = fp16_round(s * (a1 - a0))       ties away from zero
y = fp16_round(a0 + p)              ties away from zero
```

For `tanh` the table is indexed on `|x|` and the sign is applied after; `a0` is the
lower knot. For `sigmoid` the table spans the whole line, and "nearer zero" means
the lower knot for `x ≥ 0` and the upper knot for `x < 0`; equivalently, the knot
whose value is nearer 0.5. The farthest knot of the `tanh` table, `x = 4.0`, is
evaluated as `s = 1` of the final interval rather than as its own entry, which is
why the device returns 1.0 there and not the table word 0.99951171875; the rule
treats `x = -8.0` the same way and lands on the table word, one ulp below the device.

| | `sigmoid` | `tanh` |
| --- | --- | --- |
| rule matches, all lanes | 1533 of 1536 | 1536 of 1536 |
| decoder arguments | 1280 of 1280 | 1280 of 1280 |
| knots | 32 of 33 | 33 of 33 |
| midpoints | 32 of 32 | 32 of 32 |
| clamp probes | 4 of 6 | 6 of 6 |
| sweep | 185 of 185 | 185 of 185 |

The alternatives it was chosen against, scored on the 1528 interior lanes of each:

| candidate | `sigmoid` | of which `x < 0` | `tanh` |
| --- | --- | --- | --- |
| exact linear interpolation, one nearest-even rounding | 1457 | 510 of 549 | 1420 |
| product rounded nearest-even, sum rounded nearest-even | 1425 | 488 of 549 | 1417 |
| product and sum rounded ties-away, lower-knot anchor | 1476 | 497 of 549 | **1528** |
| product and sum rounded ties-away, upper-knot anchor | 1497 | **549 of 549** | 1311 |

Two facts carry the identification. First, ties-away rounding on the sum is
distinguishable from nearest-even on these samples: `a0` and `p` sit on different
fp16 grids, so the arithmetic lands on an exact half-ulp often (81 product ties and
152 sum ties on the `sigmoid` interior lanes, 81 and 215 on `tanh`), and the device
resolves every one of them away from zero.
Second, the anchor is not "the lower knot": on the negative half of the `sigmoid`
table the lower-knot form misses 52 lanes with a signature that is small-`s` lanes
rounded up and large-`s` lanes rounded down, and anchoring at the knot nearer zero
removes all 52 while leaving the positive half at 977 of 977. That is the behaviour
of a magnitude-indexed interpolator, the same organisation the `tanh` table declares
in its header with a clamp low of 0.0. The `sigmoid` table is stored over the full
line, but the hardware walks it outward from the centre.

This also explains two facts earlier receipts recorded as unexplained. The
tiny-sigmoid receipt's "midpoints only 27 of 32 equal to `fp16(avg)`" is the
ties-away sum: the average of two knots is a tie exactly when their sum has an odd
low bit, 13 of the 32 `sigmoid` midpoints are ties, and the five that differ are
exactly the ties where nearest-even lands toward zero (`tanh`: 15 ties, 9 differ,
same reason). The `tanh` far knot returning 1.0 where the table word is
0.99951171875 is the `s = 1` evaluation of the final interval, `raz16(a0 + raz16(a1 - a0))`,
and the rule reproduces it. The `sigmoid` far knot at `x = -8` is not reproduced and
stays in the exception table below.

Exceptions, all three on `sigmoid`, none inside the argument range of any decoder
lane:

| x | device | rule | reading |
| --- | --- | --- | --- |
| -8.0 | 0.000335693359375 | 0.00033545494 (`table[0]`) | one fp16 ulp above the rule's `s = 1` evaluation; `tanh` at 4.0 matches the rule exactly, so the far-negative `sigmoid` edge has one more rounding this sample set cannot place |
| 8.3125 | 1.0 | 0.99951171875 | between the last knot at 8.0 and the header clamp at 8.3203125 the device already emits the clamp's "above" value |
| -9.9296875 | 1.9073486328125e-06 | negative extrapolation | between the header clamp at -9.9375 and the first knot at -8.0 the device emits a subnormal, not the "below" word and not an extrapolation |

So the header clamps are exact where probed (`8.3203125 → 1.0`, `-9.9375 → 0.0`,
`±4.0 → ±1.0`, and one step outside each), and the two 1.9-wide and 0.32-wide gaps
between the `sigmoid` knot span and its clamps hold a behaviour with one sample each.
Nothing the decoder evaluates lives there.

## What changes

Nothing in the shipped path. For mil-hwx-compiler, `docs/ane/numerics.md`'s open
question about the interpolator now has a measured answer above; that doc is not
edited here because this receipt lives in mlx-omarchy and the compiler slice is
owned elsewhere. `analyze.py` carries `h13_lut()` as the runnable form of the rule
and regenerates every count in this receipt from the files in its directory plus the
pinned capture and package.
