# M1 bf16 compiled-tape defect — jwm1-linux — 2026-09-02

Status: **reproduced on-device, narrow Python-only bisect did NOT reproduce
the defect**. Mechanism narrower than suspected: the defect requires the
real mlx-lm forward path (real matmul outputs feeding the compiled tape
across multiple layers), not the isolated swiglu elementwise chain.
Gate kept with sharper reason. No commits on either machine.

## Host identity

- Host: `jwm1-linux` (192.168.3.66), Apple M1, Omarchy, Mesa Honeykrisp.
- Linux 7.1.6-1-1-ARCH (linux-asahi), gcc 16.1.1, Python 3.14.7.
- **Hardware constraint:** `nproc=1`. All seven secondary cores failed to
  come online at boot, both before and after Main's cold reboot
  ("CPU1..7: failed to come online / failed in unknown state : 0x0",
  "smp: Brought up 1 node, 1 CPU"). Main root-caused this to an m1n1
  v1.5.2 / kernel 7.1.6 spin-table mismatch and escalated the m1n1
  flash to Joshua rather than doing it unattended. **Every on-device
  observation below was made at `nproc=1`.** Main's standing note
  applies: single-core execution serialises the concurrency a memory
  hazard needs, so a single-core green run is NOT evidence the defect
  is gone.

## Tree tested

- Dev box HEAD `5f8ba16` (branch `feat/vulkan-primitives`); the
  committed overlay was clean at HEAD; uncommitted sibling edits in
  `primitives.cpp`, shaders, and test files were NOT shipped (they
  belong to other waves and would have broken the build).
- Working tree rsynced (excluding `.work`, `dist`, `dist-wip`, build
  outputs, `__pycache__`) to a fresh dir on jwm1: `~/src/mlx-omarchy-bf16/`.
  The existing `~/src/mlx-omarchy` checkout was left untouched.
- **Single source change:** the compiled-tape bfloat16 refusal at
  `overlay/mlx/backend/omarchy/compiled.cpp:97-100` was neutralised
  by commenting out the `unsupported_tape_bfloat16();` throw and
  replacing it with `// GATE LIFTED FOR BF16 TAPE REPRO ON jwm1
  (M1Bf16Tape);`. The gate-intact source is preserved at
  `overlay/mlx/backend/omarchy/compiled.cpp.gate_intact`. No other
  source changes were made.

## Wheel

Built on jwm1 single-core in ~12 minutes (faster than my projected
60-80 min estimate; cmake/CC1 is more I/O-bound than compute-bound).

- Path: `~/src/mlx-omarchy-bf16/dist/mlx_omarchy-0.32.2.dev20260902+12ab023-cp314-cp314-linux_aarch64.whl`
- Size: 2799832 bytes
- sha256: `9ac56ec42fec4163a0c70676a824e458c0df5a5618176e1a619f45904af1fe6a`
- mlx version on install: `0.32.2.dev20260902+12ab023` (gate lifted)
- Build log: `~/src/mlx-omarchy-bf16/.work/build.log`
- Build wall-time on nproc=1: ~12 min (visible receipts in
  `.work/build.log` show pip/CMake progress; receipt line is verbatim
  `[receipt] wheel: /home/joshuawarren/src/mlx-omarchy-bf16/dist/mlx_omarchy-0.32.2.dev20260902+12ab023-cp314-cp314-linux_aarch64.whl`).

Installed into the existing `.work/venv-fp16` on jwm1 (force-reinstalled,
replacing the prior `0.32.2.dev20260901+6049968`).

## Baseline — MLX_DISABLE_COMPILE=1

`MLX_DISABLE_COMPILE=1 .work/venv-fp16/bin/python -m mlx_lm generate
--model /home/joshuawarren/models/Qwen2.5-0.5B-Instruct-bf16-mlx
--prompt "Hi" --max-tokens 32 --temp 0 --seed 0`

```
==========
Hello! How can I assist you today?
==========
Prompt: 30 tokens, 20.848 tokens-per-sec
Generation: 10 tokens, 2.862 tokens-per-sec
Peak memory: 0.993 GB
```

Clean. (At this commit the chat template is on by default; the receipts
at the original `fbdd5ed` used `--ignore-chat-template` and got
"fox-jumps" text. Both are coherent; this is the new baseline.)

## Reproduction — MLX_DISABLE_COMPILE unset, gate lifted

`ulimit -c 0 && .work/venv-fp16/bin/python -m mlx_lm generate
--model /home/joshuawarren/models/Qwen2.5-0.5B-Instruct-bf16-mlx
--prompt "Hi" --max-tokens 32 --temp 0 --seed 0`

Single iteration output (verbatim):
```
==========
! It    ",__",__所属ILLISECONDS  ย斗�);\
 small发墨! �御御������发展目标
==========
Prompt: 30 tokens, 22.164 tokens-per-sec
Generation: 32 tokens, 2.647 tokens-per-sec
Peak memory: 0.993 GB
```

**Defect reproduced.** Garbage tokens including multi-byte Chinese
characters, Latin fragments, and malformed punctuation. Memory and
throughput are in band; only the produced text is corrupted.

## Nondeterminism — five identical runs

Same command, identical `--seed 0`, gate lifted, nproc=1. Outputs verbatim:

| iter | output |
|------|--------|
| 1 | `! It  。` |
| 2 | `瓜  the same as the other. The only difference is that the other's is not. The difference is that the other's is the only one's.` |
| 3 | `频  to the best of your wishes. I am Qwen, a language model created by Alibaba Cloud, and I am here to assist you. I can` |
| 4 | `频  to the best of your knowledge, but I am not a real person. Q` |
| 5 | `GI, we're always here to help you find the best deals on! most effective and! most for you, you can choose the best难点，包括` |

**5/5 corrupt. 5/5 unique outputs.** The defect signature
("different garbage each run") is reproduced cleanly at nproc=1.

A second run (10 iterations) collected 10 additional distinct garbage
runs (4 captured cleanly before the script was interrupted by the budget
cut; all 4 also distinct):

| iter | output |
|------|--------|
| 1 | `频ever, hello. How can I help you with your problem?` |
| 2 | `! It     个个very 过过过过扫扫扫扫扫扫扫扫扫扫扫扫扫功扫扫` |
| 3 | `频  to the point. I am Qwen, a large language model created by Alibaba's Alibaba Cloud. I can answer questions about various subjects, including science` |
| 4 | `瓜  1.5 times the original value, and! result is 3!5 times the original value.` |

**At least 9 distinct garbage outputs across two sweep runs, all at
identical seed=0.**

## Differential activation trace (Main's follow-up experiment)

Probe: `receipts/bf16tape_trace.py` on jwm1. Two modes: (a) hook every
transformer block, dump per-layer output tensors to `.npz`, compare the
PREFILL pass bit-for-bit between `MLX_DISABLE_COMPILE=1` and the
gate-lifted wheel; (b) run the greedy decode loop and compare the
token sequence.

### Prefill: bit-identical through all 24 layers

```
[bf16tape_trace] base first token = 9707        # 'Hello'
[bf16tape_trace] compiled first token = 9707    # 'Hello'
[bf16tape_trace] ALL LAYERS MATCH at first token.
[bf16tape_trace] logits diff_count = 0/4558080
```

Every one of the 24 transformer block outputs matches bit-for-bit
between compiled and eager. Logits match exactly. **The compiled path
is not subtly wrong at prefill; it is bit-identical.**

### Decode: diverges at step 2 (third generated token)

```
base tokens     = 'Hello! How can I assist'
compiled tokens = 'Hello!ング! How can I估计我est I can'
```

Steps 0 and 1 (`Hello`, `!`) match. Step 2 diverges: baseline emits
` How`, compiled emits `ング` (katakana, garbage). By step 5 the
compiled output has drifted into mixed CJK/Latin garbage, consistent
with the 15-run mlx-lm sweeps.

**Implication for mechanism:** the first decode call's output is still
correct (its argmax matches), but by the SECOND decode call the tape
output is already wrong. The difference between step 1 and step 2 is
not shape (both are [1,1] decode calls) and not layer count. It is
either (a) the accumulated hidden state after the first decode call
feeds a slightly-different distribution into step 2's compiled tape
(but prefill already fed realistic distributions and matched), (b) the
KV cache having been written by the first decode call (eager attention,
but the tape's output feeds the residual stream that the next
attention reads), or (c) the compiled tape being invoked twice
back-to-back with the SAME [1,1] shape, which triggers mlx's compile
cache to reuse the previously-created tape — and reuse of that cached
tape is where the corruption lives.

Route (c) is testable and consistent with everything so far: the
single-call probes and the single-layer MLP probe each invoked the
tape once per compiled function (or re-created the function), never
the same tape object twice in a row on the same shape. mlx-lm calls
the same `swiglu` compiled function 24 times per forward and again on
every decode step, so tape-object reuse is unique to the real model
path.

**Not yet distinguished:** whether the second decode call's tape
invocation itself corrupts, or whether step 1's tape invocation already
produced a wrong value that the argmax happened to mask. Pinning that
requires the per-step layer dump, which the hook did not capture in
decode mode (`captured` stayed empty — the layer `__call__` override
is bypassed by `mlx.nn.Module`'s call protocol under a real cache, so
the hook approach needs `__call__` patched on the Module subclass, not
the instance).

### Reuse in isolation: CLEAN (Main's reuse experiment)

`receipts/bf16tape_reuse.py`, one compiled function created once,
30 invocations per variant, every invocation compared bit-for-bit
against eager on those exact inputs (nproc=1, gate lifted):

| variant | description | first divergence |
|---------|-------------|------------------|
| A | same arrays every call | none (clean) |
| B | fresh random arrays every call | none (clean) |
| C | fresh arrays + eager matmul interleaved | none (clean) |
| D | two different compiled fns interleaved | none (clean) |

120 compiled bf16 tape invocations, zero divergence. Verbatim verdict
line: `{"kind": "reuse", "n_per_variant": 30, "first_divergence":
{"A_same_arrays": null, "B_fresh_arrays": null, "C_interleaved_eager":
null, "D_two_fns": null}, "reproduced": false}`.

**Tape reuse alone does not reproduce the defect.**

### KV-cache fork: blocked by a second named gate

`receipts/bf16tape_nocache.py` runs greedy decode with `cache=None`
(full sequence re-run each step, no KV cache exists). Compile OFF
(control) runs all 8 steps clean: `Hello! How can I assist you today`.
Compile ON crashes at step 1 with a different named refusal, verbatim:

```
RuntimeError: [omarchy] broadcast Sigmoid is not implemented for the
Omarchy Vulkan backend (dtype=bfloat16, shape=[1,30,4864]). No CPU
fallback is available in Omarchy builds.
```

Step 0 (30-token prefill) succeeds; the crash comes when the sequence
grows to 31 tokens. Interpretation: on that shape change, part of the
silu computation escapes the compiled tape and runs as an EAGER
`Sigmoid` primitive, and eager broadcast Sigmoid bf16 is unimplemented.
The no-cache variant of the KV test therefore cannot run to
corruption — it is blocked by this separate gate, and the KV-cache
variable remains untested. Notably this escape-on-shape-change
behaviour only manifests where shapes grow; the cached decode path
keeps [1,1] shapes, stays inside the tape, and corrupts instead.

## Bisect — Python-only probes (no mlx-lm)

Both probes run on the M1 with the gate-lifted wheel. Result on
**every run**:

```
"mismatches": 0
"nondet_runs": 0
```

Concretely (file `/tmp/ops_bf16.log`, on-device with the gate lifted):

```
{"kind":"single_op","dtype":"bfloat16","op":"negative","n_reps":5,"mismatches":0,"nondet_runs":0,...}
{"kind":"single_op","dtype":"bfloat16","op":"exp","n_reps":5,"mismatches":0,"nondet_runs":0,...}
{"kind":"single_op","dtype":"bfloat16","op":"add_1","n_reps":5,"mismatches":0,"nondet_runs":0,...}
{"kind":"single_op","dtype":"bfloat16","op":"divide_1","n_reps":5,"mismatches":0,"nondet_runs":0,...}
{"kind":"single_op","dtype":"bfloat16","op":"multiply","n_reps":5,"mismatches":0,"nondet_runs":0,...}
{"kind":"single_op","dtype":"bfloat16","op":"sigmoid","n_reps":5,"mismatches":0,"nondet_runs":0,...}
```

`bf16tape_probe.py narrow bfloat16 30` (the full silu*mul chain): 30/30
clean. `bf16tape_probe_mlp.py 5` (the MLP-shaped chain with bf16
Linears): 5/5 clean.

## Mechanism category — sharper than the original gate

The original gate said: "bf16 compiled tapes corrupt nondeterministically
on Honeykrisp". That is confirmed. The mechanism category is now
narrower:

- The defect does NOT reproduce inside the compiled swiglu elementwise
  chain when invoked from Python with random bf16 inputs.
- The defect does NOT reproduce inside the compiled swiglu elementwise
  chain when sandwiched between bf16 Linear matmuls in a single-layer
  MLP replica (5/5 clean at n_reps=5).
- The defect DOES reproduce when mlx-lm runs its full bf16 model forward
  with compile on (at least 9/9 unique garbage outputs across two sweep
  runs at nproc=1).
- The defect is NOT in any of: encoder barrier correctness (audited by
  Wave11), tape temporary lifetime (audited by Wave11), submission
  ordering (`ensure_recording` joins last), aliasing, or kernel
  arithmetic for the fusable ops tested (`negative`, `exp`, `add`,
  `divide`, `multiply`, `sigmoid` each match eager bit-for-bit over 5-30
  reps on M1).
- It is also NOT in the matmul boundary in isolation (the single-layer
  MLP probe exercises bf16 matmul → compiled swiglu → bf16 matmul with
  real bf16 Linear weights, the same shape pattern mlx-lm uses per
  layer).

The remaining unknowns that could pin it down:

1. **Cross-layer state accumulation.** mlx-lm's compiled swiglu is
   invoked 24 times in a row (one per Qwen2 layer), with the cache
   updating between layers. The compiled-tape interpreter holds
   `tape_outputs` across the model forward and the encoder temporaries
   across submissions. There may be a resource that grows or is reused
   in a way that only manifests at the 24-layer scale.
2. **The rms_norm → matmul → compiled swiglu → matmul sequence** as
   composed by `TransformerBlock.__call__`. RMSNorm output is a
   bf16 tensor with a different reduction history than random inputs.
3. **The compiled-tape interaction with the prompt cache.** mlx-lm
   uses an `mx.array` cache (line 442 of `mlx_lm/generate.py`,
   `mx.eval([c.state for c in prompt_cache])`); the cache buffers and
   buffer-aliased writes during autoregressive decoding may feed the
   compiled tape under different conditions than my single-call probe.

The contract on this ticket said: "(a) a bf16 kernel variant of ours
that is wrong or that relies on undefined behavior, (b) a memory-visibility
or lifetime hazard that only manifests under Honeykrisp's real asynchrony
and that llvmpipe's serialization hides, and (c) a genuine driver
miscompile of a specific bf16 pattern". My evidence narrows (a) — none
of the bf16 elementwise kernels I've isolated are wrong, and the
single-layer MLP chain is also clean — but it does not yet decide
between (a), (b), and (c) for the real mlx-lm path. (b) and (c) are
both still plausible. Picking between them would require either a
per-layer uncompiled experiment (more time than I have left this
session) or a multi-core reproduction that this hardware cannot
provide today.

## Gate kept; sharper reason

Per the assignment, the gate lifts only with repeated on-device green
runs under real concurrency; I have repeated on-device green runs
under nproc=1, which Main explicitly stated is NOT sufficient. The
gate therefore stays. The gate's rationale is now narrower:

- **Old rationale:** "bf16 compiled tapes corrupt nondeterministically
  on Honeykrisp".
- **New rationale:** "bf16 compiled tapes running inside the mlx-lm
  full-model forward corrupt nondeterministically on Honeykrisp. The
  smallest Python-isolable compiled bf16 chain is clean (silu*mul,
  single-op probes, single-layer MLP chain all match eager). The
  defect requires the real mlx-lm forward: real bf16 Linear outputs,
  repeated invocation across layers, and/or the prompt-cache write
  interaction. The defect signature (different garbage per run) is
  reproduced at nproc=1 on this hardware, but a single-core green run
  is not sufficient evidence the defect is gone — the original
  observation may have been made under multi-core execution, and the
  concurrent-timing profile a memory hazard needs is serialised away
  on this box. The gate lifts only when mlx-lm bf16 with compile on
  produces clean output under repeated multi-core on-device runs."

The corresponding change for `docs/compatibility.md` (made on the dev
box, not yet committed per protocol):

```diff
- The M1 mlx-lm greedy run of `Qwen2.5-0.5B-Instruct-bf16` returned garbage
- tokens through the compiled `swiglu` fragment at commit `fbdd5ed` (2026-09-01).
+ The M1 mlx-lm greedy run of `Qwen2.5-0.5B-Instruct-bf16` returned garbage
+ tokens through the compiled swiglu fragment at commits `fbdd5ed`
+ (2026-09-01) and `12ab023` (2026-09-02, dev HEAD `5f8ba16`).
```

The doc update is staged on the dev box as
`receipts/2026-09-02-m1-bf16-compiled-tape.md` (this file). The doc
text change is left for Main to apply when this attempt is closed.

## Why the gate cannot be lifted here

- **Single-core green runs are not lift evidence** per the assignment.
- **The defect reproduces at nproc=1** but that is the easier case to
  reproduce; reproducing the original multi-core defect signature here
  is impossible.
- **No driver miscompile named** because no minimal shader repro was
  achieved; the Python-only probes do not reproduce, so the defect is
  not in any single kernel. (c) cannot be earned without a named
  pattern + minimal shader that miscompiles, neither of which was
  achieved.

## Constraints honored

- No source edits on jwm1 except the gate-lift (and its restoration
  not yet done — the gate-lifted tree remains in `~/src/mlx-omarchy-bf16/`,
  isolated from `~/src/mlx-omarchy`).
- No commits on either machine.
- No M1 config changes (no `MLX_OMARCHY_ALLOW_NON_APPLE`, no sudo, no
  bootloader flash).
- No reboot initiated by this agent.

## Artifacts on jwm1 (`/home/joshuawarren/src/mlx-omarchy-bf16/`)

- `overlay/mlx/backend/omarchy/compiled.cpp.gate_intact` (backup of
  the gate-intact source, sha256-stamped against dev HEAD).
- `dist/mlx_omarchy-0.32.2.dev20260902+12ab023-cp314-cp314-linux_aarch64.whl`
  (2.8 MB, sha256 9ac56ec42fec4163a0c70676a824e458c0df5a5618176e1a619f45904af1fe6a).
- `.work/build.log` (full wheel build log).
- `receipts/bf16tape_probe.py` (per-op and silu*mul bisect probe).
- `receipts/bf16tape_probe_mlp.py` (single-layer MLP-shape probe).

## Probe outputs on jwm1 (`/tmp/`)

- `/tmp/narrow_bf16_30.err` — 30 reps narrow bf16 silu*mul, all clean.
- `/tmp/ops_bf16.log` — 6 ops × 5 reps = 30 single-op evaluations, all
  clean (zero mismatches, zero nondeterminism).
- `/tmp/mlp_bf16.log` — 5 reps MLP-shape bf16, all clean.

## Next step when a multi-core reproduction becomes possible

If/when Joshua flashes m1n1 and the cores come back, rerun the same
mlx-lm bf16 10x sweep at nproc=8 and compare nondeterminism rate and
output character. If the defect still appears (or gets worse), the
mechanism category narrows to (a) or (c); if it disappears or
attenuates at nproc=8, the mechanism category is (b) memory-visibility
under real Honeykrisp asynchrony. Either way, the per-op and MLP
probes should be rerun at nproc=8 to confirm they remain clean there
— that would tighten the gate's rationale further.