# AGX wait-batching driver patch — make it fire, then qualify (2026-09-12)

Mesa fork `github.com/joshuaswarren/mesa`, branch `hk/agx-wait-batching`.
Baseline `6f6afc89684` (= installed package `mesa-honeykrisp-omarchy
26.3.0.devel.hk6f6afc8-1`). Host placeholder `jwm1` (Apple M1 G13G B1,
aarch64, Omarchy kernel `7.1.6-1-1-ARCH`), all numbers on that host, all
windows under `/tmp/m1-gpu.lock`.

## Verdict

**The pass fires and the ISA moved.** Patched driver @ `30ffef135e0`,
shipped qmm coopmat kernel (`shaders/qmm_coopmat.comp`, 64-lane shader),
final packed ISA: **749 → 699 instructions (−50)**, **wait 32 → 19**,
**load/lload with a wait within 2 instructions 15 → 6**. f16 output digest
`5179630cd4a7c3f9` **held** (bit-identical to the stock pin committed in
`receipts/2026-09-12-agx-qmm-codegen`).

Mechanism, honest version: waits move from block exits to the first true
hazard. The 4 x-load guards still serialize internally — register
allocation reuses each guarded load's destination register for the else
arm's default value, so a WAW wait is forced inside every guard. The win
on this kernel is −50 issued instructions and −13 waits, **not** the full
5-latency overlap the original drain theory hoped for. This corroborates
the de-divergence finding: removing the exec-mask block-exit drains alone
does not recover all the staged-load latency.

## Commit chain (all on `hk/agx-wait-batching`)

| commit | what | outcome |
|---|---|---|
| `84fcd220de1` | eager frame carry (inherited) | inert: merge resolved via `order[]` before visitation, every frame failed |
| `8f3169a4342` | lazy merge resolution at else-arrival | committed unbuilt when inherited |
| `c3cebc011bd` | + firing instrumentation, wait_mask/fold fixes | frames fire (75 ACCEPT) but 25 FOLDs: fallthrough shapes never resolve; also unsafe 4-slot assignment |
| `30ffef135e0` | conservative rewrite: 2 slots, pop_exec-lazy close, sum-carry | **fires fully (75/75/75), ISA moved, digest held** |
| `f11106ddfb0` | host capacity/hazard gtests | this commit |

## The three root causes found on the way

1. **Eager merge resolution** (the known one): `order[]` lookup at frame
   push, before the then arm's last block was visited. But the deeper
   shape problem: AGX if/else guards are *fallthrough* shapes — the then
   arm falls into the else block (stock ISA `load @0x1fc → else @0x206`),
   so there is often **no jump to resolve a merge from**. Fix in
   `30ffef135e0`: regions close lazily when a block is entered whose
   predecessor ends with `pop_exec` — `emit_if` (agx_compile.c) appends
   pop_exec after the last else-arm block of *every* region, so one rule
   covers all shapes.
2. **`wait_mask` hoisted to block scope** (`84fcd220de1`): once a slot's
   hazard bit set, every later instruction in the block re-drained it
   (the 216 → 1966 wait explosion across shaders). Restored
   per-instruction scoping.
3. **4-slot assignment aliased the 1-bit IR field**:
   `agx_instr::scoreboard` is `unsigned scoreboard : 1` — "Scoreboard
   index, 0 or 1" (agx_compiler.h) — so slots 2/3 truncated to 0/1 at
   the IR level. Pack formats agree (load/store pack 2 bits @30-31,
   texture sample packs **1 bit** @63, agx_pack.c). `AGX_NUM_SLOTS` back
   to 2, pinned by `_Static_assert`.

## Safety model (no unproven hardware assumptions)

- Scoreboard queues: **no per-lane claim is made anywhere**. Carried
  state unions at every else arrival (`entry | then_exit`), pending
  counts **sum** (not max), and any slot driven past `AGX_MAX_PENDING`
  is force-drained at the merge. Conservative under either per-lane or
  shared-FIFO accounting.
- RA runs before this pass, so a physical register rewritten by the else
  arm must wait out the then arm's in-flight writeback — which is why
  the else arm sees the then arm's carried pending state.
- Pre-issuance bound: the upstream per-block check (count ≥ 8 ⇒ wait
  before the async instruction) is retained in both arms, so a ninth
  tracked message on a slot can never issue.
- Unassigned slots are safe by construction: `agx_alloc_instr`
  zero-allocates, so a stale/unassigned `scoreboard` is 0 (valid);
  asserted zero-invariant rather than guarded.
- Real branches, loop back-edges and barriers still drain as upstream.
- f16 digest of the target kernel: held (see below).

## Firing proof (per-site log, committed)

Instrumentation (temporary, unconditional): pass-entry line, per-site
ACCEPT/reject-with-reason, else-arrival, close, carry, drain,
overflow-drain. Run on the shipped qmm coopmat kernel dispatch on the
patched driver @ `30ffef135e0`:

- all shaders: 469 AGXWAITS lines, 75 ACCEPT, 75 else-arrivals,
  75 closes, **0 folds**, 4 overflow-drains (conservative backstop)
- coopmat kernel section: `logs/waits-firing-coopmat-kernel.log`
- full dump @ 30ffef: `logs/full-dump-patched-30ffef.log.gz`

## ISA diff (committed)

`isa/kernel-final-isa.diff` (full text), `isa/kernel-final-isa-{stock,
patched}.txt`. Stock extraction: final packed disassembly of the 64-lane
kernel, identical extraction applied to both dumps (stock reference:
`receipts/2026-09-12-agx-qmm-codegen/dumps/coopmat-kernel-final-isa.txt.gz`).

Example, first x-load guard — stock:

```
1fc: load r39, du28, r39, i32, x, a
204: wait a                    <- drain at exec-mask block exit
206: else 0.0, 0.0, 1, feq
20c: ldimm r39, 0
```

patched @ 30ffef:

```
1fc: load r39, du28, r39, i32, x, a
204: else 0.0, 0.0, 1, feq
20a: wait a                    <- deferred to the WAW (RA reuses r39)
20c: ldimm r39, 0
```

## Hardware-execution disclosure (honest record)

The 13:09Z 2026-09-12 run on the modified driver @ `30ffef135e0` was a
**real dispatch**, not compile-only. It ran inside the phase's single
flock on `/tmp/m1-gpu.lock` (enclosing pattern in `window.sh`). Probe
command:

```
VK_DRIVER_FILES=~/benchq/qmm-coop-bench/icd-waitbatch-patched.json \
AGX_SIMDMAT=1 MESA_SHADER_CACHE_DISABLE=true \
MESA_SHADER_CACHE_DIR=$(mktemp -d) AGX_MESA_DEBUG=shaders \
~/venv-qmmcoop/bin/python ~/benchq/qmm-coop-bench/qmm_dump_probe.py

(`qmm_dump_probe.py` compiles and executes the kernel: 5 warmup + 6
timed evals + a 7th eval for the digest.) Output: median 17.2814 ms,
1062.2 GFLOP/s, digest `5179630cd4a7c3f9`. Stock same-day reference
single-run: 17.5371 ms / 1046.7 GFLOP/s. **These are single unpaired
runs in different windows — reported for continuity, not as a
performance claim.** Paired qualification is gated on host checks and a
separate window.

An earlier dispatch run @ `c3cebc011bd` (12:47Z) likewise executed the
kernel on a then-unqualified driver; its log informed the fold-shape
diagnosis only and no numbers from it are used.

## Status

- [x] Pass firing (log committed)
- [x] ISA before/after diff committed
- [x] Safety model conservative (reviewer + root cleared for host tests)
- [x] Host capacity/hazard gtests (`f11106ddfb0`)
- [ ] Host gtests green on jwm1 (running)
- [ ] Paired interleaved A/B, same window, 3+ reps (kernel probe GFLOP/s
      at 1053x9728x896 + bench_matrix prefill tok/s legs), patched vs
      base driver
- [ ] Six canonical digest legs x3 on the patched driver against the
      committed pins (movement = rejection, not re-pin)
- [ ] Package/ICD provenance for both sides
