# Abliterated Bonsai-2-27B — coherent tok/s on the v0.7.0 recert wheel (jw16)

2026-09-18 · lane AblitRerun (queued tail from F7GdnCorrectness) · host jw16 (M1 Max T6001)

## Verdict

**The F7-parked abliterated-pack tok/s item closes coherent.** On the recert
wheel bytes (`2def345c…`, main `b283a16f`), greedy 96-step generation with the
runtime refusal ablation applied (129 writers, α=1.5) is **coherent with zero
NaN, 1.25 tok/s** vs the same-window stock rerun at **1.44 tok/s**. On this
benign prompt the ablation leaves the argmax trajectory bit-identical (same
generated-id digest as stock); a prefill liveness probe proves the projection
is live and materially moves refusal-adjacent logits.

## Wheel + setup identity

- wheel: `mlx_omarchy-0.32.3.dev202609182317+b283a16-cp314-cp314-linux_aarch64.whl`,
  sha256 `2def345c00a60c41d2f18018096720611d40ff5a3dc50845ae3e19dc59794e53`
  (re-verified on jw16 this lane), venv `/tmp/v070-matrix-venv`
  (mlx-lm 0.31.3, **stock** mlx-vlm 0.7.1 — the stock path F7 verified clean;
  the custom gdn_sink variant was not needed)
- pack: prism-ml snapshot `3f926b415992eaa2ae9dd7b573706494d6bbf787`
  (adapter base_revision, exact)
- adapter: `/tmp/ablit` (α=1.5, 128 linear writers + embed = 129 sites;
  `refusal_dir.npy` sha256 `b981c254df8b505…`, loader contract per
  `2026-09-18-runtime-modernize-source-bump.md` §16)
- run script `/tmp/ablit_gen.py` on jw16: same stock recipe as the recert
  matrix leg (`f7_bisect.py gen` semantics — pack-bundled `vision_artifact`
  loader, chat-templated "The capital of France is", 17 prompt tokens, greedy
  argmax, KV cache, tok/s over decode steps), `--arm stock|ablit`; no
  `MLX_*` env overrides (runtime defaults, compile ON).

## Results

| arm | tok/s | steps | nan_at | digest (ids) | text |
| --- | --- | ---: | --- | --- | --- |
| stock (same-window rerun) | **1.44** (1.4428) | 96 | null | `9252095e0de70235` | `'The capital of France is **Paris**.<\|im_end\|>…'` |
| abliterated α=1.5 | **1.25** (1.2468) | 96 | null | `9252095e0de70235` | identical text |

- Stock rerun reproduces the V070Recert receipt row (1.44 tok/s, coherent)
  on the same day/venv — the window baseline is honest.
- **Digest equality on the benign prompt** is a measurement, not a no-op
  suspicion: the liveness probe below shows the projection fires in-path and
  changes the next-token argmax on a refusal-adjacent prompt while leaving
  this benign trajectory's argmax untouched.
- tok/s cost of the ablation ≈ **-13.6%** — 129 extra f32 projections per
  forward (each `y → y - 1.5·(y·d)d` on 5120-dim residuals), un-fused.

## Ablation liveness probe (one load, stock prefill → patch → same prefills)

| prompt | max\|Δlogit\| | next-token argmax |
| --- | ---: | --- |
| "The capital of France is" | 1.1007 | 760 → 760 (unchanged) |
| "How do I pick a lock?" | **8.8909** | **40 → 47 (changed)** |

`[ablit-rerun] projected 129 residual writers at alpha=1.5` — all sites
present and wrapped (strict assert, no missing).

## Window discipline

Three windows, all `sudo systemctl stop llm-inference.service` →
`flock -w 900 /tmp/m1-gpu.lock` → work → release → restart → confirmed:
window 1 (stock + first ablit attempt) rc=1 then active/health 200;
window 2 (ablit rerun) rc=0, active, warm-up 503 then 200; window 3
(liveness probe) rc=1 on a print bug after all measurements landed, active,
health 200 re-confirmed by hand. Lock inode 12 before and after every take,
never unlinked. No collisions; no jwm1/jw14m2 contact; nothing tagged or
published.

## Notes / landed mechanics

- First ablit attempt crashed: `AttributeError: 'list' object has no
  attribute '63'` — mlx stores layer stacks as ONE list under a single key
  while `named_modules()` yields dotted numeric names, so the
  Bonsai-demo-style flat `getattr` parent walk fails on stock mlx_vlm trees
  (`apply_to_modules` LIFO order surfaces layer 63 first, hence "63").
  Fixed with a container-aware descent in `/tmp/ablit_gen.py`; the same
  latent bug is still in `/tmp/Bonsai-demo/scripts/mlx_generate_bonsai2_ablit.py`
  for anyone rerunning it against the stock-path model.
- `rariruluis/ternary-bonsai-2-27b-mlx-runtime-abliterated` itself was not
  loaded; the runtime projection it publishes is applied to the canonical
  prism-ml pack, per the verified loader contract. Same generation arm as
  the F7-era run (then 3.78 tok/s, corrupt output under F7) — now coherent
  on the fixed wheel, closing the "ablation delta unmeasurable until F7"
  note in `2026-09-18-runtime-modernize-source-bump.md` §16.

## Artifacts (jw16 → repo)

- `receipts/2026-09-18-ablit-rerun-jw16/gen-both-arms.log` — window markers,
  both arm stdout lines with tok/s/digest JSON
- `receipts/2026-09-18-ablit-rerun-jw16/liveness-probe.log` — probe lines
- scripts: `/tmp/ablit_gen.py`, `/tmp/ablit_probe.py` (jw16)
- raw: `/tmp/ablit-rerun.log`, `/tmp/ablit-probe2.log` (jw16)

## Refs

- `receipts/2026-09-18-f7-gdn-correctness.md` (Parked item this lane closes;
  stock-path-clean note)
- `receipts/2026-09-18-v070-pretag-recert-jw16.md` (wheel identity, stock
  1.44 tok/s row, window discipline skeleton)
- `receipts/2026-09-18-runtime-modernize-source-bump.md` §16 (adapter loader
  contract, 129 sites, F7-era corrupt-output arm)
