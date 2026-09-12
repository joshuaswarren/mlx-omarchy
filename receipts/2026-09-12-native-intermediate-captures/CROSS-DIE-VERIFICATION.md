# Cross-die verification — 2026-09-12

The full capture harness was re-run on **macstudio** (Apple M1 Ultra,
`applegpu_g13d`, macOS 26.6.2) against the same pins — mlx==0.32.2 /
mlx-lm==0.31.3 in a fresh venv, same HF snapshot revisions
(`a5339a41…`, `56d07e76…`), same expanded prompts, greedy temp 0 seed 0,
EOS suppressed, `MLX_DISABLE_COMPILE=1` — and every regenerated array
was sha256-compared against `BULK-MANIFEST.json` (16m1mbp / M1 Max
origin). Comparison tool: `harness/compare_cross_die.py`; raw result in
`cross_die_comparison.json` (second pass, RNG-aligned).

## Verdict: outcome 2, with a strong sub-claim

1. **All six canonical digests reproduce on M1 Ultra** — identical to
   the committed native oracle and to the M1 Max captures:
   Q4 `7fd25a86…` / `254d73fd…` / `7da83f06…`,
   BF16 `7fc0f968…` / `407b7624…` / `ff502900…`.
2. **Byte-identical across the two dies** (159 arrays): the entire
   decode-attention capture set that was sampled into the manifest
   (per-layer q/k/v/out at KV 30/31/32, 262/263, 1053/1054, both
   models), the qmm layer-0 arrays for the long and 1K-ctx legs
   (M=262 split_k=2 and M=1053 plain qmm — including 20 MB gate/up/down
   tensors), the `fast::exp` I/O tables (2 M inputs, zero differing
   bytes), and every synthetic dispatch capture once the RNG stream was
   aligned. Metal `simd_sum`-consuming scores therefore also match
   across dies on every decode capture.
3. **Die-dependent arrays (48 of the original 274; do not use as a
   die-agnostic rule-1 oracle without tagging)** — all confined to
   **q4/short layer-0 prefill**, the M=30 shapes where qmm dispatch
   picks `qmm_t_splitk` with split_k=14 (the heaviest split in the
   ladder):
   `qmm/{q_y, o_x, o_y, gate_x, gate_y, up_x, up_y, down_x, down_y}`
   plus the two downstream `sdpa/prefill0_L0_{q,out}` files and the
   manifest checksums. The divergence root is the q_proj split-K
   output (its input `q_x` and packed weights are byte-identical);
   q4/long (split_k=2) and every non-split shape are identical, so the
   suspected sensitivity is the many-partition f16 partial-sum path,
   not the MMA order (plain qmm at M=1053 is byte-identical across
   dies). Correlation stated as observation; mechanism not proven.
4. **Harness artifacts excluded from evidence** (48 → after
   alignment): 36 "different" dispatch files in the first pass were
   the probe's RNG stream being consumed differently when the dump
   step is skipped — fixed-input discipline requires running identical
   probe-step sequences, not just identical seeds. 80 dump-trail files
   differ between all runs because that instrument was broken at
   capture time (see KNOWN-DEFECTS.md items 1–2); they are invalid on
   both dies and carry no cross-die information.
5. Not regenerated on macstudio: the simd_sum probe files
   (`probe_simd_order.py` was not re-run there); 23 manifest entries.

## Storage

Regenerated arrays live at `macstudio:~/src/native-captures-macstudio`
and `jwm1:~/src/native-captures-macstudio` (309 MB). With this run,
the capture set exists independently on two Apple dies and two Linux
hosts, and is regenerable from the committed harness + pins on any
macOS+MLX machine.
