# Compiled fusion reduces model dispatches; gains remain small

On Apple M1 with Honeykrisp, development source `2f54fcb0738bc23e6b5987bcc6b341825759d3c4` combines the model's three-operation SwiGLU chains. Across 127 decode intervals, fusion cuts GPU dispatches per token from 585 to 537 and tape dispatches from 72 to 24. The ON profile records `FusedChainF16` with three instructions; OFF records separate elementwise operations.

Fusion remains opt-in through `MLX_OMARCHY_FUSED_CHAIN=1`. Five balanced pairs per workload show small gains on one pinned model, not broad-model qualification for a default change. Published v0.3.5 wheels do not contain this implementation.

## What changed

`fused_chain.{h,cpp}` and `shaders/fused_chain.comp` combine dependent float32/float16 tape operations into one dispatch, with at most eight instructions and three input buffers. Intermediate values round to their storage dtype after each instruction. Shared outputs, unsupported layouts, and unsupported dtypes keep the per-node path. The gate-off path constructs no fusion state.

The first implementation selected the fused kernel but did not combine model dispatches. Four identity broadcasts split the model's seven-node tape and inflated its use counts. The new prepass resolves identity aliases and counts their actual data uses. It uses upstream broadcast shape rules, including zero dimensions and invalid shapes, rather than dimension-wise maximums. Real broadcasts and shared outputs still form boundaries. Per-node submission diagnostics disable fusion.

The first M1 float16 tests had failed 89 assertions across two cases. Replacing the conversion round trip with `packHalf2x16` / `unpackHalf2x16` fixed those cases without weakening comparisons. The responsible driver/compiler transformation remains unknown.

## Fixed-source native checks

The normal `2f54fcb` build passed these M1 suites:

| Suite | Cases | Assertions |
|---|---:|---:|
| Fast operations | 23 | 29,867 |
| Copy offsets | 13 | 93 |
| Fused chains, OFF | 23 | 5,014 |
| Fused chains, ON | 23 | 5,014 |
| Compiled tapes, no overrides | 11 | 1,765 |
| Runtime | 26 | 6,252 |
| Compiled tapes, fusion OFF | 11 | 1,765 |
| Compiled tapes, fusion ON | 11 | 1,765 |

Doctest reports 26 passing runtime cases, but the no-qualifying-GPU refusal
case returns early because this machine has a qualifying GPU. That refusal
path was not exercised by this hardware run.

The build includes the CPU backend, as `scripts/build-wheel.sh` specifies for explicit CPU streams. These tests and model captures select the Apple M1 GPU; the touched GPU paths do not retry on CPU. The bf16 RoPE fix in this build has a separate [numerical and full-token receipt](2026-09-04-bf16-rope-layout-m1.json).

## Normal-wheel measurements

Five pairs alternate OFF/ON and ON/OFF order. Both sides use the same normal wheel, with profiling absent and `MLX_DISABLE_COMPILE` unset. Each leg uses temperature 0, seed 0, four warmup tokens, and EOS suppression. The Qwen2.5-0.5B 4-bit revision is `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`.

All 30 legs passed provenance, prompt-count, generated-count, and digest checks. Every one of the 15 paired comparisons has equal full token arrays, retained in the [machine-readable receipt](2026-09-04-fusion-identity-gate.json).

| Prompt / generated tokens | Median decode tok/s OFF | ON | Median paired ON/OFF ratio |
|---|---:|---:|---:|
| 30 / 32 | 31.4143 | 31.9638 | 1.0153 |
| 262 / 128 | 29.0487 | 29.3250 | 1.0097 |
| 1053 / 32 | 24.3493 | 24.5867 | 1.0103 |

Median prefill seconds changed from 0.377109 to 0.368751, 1.396401 to 1.371083, and 5.310921 to 5.223664, respectively. One long128 pair decoded more slowly with fusion ON. The roughly 1% decode gains are measured on Linux, not against native Metal.

Normal wheel SHA-256: `f093933b1a0634c0ae58280fd8984009925cc42b6974235731317c84faa3612e`. The separate same-source diagnostics wheel has SHA-256 `a93f5725dfb60ec0e3b6ea5c7fda3a961d20f92ae888c60536954ced6ef563e3`; it supplies dispatch evidence only, not the timing table.

Raw artifacts on the measurement host are in `~/benchq/logs/fusion-pairs-2f54fcb0/` and `~/benchq/logs/combined-gate-2f54fcb0/`. Local copies use the same directory names under `/tmp/`.

## Historical comparisons

The initial five pairs at `db10f538` set `MLX_DISABLE_COMPILE=0`. Upstream disables compilation whenever that variable exists, so those runs emitted no tape dispatches and were rejected as fusion evidence.

A corrected `db10f538` run with the variable absent still recorded 585 total and 72 tape dispatches per token on both sides. Its five-pair decode medians were 31.48/31.58, 29.04/29.19, and 24.32/24.37 tok/s. The recorded token digests matched; those matrix files did not retain full token arrays. That build did not establish model fusion or a speed gain. Its raw artifacts remain under `~/benchq/logs/performance-final-db10f53-20260904-batch4/`.
