# SDPA f16-scores rework: M1 A/B

2026-09-04, jwm1 (Apple M1, Honeykrisp 1.4.354). Before = main at 643a464
(v0.3.5 plus docs). After = sdpa-f16-scores at 1a29e26, rebased onto that
main so the diff is the rework alone. Both wheels built on the M1 from
detached checkouts, sha256 differ (a30ee42b vs e24a4808), fresh venv per
side, every run printed a provenance line with verified=match. Qwen2.5-0.5B,
pinned 64 tokens, EOS suppressed, decode over the last 63. bf16 eager
(compiled bf16 tapes are gated on this GPU).

## Equivalence on hardware

`scripts/sdpa_equivalence.py` on the after wheel: ALL PASS, 19 cases,
including the fully-masked row and the q,k~N(0,40) overflow boundary.

## Decode, tok/s

| dtype | before r1-r5 | median | after r1-r5 | median | delta |
|---|---|---|---|---|---|
| 4-bit | 12.69 12.65 12.53 12.52 12.56 | 12.56 | 12.82 12.86 12.85 12.82 12.66 | 12.82 | +2.1% |
| bf16 | 8.75 8.78 8.74 8.73 8.78 | 8.75 | 8.75 8.72 8.73 8.77 8.72 | 8.73 | -0.2% |

After legs ran 10:40-10:42, before legs 10:44-10:46; the box was idle
between them.

## Reading

The rework removes about 120 dispatches and 96 casts per token
(753 -> 633 on llvmpipe, receipts/2026-09-04-sdpa-f16-scores-rework.md)
and buys 2% on 4-bit decode and nothing on bf16. The removed work was
cheap: the remaining wall is kernel time, not dispatch count, which is
the same conclusion the GPU-busy breakdown reached from the other
direction. Merged because it is correct on hardware, not slower, and
simpler; not because it is a performance win.

Logs: jwm1:~/benchq/logs/sdpa-ab/ (per-run) and ~/benchq/logs/sdpa-ab-session.log.

## Instrument note

The first pass of this session ran both sides in the after venv (a loop
variable not reset) and the before side refused to emit every number
because its claimed wheel did not match the loaded library. Third refusal
tonight, third real mistake caught. Fixed in ~/benchq/sdpa-ab.sh on jwm1.
