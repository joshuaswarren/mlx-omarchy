# Bisect observations — suite failures at tonight's checkpoints

All runs on jwm1 (Apple M1, aarch64), fresh `scripts/prepare-mlx.sh` staging
per checkpoint, Release test build, suite binaries run under
`/tmp/m1-gpu.lock`. Every checkpoint ran build and suite in the SAME
checkout; the primitive-suite case count printed per run identifies which
tree's binary produced the result (early batched loops that violated this
were discarded and re-run; only same-checkout build+run pairs are recorded
here). Raw cmake build logs: `bisect-*.log`, `iso-*.log`, `rep-*.log` in
this directory (build evidence; run results transcribed below from the
session transcript).

| checkpoint (first-parent) | copy_offset: scalar-fill ordering | primitive: affine storage-offset | prim case count |
|---|---|---|---|
| `bddc061f` pre-tonight main | PASS ×5 | FAIL ×3 | 100 |
| `5aab281f` qmm-layout merge | PASS | FAIL | 100 |
| `8fd294ed` Q4FusedProjections merge | PASS | FAIL | 103 |
| `de780aa2` scalar-fill host drain restore | **FAIL ×3** (25 passed / 1 failed) | FAIL | 103 |
| `485404cf` recycled-storage drain gate | **FAIL ×3** | FAIL | 103 |
| `f425f46f` narrow recycled-drain receipt | **FAIL** (7 submissions vs 6) | FAIL | 103 |
| `ab08be8b` composed main | **FAIL ×3 cold** + window FAIL | FAIL (window + cold) | 103 |

## copy_offset — "back-to-back scalar fills stay ordered in one submission"

- Introduced by `de780aa2` ("omarchy: restore the scalar-fill host drain
  the batching commit removed"): deterministic 25/26 from that commit
  onward, cold reproducible (no warm-driver state required).
- `485404cf` ("gate the scalar-fill drain on recycled storage instead of
  paying it always") did NOT restore the batching contract — the failure
  signature is identical before and after it (expected 6 submissions,
  got 7; the value checks at `test_copy_offsets.cpp:182-183` pass, so
  outputs remain correct — the lost property is the single-submission
  batching invariant pinned by `e5072671` earlier in this stack).
- The wrong-value motivation for the drain and the batching contract of
  the test are in direct conflict as landed; resolution (which side moves,
  or a gate that actually recovers batching for non-recycled storage) is
  routed to the regression owner.

## primitive — "quantized matmul binds affine streams at storage offsets"

- The test case exists at `bddc061f` (same TEST_CASE name, same
  `:5514` CHECK, epsilon 4e-3) and fails there deterministically (×3,
  99/100), and at every later checkpoint through `ab08be8b` (102/103),
  including the qualification window.
- m=1 element 0.416748 vs expected 0.409468 (~1.8 % off) on M1 hardware.
- Attribution: pre-existing on main before tonight's landings; NOT
  introduced or worsened by any of tonight's five. This matches the
  qmm-layout worker's "pre-existing on `bddc061f`" report; any claim that
  this case is green on M1 hardware does not reproduce.
