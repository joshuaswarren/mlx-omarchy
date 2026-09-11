# F16 SDPA causal-ragged wrong values and NaN: root cause and fix

Snapshot: 2026-09-11 upstream Python suite, cluster "f16 SDPA with causal
mask on ragged lengths" (`receipts/2026-09-11-upstream-suite`, cluster 4 in
`notes.md`). Four cases, all at `B=1, qsl=127, ksl=65, n_q_heads=32,
n_kv_heads=8` (GQA), causal, f16: errors 0.1247 / 0.0770 / 0.0787 against a
3e-4 tolerance, and one NaN at `head_dim=128, transpose=True`. Fixed on
branch `sdpa/f16-causal-ragged` at commit `a7a6d94f`.

## Root cause

The f16 fast SDPA route never materializes the causal mask. Two
dispatch-level effects combine:

1. The scores matmul (`MatmulRbF16`, `CausalSkip::Columns`, flags bit 3)
   legitimately leaves score tiles unwritten when every column of the tile
   lies past every row's last valid key. This is safe ONLY if the softmax
   that consumes the scores is itself causal.
2. `dispatch_softmax` selected causal mode with a value sentinel:
   `if (causal_offset >= 0)` where the offset is `k_len - q_len` and `-1`
   means "off". A query longer than the key produces a legitimately negative
   offset (`65 - 127 = -62`), which the sentinel silently read as "causal
   off". The softmax then ran over the full row, consuming the unwritten
   tiles.

Consequences at `qsl=127, ksl=65`:

- Every valid row's softmax max and denominator are contaminated by garbage
  scores past the causal boundary -> the 0.077-0.125 wrong values.
- Uninitialized buffer memory routinely contains f16 NaN patterns; the
  softmax max loop (`isnan(next) || next > local_max`) deliberately
  propagates NaN inputs, so one NaN-pattern garbage score turns the whole
  row NaN -> the hd=128/transpose=True NaN.

The failure set matches the mechanism exactly: every causal shape in the
upstream list has `qsl <= ksl` (offset >= 0, causal mode engaged) except
`(1, 127, 65)` at hd 64/128 x transpose False/True - precisely the four
 failing subtests. The upstream snapshot's 2026-09-06 run
 watchdog-terminated `test_fast_sdpa.py` before reaching these shapes,
 which is why they are new failures, not regressions of a passing state.

## Shared-site audit (every causal softmax consumer)

`dispatch_softmax` has exactly three callers
(`overlay/mlx/backend/omarchy/primitives.cpp`):

| Caller | Causal? | Negative offset possible? | Action |
|---|---|---|---|
| Softmax primitive (line 8947) | never (defaults) | n/a - no offset exists | none needed |
| f16/bf16 fast SDPA arm (line 10491) | `do_causal_` | yes, `k_len - q_len` | fixed here - shared site |
| f32 composition arm (line 10619) | never - mask is materialized additively | offset handled signed at line 10598 (`int offset = k_len - q_len`), mask 0 / -1e30; fully masked rows get the finite floor and a defined uniform softmax | none needed |

The f16 and bf16 fast arms share one call site and one shader
(`softmax_suffix.comp` with dtype variants), so the fix repairs both arms
once; there is no f16-only guard. The only other offset consumers are the
two `dispatch_matmul` causal pairs inside the same fast arm (scores
`CausalSkip::Columns`, probs `CausalSkip::K`), whose u32 tile arithmetic
evaluates small negative offsets correctly mod 2^32.

## Is the RB matmul leaving tiles unwritten intended?

Yes, and it stays. `matmul_rb.comp`'s header documents the contract: flags
bit 3 skips output tiles whose every column lies past every row's last
valid key, "the causal softmax never reads them". The producer/consumer
pairing is enforced at the single shared dispatch site - the same
`do_causal_` boolean drives `CausalSkip::Columns` on the matmul and the
causal flag on the softmax - so after this fix there is no reachable state
where the matmul skips tiles and the softmax reads them (the sentinel bug
was exactly such a state).

Writing exact zeros for skipped tiles was considered and rejected: it
costs the write bandwidth of the whole skipped upper-triangle region on
ragged causal prefill (up to roughly half the scores tensor at heavy
raggedness; zero effect on the shipped decode legs, where offset >= 0
skips nothing), and it would not actually remove the hazard - zeros are
still wrong scores to a non-causal consumer, turning a loud garbage/NaN
failure into a silently wrong one. The structural pairing fix is the
protection.

`matmul_rb.comp` needs no change: `last_valid = aux_size + row0 + TILE - 1`
evaluates in u32, and for small negative offsets the mod-2^32 wrap equals
the true signed value, so its skip decisions stay correct (and over-keeping
a tile is harmless; the causal softmax masks it).

## The fix (a7a6d94f)

`overlay/mlx/backend/omarchy/primitives.cpp`:
`dispatch_softmax` takes an explicit `bool causal` plus the signed offset;
the offset rides the push-constant u32 two's complement.

`overlay/mlx/backend/omarchy/shaders/softmax_suffix.comp`: the admissible
key count is computed signed and clamped,
`valid = clamp(int(aux_offset) + int(row % aux_size) + 1, 0, row_length)`.
A row with no admissible key gets zero valid columns.

### NaN impossible by construction

For a fully masked row, `valid == 0`: the max and exp-sum loops never
execute, and the store loop writes `0.0` for every column (the
`index < valid` branch is never taken, so the `1/0` normalizer is never
applied to any store). For a row with at least one admissible key, the max
is a real score, `exp(max - max) = 1` joins the sum, and the normalizer is
finite; masked columns store exact zeros. No code path stores NaN for a
causal row. The regression test asserts the zero rows are exact zeros and
that no output element anywhere is NaN.

## Regression test

`overlay/tests/omarchy/test_sdpa_causal_ragged.cpp`
(`omarchy_sdpa_causal_ragged_tests`): the four upstream combos
(hd 64/128 x transpose False/True at `qsl=127, ksl=65`, GQA 32:8, f16)
against a double host reference under the upstream tolerance criterion,
plus exact-zero assertions on the fully masked rows and a global no-NaN
assert, plus a small `qsl=4, ksl=2` control pinning the same construction.
Fail-proof on llvmpipe with pre-fix sources: all four combos fail with
worst diffs 0.1125 / 0.0797 / 0.1104 - the same signature as upstream.

One `/tmp/m1-gpu.lock` window plus a digest-only redo window
(`m1-window.sh`, `m1-window-digests.sh`, combined `window.log`, artifacts
under `m1-logs/` and `matrix/`):

1. Pre-fix repro on the pinned 2a9add42 requal wheel
   (`m1-logs/prefix-repro.log`): exactly the four subtests fail
   (`B=1, qsl=127, ksl=65, head_dim=64/128, GQA 32:8, causal,
   transpose=False/True`), including
   `AssertionError: nan not less than or equal to 0.0003` at
   head_dim=128, transpose=True - the NaN reproduced.
2. Branch wheel `mlx_omarchy-0.32.2.dev202609112233+a7a6d94f`
   (sha256 `9f83f4782d5635dd8846ce48beb60367bc8fb2c8fd145c0a73cc548b5e9bf4aa`,
   `python-provenance.json`: SOURCE-VERIFIED match, stamp a7a6d94f):
   `test_fast_sdpa.py` reports `20 passed, 3 skipped, 466 subtests passed`
   with zero failures (`m1-logs/fix-sdpa.log`, junit `sdpa-fixed.xml`).
3. C++ regression binaries: `m1-logs/cpp-causal-ragged.log` - the new
   ragged test passes 20/20 assertions; `m1-logs/cpp-sdpa-norm.log` - the
   existing W9/W10/W11 SDPA suite passes 16/16.
4. Canonical digests, fork driver, warmup + r1 + r2 (`digest-gates.json`):
   `all_held: true` - all six canonical Q4 digests
   (qwen25-0.5b-4bit short/long/1K-ctx: 7fd25a869ff21678,
   4cc08910089477fd, 7da83f06ec9f001d) and every BF16 pin
   (qwen25-0.5b-bf16 short/long/1K-ctx: f26175202f3dabe9, 8690dc83246b39f8,
   ff502900d2a179a5) hold in every rep, bit-identical. Expected: decode
   legs run at q_len=1 (offset >= 0) and never reach the changed path;
   the bf16 fast route shares the fixed call site and only changes
   behavior for causal with negative offset, absent from every leg.

The first phase-5 attempt measured nothing (every leg "skipped:
mlx_lm not importable"): the copied pytest venv lacked mlx_lm. Installed
`mlx_lm==0.31.3` (the version the canonical matrices were produced with)
and reran the digest phase in its own window; phases 1-4 were unaffected.

## Non-goals

- The narrow-type score/prob storage (known defect,
  docs/known-defects.md) is untouched; the valid-row arithmetic after this
  fix is the same arithmetic the passing causal shapes already used.
- Quantized clusters and the non-attention wrong-value sweep are owned by
  siblings.
