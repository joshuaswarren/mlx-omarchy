# When a candidate may change generated token ids

Decided 2026-09-10 while closing the M1 performance gap. The gate that had
been in force ("bit-identical to the previous Linux release on every leg")
was blocking the two largest remaining levers, and it was protecting the
wrong invariant: it froze Linux-specific arithmetic that does not match
macOS MLX, which is the parity target.

## The rule

A candidate may change a generated token stream only when both hold:

1. **The new arithmetic is native's arithmetic, proven.** The kernel must
   reproduce captured native intermediates exactly on fixed inputs (the
   oracle captures under `receipts/2026-09-09-*-control` and
   `receipts/native-baseline-2026-09-06`), not merely produce "plausible"
   output. A digest that is neither the old Linux value nor native's is
   rejected; that is what disqualified the fused SDPA attempt in
   `receipts/2026-09-09-decode-sdpa`.
2. **No leg that already matches native regresses.** Legs whose digest
   equals the native digest (today: Q4 short and Q4 1K context, BF16 1K
   context) must still equal it after the change.

Legs whose Linux digest already differs from native (today: Q4 262-token,
BF16 short, BF16 262-token) may move to native's digest. They may not move
to anything else.

## Consequences

- Release notes must state any digest change and which legs moved toward
  native. Both drivers (stock Mesa and the Honeykrisp fork) are checked;
  `receipts/2026-09-10-stock-prefill-digest` is the pattern.
- The stock-driver fallback paths are held to the same rule; a fork-only
  arithmetic change that silently alters the stock stream is a defect
  (`receipts/2026-09-10-v0.4.1-release.md`).
- Kernels kept only to preserve Linux-specific rounding, once native's
  order is implemented and proven, are deleted rather than retained behind
  a flag.
