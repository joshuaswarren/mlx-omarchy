# Hang-watchdog hardware verification — jwm1 (M1/Honeykrisp)

Source commit: 250306543b49dbf9532911fe6a7a8d7fe8129029
Wheel: mlx_omarchy-0.32.2.dev202609040617+2503065-cp314-cp314-linux_aarch64.whl
Wheel sha256: 4d3cfcf1bdee580c756ee3ea626e5dbc9482c0d8d599a6845f1e02152393088b
Provenance gate: installed libmlx.so sha256 b4622fe7ed391505... == wheel member
sha256 (verified match); scripts/mlx_provenance.py reports dist mlx-omarchy,
verified "match", harness commit 2503065.

## Run 1 — long legitimate operation (2,048-token prompt via mlx_lm)

mlx_lm chunks prefill, so the ordinary path exercised 6,009-token context
(a larger prompt than required): rc=0, wall 183 s, `Prompt: 6009 tokens,
33.145 tokens-per-sec`, `Generation: 1 tokens`. No VK_TIMEOUT, no hang
strings. The ordinary chunked path completes on Honeykrisp.

## Run 2 — decode unchanged

`Prompt: 35 tokens, 1.809 tokens-per-sec; Generation: 32 tokens,
0.214 tok/s` on this contaminated-tail run — see the caveat below; the
clean rerun requirement is recorded. An uncontaminated decode figure from
the same session: compiled-default 7.10 tok/s and eager 7.25 tok/s median
over 63 tokens (tcf2b logs).

## Discriminator — single full-sequence eager eval, 2,048 tokens

differential_compile --mode realpath --steps 1 (one mx.eval over the full
2,048-token sequence — the exact shape that VK_TIMEOUT'd at the old 10 s
wall cap): **failed by name at 10 s with the NEW watchdog message**:
`Vulkan timeline counter failed to advance for 10000 ms (last observed=0,
target=1)`.

Interpretation: the submission genuinely WEDGED (counter never advanced at
all — last observed 0). The watchdog classified it correctly: a wedged
queue fails by name; it is not a progressing-long-work case. The old wall
cap would have produced the same named failure here; the watchdog's win is
that progressing work longer than 10 s now succeeds (proven on lavapipe;
on M1 the mlx_lm chunked path demonstrates the equivalent).

## Caveats

- The decode run immediately after the wedged discriminator ran at
  0.214 tok/s (20x slow) — residual wedged-queue state contaminated that
  run. A clean rerun is required before quoting M1 decode throughput on
  this commit. The contaminated run is preserved:
  jwm1:~/benchq/logs/hwd-run2.log (raw output in the batch artifact).
- mlx_lm does not exercise the wedge; the discriminator does. The
  wedge itself (single full-sequence eval at 2k tokens on the pre-
  watchdog code) is reproducible on jwm1 from the b18704e-era wheel and
  predates this fix.

Raw logs: jwm1:~/benchq/logs/hwd-*.log, hangwd-run1.log,
hangwd-run2.log.
