# TermA companion: the ~10 µs/dispatch decode cost is Honeykrisp's per-launch CDM_BARRIER — trimmed and packaged

Date: 2026-09-16. Companion to
`ane-linux-experiments/receipts/2026-09-16-termA-dispatch-attribution.md`
(full protocol, artifacts, commands). This repository shipped no source
change: the wheel under test is the untouched v0.6.0 release artifact
and `63c1d3cf` is not merged.

Verdict: the term-A serial cost (~10 µs/dispatch, device-invariant) is
the kitchen-sink CDM_BARRIER honeykrisp emits after every compute
launch. On G13G (jwm1, t8103) the landed mesa change
(`joshuaswarren/mesa` `hk/cdm-barrier-trim` `f96e090b382`, merged into
`honeykrisp-omarchy`) emits bits {4,5,6,7,8} instead of the 20-bit sink:

- packaged A/B (hk f96e090-2 vs hk 6f6afc8-1, 6 interleaved rounds):
  **ctx1053 99.49 → 105.78 tok/s (+6.32%), short 113.68 → 116.97
  (+2.90%)**, pins `7fd25a869ff21678` / `7da83f06ec9f001d` held 24/24;
- against the boundary-receipt base medians (102.35 ctx1024 at 201
  dispatches) the packaged driver stands at **+3.35%**;
- omarchy runtime suite on the packaged driver: 22 cases / 6189
  assertions, 0 failed;
- package `mesa-honeykrisp-omarchy 26.3.0.devel.hkf96e090-2` built with
  the unchanged PKGBUILD recipe and installed on jwm1 (rolling the
  boundary receipt's "next attack (2) per-dispatch submit-path cost";
  items (1) trig invariance and (3) L2 sector policy remain open);
- diagnostic knobs used for the attribution live on mesa branch
  `hk/app-barrier` (`HK_CDMBARBITS=<hex>`, `HK_APPBAR=1`), default off.

Boundary-receipt correction: its "no source-level change in this
repository" framing stands, but the residual is no longer unscheduled —
the driver-side lever was found, landed, packaged, and verified on the
pinned decode protocol.

## Addendum: G13X (M1 Max) port measured, not shipped

jw16 (t6001, G13C) = AGX_CHIP_G13X per agx_device.c (gen13 + multi-cluster).
The sourced designed set {4,5,6,8} (upstream's own pre-sink G13X block) was
packaged (hkd71c94e-2) and measured on jw16, 12-round interleaved: short
190.66 -> 215.47 tok/s (+13.0%) but ctx1053 142.12 -> 137.62 (-3.17%), pins
48/48, suite 41 cases / 22694 assertions green. Regressive on the Max's weak
leg -> not shipped; mesa 5deac1c8068 restores the G13X sink (merged to
honeykrisp-omarchy), jw16 rolled back to hk6f6afc8-1, llm-inference
restarted. The trim ships G13G-only.
