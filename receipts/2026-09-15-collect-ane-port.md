# Receipt: collect_quick `ane_port` driver-port capture — 2026-09-15

## What landed

- `scripts/collect_quick.py`: new `probe_ane_port` section. `_ane_port_devicetree`
  walks the live devicetree and captures, per SoC: every `ane` node in full
  (`reg` decoded as address/size using the node's `#address-cells`/
  `#size-cells`, `reg-names`, `interrupts`, `interrupt-parent`, `iommus`
  with phandles resolved to DART paths via each DART's `#iommu-cells`,
  `power-domains`, `status`, `compatible`, and remaining props), every DART
  node, PMGR power-domain children with labels, the AIC compatible, and the
  phandle map. `_ane_port_runtime` captures `/proc/iomem` ane/dart lines,
  `/sys/module/ane/{version,srcversion}`, the `/proc/modules` ane line, and
  redacted capped `dmesg | grep -iE 'ane|dart|pmgr'` output. A tree with no
  `ane` node (stock t8103 dtb) reports `ane_node_present: false` and still
  dumps DART/PMGR/AIC so the overlay can be authored off-machine. String
  values pass the shared Redactor; all lists capped.
- `scripts/collect_common.py`: `build_payload` gains one bounded column,
  `ane_port` (≤1024 chars): presence, ane node reg base, dart count,
  pmgr domain count, AIC compatible. Null when the section is absent.
- `services/community-data/schema/payload-v1.schema.json`: `ane_port`
  added (string|null, maxLength 1024) to keep the pinned schema in sync.
- `scripts/test_collect.py`: `AnePortDevicetreeProbe` — fake t6001-style
  tree (ane node present, full prop assertions incl. iommu phandle
  resolution) and fake t8103-style stock tree (no ane node, DART/PMGR/AIC
  still dumped), plus the bounded payload summary. Pinned key-set test and
  section list updated.
- `CONTRIBUTING.md`: new section "Porting omarchy-ane to a new SoC" listing
  exactly which fields the report captures and pointing at omarchy-ane's
  `ane/t6001-j316c-set-domains.dts`.

## Evidence

- `python3 scripts/test_collect.py` → Ran 76 tests, OK.
- `python3 scripts/test_collect_macos.py` → Ran 25 tests, OK.
- `python3 scripts/collect_quick.py` on x86 Linux (no Apple DT) → clean
  report, `ane_port.devicetree` all-empty, runtime facts null, exit 0.
