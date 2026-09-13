# T6000 ANE DART stream-ID cells

Date: 2026-09-13

Input mapping: `receipts/2026-09-13-t6000-live-mapping.md`

Linux source: omarchy-linux `945766977286778643c6149d86eb46d1f473a0e5`

mlx-omarchy base: `f92d1cc1e027937c422e28021f533310af74adf8`

## Result

**STOP: no source-qualified `iommus` tuple can be emitted.**

The missing source symbol is the ANE master's emitted T6000 DART stream-ID cell, named here only as the placeholder `ANE_T6000_STREAM_ID`. No symbol or T600x ANE node in the reviewed kernel source supplies its numeric value. The required shape is:

```dts
iommus = <&[unresolved ANE DART provider] [ANE_T6000_STREAM_ID missing]>;
```

The live raw ADT value `sids = <0xa001>` is not a Linux stream-ID cell. The `apple,dart` binding defines one cell as the stream ID emitted by the master. The T6000 driver supports 16 streams, so its Linux stream-ID domain is `0` through `15`; `0xa001` is `40961`. The driver's `of_xlate` path reads the whole cell literally as `args->args[0]` and sets that bit. It contains no ADT `sids` decoder. Copying `0xa001`, taking its low nibble, or treating its set bits as stream IDs would each be an unsupported decode.

## Live range conversion

The input receipt records four live `arm-io/dart-ane0` windows after `arm-io` range translation. Their mechanical Linux address and size cell splits are:

| ADT range | Physical window | Linux `reg` cells |
| --- | --- | --- |
| 0 | `0x285800000 + 0x4000` | `<0x2 0x85800000 0x0 0x00004000>` |
| 1 | `0x285810000 + 0x4000` | `<0x2 0x85810000 0x0 0x00004000>` |
| 2 | `0x285820000 + 0x4000` | `<0x2 0x85820000 0x0 0x00004000>` |
| 3 | `0x285804000 + 0x4000` | `<0x2 0x85804000 0x0 0x00004000>` |

These are address-cell conversions, not four approved DART nodes. The binding permits one `reg` range per provider. T600x source follows that shape: each DART provider has one `0x4000` range. Where one master reaches several DARTs, the source creates separate provider nodes and lists one `<&provider stream-id>` tuple per provider; the ISP uses three DART nodes and three tuples. Kernel source does not map the four live ANE windows to provider labels or state how many of them `mapper-ane0` uses. That provider mapping remains unresolved independently of the missing stream ID.

## Source proof

- `receipts/2026-09-13-t6000-live-mapping.md:63` records that the live ANE IOMMU parent resolves to ADT `arm-io/dart-ane0/mapper-ane0`, not to a Linux tuple.
- `receipts/2026-09-13-t6000-live-mapping.md:66-78` records the four translated `dart-ane0` ranges, raw `sids = <0xa001>`, and the existing warning that no reviewed rule converts it to a Linux cell.
- omarchy-linux `Documentation/devicetree/bindings/iommu/apple,dart.yaml:24-38` admits `apple,t6000-dart` and allows one `reg` item.
- The same binding at lines `48-52` requires `#iommu-cells = <1>` and defines the cell as the stream ID emitted by a master. Its example at lines `67-75` uses the literal tuple `<&dart1 0>`.
- `drivers/iommu/apple-dart.c:993-1006` requires one argument and assigns `sid = args->args[0]`; lines `1030-1039` set that literal SID in the selected DART's bitmap. There is no ADT `sids` decode in this path.
- `drivers/iommu/apple-dart.c:1528-1534` sets `apple_dart_hw_t6000.max_sid_count = 16`.
- `arch/arm64/boot/dts/apple/t600x-die0.dtsi:37-44` defines a T6000 DART provider with one `0x4000` `reg`, one IOMMU cell, and label `pmp_dart`; lines `209-218` attach its master with the literal tuple `<&pmp_dart 0>`.
- `arch/arm64/boot/dts/apple/t600x-die0.dtsi:380-408` attaches two masters to the same `aop_dart` with distinct literal stream IDs `7` and `0`. This confirms that the cell is the per-master stream ID, not an encoded DART address or range selector.
- `arch/arm64/boot/dts/apple/t600x-die0.dtsi:803-835` represents three ISP DART windows as three provider nodes and attaches the ISP with three `<&provider 0>` tuples. This is the sourced T600x pattern for a master using several DART windows.
- A scoped `git grep` over the binding, Apple DART driver, and T6000/T6001/T6002/T600x DTS sources found no `"sids"` property reader, `of_*sids` decoder, or `0xa001` mapping (`git-grep-exit=1`). A second scoped search found no T600x `ane_dart`, `dart-ane`, or ANE `iommus` node (`git-grep-exit=1`).

## Verification

The documentation check ran from the isolated mlx-omarchy feature worktree:

~~~sh
python3 - <<'PY'
from pathlib import Path

linux = Path('/home/joshuawarren/src/omarchy-linux')
receipt = Path('receipts/2026-09-13-t6000-ane-dart-cells.md')
binding = (linux / 'Documentation/devicetree/bindings/iommu/apple,dart.yaml').read_text()
driver = (linux / 'drivers/iommu/apple-dart.c').read_text()
t600x = (linux / 'arch/arm64/boot/dts/apple/t600x-die0.dtsi').read_text()

ranges = [
    (0x285800000, 0x4000, (0x2, 0x85800000, 0x0, 0x4000)),
    (0x285810000, 0x4000, (0x2, 0x85810000, 0x0, 0x4000)),
    (0x285820000, 0x4000, (0x2, 0x85820000, 0x0, 0x4000)),
    (0x285804000, 0x4000, (0x2, 0x85804000, 0x0, 0x4000)),
]
for start, size, cells in ranges:
    assert (start >> 32, start & 0xffffffff, size >> 32, size & 0xffffffff) == cells

assert 'The single cell describes the stream id emitted by' in binding
assert 'sid = args->args[0];' in driver
assert '.max_sid_count = 16,' in driver
assert 'sids' not in driver
assert 'iommus = <&aop_dart 7>;' in t600x
assert 'iommus = <&isp_dart0 0>, <&isp_dart1 0>, <&isp_dart2 0>;' in t600x
assert 'STOP: no source-qualified' in receipt.read_text()
print('PASS stop=missing-ANE-T6000-stream-ID ranges=4 raw_sids=0xa001 t6000_streams=16')
PY
~~~

Observed output:

~~~text
PASS stop=missing-ANE-T6000-stream-ID ranges=4 raw_sids=0xa001 t6000_streams=16
~~~

## Boundary

No DTS was generated or applied. No module was loaded, no Mesa source was changed, no GPU process was stopped, and no hardware command was run.
