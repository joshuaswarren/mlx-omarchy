# T6000 ANE engine binding cells

Date: 2026-09-13
Input mapping: `receipts/2026-09-13-t6000-live-mapping.md`
ANE driver source: omarchy-ane `061b461410a595eb9884e48a250dcc47fc77be8d`
Linux source: omarchy-linux `945766977286778643c6149d86eb46d1f473a0e5`

## Result

The first live `arm-io/ane0` register range is the current driver's `engine` resource. Under the T6001 parent bus, its Linux cells are:

```dts
reg = <0x2 0x84000000 0x0 0x02000000>;
reg-names = "engine";
```

This resolves only the named engine prerequisite. It is not a complete ANE node.

## Source proof

The input receipt records these range-translated candidates from the live ADT:

| ADT range | Physical address | Size |
| --- | ---: | ---: |
| range 0 | `0x284000000` | `0x02000000` |
| range 1 | `0x28e080000` | `0x0000c02c` |

The current driver requires a memory resource named `engine`: `ane/src/ane_drv.c:624-628` calls `devm_platform_ioremap_resource_byname(pdev, "engine")`. Linux resolves that name to an `IORESOURCE_MEM` and maps that resource in `drivers/base/platform.c:142-148`. The mapping uses the resource's exact size in `lib/devres.c:142` and `lib/devres.c:159`.

The driver's first engine accesses are not at offset zero. `ane/src/ane_tm.c:13-14` defines task-manager bases `0x20000` and `0x21000`; lines 54-57 add those offsets to `ane->engine` for every task-manager and task-queue access. Range 1 is only `0xc02c` bytes, so it ends before the first `0x20000` access and cannot satisfy this driver's mapped `engine` contract. Range 0 is `0x2000000` bytes and contains both bases. This size exclusion assigns `engine` to range 0 without using an address guess.

The existing T8103 staging node is supporting, not cross-SoC, evidence: `arch/arm64/boot/dts/apple/t8103.dtsi:1443-1449` also assigns `engine` to the first ANE register tuple.

## Cell conversion

The M1 Max source places devices below a `soc` bus with `#address-cells = <2>`, `#size-cells = <2>`, and identity `ranges;` in `arch/arm64/boot/dts/apple/t6001.dtsi:28-36`. Splitting the physical start and size into 32-bit cells gives:

| Value | High cell | Low cell |
| --- | ---: | ---: |
| start `0x284000000` | `0x2` | `0x84000000` |
| size `0x02000000` | `0x0` | `0x02000000` |

The resolved node unit address would therefore be `ane@284000000` if the remaining prerequisites establish an ANE node there.

## Verification

The live input remains the `jw16mbp1-linux` M1 Max capture on kernel `7.1.6-1-1-ARCH`; this follow-up used the scoped values recorded in the input receipt and did not access that host. The documentation check ran from the isolated mlx-omarchy worktree:

~~~sh
python3 - <<'PY'
from pathlib import Path
start, size, small, tm, tq = 0x284000000, 0x02000000, 0x0000c02c, 0x20000, 0x21000
assert (start >> 32, start & 0xffffffff) == (0x2, 0x84000000)
assert (size >> 32, size & 0xffffffff) == (0x0, 0x02000000)
assert small < tm < tq < size
ane = Path('/home/joshuawarren/.config/superpowers/worktrees/omarchy-ane/m1max-bringup/ane/src')
linux = Path('/home/joshuawarren/src/omarchy-linux')
assert 'devm_platform_ioremap_resource_byname(pdev, "engine")' in (ane / 'ane_drv.c').read_text()
tm_src = (ane / 'ane_tm.c').read_text()
assert '#define ANE_TM_BASE\t\t  0x20000' in tm_src
assert '#define ANE_TQ_BASE\t\t  0x21000' in tm_src
dts = (linux / 'arch/arm64/boot/dts/apple/t6001.dtsi').read_text()
assert '#address-cells = <2>;' in dts and '#size-cells = <2>;' in dts
receipt = Path('receipts/2026-09-13-t6000-engine-binding-cells.md').read_text()
assert 'reg = <0x2 0x84000000 0x0 0x02000000>;' in receipt
print('PASS engine=range0 cells=<0x2 0x84000000 0x0 0x02000000> range1_size=0xc02c first_access=0x20000')
PY
~~~

Observed output:

~~~text
PASS engine=range0 cells=<0x2 0x84000000 0x0 0x02000000> range1_size=0xc02c first_access=0x20000
~~~

## Still unresolved

No cells are supplied here for `compatible`, interrupts, `iommus`, `power-domains`, firmware memory, or the second register range. The AIC trigger type, Linux stream ID, firmware ownership, and ordered T6000 PMGR leaf domains still require source evidence. No DTS was generated or applied, and `ane.ko` was not loaded.
