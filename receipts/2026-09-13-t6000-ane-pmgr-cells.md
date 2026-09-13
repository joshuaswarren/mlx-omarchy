# T6000 ANE PMGR power-domain cells

Date: 2026-09-13

Input mapping: `receipts/2026-09-13-t6000-live-mapping.md`

Linux source: omarchy-linux `945766977286778643c6149d86eb46d1f473a0e5`

mlx-omarchy base: `864744c943953fa23fb0df485ff285f333c1c22f`

## Result

**STOP: no source-qualified ordered `power-domains` list can be emitted.**

The missing source symbol is a T6000 generic power-domain provider for the live ADT leaf `ANE-SYS-V`, named here only as the placeholder `T6000_ANE_SYS_V_GENPD`. The reviewed T6000-family source defines no `ps_ane_sys_v`, `ps_ane_base`, or `ps_ane_set*` provider that can represent that leaf. Therefore this receipt does not substitute an ancestor and does not emit a DTS property.

The source-qualified ancestry stops at the missing leaf:

```text
ADT power-gate 0x1cf: ANE-SYS-V
  -> Linux [T6000_ANE_SYS_V_GENPD missing]
       -> &ps_ane_sys_cpu
            -> &ps_ane_sys
                 -> &ps_afr
```

The three existing phandles are an ancestor chain, not an ordered ANE leaf list. This is not valid output:

```dts
/* INVALID: bypasses the unsourced ANE-SYS-V leaf */
power-domains = <&ps_ane_sys_cpu>, <&ps_ane_sys>, <&ps_afr>;
```

## Live mapping ancestry

The input receipt records the live relation at `receipts/2026-09-13-t6000-live-mapping.md:86-98`: ANE power gate `0x1cf` resolves to ADT device `ANE-SYS-V`, whose parent is `ANE_SYS_CPU`. It also records the live Linux FDT chain `ane_sys_cpu -> ane_sys -> afr` and says the `ANE-SYS-V` leaf has no Linux provider. This receipt uses that captured ancestry without reading hardware.

For the M1 Max die-zero source, `arch/arm64/boot/dts/apple/t6001.dtsi:42-52` defines an empty `DIE` suffix and includes `t600x-pmgr.dtsi`. The resulting labels are therefore `ps_ane_sys_cpu`, `ps_ane_sys`, and `ps_afr`.

## Linux PMGR source proof

The Apple PMGR binding establishes the representation rule:

- `Documentation/devicetree/bindings/power/apple,pmgr-pwrstate.yaml:21-25` says each PMGR power-controller node provides one generic power domain for one SoC block, and `power-domains` relationships between provider nodes represent the hierarchy.
- Lines `54-58` allow one register range and require zero-cell phandles.
- Lines `63-68` define a provider node's `power-domains` property as its parent-domain references. A node may have multiple parents, and all are powered with the child.
- Lines `70-73` say `label` names the SoC domain controlled by that provider.

The T600x provider nodes establish only this chain:

- `arch/arm64/boot/dts/apple/t600x-pmgr.dtsi:222-229` defines `ps_afr` at PMGR offset `0x1e8`, label `afr`.
- Lines `374-380` define `ps_ane_sys` at offset `0x268`, label `ane_sys`, with parent `<&ps_afr>` on die zero.
- Lines `412-418` define `ps_ane_sys_cpu` at offset `0x2c8`, label `ane_sys_cpu`, with parent `<&ps_ane_sys>` on die zero.

A scoped source search at the pinned commit found no `ANE-SYS-V`, `ANE_SYS_V`, `ane_sys_v`, `ps_ane_base`, or `ps_ane_set*` symbol in the T6000, T6001, T6002, or shared T600x DTS sources. The command exited `1` with no output. Thus the source has no provider node for the captured leaf and no T600x ANE consumer node that supplies an ordered leaf list.

## Why another SoC does not supply the order

T8103 is evidence for the required shape, not a T6000 conversion rule. Its ANE consumer lists five leaves in an explicit order at `arch/arm64/boot/dts/apple/t8103.dtsi:1454-1455`:

```dts
power-domains = <&ps_ane_set1>, <&ps_ane_set2>,
                <&ps_ane_set3>, <&ps_ane_set4>, <&ps_ane_set5>;
```

Its PMGR source at `arch/arm64/boot/dts/apple/t8103-pmgr.dtsi:995-1055` separately defines `ane_sys_cpu -> ane_sys`, an `ane_base -> ane_sys_cpu` provider, and five `ane_set* -> ane_base` leaf providers. T600x source has only the two ANE ancestor providers. Copying the T8103 five-leaf list, inventing T600x `ps_ane_set*` nodes, or replacing the missing leaf with `ps_ane_sys_cpu` would guess both the T6000 leaf set and its order.

The current ANE driver also preserves list order: omarchy-ane `ane/src/ane_drv.c:551-569` counts `power-domains` and attaches each entry by increasing index; its cleanup at lines `534-543` detaches in reverse. That consumer behavior cannot reconstruct provider labels or an order absent from the DTS source.

## Required source before conversion

Conversion can continue only when a reviewed T6000 source supplies:

1. the PMGR power-controller node or nodes that implement live ADT `ANE-SYS-V`, including each register offset and parent relation; and
2. the exact ordered set of those leaf provider labels consumed by the T6000 ANE node.

Until then, the explicit missing symbol is `T6000_ANE_SYS_V_GENPD`, and the ordered leaf list remains unresolved.

## Verification

The documentation check ran from the isolated mlx-omarchy feature worktree:

~~~sh
python3 - <<'PY'
from pathlib import Path
import subprocess

linux = Path('/home/joshuawarren/src/omarchy-linux')
receipt = Path('receipts/2026-09-13-t6000-ane-pmgr-cells.md')
binding = (linux / 'Documentation/devicetree/bindings/power/apple,pmgr-pwrstate.yaml').read_text()
t600x = (linux / 'arch/arm64/boot/dts/apple/t600x-pmgr.dtsi').read_text()
t8103 = (linux / 'arch/arm64/boot/dts/apple/t8103.dtsi').read_text()
t8103_pmgr = (linux / 'arch/arm64/boot/dts/apple/t8103-pmgr.dtsi').read_text()

assert subprocess.check_output(['git', '-C', str(linux), 'rev-parse', 'HEAD'], text=True).strip() == '945766977286778643c6149d86eb46d1f473a0e5'
assert 'The provider controls a single SoC block.' in binding
assert 'The power hierarchy is' in binding
assert 'Reference to parent power domains.' in binding
assert 'DIE_NODE(ps_afr): power-controller@1e8' in t600x
assert 'DIE_NODE(ps_ane_sys): power-controller@268' in t600x
assert 'power-domains = <&DIE_NODE(ps_afr)>;' in t600x
assert 'DIE_NODE(ps_ane_sys_cpu): power-controller@2c8' in t600x
assert 'power-domains = <&DIE_NODE(ps_ane_sys)>;' in t600x
for missing in ('ANE-SYS-V', 'ANE_SYS_V', 'ane_sys_v', 'ps_ane_base', 'ps_ane_set1'):
    assert missing not in t600x
assert 'power-domains = <&ps_ane_set1>, <&ps_ane_set2>,' in t8103
for leaf in range(1, 6):
    assert f'ps_ane_set{leaf}: power-controller@' in t8103_pmgr
text = receipt.read_text()
assert 'STOP: no source-qualified ordered' in text
assert 'T6000_ANE_SYS_V_GENPD' in text
print('PASS stop=missing-T6000_ANE_SYS_V_GENPD ancestry=ps_ane_sys_cpu->ps_ane_sys->ps_afr t600x_leaf_count=0')
PY
~~~

Observed output:

~~~text
PASS stop=missing-T6000_ANE_SYS_V_GENPD ancestry=ps_ane_sys_cpu->ps_ane_sys->ps_afr t600x_leaf_count=0
~~~

## Boundary

No DTS was generated or applied. No module was loaded, no register or hardware access was performed, and no GPU process was stopped.
