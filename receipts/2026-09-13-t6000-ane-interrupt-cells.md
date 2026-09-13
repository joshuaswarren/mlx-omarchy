# T6000 ANE interrupt binding cells

Date: 2026-09-13

Input mapping: `receipts/2026-09-13-t6000-live-mapping.md`

Linux source: omarchy-linux `945766977286778643c6149d86eb46d1f473a0e5`

ANE driver source: omarchy-ane `061b461410a595eb9884e48a250dcc47fc77be8d`

## Result

The T6000-family AIC2 specifier has four cells. Three cells are source-qualified for the captured ANE interrupt:

| Cell | Sourced value | Basis |
| --- | --- | --- |
| interrupt type | `AIC_IRQ` (`0`) | The capture identifies a hardware interrupt, and the AIC binding assigns `0` to hardware IRQs. |
| die ID | `0` | T6000 inherits the T6001 die-0 description, whose AIC2 users put `0` in this cell. |
| interrupt number | `0x302` (`770`) | Exact raw ANE ADT value. |
| interrupt flags | **unresolved** (`IRQ_TYPE_LEVEL_HIGH` candidate) | No T6000 ANE source establishes the line polarity. |

The sourced prefix is therefore:

```text
<AIC_IRQ 0 0x302 [IRQ_TYPE_LEVEL_HIGH UNPROVED]>
```

The Linux consumer name is:

```dts
interrupt-names = "ane";
```

This is an explicit missing-symbol stop: the candidate kernel symbol `IRQ_TYPE_LEVEL_HIGH` lacks T6000-specific source qualification. Do not emit an `interrupts` property until a T6000 source establishes the fourth-cell `IRQ_TYPE_*` value. The AIC2 binding says `IRQ_TYPE_LEVEL_HIGH` is normal, and the T8103 ANE uses it. Neither statement proves the polarity of T6000 ANE interrupt `0x302`, so the symbol is not recorded as a resolved cell.

The raw ADT parent `0x7e` is not copied into Linux DTS. The Linux reference is `interrupt-parent = <&aic>;`, where `aic` is the T6000-family four-cell AIC2 provider inherited through the T6000 include chain.

## Source proof

- `receipts/2026-09-13-t6000-live-mapping.md:60-61` records ANE raw interrupt `0x302` and raw parent phandle `0x7e`; the same row warns that `0x7e` is not a Linux phandle.
- `receipts/2026-09-13-t6000-live-mapping.md:33-38` records the matching live FDT AIC2 provider at `interrupt-controller@28e100000`, compatible `apple,t6000-aic`, with four interrupt cells.
- `arch/arm64/boot/dts/apple/t6000.dtsi:10-17` says T6000 is a cut-down T6001, includes `t6001.dtsi`, and declares `apple,t6000`.
- `arch/arm64/boot/dts/apple/t6001.dtsi:42-48` sets `DIE_NO 0` and includes `t600x-die0.dtsi` under `/soc`.
- `arch/arm64/boot/dts/apple/t600x-die0.dtsi:17-24` defines label `aic`, compatible `apple,t6000-aic`, with `#interrupt-cells = <4>`.
- `Documentation/devicetree/bindings/interrupt-controller/apple,aic2.yaml:54-73` defines the cells as interrupt type, T6000 die ID, interrupt number, and interrupt flags. It identifies type `0` as a hardware IRQ and says the final flag is normally `IRQ_TYPE_LEVEL_HIGH (4)`.
- `include/dt-bindings/interrupt-controller/apple-aic.h:5-8` defines `AIC_IRQ` as `0`.
- `arch/arm64/boot/dts/apple/t600x-die0.dtsi:37-43` is a direct T6000-family use of the four-cell form: `<AIC_IRQ 0 1010 IRQ_TYPE_LEVEL_HIGH>`. This proves the cell placement and die-0 value, not the trigger of ANE IRQ `0x302`.
- `include/dt-bindings/interrupt-controller/irq.h:13-18` defines the available standard symbols, including `IRQ_TYPE_LEVEL_HIGH` as `4`.
- omarchy-ane `ane/src/ane_drv.c:615-618` requests the IRQ by the exact name `"ane"`; lines `734-737` include `apple,t6000-ane` in the same driver's match table.
- `arch/arm64/boot/dts/apple/t8103.dtsi:1450-1453` names the first T8103 ANE interrupt `"ane"` and gives it `IRQ_TYPE_LEVEL_HIGH`. This supports the name and candidate trigger only. It is not cross-SoC proof for T6000.

## Boundary

No DTS was generated or applied. No module was loaded, no Mesa source was changed, no GPU process was stopped, and no hardware command was run.
