import { describe, expect, test } from "bun:test";
import payloadSchemaJson from "../../schema/payload-v1.schema.json";
import { SchemaNode, validateSchemaRoot } from "../../src/schema";
import fixture from "./fixtures/payload-v1.json";
import e2eFixture from "./fixtures/payload-v1-e2e.json";

const schema = payloadSchemaJson as SchemaNode;

function mutate(overrides: Record<string, unknown>): Record<string, unknown> {
  return { ...fixture, ...overrides };
}

describe("payload schema v1", () => {
  test("collector-generated fixture validates", () => {
    expect(validateSchemaRoot(fixture, schema)).toEqual([]);
  });

  test("e2e fixture validates with the extended kind", () => {
    expect(validateSchemaRoot(e2eFixture, schema)).toEqual([]);
  });

  test("e2e kind but missing e2e fields still validates (fields optional)", () => {
    const { test_id, install_path, asahi_image, encryption, boot_separate, overall, ...rest } =
      e2eFixture as Record<string, unknown>;
    expect(validateSchemaRoot(rest, schema)).toEqual([]);
  });

  test("schema_version is pinned to 1", () => {
    expect(validateSchemaRoot(mutate({ schema_version: 2 }), schema).length).toBeGreaterThan(0);
    expect(validateSchemaRoot(mutate({ schema_version: "1" }), schema).length).toBeGreaterThan(0);
  });

  test("additional properties are rejected", () => {
    expect(validateSchemaRoot(mutate({ hostname: "macbook.local" }), schema).length).toBeGreaterThan(0);
  });

  test("kind is an enum", () => {
    expect(validateSchemaRoot(mutate({ kind: "telemetry" }), schema).length).toBeGreaterThan(0);
  });

  test("generated_at shape is enforced", () => {
    expect(validateSchemaRoot(mutate({ generated_at: "yesterday" }), schema).length).toBeGreaterThan(0);
  });

  test("file entries need path, bytes, sha256 and nothing else", () => {
    const files = [
      { path: "quick.json", bytes: 10, sha256: "a".repeat(64), extra: true },
    ];
    expect(validateSchemaRoot(mutate({ files }), schema).length).toBeGreaterThan(0);
    const badHash = [{ path: "quick.json", bytes: 10, sha256: "nothex" }];
    expect(validateSchemaRoot(mutate({ files: badHash }), schema).length).toBeGreaterThan(0);
    const tooMany = Array.from({ length: 65 }, (_, i) => ({
      path: `f${i}.json`,
      bytes: 1,
      sha256: "a".repeat(64),
    }));
    expect(validateSchemaRoot(mutate({ files: tooMany }), schema).length).toBeGreaterThan(0);
  });

  test("redaction_summary values must be integers", () => {
    expect(
      validateSchemaRoot(mutate({ redaction_summary: { mac: "two" } }), schema).length,
    ).toBeGreaterThan(0);
  });

  test("identity fields accept string or null but not numbers", () => {
    expect(validateSchemaRoot(mutate({ chip: 42 }), schema).length).toBeGreaterThan(0);
    expect(validateSchemaRoot(mutate({ chip: null }), schema)).toEqual([]);
  });

  test("cpu_present above the schema cap is rejected", () => {
    expect(validateSchemaRoot(mutate({ cpu_present: 65536 }), schema)).toEqual([]);
    expect(
      validateSchemaRoot(mutate({ cpu_present: 65537 }), schema).length,
    ).toBeGreaterThan(0);
    expect(validateSchemaRoot(mutate({ cpu_present: -1 }), schema).length).toBeGreaterThan(0);
  });

  test("ane_port is bounded to 1024 chars and accepts null", () => {
    expect(validateSchemaRoot(mutate({ ane_port: null }), schema)).toEqual([]);
    expect(
      validateSchemaRoot(mutate({ ane_port: "x".repeat(1024) }), schema),
    ).toEqual([]);
    expect(
      validateSchemaRoot(mutate({ ane_port: "x".repeat(1025) }), schema).length,
    ).toBeGreaterThan(0);
  });

  test("ane_port_detail rejects unknown nested properties", () => {
    const bad = {
      devicetree: { ane_node_present: true },
      runtime: { unknown_field: "x" },
    };
    expect(validateSchemaRoot(mutate({ ane_port_detail: bad }), schema).length)
      .toBeGreaterThan(0);
  });

  test("ane_port_detail accepts the bounded shape with truncation flag", () => {
    const good = {
      devicetree: {
        ane_node_present: true,
        ane_nodes: { "ane@0": { compatible: ["apple,t6001-ane"] } },
        ane_reg: ["0x26bc04000/0x24000"],
        darts: {},
        pmgr_domains: [{ path: "pmgr/p", label: "p", compatible: ["x"] }],
        pmgr_blocks: [{
          path: "soc/power-management@23b700000",
          reg: ["0x23b700000/0x14000"],
          children: [
            { name: "power-controller@c000", label: "ane_sys_cpu",
              compatible: ["apple,t8103-pmgr-pwrstate"] },
          ],
          children_total: 132,
        }],
        aic: { path: "aic", compatible: ["apple,aic"] },
        phandles: { "1": "dart@0" },
        boot: {
          model: "MacBook Pro (14-inch, 2021)",
          compatible: ["apple,t8103", "apple,arm-platform"],
          chosen: { "asahi,m1n1-stage1-version": "m1n1 1.2.1" },
        },
        dtb_sha256: "ab".repeat(32),
      },
      runtime: {
        iomem: ["ane: 0x0-0x1000"],
        module_version: "0.1",
        srcversion: "DEAD",
        loaded: "ane 32768 0 - Live",
        dmesg: ["ane: ok"],
      },
      truncated: ["ane_nodes:12"],
    };
    expect(validateSchemaRoot(mutate({ ane_port_detail: good }), schema))
      .toEqual([]);
  });

  test("ane_port_detail new fields accept null and reject bad shapes", () => {
    const nulled = {
      devicetree: {
        ane_node_present: false,
        ane_nodes: {},
        ane_reg: null,
        darts: {},
        pmgr_domains: [],
        pmgr_blocks: [],
        phandles: {},
        boot: null,
        dtb_sha256: null,
      },
    };
    expect(validateSchemaRoot(mutate({ ane_port_detail: nulled }), schema))
      .toEqual([]);
    // dtb_sha256 must be a lowercase 64-hex digest or null.
    expect(
      validateSchemaRoot(mutate({ ane_port_detail: {
        devicetree: { ane_node_present: false, ane_nodes: {}, darts: {},
                      pmgr_domains: [], phandles: {},
                      dtb_sha256: "ZZ" + "ab".repeat(31) },
      } }), schema).length,
    ).toBeGreaterThan(0);
    // pmgr block children reject unknown properties.
    expect(
      validateSchemaRoot(mutate({ ane_port_detail: {
        devicetree: { ane_node_present: false, ane_nodes: {}, darts: {},
                      pmgr_domains: [],
                      pmgr_blocks: [{ path: "p", children:
                        [{ name: "power-controller@0", reg: "0x0" }] }] },
      } }), schema).length,
    ).toBeGreaterThan(0);
    // boot.chosen values are strings only.
    expect(
      validateSchemaRoot(mutate({ ane_port_detail: {
        devicetree: { ane_node_present: false, ane_nodes: {}, darts: {},
                      pmgr_domains: [], phandles: {},
                      boot: { chosen: { "asahi,x": 42 } } },
      } }), schema).length,
    ).toBeGreaterThan(0);
  });

  test("devicetree set_base_candidate accepted (collector always emits it on Linux)", () => {
    // 2026-09-18 regression: _cap_port_detail emits set_base_candidate on
    // every Linux run, but the schema declared it only under macos/, so
    // every ANE Linux submit 422'd as schema_invalid.
    const base = {
      devicetree: {
        ane_node_present: true,
        ane_nodes: {},
        ane_reg: null,
        darts: {},
        pmgr_domains: [],
        pmgr_blocks: [],
        phandles: {},
        boot: null,
        dtb_sha256: null,
      },
    };
    expect(
      validateSchemaRoot(
        mutate({ ane_port_detail: { ...base,
          devicetree: { ...base.devicetree, set_base_candidate: null } } }),
        schema,
      ),
    ).toEqual([]);
    expect(
      validateSchemaRoot(
        mutate({ ane_port_detail: { ...base,
          devicetree: { ...base.devicetree, set_base_candidate: {
            pmgr_block: "/soc/pmu@23d100080",
            offset: "0xc000",
            base: "0x23d100000",
            driver_window_confirms: true,
          } } } }),
        schema,
      ),
    ).toEqual([]);
    // Junk inside the candidate is still rejected — strictness is retained.
    expect(
      validateSchemaRoot(
        mutate({ ane_port_detail: { ...base,
          devicetree: { ...base.devicetree, set_base_candidate: {
            offset: 49152 } } } }),
        schema,
      ).length,
    ).toBeGreaterThan(0);
    // 2026-09-18 follow-up: the Linux collector emits the ane_pwrstate_cells
    // shape (status/ane_pwrstate_cells/note), not the macOS 4-prop shape —
    // it must validate too or every ANE Linux submit 422s for real.
    expect(
      validateSchemaRoot(
        mutate({ ane_port_detail: { ...base,
          devicetree: { ...base.devicetree, set_base_candidate: {
            status: "not_available_from_device_tree",
            ane_pwrstate_cells: [
              { label: "ane_cpu", offset: 0x2e0 },
              { label: "ane_set1", offset: 0x4018 },
            ],
            note: "ane_set* entries are 4-byte power-controller pwrstate "
                + "cells, not the SET MMIO window",
          } } } }),
        schema,
      ),
    ).toEqual([]);
    expect(
      validateSchemaRoot(
        mutate({ ane_port_detail: { ...base,
          devicetree: { ...base.devicetree, set_base_candidate: {
            status: "not_available_from_device_tree",
            ane_pwrstate_cells: [{ label: "ane_cpu", offset: "0x2e0" }],
          } } } }),
        schema,
      ).length,
    ).toBeGreaterThan(0);
  });

  test("t6021 bring-up fields: macos children, dart_options, interrupt_controllers", () => {
    const macos = {
      available: true,
      instances: [],
      ane_nodes: [{
        name: "ane0",
        compatible: ["ane,t6021"],
        reg: "0x0",
        phandle: 0x169,
        children: [{ name: "engine-sub", reg: "0x285c04000:0x24000" }],
      }],
      dart_nodes: [{
        name: "dart-ane0",
        phandle: 0x16a,
        dart_id: 0x25,
        iommu_cells: null,
        dart_options: "0x25",
      }],
      interrupt_controllers: [{
        name: "aic",
        compatible: ["apple,t6021-aic"],
        phandle: 0x16b,
        interrupt_cells: null,
      }],
      coreml: { available: false, compute_units: null, error: "x" },
      powermetrics: { available: false, power_mw: null, error: "x" },
    };
    expect(
      validateSchemaRoot(mutate({ ane_port_detail: { macos } }), schema),
    ).toEqual([]);
    // children entries stay closed-world.
    expect(
      validateSchemaRoot(mutate({ ane_port_detail: { macos: { ...macos,
        ane_nodes: [{ name: "ane0", children: [{ name: "x", junk: 1 }] }] } } }),
        schema).length,
    ).toBeGreaterThan(0);
    // interrupt_controllers entries stay closed-world.
    expect(
      validateSchemaRoot(mutate({ ane_port_detail: { macos: { ...macos,
        interrupt_controllers: [{ name: "aic", flags: "0x3" }] } } }),
        schema).length,
    ).toBeGreaterThan(0);
  });

  test("t6021 bring-up fields: devicetree adt artifact and adt_nodes", () => {
    const base = {
      devicetree: {
        ane_node_present: true,
        ane_nodes: {},
        ane_reg: null,
        darts: {},
        pmgr_domains: [],
        pmgr_blocks: [],
        phandles: {},
        boot: null,
        dtb_sha256: null,
        adt: {
          found: true,
          path: "/boot/efi/m1n1",
          sha256: "ab".repeat(32),
          source: "m1n1",
        },
        adt_nodes: {
          ane0: { reg: "0x285c04000:0x24000", reg_ranges: null },
          dart_ane0: null,
          ane0_iommus: [{ stream_id: 8 }],
          ane0_interrupts: ["0x374 0x0 0x4"],
        },
      },
    };
    expect(
      validateSchemaRoot(mutate({ ane_port_detail: base }), schema),
    ).toEqual([]);
    // adt without the artifact records the miss instead of lying.
    const absent = { ...base.devicetree,
      adt: { found: false, path: null, sha256: null, source: null } };
    expect(
      validateSchemaRoot(
        mutate({ ane_port_detail: { devicetree: absent } }), schema),
    ).toEqual([]);
    // junk in adt_nodes stays rejected.
    expect(
      validateSchemaRoot(mutate({ ane_port_detail: { devicetree: {
        ...base.devicetree,
        adt_nodes: { ane0: { stream_map: "0xff" } } } } }), schema).length,
    ).toBeGreaterThan(0);
  });

  test("macos probe payload accepts pmgr_nodes and rejects junk there", () => {
    const base = {
      available: true,
      instances: [],
      ane_nodes: [],
      dart_nodes: [],
      coreml: { available: false, compute_units: null, error: "x" },
      powermetrics: { available: false, power_mw: null, error: "x" },
    };
    expect(
      validateSchemaRoot(mutate({ ane_port_detail: {
        macos: { ...base, pmgr_nodes: [
          { name: "pmgr", location: "8E080000", reg: "0000088e" },
        ] },
      } }), schema),
    ).toEqual([]);
    expect(
      validateSchemaRoot(mutate({ ane_port_detail: {
        macos: { ...base, pmgr_nodes: [
          { name: "pmgr", unknown_key: true },
        ] },
      } }), schema).length,
    ).toBeGreaterThan(0);
  });
});
