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
