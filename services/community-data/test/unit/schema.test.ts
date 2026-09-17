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
        darts: {},
        pmgr_domains: [{ path: "pmgr/p", label: "p", compatible: ["x"] }],
        aic: { path: "aic", compatible: ["apple,aic"] },
        phandles: { "1": "dart@0" },
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
});
