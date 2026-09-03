import { describe, expect, test } from "bun:test";
import payloadSchemaJson from "../../schema/payload-v1.schema.json";
import { SchemaNode, validateSchemaRoot } from "../../src/schema";
import fixture from "./fixtures/payload-v1.json";

const schema = payloadSchemaJson as SchemaNode;

function mutate(overrides: Record<string, unknown>): Record<string, unknown> {
  return { ...fixture, ...overrides };
}

describe("payload schema v1", () => {
  test("collector-generated fixture validates", () => {
    expect(validateSchemaRoot(fixture, schema)).toEqual([]);
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
});
