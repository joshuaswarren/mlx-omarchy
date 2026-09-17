import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "bun:test";
import payloadSchemaJson from "../../schema/payload-v1.schema.json";
import { SCHEMA_IDENTITY } from "../../src/routes";
import { sha256Hex } from "../../src/hash";

const schemaBytes = readFileSync(
  join(import.meta.dir, "..", "..", "schema", "payload-v1.schema.json"),
);
const expectedFieldsHash = sha256Hex(
  Object.keys(payloadSchemaJson.properties).sort().join("\n"),
);
const expectedSchemaHash = sha256Hex(new Uint8Array(schemaBytes));

describe("SCHEMA_IDENTITY", () => {
  test("schema_version matches the bundled JSON schema", () => {
    const declared = payloadSchemaJson.properties.schema_version.const;
    expect(SCHEMA_IDENTITY.schema_version).toBe(declared);
  });

  test("fields_sha256 matches the sorted field names of the bundled schema", async () => {
    expect(SCHEMA_IDENTITY.fields_sha256).toBe(await expectedFieldsHash);
  });

  test("schema_sha256 matches the raw bundled schema bytes", async () => {
    expect(SCHEMA_IDENTITY.schema_sha256).toBe(await expectedSchemaHash);
  });

  test("ane_port is declared in the bundled schema (the 2026-09-17 incident)", () => {
    expect(payloadSchemaJson.properties).toHaveProperty("ane_port");
  });

  test("ane_port_detail is declared in the bundled schema (v0.6.1)", () => {
    expect(payloadSchemaJson.properties).toHaveProperty("ane_port_detail");
  });
});
