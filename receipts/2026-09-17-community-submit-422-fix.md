# Community submit 422 on `ane_port`: stale deploy, fixed and guarded (2026-09-17)

A user with invite access could not submit: `POST /v1/submit` returned
422 `{"error":"schema_invalid","detail":{"errors":["$: additional property
ane_port is not allowed"]}}`. Their diagnosis was correct — PoW (difficulty
18), the dedup probe (404), and the 13.6 KB archive all passed; the payload
failed on that one field.

## Root cause: stale deploy, not a code defect

`services/community-data/src/routes.ts:1` imports the allowlist at BUILD
time:

```ts
import payloadSchemaJson from "../schema/payload-v1.schema.json";
```

`ane_port` was added to that schema in commit `7ab04f73`
(2026-09-15 16:38 CDT, "collect: probe ane_port and document SoC overlay
porting"), and the collector has sent it since
(`scripts/collect_common.py:401`, string or null, ≤1024 chars — matching
`schema/payload-v1.schema.json:33`). The running Worker was built before
that commit, so it rejected a field its own repository schema had carried
for two days. Both `main@99b7104c` and the `v0.6.0` tag collectors were
affected, exactly as reported.

Fix: redeploy from current `main`. No client change, no schema change, no
loosening of `additionalProperties: false`.

## Deploys

| version id | contents |
| --- | --- |
| `b5e02fe5-f999-4ce9-8e70-4458b5a7460a` | first redeploy from `4aa35ee5` — picks up the `ane_port` schema bump |
| `a750341c-28af-4438-a045-ce88e019ffe1` | sibling redeploy of the same tree |
| `21969d9a-cf0f-44cb-b245-e7bfe728c887` | adds `GET /v1/schema` + identity unit tests |

Unit tests ran green BEFORE each deploy (55/55 pre-guard, 59/59 with the
guard's 4 new tests); `bun test/smoke.ts` 12/12 against `wrangler dev` on
D1 sqlite. D1 binding and the hourly cron (`17 * * * *`) re-registered by
the deploy; no migration was required or applied.

## Verification on the live endpoint

- `scripts/collect_quick.py --submit <url>` → 200, `deduplicated=false`,
  `content_sha256=202540fd6809f3d2c383f52c4dd3cbe252b88745838a6a230d0945410f3d48ba`
- second quick submit → 200,
  `content_sha256=a153d0a28555ae858ea29db4c73b70a3d300cd04950050f01a5f4ffe38628bf3`
- `scripts/collect_deep.py --out … --submit <url>` → 200,
  `content_sha256=885fdf6e73d769146b4d11cbcc7b357285f61d84aee02a659a374d668eedc7fe`
- post-guard quick submit → 200,
  `content_sha256=b13e4eea6eb6d51e9e92007281496b7697d93dec18ffdc7408ad385dbca7b933`
- fetched a stored row back: `ane_port` present, alongside every other
  field added since the last deploy — `cpu_present`, `hotplug_control`,
  `ane_dt_node`, `ane_dt_compatible`, `boot_chain`, `cmdline`,
  `core_shortfall`, `benchmark[]`, `redaction_summary`, `files[]`. The
  reporter's failure class is gone for all of them, not just `ane_port`.
- `GET /v1/results` count 17, `generated_at` later than the submits —
  the hourly `rebuildCaches` ran on its own, no manual step.
- `GET /v1/dataset/latest.jsonl` 17 lines.

## Guard against recurrence

The failure was invisible until a user hit it, so the deployed schema
identity is now observable and checkable:

- `GET /v1/schema` → `{schema_version, fields_sha256, schema_sha256}`
  (live: `fields_sha256 ff51f59b…`, `schema_sha256 28b49062…`).
- `services/community-data/scripts/check_schema_identity.py` compares the
  live endpoint to the repository file: exit 0 match, 2 drift (naming the
  offending fields), 1 network error. Verified in all three states,
  including a monkey-patched stale-deploy simulation and an unreachable
  endpoint.
- `services/community-data/scripts/compute_schema_identity.py` prints the
  new identity after any schema edit.
- `test/unit/schema-identity.test.ts` pins the `SCHEMA_IDENTITY` constant
  against the bundled JSON so the constant cannot drift from the file.

Strict validation is retained deliberately: unknown fields stay rejected.
The defect was deploy hygiene, and it is now checkable rather than traded
away for permissiveness.

## Secret hygiene

The canonical Cloudflare token (`CF_API_TOKEN` + `CF_ACCOUNT_ID`) was read
into a 0600 temporary file by direct redirect, sourced only inside the
`wrangler deploy` invocation, filtered out of all command output, and
deleted immediately afterward (absence confirmed). It was never printed,
logged, or committed. esper was unreachable over SSH at the time; the CT 158
copy at `/etc/paperless-mail-ingest/env` served as the documented second
source (`homelab-infra/docs/cloudflare.md`).

## Known residue

Four verification rows are published and cannot be removed through the
public API (no delete route; the cron GCs only unpublished rows older than
7 days). They are x86_64 dev-box reports with null chip/Mesa fields,
submitted 2026-09-17T05:24Z–05:30Z, and will appear in the next daily
`community-data` snapshot. Downstream consumers can filter them by
`generated_at` plus `arch=x86_64` with `mesa_driver=null`. Removing them
needs either a D1 delete or an authenticated admin route — owner's call,
not taken here.

## User-facing answer

No client change is required. Resubmit as-is; the Worker now accepts
current-collector payloads including `ane_port`. If a future deploy
regresses, `check_schema_identity.py` names the drifted field instead of
leaving submitters to decode a 422.
