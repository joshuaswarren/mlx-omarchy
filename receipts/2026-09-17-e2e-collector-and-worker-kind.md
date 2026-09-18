# Omarchy Mac E2E collector + community-data kind extension (2026-09-17)

Naeem Malik's ping ask (this repo's
`receipts/naeem-ping-2026-09-17/`): generalize the mlx-omarchy hardware
collector so it captures everything the E2E volunteer checklist covers.

## Answer to "same worker or a second one?"

**Same worker.** `mlx-omarchy-community-data` stores `kind` as a plain
string column and serves summaries from the payload JSON, so a new kind
needed no second Cloudflare resource — only an enum extension. Worker
changes (mlx-omarchy branch `e2e-kind`, commit `51158400`):

- `schema/payload-v1.schema.json`: kind enum += `omarchy-mac-e2e`;
  new optional nullable fields `test_id`, `install_path`, `asahi_image`,
  `encryption`, `boot_separate`, `overall`.
- `src/validate.ts`: kind union + check.
- `SCHEMA_IDENTITY` regenerated (`fields 4484ecd9…`, `schema 5781ea64…`).
- New fixture `test/unit/fixtures/payload-v1-e2e.json`; unit 65/65,
  smoke 13/13 (incl. `GET /v1/schema` reporting the new identity).
- Old collectors unaffected: their fields are unchanged and the new ones
  are optional.

## Collector (omarchy-mac-fork branch `e2e-collector`, commit `b4b53f4`)

`scripts/collect_e2e.py` + vendored `collect_common.py`,
`collect_submit.py`, `collect_quick.py`, `collect_macos.py` (one lazy
`bench_matrix` import so the set is self-contained).

Sections, mapped to the checklist:

| section | checklist coverage |
| --- | --- |
| identity | §1 test assignment prompts (reused from ./e2e-identity.json) |
| baseline | §3 baseline commands + storage facts: LUKS present, /boot separate, freshness marker |
| mlx | full mlx-omarchy quick report: host/devicetree, Mesa/Vulkan, ANE devicetree, installed distributions, default device |
| install-logs | §4–5: setup/install/pacman log tails, `omarchy-mac-setup --status`, service state |
| boot | §5–7: `journalctl --list-boots`, current-boot warnings, failed system+user units, /boot tree |
| install-state | §6: omarchy version/path, pinned packages, `pacman -Dk`, `omarchy-migrate --pending` (exit semantics recorded, not interpreted), `omarchy-done` checks, swapon |
| security | §7: setup conf/sudoers presence, passwordless-sudo probe, sshd, nft ruleset, user journal warnings |
| macos | optional `--from-macos FILE` for the pasted macOS-side output |
| interview | `--interview`: every human-judgment checkbox (§5–9) as PASS/FAIL/SKIP/NA + notes, persisted to ./e2e-answers.json |

Output: deterministic redacted archive + `submission.md` that pre-fills
the checklist report template (model, SoC, encryption and boot-separate
flags auto-filled; stage verdicts from interview answers). `--submit URL`
speaks the community-data protocol with kind `omarchy-mac-e2e`.

Verified on a non-Apple host: all sections complete, unavailability
recorded as data, redaction summary present (402 hostname / 9 home / 8
username redactions), archive 15,479 B, submission renders.

## Deploy (BLOCKED on credentials)

Canonical token lives at esper `~/.config/cloudflare/thewarrens-co.env`;
esper SSH is down (LAN + Tailscale both time out, 2026-09-17 ~18:30 CDT),
and the documented CT 158 copy (`/etc/paperless-mail-ingest/env`) is not
present on raptor. Deploy once esper is reachable:

```sh
# on esper, from a checkout of mlx-omarchy branch e2e-kind
cd services/community-data
set -a; . ~/.config/cloudflare/thewarrens-co.env; set +a
CLOUDFLARE_API_TOKEN=$CF_API_TOKEN CLOUDFLARE_ACCOUNT_ID=$CF_ACCOUNT_ID \
  bunx wrangler deploy
scripts/check_schema_identity.py   # must exit 0 against the live URL
```

Post-deploy smoke: `python3 scripts/collect_e2e.py --out smoke.tar.gz
--submit https://mlx-omarchy-community-data.joshua-s-warren.workers.dev
--skip interview` from any machine (one labeled `test_id` row; note it in
this receipt), or negative-path first: a `kind: omarchy-mac-e2e` initiate
with a stale payload must 422 naming the first unknown field.

## Deployed and live-verified (2026-09-18)

- Token transferred to this CT by Joshua (esper → CT, key-based).
- `wrangler deploy` → version 7571e527; `check_schema_identity.py --url` →
  OK (fields 4484ecd9…, schema 5781ea64…).
- First live submit 500'd: `initiate: insert succeeded but row is missing`.
  Root cause: `0001_init.sql` pinned `kind CHECK (kind IN ('quick','deep'))`;
  `INSERT OR IGNORE` silently dropped the e2e row. Fixed in
  `migrations/0002_e2e_kind.sql` (rebuild without the kind list; API
  validator is the gate), applied remotely, proven in real sqlite, and
  guarded by a new smoke scenario.
- Live round trip: collector `--submit` → 200, receipt
  https://mlx-omarchy-community-data.joshua-s-warren.workers.dev/v1/results/ef89a1c651ce3cb599be62dc7baa49f8bacb5e76e9e3a56d41b0fddd60627ff3
  stored kind=omarchy-mac-e2e, test_id=collector-smoke-2026-09-17,
  arch=x86_64, archive 19,353 bytes. That row is a labeled smoke row
  (non-Apple dev box), not a volunteer submission.
- Local `wrangler dev` esbuild watcher crash on this CT is a tooling flake
  (same wrangler passed 13/13 earlier today); migration proven in sqlite.

## Merge and processing pipeline (2026-09-18)

- Merged to mlx-omarchy main: dashboard Kind column added; merged
  origin/main's schema growth with the e2e fields (recomputed identity
  4484ecd9…/5199ea0f… — fields hash unchanged, so the merged property set
  equals the e2e-kind set; main's lane fields were already a subset).
  Local main was diverged (stale /tmp/mlxmerge worktree + active encoder
  lane); resolved by taking origin/main's side for the encoder file and
  recomputing the schema identity. Deployed version 0ae54766.
- Live pipeline verification: dataset serves quick=34 / deep=30 /
  omarchy-mac-e2e=1 (caches rebuild immediately on publish, no cron wait);
  e2e initiate passes the kind gate (reaches PoW enforcement); dashboard
  serves the Kind column; old quick/deep rows unchanged.
- Omarchy Mac side: PR opened — omacom/omarchy-mac#464
  (joshuaswarren:e2e-collector → quattro) with docs/e2e-collector.md,
  the no-clone bootstrap bin/omarchy-mac-e2e-collect, and the checklist.
  The bootstrap defaults to omacom@quattro, so it goes live for
  volunteers on merge.
