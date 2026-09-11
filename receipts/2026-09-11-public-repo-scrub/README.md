# Public-repo scrub - 2026-09-11

Forward-only redaction of private infrastructure detail from tracked files,
at the repository owner's direction. Git history is deliberately unchanged:
a rewrite would invalidate every commit SHA cited as evidence inside the
receipts, and the owner chose the evidence chain over history purity.

## What was removed

| Class | Replacement | Files |
|---|---|---|
| VPN/LAN host addresses | `<m1-host>`, `<mac-host>`, `<m1-lan>`, `<lan-host>` | 43 |
| NAS backup target + keychain observation | `<redacted: backup target>` | 1 |
| Resident service names, paths and ports from `ps` dumps | `<resident-inference-service>`, `<resident-cpu-service>`, or a count | 7 |
| Battery serial from `pmset` output | `id=<redacted>` | 16 |
| Router container, config path, service unit, guard script, six model alias names | removed outright | `docs/plans/2026-09-11-m1-max-bringup.md` |

Host nicknames (`jwm1`, `16m1mbp`, `macstudio`) are kept: they identify which
chip produced which measurement and carry no address or topology. Receipt
directory names that embed them are unchanged so cited paths keep resolving.

## Source change, not just a scrub

`receipts/2026-09-10-native-macos-metal-baseline/harness/run_native_matrix.py`
recorded full `ps` command lines under `resident_idle_services_recorded`.
It now records `resident_idle_service_count` - a count. That field is what
leaked the user paths, the service port and the fleet's service inventory
into five published receipts.

`scripts/test_collect.py` had the owner's real LAN address as the fixture
proving the redactor removes addresses. Swapped to the RFC 5737
documentation range (`198.51.100.7`); 73/73 tests still pass.

## Verification

    git grep -nE '100\.84\.184\.102|100\.67\.134\.6|192\.168\.(3\.66|8\.2|7\.141)' ; # no matches
    git grep -n 'omlx-server\|TMBackup\|InternalBattery-0 (id=[0-9]'        ; # no matches
    python3 scripts/test_collect.py -q                                        ; # 73 tests OK

Every touched JSON file was re-parsed after editing; all valid.

## Root cause

Two different failures, same shape. The harness treated "record the
environment honestly" as "dump whatever the OS printed", and a planning
document copied an operational standing order - written for a private notes
file - into a public repo. Neither was needed to judge a measurement.
