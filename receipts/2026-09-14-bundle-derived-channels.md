# 2026-09-14 bundle derived-channel check

Host-only. Not a hardware pass. No GPU lock. No ANE execute. No merge.

omarchy-ane `origin/main` pin named by the assignment: `6fa243a` (STRICT_BIND
refuses a positional map). libane `5113ef6` reports a positional channel map
and refuses it under `STRICT_BIND`; `20d24ad` reads the role-to-channel map
out of the task stream. This slice does not merge `63c1d3cf`.

## Old check

`overlay/mlx/backend/omarchy/ane/bundle.cpp` compared each manifest binding
to the positional formula:

```
output_bdx(i) = 4 + i
input_bdx(header, i) = 4 + header.destination_count + i
```

The schema-4 adapter writes those same numbers. The check compared a value
to the formula that produced it and could not fail. A task stream that
names source 4 / destination 5 (exported 1x512/1x896) still loaded if the
manifest said destination 4 / source 5.

## New check

`validate_program_contract` reads the ANEC payload and derives the
role-to-channel map the same way `ane_bind_init` does: tile-DMA selectors
in task word 8, live DMA configs at `0x13800` / `0x13804` / `0x17800`,
roles ordered by ascending allocated channel >= 4. If the walk does not
name exactly `source_count` sources and `destination_count` destinations,
it throws

`task stream does not name every surface; channel map is positional`

matching libane `STRICT_BIND` (`ane.c` around the `ane_bind_init` return).
A derived map is then compared to the manifest `channel` fields.

## Test that fails on a positional map

`TEST_CASE("positional channel map is refused")` in
`overlay/tests/omarchy/ane/test_bundle.cpp`.

It keeps the positional declaration (outputs 4, inputs 5 and 6) and zeros
the planted task image. Before this change that bundle loaded. Now
`load_bundle` throws `channel map is positional`.

Also added, both host-side:

- `derived reverse map rejects a positional declaration` — selector
  `0x00025864` derives dst=5 src=4; a positional declaration still fails.
  Vacuous under the old `4+i` check.
- `derived reverse map accepts the task-stream channels` — the same
  selector loads when the manifest names dst=5 src=4.

Default fixtures plant selector `0x00024966` (dst=4 src=5,6) so existing
cases keep a derived map equal to the old formula.

## Proof

Host: `omp-studio-local`, kernel `6.17.2-1-pve`, x86_64. No device.

```text
scripts/prepare-mlx.sh
cmake -S .work/mlx -B .work/build-bundle-derived \
  -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=OFF -DMLX_BUILD_METAL=OFF \
  -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=ON -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF -DMLX_BUILD_PYTHON_BINDINGS=OFF
cmake --build .work/build-bundle-derived --target omarchy_ane_bundle_tests -j8
./.work/build-bundle-derived/tests/omarchy/omarchy_ane_bundle_tests
```

```text
[doctest] test cases:   27 |   27 passed | 0 failed | 0 skipped
[doctest] assertions: 4732 | 4732 passed | 0 failed |
[doctest] Status: SUCCESS!
```

That binary is a host load of synthetic ANEC bytes. It is not a jwm1 or
jw16 execute.

## Diff

Against HEAD `7f8786b02f88d56cb98f5779aea7b9ab4a715972`, uncommitted:

```text
 overlay/mlx/backend/omarchy/ane/bundle.cpp | 172 +++++++++++++++++++++++++++--
 overlay/tests/omarchy/ane/test_bundle.cpp  |  80 +++++++++++++-
 2 files changed, 243 insertions(+), 9 deletions(-)
```

| file | sha256 |
| --- | --- |
| `overlay/mlx/backend/omarchy/ane/bundle.cpp` | `7c4716a467b44b0e6b616f26a3aa39a47cc0d11ea7e64ea4efcfb964c5f003ba` |
| `overlay/tests/omarchy/ane/test_bundle.cpp` | `38cdb5d669eeb80a64b5e818f07f1cf55ee39d6b2ca37eb5d0fba6a57c667a9c` |

## Remaining

- Existing schema-4 manifests that still declare formula-derived channels
  will fail this loader until they are re-adapted against the task stream.
- `mlx-omarchy-info` still prints `4+i` as a comment channel; not this slice.
- The worker still binds libane by role index. This check only fails closed
  at bundle load.
- Not landed. Hand to LandHostLeaves.

- resolved_model: `xai-oauth/grok-4.6`
- fallback: false
