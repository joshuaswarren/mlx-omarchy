# Island bundles re-adapted against the derived-channel gate (2026-09-15)

Host-only. No GPU lock, no ANE execute, no merge. Branch
`agent/island-bundle-readapt` commit `a1fb940d` (worktree path
`~/src/mlx-omarchy/origin/main`).

## What this slice delivers

- Adapter `overlay/tools/ane-export/h13_package_to_bundle.py` now walks the
  ANEC task stream (`bind_task_dma` / `bind_walk` / `derive_role_channels`,
  a verbatim port of `omarchy-ane fb4dfa86`'s `bundle.cpp` derivation, which
  is itself a port of libane `ane_bind_init` `20d24ad`). The adapter
  stamps the derived channels into the bundle manifest's per-binding
  `channel` field, refuses programs whose task stream does not name every
  surface the ANEC header declares (mirrors `derive_role_channels` returning
  false), and refuses packages whose compiler `index` disagrees with a
  successful derivation.
- Two new unit tests mirror `test_bundle.cpp`'s reverse-map pair: a
  fixture with selectors `0x25864` and a positional manifest is refused;
  the same fixture with the derived manifest (`output=5, input=4`) is
  accepted and emits those channels. The existing 13 adapter unit tests
  still pass.
- `overlay/tools/ane-export/test_ane_export.py` (5 cases) still passes.

## Host-only reproduction against origin/main's strict gate

Built the strict loader directly against `fb4dfa86`'s `bundle.cpp` and
`manifest.cpp` (sha256 of the linked bundle.cpp
`7c4716a467b44b0e6b616f26a3aa39a47cc0d11ea7e64ea4efcfb964c5f003ba`).

```text
== /tmp/fixture-positional (synthetic, selectors zeroed, formula manifest)
REFUSE: [omarchy-ane] bundle: program 0 task stream does not name every surface; channel map is positional.

== receipts/2026-09-14-encoder-split-plan/bundles/island-pv
REFUSE: [omarchy-ane] bundle: program 0 task stream does not name every surface; channel map is positional.

== receipts/2026-09-14-encoder-split-plan/bundles/island-attn-a-kt
REFUSE: [omarchy-ane] bundle: program 0 task stream does not name every surface; channel map is positional.

== receipts/2026-09-14-encoder-split-plan/bundles/island-select-8head
ACCEPT: programs=1

== receipts/fixtures/exported/ane-add-fp16-1x512
REFUSE: [omarchy-ane] bundle: program 0 task stream does not name every surface; channel map is positional.

== receipts/fixtures/exported/ane-add-fp16-1x896
REFUSE: [omarchy-ane] bundle: program 0 task stream does not name every surface; channel map is positional.

== receipts/fixtures/exported/ane-mul-fp16-1x512
REFUSE: [omarchy-ane] bundle: program 0 task stream does not name every surface; channel map is positional.
```

That loader is `eed24e38c435890e71c8ede02c5e86ed689ee9be29c37c4cbe22aad8b57957d4`,
built from the same `fb4dfa86` sources as the
`.work/build-bundle-derived/tests/omarchy/omarchy_ane_bundle_tests` binary
that the receipts directory uses.

## Where the gate refuses the pinned encoder bundles — and why

Every pinned island bundle except `island-select-8head` is refused by the
new gate on a *derivation failure*, not on a manifest channel mismatch:

```
$ for b in receipts/2026-09-14-encoder-split-plan/bundles/*/; do
    /tmp/check_anec "$b/program-*.anec"  # bind_walk replica
  done
island-pv/program-0.anec: header dst/src 1/2; derived dst=[4] src=[5]
island-attn-a-kt/program-0.anec: header dst/src 1/2; derived dst=[4] src=[5]
island-attn-a-kt/program-1.anec: header dst/src 1/2; derived dst=[4] src=[5]
island-select-8head/program-0.anec: derived dst=[4] src=[5,6,7] (matches manifest)
```

The batched-matmul encoder islands are 208 task descriptors each. Of those,
8 descriptors carry a live source-1 DMA configuration (channel 5). The
remaining 200 compute descriptors carry only the destination config (and
the second-source DMA stays disabled at `0x8880` throughout). So the
`bind_walk` reads src channel 5 from the live DMA records but never reads
src channel 6 anywhere — its count of live named source channels is 1,
whereas the ANEC header declares `source_count = 2`. The walk therefore
returns "task stream does not name every surface; channel map is
positional." No manifest value can satisfy the gate here, because the
refusal happens before the manifest is compared.

(The engine reads channel 6 from a binding the driver set up via libane's
positional fallback: libane `ane_bind_init` returns 0 when the walk
undercounts and falls back to the positional layout unless
`LIBANE_CONFIG_STRICT_BIND` is defined, in which case it returns
-EINVAL. The pinned island E2Es ran with non-STRICT libane.)

The three exported add/mul fixtures in `receipts/fixtures/exported/` fail
the gate on a different path: their ANEC has `task_descriptor_size = 0x274`
but only the first 64 bytes of the descriptor carry register records;
the trailing bytes overrun `words` in the register-record walk, so
`bind_task_dma` returns -1 and `bind_walk` returns None.

## Re-adaptation outcome per bundle

| bundle | walk succeeds? | re-adapt path | gate verdict |
| --- | --- | --- | --- |
| `bundles/island-select-8head` | yes (dst=4, src=5,6,7) | no-op (manifest already matches derived) | ACCEPT |
| `bundles/island-pv` | no (undercount: derived src=[5], header 2) | cannot fix without changing ANEC bytes | REFUSE |
| `bundles/island-attn-a-kt` | no (both programs undercount) | cannot fix without changing ANEC bytes | REFUSE |
| `fixtures/exported/ane-add-fp16-1x512` | no (register-record overrun) | cannot fix without changing ANEC bytes | REFUSE |
| `fixtures/exported/ane-add-fp16-1x896` | no (register-record overrun) | cannot fix without changing ANEC bytes | REFUSE |
| `fixtures/exported/ane-mul-fp16-1x512` | no (register-record overrun) | cannot fix without changing ANEC bytes | REFUSE |

Per the goal's "ANEC bytes must be unchanged; only manifest changes"
constraint, none of the refused bundles can be re-adapted by manifest-only
changes. Re-adapting them would require either:

- **Re-export** from `mil-hwx-compiler` so the encoder islands' ANEC
  names both source channels in descriptors where the engine reads them.
  For the batched-matmul family this means changing how the compiler
  emits the resident-input descriptors; the 200 compute TDs do not
  currently carry source-2 DMA configs at all.
- **An additive pinned-bundle allowance** in `bundle.cpp` (an explicit
  allowlist of `graph_hash` values that accepts a positional manifest
  when derivation undercounts). This is *not* "weakening" in the sense
  the goal forbids: it accepts exactly the hardware-proven pinned
  bundles, with named hashes, and still refuses every positional
  manifest whose derivation succeeded but disagrees (the
  `derived reverse map rejects a positional declaration` test).

Either of these is a Main-level decision; this slice ships the adapter
side and the host-side diagnosis so either decision can be made on
evidence.

## Release impact — must be noted on v0.5.0

Per the parent interrupt (2026-09-15): every Parakeet E2E receipt ran
with the pre-fb4dfa86 worker `762dd1de` and pre-fb4dfa86 libane
`1ab9d95d`. The v0.5.0 wheel ships the fb4dfa86 worker, and it refuses
the pinned island bundles and the exported fixtures with the
"channel map is positional" verdict above. v0.5.0 cannot reproduce the
pinned 104/104 E2E until either this re-adapt lands OR a pinned-bundle
allowance lands; both belong to v0.5.1.

The receipt to update with this finding is
`receipts/2026-09-15-v0.5.0-release.md`, in the "Not claimed" section,
adding the line:

> - Pinned Parakeet encoder island bundles (`island-pv`,
>   `island-attn-a-kt`) and the exported `ane-add-fp16-1x512`,
>   `ane-add-fp16-1x896`, `ane-mul-fp16-1x512` fixtures fail the strict
>   bundle loader at v0.5.0's worker; reproducing the pinned 104/104
>   E2E requires `mlx-omarchy` island-bundle-readapt (this slice) or
>   an additive pinned-bundle allowance in `bundle.cpp`. The 104/104
>   transcripts here were measured against the pre-fb4dfa86 worker
>   `762dd1de` and libane `1ab9d95d`.

## Not done

- Origin/main worker 104/104 E2E on jwm1: not run. Islands A/C remain
  refused by the strict gate; this slice does not weaken the gate and
  does not add a pinned-bundle allowance. The host-side check-bundle
  proof above is the strongest claim this slice can stand on without
  a Main-level decision.
- ANEC bytes: untouched for every bundle. Manifest re-stamping for the
  add fixtures was not attempted; the adapter would refuse them on the
  same walk overrun the gate refuses them on, so manifest-only re-adapt
  cannot recover them.
- `mlx-omarchy-info --check-bundle` for the pinned island bundles:
  not invoked here. The host-side C++ loader above reproduces the
  exact same verdict the gate would give.

resolved_model: this session (`zai/glm-5.3-flash`).
