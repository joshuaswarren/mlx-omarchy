# Block-rounded coopmat linear — reverted, slower (2026-09-16)

**Reverted, slower.** Merge `679a643a` is off origin/main. The block-rounded
fp16 linear was bit-identical (encoder_hidden pin `38c73261…`, 104/104,
transcript `db501a8c…`) but encoder wall went 12638.7 → 13310–13488 ms
(+5–7%, reproduced). The goal was faster with identical bytes; slower
does not land. Encoder path is again the `f735a675` batched linear plus
chain-reduce kernel (file bytes match `f20c634d`).

## Branch, wheel, provenance

- Branch `encoder-host-coopmat` tip `dad54151` (kernel `a90b58c2`; test
  fixes `5ea9d231`, `dad54151`). The `dad54151` test fix landed during this
  acceptance: `split()` keeps the split axis, so the per-block reference
  parts are `{m,1,16}` / `{n,1,16}` and the 2-axis transpose threw
  `[transpose] Recived 2 axes for array with 3 dimensions` — never green
  anywhere (x86 box has no Vulkan device, battery skips). Fix: reshape the
  parts to `{m,16}` / `{n,16}` before the `{1,0}` transpose; approved by
  the branch author and committed by them mid-acceptance.
- Wheel under test:
  `mlx_omarchy-0.32.2.dev202609160206+a90b58c2-cp314-cp314-linux_aarch64.whl`,
  sha256 `f2c985b7e39d35e7f02b1384c8d0a689aa97ecae7e704068b50fca05ce560254`.
  Test-only commits after `a90b58c2` do not affect wheel contents.
- Installed to `/var/tmp/ParakeetE2ECoopmat/site`;
  `scripts/mlx_provenance.py --expect-wheel <whl>` → **`verified: "match"`**
  (every run; r2/r3/r4/r5).
- `63c1d3cf` is not an object in this repository (`git cat-file -t` →
  fatal). The do-not-merge instruction is inert here; nothing named
  `63c1d3cf` was merged.

## Battery (omarchy_fast_ops_tests)

|build|cases|passed|failed|assertions|
|---|---:|---:|---:|---:|
|origin/main control `9b02870c`, same box, same day|35|25|10|1079751|
|branch `dad54151` (r4/r5, exclusive lock)|37|**27**|10|**1097497**|

- The 10 failures on both builds are the **same fused-rope test set**
  throwing `[omarchy] Vulkan timeline counter failed to advance for 10000
  ms (last observed=0, target=1)` — **pre-existing on origin/main**,
  reproduced 1:1 with the same-day control build. Not a branch regression;
  root cause owned elsewhere (jwm1 GPU investigation, active).
- The branch's contribution is +2 test cases (block-rounded), both green:
  **`block_rounded_matmul matches the per-block chain reference bit for
  bit` passes with 0 failed assertions** (+7746 assertions vs main).
- Earlier "contention" attributions, for the record: the 22:42 run died
  before testing (orphaned by session teardown); r1 (22:46) ran unlocked
  against TdtGpuLoop's exclusive GPU hold; r2 (22:53) ran against the
  wedged ANE (AneTmRecovery's provoked -110 had engaged the preserve path;
  recovered 0.2 ms post-swap, no reboot needed). r3's E2E was concurrent
  with an unlocked rope probe. r4/r5 are the clean, exclusive, healthy-ANE
  runs quoted above.

## E2E result (jwm1-linux, real pipeline — run r4b, exclusive lock)

Same fixture, golden capture, islands, worker and libane as the
2026-09-15 baseline. r4 (in-run E2E) and r4b (standalone rerun) agree on
every correctness field; r4b quoted, r4 in parentheses.

|quantity|value|
|---|---|
|status|**match**|
|emissions actual / native|**104 / 104**, matching_prefix_length 104|
|transcript|match, sha256 `db501a8c080380ea027ffa50a4b4956c39df77cb692c4fb78e556311a11a0790` (identical r3/r4/r4b)|
|encoder_hidden sha256|**`38c73261f29230276ed76f1fc017b76b024156d79218bd5f1347fdc7e7d43ec7`** — pin, byte-identical (r3/r4/r4b)|
|mel|bit-exact|
|ANE|islands ABC, 72 submits, 0 timeouts, exec 5032.9 ms (r4: 4975.0; baseline 5041.5)|

## Stage wall

|Stage|r4b ms|r4 ms|baseline `b5bf90e` ms|notes|
|---|---:|---:|---:|---|
|audio_load|169.3|152.4|187.4||
|mel_frontend|247.9|188.3|250.1|warm SPIR-V cache; mel bit-exact|
| encoder_ane | **13310.4** | 13487.7 | 12638.7 | **+5.3–6.7% vs baseline, reproduced** |
|decoder_load|99.4|115.4|99.1||
|tdt_decode|2853.9|2847.1|2880.3|warm compiled-cache|
|detokenize|12.4|12.1|12.3||
|**total_pipeline**|**16693.4**|—|16068.0||

The encoder delta is real (two clean runs within ~1.3% of each other, well
outside their spread) and unattributed between the coopmat left-linear
dispatch (`32783000`), the changed translator runner (branch
`vulkan_encoder.py`, not baseline `6c175adf`), and same-day device state.
ANE exec matches the long-standing ~5.0 s attribution, so the delta is in
the Vulkan dispatch portion. Correctness is unaffected; flagging for the
wall-attribution owner.

## Identity

- Host `jwm1-linux`, aarch64, kernel `7.1.6-1-1-ARCH`, Python 3.14.7,
  `/dev/accel/accel0`, device Apple M1 (G13G B1), Vulkan Mesa Honeykrisp.
- `MLX_OMARCHY_SPIRV_CACHE=/var/tmp/MelFrontendPerf/spirv-ab.3KDGHZ`;
  `PYTHONPATH=$SITE:/var/tmp/MelFrontendPerf/venv-cache/lib/python3.14/site-packages`.
- Worker `762dd1de…` (`jwm1-select-island-exec2`), libane `1ab9d95d…`,
  bundles `island-attn-a-kt` / `island-select-8head-scratch417` /
  `island-pv`; ANE module `b52064c` (tm-recovery build).
- Encoder runner: branch `overlay/tools/coreml/vulkan_encoder.py`, staged
  copy in this receipt dir. e2e runner `/var/tmp/ParakeetE2E/parakeet_e2e.py`,
  command in `run-e2e.sh` in this receipt dir.
- Locks: `/tmp/m1-gpu.lock` + `/run/lock/mlx-omarchy-ane/device.lock`,
  `flock -w 900`, never stolen, never unlinked; free after each run.

## Artifacts

`receipts/2026-09-15-encoder-host-coopmat/`: `e2e-report.json` (r4b),
`transcript.txt` (`db501a8c…`), `token_ids.json`, `encoder_hidden.npy`
(`38c73261…` pin), `mel.npy`, `vulkan_encoder.py`, `run-e2e.sh`.
Run logs: `/var/tmp/ParakeetE2ECoopmat/` (`fast-ops-full.log` r4/r5,
`e2e.log`, `provenance.txt`, `out/` r4, `out-r4b/`), main control battery
`/var/tmp/mainctl-battery.log`.

## Not claimed

- No root cause for the pre-existing fused-rope Vulkan timeline hangs
  (reproduced on origin/main control; owned by the jwm1 GPU investigation).
- No attribution of the +5.3–6.7% encoder wall delta to a specific commit;
  the three candidates are listed above.
- No cold-SPIR-V mel run; no claim on Phase 9 latency, macOS, other
  fixtures, or batching/resident-worker improvements.
