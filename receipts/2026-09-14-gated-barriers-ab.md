# jwm1 gated-barriers decode A/B

Date: 2026-09-14

## Verdict

Do not land. `MLX_OMARCHY_GATED_BARRIERS` stays default off.

Five paired off/on repetitions of Qwen2.5-0.5B-Instruct-4bit greedy decode on `jwm1-linux` held both pinned generated-ID digests. Median decode tok/s did not rise on either the 30-token or the 1053-token leg. Prefill medians were also flat.

This is the named discriminating experiment from `receipts/2026-09-14-decode-gap-attribution.md` (dependency-gated barriers, no rebuild). It does not resolve term A's host/GPU split: dropping the post-dispatch barrier and gating the pre-barrier did not move the wall.

## Identity

- Receipt checkout: `7f8786b02f88d56cb98f5779aea7b9ab4a715972` (`v0.4.2-467-g7f8786b0-dirty`; dirty from unrelated sibling work, not used for the measurement).
- Measured source / wheel: `b41e2b74c330f910b24cab0e7516e306527858f0`, `mlx-omarchy==0.32.2.dev202609122355+b41e2b74`.
- Wheel SHA-256: `81743cd1a631f6c5d8aa7ff9538d1ec4ddcb2a57c184da7c55345e48d349a240`.
- Python: `3.14.7`; mlx-lm: `0.31.3`.
- Host: `jwm1-linux`; `aarch64`; `nproc=8`; Apple M1 T8103, `Apple M1 (G13G B1)`.
- Runtime device: `Device(gpu, 0)`.
- Agent model: `xai-oauth/grok-4.6`; routing fallback: true (requested OpenAI). Numbers are from the locked `bench_decode` subprocesses, not from the agent.
- Model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit` snapshot at `/home/joshuawarren/models/Qwen2.5-0.5B-Instruct-4bit-mlx`.
- `scripts/bench_decode.py` SHA-256: `f5062d88f34b0845c1e59b0b35d2e33ae02180f636a0f65c543a4366ec2bef7f`.
- `scripts/bench_matrix.json` SHA-256: `df8eb9f3ed84182604379b7cde070ade706bfe590fbf83a35b1877cfaf43f258`.
- Runner SHA-256: `08158ce9330980d557444839caf934cbbabcbff8dd4febfc967dea149d68859b`.
- Raw capture SHA-256: `efaf00e11ddcbb4f9149534bbc63914be3ff37724e35e318c20d125d1801620b`.

Provenance on every leg (`verified=match`; harness=unknown because the hashed scripts ran from `/tmp/gated-barriers-ab/scripts`, not a git tree):

```text
provenance: mlx-omarchy 0.32.2.dev202609122355+b41e2b74 mx=0.32.2.dev202609122355+b41e2b74 verified=match harness=unknown core.cpython-314-aarch64-linux-gnu.so=sha256:4ad850da16c300ea libmlx.so=sha256:03b3f4b9024927f9
```

`libmlx.so` and the core extension hashes match `receipts/2026-09-14-jwm1-gpu-parity-rerun.md`.

Power during the locked run and after release:

```text
/sys/class/power_supply/macsmc-ac/online=1
/sys/class/power_supply/tps6598x-source-psy-0-0038/online=1
/sys/class/power_supply/tps6598x-source-psy-0-003f/online=0
/sys/class/power_supply/macsmc-battery/status=Full
```

## Protocol

Same 30/32 and 1053/32 Q4 decode protocol as `receipts/2026-09-14-jwm1-gpu-parity-rerun.md`, with an explicit `MLX_OMARCHY_GATED_BARRIERS=0` vs `=1` pair. Interpreter:

```text
/home/joshuawarren/src/mlx-bf16-grouped-base-b41e2b74/.work/venv-run/bin/python
```

One outer lock for all 20 legs:

```sh
timeout -k 10s 900s flock -w 60 /tmp/m1-gpu.lock \
  /home/joshuawarren/src/mlx-bf16-grouped-base-b41e2b74/.work/venv-run/bin/python \
  /tmp/gated-barriers-ab/run.py
```

Each decode leg was a fresh subprocess with:

```text
MLX_DISABLE_COMPILE=1
HF_HUB_OFFLINE=1
MLX_OMARCHY_GATED_BARRIERS=0|1
--tokens 32 --temp 0.0 --seed 0 --warmup-tokens 4
--wheel .../mlx_omarchy-0.32.2.dev202609122355+b41e2b74-cp314-cp314-linux_aarch64.whl
```

Five paired repetitions. Order inside each rep: short-off, short-on, ctx1053-off, ctx1053-on. `bench_matrix.prompt_text` expanded `short` (2 UTF-8 bytes) and `ctx1024` (4759 UTF-8 bytes). `bench_decode.py` asserted 30 and 1053 chat-template tokens. Decode rate is over 31 inter-token gaps with EOS suppressed. Median is the middle of the five sorted rates.

No diagnostics profile, rebuild, overlay edit, ANE unload, reboot, 1x896, or SET write. `63c1d3cf` was not merged.

## Results

Median of 5. A positive delta means gated-on is faster.

| Metric | off (default) | on (`GATED_BARRIERS=1`) | delta | pin |
| --- | ---: | ---: | ---: | --- |
| Q4 short decode, 30/32 | 111.4096 tok/s | 111.3977 tok/s | -0.0119 | `7fd25a869ff21678` |
| Q4 short prefill, 30 tokens | 330.8835 tok/s | 329.767 tok/s | -1.1165 | |
| Q4 1K-context decode, 1053/32 | 96.2628 tok/s | 96.1491 tok/s | -0.1137 | `7da83f06ec9f001d` |
| Q4 1K-context prefill, 1053 tokens | 1113.4323 tok/s | 1113.0488 tok/s | -0.3835 | |

All 20 generated-ID digests matched their pins. IDs:

```text
short:   7fd25a869ff21678, n=32, first=9707,0,2585 last=646,387,7881
1053:    7da83f06ec9f001d, n=32, first=13060,498,369 last=3897,553,279
```

Raw decode tok/s:

```text
short off: 107.0367 111.4249 111.4096 110.8855 111.4510
short on:  106.3875 111.8994 111.5840 111.3977 110.0660
1053 off:  95.7481  98.0143  96.4359  93.9773  96.2628
1053 on:   96.9599  96.1491  96.6764  96.0647  95.6340
```

Paired on-minus-off decode deltas mix sign on both legs. The gated-on movement is inside the same-wheel rerun noise already seen between `receipts/2026-09-14-jwm1-gpu-parity-refresh.md` (109.4675 / 96.2046) and `receipts/2026-09-14-jwm1-gpu-parity-rerun.md` (107.285 / 95.0352).

Wall clock for the locked script was 57 s (`started_unix=1789414922`, `finished_unix=1789414979`).

## Land decision

`gated_wins_both_decode` is false. Default remains off in `CommandEncoder::gated_barriers()` (`overlay/mlx/backend/omarchy/encoder.cpp`). `docs/install-omarchy.md` still records the mode as defaults off. No source change.

The 2026-09-08 screen (`receipts/2026-09-08-gated-barrier-screen/verdict.json`) reached the same default-off decision at ~41 tok/s ctx1024. This receipt repeats it on the current 95 tok/s 1053/32 wheel.

## Lock and hardware safety

- `/tmp/m1-gpu.lock` inode 29 held exclusively; nested `flock -n` returned 1 while held and at end-still-held.
- Lock was not unlinked. After release: `flock -n` free, `fuser` empty.
- No leftover decode/profile/`deqp` process.
- `ane.ko` was not touched. No ANE command, unload, reboot, 1x896, or SET write.
- Grouped-base `dist/` wheel and `.work/venv-run` were not modified.

## Artifacts

`receipts/2026-09-14-gated-barriers-ab/`:

| file | SHA-256 |
| --- | --- |
| `run.py` | `08158ce9330980d557444839caf934cbbabcbff8dd4febfc967dea149d68859b` |
| `results.json` | `efaf00e11ddcbb4f9149534bbc63914be3ff37724e35e318c20d125d1801620b` |
