# Contributing

`mlx-omarchy` still needs more MLX features for real models.
You can help with small backend changes and M1 tests that others can repeat.

## Before you start

1. Read `docs/architecture.md` and `docs/roadmap.md`.
2. Check `docs/compatibility.md` for the current gate.
3. Run `scripts/prepare-mlx.sh` to create `.work/mlx`.
4. Open an issue before you change a public API, driver ABI, or bundle format.
5. Keep private Apple software, firmware, keys, and model weights out of the repository.

## Source rules

- Add new Omarchy files under `overlay/`.
- Put changes to current MLX files in `patches/`.
- Do not copy the MLX source into this repository.
- Update `mlx.lock` only with an exact commit and checked archive hash.
- Keep Omarchy changes behind `MLX_BUILD_OMARCHY`.

## Code rules

- Keep the public `mlx` import and `mx.gpu` device.
- Use Vulkan for all tensor work.
- Keep ANE choice inside the GPU evaluator.
- Never add a silent CPU tensor fallback.
- Reuse shared MLX code before you add Omarchy-only code.
- Keep model rules out of core operations.

## Proof required

Name the changed contract in each pull request.
Add the smallest test that proves the change.

For a hardware result, record these facts:

- source commit
- kernel
- Mesa
- device
- firmware
- command
- model hash
- output
- dispatch trace

For a speed result, also record these facts:

- warmup
- repeat count
- memory use
- temperature
- test tool and settings

Do not lower a tolerance or remove work to make a check pass.
Do not claim M1 results for a different Apple Silicon model.

## Community hardware data

You can share a hardware result without write access and without a pull
request. The collector runs on your machine, redacts before it writes,
and uploads nothing until you type `SUBMIT`.

Both scripts run on Python 3 alone, need no build, and download no
models. The default run uploads nothing; you always see the full preview
first.

Report a machine in minutes:

```bash
python3 scripts/collect_quick.py
```

The command prints one JSON report: CPU and memory, kernel, Apple
Silicon model, the Mesa Honeykrisp and Vulkan stack, ANE visibility, and
the installed mlx-omarchy wheel with its capability dump. The JSON is
small and carries no personal data; paste it wherever you discuss the
project.

Report a performance or correctness problem in depth:

```bash
python3 scripts/collect_deep.py                  # preview only, writes nothing
python3 scripts/collect_deep.py --out mlx-omarchy-deep.tar.gz
```

The archive is deterministic: the same workspace produces the same bytes
as the previewed manifest, and each member carries a SHA-256 there. A
missing precondition, such as an unbuilt benchmark binary, is recorded
as `available: false` instead of failing the run, and a timeout keeps
the sections that finished. The run also writes a paste-ready
`mlx-omarchy-deep.submission.md` cover text.

To publish a result:

```bash
export MLX_OMARCHY_SUBMIT_URL=https://mlx-omarchy-community-data.joshua-s-warren.workers.dev/v1/submit
python3 scripts/collect_deep.py --out mlx-omarchy-deep.tar.gz
# type SUBMIT at the prompt, or pass --submit "$MLX_OMARCHY_SUBMIT_URL"
```

Only the already-redacted archive is sent. The endpoint deduplicates by
content SHA-256 and answers with a public receipt URL, which the script
prints. Keep that URL: it is the stable public link to your result. A
failed upload keeps the local archive and submission file.

Privacy rules that hold for every submission:

- Only redacted, allowlisted fields leave your machine: host model,
  architecture, kernel release, GPU and driver names, package versions,
  probe results, and benchmark numbers.
- The redactor removes user names, host names, home paths, IP and MAC
  addresses, serial numbers, UUIDs, and credential-shaped strings, and
  the manifest counts every replacement it made.
- The scripts never read the devicetree serial number, machine IDs, or
  network configuration, and collection never touches the network;
  upload code lives in one separate module and runs only after you
  confirm.
- Raw command output is capped and passes through the redactor; the
  archive never carries an unredacted log.
- Upload is never automatic. A failed upload keeps your local files.
- A submission never opens a pull request, issue, or discussion. The
  dataset mirror commits snapshots as `github-actions[bot]`, so
  submitters never appear in the contributor graph.

Results land on the public read API and in a mirrored snapshot on the
`community-data` branch. Coding agents working in this repository find
query commands under "Community hardware data" in `AGENTS.md`.

## Driver changes

Test early driver work in [`ane-linux-experiments`](https://github.com/joshuaswarren/ane-linux-experiments).
Send stable DRM and `libane` changes to [`eiln/ane`](https://github.com/eiln/ane).
Do not copy a private driver fork into this repository.

## License

Contributions use the repository's MIT license.
Keep upstream copyright notices in patch context and prepared source.
