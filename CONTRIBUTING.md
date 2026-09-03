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
and uploads nothing until you type `SUBMIT` or pass `--submit`. The
default run uploads nothing; you always see the full preview first.

Clone this repository first; both scripts live in `scripts/`:

```bash
git clone https://github.com/joshuaswarren/mlx-omarchy.git
cd mlx-omarchy
```

Two paths exist. They need different setups.

### Path 1: quick capability report, zero install

One command. It runs on your system `python3`. It needs no install, no
build, no mlx-omarchy wheel, and downloads no models:

```bash
python3 scripts/collect_quick.py

# or print it and publish it in one command:
python3 scripts/collect_quick.py \
  --submit https://mlx-omarchy-community-data.joshua-s-warren.workers.dev
```

The command prints one JSON report: CPU and memory, kernel, Apple
Silicon model, the Mesa Honeykrisp and Vulkan stack, ANE visibility, and
the installed mlx-omarchy wheel with its capability dump. The JSON is
small and carries no personal data; paste it wherever you discuss the
project.

### Path 2: full report, needs the released wheel

`scripts/collect_deep.py` runs five sections: `quick`, `environment`,
`correctness`, `benchmark`, and `profile`. The `correctness` and
`benchmark` sections import `mlx`, so they report `available: false`
when no wheel is installed for the interpreter that runs the collector.

Prerequisites, in order:

1. Python 3.14. The aarch64 wheel asset is `cp314` only, so Python 3.13
   refuses it with `is not a supported wheel on this platform`. Check
   yours with `python3 --version`; if it is not 3.14.x, install a 3.14
   interpreter and use its name in the commands below.
2. The aarch64 wheel asset from a stable release at
   [github.com/joshuaswarren/mlx-omarchy/releases](https://github.com/joshuaswarren/mlx-omarchy/releases).
   The name pattern is `mlx_omarchy-*-cp314-cp314-linux_aarch64.whl`;
   v0.3.2 ships
   `mlx_omarchy-0.32.2.dev202609030512-cp314-cp314-linux_aarch64.whl`.
   Use a stable wheel, not the diagnostics prerelease below; the
   prerelease slows every dispatch on purpose, so its benchmark
   numbers are not comparable.
3. Install that wheel into one interpreter, and always run the
   collector with that same interpreter. A venv pins the pair:

   ```bash
   python3.14 -m venv ~/.venvs/mlx-collect
   ~/.venvs/mlx-collect/bin/pip install \
     https://github.com/joshuaswarren/mlx-omarchy/releases/download/v0.3.2/mlx_omarchy-0.32.2.dev202609030512-cp314-cp314-linux_aarch64.whl
   ```

   That URL pins v0.3.2. To take whatever the newest stable release
   ships instead:

   ```bash
   gh release download --repo joshuaswarren/mlx-omarchy \
     --pattern '*cp314*linux_aarch64.whl'
   ~/.venvs/mlx-collect/bin/pip install ./mlx_omarchy-*-cp314-cp314-linux_aarch64.whl
   ```

4. Run the collector with that interpreter:

   ```bash
   # preview only, writes nothing:
   ~/.venvs/mlx-collect/bin/python scripts/collect_deep.py

   # archive plus paste-ready .submission.md cover text:
   ~/.venvs/mlx-collect/bin/python scripts/collect_deep.py \
     --out mlx-omarchy-deep.tar.gz

   # publish in the same command (--submit requires --out):
   ~/.venvs/mlx-collect/bin/python scripts/collect_deep.py \
     --out mlx-omarchy-deep.tar.gz \
     --submit https://mlx-omarchy-community-data.joshua-s-warren.workers.dev
   ```

The archive is deterministic: the same workspace produces the same bytes
as the previewed manifest, and each member carries a SHA-256 there. A
missing precondition, such as an unbuilt benchmark binary, is recorded
as `available: false` instead of failing the run, and a timeout keeps
the sections that finished. The run also writes a paste-ready
`mlx-omarchy-deep.submission.md` cover text. No section downloads a
model; the probes generate their own data.

What each section contributes:

- `quick`: the Path 1 capability report. Needs no wheel.
- `environment`: Python, kernel, and tool versions. Needs no wheel.
- `correctness`: value probes of the installed wheel's operations.
  Needs the wheel.
- `benchmark`: the matmul and attention sweep. Works on the released
  wheel, and its numbers ride in the published summary, so results are
  comparable across machines without downloading anyone's archive.
- `profile`: the GPU dispatch profile. Reports `available: false` on a
  released install; see the profile section below.

The GPU profile section needs more than the wheel, and it reports
`available: false` on a released install. Release wheels are compiled
with the profiling harness OFF on purpose, so the dispatch path
carries zero profiling code. Two ways to turn it on:

Easiest: install the dev diagnostics prerelease from GitHub Releases
at <https://github.com/joshuaswarren/mlx-omarchy/releases/tag/v0.3.3-diag.1>.
It carries the profiling harness compiled IN and the `mlx-omarchy-info`
tool. The harness slows every dispatch down, so use this build to
collect profiles only; it is not for production work. Attach the dev
wheel to a venv:

```bash
python3.14 -m venv ~/venv-mlx-diag
~/venv-mlx-diag/bin/pip install \
  'https://github.com/joshuaswarren/mlx-omarchy/releases/download/v0.3.3-diag.1/mlx_omarchy-0.32.2.dev202609031348%2Bdiag-cp314-cp314-linux_aarch64.whl'
```

Set the env var to a file path and run your workload. The runtime
appends one NDJSON event stream to that path for the life of the
process:

```bash
export MLX_OMARCHY_GPU_PROFILE=$HOME/mlx-profile.jsonl
python3 your_workload.py
```

Or build from source with the harness on:

```bash
cmake -B build -DMLX_OMARCHY_GPU_PROFILING=ON   # plus your usual flags
```

Then run the deep collector against that venv or build. Hardware,
kernel, Mesa, Vulkan, ANE visibility, correctness probes and the
benchmark sweep all work from the plain wheel, so a wheel-only report
is still useful.

To publish a result:

```bash
export MLX_OMARCHY_SUBMIT_URL=https://mlx-omarchy-community-data.joshua-s-warren.workers.dev
~/.venvs/mlx-collect/bin/python scripts/collect_deep.py --out mlx-omarchy-deep.tar.gz
# type SUBMIT at the prompt, or pass --submit "$MLX_OMARCHY_SUBMIT_URL"
```

Only the already-redacted archive is sent. The endpoint deduplicates by
content SHA-256 and answers with a public receipt URL, which the script
prints. Keep that URL: it is the stable public link to your result. A
failed upload keeps the local archive and submission file.

### Troubleshooting

- `pip install` fails with `is not a supported wheel on this platform`:
  the interpreter is not Python 3.14. The aarch64 wheel is `cp314`
  only. Run the install with a 3.14 interpreter.
- The report shows `"mlx": {"available": false}`, or `correctness` and
  `benchmark` both say `available: false`: the collector ran under a
  Python without the mlx-omarchy wheel. Test the exact interpreter you
  invoke with `.../bin/python -c "import mlx"`; a wheel installed into
  a different `python3` does not count.
- `profile` says `available: false` on a released wheel: expected, not
  a fault. Release wheels compile the profiling harness off and do not
  ship `mlx-omarchy-info`; see the profile section above.
- The upload fails with a route error: you passed the API route in the
  URL. `--submit` and `MLX_OMARCHY_SUBMIT_URL` take the origin only,
  for example
  `https://mlx-omarchy-community-data.joshua-s-warren.workers.dev`.
  The collector appends `/v1/submit/<sha256>` itself.

### Check your core count before you benchmark

A machine that silently runs fewer cores than it has produces
misleading host-bound numbers. One line shows your situation:

```bash
nproc && cat /sys/devices/system/cpu/present
```

If `nproc` prints fewer cores than the `present` range covers, fix the
boot before you benchmark. The collector records the same facts in
every report: `host.cpu_online` counts the cores your process may use,
`host.cpu.present` and `host.cpu.online` come from sysfs, and
`host.core_shortfall` is set only when fewer cores are online than
present and no `maxcpus=` or `nosmp` boot clamp explains the gap.

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
