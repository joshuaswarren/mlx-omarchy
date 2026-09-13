# Native macOS collector smoke

Source: `e1ca0b61f436093155f2e8f55bc962c7396ef9c6`. The worktree was clean for both runs.

A MacBook Air M2 (Mac14,2), 24 GiB RAM, macOS 26.6.2, and Python 3.14.7
completed the existing quick and deep collection paths with native MLX 0.32.1.
Both installations passed all six correctness checks and completed all three
matrix sizes. [results.json](results.json) contains commands, binary hashes,
measurements, power and process context, archive hashes, and validation results.
The two archives contain the full generated reports.

- Homebrew: fingerprints recorded as `unverified` because RECORD is absent.
- Clean pip venv: the loaded extension and `libmlx.dylib` matched the RECORDs
  from `mlx` and `mlx-metal`. Provenance reports `match`.
- Empty venv: quick collection retained hardware facts. Both deep MLX sections
  reported `available: false`.
- Both archives: member hashes matched. The service schema accepted each
  derived summary. Its PII scanner found no matches across each summary and
  eight archive members. A separate scan found no local user name, host name,
  or home path in any member.

These are collector smoke results. The Mac was on battery with one known
model process present. Other GPU activity and temperature were not measured.
There is no dispatch trace, Linux compatibility proof, ANE result, or
performance parity claim.

Offline tests passed: 73 existing collector tests, 14 macOS tests, three
provenance tests, and the benchmark-matrix self-test. The same checks passed
on [Linux and macOS CI](https://github.com/fgilio/mlx-omarchy/actions/runs/34776275784)
with Python 3.11 and 3.14. No backend code changed or backend build was tested.

To reproduce the pip setup:

```bash
python3 -m venv /tmp/mlx-macos-pip-test
/tmp/mlx-macos-pip-test/bin/python -m pip install mlx==0.32.1
/tmp/mlx-macos-pip-test/bin/python scripts/collect_deep.py \
  --out /tmp/mlx-macos-pr-final-pip.tar.gz \
  --workspace /tmp/mlx-macos-pr-final-pip
```

Use a new workspace for a new run. To check the absent-package path:

```bash
python3 -m venv --without-pip /tmp/mlx-empty-collector-test
/tmp/mlx-empty-collector-test/bin/python scripts/collect_quick.py
/tmp/mlx-empty-collector-test/bin/python scripts/collect_deep.py
```
