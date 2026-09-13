# Invalid attempt 0: false contender classification

This attempt is not performance evidence. The window acquired `/tmp/m1-gpu.lock` at `2026-09-13T03:51:24Z`, verified the pinned `6b1ac029` tree and wheel, then exited before any decode measurement.

The original `pgrep -f` guard matched PID `383344`, a dormant Bash monitor whose quoted command contained an old MLX probe path. `window.log` records the false match, `CHILD_PIDS_CLEAR`, and lock release at `2026-09-13T03:51:24Z` with `rc=4`. The corrected `contenders.py` classifier excludes shell monitors and retains exact service and MLX Python-job detection.
