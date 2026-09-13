# Invalid attempt 2: scanner self-match

This attempt is not performance evidence. The window acquired `/tmp/m1-gpu.lock` at `2026-09-13T04:22:48Z`, completed identity checks, then exited before decode because the Python contender classifier interpreted the MLX worktree path in its own interpreter command as accelerator work.

The launcher used PGID `3300655`; `window.sh` ran as PID `3300657`. `window.log` records `CHILD_PIDS_CLEAR`, lock release at `2026-09-13T04:22:48Z`, and `rc=4`.

The initial synthetic correction passed, but the subsequent live local scan exited 1 after also matching two unrelated `worker_watch.py` processes whose `--config` arguments contained `mlx`. A message that incorrectly said the live scan passed was corrected immediately. The final classifier uses the Python entry point or `-m` module rather than arbitrary argument text.
