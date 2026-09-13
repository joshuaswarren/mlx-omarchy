# Invalid attempt 1: transfer-name mismatch

This attempt is not performance evidence. The bounded launcher used PGID `3299738`; `window.sh` acquired `/tmp/m1-gpu.lock` as PID `3299740` at `2026-09-13T04:18:27Z`, then failed its preflight because `/tmp/LongContextCostAttribution-contenders.py` was absent. The files had been copied under short local basenames.

`window.log` and `launcher.log` record `CHILD_PIDS_CLEAR`, lock release at `2026-09-13T04:18:27Z`, and `rc=2`. No identity or decode measurement ran.
