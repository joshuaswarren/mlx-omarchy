#!/usr/bin/env bash
set -euo pipefail
printf 'smoke_workload=true pid=%s pgid=%s\n' "$$" "$(ps -o pgid= -p $$ | tr -d ' ')"
