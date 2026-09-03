#!/usr/bin/env python3
"""Trace collect_deep submit traffic: print each endpoint response body."""
import sys

sys.path.insert(0, "/home/joshuawarren/src/mlx-omarchy-deepcheck/scripts")
sys.argv = [
    "collect_deep.py",
    "--out", "/home/joshuawarren/benchq/deep-trace.tar.gz",
    "--submit", "https://mlx-omarchy-community-data.joshua-s-warren.workers.dev",
]

import collect_deep  # noqa: E402
import collect_submit  # noqa: E402

_orig_request = collect_submit._request


def traced_request(urlopen, req, timeout):
    status, body = _orig_request(urlopen, req, timeout)
    print(f"[trace] {req.get_method()} {req.get_full_url()} -> {status}",
          flush=True)
    if status != 200:
        print(f"[trace] body: {body[:1200]}", flush=True)
    return status, body


collect_submit._request = traced_request
collect_deep.main()
