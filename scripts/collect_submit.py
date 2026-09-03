#!/usr/bin/env python3
"""Upload path for the mlx-omarchy deep diagnostics archive.

The only network code in the collectors lives here. `collect_quick.py`
and `collect_deep.py` never import this module at collection time; the
deep script imports it lazily, only after the user has seen the preview
and asked to submit explicitly.

Protocol (boring HTTP JSON):

  GET  <endpoint>/<archive sha256>   dedup probe
       200 -> {"url": ...}           already present; nothing is sent
       404 -> continue
  POST <endpoint>                    body: the redacted .tar.gz bytes
       headers: Content-Sha256, Content-Type
       200/201 -> {"url": ...}       the public receipt URL

Only the already-redacted archive and its hashes are sent. The endpoint
is caller-supplied (--submit URL or MLX_OMARCHY_SUBMIT_URL); there are no
embedded endpoints and no embedded secrets. If a deployment needs a
bearer token, the caller sets MLX_OMARCHY_SUBMIT_TOKEN in its own
environment; the token is never written to disk or logged.

Every failure raises SubmitError. The caller keeps the local archive and
submission file and prints the error; a failed submit never destroys
local output.
"""

import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_TIMEOUT = 60


class SubmitError(RuntimeError):
    pass


def _request(opener, req, timeout):
    try:
        with opener.open(req, timeout=timeout) as resp:
            body = resp.read()
            return resp.status, body
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise SubmitError(f"endpoint unreachable: {exc}") from exc


def submit(endpoint, archive_bytes, timeout=DEFAULT_TIMEOUT, urlopen=None,
           token=None):
    """Submit one archive; returns {"url", "deduplicated", "status"}.

    `urlopen` is injectable for tests. The default opener is urllib's.
    """
    if urlopen is None:
        opener = urllib.request.build_opener()
        urlopen = opener.open
    digest = hashlib.sha256(archive_bytes).hexdigest()
    base = endpoint.rstrip("/")
    headers = {
        "Content-Type": "application/octet-stream",
        "Content-Sha256": digest,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    status, body = _request(
        urlopen, urllib.request.Request(f"{base}/{digest}", headers=headers),
        timeout)
    if status == 200:
        return _receipt(body, deduplicated=True, status=status)
    if status != 404:
        raise SubmitError(f"dedup probe failed with HTTP {status}")

    status, body = _request(
        urlopen,
        urllib.request.Request(base, data=archive_bytes, headers=headers,
                               method="POST"),
        timeout)
    if status not in (200, 201):
        raise SubmitError(f"upload failed with HTTP {status}")
    return _receipt(body, deduplicated=False, status=status)


def _receipt(body, deduplicated, status):
    try:
        payload = json.loads(body.decode("utf-8"))
        url = payload["url"]
    except (ValueError, KeyError, UnicodeDecodeError) as exc:
        raise SubmitError(f"endpoint returned no receipt URL: {exc}") from exc
    return {"url": url, "deduplicated": deduplicated, "status": status}


def endpoint_from_args(args):
    """Resolve the endpoint: --submit flag first, then the environment."""
    if getattr(args, "submit", None):
        return args.submit
    return os.environ.get("MLX_OMARCHY_SUBMIT_URL") or None


def token_from_env():
    return os.environ.get("MLX_OMARCHY_SUBMIT_TOKEN") or None


def sys_stdin_isatty():
    return os.environ.get("MLX_OMARCHY_ASSUME_YES") != "1" and \
        sys.stdin.isatty()


def confirm_interactive(endpoint, archive_name, digest):
    """Ask for the exact word SUBMIT on a terminal; anything else declines."""
    if not sys_stdin_isatty():
        return False
    print(f"[submit] endpoint: {endpoint}")
    print(f"[submit] archive: {archive_name} sha256={digest}")
    print("[submit] only the redacted archive is sent. "
          "Type SUBMIT to upload, anything else to keep it local:")
    try:
        answer = input().strip()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer == "SUBMIT"
