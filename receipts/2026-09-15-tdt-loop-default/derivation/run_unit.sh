#!/bin/bash
# Host-routing unit tests (no GPU required) against the deployed branch tree.
set -e
cd /var/tmp/TdtLoopDefault/overlay/tests/omarchy/coreml
~/venv-agxgen/bin/python test_tdt_decode_path.py 2>&1
~/venv-agxgen/bin/python test_tdt_control.py 2>&1
