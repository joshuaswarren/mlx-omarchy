#!/usr/bin/env python3
"""Tests the canonical-channel integrity check inside h13_package_to_bundle.py."""
import re
import sys

ADAPTER_PATH = "/overlay/tools/ane-export/h13_package_to_bundle.py"
src = open(ADAPTER_PATH).read()

# Anchor on the outer loop and the inner for-direction block
m = re.search(
    r"for program_index, program in enumerate\(converted_programs\):"
    r"(.*?)\n    for name, role in tensor_roles",
    src, re.DOTALL,
)
assert m, "could not find canonical-check region"
body = m.group(1)
# Trim the role-direction check that we don't want to test.
role_idx = body.find('        for direction, bindings in (("inputs"')
integrity_text = body[:role_idx].rstrip()

# Promote indented body (4 spaces) to a function body (8 spaces)
integrity_text2 = integrity_text.replace(
    "programs[{program_index}]", "programs[0]"
)
function_text = (
    "def check_canonical(program, fail):\n"
    + integrity_text2.replace("\n", "\n    ")
)
exec(function_text, globals())

tests_run = 0
tests_pass = 0

def program(inputs_channels, outputs_channels):
    return {
        "inputs": [{"channel": c} for c in inputs_channels],
        "outputs": [{"channel": c} for c in outputs_channels],
    }

class _FailCalled(Exception):
    pass

def expect_pass(label, **program_kwargs):
    global tests_run, tests_pass
    tests_run += 1
    def fail(m):
        raise _FailCalled(m)
    try:
        check_canonical(program_kwargs, fail)
        tests_pass += 1
        print(f"[PASS] {label}")
    except _FailCalled as e:
        print(f"[FAIL] {label}: unexpected fail: {e}")
    except Exception as e:
        print(f"[ERROR] {label}: {e}")

def expect_fail(label, **program_kwargs):
    global tests_run, tests_pass
    tests_run += 1
    def fail(m):
        raise _FailCalled(m)
    try:
        check_canonical(program_kwargs, fail)
        print(f"[FAIL] {label}: should have failed but didn't")
    except _FailCalled as e:
        tests_pass += 1
        print(f"[PASS] {label}: correctly failed with {e}")

# 1. Input-first single (the new schema from db8ffba + f122644)
expect_pass("input-first single in/out", **program([4], [5]))
# 2. Output-first single (the old schema from 2d11b2a and earlier)
expect_pass("output-first single in/out", **program([5], [4]))
# 3. Multi-i/o (any order)
expect_pass("multi i/o", **program([4, 5, 6], [7]))
# 4. Duplicate channel across inputs and outputs is REJECTED
expect_fail("input/output channel collision", **program([4], [4]))
# 5. Duplicate channel within outputs is REJECTED
expect_fail("duplicate output channel", **program([4], [5, 5]))
# 6. Duplicate channel within inputs is REJECTED
expect_fail("duplicate input channel", **program([4, 4], [5]))
# 7. Out-of-range channel is REJECTED
expect_fail("channel below 4", **program([3], [5]))
expect_fail("channel at 32", **program([4], [32]))

print(f"\n{tests_pass}/{tests_run} passed")
sys.exit(0 if tests_pass == tests_run else 1)