# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Hardware-run stub: only the symbols inference_driver imports."""

ANE_COUNTER_FIELDS = (
    "ane_models_loaded",
    "ane_packages_compiled",
    "ane_package_cache_hits",
    "ane_worker_starts",
    "ane_submissions",
    "ane_timeouts",
    "ane_input_bytes",
    "ane_output_bytes",
    "ane_exec_ns",
)


def evaluate_invariants(report: dict) -> list[str]:
    violations = []
    for field in ANE_COUNTER_FIELDS:
        if field not in report:
            violations.append(f"missing counter '{field}'")
    if report.get("ane_timeouts", 0) != 0:
        violations.append(
            f"ane_timeouts={report['ane_timeouts']} (must be 0 on a "
            "clean parity run)"
        )
    return violations
