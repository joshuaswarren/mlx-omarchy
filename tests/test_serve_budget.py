"""Serve budget: estimate math, context admission, reservations, fit check."""

import contextlib
import unittest.mock
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_serve import budget  # noqa: E402

GiB = 1024**3


def meminfo(available_gib: float) -> str:
    kb = int(available_gib * GiB / 1024)
    return f"MemTotal:       999999999 kB\nMemAvailable:   {kb} kB\n"


def memory(weights_gib: float, kv_per_tok=None, peak=None):
    return {
        "weights_bytes": int(weights_gib * GiB),
        "kv_bytes_per_token": kv_per_tok,
        "peak_estimate_bytes": int(peak * GiB) if peak is not None else None,
    }


class EstimateTests(unittest.TestCase):
    def test_full_formula_with_known_kv(self):
        est = budget.estimate_required(memory(16.0, kv_per_tok=256 * 1024), 4096)
        kv = 256 * 1024 * 4096
        self.assertEqual(est.kv, kv)
        self.assertTrue(est.kv_known)
        self.assertEqual(est.workspace, max(int(0.25 * 16 * GiB), GiB))
        self.assertEqual(est.total, int(16 * GiB) + kv + est.workspace)

    def test_unknown_kv_adds_flat_margin(self):
        est = budget.estimate_required(memory(16.0), 4096)
        self.assertFalse(est.kv_known)
        self.assertEqual(est.kv, 0)
        expected = max(int(0.25 * 16 * GiB), GiB) + budget.UNKNOWN_KV_MARGIN
        self.assertEqual(est.workspace, expected)

    def test_peak_estimate_overrides(self):
        est = budget.estimate_required(memory(16.0, peak=20.5), 8192)
        self.assertTrue(est.peak_override)
        self.assertEqual(est.total, int(20.5 * GiB))

    def test_peak_never_hides_context_growth(self):
        # A peak measured at a small context must not cap the estimate when
        # the requested context is large: total >= weights + kv always.
        est = budget.estimate_required(memory(1.0, kv_per_tok=1_000_000, peak=2 * GiB),
                                       1_000_000)
        self.assertGreaterEqual(est.total, int(1 * GiB) + 1_000_000 * 1_000_000)
        self.assertGreater(est.total, 2 * GiB)

    def test_meminfo_nonpositive_and_empty_rejected(self):
        with self.assertRaises(budget.BudgetError):
            budget.parse_meminfo("MemAvailable:   -5 kB\n")
        with self.assertRaises(budget.BudgetError):
            budget.parse_meminfo("MemAvailable:\n")
        with self.assertRaises(budget.BudgetError):
            budget.parse_meminfo("MemAvailable:   kB\n")

    def test_moe_counts_total_weights(self):
        # 35B-A3B: admission uses the FULL 35B checkpoint, not 3B active.
        est = budget.estimate_required(memory(66.97, kv_per_tok=1024), 1024)
        self.assertEqual(est.weights, int(66.97 * GiB))


class ContextTests(unittest.TestCase):
    def test_limit_is_default(self):
        self.assertEqual(budget.resolve_context({"max_tokens": 40960}, None), 40960)

    def test_fallback_default(self):
        self.assertEqual(budget.resolve_context({"max_tokens": None}, None),
                         budget.DEFAULT_CONTEXT_TOKENS)

    def test_over_limit_is_error_not_clamp(self):
        with self.assertRaises(budget.BudgetError):
            budget.resolve_context({"max_tokens": 512}, 1024)

    def test_under_limit_ok(self):
        self.assertEqual(budget.resolve_context({"max_tokens": 1024}, 512), 512)


class MeminfoTests(unittest.TestCase):
    def test_parses_memavailable(self):
        self.assertEqual(budget.parse_meminfo(meminfo(8.0)), int(8 * GiB))

    def test_missing_memavailable_raises(self):
        with self.assertRaises(budget.BudgetError):
            budget.parse_meminfo("MemTotal: 1 kB\n")


class AdmissionTests(unittest.TestCase):
    def test_fit_and_reject_with_reservations(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            budget.set_reservation("laya", int(1.1 * GiB), "decisions server", home)
            with unittest.mock.patch.object(budget, "mem_available",
                                            lambda: int(16 * GiB)):
                fits = budget.admit(int(11 * GiB), home)
                self.assertTrue(fits.fits)
                # 16 - 2 safety - 1.1 laya - 11 < 0 -> refuse
                over = budget.admit(int(14 * GiB), home)
                self.assertFalse(over.fits)
                self.assertIn("laya", json.dumps(budget.load_reservations(home)))

    def test_rejected_headroom_is_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            with unittest.mock.patch.object(budget, "mem_available",
                                            lambda: int(4 * GiB)):
                result = budget.admit(int(3 * GiB), Path(tmp))
                self.assertFalse(result.fits)
                self.assertLess(result.headroom, 0)

    def test_negative_reservation_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            home.mkdir(exist_ok=True)
            (home / budget.RESERVATIONS_FILE).write_text(
                json.dumps({"cheat": {"bytes": -4, "note": ""}}))
            with self.assertRaises(budget.BudgetError):
                budget.load_reservations(home)

    def test_two_phase_reservation_pending_vs_resident(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            budget.set_reservation("laya", int(1 * GiB), "before load", home,
                                   state="pending")
            budget.set_reservation("chat", int(5 * GiB), "running", home,
                                   state="resident")
            with unittest.mock.patch.object(budget, "mem_available",
                                            lambda: int(16 * GiB)):
                # resident already lives inside MemAvailable: only pending counts
                result = budget.admit(int(13 * GiB), home)
                self.assertTrue(result.fits)
                self.assertEqual(result.reserved, int(1 * GiB))
            budget.set_reservation_state("laya", "resident", home)
            self.assertEqual(budget.load_reservations(home)["laya"]["state"], "resident")
            with self.assertRaises(budget.BudgetError):
                budget.set_reservation("x", 1, "", home, state="ghost")

    def test_concurrent_reservation_writes_stay_consistent(self):
        import threading

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            def writer(i):
                budget.set_reservation(f"svc{i}", 1024 * (i + 1), "", home)
            threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            data = budget.load_reservations(home)
            self.assertEqual(len(data), 8)
            leftovers = [p.name for p in home.iterdir()
                         if p.name.endswith(".tmp")]
            self.assertEqual(leftovers, [])

    def test_corrupt_reservations_refuse_to_guess(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            home.mkdir(exist_ok=True)
            (home / budget.RESERVATIONS_FILE).write_text("[]")
            with self.assertRaises(budget.BudgetError):
                budget.load_reservations(home)

    def test_clear_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            budget.set_reservation("laya", 1, "note", home)
            self.assertTrue(budget.clear_reservation("laya", home))
            self.assertFalse(budget.clear_reservation("laya", home))
            self.assertEqual(budget.load_reservations(home), {})

    def test_set_reservation_rejects_garbage(self):
        with self.assertRaises(budget.BudgetError):
            budget.set_reservation("", 5)
        with self.assertRaises(budget.BudgetError):
            budget.set_reservation("x", -1)


class DiskTests(unittest.TestCase):
    def test_disk_check_compares_free_to_need(self):
        with tempfile.TemporaryDirectory() as tmp:
            free = budget.disk_free(Path(tmp))
            ok, got_free, _ = budget.disk_check(free - 1, Path(tmp))
            self.assertTrue(ok)
            self.assertEqual(got_free, free)
            ok, _, _ = budget.disk_check(free + 1, Path(tmp))
            self.assertFalse(ok)




if __name__ == "__main__":
    unittest.main()
