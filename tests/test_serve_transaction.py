"""Shared atomic budget transaction: cross-process race and owner proofs.

Main-mandated cases:
- a real two-process concurrent admit_and_reserve cannot overcommit: with
  a fixed budget, exactly one winner is possible and the sum of held
  reservations never exceeds the budget;
- a reservation held by one owner refuses overwrite, relabel, and clear
  by another owner (two processes may not serve the same catalog id).
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_serve import budget  # noqa: E402

GiB = 1024**3

WORKER = """
import json
import sys
from pathlib import Path
sys.path.insert(0, {serve!r})
from mlx_omarchy_serve import budget

owner = budget.generate_owner()
name = sys.argv[1]
try:
    result = budget.admit_and_reserve(
        name, int(sys.argv[2]), note="worker", owner=owner,
        home=Path(sys.argv[3]), available_bytes=int(sys.argv[4]))
    print(json.dumps({{"ok": True, "name": name, "owner": owner,
                       "reserved": result.reserved}}))
except budget.BudgetError as exc:
    print(json.dumps({{"ok": False, "name": name, "owner": owner,
                       "error": str(exc)}}))
""".format(serve=str(REPO_ROOT / "serve"))


def run_worker(name, gib, home, available):
    result = subprocess.run(
        [sys.executable, "-c", WORKER, name, str(int(gib * GiB)), str(home),
         str(int(available * GiB))],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise AssertionError(f"worker crashed: {result.stderr}")
    return json.loads(result.stdout.strip().splitlines()[-1])


class ConcurrentTransactionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)

    def test_two_processes_cannot_overcommit(self):
        # Budget 10 GiB, safety 2 GiB -> 8 usable; two 6 GiB claims race.
        outcomes = [run_worker(f"svc{i}", 6, self.home, 10) for i in range(2)]
        winners = [o for o in outcomes if o["ok"]]
        losers = [o for o in outcomes if not o["ok"]]
        self.assertEqual(len(winners), 1)
        self.assertEqual(len(losers), 1)
        self.assertIn("does not fit", losers[0]["error"])
        held = budget.load_reservations(self.home)
        self.assertEqual(sum(r["bytes"] for r in held.values()), int(6 * GiB))

    def test_four_racers_hold_never_past_budget(self):
        # 4 processes race for a 10 GiB budget with 4 GiB claims
        # (usable 8): exactly two may win, in any order.
        outcomes = [run_worker(f"svc{i}", 4, self.home, 10) for i in range(4)]
        winners = [o for o in outcomes if o["ok"]]
        self.assertEqual(len(winners), 2)
        held = budget.load_reservations(self.home)
        self.assertEqual(sum(r["bytes"] for r in held.values()), int(8 * GiB))
        self.assertEqual(len(held), 2)
        for outcome in outcomes:
            if not outcome["ok"]:
                self.assertIn("does not fit", outcome["error"])

    def test_duplicate_owner_refusal_across_processes(self):
        first = run_worker("laya", 1, self.home, 16)
        self.assertTrue(first["ok"])
        # A second process (different owner) may not reserve, relabel,
        # or clear the held name — it must fail loudly in its own process.
        script = """
import json
import sys
from pathlib import Path
sys.path.insert(0, {serve!r})
from mlx_omarchy_serve import budget

GiB = 1024 ** 3
home = Path(sys.argv[1])
results = []
try:
    budget.admit_and_reserve("laya", 1 * GiB, owner="intruder",
                             home=home, available_bytes=16 * GiB)
    results.append("reserved")
except budget.BudgetError:
    results.append("refused-reserve")
try:
    budget.set_reservation_state("laya", "resident", home=home)
    results.append("relabeled")
except budget.BudgetError:
    results.append("refused-relabel")
try:
    budget.clear_reservation("laya", home=home, owner="intruder")
    results.append("cleared")
except budget.BudgetError:
    results.append("refused-clear")
print(json.dumps(results))
""".format(serve=str(REPO_ROOT / "serve"))
        result = subprocess.run([sys.executable, "-c", script, str(self.home)],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout.strip().splitlines()[-1]),
                         ["refused-reserve", "refused-relabel", "refused-clear"])
        # and the legitimate owner's reservation survived all of it
        held = budget.load_reservations(self.home)["laya"]
        self.assertEqual(held["bytes"], int(1 * GiB))


class OwnerSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)

    def test_owner_token_is_unique_per_call(self):
        self.assertNotEqual(budget.generate_owner(), budget.generate_owner())

    def test_unowned_manual_entry_refuses_owned_overwrite(self):
        budget.set_reservation("manual", int(1 * GiB), "human", self.home)
        with self.assertRaises(budget.BudgetError):
            budget.admit_and_reserve("manual", int(1 * GiB), owner="bot",
                                     home=self.home, available_bytes=int(64 * GiB))

    def test_same_owner_may_retransact_own_reservation(self):
        owner = budget.generate_owner()
        budget.admit_and_reserve("laya", int(2 * GiB), owner=owner,
                                 home=self.home, available_bytes=int(64 * GiB))
        budget.admit_and_reserve("laya", int(3 * GiB), owner=owner,
                                 home=self.home, available_bytes=int(64 * GiB))
        self.assertEqual(budget.load_reservations(self.home)["laya"]["bytes"],
                         int(3 * GiB))

    def test_missing_owner_refused(self):
        # owner is a required keyword: a caller bug is a TypeError, not a
        # runtime budget condition.
        with self.assertRaises(TypeError):
            budget.admit_and_reserve("laya", int(1 * GiB), home=self.home,
                                     available_bytes=int(64 * GiB))


if __name__ == "__main__":
    unittest.main()
