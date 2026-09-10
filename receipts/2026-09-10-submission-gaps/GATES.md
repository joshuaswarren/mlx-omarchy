# Gates: Submission gaps

OWNS: overlay/mlx/backend/omarchy/encoder.cpp, overlay/mlx/backend/omarchy/encoder.h, overlay/mlx/backend/omarchy/eval.cpp, overlay/mlx/backend/omarchy/compiled.cpp, overlay/mlx/backend/omarchy/device.cpp, overlay/mlx/backend/omarchy/gpu_profiler.h, overlay/tests/omarchy/**, receipts/2026-09-10-submission-gaps/**

Scope: Attribute host time at every submission boundary, remove only the measured dominant cause without violating ordering or visibility, and verify the result on llvmpipe and both M1 drivers. Since the base fix landed on main, compare the remaining refinement with exact current main and reject it unless it recovers decode without losing correctness.

- [x] G1: The receipt contains measured before/after submission-gap summaries for 262 and 1053 tokens.
  CHECK: python3 -c "import json; d=json.load(open('receipts/2026-09-10-submission-gaps/verdict.json')); assert {'262','1053'} <= d['gaps'].keys(); print('gap receipt valid')"
  EXPECT: gap receipt valid
  EVIDENCE: verdict.json; baseline-a7b9-m*.json; after-5fac-m*.json.

- [x] G2: Runtime and targeted tests pass on llvmpipe, M1 fork Honeykrisp, and M1 stock Honeykrisp.
  CHECK: python3 -c "import json; d=json.load(open('receipts/2026-09-10-submission-gaps/verdict.json')); t=d['tests']; assert all(t[k]['passed'] for k in ('llvmpipe','m1_fork','m1_stock')); print('targeted tests valid')"
  EXPECT: targeted tests valid
  EVIDENCE: tests-final/*.log.

- [x] G3: All six workload digests match the required values on both M1 drivers.
  CHECK: python3 -c "import json; d=json.load(open('receipts/2026-09-10-submission-gaps/verdict.json')); assert d['paired_stock_digests']['all_match']; assert all(x['digest_same'] for x in d['paired_fork_refinement_vs_current_main']); print('digest matrix valid')"
  EXPECT: digest matrix valid
  EVIDENCE: refinement-vs-main/*.json; final-stock/submission-gaps-final-stock.json.

- [x] G4: The receipt contains an identity-logged, three-pair six-leg fork comparison against exact current main, stock digests, wheel SHA-256 values, source commits, decision, and next action.
  CHECK: python3 -c "import json; d=json.load(open('receipts/2026-09-10-submission-gaps/verdict.json')); assert len(d['paired_fork_refinement_vs_current_main'])==6; assert d['candidate']['wheel_sha256']; assert d['current_main_baseline']['wheel_sha256']; assert d['decision']; assert d['next']; print('acceptance receipt complete')"
  EXPECT: acceptance receipt complete
  EVIDENCE: verdict.json; refinement-vs-main/*.json; final-stock/*; submission-gaps-current-main-build.log.

- [x] G5: The refinement decision applies the 0.99 decode floor against current main.
  CHECK: python3 -c "import json; d=json.load(open('receipts/2026-09-10-submission-gaps/verdict.json')); assert not d['policy_ok']; assert any(x['decode_ratio'] < .99 for x in d['paired_fork_refinement_vs_current_main']); assert d['decision'].startswith('REJECT REFINEMENT'); print('refinement correctly rejected')"
  EXPECT: refinement correctly rejected
  EVIDENCE: verdict.json; Q4 short decode ratio 0.988715 and Q4 262 decode ratio 0.984542.

- [x] G6: The committed branch is pushed to origin and Main receives the current-main comparison and rejection decision.
  EVIDENCE: final branch ref and hub delivery receipt reported at completion.
