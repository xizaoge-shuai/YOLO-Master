# Joint intervention/native-gradient implementation verification

- 46 targeted tests passed; four existing pytest temporary-directory cleanup warnings. Ruff, Ruff format, codespell for the five new Python files passed; bash syntax passed.
- Failure-first helper tests exercised missing implementations; the integration review found Tensor-versus-three-level-tuple mismatch. A three-level test reproduced the failure before the helper accepted validated, matched sequences. All tests were rerun after that fix.
- Independent focused review found no remaining launch-blocking issue. Actual GPU smoke is still required after this implementation commit.
- Required repository-wide checks: existing 2768 Ruff errors; formatting check reports five files plus parse errors; full codespell timed out at 45 seconds. No unrelated files were changed. Full noisy logs remain on the server; compact global-checks.json is archived.
- Source hashes and final check output: verification.json. Frozen protocol: research/JOINT_GRADIENT_IMPLEMENTATION_20261018.md.
- Experiment SUCCEEDED and push status are separate: inspect archive-status.json for remote verification. Failed pushes are preserved for retry without discarding diagnostic results.
- This campaign only runs execution smoke and fixed 128-image training-development diagnostics. Native-gradient enumeration at p=.25 is not actual reduced-query training. No AP or novelty claim.
