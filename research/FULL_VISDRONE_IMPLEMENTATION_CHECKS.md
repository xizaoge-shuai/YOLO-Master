# Full VisDrone implementation checks

- Five contract tests passed locally and on the server: full tail-preserving image quota, paired plans,
  zero-area label filtering, train/val isolation, rounded letterbox inversion and memmap reads.
- GPU0 smoke job job-smoke-2TKOvomC: six methods, 17 train / 8 val, one epoch each.
  SUCCEEDED; paired 25% arms each queried 4 images, online queried 17; cache parity max about 0.000205.
  This is a pipeline test; near-zero AP is not a method performance result.
- Full data preparation: 6471 train / 548 val, no test-dev. One zero-height box filtered in derived labels:
  train/9999985_00000_d_0000020.txt line 11.
- Verification job verification-1790393998421078270: two-epoch correct25 interrupted after epoch 1,
  restored and compared with uninterrupted training; detector and corrector maximum absolute weight
  differences both 0.0; both consumed 8 training queries.
- Initial verification comparison failed on non-tensor state_dict metadata. Fixed the verification
  comparison, not training, then reran the full check successfully.
- Required global Ruff reports 2768 pre-existing violations; global formatter reports five existing
  files. They are historical experiment dependencies, left unchanged. codespell is not installed.
- New-scope lint, formatting, shell syntax and final GPU recheck results are recorded before final commit.

Limit: official VisDrone evaluator runtime is absent; exported predictions await official rescoring.
The 100-epoch full campaign has not been launched by the assistant.

## Final review and recovery checks

- Independent review found and resolved: duplicate launch clobbering, smoke/full resume confusion, cache hash verification, and final-epoch artifact consistency.
- Seven data contracts pass. New source lint/format and Bash syntax pass.
- Final verification-1790394572184101177: model/corrector resumed weight difference 0.0; final-commit os._exit(99) fault recovered with coherent best weights/predictions and DONE.
- Lightweight evidence: full-visdrone-validation-20260926/. Reserve cache plus 30 GiB for retained best artifact versions.
- Official VisDrone scoring and full 100-epoch results remain pending. No production training launched in this implementation turn.

- Final related tests: 11 passed (7 full-data contracts + 4 existing budget-corrector tests); scoped Ruff/format and bash -n passed.
