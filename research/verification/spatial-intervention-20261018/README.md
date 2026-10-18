# Spatial intervention pre-launch verification

23 targeted tests passed (4 pytest temporary-directory cleanup warnings); four new Python files pass Ruff check/format and codespell; launcher Bash syntax passes. Raw check logs are retained at runs/paper/spatial-intervention/checks.

Repository-required global checks were run: Ruff reports the existing 2768 errors, format reports 5 files plus parse errors, and full codespell times out after 45 seconds. No unrelated global fixes were attempted.

Independent reviewer stopped on usage limits before a final review; manual checks are described in the frozen protocol. No GPU success is claimed here. The live campaign must pass the two-image native-loss/state/row/source integrity gates before proceeding to 128 training images.
