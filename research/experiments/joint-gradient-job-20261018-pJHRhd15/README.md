# Joint spatial/native-gradient campaign

The two-image execution smoke passed and was committed/pushed; the fixed 128-training-image development stage is running in the background on GPU0. Eight arms, six views, two frozen seed0 heads. This is a mechanism and native-gradient variance diagnostic, not full-data AP training or an actual 25%-query training run.

Status: `bash online_experiments/run_joint_gradient.sh status`.
Full log: `JOB="$(cat runs/paper/joint-gradient/latest-job.txt)"; tail -f "$JOB/job.log"`.
When development succeeds, inspect `development/screening.json`, `development/comparison.txt` and `archive-status.json`. Completed stages are verified, archived, committed and pushed automatically. A push failure is recorded separately from experiment success.

Launch/progress observation: launch.json. Real-smoke integrity/adversarial checks: research/checks/joint-gradient-20261018/real-smoke-verification.json. Source protocol: research/JOINT_GRADIENT_IMPLEMENTATION_20261018.md.
