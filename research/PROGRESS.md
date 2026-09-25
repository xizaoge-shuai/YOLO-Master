# D1 experiment progress

## 2026-09-25: cached-feature augmentation baselines

- Completed: 10 configurations, 3 seeds each, 100 epochs, fixed 400/100 split.
- Actual head: mean_value; identical unaugmented validation cache.
- Strong cheap reference: two real cached views + geometric feature transport.
- Mean best AP: C1 2.5694; C2 3.6405; Transport C1 3.1775;
  Transport C2 3.7666; LOFF-TA-style C2 3.7675.
- Noise adds approximately 0.0009 mean best AP over Transport C2 in this pilot;
  this is not evidence of a reliable improvement.
- Next: 10%/25% matched-backbone-query C2-Mix / C2-Transport / C2-Correct.
- Source code and experimental evidence are recorded in separate commits.
- [Evidence and analysis](experiments/feature-baselines-20260925-job-6JnMcLAf/README.md).

Earlier online, fusion, mean-sweep and flip-cache runs remain historical
references. Their helper scripts are retained as run; this entry does not
claim that their weights/predictions have been independently re-evaluated.
