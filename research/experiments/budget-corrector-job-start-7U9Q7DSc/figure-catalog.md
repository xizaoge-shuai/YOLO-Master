# Figure catalog

## budget-comparison.pdf / .png
- Purpose: compare matched methods at each online budget without hiding seed variability.
- Source: original/train/*/metrics.csv and summary.json; AP multiplied by 100 only.
- Variables: best/final validation AP; mean +/- sample SD; dots show three training seeds.
- Caption must state 400/100 internal split and selection of best on validation.
- Interpretation: compare Correct to Transport as well as Mix; p25 final averages nearly tie.
- Caveat: a small development pilot, not official benchmark or independent dataset uncertainty.

## reconstruction-and-timing.pdf / .png
- Purpose: separate successful dense reconstruction from detector benefit; expose timing anomaly.
- Source: correction.jsonl and epoch metrics.csv.
- Top: mean over three seeds of pre-update query relative MSE; queries are training images.
- Bottom: all three Transport p25 timing trajectories, including the anomalous seed.
- Caption must retain the anomaly and say that no hardware telemetry identifies its cause.
- Interpretation: lower reconstruction loss alone does not demonstrate task utility.
- Caveat: no held-out feature probe or foreground/object-size decomposition exists yet.
