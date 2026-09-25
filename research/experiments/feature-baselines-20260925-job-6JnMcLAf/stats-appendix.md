# Statistical appendix

This is a descriptive, exploratory pilot, not a confirmatory benchmark.
The unit is a training seed (n=3), conditional on one fixed 400/100-image split.
Epochs and individual detections are not treated as independent replicates.
Best AP uses validation checkpoint selection; two brightness strengths were also explored.
All settings are reported, rather than publishing only the best strength.

The tables include mean differences in AP points and paired differences.
Paired two-sided exact sign-flip tests enumerate all 2^3 sign assignments.
Their null requires sign exchangeability/symmetry of paired differences.
With three seeds the smallest attainable two-sided p-value is 0.25.
Holm adjustment is applied to the nine exploratory best-AP contrasts.
Thus these tests cannot establish conventional significance in this experiment.

Student-t 95% intervals use df=2 and multiplier 4.30265273.
Normality cannot be meaningfully assessed with three seeds; these intervals
are assumption-dependent descriptors, not evidence of generalization across datasets.
Per-method intervals and seed SDs are available in analysis.json and summary.csv.

| Contrast | Best AP delta | Paired seed deltas | Paired t 95% interval | Exact p | Holm p | Last AP delta |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| cache2 - cache1 | +1.0711 | +1.2488, +0.9137, +1.0509 | [+0.6527, +1.4896] | 0.250 | 1.000 | +1.0176 |
| frofa-c1-d01 - cache1 | +0.1946 | +0.2528, +0.2562, +0.0747 | [-0.0633, +0.4525] | 0.250 | 1.000 | +0.3548 |
| frofa-c1-d03 - cache1 | -0.1019 | +0.0175, -0.2996, -0.0238 | [-0.5302, +0.3263] | 0.500 | 1.000 | +0.1362 |
| frofa-c2-d01 - cache2 | -0.0654 | -0.2700, -0.0141, +0.0881 | [-0.5235, +0.3928] | 0.750 | 1.000 | +0.1641 |
| frofa-c2-d03 - cache2 | -0.3164 | -0.3846, -0.1919, -0.3728 | [-0.5847, -0.0482] | 0.250 | 1.000 | -0.0030 |
| transport-c1 - cache1 | +0.6081 | +0.6004, +0.5119, +0.7121 | [+0.3590, +0.8573] | 0.250 | 1.000 | +0.9316 |
| loffta-c1-n005 - transport-c1 | +0.1078 | +0.3337, -0.0949, +0.0846 | [-0.4270, +0.6426] | 0.750 | 1.000 | +0.2368 |
| transport-c2 - cache2 | +0.1260 | -0.2084, +0.3356, +0.2510 | [-0.6012, +0.8533] | 0.500 | 1.000 | +0.6288 |
| loffta-c2-n005 - transport-c2 | +0.0009 | +0.0818, -0.0251, -0.0539 | [-0.1767, +0.1785] | 1.000 | 1.000 | -0.0244 |
