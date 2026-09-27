# Full VisDrone reference-protocol rescoring

Source training: runs/paper/full-visdrone/job-start-eDmH0odX (6471 train / 548 val, seed 0).
Source evaluation: runs/paper/visdrone-eval/job-full-20260927.
Completed: 12 prediction sets (six methods, final and custom-selected best).

The Python port follows VisDrone2018-DET-toolkit commit
005445782213e20cb91bc50a597db3dd949e749a, including class-image occurrence
weighting, ignored regions/GT, global detection prefixes, and VOC-integral AP.
Do not substitute the separately reported unweighted macro diagnostic.
Reference MATLAB source was executed with Octave 10.3 on 24 synthetic and two
real-image parity cases; it was not run with commercial MATLAB or on all 548
images in the native environment. Preprocessed arrays matched exactly; maximum
seven-metric error was 4.41e-13 AP points. All 548 images were scored by the port.
Actual reference file bytes and complete annotation/prediction coverage were checked.
This does not establish that the YOLO-Master paper used the identical protocol.

Final checkpoint is the prospectively declared primary endpoint. 'best' remains
the checkpoint selected by the historical custom DetMetrics evaluator, not the
maximum reference AP over all epochs. Old metrics and predictions are preserved.
Single seed: no SD/significance/stable cross-seed advantage claim.

Final AP: C1 9.1186, C2 10.6568, Mix25 11.3526, Transport25 9.6350,
Correct25 9.7656, Online-Geom 11.3982. Correct25 minus Transport25 is +0.1306;
Correct25 minus Mix25 is -1.5870. The reconstruction corrector is not a finalized
superior method. A frozen held-out feature/task diagnostic is the next step.
No new AP-small result, cross-dataset result, or inference-speed result is claimed.

Verification: ten reference unit tests and two feature-metric unit tests pass.
Scoped Ruff passes. Required global checks retain 2768 historical Ruff findings
and five formatting mismatches; full codespell status is recorded in validation.
Large predictions, annotations, native parity inputs, weights and cache are kept
in the run directory, not duplicated into Git. Input/source hashes are archived.
Initial loader, Octave-prefix and threshold-parity failures are retained.

## Results

~~~text
VisDrone pinned-reference Python port; Octave parity passed.
AP 0-100; last is primary, best was selected by custom training AP.
method checkpoint AP AP50 AP75 AR500 macro_AP_diagnostic
cache1 last 9.1186 20.3817 7.1366 22.2363 8.8044
cache1 best 9.4065 20.8906 7.5036 22.2945 9.2781
cache2 last 10.6568 23.6290 8.3528 24.2160 10.3244
cache2 best 10.7025 23.6397 8.4385 24.2687 10.4061
mix25 last 11.3526 24.5600 9.1624 24.5009 11.2540
mix25 best 11.3878 24.5839 9.1635 24.5401 11.2933
transport25 last 9.6350 20.9255 7.9773 21.8929 9.5717
transport25 best 9.6019 20.8466 7.8775 21.7519 9.5399
correct25 last 9.7656 21.1736 7.9369 22.1878 9.7314
correct25 best 9.7867 21.1914 7.9808 22.2391 9.7540
online-geom last 11.3982 24.5792 9.2883 25.0183 11.2361
online-geom best 11.2903 24.3648 9.1573 24.6667 11.2212
~~~
