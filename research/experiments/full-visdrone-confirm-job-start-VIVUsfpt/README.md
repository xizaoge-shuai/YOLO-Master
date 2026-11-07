# Historical full-VisDrone confirmation recovery

Recovered 2026-11-07 from `runs/paper/full-visdrone-confirm/job-start-VIVUsfpt`. These are six existing runs, not six newly trained repeats. Full 6471/548 VisDrone, epoch100, batch8,640,mean_value.

| Method | seed1 final saved AP | seed2 final saved AP |
| --- | ---: | ---: |
| C2 | 10.159850951436841 | 10.337528652236932 |
| Mix25 | 10.88831702796182 | 10.809557307833769 |
| Online100 | 11.398457511328816 | 11.691563635468281 |

The saved reference evaluation reports PASS, 548 images and pinned reference `005445782213e20cb91bc50a597db3dd949e749a`. All six DONE markers and 100-epoch CSVs were checked, together with 600 schedule digests. Original and archived metadata SHA256 values are in recovery-check.json. JSON repository paths are normalized; numbers are preserved.

No fresh AP was calculated. The inspected historical evaluation/summary records do not bind checkpoint and prediction hashes to that scoring event; current file presence is insufficient to retroactively authenticate the saved evaluation. Best links retain the pre-migration prefix. Historical provenance names commit `aa56d5b050f982734e99c2e4a37b4a0931aebe90`, 1311 source hashes and two dirty files; it must not be described as a clean immutable checkout. No checkpoint, predictions, schedules, datasets or caches are copied into Git.
