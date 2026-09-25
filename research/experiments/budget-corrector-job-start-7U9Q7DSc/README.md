# Fixed-budget correction pilot

Status: SUCCEEDED

400 training / 100 validation images; three training seeds in the full plan.
Use original/comparison.txt for paired differences and completed-run counts.
C2-Mix, C2-Transport and C2-Correct share query identities and augmentation proposals.
The corrector is supervised by queried training features only; validation is uncorrected.
Schedule hashes are archived; full schedules remain at the source job path.
Cache construction is charged once per run; prior validation-cache build is excluded.
Three seeds on this pilot do not establish official benchmark performance.
