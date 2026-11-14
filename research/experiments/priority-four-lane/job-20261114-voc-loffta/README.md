# Fourth lane: VOC LOFF-TA

Actual detached launch 2026-11-14 19:17CST. New supervisor2300793 preserves live VOC FroFA1467410 and starts cached LOFF-TA2300847 in slot3. Existing SKU INT81881349 and Detail252105170 retain their original lifetimes and slot1/2. Source committed/pushed068d9f2 before launch. Full VOC16551train/4952test,seed0,final100,batch8,640; no new method or training recipe.

Four-worker cap, extra GPU>=12GiB/host>=18GiB admission. LOFF validates/archives once; VOC waits both current FroFA/LOFF before remaining INT8,Adapter,Detail. Canonical outputs/log aliases are visible to the unchanged CPU metric watcher6/14complete,0failed. GPU activity and original1366source hashes verified; first epoch and final AP remain pending. Shared GPU timing is observational.

Check `runs/paper/priority-four-lane/job-20261114-voc-loffta/status.json` and `train-loffta-c2-n005.log`. A blank log before epoch1 is expected; do not launch duplicate workers. Full raw outputs stay outside Git.
