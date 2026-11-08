# Verified findings

- The formal completed main dataset is VisDrone6471/548. COCO128, COCO val1000 and VOC2007 subset are pilot evidence.
- Server COCO train2017 has zero images, val2017 has5000; no full COCO result exists.
- YOLO-Master arXiv2512.23273v1 §4.1 uses COCO2017, VOC07+12, VisDrone2019, KITTI and SKU-110K.
- Additional formal datasets selected: COCO2017, VOC07+12, SKU-110K. They retain horizontal-box detection and offer distinct natural/retail domains.
- Free disk before cleanup:1400406491136 bytes. Original project runs occupy about183GB; most disk usage is outside this project and outside cleanup scope.
- Active reuse covers approximately135GiB of VisDrone feature banks. Original raw datasets are much smaller; deleting VisDrone now would break remaining seeds and later tuning.
- The COCO temporary extraction held122225 files. Every non-metadata file matched canonical data or a member of the retained labels ZIP. No unique data removed.
- Cleanup observed free-space increase530046976 bytes, approximately0.494GiB; full evidence retained in runtime cleanup plan and compact result.
- Server system Python lacks rg; use Python traversal for repository-scoped fallback. The training conda environment has numpy/torch/PIL but pycocotools is absent.

2026-11-09 migration findings:
- AutoDL has existing /root/miniconda3 basePython3.12.3/Pillow11.3; isolated yolomaster3.10.20 is being installed. New GPU inventory is RTX4080 SUPER32760MiB; old and new observed times must remain separate.
- /root/autodl-tmp is still50GiB. Complete VOC C2+test bank is about130GiB; full COCO C2+val about829GiB before images and reserve.
- No local full raw dataset, checkpoint or feature bank was found; Git restores source and lightweight records only.
- VOC preparation is a data-only script. Existing paper_baseline_suite hardcodes VisDrone counts/class/metrics and must not be invoked on VOC by only changing its paths.
