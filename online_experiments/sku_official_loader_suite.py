"""SKU-only complete-split protocol with disclosed tolerant JPEG decoding.

Reuses the frozen detector, schedule, losses, and official-loader-compatible
boundary annotations. This wrapper changes neither VOC nor old source files.
The five damaged official JPEGs remain byte-for-byte intact and are retained.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ANCHOR_RELATIVE = "runs/paper/crossdataset/job-20261110-voc0712-s0-v2/provenance.json"
ANCHOR_SHA256 = "9564e537b32f18fe51a949888a6656a047c2a02797fd9ce863247e1e0b37d932"
EVIDENCE_RELATIVE = "research/checks/sku-preparation-recovery-20261110"
ADDED_SOURCES = (
    "online_experiments/sku_official_loader_suite.py",
    "online_experiments/sku_official_loader_campaign.py",
)
DAMAGED_JPEGS = {"test_274.jpg", "train_4222.jpg", "train_5822.jpg", "train_882.jpg", "train_924.jpg"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked_anchor():
    path = ROOT / ANCHOR_RELATIVE
    if sha(path) != ANCHOR_SHA256:
        raise RuntimeError("Frozen source anchor changed")
    anchor = read(path)
    if len(anchor["source_sha256"]) != 1366:
        raise RuntimeError("Expected reviewed 1366-file frozen source anchor")
    for relative, expected in anchor["source_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError("Frozen source changed: " + relative)
    return anchor


def checked_decoder_evidence():
    folder = ROOT / EVIDENCE_RELATIVE
    strict_path = folder / "sku-full-image-decode.json"
    tolerant_path = folder / "sku-full-image-decode-tolerant.json"
    strict, tolerant = read(strict_path), read(tolerant_path)
    if strict["images"] != 11743 or strict["decoded"] != 11743:
        raise RuntimeError("Strict decoder evidence must cover all official SKU images")
    if {row["file"] for row in strict["failures"]} != DAMAGED_JPEGS:
        raise RuntimeError("Official damaged JPEG set differs from reviewed five-image evidence")
    if (
        tolerant["status"] != "PASS"
        or tolerant["images"] != 11743
        or tolerant["decoded"] != 11743
        or tolerant["failures"]
    ):
        raise RuntimeError("Tolerant decoder must retain and decode every original split image")
    return {
        "strict_decode_evidence": {"file": str(strict_path), "sha256": sha(strict_path)},
        "tolerant_decode_evidence": {"file": str(tolerant_path), "sha256": sha(tolerant_path)},
        "policy": "Pillow ImageFile.LOAD_TRUNCATED_IMAGES=True in this process only; identical decoder for every SKU arm. Retain all11743 original JPEGs and full official annotations; raw bytes unchanged; no sample drop. Tolerant decode may conceal missing JPEG entropy data; not a byte repair or pixel-identical claim to author OpenCV.",
        "damaged_original_jpegs": sorted(DAMAGED_JPEGS),
        "annotation_policy": "Author CSVGenerator boundary rules, original CSV boxes retained; author-loader compatibility COCO AP, not a quoted original paper number.",
    }


def frozen_source_snapshot():
    anchor = checked_anchor()
    decoder = checked_decoder_evidence()
    sources = dict(anchor["source_sha256"])
    sources.update({relative: sha(ROOT / relative) for relative in ADDED_SOURCES})
    for record in (decoder["strict_decode_evidence"], decoder["tolerant_decode_evidence"]):
        sources[Path(record["file"]).relative_to(ROOT).as_posix()] = record["sha256"]
    return {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": sources,
        "source_scope_policy": "Exact1366 frozen keys from pinned VOC source anchor, plus reviewed SKU wrappers and decoder evidence; unrelated new project scripts deliberately excluded.",
        "source_anchor": {"file": str(ROOT / ANCHOR_RELATIVE), "sha256": ANCHOR_SHA256},
        "sku_image_decoder": decoder,
    }


def install():
    checked_anchor()
    checked_decoder_evidence()
    from PIL import ImageFile

    from online_experiments import crossdataset_suite as base

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    base.VERSION = "crossdataset-sku-official-loader-v2"
    base.source_snapshot = frozen_source_snapshot
    previous = base.load_dataset

    def load_sku_only(*args, **kwargs):
        result = previous(*args, **kwargs)
        if result[0]["dataset"] != "sku110k":
            raise RuntimeError("SKU decoder wrapper cannot change another dataset's protocol")
        return result

    base.load_dataset = load_sku_only
    return base


if __name__ == "__main__":
    install().main()
