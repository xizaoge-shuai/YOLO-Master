# D1 8.24 Feature Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic 100-image FP16 DINOv3 multi-level feature cache with integrity, interface, disk, I/O, latency, and peak-VRAM evidence for D1 admission.

**Architecture:** A small cache contract module owns stable sample IDs, manifests, checksums, and validation. A standalone CLI resolves a YOLO dataset split, letterboxes images deterministically, extracts configured DINOv3 backbone feature maps, writes one safetensors file per image atomically, and emits machine-readable admission reports. Detector training integration is deliberately deferred to the next plan.

**Tech Stack:** Python 3.10, PyTorch, Transformers DINOv3ViTBackbone, safetensors, Pillow, Ultralytics dataset utilities, pytest.

---

### Task 1: Cache contract and integrity validation

**Files:**
- Create: `ultralytics/data/foundation_cache.py`
- Test: `tests/test_d1_foundation_cache.py`

- [ ] **Step 1: Write failing contract tests**

```python
from pathlib import Path

import pytest

from ultralytics.data.foundation_cache import CacheIdentity, image_sha256, stable_sample_id, validate_manifest


def test_stable_sample_id_is_path_and_content_bound(tmp_path: Path):
    image = tmp_path / "images" / "a.jpg"
    image.parent.mkdir()
    image.write_bytes(b"image-a")
    first = stable_sample_id(image, tmp_path)
    assert first == stable_sample_id(image, tmp_path)
    image.write_bytes(b"image-b")
    assert first != stable_sample_id(image, tmp_path)


def test_manifest_rejects_preprocess_mismatch():
    identity = CacheIdentity("model", "revision", (3, 7, 11), 640, "abc")
    manifest = {"schema_version": 1, "identity": identity.to_dict(), "samples": []}
    with pytest.raises(ValueError, match="preprocess_fingerprint"):
        validate_manifest(manifest, identity.__class__("model", "revision", (3, 7, 11), 640, "different"))
```

- [ ] **Step 2: Run tests and confirm import failure**

Run: `pytest -q tests/test_d1_foundation_cache.py`

Expected: FAIL because `ultralytics.data.foundation_cache` does not exist.

- [ ] **Step 3: Implement immutable identity, SHA256 helpers, atomic JSON writes, and strict manifest validation**

```python
@dataclass(frozen=True)
class CacheIdentity:
    model_id: str
    revision: str
    layers: tuple[int, ...]
    imgsz: int
    preprocess_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"layers": list(self.layers)}


def stable_sample_id(path: Path, root: Path) -> str:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    return hashlib.sha256(f"{relative}\0{image_sha256(path)}".encode()).hexdigest()[:24]
```

Validation must reject schema, model ID, revision, layer, image-size, preprocessing fingerprint, duplicate sample ID, missing cache file, and checksum mismatches with field-specific messages.

- [ ] **Step 4: Run contract tests**

Run: `pytest -q tests/test_d1_foundation_cache.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ultralytics/data/foundation_cache.py tests/test_d1_foundation_cache.py
git commit -m "feat: add D1 foundation cache contract"
```

### Task 2: Deterministic dataset selection and preprocessing

**Files:**
- Modify: `ultralytics/data/foundation_cache.py`
- Modify: `tests/test_d1_foundation_cache.py`

- [ ] **Step 1: Add failing tests for ordering and letterbox metadata**

```python
def test_select_images_uses_sorted_stable_ids(tmp_path: Path):
    images = []
    for name in ("z.jpg", "a.jpg", "m.jpg"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        images.append(path)
    selected = select_images(images, root=tmp_path, limit=2)
    assert [x.sample_id for x in selected] == sorted(x.sample_id for x in selected)
    assert len(selected) == 2


def test_letterbox_is_fixed_square(tmp_path: Path):
    image = Image.new("RGB", (80, 40), color="white")
    tensor, meta = load_letterboxed_tensor(image, imgsz=64)
    assert tensor.shape == (3, 64, 64)
    assert meta["original_shape"] == [40, 80]
    assert meta["resized_shape"] == [32, 64]
    assert meta["pad"] == [0, 16]
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `pytest -q tests/test_d1_foundation_cache.py -k 'select_images or letterbox'`

Expected: FAIL because the new functions are undefined.

- [ ] **Step 3: Implement selection and deterministic preprocessing**

The loader returns an RGB float32 CHW tensor in `[0, 1]`, uses bilinear resize, symmetric constant padding value `114/255`, and records original shape, resized shape, scale, and top-left pad. Selection sorts by stable sample ID before applying the limit.

- [ ] **Step 4: Run the full contract suite**

Run: `pytest -q tests/test_d1_foundation_cache.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ultralytics/data/foundation_cache.py tests/test_d1_foundation_cache.py
git commit -m "feat: add deterministic D1 cache preprocessing"
```

### Task 3: Multi-level DINOv3 extraction and atomic safetensors writer

**Files:**
- Create: `scripts/d1_build_feature_cache.py`
- Create: `tests/test_d1_feature_cache_builder.py`

- [ ] **Step 1: Write a failing fake-backbone integration test**

```python
def test_builder_writes_three_fp16_levels(tmp_path: Path):
    image = tmp_path / "images" / "sample.jpg"
    image.parent.mkdir()
    Image.new("RGB", (32, 24), color="red").save(image)
    result = build_cache(
        image_paths=[image],
        dataset_root=tmp_path,
        output_dir=tmp_path / "cache",
        extractor=FakeExtractor(),
        identity=CacheIdentity("fake", "rev", (3, 7, 11), 32, "fp"),
        limit=1,
    )
    tensors = load_file(result.samples[0]["cache_path"])
    assert set(tensors) == {"layer_3", "layer_7", "layer_11"}
    assert all(x.dtype == torch.float16 for x in tensors.values())
```

- [ ] **Step 2: Run test and confirm import failure**

Run: `pytest -q tests/test_d1_feature_cache_builder.py`

Expected: FAIL because the builder module does not exist.

- [ ] **Step 3: Implement `DINOv3MultiLevelExtractor`**

Load `DINOv3ViTBackbone.from_pretrained(model_id, revision=revision, out_indices=layers)`, freeze all parameters, enforce eval mode, reuse `prepare_image_tensor`, and call the backbone under `torch.inference_mode()`. Require exactly one finite BCHW feature map per requested layer and record actual channel/spatial shapes from outputs.

- [ ] **Step 4: Implement `build_cache`**

Write each sample to `<output>/features/<sample_id>.safetensors.tmp`, fsync, rename atomically, hash the final file, then update `manifest.json` atomically. Manifest sample records include image path/hash, tensor names/shapes/dtype, cache path/hash, letterbox metadata, and bytes. Non-deterministic timing and VRAM measurements are written only to `resource-report.json`, so repeated builds can produce identical manifests.

- [ ] **Step 5: Run fake-backbone tests**

Run: `pytest -q tests/test_d1_feature_cache_builder.py`

Expected: PASS without downloading a model.

- [ ] **Step 6: Commit**

```bash
git add scripts/d1_build_feature_cache.py tests/test_d1_feature_cache_builder.py
git commit -m "feat: build deterministic multi-level DINOv3 cache"
```

### Task 4: YOLO dataset CLI and admission reports

**Files:**
- Modify: `scripts/d1_build_feature_cache.py`
- Modify: `tests/test_d1_feature_cache_builder.py`

- [ ] **Step 1: Add failing CLI/report tests**

```python
def test_resource_report_contains_admission_fields(tmp_path: Path):
    report = build_resource_report(
        samples=[{"bytes": 1024, "extraction_seconds": 0.1}],
        peak_vram_bytes=2048,
        read_seconds=0.01,
    )
    assert report["sample_count"] == 1
    assert report["cache_total_bytes"] == 1024
    assert report["peak_vram_bytes"] == 2048
    assert report["sequential_read_mb_s"] > 0
```

- [ ] **Step 2: Run report tests and confirm failure**

Run: `pytest -q tests/test_d1_feature_cache_builder.py -k report`

Expected: FAIL because `build_resource_report` is undefined.

- [ ] **Step 3: Implement CLI dataset resolution and reports**

CLI arguments are `--dataset`, `--split`, `--limit`, `--imgsz`, `--model`, `--revision`, `--layers`, `--device`, `--dtype`, `--output`, and `--overwrite`. Resolve the dataset through `check_det_dataset`, recursively collect supported image files, build the cache, sequentially read every safetensors file once, and write `resource-report.json`, `interface-dimensions.json`, and `reproduce-command.txt`.

- [ ] **Step 4: Run all new tests and existing foundation tests**

Run: `pytest -q tests/test_d1_foundation_cache.py tests/test_d1_feature_cache_builder.py tests/test_foundation_distill_model.py tests/test_foundation_checkpoint.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/d1_build_feature_cache.py tests/test_d1_feature_cache_builder.py
git commit -m "feat: report D1 cache admission resources"
```

### Task 5: Real 4-image smoke, then 100-image admission cache

**Files:**
- Modify: `docs/d1/824-admission-status.md`
- Create at runtime: `runs/d1/admission-824/cache100/manifest.json`
- Create at runtime: `runs/d1/admission-824/cache100/resource-report.json`
- Create at runtime: `runs/d1/admission-824/cache100/interface-dimensions.json`

- [ ] **Step 1: Run a real four-image smoke**

```bash
python scripts/d1_build_feature_cache.py \
  --dataset coco8.yaml --split train --limit 4 --imgsz 128 \
  --model Tooony133/dinov3-vits16-pretrain-lvd1689m \
  --revision fc6921f7a0b44d5b33ab4482cfed5443db6ccd81 \
  --layers 3 7 11 --device cuda:0 --dtype fp16 \
  --output runs/d1/admission-824/cache4 \
  2>&1 | tee runs/d1/admission-824/cache4.log
```

Expected: 4 cache files, one manifest, finite three-level tensors, and no trainable teacher parameters.

- [ ] **Step 2: Validate the smoke cache**

Run: `python scripts/d1_build_feature_cache.py --validate-only --output runs/d1/admission-824/cache4`

Expected: `D1 cache validation: PASS (4 samples)`.

- [ ] **Step 3: Run the real 100-image admission build on COCO-mini**

```bash
python scripts/d1_build_feature_cache.py \
  --dataset ultralytics/cfg/datasets/coco-mini.yaml --split train --limit 100 --imgsz 640 \
  --model Tooony133/dinov3-vits16-pretrain-lvd1689m \
  --revision fc6921f7a0b44d5b33ab4482cfed5443db6ccd81 \
  --layers 3 7 11 --device cuda:0 --dtype fp16 \
  --output runs/d1/admission-824/cache100 \
  2>&1 | tee runs/d1/admission-824/cache100.log
```

Expected: 100 cache files plus the three admission reports. If `coco-mini.yaml` is absent, stop and create the dataset split explicitly; do not substitute coco8 because it has only four training images.

- [ ] **Step 4: Update admission status with measured values**

Copy exact values from the JSON reports into `docs/d1/824-admission-status.md`, including the cache model revision, sample count, total bytes, bytes/sample, sequential MB/s, peak VRAM, build seconds, and feature shapes.

- [ ] **Step 5: Commit code and measured admission documentation**

```bash
git add docs/d1/824-admission-status.md
git commit -m "docs: record D1 100-image cache admission evidence"
```
