"""CPU-only, GT-selected VisDrone image plates from frozen final predictions.

No model inference, training, threshold tuning, or result-dependent selection.
The figures are qualitative diagnostics, not AP or a complete error taxonomy.
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from online_experiments.analyze_detection_evidence import load_annotations, rows
from online_experiments.visdrone_reference_eval import match_thresholds, overlap_matrix, prepare_arrays

METHODS = ("blank25", "detail25", "lowpass25", "vit-adapter-cache25")
LABELS = ("Blank25", "Detail25", "Lowpass25", "Adapter25 (adaptation)")
COHORT_LABELS = {
    "small_gt": "Small-object count",
    "objects_per_megapixel": "Object density",
    "heavy_occluded_gt": "Heavy occlusion count",
    "truncated_gt": "Truncation count",
}
CONFIDENCE = 0.25
IOU = 0.5
COLORS = {"tp": "#0072B2", "fp": "#D55E00", "miss": "#CC79A7", "gt": "#56B4E9", "ignored": "#AAAAAA"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def matching(gt, det):
    """Class-wise score greedy; exact reference IoU/ignore and final-GT tie rule."""
    gt = np.asarray(gt, dtype=float).reshape(-1, 8)
    det = np.asarray(det, dtype=float).reshape(-1, 8)
    eligible = np.isin(gt[:, 5], np.arange(1, 11))
    positive = eligible & (gt[:, 4] == 0)
    matched = np.zeros(len(gt), dtype=bool)
    flags = np.zeros(len(det), dtype=int)
    for category in range(1, 11):
        gi = np.flatnonzero(eligible & (gt[:, 5] == category))
        di = np.flatnonzero(det[:, 5] == category)
        values = overlap_matrix(det[di], gt[gi])
        for j, d in enumerate(di):
            choices = np.flatnonzero(positive[gi] & ~matched[gi] & (values[j] >= IOU))
            if len(choices):
                best = choices[::-1][np.argmax(values[j, choices[::-1]])]
                matched[gi[best]] = True
                flags[d] = 1
            elif np.any((gt[gi, 4] != 0) & (values[j] >= IOU)):
                flags[d] = -1
    return flags, np.flatnonzero(positive & ~matched)


def crop_from_gt(gt, shape, cohort=None):
    """Fixed 512-pixel window; maximize cohort GT centers, never predictions."""
    h, w = map(int, shape)
    positive = (gt[:, 4] == 0) & np.isin(gt[:, 5], np.arange(1, 11))
    if cohort == "small_gt":
        positive &= gt[:, 2] * gt[:, 3] <= 1024
    elif cohort == "heavy_occluded_gt":
        positive &= gt[:, 7] == 2
    elif cohort == "truncated_gt":
        positive &= gt[:, 6] > 0
    boxes = gt[positive]
    if not len(boxes):
        boxes = gt[(gt[:, 4] == 0) & np.isin(gt[:, 5], np.arange(1, 11))]
    cw, ch = min(w, 512), min(h, 512)
    centers = boxes[:, :2] + boxes[:, 2:4] / 2
    candidates = {(max(0, min(w - cw, int(x - cw / 2))), max(0, min(h - ch, int(y - ch / 2)))) for x, y in centers}
    candidates.add(((w - cw) // 2, (h - ch) // 2))
    scored = []
    for x, y in candidates:
        count = int(
            np.count_nonzero(
                (centers[:, 0] >= x) & (centers[:, 0] < x + cw) & (centers[:, 1] >= y) & (centers[:, 1] < y + ch)
            )
        )
        scored.append((-count, y, x))
    _, y, x = min(scored)
    return [x, y, x + cw, y + ch]


def draw_boxes(axis, boxes, color, linestyle="-", linewidth=0.55):
    from matplotlib.patches import Rectangle

    for box in boxes:
        x, y, w, h = box[:4]
        if w <= 0 or h <= 0:
            axis.plot(x, y, "+", color=color, markersize=3)
        else:
            axis.add_patch(
                Rectangle((x, y), w, h, facecolor="none", edgecolor=color, linewidth=linewidth, linestyle=linestyle)
            )


def image_panel(axis, image, gt, result=None, crop=None):
    from PIL import Image

    h, w = image.shape[:2]
    if crop:
        x1, y1, x2, y2 = crop
        displayed = image[y1:y2, x1:x2]
        extent = [x1, x2, y2, y1]
    else:
        displayed = image
        extent = [0, w, h, 0]
        if max(h, w) > 1100:
            ratio = 1100 / max(h, w)
            displayed = np.asarray(
                Image.fromarray(image).resize((round(w * ratio), round(h * ratio)), Image.Resampling.LANCZOS)
            )
    axis.imshow(displayed, extent=extent, interpolation="nearest")
    if result is None:
        valid = (gt[:, 4] == 0) & np.isin(gt[:, 5], np.arange(1, 11))
        draw_boxes(axis, gt[valid], COLORS["gt"])
        draw_boxes(axis, gt[~valid], COLORS["ignored"], ":")
    else:
        det, flags, missing = result
        draw_boxes(axis, det[flags == 1], COLORS["tp"])
        draw_boxes(axis, det[flags == 0], COLORS["fp"])
        draw_boxes(axis, det[flags == -1], COLORS["ignored"], ":")
        draw_boxes(axis, gt[missing], COLORS["miss"], "--", 0.65)
    if crop:
        axis.set_xlim(crop[0], crop[2])
        axis.set_ylim(crop[3], crop[1])
    axis.set_axis_off()


def residual_image(heat_path, sample):
    """Project the native 80x80 energy map through the recorded letterbox."""
    from PIL import Image

    with np.load(heat_path) as maps:
        residual = maps["rgb_residual_p3_mean_squared"].astype(float)
        final = maps["final_p3_mean_squared"].astype(float)
        valid = maps["original_image_area_fraction"].astype(float)
        if residual.shape != (80, 80) or np.any(~np.isfinite(residual)) or np.any(residual < 0):
            raise ValueError(f"Invalid residual map: {heat_path}")
        denominator = float(np.sum(final * valid) / np.sum(valid))
        if denominator <= 0 or not np.isfinite(denominator):
            raise ValueError("Residual energy normalization needs positive finite final P3 energy")
        normalized = residual / denominator
    lb = sample["letterbox"]
    ph, pw = lb["pad"][1], lb["pad"][0]
    rh, rw = lb["resized_shape"]
    h, w = lb["original_shape"]
    canvas = np.asarray(Image.fromarray(normalized.astype(np.float32)).resize((640, 640), Image.Resampling.BILINEAR))
    original = Image.fromarray(canvas[ph : ph + rh, pw : pw + rw]).resize((w, h), Image.Resampling.BILINEAR)
    return np.asarray(original), denominator


def checks():
    gt = np.array([[0, 0, 10, 10, 0, 1, 0, 0], [20, 0, 10, 10, 1, 1, 0, 0]], dtype=float)
    det = np.array(
        [
            [0, 0, 10, 10, 0.9, 1, -1, -1],
            [0, 0, 10, 10, 0.8, 1, -1, -1],
            [20, 0, 10, 10, 0.7, 1, -1, -1],
            [0, 0, 10, 10, 0.6, 2, -1, -1],
        ],
        dtype=float,
    )
    flags, missing = matching(gt, det)
    assert flags.tolist() == [1, 0, -1, 0] and len(missing) == 0
    flags, missing = matching(gt, np.empty((0, 8)))
    assert len(flags) == 0 and missing.tolist() == [0]
    crop = crop_from_gt(gt, (100, 150))
    assert crop == [0, 0, 150, 100]
    gt2 = np.array([[900, 500, 20, 20, 0, 1, 1, 2]], dtype=float)
    assert crop_from_gt(gt2, (1080, 1920), "truncated_gt") == [654, 254, 1166, 766]
    d = det[det[:, 4] >= CONFIDENCE]
    assert len(d) == 4
    assert "torch" not in sys.modules
    return {
        "status": "PASS",
        "contracts": [
            "class-aware greedy one-to-one TP",
            "duplicate and wrong class remain FP",
            "same-class ignored object absorbs predictions",
            "unmatched positive GT is miss",
            "original-coordinate crop has no letterbox conversion",
            "fixed512 GT-only crop deterministic",
            "no torch import",
        ],
    }


def run(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from PIL import Image

    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError("Output must be fresh; prior results are preserved")
    args.output.mkdir(parents=True)
    (args.output / "status.txt").write_text("RUNNING CPU qualitative27\n")
    try:
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["DejaVu Sans"],
                "font.size": 8,
                "svg.fonttype": "none",
                "pdf.fonttype": 42,
            }
        )
        cohorts_path = args.analysis / "qualitative-cohorts-gt-only.json"
        cohorts = read(cohorts_path)
        if not cohorts["selection_before_prediction_inspection"] or cohorts["n_per_cohort"] != 8:
            raise ValueError("Require the existing predeclared GT-only top8 cohorts")
        selected = sorted({sid for ids in cohorts["cohorts"].values() for sid in ids})
        heat_selection = read(args.features / "heatmap-selection.json")
        if selected != sorted(heat_selection["sample_ids"]) or len(selected) != 27:
            raise ValueError("Exactly27 preselected samples must match prior heatmap selection")
        source = Path(read(args.analysis / "protocol.json")["input_source"])
        manifest_path = source / "dataset-manifest.json"
        manifest = read(manifest_path)
        if sha(manifest_path) != read(args.analysis / "protocol.json")["dataset_manifest_sha256"]:
            raise ValueError("Dataset manifest identity changed")
        records, annotation_hashes = load_annotations(source, args.archive)
        records = {r["id"]: r for r in records}
        samples = {Path(s["image_path"]).stem: s for s in manifest["splits"]["val"]}
        bindings, all_predictions = {}, {}
        spatial_protocol = read(args.features / "protocol.json")
        feature_completion = read(args.features / "completion.json")
        if (
            feature_completion["status"] != "SUCCEEDED"
            or feature_completion["images_each"] != 548
            or feature_completion["heatmap_n_images_per_method"] != 27
            or tuple(feature_completion["methods"]) != METHODS
        ):
            raise ValueError("Full548 feature stage with all27 heatmaps per method must be complete")
        if (
            sha(ROOT / "online_experiments/analyze_detection_evidence.py")
            != spatial_protocol["offline_analysis_source_sha256"]
        ):
            raise ValueError("Imported frozen analysis source changed")
        for method in METHODS:
            analysis = read(args.analysis / f"{method}.json")
            run_path = Path(analysis["run"])
            official_path = run_path / "official-final.json"
            official = read(official_path)
            if sha(official_path) != analysis["bindings"]["official_final_sha256"]:
                raise ValueError(f"Official evaluation identity changed: {method}")
            if official["annotation_sha256"] != annotation_hashes:
                raise ValueError("GT annotation identity differs between arms")
            sp = spatial_protocol["bindings"][method]
            if sp["checkpoint_sha256"] != official["checkpoint_sha256"] or sp["official_final_sha256"] != sha(
                official_path
            ):
                raise ValueError("Heatmap and prediction checkpoint identity differ")
            if sha(run_path / "weights/last.pt") != official["checkpoint_sha256"]:
                raise ValueError("Final checkpoint changed since prediction export")
            bindings[method] = {
                "run": str(run_path),
                "official_final_sha256": sha(official_path),
                "checkpoint_sha256": official["checkpoint_sha256"],
                "predictions": {},
                "heatmaps": {},
            }
            for sid in selected:
                pp = run_path / "predictions/last" / (sid + ".txt")
                hp = args.features / "heatmaps" / method / (sid + ".npz")
                pp_sha = sha(pp)
                if pp_sha != official["prediction_sha256"][sid]:
                    raise ValueError(f"Prediction hash changed: {method} {sid}")
                det = rows(pp.read_bytes())
                if len(det) > 500 or np.any(np.diff(det[:, 4]) > 0):
                    raise ValueError("Predictions must be score-descending max500")
                all_predictions[(method, sid)] = det[det[:, 4] >= CONFIDENCE]
                bindings[method]["predictions"][sid] = pp_sha
                bindings[method]["heatmaps"][sid] = sha(hp)
        legend = [
            Line2D(
                [0],
                [0],
                color=COLORS[k],
                lw=1.2,
                linestyle="--" if k == "miss" else ":" if k == "ignored" else "-",
                label=t,
            )
            for k, t in (
                ("gt", "GT"),
                ("tp", "Matched prediction"),
                ("fp", "Unmatched prediction"),
                ("miss", "Missed GT"),
                ("ignored", "Ignored"),
            )
        ]
        entries, counts = [], []
        main = [(key, ids[0]) for key, ids in cohorts["cohorts"].items()]
        fig_main, axes_main = plt.subplots(4, 5, figsize=(14.5, 11.3), squeeze=False)
        main_axes = {key: axes_main[i] for i, (key, _) in enumerate(main)}
        gallery = args.output / "gallery"
        gallery.mkdir()
        with PdfPages(args.output / "hard-scenes-gallery-27.pdf") as pdf:
            for index, sid in enumerate(selected):
                sample, record = samples[sid], records[sid]
                raw_image = Path(sample["image_path"])
                if sha(raw_image) != sample["image_sha256"]:
                    raise ValueError(f"Image identity changed: {sid}")
                with Image.open(raw_image) as opened:
                    image = np.asarray(opened.convert("RGB"))
                if list(image.shape[:2]) != record["shape"]:
                    raise ValueError("Original image coordinates/shape mismatch")
                gt, _ = prepare_arrays(record["gt"], np.empty((0, 8)), record["shape"])
                crop = crop_from_gt(gt, record["shape"])
                method_results, maps, norms = {}, [], []
                for method in METHODS:
                    _, det = prepare_arrays(record["gt"], all_predictions[(method, sid)], record["shape"])
                    flags, missing = matching(gt, det)
                    for category in range(1, 11):
                        gi, di = gt[:, 5] == category, det[:, 5] == category
                        expected_flags = match_thresholds(gt[gi, :5], det[di, :5])[0]
                        if not np.array_equal(flags[di], expected_flags):
                            raise ValueError("Qualitative flags differ from frozen reference")
                    method_results[method] = (det, flags, missing)
                    counts.append(
                        {
                            "sample_id": sid,
                            "method": method,
                            "GT": int(np.count_nonzero((gt[:, 4] == 0) & np.isin(gt[:, 5], np.arange(1, 11)))),
                            "TP": int(np.count_nonzero(flags == 1)),
                            "FP": int(np.count_nonzero(flags == 0)),
                            "FN": len(missing),
                            "ignored_predictions": int(np.count_nonzero(flags == -1)),
                            "raw_thresholded_predictions": len(all_predictions[(method, sid)]),
                            "ignored_region_removed_predictions": len(all_predictions[(method, sid)]) - len(det),
                        }
                    )
                    energy, denominator = residual_image(args.features / "heatmaps" / method / (sid + ".npz"), sample)
                    maps.append(energy)
                    norms.append(denominator)
                # A single shared per-image scale across arms; no local contrast changes.
                vmax = max(float(np.quantile(np.concatenate([m.ravel() for m in maps]), 0.99)), 1e-12)
                fig, axes = plt.subplots(3, 5, figsize=(14.5, 9.2), squeeze=False)
                for row in (0, 1):
                    cr = None if row == 0 else crop
                    image_panel(axes[row, 0], image, gt, crop=cr)
                    axes[row, 0].set_title("GT: full scene" if row == 0 else "GT: fixed512 crop", fontsize=9)
                    for col, (method, label) in enumerate(zip(METHODS, LABELS), 1):
                        result = method_results[method]
                        image_panel(axes[row, col], image, gt, result, cr)
                        c = counts[-4 + col - 1]
                        axes[row, col].set_title(
                            f"{label}\nTP {c['TP']} / FP {c['FP']} / FN {c['FN']} (full image)" if row == 0 else label,
                            fontsize=8,
                        )
                image_panel(axes[2, 0], image, gt, crop=crop)
                axes[2, 0].set_title("GT crop (repeated for alignment)", fontsize=8)
                for col, (energy, label) in enumerate(zip(maps, LABELS), 1):
                    ax = axes[2, col]
                    x1, y1, x2, y2 = crop
                    im = ax.imshow(
                        energy[y1:y2, x1:x2],
                        extent=[x1, x2, y2, y1],
                        cmap="magma",
                        vmin=0,
                        vmax=vmax,
                        interpolation="nearest",
                    )
                    ax.set_xlim(crop[0], crop[2])
                    ax.set_ylim(crop[3], crop[1])
                    ax.set_axis_off()
                    ax.set_title(f"{label}\nRGB residual energy", fontsize=8)
                color_axis = fig.add_axes([0.958, 0.10, 0.008, 0.20])
                colorbar = fig.colorbar(im, cax=color_axis)
                colorbar.set_label("Residual MSE / image mean final P3 MSE", fontsize=7)
                colorbar.ax.tick_params(labelsize=7)
                memberships = [key for key, ids in cohorts["cohorts"].items() if sid in ids]
                fig.suptitle(
                    f"{sid} | GT-selected: {', '.join(memberships)}\nFinal epoch100, seed0; score >=0.25; class-aware IoU >=0.50; fixed thresholds, no additional NMS",
                    fontsize=11,
                )
                fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=8)
                fig.subplots_adjust(left=0.015, right=0.945, bottom=0.06, top=0.89, wspace=0.04, hspace=0.20)
                pdf.savefig(fig, dpi=300)
                name = f"{index + 1:02d}-{sid}"
                fig.savefig(gallery / f"{name}.png", dpi=300)
                plt.close(fig)
                entries.append(
                    {
                        "sample_id": sid,
                        "image_sha256": sample["image_sha256"],
                        "original_shape": record["shape"],
                        "cohorts": memberships,
                        "gallery_crop_xyxy": crop,
                        "residual_final_energy_normalizers": dict(zip(METHODS, norms)),
                        "shared_heatmap_vmax": vmax,
                        "heatmap_clipped_fraction": dict(zip(METHODS, [float(np.mean(m > vmax)) for m in maps])),
                    }
                )
                for cohort, main_sid in main:
                    if main_sid != sid:
                        continue
                    cr = crop_from_gt(gt, record["shape"], cohort)
                    row_axes = main_axes[cohort]
                    image_panel(row_axes[0], image, gt, crop=cr)
                    row_axes[0].set_title(f"{COHORT_LABELS[cohort]}\nGT: {sid}", fontsize=8)
                    for col, (method, label) in enumerate(zip(METHODS, LABELS), 1):
                        image_panel(row_axes[col], image, gt, method_results[method], cr)
                        row_axes[col].set_title(label, fontsize=9)
                    entries[-1].setdefault("main_preview_crops", {})[cohort] = cr
                print(f"DONE qualitative {index + 1}/27 {sid}", flush=True)
        fig_main.suptitle(
            "GT-selected hard scenes: four fixed final-checkpoint methods\nEach cohort uses its predeclared rank1 image; 512x512 GT-only crop; seed0, score >=0.25, IoU >=0.50",
            fontsize=12,
        )
        fig_main.legend(handles=legend, loc="lower center", ncol=5, fontsize=9)
        fig_main.subplots_adjust(left=0.012, right=0.99, bottom=0.045, top=0.92, hspace=0.18, wspace=0.025)
        fig_main.savefig(args.output / "hard-scenes-main-preview.png", dpi=300)
        fig_main.savefig(args.output / "hard-scenes-main-preview.pdf", dpi=300)
        fig_main.savefig(args.output / "hard-scenes-main-preview.svg", dpi=300)
        plt.close(fig_main)
        with (args.output / "qualitative-match-counts.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(counts[0]))
            writer.writeheader()
            writer.writerows(counts)
        contract = {"status": "PASS", **checks()}
        dump(args.output / "contracts.json", contract)
        protocol = {
            "schema": "visdrone-gt-selected-qualitative-v1",
            "archetype": "image plate + quant",
            "claim": "Inspect successes and remaining failures on predeclared difficult images; no claim that the candidate wins every scene.",
            "analysis_job": str(args.analysis),
            "features_job": str(args.features),
            "figure_source_sha256": sha(__file__),
            "analysis_protocol_sha256": sha(args.analysis / "protocol.json"),
            "heatmap_protocol_sha256": sha(args.features / "protocol.json"),
            "selection_sha256": sha(cohorts_path),
            "dataset_manifest_sha256": sha(manifest_path),
            "selection": "union of predeclared top8 per GT-only cohort, 27 unique images; all retained",
            "main_selection": "first image in each original cohort; duplicate image in small/occluded rows intentionally retained",
            "prediction_independent_crop": "512 original-pixel window maximizing relevant GT-center count; tie: topmost then leftmost; no rescaling of coordinates",
            "confidence_threshold": CONFIDENCE,
            "IoU_threshold": IOU,
            "matching": "score-descending within class; valid positive GT first, best continuous IoU then final-GT tie; one-to-one TP; remaining same-class ignore may absorb multiple predictions; unmatched valid GT shown as missed; all unmatched predictions retained",
            "ignore_regions": "frozen reference prepare_arrays union-mask >=0.5 filter; original score-zero GT ignored; category11 not scored",
            "classification_errors": "wrong class appears as unmatched prediction plus missed GT; diagnostic is not full TIDE taxonomy",
            "no_extra_nms": True,
            "checkpoint": "fixed final epoch100, seed0",
            "heatmap": {
                "quantity": "RGB residual mean-squared P3 channel energy / per-image valid-area weighted mean final P3 channel energy",
                "projection": "bilinear80->640, remove recorded letterbox padding, bilinear to original shape",
                "limits": "zero to pooled4-method per-image99th percentile; clipped fractions reported, full arrays preserved",
                "colormap": "magma",
                "interpretation": "descriptive feature activation energy, not calibrated uncertainty or causal evidence; blank residual should be zero",
            },
            "image_processing": "unchanged RGB; no brightness, contrast, gamma, denoise, or retouch; crop coordinates recorded; uniform panel downsampling only",
            "methods": list(METHODS),
            "bindings": bindings,
            "samples": entries,
            "statistics": "single training seed; selected images not independent model replicates; no statistical significance claims",
            "no_gpu_no_torch": "torch" not in sys.modules,
        }
        dump(args.output / "source-selection-manifest.json", protocol)
        (args.output / "README.md").write_text(
            "# Fixed GT-selected qualitative evidence\n\n27 unique images (108 method-image comparisons), all four predefined top8 cohorts, fixed final epoch100 seed0. Main preview uses each cohort's original rank1; small/occluded rank1 is the same image and is intentionally repeated. Threshold0.25, class-aware IoU0.50, no threshold optimization or extra NMS.\n\nBlue solid prediction = matched; orange solid = unmatched prediction; magenta dashed GT = missed; gray dotted = ignored; cyan = GT. These are thresholded diagnostic matches, not official AP or a full error taxonomy. Retains duplicate, background, localization/classification failures rather than suppressing them.\n\nGallery pages show original full scenes, deterministic GT-only512px crops and RGB residual feature energy. Heatmap scalar normalization, shared limits, clipping fractions, provenance hashes, exact crop and matching rules are in source-selection-manifest.json. Feature energy is descriptive and cannot establish causal AP gains. Image adjustments: no brightness, contrast, gamma, denoising or retouch.\n\nPNG is an electronic review image; PDF/SVG retain editable vector overlays. The large full gallery is supplementary audit material; the main preview requires final manuscript resizing before submission. The raw photos remain only within these derived scientific panels and are not repackaged as a dataset.\n",
            encoding="utf-8",
        )
        files = {
            str(p.relative_to(args.output)): {"sha256": sha(p), "bytes": p.stat().st_size}
            for p in sorted(args.output.rglob("*"))
            if p.is_file() and p.name != "status.txt"
        }
        dump(
            args.output / "completion.json",
            {
                "status": "SUCCEEDED",
                "images": 27,
                "methods": 4,
                "method_image_pairs": 108,
                "seconds": time.monotonic() - started,
                "torch_imported": "torch" in sys.modules,
                "files": files,
            },
        )
        (args.output / "status.txt").write_text("SUCCEEDED CPU qualitative27 methods4 pairs108\n")
    except Exception as error:
        dump(args.output / "failure.json", {"type": type(error).__name__, "message": str(error)})
        (args.output / "status.txt").write_text(f"FAILED {type(error).__name__}: {error}\n")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("selfcheck", "run"))
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--archive", type=Path, default=ROOT / "datasets/VisDrone/VisDrone2019-DET-val.zip")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "selfcheck":
        print(json.dumps(checks(), indent=2))
    else:
        if not all((args.analysis, args.features, args.output)):
            parser.error("run requires --analysis --features --output")
        run(args)


if __name__ == "__main__":
    main()
