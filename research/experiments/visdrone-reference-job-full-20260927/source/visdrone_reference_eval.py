"""NumPy port of VisDrone DET toolkit commit 005445782213e20cb91bc50a597db3dd949e749a.

Preserve reference behavior, including occurrence-weighted class aggregation,
the ignored-GT denominator and global maxDets before class filtering.
This module consumes original-coordinate [x,y,w,h,score,class,truncation,occlusion].
"""

import numpy as np

THRESHOLDS = 0.5 + np.arange(10, dtype=np.float64) * 0.05
CAPS = (1, 10, 100, 500)
REFERENCE_COMMIT = "005445782213e20cb91bc50a597db3dd949e749a"


def prepare_arrays(gt, det, shape):
    """Port dropObjectsInIgr and saveAnnoRes without changing their pixel conventions."""
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 8).copy()
    det = np.asarray(det, dtype=np.float64).reshape(-1, 8).copy()
    if not np.isfinite(gt).all() or not np.isfinite(det).all():
        raise ValueError("Nonfinite annotation or detection")
    if len(det) and np.any(np.diff(det[:, 4]) > 0):
        raise ValueError("Global maxDets prefix optimization requires score-descending detections")
    height, width = map(int, shape)
    regions = np.maximum(1, gt[gt[:, 5] == 0, :4])
    gt = gt[gt[:, 5] != 0]
    if len(regions):
        if np.any(regions != np.floor(regions)):
            raise ValueError("Reference ignored-region rasterization requires integer coordinates")
        mask = np.zeros((height, width), dtype=np.int32)
        for x, y, w, h in regions.astype(np.int64):
            mask[y - 1 : min(height, y + h), x - 1 : min(width, x + w)] = 1
        integral = mask.cumsum(axis=0, dtype=np.int64).cumsum(axis=1, dtype=np.int64)

        def keep(boxes):
            # MATLAB round is half away from zero; max(1,round(...)) removes negative differences.
            positions = np.maximum(1, np.floor(boxes[:, :4] + 0.5)).astype(np.int64)
            x, y = np.minimum(positions[:, 0], width), np.minimum(positions[:, 1], height)
            w, h = positions[:, 2], positions[:, 3]
            right, bottom = np.minimum(width, x + w), np.minimum(height, y + h)
            area = (
                integral[y - 1, x - 1]
                + integral[bottom - 1, right - 1]
                - integral[y - 1, right - 1]
                - integral[bottom - 1, x - 1]
            )
            return area / (w * h) < 0.5

        gt, det = gt[keep(gt)], det[keep(det)]
    flags = gt[:, 4].copy()
    gt[flags == 0, 4] = 1
    gt[flags == 1, 4] = 0
    if np.any(~np.isin(gt[:, 4], [0, 1])):
        raise ValueError("Ground-truth score must be 0 or 1")
    return gt, det


def overlap_matrix(det, gt):
    """Reference compOas: continuous xywh IoU or intersection/detection-area for ignored GT."""
    if not len(det) or not len(gt):
        return np.zeros((len(det), len(gt)), dtype=np.float64)
    left = np.maximum(det[:, None, :2], gt[None, :, :2])
    right = np.minimum(det[:, None, :2] + det[:, None, 2:4], gt[None, :, :2] + gt[None, :, 2:4])
    size = np.maximum(0, right - left)
    inter = size[..., 0] * size[..., 1]
    da = det[:, 2] * det[:, 3]
    ga = gt[:, 2] * gt[:, 3]
    union = da[:, None] + ga[None] - inter
    denominator = np.where(gt[None, :, 4] != 0, da[:, None], union)
    return np.divide(inter, denominator, out=np.zeros_like(inter), where=(inter > 0) & (denominator > 0))


def match_thresholds(gt, det):
    """Greedy reference evalRes at all ten thresholds; input detections already sorted."""
    gt = gt[np.argsort(gt[:, 4], kind="stable")]
    matched = np.zeros((len(THRESHOLDS), len(det)), dtype=np.int8)
    if not len(gt) or not len(det):
        return matched
    overlaps = overlap_matrix(det, gt)
    state = np.broadcast_to(-gt[:, 4], (len(THRESHOLDS), len(gt))).copy()
    thresholds = THRESHOLDS[:, None]
    for d, values in enumerate(overlaps):
        eligible = (state == 0) & (values[None] >= thresholds)
        choices = np.where(eligible, values[None], -1)
        # The reference uses >=, so the final GT wins exact overlap ties.
        index = len(gt) - 1 - np.argmax(choices[:, ::-1], axis=1)
        good = choices[np.arange(len(THRESHOLDS)), index] >= 0
        matched[good, d] = 1
        state[np.flatnonzero(good), index[good]] = 1
        ignored = (~good) & np.any((state == -1) & (values[None] >= thresholds), axis=1)
        matched[ignored, d] = -1
    return matched


def voc_ap(recall, precision):
    """All-point precision-envelope integration used by the reference VOCap."""
    r = np.concatenate(([0.0], recall, [1.0]))
    p = np.concatenate(([0.0], precision, [0.0]))
    p = np.maximum.accumulate(p[::-1])[::-1]
    changes = np.flatnonzero(r[1:] != r[:-1]) + 1
    return float(np.sum((r[changes] - r[changes - 1]) * p[changes]))


def evaluate_prepared(cases):
    """Evaluate processed image arrays with the reference's exact aggregation."""
    ap = np.zeros((10, 10))
    ar = np.zeros((10, 10, 4))
    occurrences = np.zeros(10, dtype=int)
    gt_counts = np.zeros(10, dtype=int)
    for category in range(1, 11):
        records, flags, ranks = [], [], []
        for gt, det in cases:
            selected_gt = gt[gt[:, 5] == category, :5]
            occurrences[category - 1] += bool(len(selected_gt))
            gt_counts[category - 1] += len(selected_gt)
            prefix = det[:500]
            global_ranks = np.flatnonzero(prefix[:, 5] == category)
            selected_det = prefix[global_ranks, :5]
            records.append(selected_det[:, 4])
            ranks.append(global_ranks)
            flags.append(match_thresholds(selected_gt, selected_det))
        scores = np.concatenate(records)
        global_ranks = np.concatenate(ranks)
        matches = np.concatenate(flags, axis=1)
        order = np.argsort(-scores, kind="stable")
        denominator = max(1, gt_counts[category - 1])
        for t in range(10):
            ordered = matches[t, order]
            tp = np.cumsum(ordered == 1)
            fp = np.cumsum(ordered == 0)
            recall = tp / denominator
            precision = tp / np.maximum(1, tp + fp)
            ap[category - 1, t] = voc_ap(recall, precision) * 100
            for c, cap in enumerate(CAPS):
                ar[category - 1, t, c] = np.count_nonzero((matches[t] == 1) & (global_ranks < cap)) / denominator * 100
    if not occurrences.sum():
        raise ValueError("No evaluable ground-truth categories")
    weighted_ap = np.average(ap, axis=0, weights=occurrences)
    weighted_ar = np.average(ar, axis=0, weights=occurrences)
    return {
        "AP": float(weighted_ap.mean()),
        "AP50": float(weighted_ap[0]),
        "AP75": float(weighted_ap[5]),
        **{f"AR{cap}": float(weighted_ar[:, c].mean()) for c, cap in enumerate(CAPS)},
        "macro_AP_diagnostic": float(ap[occurrences > 0].mean()),
        "class_AP": ap.mean(axis=1).tolist(),
        "class_AP50": ap[:, 0].tolist(),
        "class_AP75": ap[:, 5].tolist(),
        "class_image_occurrences": occurrences.tolist(),
        "class_gt_counts_reference_denominator": gt_counts.tolist(),
        "aggregation": "official calcAccuracy occurrence-weighted classes, not uniform macro",
        "reference_commit": REFERENCE_COMMIT,
    }


def evaluate_raw(cases):
    """Evaluate original annotations; cases contain (GT, detections, (height,width))."""
    if not cases:
        raise ValueError("Empty image list")
    return evaluate_prepared([prepare_arrays(*case) for case in cases])
