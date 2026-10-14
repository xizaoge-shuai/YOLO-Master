"""Detached assignment interventions for frozen-head diagnostics, without training edits."""

import hashlib
import json

import torch
import torch.nn.functional as F

from ultralytics.utils.tal import make_anchors


def state_hash(model):
    """Hash tensor parameters/buffers and nested module extra state deterministically."""
    digest = hashlib.sha256()

    def update(value):
        if isinstance(value, torch.Tensor):
            digest.update(f"tensor:{value.dtype}:{tuple(value.shape)}:".encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            digest.update(b"dict{")
            for key in sorted(value, key=str):
                update(key)
                update(value[key])
            digest.update(b"}")
        elif isinstance(value, (tuple, list)):
            digest.update(type(value).__name__.encode() + b"[")
            for item in value:
                update(item)
            digest.update(b"]")
        else:
            digest.update(json.dumps(value, sort_keys=True, allow_nan=False).encode() + b";")

    update(model.state_dict())
    return digest.hexdigest()


def capture_assignment(criterion, predictions, batch):
    """Capture native targets and loss without changing the assigner or predictions."""
    captured = []

    def hook(module, inputs, output):
        captured.append(tuple(x.detach().clone() for x in output))

    handle = criterion.assigner.register_forward_hook(hook)
    try:
        with torch.no_grad():
            _, components = criterion(predictions, batch)
    finally:
        handle.remove()
    if len(captured) != 1 or predictions["scores"].shape[0] != 1:
        raise ValueError("This diagnostic requires exactly one image and one assignment call")
    return captured[0], components.detach()


def fixed_loss(criterion, predictions, assignment, protocol, normalizer=None):
    """Evaluate native soft targets or unit hard targets under a fixed donor assignment."""
    if predictions["scores"].shape[0] != 1:
        raise ValueError("Per-image normalization requires batch=1")
    labels, target_boxes, target_scores, foreground, _ = assignment
    foreground = foreground.bool()
    distribution = predictions["boxes"].permute(0, 2, 1).contiguous()
    logits = predictions["scores"].permute(0, 2, 1).contiguous()
    anchors, strides = make_anchors(predictions["feats"], criterion.stride, 0.5)
    image_size = logits.new_tensor(predictions["feats"][0].shape[2:]) * criterion.stride[0]
    boxes = criterion.bbox_decode(anchors, distribution)
    if protocol == "native_soft":
        denominator = target_scores.sum().clamp_min(1)
    elif protocol == "hard_common":
        if normalizer is None or float(normalizer) < 1:
            raise ValueError("hard_common needs a fixed positive common denominator >=1")
        if ((labels[foreground] < 0) | (labels[foreground] >= criterion.nc)).any():
            raise ValueError("Invalid positive class")
        target_scores = F.one_hot(labels.long().clamp(0, criterion.nc - 1), criterion.nc).to(logits.dtype)
        target_scores = target_scores * foreground[..., None]
        denominator = normalizer
    else:
        raise ValueError(f"Unknown protocol: {protocol}")
    bce = criterion.bce(logits, target_scores.to(logits.dtype))
    if criterion.class_weights is not None:
        bce = bce * criterion.class_weights
    classification = bce.sum() / denominator
    box, dfl = logits.new_zeros(()), logits.new_zeros(())
    if foreground.any():
        box, dfl = criterion.bbox_loss(
            distribution,
            boxes,
            anchors,
            target_boxes / strides,
            target_scores,
            denominator,
            foreground,
            image_size,
            strides,
        )
    return torch.stack((box * criterion.hyp.box, classification * criterion.hyp.cls, dfl * criterion.hyp.dfl))


def loss_and_gradient(criterion, predictions, assignment, protocol, normalizer=None):
    """Return gradients in raw output coordinates, not feature/parameter gradients."""
    leaves = {key: predictions[key].detach().clone().requires_grad_(True) for key in ("scores", "boxes")}
    raw = {**predictions, **leaves}
    loss = fixed_loss(criterion, raw, assignment, protocol, normalizer)
    gradients = torch.autograd.grad(loss.sum(), tuple(leaves.values()), allow_unused=True)
    result = {
        name: (torch.zeros_like(leaf) if grad is None else grad.detach()).flatten()
        for name, leaf, grad in zip(("cls", "box"), leaves.values(), gradients)
    }
    if not torch.isfinite(loss).all() or not all(torch.isfinite(g).all() for g in result.values()):
        raise ValueError("Non-finite loss or gradient")
    return loss.detach(), result


def gradient_comparison(value, reference):
    """Compare two gradients; undefined ratios/cosines are missing, not perfect."""
    value, reference = value.double(), reference.double()
    nv, nr = value.norm(), reference.norm()
    return {
        "cosine": float(torch.dot(value, reference) / (nv * nr)) if nv > 0 and nr > 0 else None,
        "relative_l2": float((value - reference).norm() / nr) if nr > 0 else None,
        "norm_ratio": float(nv / nr) if nr > 0 else None,
    }


def assignment_metrics(exact, approximate, gt_areas):
    """Describe assignment agreement and input-space object-size coverage for one image."""
    ef, af = exact[3].bool()[0], approximate[3].bool()[0]
    ei, ai = exact[4].long()[0], approximate[4].long()[0]

    def jaccard(a, b):
        union = int((a | b).sum())
        return float((a & b).sum()) / union if union else None

    shared = ef & af
    output = {
        "fg_jaccard": jaccard(ef, af),
        "exact_positives": int(ef.sum()),
        "approx_positives": int(af.sum()),
        "shared_positives": int(shared.sum()),
        "gt_switch_fraction": float((ei[shared] != ai[shared]).float().mean()) if shared.any() else None,
        "exact_score_sum": float(exact[2].sum()),
        "approx_score_sum": float(approximate[2].sum()),
    }
    for name, low, high in (("small", 0, 32**2), ("medium", 32**2, 96**2), ("large", 96**2, float("inf"))):
        size_gts = (gt_areas > low) & (gt_areas <= high)
        masks = []
        for label, assignment, fg, ids in (("exact", exact, ef, ei), ("approx", approximate, af, ai)):
            area = (assignment[1][0, :, 2:] - assignment[1][0, :, :2]).prod(-1)
            masks.append(fg & (area > low) & (area <= high))
            covered = torch.zeros_like(size_gts)
            if fg.any():
                covered[ids[fg]] = True
            output[f"{label}_coverage_{name}"] = (
                float((covered & size_gts).sum()) / int(size_gts.sum()) if size_gts.any() else None
            )
        output[f"gt_count_{name}"] = int(size_gts.sum())
        output[f"fg_jaccard_{name}"] = jaccard(*masks)
    return output
