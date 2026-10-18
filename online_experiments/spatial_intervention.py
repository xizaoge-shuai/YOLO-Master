"""Spatial support, local feature interventions and parameter-gradient summaries."""

import torch
import torch.nn.functional as F

from online_experiments.assignment_intervention import gradient_comparison
from online_experiments.feature_baselines import warp_features


def spatial_regions(letterbox, boxes, flip, scale, imgsz=640, grid_size=40):
    """Partition tokens using pixel support and feature interpolation; preserve visible fragments."""
    if imgsz % grid_size or scale <= 0:
        raise ValueError("Expected a positive scale and integral patch size")
    height, width = map(int, letterbox["resized_shape"])
    left, top = map(int, letterbox["pad"])
    if min(left, top) < 0 or left + width > imgsz or top + height > imgsz:
        raise ValueError("Invalid letterbox support")
    canvas = torch.ones(1, 1, imgsz, imgsz)
    content = torch.zeros_like(canvas)
    content[..., top : top + height, left : left + width] = 1
    theta = torch.tensor([[[1 / scale, 0.0, 0.0], [0.0, 1 / scale, 0.0]]])
    grid = F.affine_grid(theta, canvas.shape, align_corners=False)
    pixel_support = torch.cat([canvas, content], 1)
    if scale != 1:
        pixel_support = F.grid_sample(pixel_support, grid, align_corners=False)
    if flip:
        pixel_support = pixel_support.flip(-1)
    pooled = F.avg_pool2d(pixel_support, imgsz // grid_size)
    full, valid = pooled[:, :1], pooled[:, 1:]
    support = warp_features(torch.ones(1, 1, grid_size, grid_size), [False], [scale]).clamp(0, 1)
    epsilon = 1e-6
    full_canvas = (full >= 1 - epsilon) & (support >= 1 - epsilon)
    new = (full <= epsilon) & (support <= epsilon)
    old = full_canvas & (valid <= epsilon)
    interior = full_canvas & (valid >= 1 - epsilon)
    foreground = torch.zeros_like(interior)
    boxes = boxes.detach().cpu().float().reshape(-1, 4)
    low = ((boxes[:, :2] - boxes[:, 2:] / 2 - 0.5) * scale + 0.5).clamp(0, 1)
    high = ((boxes[:, :2] + boxes[:, 2:] / 2 - 0.5) * scale + 0.5).clamp(0, 1)
    if flip:
        low[:, 0], high[:, 0] = 1 - high[:, 0].clone(), 1 - low[:, 0].clone()
    edge = torch.arange(grid_size + 1) / grid_size
    for lo, hi in zip(low, high):
        if (hi <= lo).any():
            continue
        xs = (edge[:-1] < hi[0]) & (edge[1:] > lo[0])
        ys = (edge[:-1] < hi[1]) & (edge[1:] > lo[1])
        foreground[0, 0] |= ys[:, None] & xs[None, :]
    foreground &= interior
    background = interior & ~foreground
    mixed = ~(new | old | foreground | background)
    return {
        "new_padding": new,
        "letterbox": old,
        "foreground": foreground,
        "background": background,
        "mixed": mixed,
    }, support


def restore_region(transported, exact, mask):
    """Replace only masked tokens with an oracle donor; never change unmasked values."""
    return torch.where(mask.to(transported.device), exact, transported)


def blank_fill(transported, blank, support):
    """Restore the missing interpolation mass using a one-time gray-image template."""
    return transported + (1 - support.to(transported.device)) * blank


def parameter_groups(model):
    """Partition every parameter into shared pyramid or task-specific branch blocks."""
    prefixes = {
        "p3": "p3.",
        "p4": "p4.",
        "p5": "p5.",
        "one2many_box": "detect.cv2.",
        "one2many_cls": "detect.cv3.",
        "one2one_box": "detect.one2one_cv2.",
        "one2one_cls": "detect.one2one_cv3.",
    }
    groups = {key: [] for key in prefixes}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        matches = [key for key, prefix in prefixes.items() if name.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"Unexpected parameter block: {name}")
        groups[matches[0]].append((name, parameter))
    if not all(groups.values()):
        raise ValueError("Missing parameter block")
    return groups


def parameter_gradients(loss, groups, retain_graph=False):
    """Use autograd.grad without populating .grad or taking optimizer steps."""
    pairs = [item for group in groups.values() for item in group]
    parameters = [p for _, p in pairs]
    if loss.requires_grad:
        values = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=retain_graph)
    else:
        values = (None,) * len(parameters)
    by_id = {
        id(p): torch.zeros_like(p).flatten() if grad is None else grad.detach().flatten()
        for p, grad in zip(parameters, values)
    }
    result = {name: torch.cat([by_id[id(p)] for _, p in pairs]) for name, pairs in groups.items()}
    if not all(torch.isfinite(v).all() for v in result.values()):
        raise ValueError("Nonfinite parameter gradient")
    return result


def compare_gradient_blocks(value, reference):
    """Keep undefined zero-gradient comparisons explicit along with their norms."""
    if value.keys() != reference.keys():
        raise ValueError("Parameter group mismatch")
    return {
        key: gradient_comparison(value[key], reference[key])
        | {
            "value_norm": float(value[key].double().norm()),
            "reference_norm": float(reference[key].double().norm()),
            "parameters": value[key].numel(),
        }
        for key in value
    }


def regional_errors(value, reference, masks):
    """Return additive SSE and energy; normalized regional errors are not additive."""
    difference = (value - reference).double().square()
    energy = reference.double().square()
    total_sse, total_energy = float(difference.sum()), float(energy.sum())
    rows = []
    for name, mask in masks.items():
        mask = mask.to(value.device).expand_as(value)
        sse, denominator = float(difference[mask].sum()), float(energy[mask].sum())
        rows.append(
            {
                "region": name,
                "tokens": int(mask[:, :1].sum()),
                "sse": sse,
                "reference_energy": denominator,
                "global_sse": total_sse,
                "global_reference_energy": total_energy,
                "sse_fraction": sse / total_sse if total_sse else None,
                "regional_relative_mse": sse / denominator if denominator else None,
            }
        )
    return rows
