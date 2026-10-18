"""Native-gradient and conditional Bernoulli variance diagnostics, without updates.

These quantities condition on one sample/transform and fixed detector parameters.
They do not establish minibatch stability, an unbiased optimizer update, or AP.
"""

import math

import torch

from online_experiments.spatial_intervention import blank_fill, parameter_gradients, restore_region


def native_parameter_gradients(model, criterion, groups, levels, targets):
    """Differentiate the actual training scalar using this arm's own native assignments.

    The caller supplies E2EDetectLoss; its first return value includes batch scaling
    and both branches. No assignment transplant, gradient clipping, or optimizer is
    used. Eval mode keeps BatchNorm/dropout fixed; autograd.grad preserves .grad.
    """
    if any(module.training for module in model.modules()):
        raise ValueError("Native gradient diagnostics require all modules in eval mode")
    parameters = [parameter for members in groups.values() for _, parameter in members]
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    ids = [id(parameter) for parameter in parameters]
    if not groups or not all(groups.values()) or len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError("Gradient groups must partition every trainable model parameter exactly once")
    predictions = model.training_predictions(levels)
    loss = criterion(predictions, targets)[0].sum()
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite native detection loss")
    gradients = parameter_gradients(loss, groups)
    return gradients, float(loss.detach())


def estimator_statistics(approximate, exact, p=0.25):
    """Enumerate the two outcomes of ga + Bernoulli(p)/p * (ge - ga).

    The __all__ row uses sums of squared norms and dot products across blocks,
    never averages block cosine similarities. Zero-reference comparisons are None.
    Variance is conditional on the supplied gradients, with no minibatch averaging.
    """
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 < p <= 1:
        raise ValueError("Query probability must satisfy 0 < p <= 1")
    if not approximate or approximate.keys() != exact.keys() or "__all__" in approximate:
        raise ValueError("Gradient block keys must match and cannot include reserved __all__")
    moments = {}
    for key, value in approximate.items():
        reference = exact[key]
        if (
            not isinstance(value, torch.Tensor)
            or not isinstance(reference, torch.Tensor)
            or value.shape != reference.shape
            or value.ndim != 1
            or value.numel() == 0
            or value.device != reference.device
            or not value.is_floating_point()
            or not reference.is_floating_point()
        ):
            raise ValueError(f"Invalid gradient block shape, device or dtype: {key}")
        if not torch.isfinite(value).all() or not torch.isfinite(reference).all():
            raise ValueError(f"Nonfinite gradient block: {key}")
        low, high = value.detach().double(), reference.detach().double()
        residual = high - low
        selected = high if p == 1 else low + residual / p
        expectation = (1 - p) * low + p * selected
        moments[key] = {
            "parameters": value.numel(),
            "approximate_squared_norm": float(low.square().sum()),
            "exact_squared_norm": float(high.square().sum()),
            "residual_squared_norm": float(residual.square().sum()),
            "selected_squared_norm": float(selected.square().sum()),
            "selected_error_squared_norm": float((selected - high).square().sum()),
            "expectation_error_squared_norm": float((expectation - high).square().sum()),
            "unselected_dot_exact": float((low * high).sum()),
            "selected_dot_exact": float((selected * high).sum()),
        }
        if not all(math.isfinite(number) for number in moments[key].values()):
            raise ValueError(f"Nonfinite estimator moment: {key}")
    moments["__all__"] = {name: sum(row[name] for row in moments.values()) for name in next(iter(moments.values()))}
    results = {}
    for key, row in moments.items():
        energy = row["exact_squared_norm"]
        residual_energy = row["residual_squared_norm"]
        variance = (1 - p) / p * residual_energy
        query_variance = (1 - p) / p * energy
        result = {
            name: row[name]
            for name in ("parameters", "approximate_squared_norm", "exact_squared_norm", "residual_squared_norm")
        }
        result.update(
            variance_trace=variance,
            query_only_variance_trace=query_variance,
            variance_ratio_vs_query_only=variance / query_variance if query_variance else None,
            expectation_relative_l2=math.sqrt(row["expectation_error_squared_norm"] / energy) if energy else None,
        )
        for name, norm, error in (
            ("unselected", row["approximate_squared_norm"], residual_energy),
            ("selected", row["selected_squared_norm"], row["selected_error_squared_norm"]),
        ):
            denominator = math.sqrt(norm) * math.sqrt(energy)
            result[f"{name}_cosine"] = (
                max(-1.0, min(1.0, row[f"{name}_dot_exact"] / denominator)) if denominator else None
            )
            result[f"{name}_norm_ratio"] = math.sqrt(norm / energy) if energy else None
            result[f"{name}_relative_l2"] = math.sqrt(error / energy) if energy else None
        if any(number is not None and not math.isfinite(number) for number in result.values()):
            raise ValueError(f"Nonfinite estimator statistic: {key}")
        results[key] = result
    return results


def joint_spatial_arms(transported, exact, masks, support, blank):
    """Combine exact interior content with either exact or cheap new padding.

    Tensor inputs return tensors; matched tuple/list inputs return tuples of levels.
    Joint oracle restores only new-padding and full-content tokens; mixed and old
    letterbox tokens remain transported. Blank fill also covers fractional support
    on mixed boundaries, then exact foreground/background replace the interior.
    """
    inputs = (transported, exact, blank)
    if any(isinstance(value, (tuple, list)) for value in inputs):
        if (
            not all(isinstance(value, (tuple, list)) for value in inputs)
            or not transported
            or len(transported) != len(exact)
            or len(transported) != len(blank)
            or not all(isinstance(level, torch.Tensor) for value in inputs for level in value)
        ):
            raise ValueError("Feature level sequences must contain matching nonempty tensor levels")
        arms = [joint_spatial_arms(low, high, masks, support, template) for low, high, template in zip(*inputs)]
        return {name: tuple(arm[name] for arm in arms) for name in arms[0]}
    content = masks["foreground"] | masks["background"]
    return {
        "joint_oracle": restore_region(transported, exact, masks["new_padding"] | content),
        "blank_content_oracle": restore_region(blank_fill(transported, blank, support), exact, content),
    }
