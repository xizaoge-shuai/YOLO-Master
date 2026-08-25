"""Tests for the D1 offline-cache detector training entry point."""

from __future__ import annotations

import torch

from scripts.d1_train_cached_detector import CachedLatentDetector, audit_model, transform_letterbox_xywhn
from ultralytics.nn.mixture_loss import _collect_mixture_aux_loss
from ultralytics.utils.loss import E2EDetectLoss


def test_letterbox_label_transform_matches_cache_geometry():
    boxes = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    metadata = {
        "original_shape": [100, 200],
        "resized_shape": [50, 100],
        "scale": 0.5,
        "pad": [0, 25],
    }

    transformed = transform_letterbox_xywhn(boxes, metadata, imgsz=100)

    assert transformed.tolist() == [[0.5, 0.5, 0.5, 0.25]]


def test_cached_detector_uses_real_yolo_loss_and_backpropagates():
    model = CachedLatentDetector(in_channels=(8, 8, 8), nc=2, imgsz=64, epochs=1)
    model.train()
    raw = tuple(torch.randn(1, 8, 4, 4) for _ in range(3))
    batch = {
        "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
    }

    predictions = model.training_predictions(raw)
    loss_vector, loss_items = E2EDetectLoss(model)(predictions, batch)
    aux = _collect_mixture_aux_loss(model, torch.device("cpu"), latent_gain=1.0, aux_budget=3.0)
    total = loss_vector.sum() + 0.1 * aux
    total.backward()

    assert torch.isfinite(total)
    assert torch.isfinite(loss_items).all()
    assert model.p3.token_projs[0][0].weight.grad is not None
    assert model.detect.cv3[0][-1].weight.grad is not None


def test_trainability_audit_excludes_teacher():
    model = CachedLatentDetector(in_channels=(8, 8, 8), nc=2, imgsz=64, epochs=1)

    report = audit_model(model)

    assert report["status"] == "PASS"
    assert report["teacher_loaded"] is False
    assert report["teacher_parameters"] == []
    assert report["unexpected_trainable"] == []
