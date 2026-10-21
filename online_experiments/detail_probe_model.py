"""A zero-initialized image-detail residual for the cached detector's P3."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from scripts.d1_train_cached_detector import CachedLatentDetector


class DetailDetector(CachedLatentDetector):
    """Keep the cached detector intact and optionally add RGB detail to P3."""

    def __init__(
        self,
        *,
        in_channels: tuple[int, ...],
        nc: int,
        imgsz: int,
        epochs: int,
        detail_mode: str = "none",
    ) -> None:
        if detail_mode not in ("none", "rgb", "lowpass"):
            raise ValueError(f"detail_mode must be none, rgb, or lowpass, got {detail_mode!r}")
        super().__init__(in_channels=in_channels, nc=nc, imgsz=imgsz, epochs=epochs)
        self.detail_mode = detail_mode
        self.detail_branch = None
        if detail_mode != "none":
            # Construct after the parent and restore CPU RNG so paired arms keep
            # identical head weights and subsequent random sampling schedules.
            with torch.random.fork_rng(devices=[]):
                self.detail_branch = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(4, 16),
                    nn.SiLU(),
                    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, groups=16, bias=False),
                    nn.Conv2d(16, 32, kernel_size=1, bias=False),
                    nn.GroupNorm(4, 32),
                    nn.SiLU(),
                    nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, groups=32, bias=False),
                    nn.Conv2d(32, 32, kernel_size=1, bias=False),
                    nn.GroupNorm(4, 32),
                    nn.SiLU(),
                    nn.Conv2d(32, 64, kernel_size=1),
                )
                nn.init.zeros_(self.detail_branch[-1].weight)
                nn.init.zeros_(self.detail_branch[-1].bias)

    @property
    def detail_parameter_count(self) -> int:
        """Return the number of parameters added by the optional branch."""
        if self.detail_branch is None:
            return 0
        return sum(parameter.numel() for parameter in self.detail_branch.parameters())

    def prepare_detail_input(self, rgb: torch.Tensor) -> torch.Tensor:
        """Filter optional lowpass RGB, then center normalized letterbox images."""
        if rgb.ndim != 4 or rgb.shape[1] != 3 or tuple(rgb.shape[-2:]) != (self.imgsz, self.imgsz):
            raise ValueError(f"RGB must have shape (batch, 3, {self.imgsz}, {self.imgsz}), got {tuple(rgb.shape)}")
        if not rgb.is_floating_point():
            raise ValueError(f"RGB must have a floating dtype normalized to [0, 1], got {rgb.dtype}")
        if self.detail_mode == "lowpass":
            size = tuple(rgb.shape[-2:])
            rgb = F.interpolate(rgb, size=(size[0] // 16, size[1] // 16), mode="area")
            rgb = F.interpolate(rgb, size=size, mode="bilinear", align_corners=False)
        return (rgb - 114.0 / 255.0) * 2.0

    def pyramid(self, raw: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        """Consume three cached levels and, for detail modes, a fourth RGB tensor."""
        expected = 3 if self.detail_branch is None else 4
        if len(raw) != expected:
            extra = "" if self.detail_branch is None else " followed by one RGB tensor"
            raise ValueError(f"Expected three cached levels{extra}, got {len(raw)} inputs")
        if self.detail_branch is None:
            return super().pyramid(raw)
        detail_input = self.prepare_detail_input(raw[3])
        features = super().pyramid(raw[:3])
        if detail_input.shape[0] != features[0].shape[0]:
            raise ValueError("RGB batch size must match the cached P3 batch size")
        features[0] = features[0] + self.detail_branch(detail_input)
        return features
