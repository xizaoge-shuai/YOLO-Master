"""A reduced, cache-side adaptation of the ViT-Adapter interaction idea.

Chen et al., Vision Transformer Adapter for Dense Predictions, ICLR 2023,
arXiv:2205.08534v4, pp. 4--5: spatial prior module, spatial feature injector,
and multi-scale feature extractor. This adaptation uses width 64, three
interactions, and a frozen layer-3/7/11 cache. It does not update or recompute
interleaved ViT blocks, and is not the original ViT-Adapter architecture.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from online_experiments.detail_probe_model import DetailDetector
from scripts.d1_train_cached_detector import CachedLatentDetector


def _tokens(features: torch.Tensor) -> torch.Tensor:
    """Flatten BCHW maps into B(HW)C tokens."""
    return features.flatten(2).transpose(1, 2)


def _maps(tokens: torch.Tensor, shapes: tuple[tuple[int, int], ...]) -> list[torch.Tensor]:
    """Recover the spatial levels in the order used for token concatenation."""
    sizes = [height * width for height, width in shapes]
    return [
        part.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], height, width)
        for part, (height, width) in zip(tokens.split(sizes, dim=1), shapes)
    ]


def _reference_grid(height: int, width: int, like: torch.Tensor) -> torch.Tensor:
    """Use pixel centers in normalized [0, 1] image coordinates, ordered x,y."""
    y = (torch.arange(height, device=like.device, dtype=like.dtype) + 0.5) / height
    x = (torch.arange(width, device=like.device, dtype=like.dtype) + 0.5) / width
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2)


class GridSparseAttention(nn.Module):
    """Multi-scale deformable cross-attention expressed with native grid_sample.

    Queries predict offsets and weights for four sampling points per head per
    source level. Offsets are measured in each source level's feature pixels.
    Sampling uses zero padding and align_corners=False; reference points use
    the corresponding pixel-center convention.
    """

    def __init__(self, levels: int, width: int = 64, heads: int = 4, points: int = 4) -> None:
        super().__init__()
        if levels < 1 or points < 1 or heads < 1 or width % heads:
            raise ValueError("Sparse-attention dimensions must be positive and width divisible by heads")
        self.width, self.heads, self.points, self.levels = width, heads, points, levels
        self.value_projection = nn.Linear(width, width)
        self.offsets = nn.Linear(width, heads * levels * points * 2)
        self.weights = nn.Linear(width, heads * levels * points)
        self.output_projection = nn.Linear(width, width)
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.weights.weight)
        nn.init.zeros_(self.weights.bias)
        # Distinct head directions and increasing radii avoid coincident initial
        # samples while preserving a local spatial prior at initialization.
        angles = torch.arange(heads, dtype=torch.float32) * (2 * math.pi / heads)
        directions = torch.stack((angles.cos(), angles.sin()), dim=-1)
        directions /= directions.abs().amax(dim=-1, keepdim=True)
        radii = torch.arange(1, points + 1, dtype=torch.float32)
        bias = directions[:, None, None, :] * radii[None, None, :, None]
        bias = bias.expand(heads, levels, points, 2).reshape(-1)
        with torch.no_grad():
            self.offsets.bias.copy_(bias)

    def forward(
        self,
        query: torch.Tensor,
        values: list[torch.Tensor],
        references: torch.Tensor,
    ) -> torch.Tensor:
        """Attend from BQD queries to a list of normalized BCHW value maps."""
        if query.ndim != 3 or query.shape[-1] != self.width or len(values) != self.levels:
            raise ValueError("Sparse-attention query width or source level count mismatch")
        batch, count, _ = query.shape
        if tuple(references.shape) != (count, 2):
            raise ValueError("Reference points must have shape (query_count, 2)")
        offsets = self.offsets(query).reshape(batch, count, self.heads, self.levels, self.points, 2)
        weights = self.weights(query).reshape(batch, count, self.heads, self.levels * self.points)
        weights = weights.softmax(dim=-1).reshape(batch, count, self.heads, self.levels, self.points)
        channels = self.width // self.heads
        result = query.new_zeros((batch, count, self.heads, channels))
        for level, value in enumerate(values):
            if value.ndim != 4 or value.shape[:2] != (batch, self.width):
                raise ValueError("Sparse-attention values must have matching batch and width")
            height, width = value.shape[-2:]
            projected = self.value_projection(_tokens(value))
            projected = projected.reshape(batch, height, width, self.heads, channels)
            projected = projected.permute(0, 3, 4, 1, 2).reshape(batch * self.heads, channels, height, width)
            normalizer = query.new_tensor((width, height))
            locations = references[None, :, None, None, :] + offsets[:, :, :, level] / normalizer
            grid = locations.permute(0, 2, 1, 3, 4).reshape(batch * self.heads, count, self.points, 2)
            sampled = F.grid_sample(
                projected,
                grid * 2 - 1,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled = sampled.reshape(batch, self.heads, channels, count, self.points)
            sampled = sampled.permute(0, 3, 1, 4, 2)
            result = result + (sampled * weights[:, :, :, level, :, None]).sum(dim=3)
        return self.output_projection(result.reshape(batch, count, self.width))


def _conv(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    """A convolutional spatial-prior block without batch-size-dependent moments."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(8, out_channels),
        nn.GELU(),
    )


class SpatialPrior(nn.Module):
    """Three-convolution stem and 1/8, 1/16, 1/32 RGB features (paper p.4)."""

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(_conv(3, 32, 2), _conv(32, 32), _conv(32, 32), nn.MaxPool2d(3, 2, 1))
        self.stage8 = _conv(32, 64, 2)
        self.stage16 = _conv(64, 128, 2)
        self.stage32 = _conv(128, 128, 2)
        self.projections = nn.ModuleList(nn.Conv2d(channels, width, 1) for channels in (64, 128, 128))

    def forward(self, rgb: torch.Tensor) -> list[torch.Tensor]:
        spatial8 = self.stage8(self.stem(rgb))
        spatial16 = self.stage16(spatial8)
        spatial32 = self.stage32(spatial16)
        return [projection(value) for projection, value in zip(self.projections, (spatial8, spatial16, spatial32))]


class CacheInteraction(nn.Module):
    """One spatial-to-semantic injection and semantic-to-spatial extraction."""

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.inject_query_norm = nn.LayerNorm(width)
        self.inject_value_norm = nn.LayerNorm(width)
        self.inject = GridSparseAttention(levels=3, width=width)
        self.gamma = nn.Parameter(torch.zeros(width))
        self.extract_query_norm = nn.LayerNorm(width)
        self.extract_value_norm = nn.LayerNorm(width)
        self.extract = GridSparseAttention(levels=1, width=width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, width // 4), nn.GELU(), nn.Linear(width // 4, width))

    def forward(
        self,
        semantic: torch.Tensor,
        spatial: torch.Tensor,
        spatial_shapes: tuple[tuple[int, int], ...],
        semantic_shape: tuple[int, int],
        semantic_references: torch.Tensor,
        spatial_references: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        injected = semantic + self.gamma * self.inject(
            self.inject_query_norm(semantic),
            _maps(self.inject_value_norm(spatial), spatial_shapes),
            semantic_references,
        )
        extracted = spatial + self.extract(
            self.extract_query_norm(spatial),
            _maps(self.extract_value_norm(injected), (semantic_shape,)),
            spatial_references,
        )
        return injected, extracted + self.ffn(self.ffn_norm(extracted))


class CacheSideAdapter(nn.Module):
    """Maintain trainable spatial state across three immutable cached ViT levels.

    At step i, semantic = project(cache_i) + previous_semantic_residual. The
    injector updates this semantic state; its difference from project(cache_i)
    is passed to step i+1. The extractor's complete spatial output is also
    carried to the next step. No trainable residual is fed through a ViT block.
    """

    def __init__(self, in_channels: tuple[int, ...], width: int = 64) -> None:
        super().__init__()
        self.spatial_prior = SpatialPrior(width)
        self.semantic_projections = nn.ModuleList(nn.Conv2d(channels, width, 1) for channels in in_channels)
        self.interactions = nn.ModuleList(CacheInteraction(width) for _ in in_channels)
        self.output_projections = nn.ModuleList(nn.Conv2d(width, channels, 1) for channels in (64, 128, 256))
        for projection in self.output_projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, raw: tuple[torch.Tensor, ...], rgb: torch.Tensor) -> list[torch.Tensor]:
        maps = self.spatial_prior(rgb)
        shapes = tuple(tuple(value.shape[-2:]) for value in maps)
        spatial = torch.cat([_tokens(value) for value in maps], dim=1)
        spatial_references = torch.cat([_reference_grid(*shape, spatial) for shape in shapes])
        semantic_shape = tuple(raw[0].shape[-2:])
        semantic_references = _reference_grid(*semantic_shape, spatial)
        residual = None
        for cached, projection, interaction in zip(raw, self.semantic_projections, self.interactions):
            base = _tokens(projection(cached))
            semantic = base if residual is None else base + residual
            injected, spatial = interaction(
                semantic, spatial, shapes, semantic_shape, semantic_references, spatial_references
            )
            residual = injected - base
        return [projection(value) for projection, value in zip(self.output_projections, _maps(spatial, shapes))]


class PaperAdapterDetector(DetailDetector):
    """Same cached head with a reduced ViT-Adapter-style RGB interaction branch."""

    def __init__(
        self,
        *,
        in_channels: tuple[int, ...],
        nc: int,
        imgsz: int,
        epochs: int,
        detail_mode: str = "rgb",
    ) -> None:
        if imgsz != 640 or in_channels != (384, 384, 384):
            raise ValueError("Cache-side paper adapter expects 640px images and three 384-channel cached levels")
        if detail_mode != "rgb":
            raise ValueError("Cache-side paper adapter requires RGB input at training and validation")
        super().__init__(in_channels=in_channels, nc=nc, imgsz=imgsz, epochs=epochs, detail_mode="none")
        self.detail_mode = "rgb"
        # Base-head initialization and subsequent global RNG sampling remain
        # identical to the existing detector; the branch has its own draw stream.
        with torch.random.fork_rng(devices=[]):
            self.detail_branch = CacheSideAdapter(in_channels)
        self.paper_contract = {
            "source": "2205.08534v4, Sections 3 and 4.4",
            "adaptation": "cache-side SPM/injector/extractor; no interleaved ViT blocks",
            "spatial_strides": [8, 16, 32],
            "width": 64,
            "interactions": 3,
            "attention_heads": 4,
            "sampling_points_per_head_per_level": 4,
            "extractor_ffn_ratio": 0.25,
            "output_channels": [64, 128, 256],
            "injector_gamma_init": 0.0,
            "output_projection_init": 0.0,
            "rgb_required_in_training_and_validation": True,
        }

    def pyramid(self, raw: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        if len(raw) != 4:
            raise ValueError("Cache-side paper adapter expects three cached levels followed by one RGB tensor")
        cached, rgb = raw[:3], raw[3]
        detail_input = self.prepare_detail_input(rgb)
        for value in cached:
            if (
                value.ndim != 4
                or value.shape[:2] != (rgb.shape[0], 384)
                or tuple(value.shape[-2:]) != (40, 40)
                or not value.is_floating_point()
                or value.device != rgb.device
                or value.dtype != rgb.dtype
            ):
                raise ValueError("Cached inputs must be matching floating Bx384x40x40 tensors on the RGB device")
        base = CachedLatentDetector.pyramid(self, cached)
        residuals = self.detail_branch(cached, detail_input)
        return [feature + residual for feature, residual in zip(base, residuals)]
