# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from threedgrut.model.ptir_helper import (
    append_ptir_metrics,
    compute_ptir_image_quality_metrics,
    linear_to_srgb,
    summarize_ptir_metrics,
)


# Existing inverse rendering methods such as R3DG and IRGS use this PSNR convention:
# compute PSNR separately for each RGB channel, then average the channel results.
IRGS_PSNR = False


def mse(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(
    img1: torch.Tensor,
    img2: torch.Tensor,
) -> torch.Tensor:
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def _to_chw_image(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(
                f"IRGS PSNR expects a single image batch, got shape {tuple(image.shape)}"
            )
        image = image.squeeze(0)

    if image.ndim != 3:
        raise ValueError(
            f"IRGS PSNR expects CHW or HWC image shape, got {tuple(image.shape)}"
        )

    if image.shape[0] == 3:
        return image
    if image.shape[-1] == 3:
        return image.permute(2, 0, 1)

    raise ValueError(
        f"IRGS PSNR expects 3 RGB channels, got shape {tuple(image.shape)}"
    )


class IRGSPeakSignalNoiseRatio(torch.nn.Module):
    def __init__(self, data_range: float = 1.0) -> None:
        super().__init__()
        self.data_range = data_range

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        img1 = _to_chw_image(img1)
        img2 = _to_chw_image(img2)
        return psnr(img1, img2).mean()


def create_psnr_criterion() -> torch.nn.Module:
    if IRGS_PSNR:
        return IRGSPeakSignalNoiseRatio(data_range=1.0)
    return PeakSignalNoiseRatio(data_range=1)


def create_image_quality_criterions(
    device: str | torch.device = "cuda",
) -> dict[str, torch.nn.Module]:
    return {
        "psnr": create_psnr_criterion().to(device),
        "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
        "lpips": LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=True
        ).to(device),
    }


def relight_environment_name(environment_path: str | Path) -> str:
    environment_path = Path(environment_path)
    return environment_path.name if environment_path.is_dir() else environment_path.stem


def _load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _strip_rgba_suffix(stem: str) -> str:
    for suffix in ("_rgba", "_rgb"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def resolve_relight_gt_path(
    dataset_root: str | Path,
    source_image_path: str | Path,
    environment_path: str | Path,
) -> Path:
    dataset_root = Path(dataset_root)
    source_image_path = Path(source_image_path)
    environment_name = relight_environment_name(environment_path)

    candidates = []
    test_rli_dir = dataset_root / "test_rli"
    if test_rli_dir.is_dir():
        frame_name = _strip_rgba_suffix(source_image_path.stem)
        candidates.append(test_rli_dir / f"{environment_name}_{frame_name}.png")

    candidates.append(source_image_path.parent / f"rgba_{environment_name}.png")

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Relight GT not found for environment '{environment_name}' and source image '{source_image_path}'. "
        f"Tried: {tried}"
    )


def _prepare_rgb_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim != 3:
        raise ValueError(f"Expected image with shape [H, W, C], got {image.shape}")
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] not in (3, 4):
        image = image[..., :3]

    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32) / float(np.iinfo(image.dtype).max)
    else:
        image = image.astype(np.float32, copy=False)
    return np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)


def read_relight_gt_image(
    image_path: str | Path,
    image_size: tuple[int, int],
    bg_color: str = "black",
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    image = _prepare_rgb_image(imageio.imread(image_path))
    if image.shape[-1] == 4:
        alpha = image[..., 3:4].clip(0.0, 1.0)
        image = image[..., :3]
        if bg_color == "black":
            image = image * alpha
        elif bg_color == "white":
            image = image * alpha + (1.0 - alpha)
        else:
            raise ValueError(f"{bg_color} is not a supported background color.")
    else:
        image = image[..., :3]

    height, width = image_size
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height))

    image = np.clip(image, 0.0, 1.0)
    image = (image * 255.0).astype(np.uint8).astype(np.float32) / 255.0
    return torch.from_numpy(image).to(device=device, dtype=torch.float32).unsqueeze(0)


@torch.no_grad()
def compute_relight_pbr_metrics(
    criterions: Mapping[str, Any],
    pred_pbr_linear: torch.Tensor,
    dataset,
    frame_index: int,
    environment_path: str | Path,
    bg_color: str = "black",
    prefix: str = "relight_pbr",
) -> tuple[dict[str, float], Path, torch.Tensor]:
    if not hasattr(dataset, "image_paths"):
        raise AttributeError(
            "Relight metrics require dataset.image_paths to resolve GT paths."
        )

    source_image_path = Path(str(dataset.image_paths[frame_index]))
    gt_path = resolve_relight_gt_path(
        dataset.root_dir, source_image_path, environment_path
    )

    pred_pbr_srgb = linear_to_srgb(pred_pbr_linear.clip(0.0, 1.0)).clip(0.0, 1.0)
    image_size = tuple(int(value) for value in pred_pbr_srgb.shape[1:3])
    gt = read_relight_gt_image(
        gt_path,
        image_size=image_size,
        bg_color=bg_color,
        device=pred_pbr_srgb.device,
    )
    metrics = compute_ptir_image_quality_metrics(criterions, pred_pbr_srgb, gt, prefix)
    return metrics, gt_path, gt


class Metric:
    def __init__(
        self, device: str | torch.device = "cuda", prefix: str = "relight_pbr"
    ) -> None:
        self.criterions = create_image_quality_criterions(device=device)
        self.prefix = prefix
        self.metric_lists: dict[str, list[float]] = {}
        self.gt_paths: list[str] = []
        self.last_gt_image: torch.Tensor | None = None
        self.last_gt_path: Path | None = None

    @torch.no_grad()
    def update_relight_pbr(
        self,
        pred_pbr_linear: torch.Tensor,
        dataset,
        frame_index: int,
        environment_path: str | Path,
        bg_color: str = "black",
    ) -> dict[str, float]:
        metrics, gt_path, gt_image = compute_relight_pbr_metrics(
            criterions=self.criterions,
            pred_pbr_linear=pred_pbr_linear,
            dataset=dataset,
            frame_index=frame_index,
            environment_path=environment_path,
            bg_color=bg_color,
            prefix=self.prefix,
        )
        append_ptir_metrics(self.metric_lists, metrics)
        self.gt_paths.append(str(gt_path.resolve()))
        self.last_gt_image = gt_image
        self.last_gt_path = gt_path
        return metrics

    def summarize_relight(self) -> dict[str, Any]:
        return summarize_ptir_metrics(self.metric_lists, include_values=False)

    def write_relight(
        self, output_dir: str | Path
    ) -> tuple[dict[str, Any], Path, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics = self.summarize_relight()
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        metrics_details_path = output_dir / "metrics_details.json"
        with open(metrics_details_path, "w") as f:
            json.dump(
                {
                    **{
                        key: [float(value) for value in values]
                        for key, values in self.metric_lists.items()
                    },
                    "relight_gt": self.gt_paths,
                },
                f,
                indent=2,
            )

        return metrics, metrics_path, metrics_details_path


def summarize_relight_directory(relight_dir: str | Path) -> dict[str, Any]:
    relight_dir = Path(relight_dir)
    if not relight_dir.is_dir():
        raise NotADirectoryError(f"Relight directory not found: {relight_dir}")

    metric_lists: dict[str, list[float]] = {}
    found_metrics = False

    for environment_dir in sorted(
        path for path in relight_dir.iterdir() if path.is_dir()
    ):
        metrics_details_path = environment_dir / "metrics_details.json"
        if not metrics_details_path.is_file():
            continue

        metrics_details = _load_json(metrics_details_path)
        for key, values in metrics_details.items():
            if key == "relight_gt":
                continue
            if not isinstance(values, list):
                continue
            found_metrics = True
            metric_lists.setdefault(key, []).extend(float(value) for value in values)

    if not found_metrics:
        raise FileNotFoundError(f"No relight metrics found under: {relight_dir}")

    return summarize_ptir_metrics(metric_lists, include_values=False)


def write_relight_summary_to_metrics(
    metrics_path: str | Path,
    relight_dir: str | Path,
) -> tuple[dict[str, Any], Path]:
    metrics_path = Path(metrics_path)
    existing_metrics = _load_json(metrics_path) if metrics_path.is_file() else {}
    relight_metrics = summarize_relight_directory(relight_dir)
    for stale_key in (
        "relight_num_environments",
        "relight_num_frames",
        "relight_per_environment",
    ):
        existing_metrics.pop(stale_key, None)

    merged_metrics = {
        **existing_metrics,
        **relight_metrics,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(merged_metrics, f, indent=2)

    return merged_metrics, metrics_path
