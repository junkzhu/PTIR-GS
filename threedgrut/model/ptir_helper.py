# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import torch

from threedgrut.utils.logger import logger
from threedgrut.utils.normal import normal_mae

if TYPE_CHECKING:
    from threedgrut.model.model import MixtureOfGaussians


def srgb_to_linear(image: torch.Tensor) -> torch.Tensor:
    """Convert sRGB values in [0, 1] to linear RGB."""
    out = torch.empty_like(image)
    linear = image <= 0.04045
    out[linear] = image[linear] / 12.92
    out[~linear] = ((image[~linear] + 0.055) / 1.055).pow(2.4)
    return out


def linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(image)
    linear = image <= 0.0031308
    out[linear] = 12.92 * image[linear]
    out[~linear] = 1.055 * image[~linear].pow(1.0 / 2.4) - 0.055
    return out


PBR_GT_MASK_APPLIED_KEY = "_pbr_gt_mask_applied"
PBR_GT_MASK_OUTPUT_KEYS = (
    "pred_pbr",
    "pred_direct",
    "pred_indirect",
    "pred_light",
    "pred_shadingnormal",
    "pred_material",
)


def apply_gt_mask_to_tensor(
    tensor: torch.Tensor | None, mask: torch.Tensor | None
) -> torch.Tensor | None:
    """Apply an NHWC foreground GT mask to an image-like tensor."""
    if tensor is None:
        return None
    if mask is None:
        return tensor
    if tensor.ndim != 4:
        return tensor

    mask = mask.detach().to(device=tensor.device, dtype=tensor.dtype).clamp(0.0, 1.0)
    if mask.ndim == 2:
        mask = mask[None, :, :, None]
    elif mask.ndim == 3:
        if mask.shape[-1] == 1 and mask.shape[:2] == tensor.shape[1:3]:
            mask = mask.unsqueeze(0)
        else:
            mask = mask.unsqueeze(-1)
    elif mask.ndim != 4:
        raise ValueError(f"GT mask must be 2D, 3D, or 4D, got {mask.ndim}D")

    if mask.shape[-1] != 1:
        mask = mask[..., :1]

    if mask.shape[0] == 1 and tensor.shape[0] != 1:
        mask = mask.expand(tensor.shape[0], -1, -1, -1)

    if mask.shape[:3] != tensor.shape[:3]:
        raise ValueError(
            f"GT mask shape {tuple(mask.shape)} is not compatible with {tuple(tensor.shape)}"
        )

    return tensor * mask


def post_processing(
    outputs: dict, gpu_batch, visualize_environment: bool = False
) -> dict:
    """Mask PTIR/PBR image outputs with the batch GT foreground mask."""
    if pbr_gt_mask_was_applied(outputs):
        return outputs

    mask = getattr(gpu_batch, "mask", None)
    if mask is None:
        return outputs

    masked_outputs = dict(outputs)
    applied = False
    for key in PBR_GT_MASK_OUTPUT_KEYS:
        if visualize_environment and key in ("pred_pbr", "pred_direct"):
            continue
        value = outputs.get(key)
        if value is None:
            continue
        masked_outputs[key] = apply_gt_mask_to_tensor(value, mask)
        applied = True

    if applied:
        masked_outputs[PBR_GT_MASK_APPLIED_KEY] = True
    return masked_outputs


def pbr_gt_mask_was_applied(outputs: dict) -> bool:
    return bool(outputs.get(PBR_GT_MASK_APPLIED_KEY, False))


def _first_tensor_options(
    *sequences: Sequence[Any],
) -> tuple[torch.device, torch.dtype]:
    for sequence in sequences:
        for value in sequence:
            if isinstance(value, torch.Tensor):
                dtype = value.dtype if value.is_floating_point() else torch.float32
                return value.device, dtype
    return torch.device("cpu"), torch.float32


def _flatten_albedo(
    albedo: Any, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if isinstance(albedo, torch.Tensor):
        tensor = albedo.detach()
    else:
        tensor = torch.as_tensor(np.asarray(albedo))

    if tensor.ndim < 2 or tensor.shape[-1] != 3:
        raise ValueError(f"Albedo must have shape (..., 3), got {tuple(tensor.shape)}")

    return tensor.to(device=device, dtype=dtype).reshape(-1, 3)


def _use_single_channel_albedo_rescale(selection_context: Any | None) -> bool:
    if selection_context is None:
        return False
    context_text = str(selection_context).lower()
    return "air_baloons" in context_text or "airbaloons" in context_text


@torch.no_grad()
def compute_albedo_rescale_ratio(
    gt_albedo_list: Sequence[Any],
    albedo_list: Sequence[Any],
    eps: float = 1.0e-6,
    selection_context: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute robust PTIR albedo rescale ratios from predicted and GT albedos.

    Only pixels with strictly positive GT albedo in all channels are used. The
    returned tensors are `(single_channel_ratio, three_channel_ratio, selected_ratio)`.
    The single-channel ratio follows the IRGS-style convention of taking the
    median ratio from channel 0, while the RGB ratio keeps an independent median
    per channel. `selected_ratio` uses the single-channel ratio for air_baloons
    scenes and the RGB ratio otherwise.
    """

    if len(gt_albedo_list) != len(albedo_list):
        raise ValueError(
            f"Expected the same number of GT and predicted albedos, got {len(gt_albedo_list)} and {len(albedo_list)}"
        )
    if len(gt_albedo_list) == 0:
        raise ValueError(
            "At least one albedo pair is required to compute a rescale ratio."
        )

    device, dtype = _first_tensor_options(gt_albedo_list, albedo_list)
    gt_albedo_flat_list = []
    albedo_flat_list = []

    for index, (gt_albedo, albedo) in enumerate(zip(gt_albedo_list, albedo_list)):
        gt_albedo_flat = _flatten_albedo(gt_albedo, device=device, dtype=dtype)
        albedo_flat = _flatten_albedo(albedo, device=device, dtype=dtype)
        if gt_albedo_flat.shape != albedo_flat.shape:
            raise ValueError(
                f"Albedo pair {index} has mismatched flattened shapes: "
                f"{tuple(gt_albedo_flat.shape)} and {tuple(albedo_flat.shape)}"
            )

        valid = (
            torch.isfinite(gt_albedo_flat).all(dim=1)
            & torch.isfinite(albedo_flat).all(dim=1)
            & (gt_albedo_flat > 0.0).all(dim=1)
        )
        if not valid.any():
            continue

        gt_albedo_flat_list.append(gt_albedo_flat[valid])
        albedo_flat_list.append(albedo_flat[valid])

    if not gt_albedo_flat_list:
        raise ValueError(
            "No valid positive GT albedo pixels were found for rescale ratio computation."
        )

    gt_all = torch.cat(gt_albedo_flat_list, dim=0)
    albedo_all = torch.cat(albedo_flat_list, dim=0)
    ratios = gt_all / albedo_all.clamp(min=eps)

    single_channel_ratio = ratios[..., 0].median()
    three_channel_ratio = ratios.median(dim=0).values
    selected_ratio = three_channel_ratio
    # selected_ratio = single_channel_ratio if _use_single_channel_albedo_rescale(selection_context) else three_channel_ratio
    return single_channel_ratio, three_channel_ratio, selected_ratio


def rescale_albedo(
    albedo: torch.Tensor,
    ratio: torch.Tensor | float,
    clamp: bool = True,
) -> torch.Tensor:
    """Apply a scalar or RGB albedo rescale ratio."""

    ratio_tensor = torch.as_tensor(ratio, device=albedo.device, dtype=albedo.dtype)
    scaled_albedo = albedo * ratio_tensor
    if clamp:
        scaled_albedo = scaled_albedo.clip(0.0, 1.0)
    return scaled_albedo


def scaled_material_albedo_preactivation(
    model: "MixtureOfGaussians",
    ratio: torch.Tensor | float,
    eps: float = 1.0e-6,
) -> torch.nn.Parameter:
    albedo = model.get_material_albedo().detach()
    scaled_albedo = rescale_albedo(albedo, ratio).clamp(eps, 1.0 - eps)
    scaled_preactivation = model.material_albedo_activation_inv(scaled_albedo)
    return torch.nn.Parameter(
        scaled_preactivation.detach().clone(),
        requires_grad=model.material_albedo.requires_grad,
    )


def save_scaled_albedo_checkpoint(
    model: "MixtureOfGaussians",
    out_dir: str | Path,
    ratio: torch.Tensor | float,
    source_checkpoint_path: str | Path | None = None,
    output_name: str = "ckpt_last_scaled.pt",
) -> Path:
    out_dir = Path(out_dir)
    output_path = out_dir / output_name
    source_path = out_dir / "ckpt_last.pt"
    if not source_path.is_file() and source_checkpoint_path is not None:
        source_path = Path(source_checkpoint_path)

    if source_path.is_file():
        checkpoint = torch.load(
            source_path, map_location=getattr(model, "device", None), weights_only=False
        )
    else:
        checkpoint = model.get_model_parameters()
        checkpoint |= {"global_step": None, "epoch": None}

    checkpoint["material_albedo"] = scaled_material_albedo_preactivation(model, ratio)
    checkpoint["albedo_rescale_rgb"] = [
        float(value) for value in torch.as_tensor(ratio).detach().cpu().reshape(-1)
    ]
    torch.save(checkpoint, output_path)
    return output_path


def _criterion_device(criterions: Mapping[str, Any]) -> torch.device | None:
    for criterion in criterions.values():
        if not isinstance(criterion, torch.nn.Module):
            continue
        for value in criterion.parameters():
            return value.device
        for value in criterion.buffers():
            return value.device
    return None


def _to_nhwc_batch(image: torch.Tensor, channels: int | None = None) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        image = torch.as_tensor(np.asarray(image))
    image = image.detach()

    if image.ndim == 3:
        if (
            channels is not None
            and image.shape[0] == channels
            and image.shape[-1] != channels
        ):
            image = image.permute(1, 2, 0)
        image = image.unsqueeze(0)
    elif image.ndim == 4:
        if (
            channels is not None
            and image.shape[1] == channels
            and image.shape[-1] != channels
        ):
            image = image.permute(0, 2, 3, 1)
    else:
        raise ValueError(
            f"Expected image with shape [H, W, C] or [B, H, W, C], got {tuple(image.shape)}"
        )

    if channels is not None and image.shape[-1] != channels:
        raise ValueError(
            f"Expected image with {channels} channels, got shape {tuple(image.shape)}"
        )

    return image.float()


def _to_metric_device(
    pred: torch.Tensor,
    gt: torch.Tensor,
    criterions: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = _criterion_device(criterions) if criterions is not None else None
    if device is None:
        device = pred.device
    return pred.to(device=device, dtype=torch.float32), gt.to(
        device=device, dtype=torch.float32
    )


@torch.no_grad()
def compute_ptir_image_quality_metrics(
    criterions: Mapping[str, Any],
    pred: torch.Tensor,
    gt: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
    """Compute full-image PSNR, SSIM, and LPIPS for NHWC images."""

    pred = _to_nhwc_batch(pred, channels=3).clip(0.0, 1.0)
    gt = _to_nhwc_batch(gt, channels=3).clip(0.0, 1.0)
    if pred.shape != gt.shape:
        raise ValueError(
            f"Metric image shapes must match, got {tuple(pred.shape)} and {tuple(gt.shape)}"
        )

    pred, gt = _to_metric_device(pred, gt, criterions)
    pred_nchw = pred.permute(0, 3, 1, 2)
    gt_nchw = gt.permute(0, 3, 1, 2)

    metrics = {
        f"psnr_{prefix}": float(criterions["psnr"](pred, gt).item()),
        f"ssim_{prefix}": float(criterions["ssim"](pred_nchw, gt_nchw).item()),
        f"lpips_{prefix}": float(criterions["lpips"](pred_nchw, gt_nchw).item()),
    }
    return metrics


def _to_roughness(roughness: torch.Tensor) -> torch.Tensor:
    roughness = _to_nhwc_batch(roughness)
    if roughness.shape[-1] == 3:
        roughness = roughness[..., :1]
    if roughness.shape[-1] != 1:
        raise ValueError(
            f"Expected roughness with 1 or 3 channels, got shape {tuple(roughness.shape)}"
        )
    return roughness


@torch.no_grad()
def compute_ptir_full_image_metrics(
    criterions: Mapping[str, Any],
    pred_pbr: torch.Tensor | None = None,
    rgb_gt: torch.Tensor | None = None,
    albedo_scaled: torch.Tensor | None = None,
    albedo_gt: torch.Tensor | None = None,
    roughness: torch.Tensor | None = None,
    roughness_gt: torch.Tensor | None = None,
    normal: torch.Tensor | None = None,
    normal_gt: torch.Tensor | None = None,
    normal_valid_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute PTIR metrics over full images.

    `pred_pbr` should be in the same color space as `rgb_gt`; in the render
    path this means the linear PTIR PBR output is converted to sRGB first. A
    metric is only computed when both the prediction and GT tensor are present.
    Normal MAE follows the 3dgrt render path: invalid normal pixels are zeroed
    by the mask while the denominator remains the full image.
    """

    metrics: dict[str, float] = {}

    if pred_pbr is not None and rgb_gt is not None:
        metrics.update(
            compute_ptir_image_quality_metrics(criterions, pred_pbr, rgb_gt, "pbr")
        )

    if albedo_scaled is not None and albedo_gt is not None:
        metrics.update(
            compute_ptir_image_quality_metrics(
                criterions, albedo_scaled, albedo_gt, "albedo_scaled"
            )
        )

    if roughness is not None and roughness_gt is not None:
        roughness = _to_roughness(roughness).clip(0.0, 1.0)
        roughness_gt = _to_roughness(roughness_gt).clip(0.0, 1.0)
        if roughness.shape != roughness_gt.shape:
            raise ValueError(
                f"Roughness metric shapes must match, got {tuple(roughness.shape)} and {tuple(roughness_gt.shape)}"
            )
        roughness, roughness_gt = _to_metric_device(roughness, roughness_gt, criterions)
        metrics["mse_roughness"] = float(
            torch.mean((roughness - roughness_gt) ** 2).item()
        )

    if normal is not None and normal_gt is not None:
        normal = _to_nhwc_batch(normal, channels=3)
        normal_gt = _to_nhwc_batch(normal_gt, channels=3)
        if normal.shape != normal_gt.shape:
            raise ValueError(
                f"Normal metric shapes must match, got {tuple(normal.shape)} and {tuple(normal_gt.shape)}"
            )
        normal, normal_gt = _to_metric_device(normal, normal_gt, criterions)
        if normal_valid_mask is not None:
            normal_valid_mask = normal_valid_mask.detach().to(device=normal.device)
        metrics["mae_normal"] = float(
            normal_mae(
                normal,
                normal_gt,
                valid_mask=normal_valid_mask,
                average_full_image=True,
            ).item()
        )

    return metrics


def append_ptir_metrics(
    metric_lists: MutableMapping[str, list[float]],
    metrics: Mapping[str, float],
) -> None:
    for key, value in metrics.items():
        metric_lists.setdefault(key, []).append(float(value))


def summarize_ptir_metrics(
    metric_lists: Mapping[str, Sequence[float]],
    include_values: bool = False,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, values in metric_lists.items():
        if not values:
            continue
        values = [float(value) for value in values]
        if include_values:
            summary[key] = values
        summary[f"{key}_mean"] = float(np.mean(values))
    return summary


def init_model_from_training_checkpoint(
    model: "MixtureOfGaussians",
    checkpoint_path: str | Path,
    setup_optimizer: bool = False,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load a training checkpoint and initialize a model from it.

    This is intended for PTIR-style initialization from stage1 training weights,
    not for full resume training semantics.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Training checkpoint not found: {checkpoint_path}")

    load_location = (
        map_location if map_location is not None else getattr(model, "device", None)
    )
    logger.info(f"🤸 Loading training checkpoint from {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location=load_location, weights_only=False
    )
    model.init_from_checkpoint(checkpoint, setup_optimizer=setup_optimizer)
    return checkpoint
