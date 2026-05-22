# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from threedgrut.utils.logger import logger


@dataclass(frozen=True)
class VisualizationRowSpec:
    name: str
    ours_key: str
    gt_attr: str
    ours_transform: Callable[[torch.Tensor], torch.Tensor]
    gt_transform: Callable[[torch.Tensor], torch.Tensor]
    error: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    fixed_max_error: Optional[float] = None


class TrainingVisualizer:
    """Save periodic training visualizations to disk."""

    def __init__(
        self,
        output_dir: str | os.PathLike,
        frequency: int,
        has_normal_gt: bool = True,
        show_pbr_material: bool = False,
    ):
        self.frequency = int(frequency)
        self.enabled = self.frequency > 0
        self.output_dir = Path(output_dir) / "visualizations"
        self.has_normal_gt = bool(has_normal_gt)
        self.show_pbr_material = bool(show_pbr_material)
        self.row_specs = [
            VisualizationRowSpec(
                name="rgb",
                ours_key="pred_rgb",
                gt_attr="rgb_gt",
                ours_transform=lambda image: image.clip(0.0, 1.0),
                gt_transform=lambda image: image.clip(0.0, 1.0),
                error=lambda ours, gt: ((ours - gt) ** 2).mean(dim=-1),
            ),
            VisualizationRowSpec(
                name="normal",
                ours_key="pred_shadingnormal",
                gt_attr="normal_gt",
                ours_transform=lambda image: (0.5 * (image + 1.0)).clip(0.0, 1.0),
                gt_transform=lambda image: (0.5 * (image + 1.0)).clip(0.0, 1.0),
                error=self._normal_angle_error,
                fixed_max_error=180.0,
            ),
        ]

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📸 Training visualizations will be saved to: {self.output_dir}")

    def should_visualize(self, step: int) -> bool:
        return self.enabled and step > 0 and step % self.frequency == 0

    @torch.no_grad()
    def save(self, step: int, outputs: dict, batch: Optional[object] = None) -> None:
        if not self.should_visualize(step):
            return

        rows = self._collect_rows(outputs, batch)
        if not rows:
            return

        image = self._concat_rows(rows)
        torchvision.utils.save_image(image, self.output_dir / f"{step:05d}.png")

    def _collect_rows(self, outputs: dict, batch: Optional[object]) -> list[torch.Tensor]:
        rows = []
        for spec in self.row_specs:
            row = self._build_row(spec, outputs, batch)
            if row is None:
                continue
            rows.append(row)

        if self.show_pbr_material:
            pbr_row = self._build_pbr_material_row(outputs)
            if pbr_row is not None:
                rows.append(pbr_row)
            pbr_component_row = self._build_pbr_component_row(outputs)
            if pbr_component_row is not None:
                rows.append(pbr_component_row)

        return rows

    def _build_row(
        self,
        spec: VisualizationRowSpec,
        outputs: dict,
        batch: Optional[object],
    ) -> Optional[torch.Tensor]:
        if batch is None:
            return None

        ours = self._to_channel_last_batch(outputs.get(spec.ours_key))
        gt = self._to_channel_last_batch(getattr(batch, spec.gt_attr, None))
        if ours is None:
            return None

        ours_image = spec.ours_transform(self._to_image_batch(ours))
        gt_image, err_image = self._build_gt_and_error_images(spec, ours, ours_image, gt)
        if gt_image is None or err_image is None:
            return None

        row_images = [
            ours_image,
            gt_image,
            err_image,
        ]
        if spec.name == "rgb":
            depth_image = self._build_depth_image(outputs.get("pred_dist"), reference=ours_image)
            row_images.insert(1, depth_image)
        elif spec.name == "normal":
            pseudo_normal_image = self._build_pseudo_normal_image(batch, reference=ours_image)
            row_images.insert(1, pseudo_normal_image)

        return self._concat_images(row_images, dim=-1)

    def _build_gt_and_error_images(
        self,
        spec: VisualizationRowSpec,
        ours: torch.Tensor,
        ours_image: torch.Tensor,
        gt: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if spec.name == "normal" and not self.has_normal_gt:
            return torch.zeros_like(ours_image), torch.zeros_like(ours_image)

        if gt is None:
            return None, None

        gt_image = spec.gt_transform(self._to_image_batch(gt))
        gt_mask = self._gt_image_mask(gt_image) if spec.name == "normal" else None

        err_map = spec.error(ours, gt)
        err_image = self._error_batch_to_image(err_map, gt_mask, fixed_max_error=spec.fixed_max_error)
        return gt_image, err_image

    @staticmethod
    def _normal_angle_error(ours: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        ours = F.normalize(ours, dim=-1)
        gt = F.normalize(gt, dim=-1)
        cos_sim = (ours * gt).sum(dim=-1).clamp(-1.0, 1.0)
        return torch.arccos(cos_sim) * 180.0 / np.pi

    @staticmethod
    def _gt_image_mask(gt_image: torch.Tensor) -> torch.Tensor:
        valid = gt_image.abs().sum(dim=1, keepdim=True) > 1e-6
        return valid.permute(0, 2, 3, 1)

    def _error_batch_to_image(
        self,
        error_map: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        fixed_max_error: Optional[float] = None,
    ) -> torch.Tensor:
        error_map = error_map.detach()
        if error_map.ndim == 2:
            error_map = error_map.unsqueeze(0)

        mask = self._to_channel_last_batch(mask)
        heatmaps = []
        for batch_idx in range(error_map.shape[0]):
            mask_item = mask[batch_idx] if mask is not None else None
            heat_rgb, _ = _error_to_inferno(
                error_map[batch_idx],
                mask=mask_item,
                apply_mask_to_visual=True,
                fixed_max_error=fixed_max_error,
            )
            heatmaps.append(torch.from_numpy(heat_rgb).permute(2, 0, 1))

        return torch.stack(heatmaps, dim=0).float()

    def _build_depth_image(
        self,
        depth: Optional[torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        depth_batch = self._to_channel_last_batch(depth)
        if depth_batch is None:
            return torch.zeros_like(reference)

        depth_map = depth_batch[..., 0].detach().clamp(min=0.0)
        depth_images = []
        eps = np.finfo(np.float32).eps
        for batch_idx in range(depth_map.shape[0]):
            depth_item = depth_map[batch_idx].cpu().numpy().astype(np.float32)
            valid = np.isfinite(depth_item) & (depth_item > 0.0)
            if valid.any():
                near = depth_item[valid].min() - eps
                far = depth_item[valid].max() + eps
            else:
                near = 0.2 - eps
                far = 13.0 + eps

            curve_fn = lambda x: -np.log(x + eps)
            near, far, depth_item = [curve_fn(x) for x in [near, far, depth_item]]
            depth_item = np.nan_to_num(
                np.clip((depth_item - np.minimum(near, far)) / np.abs(far - near), 0.0, 1.0)
            )
            depth_item = (depth_item * 255.0).astype(np.uint8)
            depth_item = cv2.applyColorMap(depth_item, cv2.COLORMAP_TURBO)
            depth_item = cv2.cvtColor(depth_item, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            depth_images.append(torch.from_numpy(depth_item).permute(2, 0, 1))

        depth_image = torch.stack(depth_images, dim=0).float().cpu()
        if depth_image.shape[-2:] != reference.shape[-2:]:
            depth_image = F.interpolate(depth_image, size=reference.shape[-2:], mode="bilinear", align_corners=False)

        return depth_image

    def _build_pseudo_normal_image(
        self,
        batch: Optional[object],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if batch is None:
            return torch.zeros_like(reference)

        image = self._to_image_batch(getattr(batch, "pseudo_normal", None))
        if image is None:
            return torch.zeros_like(reference)

        image = (0.5 * (image + 1.0)).clip(0.0, 1.0)
        if image.shape[-2:] != reference.shape[-2:]:
            image = F.interpolate(image, size=reference.shape[-2:], mode="bilinear", align_corners=False)

        return image

    def _build_pbr_material_row(self, outputs: dict) -> Optional[torch.Tensor]:
        material = self._to_material_batch(outputs.get("pred_material"))
        pbr_image = self._to_image_batch(outputs.get("pred_pbr"))
        pbr_image_is_linear = pbr_image is not None
        if pbr_image is None:
            pbr_image = self._to_image_batch(outputs.get("pred_rgb"))
        if pbr_image is None:
            if material is None:
                return None
            pbr_image = torch.zeros(
                material.shape[0],
                3,
                material.shape[1],
                material.shape[2],
                dtype=material.dtype,
            )
        else:
            pbr_image = self._linear_to_srgb(pbr_image) if pbr_image_is_linear else pbr_image.clip(0.0, 1.0)

        if material is None:
            albedo_image = torch.zeros_like(pbr_image)
            roughness_image = torch.zeros_like(pbr_image)
            metallic_image = torch.zeros_like(pbr_image)
        else:
            albedo_image = material[..., 0:3].permute(0, 3, 1, 2).clip(0.0, 1.0)
            roughness_image = material[..., 3:4].permute(0, 3, 1, 2).repeat(1, 3, 1, 1).clip(0.0, 1.0)
            metallic_image = material[..., 4:5].permute(0, 3, 1, 2).repeat(1, 3, 1, 1).clip(0.0, 1.0)

        row_images = [
            pbr_image,
            albedo_image,
            roughness_image,
            metallic_image,
        ]
        return self._concat_images(row_images, dim=-1)

    def _build_pbr_component_row(self, outputs: dict) -> Optional[torch.Tensor]:
        direct_image = self._to_image_batch(outputs.get("pred_direct"))
        indirect_image = self._to_image_batch(outputs.get("pred_indirect"))
        if direct_image is None or indirect_image is None:
            return None

        direct_image = self._linear_to_srgb(direct_image)
        indirect_image = self._linear_to_srgb(indirect_image)
        environment_image = self._build_environment_image(
            outputs.get("environment"),
            reference=direct_image,
            cell_span=2,
        )

        row_images = [
            direct_image,
            indirect_image,
            environment_image,
        ]
        return self._concat_images_preserve_width(row_images)

    def _build_environment_image(
        self,
        environment: Optional[torch.Tensor],
        reference: torch.Tensor,
        cell_span: int = 1,
    ) -> torch.Tensor:
        target_height = reference.shape[-2]
        target_width = reference.shape[-1] * cell_span
        image = self._environment_to_image(environment)
        if image is None:
            return torch.zeros(reference.shape[0], 3, target_height, target_width, dtype=reference.dtype)

        image = image.clip(0.0, 1.0)
        image = F.interpolate(image, size=(target_height, target_width), mode="bilinear", align_corners=False)
        if image.shape[0] != reference.shape[0]:
            image = image[:1].expand(reference.shape[0], -1, -1, -1)
        return image

    @staticmethod
    def _linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
        image = image.clip(0.0, 1.0)
        return torch.where(
            image <= 0.0031308,
            12.92 * image,
            1.055 * image ** (1.0 / 2.4) - 0.055,
        )

    @staticmethod
    def _environment_to_image(environment: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if environment is None:
            return None

        image = environment.detach()
        if image.ndim == 4:
            image = image[0]
        if image.ndim != 3:
            return None

        if image.shape[-1] in (3, 4):
            image = image[..., :3].permute(2, 0, 1).unsqueeze(0)
        elif image.shape[0] in (3, 4):
            image = image[:3].unsqueeze(0)
        else:
            return None

        return image.float().cpu()

    @staticmethod
    def _concat_rows(rows: list[torch.Tensor]) -> torch.Tensor:
        width = rows[0].shape[-1]
        resized_rows = []

        for row in rows:
            if row.shape[-1] != width:
                row = F.interpolate(row, size=(row.shape[-2], width), mode="bilinear", align_corners=False)
            resized_rows.append(row)

        return torch.cat(resized_rows, dim=-2)

    @staticmethod
    def _concat_images(images: list[torch.Tensor], dim: int) -> torch.Tensor:
        height, width = images[0].shape[-2:]
        resized_images = []

        for image in images:
            if image.shape[-2:] != (height, width):
                image = F.interpolate(image, size=(height, width), mode="bilinear", align_corners=False)
            resized_images.append(image)

        return torch.cat(resized_images, dim=dim)

    @staticmethod
    def _concat_images_preserve_width(images: list[torch.Tensor]) -> torch.Tensor:
        height = images[0].shape[-2]
        resized_images = []

        for image in images:
            if image.shape[-2] != height:
                image = F.interpolate(image, size=(height, image.shape[-1]), mode="bilinear", align_corners=False)
            resized_images.append(image)

        return torch.cat(resized_images, dim=-1)

    @staticmethod
    def _to_image_batch(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None

        tensor = tensor.detach()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        if tensor.ndim != 4:
            return None

        if tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(0, 3, 1, 2)
        elif tensor.shape[1] not in (1, 3, 4):
            return None

        return tensor.float().cpu()

    @staticmethod
    def _to_channel_last_batch(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None

        tensor = tensor.detach()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        if tensor.ndim != 4:
            return None

        if tensor.shape[-1] in (1, 3, 4):
            return tensor.float()
        if tensor.shape[1] in (1, 3, 4):
            return tensor.permute(0, 2, 3, 1).float()

        return None

    @staticmethod
    def _to_material_batch(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None

        tensor = tensor.detach()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        if tensor.ndim != 4:
            return None

        if tensor.shape[-1] == 5:
            return tensor.float().cpu()
        if tensor.shape[1] == 5:
            return tensor.permute(0, 2, 3, 1).float().cpu()

        return None


def _error_to_inferno(
    error_map: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    apply_mask_to_visual: bool = False,
    fixed_max_error: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    error_map = error_map.detach().clamp(min=0.0)
    valid = None
    if mask is not None:
        if mask.ndim == 3:
            valid = mask[..., 0] > 0
        else:
            valid = mask > 0

    if fixed_max_error is not None:
        max_error = float(max(fixed_max_error, 0.0))
    elif valid is not None and valid.any():
        max_error = float(error_map[valid].max().item())
    else:
        max_error = float(error_map.max().item())

    denom = max(max_error, 1e-8)
    norm = (error_map / denom).clamp(0.0, 1.0)
    norm_np = norm.cpu().numpy().astype(np.float32)

    heat_rgb = _apply_inferno_colormap(norm_np)
    if apply_mask_to_visual and valid is not None:
        valid_np = valid.cpu().numpy()
        heat_rgb = np.where(valid_np[..., None], heat_rgb, 0)
    return heat_rgb, max_error


def _apply_inferno_colormap(norm_np: np.ndarray) -> np.ndarray:
    norm_u8 = (np.clip(norm_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(norm_u8, cv2.COLORMAP_INFERNO)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    return heat_rgb.astype(np.float32) / 255.0
