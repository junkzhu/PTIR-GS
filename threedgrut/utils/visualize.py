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


def _as_detached_float_tensor(
    value: torch.Tensor | np.ndarray,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    else:
        raise TypeError(f"Expected torch.Tensor or numpy.ndarray, got {type(value)!r}")

    if device is not None:
        tensor = tensor.to(device=device)
    return tensor.float()


def _channel_axis(tensor: torch.Tensor) -> Optional[int]:
    if tensor.ndim < 3:
        return None
    if tensor.shape[-1] in (1, 3, 4):
        return tensor.ndim - 1
    if tensor.shape[-3] in (1, 3, 4):
        return tensor.ndim - 3
    return None


def _broadcast_psnr_mask(mask: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
    mask = mask.detach().float()
    if mask.shape == error.shape:
        return mask

    if mask.ndim == error.ndim - 1:
        channel_axis = _channel_axis(error)
        if channel_axis is None:
            mask = mask.unsqueeze(-1)
        else:
            mask = mask.unsqueeze(channel_axis)

    if mask.shape == error.shape:
        return mask

    try:
        return torch.broadcast_to(mask, error.shape)
    except RuntimeError:
        pass

    if mask.ndim == error.ndim == 4:
        if mask.shape[-1] == 1 and error.shape[1] in (1, 3, 4):
            mask = mask.permute(0, 3, 1, 2)
        elif mask.shape[1] == 1 and error.shape[-1] in (1, 3, 4):
            mask = mask.permute(0, 2, 3, 1)
    elif mask.ndim == error.ndim == 3:
        if mask.shape[-1] == 1 and error.shape[0] in (1, 3, 4):
            mask = mask.permute(2, 0, 1)
        elif mask.shape[0] == 1 and error.shape[-1] in (1, 3, 4):
            mask = mask.permute(1, 2, 0)

    try:
        return torch.broadcast_to(mask, error.shape)
    except RuntimeError as exc:
        raise ValueError(f"Mask shape {tuple(mask.shape)} is not broadcastable to {tuple(error.shape)}") from exc


def _compute_psnr_tensor(
    pred: torch.Tensor | np.ndarray,
    gt: torch.Tensor | np.ndarray,
    mask: Optional[torch.Tensor | np.ndarray] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    with torch.no_grad():
        device = pred.device if isinstance(pred, torch.Tensor) else None
        if device is None and isinstance(gt, torch.Tensor):
            device = gt.device

        pred_tensor = _as_detached_float_tensor(pred, device=device).clamp(0.0, 1.0)
        gt_tensor = _as_detached_float_tensor(gt, device=pred_tensor.device).clamp(0.0, 1.0)
        if pred_tensor.shape != gt_tensor.shape:
            raise ValueError(f"PSNR expects matching shapes, got {tuple(pred_tensor.shape)} and {tuple(gt_tensor.shape)}")

        error = (pred_tensor - gt_tensor) ** 2
        if mask is not None:
            mask_tensor = _as_detached_float_tensor(mask, device=error.device)
            mask_tensor = (_broadcast_psnr_mask(mask_tensor, error) > 0).to(dtype=error.dtype)
            valid_count = mask_tensor.sum()
            masked_mse = (error * mask_tensor).sum() / valid_count.clamp_min(1.0)
            mse = torch.where(
                valid_count > 0,
                masked_mse,
                torch.full((), float("nan"), device=error.device, dtype=error.dtype),
            )
        else:
            mse = error.mean()

        return -10.0 * torch.log10(mse + eps)


def compute_psnr(
    pred: torch.Tensor | np.ndarray,
    gt: torch.Tensor | np.ndarray,
    mask: Optional[torch.Tensor | np.ndarray] = None,
    eps: float = 1e-8,
) -> float:
    psnr = _compute_psnr_tensor(pred, gt, mask=mask, eps=eps)
    return float(psnr.item())


class PBRPSNRTracker:
    def __init__(self, max_history: int = 10):
        self.max_history = int(max_history)
        self.pending_psnrs: list[torch.Tensor] = []
        self.history: list[float] = []

    def update(
        self,
        pred_pbr: Optional[torch.Tensor | np.ndarray],
        gt: Optional[torch.Tensor | np.ndarray],
        mask: Optional[torch.Tensor | np.ndarray] = None,
    ) -> Optional[torch.Tensor]:
        if pred_pbr is None or gt is None:
            return None

        psnr = _compute_psnr_tensor(pred_pbr, gt, mask=mask)
        self.pending_psnrs.append(psnr.detach())
        return psnr

    def finalize_visualization_step(self) -> Optional[float]:
        if not self.pending_psnrs:
            return None

        device = self.pending_psnrs[0].device
        pending = torch.stack([psnr.to(device=device) for psnr in self.pending_psnrs])
        finite = torch.isfinite(pending)
        if not finite.any():
            self.pending_psnrs.clear()
            return None

        mean_psnr = float(pending[finite].mean().item())
        self.history.append(mean_psnr)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]
        self.pending_psnrs.clear()
        return mean_psnr


def draw_psnr_sparkline_on_image(image: np.ndarray, psnr_history: list[float]) -> np.ndarray:
    if not psnr_history:
        return image

    image_np = np.asarray(image)
    if image_np.ndim != 3 or image_np.shape[-1] < 3:
        return image

    values = np.asarray(psnr_history[-10:], dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return image

    output = image_np.copy()
    dtype = output.dtype
    rgb = output[..., :3].astype(np.float32, copy=False)
    max_value = 255.0 if np.issubdtype(dtype, np.integer) or float(np.nanmax(rgb)) > 1.5 else 1.0
    canvas = np.clip(rgb, 0.0, max_value)
    if max_value <= 1.0:
        canvas = canvas * 255.0

    height, width = canvas.shape[:2]
    pad = max(4, int(round(min(height, width) * 0.015)))
    max_box_width = max(1, width - 2 * pad)
    max_box_height = max(1, height - 2 * pad)
    box_width = min(max_box_width, max(88, int(round(width * 0.34))))
    box_height = min(max_box_height, max(42, int(round(height * 0.20))))
    x0 = pad
    y0 = height - pad - box_height
    x1 = x0 + box_width
    y1 = y0 + box_height

    canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * 0.35

    text = f"PSNR {values[-1]:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.32, min(0.55, box_height / 90.0))
    thickness = max(1, int(round(min(height, width) / 420.0)))
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = x0 + max(6, int(round(box_width * 0.06)))
    text_y = y0 + max(text_size[1] + 5, int(round(box_height * 0.28)))
    cv2.putText(canvas, text, (text_x, text_y), font, font_scale, (232, 242, 255), thickness, cv2.LINE_AA)

    plot_x0 = x0 + max(7, int(round(box_width * 0.08)))
    plot_x1 = x1 - max(7, int(round(box_width * 0.06)))
    plot_y0 = y0 + max(text_y - y0 + baseline + 4, int(round(box_height * 0.42)))
    plot_y1 = y1 - max(6, int(round(box_height * 0.12)))
    if plot_x1 <= plot_x0 or plot_y1 <= plot_y0:
        return output

    vmin = float(values.min())
    vmax = float(values.max())
    if abs(vmax - vmin) < 1e-6:
        vmin -= 1.0
        vmax += 1.0

    if values.size == 1:
        x_coords = np.array([(plot_x0 + plot_x1) * 0.5], dtype=np.float32)
    else:
        x_coords = np.linspace(plot_x0, plot_x1, values.size, dtype=np.float32)
    norm = (values - vmin) / (vmax - vmin)
    y_coords = plot_y1 - norm * (plot_y1 - plot_y0)
    points = np.stack([x_coords, y_coords], axis=1).round().astype(np.int32)

    cv2.line(canvas, (plot_x0, plot_y1), (plot_x1, plot_y1), (105, 125, 145), 1, cv2.LINE_AA)
    if points.shape[0] > 1:
        cv2.polylines(canvas, [points.reshape(-1, 1, 2)], False, (176, 231, 255), thickness, cv2.LINE_AA)
    cv2.circle(canvas, tuple(points[-1]), max(2, thickness + 1), (255, 246, 175), -1, cv2.LINE_AA)

    if max_value <= 1.0:
        output[..., :3] = np.clip(canvas / 255.0, 0.0, 1.0)
    else:
        output[..., :3] = np.clip(canvas, 0.0, 255.0)

    if np.issubdtype(dtype, np.integer):
        return np.rint(output).astype(dtype)
    return output.astype(dtype, copy=False)


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
        self.pbr_psnr_tracker = PBRPSNRTracker()
        self._draw_pbr_psnr_history = False
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
        if self.enabled:
            self._update_pbr_psnr(outputs, batch)
        if not self.should_visualize(step):
            return

        self._draw_pbr_psnr_history = self.pbr_psnr_tracker.finalize_visualization_step() is not None
        rows = self._collect_rows(outputs, batch)
        if not rows:
            self._draw_pbr_psnr_history = False
            return

        image = self._concat_rows(rows)
        torchvision.utils.save_image(image, self.output_dir / f"{step:05d}.png")
        self._draw_pbr_psnr_history = False

    def _update_pbr_psnr(self, outputs: dict, batch: Optional[object]) -> None:
        if batch is None:
            return

        pred_pbr = outputs.get("pred_pbr")
        rgb_gt = getattr(batch, "rgb_gt", None)
        if pred_pbr is None or rgb_gt is None:
            return

        pred_pbr_srgb = self._linear_to_srgb(pred_pbr.detach())
        self.pbr_psnr_tracker.update(pred_pbr_srgb, rgb_gt)

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
            if pbr_image_is_linear:
                pbr_image = self._draw_psnr_history_on_batch(pbr_image)

        if material is None:
            albedo_image = torch.zeros_like(pbr_image)
            roughness_image = torch.zeros_like(pbr_image)
            metallic_image = torch.zeros_like(pbr_image)
        else:
            albedo_image = self._linear_to_srgb(material[..., 0:3].permute(0, 3, 1, 2))
            roughness_image = material[..., 3:4].permute(0, 3, 1, 2).repeat(1, 3, 1, 1).clip(0.0, 1.0)
            metallic_image = material[..., 4:5].permute(0, 3, 1, 2).repeat(1, 3, 1, 1).clip(0.0, 1.0)

        row_images = [
            pbr_image,
            albedo_image,
            roughness_image,
            metallic_image,
        ]
        return self._concat_images(row_images, dim=-1)

    def _draw_psnr_history_on_batch(self, image_batch: torch.Tensor) -> torch.Tensor:
        if not self._draw_pbr_psnr_history or not self.pbr_psnr_tracker.history:
            return image_batch

        images = []
        for image in image_batch:
            image_np = image.permute(1, 2, 0).detach().cpu().numpy()
            image_np = draw_psnr_sparkline_on_image(image_np, self.pbr_psnr_tracker.history)
            images.append(torch.from_numpy(np.ascontiguousarray(image_np)).permute(2, 0, 1))
        return torch.stack(images, dim=0).to(device=image_batch.device, dtype=image_batch.dtype)

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

        image = self._linear_to_srgb(image)
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
