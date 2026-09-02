# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch
from kornia.filters import spatial_gradient
from fused_ssim import fused_ssim


@torch.cuda.nvtx.range("l1_loss")
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


@torch.cuda.nvtx.range("l2_loss")
def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


@torch.cuda.nvtx.range("masked_l2_loss")
def masked_l2_loss(network_output, gt, mask=None, eps=1e-6):
    loss = (network_output - gt) ** 2

    if mask is None:
        return loss.mean()

    mask = _mask_to_bhw1(mask, loss)
    return (loss * mask).sum() / torch.clamp(mask.sum() * loss.shape[-1], min=eps)


@torch.cuda.nvtx.range("ssim")
def ssim(img1, img2, window_size=11, size_average=True):
    # predicted_image, gt_image: [BS, CH, H, W], predicted_image is differentiable
    return fused_ssim(img1, img2, padding="same")


@torch.cuda.nvtx.range("pseudo_normal_loss")
def pseudo_normal_loss(
    render_normal, pseudo_normal, valid_mask=None, eps=1e-6, detach_pseudo_normal=True
):
    """
    Args:
        render_normal: [B, H, W, 3] or [H, W, 3], world-space rendered normal.
        pseudo_normal: Same shape as render_normal, world-space pseudo normal.
        valid_mask: Optional bool/float mask shaped [B, H, W, 1], [B, H, W], [H, W, 1], or [H, W].
        eps: Normalization epsilon.
        detach_pseudo_normal: Whether to stop gradients through pseudo_normal.
    """
    n_render = render_normal
    n_pseudo = pseudo_normal.detach() if detach_pseudo_normal else pseudo_normal

    loss = 1.0 - (n_render * n_pseudo).sum(dim=-1)

    if valid_mask is not None:
        valid_mask = valid_mask.bool()
        if valid_mask.ndim == loss.ndim + 1:
            valid_mask = valid_mask.squeeze(-1)
        loss = loss[valid_mask]

    if loss.numel() == 0:
        return render_normal.sum() * 0.0

    return loss.mean()


@torch.cuda.nvtx.range("prior_normal_alignment_loss")
def prior_normal_alignment_loss(render_normal, prior_normal, valid_mask=None, eps=1e-6):
    """
    Alignment loss between rendered shading normals and diffusion normal priors.

    Args:
        render_normal: [B, H, W, 3] rendered shading normal.
        prior_normal: Same shape as render_normal, diffusion prior normal.
        valid_mask: Optional mask shaped [B, H, W, 1], [B, H, W], [H, W, 1], or [H, W].
        eps: Minimum denominator for masked reduction.
    """
    render_normal = render_normal
    prior_normal = prior_normal.detach()
    loss = 1.0 - (render_normal * prior_normal).sum(dim=-1)

    if valid_mask is not None:
        valid_mask = valid_mask.detach().to(
            device=render_normal.device, dtype=render_normal.dtype
        )
        if valid_mask.ndim == loss.ndim + 1:
            valid_mask = valid_mask.squeeze(-1)
        elif valid_mask.ndim == loss.ndim:
            pass
        else:
            raise ValueError(
                f"valid_mask must be compatible with {tuple(loss.shape)}, got {valid_mask.ndim}D"
            )

        if valid_mask.shape[0] == 1 and loss.shape[0] != 1:
            valid_mask = valid_mask.expand(loss.shape[0], -1, -1)
        if valid_mask.shape != loss.shape:
            raise ValueError(
                f"valid_mask shape {tuple(valid_mask.shape)} is not compatible with {tuple(loss.shape)}"
            )

        loss = loss * valid_mask
        return loss.sum() / torch.clamp(valid_mask.sum(), min=eps)

    return loss.mean()


@torch.cuda.nvtx.range("mask_entropy_loss")
def mask_entropy_loss(pred_opacity, mask, eps=1e-6):
    """
    Binary cross-entropy between rendered opacity and a foreground mask.

    Args:
        pred_opacity: Rendered opacity in [0, 1], shaped [B, H, W, 1] or [B, H, W].
        mask: Foreground mask, shaped like pred_opacity or with a trailing singleton channel.
        eps: Clamp epsilon for numerical stability.
    """
    if pred_opacity is None:
        raise ValueError("pred_opacity must be provided for mask_entropy_loss")
    if mask is None:
        return pred_opacity.sum() * 0.0

    image_mask = (
        mask.detach()
        .to(device=pred_opacity.device, dtype=pred_opacity.dtype)
        .clamp(0.0, 1.0)
    )
    rendered_opacity = pred_opacity.clamp(eps, 1.0 - eps)

    if image_mask.ndim == rendered_opacity.ndim - 1:
        image_mask = image_mask.unsqueeze(-1)
    elif rendered_opacity.ndim == image_mask.ndim - 1:
        rendered_opacity = rendered_opacity.unsqueeze(-1)

    if image_mask.shape != rendered_opacity.shape:
        image_mask = image_mask.expand_as(rendered_opacity)

    return -(
        image_mask * torch.log(rendered_opacity)
        + (1.0 - image_mask) * torch.log(1.0 - rendered_opacity)
    ).mean()


@torch.cuda.nvtx.range("depth_distortion_loss")
def depth_distortion_loss(pred_depth_distortion):
    """
    Mean depth distortion regularizer over rendered pixels.

    Args:
        pred_depth_distortion: Per-pixel distortion map, shaped [B, H, W, 1] or [B, H, W].
    """
    if pred_depth_distortion is None:
        raise ValueError(
            "pred_depth_distortion must be provided for depth_distortion_loss"
        )

    if pred_depth_distortion.numel() == 0:
        return pred_depth_distortion.sum() * 0.0

    return pred_depth_distortion.mean()


@torch.cuda.nvtx.range("light_consistency_loss")
def light_consistency_loss(light):
    if light is None:
        raise ValueError("light must be provided for light_consistency_loss")
    if light.ndim != 4:
        raise ValueError(f"light must be shaped [B, H, W, C], got {light.ndim}D")
    if light.numel() == 0:
        return light.sum() * 0.0

    mean_light = light.mean(dim=-1, keepdim=True).expand_as(light)
    return torch.nn.functional.l1_loss(light, mean_light)


def _mask_to_bhw1(mask, reference):
    mask = mask.detach().to(device=reference.device, dtype=reference.dtype)

    if mask.ndim == 3:
        mask = mask.unsqueeze(-1)
    elif mask.ndim == 4:
        pass
    else:
        raise ValueError(
            f"mask must be shaped [B, H, W] or [B, H, W, 1], got {mask.ndim}D"
        )

    if mask.shape[-1] != 1:
        mask = mask[..., :1]

    if mask.shape[0] == 1 and reference.shape[0] != 1:
        mask = mask.expand(reference.shape[0], -1, -1, -1)

    if mask.shape[:3] != reference.shape[:3]:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} is not compatible with {tuple(reference.shape)}"
        )

    return mask.clamp(0.0, 1.0)


def _edge_aware_spatial_gradients(value_map, gt_image):
    value_map_nchw = value_map.permute(0, 3, 1, 2)
    gt_image_nchw = gt_image.permute(0, 3, 1, 2)

    value_grad = spatial_gradient(value_map_nchw, order=1).abs()
    gt_grad = spatial_gradient(gt_image_nchw, order=1).abs()
    return value_grad, gt_grad


def _edge_aware_gradient_mask(mask, reference):
    mask = _mask_to_bhw1(mask, reference).permute(0, 3, 1, 2)

    left = torch.zeros_like(mask)
    right = torch.zeros_like(mask)
    up = torch.zeros_like(mask)
    down = torch.zeros_like(mask)

    left[:, :, :, 1:] = mask[:, :, :, :-1]
    right[:, :, :, :-1] = mask[:, :, :, 1:]
    up[:, :, 1:, :] = mask[:, :, :-1, :]
    down[:, :, :-1, :] = mask[:, :, 1:, :]

    mask_x = mask * left * right
    mask_y = mask * up * down
    return torch.stack((mask_x, mask_y), dim=2)


@torch.cuda.nvtx.range("edge_aware_smoothness_loss")
def edge_aware_smoothness_loss(value_map, gt_image, mask=None, eps=1e-3, scale=1.0):
    """
    Edge-aware smoothness loss for rendered maps.

    Args:
        value_map: Rendered map to smooth, shaped [B, H, W, C].
        gt_image: RGB guidance image in [0, 1], shaped [B, H, W, 3].
        mask: Optional valid-pixel mask. Smoothness is applied only where the
            finite-difference stencil stays inside the mask.
        eps: Minimum denominator for masked normalization.
        scale: Multiplier for the GT image gradient inside exp(-scale * gt_grad).
    """
    if value_map.ndim != 4:
        raise ValueError(
            f"value_map must be shaped [B, H, W, C], got {value_map.ndim}D"
        )
    if gt_image.ndim != 4:
        raise ValueError(f"gt_image must be shaped [B, H, W, C], got {gt_image.ndim}D")

    if value_map.shape[:3] != gt_image.shape[:3]:
        raise ValueError(
            f"value_map and gt_image spatial shapes must match, got {tuple(value_map.shape)} and {tuple(gt_image.shape)}"
        )

    value_grad, gt_grad = _edge_aware_spatial_gradients(value_map, gt_image)
    edge_weight = torch.exp(-scale * gt_grad.mean(dim=1, keepdim=True))
    weighted_grad = value_grad * edge_weight
    if mask is None:
        return weighted_grad.sum(dim=2).mean()

    gradient_mask = _edge_aware_gradient_mask(mask, value_map)
    masked_grad = weighted_grad * gradient_mask
    valid_pixels = (gradient_mask.sum(dim=2) > 0.0).to(dtype=value_map.dtype)
    denom = torch.clamp(valid_pixels.sum() * value_map.shape[-1], min=eps)
    return masked_grad.sum() / denom
