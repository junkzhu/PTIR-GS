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
import torch.nn.functional as F
from fused_ssim import fused_ssim


@torch.cuda.nvtx.range("l1_loss")
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


@torch.cuda.nvtx.range("l2_loss")
def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


@torch.cuda.nvtx.range("ssim")
def ssim(img1, img2, window_size=11, size_average=True):
    # predicted_image, gt_image: [BS, CH, H, W], predicted_image is differentiable
    return fused_ssim(img1, img2, padding="valid")


@torch.cuda.nvtx.range("pseudo_normal_loss")
def pseudo_normal_loss(render_normal, pseudo_normal, valid_mask=None, eps=1e-6):
    """
    Args:
        render_normal: [B, H, W, 3] or [H, W, 3], world-space rendered normal.
        pseudo_normal: Same shape as render_normal, world-space pseudo normal.
        valid_mask: Optional bool/float mask shaped [B, H, W, 1], [B, H, W], [H, W, 1], or [H, W].
        eps: Normalization epsilon.
    """
    n_render = render_normal
    n_pseudo = pseudo_normal.detach()

    loss = 1.0 - (n_render * n_pseudo).sum(dim=-1)

    if valid_mask is not None:
        valid_mask = valid_mask.bool()
        if valid_mask.ndim == loss.ndim + 1:
            valid_mask = valid_mask.squeeze(-1)
        loss = loss[valid_mask]

    if loss.numel() == 0:
        return render_normal.sum() * 0.0

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

    image_mask = mask.detach().to(device=pred_opacity.device, dtype=pred_opacity.dtype).clamp(0.0, 1.0)
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
