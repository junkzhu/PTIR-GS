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

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import torch


@dataclass
class BatchPrior:
    normal: Optional[torch.Tensor] = None
    albedo: Optional[torch.Tensor] = None
    roughness: Optional[torch.Tensor] = None

    def validate(self, batch_size: int):
        if self.normal is not None:
            assert self.normal.ndim == 4, (
                "prior.normal must be a 4D tensor [B, H, W, 3]"
            )
            assert self.normal.shape[0] == batch_size, (
                "prior.normal must have the same batch size"
            )
        if self.albedo is not None:
            assert self.albedo.ndim == 4, (
                "prior.albedo must be a 4D tensor [B, H, W, 3]"
            )
            assert self.albedo.shape[0] == batch_size, (
                "prior.albedo must have the same batch size"
            )
        if self.roughness is not None:
            assert self.roughness.ndim == 4, (
                "prior.roughness must be a 4D tensor [B, H, W, 1]"
            )
            assert self.roughness.shape[0] == batch_size, (
                "prior.roughness must have the same batch size"
            )


@dataclass
class Batch:
    rays_ori: torch.Tensor  # [B, H, W, 3] ray origins in arbitrary space
    rays_dir: torch.Tensor  # [B, H, W, 3] ray directions in arbitrary space
    T_to_world: torch.Tensor  # [B, 4, 4] transformation matrix from the ray space to the world space (START pose)
    T_to_world_end: Optional[torch.Tensor] = (
        None  # [B, 4, 4] END pose for rolling shutter
    )
    rays_in_world_space: bool = (
        False  # True if rays are already in world space (no transform needed)
    )
    rgb_gt: Optional[torch.Tensor] = None
    normal_gt: Optional[torch.Tensor] = None
    material_gt: Optional[torch.Tensor] = None
    material_albedo_gt: Optional[torch.Tensor] = None
    material_roughness_gt: Optional[torch.Tensor] = None
    material_metallic_gt: Optional[torch.Tensor] = None
    pseudo_normal: Optional[torch.Tensor] = None
    pseudo_normal_mask: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    gradient_mask: Optional[torch.Tensor] = None
    intrinsics: Optional[list] = None
    intrinsics_OpenCVPinholeCameraModelParameters: Optional[dict] = None
    intrinsics_OpenCVFisheyeCameraModelParameters: Optional[dict] = None
    intrinsics_FThetaCameraModelParameters: Optional[dict] = None
    # Camera/frame indices for post-processing
    camera_idx: int = -1  # 0-based camera index
    frame_idx: int = -1  # 0-based frame index (global across split)
    # Pixel coordinates for post-processing
    pixel_coords: Optional[torch.Tensor] = (
        None  # [B, H, W, 2] (x, y) with +0.5 center offset
    )
    # Exposure prior from EXIF metadata (mean-normalized log2 exposure [1], None if unavailable)
    exposure: Optional[torch.Tensor] = None
    lights: Optional[torch.Tensor] = None  # packed scene lights [N, 9]
    light_alias_table: Optional[torch.Tensor] = None  # top-level light alias table [5, N]
    mesh_light_vertices: Optional[torch.Tensor] = None  # packed mesh-light vertices [V, 3]
    mesh_light_triangles: Optional[torch.Tensor] = None  # packed mesh-light triangles [T, 3]
    mesh_lights: Optional[torch.Tensor] = None  # packed mesh-light PBR params [M, 14]
    mesh_light_triangle_alias_table: Optional[torch.Tensor] = None  # triangle alias table [3, T]
    prior: Optional[BatchPrior] = None

    def __post_init__(self):
        batch_size = self.T_to_world.shape[0]
        assert self.rays_ori.shape[0] == batch_size, (
            "rays_ori must have the same batch size"
        )
        assert self.rays_dir.shape[0] == batch_size, (
            "rays_dir must have the same batch size"
        )
        if self.rgb_gt is not None:
            assert self.rgb_gt.ndim == 4, "rgb_gt must be a 4D tensor [B, H, W, 3]"
            assert self.rgb_gt.shape[0] == batch_size, (
                "rgb_gt must have the same batch size"
            )
        if self.normal_gt is not None:
            assert self.normal_gt.ndim == 4, (
                "normal_gt must be a 4D tensor [B, H, W, 3]"
            )
            assert self.normal_gt.shape[0] == batch_size, (
                "normal_gt must have the same batch size"
            )
        if self.material_gt is not None:
            assert self.material_gt.ndim == 4, (
                "material_gt must be a 4D tensor [B, H, W, 5]"
            )
            assert self.material_gt.shape[0] == batch_size, (
                "material_gt must have the same batch size"
            )
            assert self.material_gt.shape[-1] == 5, (
                "material_gt last dimension must be 5"
            )
        if self.material_albedo_gt is not None:
            assert self.material_albedo_gt.ndim == 4, (
                "material_albedo_gt must be a 4D tensor [B, H, W, 3]"
            )
            assert self.material_albedo_gt.shape[0] == batch_size, (
                "material_albedo_gt must have the same batch size"
            )
        if self.material_roughness_gt is not None:
            assert self.material_roughness_gt.ndim == 4, (
                "material_roughness_gt must be a 4D tensor [B, H, W, 1]"
            )
            assert self.material_roughness_gt.shape[0] == batch_size, (
                "material_roughness_gt must have the same batch size"
            )
        if self.material_metallic_gt is not None:
            assert self.material_metallic_gt.ndim == 4, (
                "material_metallic_gt must be a 4D tensor [B, H, W, 1]"
            )
            assert self.material_metallic_gt.shape[0] == batch_size, (
                "material_metallic_gt must have the same batch size"
            )
        if self.pseudo_normal is not None:
            assert self.pseudo_normal.ndim == 4, (
                "pseudo_normal must be a 4D tensor [B, H, W, 3]"
            )
            assert self.pseudo_normal.shape[0] == batch_size, (
                "pseudo_normal must have the same batch size"
            )
        if self.pseudo_normal_mask is not None:
            assert self.pseudo_normal_mask.ndim == 4, (
                "pseudo_normal_mask must be a 4D tensor [B, H, W, 1]"
            )
            assert self.pseudo_normal_mask.shape[0] == batch_size, (
                "pseudo_normal_mask must have the same batch size"
            )
        if self.mask is not None:
            assert self.mask.ndim == 4, "mask must be a 4D tensor [B, H, W, 1]"
            assert self.mask.shape[0] == batch_size, (
                "mask must have the same batch size"
            )
        if self.gradient_mask is not None:
            assert self.gradient_mask.ndim == 4, (
                "gradient_mask must be a 4D tensor [B, H, W, 1]"
            )
            assert self.gradient_mask.shape[0] == batch_size, (
                "gradient_mask must have the same batch size"
            )
        if self.intrinsics:
            assert isinstance(self.intrinsics, list), "intrinsics must be a list"
            assert len(self.intrinsics) == 4, (
                "intrinsics must have 4 elements [fx, fy, cx, cy]"
            )
        if self.pixel_coords is not None:
            assert self.pixel_coords.ndim == 4, (
                "pixel_coords must be a 4D tensor [B, H, W, 2]"
            )
            assert self.pixel_coords.shape[0] == batch_size, (
                "pixel_coords must have the same batch size"
            )
            assert self.pixel_coords.shape[3] == 2, (
                "pixel_coords last dimension must be 2 (x, y)"
            )
        if self.prior is not None:
            self.prior.validate(batch_size)


class BoundedMultiViewDataset(Protocol):
    """Defines the basic functionality required from all datasets that can be used with the 3dgrut Trainer."""

    def get_scene_bbox(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the bounding box of the scene as a tuple of vec3 (min,max)"""
        ...

    def get_scene_extent(self) -> float:
        """TODO"""
        ...

    def get_observer_points(self) -> np.ndarray:
        """TODO"""
        ...

    def get_poses(self) -> np.ndarray:
        """Get camera poses as 4x4 transformation matrices.

        Returns camera-to-world (C2W) transformation matrices using the 3DGRUT
        "right down front" camera coordinate system convention.

        Camera Coordinate System:
            - Right: +X axis points to the camera's right
            - Down: +Y axis points downward
            - Front: +Z axis points forward (viewing direction)

        Returns:
            np.ndarray: Camera poses with shape (N, 4, 4)
        """
        ...

    def get_gpu_batch_with_intrinsics(self, batch: dict) -> Batch:
        """Add the intrinsics to the batch and move data to GPU."""
        ...

    def get_camera_idx(self, frame_idx: int) -> int:
        """Return 0-based camera index for a given train split frame index."""
        ...

    def get_frames_per_camera(self) -> list[int]:
        """Return list of frame counts per camera.

        Returns a list where index i contains the number of frames captured
        by camera i. Derived values:
        - num_cameras = len(frames_per_camera)
        - num_frames = sum(frames_per_camera)
        """
        ...

    def __getitem__(self, index: int) -> dict: ...

    def __len__(self) -> int: ...


@runtime_checkable
class DatasetVisualization(Protocol):
    """Defines the basic functionality required from all datasets that can be visualized in the GUI app."""

    def create_dataset_camera_visualization(self): ...
