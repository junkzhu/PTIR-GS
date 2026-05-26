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

import json
import os

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from einops import rearrange
from kornia import create_meshgrid
from PIL import Image
from torch.utils.data import Dataset

from threedgrut.utils.logger import logger

from .protocols import Batch, BatchPrior, BoundedMultiViewDataset, DatasetVisualization
from .utils import (
    create_camera_visualization,
    create_pixel_coords,
    get_center_and_diag,
    get_worker_id,
)


class NeRFDataset(Dataset, BoundedMultiViewDataset, DatasetVisualization):
    def __init__(
        self,
        path,
        device="cuda",
        split="train",
        ray_jitter=None,
        bg_color=None,
        load_normals=False,
        load_materials=False,
    ):
        self.root_dir = path
        self.device = device
        self.split = split
        self.ray_jitter = ray_jitter
        self.bg_color = bg_color
        self.load_normals = load_normals
        self.load_materials = load_materials

        # Cache for per-worker GPU tensors (thread-local storage)
        self._worker_gpu_cache = {}

        # (Re)load intrinsics and extrinsics
        self.reload()

    def reload(self):
        self.read_intrinsics()
        self.read_meta(self.split)
        self.center, self.length_scale, self.scene_bbox = self.compute_spatial_extents()

        # Store ray computation parameters on CPU for multiprocessing compatibility
        # Equivalent to _store_camera_params_cpu in ColmapDataset
        self._ray_cache_params = {
            "image_h": None,  # Will be set when needed
            "image_w": None,  # Will be set when needed
            "K": self.K.copy(),  # CPU numpy array
            "device": self.device,
            "ray_jitter": self.ray_jitter,
        }

        # Clear existing worker caches to force recreation with new intrinsics
        self._worker_gpu_cache.clear()

    def _resolve_transforms_path(self, split: str) -> str:
        if split == "val":
            candidates = ("transforms_val.json", "transforms_test.json")
        elif split == "trainval":
            candidates = ("transforms_train.json", "transforms_val.json", "transforms_test.json")
        else:
            candidates = (f"transforms_{split}.json",)

        for filename in candidates:
            path = os.path.join(self.root_dir, filename)
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            f"No transforms file found for split={split!r} in {self.root_dir}. Tried: {', '.join(candidates)}"
        )

    def _lazy_worker_ray_tensors_cache(self):
        """Create GPU-cached ray directions and pixel coordinates for current worker."""
        worker_id = get_worker_id()

        # Check if this worker already has cached tensors
        if worker_id not in self._worker_gpu_cache:
            # Create GPU tensors for this worker
            directions = NeRFDataset.__get_ray_directions(
                self.image_h,
                self.image_w,
                torch.tensor(self._ray_cache_params["K"], device=self.device),
                device=self.device,
                ray_jitter=self._ray_cache_params["ray_jitter"],
            )
            rays_o_cam = torch.zeros(
                (1, self.image_h, self.image_w, 3),
                dtype=torch.float32,
                device=self.device,
            )
            rays_d_cam = directions.reshape((1, self.image_h, self.image_w, 3)).contiguous()

            # Generate pixel coordinates with +0.5 center offset for post-processing
            pixel_coords = create_pixel_coords(self.image_w, self.image_h, device=self.device)

            # Cache for this worker
            self._worker_gpu_cache[worker_id] = (rays_o_cam, rays_d_cam, pixel_coords)

        return self._worker_gpu_cache[worker_id]

    def read_intrinsics(self):
        with open(self._resolve_transforms_path("train"), "r") as f:
            meta = json.load(f)

        # !! Assumptions !!
        # 1. All images have the same intrinsics
        # 2. Principal point is at canvas center
        # 3. Camera has no distortion params
        first_frame_path = meta["frames"][0]["file_path"]
        img_path = self._resolve_image_path(first_frame_path)

        frame = Image.open(img_path)

        w = frame.width
        h = frame.height
        self.img_wh = (w, h)

        fx = fy = 0.5 * w / np.tan(0.5 * meta["camera_angle_x"])

        self.K = np.float32([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]])
        self.intrinsics = [fx, fy, w / 2, h / 2]

    def read_meta(self, split):
        self.poses = []
        self.image_paths = []
        self.mask_paths = []
        self.gradient_mask_paths = []
        self.normal_paths = []
        self.material_albedo_paths = []
        self.material_roughness_paths = []

        if split == "trainval":
            with open(self._resolve_transforms_path("train"), "r") as f:
                frames = json.load(f)["frames"]
            with open(self._resolve_transforms_path("val"), "r") as f:
                frames += json.load(f)["frames"]
        else:
            with open(self._resolve_transforms_path(split), "r") as f:
                frames = json.load(f)["frames"]

        cam_centers = []
        for frame in logger.track(frames, description=f"Load Dataset ({split})", color="salmon1"):
            c2w = np.array(frame["transform_matrix"], dtype=np.float32)
            c2w[:, 1:3] *= -1  # [right up back] to [right down front]
            cam_centers.append(c2w[:3, 3])
            self.poses.append(c2w)

            img_path = self._resolve_image_path(frame["file_path"])
            self.image_paths.append(img_path)
            self.normal_paths.append(self._normal_path_from_image_path(img_path))
            material_paths = self._material_paths_from_image_path(img_path)
            self.material_albedo_paths.append(material_paths["albedo"])
            self.material_roughness_paths.append(material_paths["roughness"])

            # We assume that the mask is stored in the same folder as the image with the same name but with _mask.png extension.
            # If the mask does not exist, we will return None in the batch
            self.mask_paths.append(os.path.splitext(img_path)[0] + "_mask.png")
            self.gradient_mask_paths.append(os.path.splitext(img_path)[0] + "_gradient_mask.png")

        self.camera_centers = np.array(cam_centers)

        # https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/__init__.py#L69
        _, diagonal = get_center_and_diag(self.camera_centers)
        self.cameras_extent = diagonal * 1.1

        self.image_paths = np.stack(self.image_paths, dtype=str)
        self.mask_paths = np.stack(self.mask_paths, dtype=str)
        self.gradient_mask_paths = np.stack(self.gradient_mask_paths, dtype=str)
        self.normal_paths = np.stack(self.normal_paths, dtype=str)
        self.material_albedo_paths = np.stack(self.material_albedo_paths, dtype=str)
        self.material_roughness_paths = np.stack(self.material_roughness_paths, dtype=str)
        self.poses = np.array(self.poses).astype(np.float32)  # (N_images, 4, 4)

    @torch.no_grad()
    def compute_spatial_extents(self):
        camera_origins = torch.FloatTensor(self.poses[:, :3, 3])
        center = camera_origins.mean(dim=0)
        dists = torch.linalg.norm(camera_origins - center[None, :], dim=-1)
        mean_dist = torch.mean(dists)  # mean distance between of cameras from center
        bbox_min = torch.min(camera_origins, dim=0).values
        bbox_max = torch.max(camera_origins, dim=0).values
        return center, mean_dist, (bbox_min, bbox_max)

    def get_length_scale(self):
        return self.length_scale

    def get_center(self):
        return self.center

    def get_scene_bbox(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.scene_bbox

    def get_scene_extent(self):
        return self.cameras_extent

    def get_observer_points(self):
        return self.camera_centers

    def get_poses(self) -> np.ndarray:
        """Get camera poses as 4x4 transformation matrices.

        NeRF Dataset Implementation:
        Converts from NeRF's "right up back" coordinate system to 3DGRUT's
        "right down front" convention by negating Y and Z axes during loading.

        Original NeRF Convention: [right, up, back]
        3DGRUT Convention: [right, down, front]
        Conversion: c2w[:, 1:3] *= -1  # Negate Y and Z columns

        Returns:
            np.ndarray: Camera poses with shape (N, 4, 4) in "right down front" convention
        """
        return self.poses

    def get_camera_idx(self, frame_idx: int) -> int:
        """Return 0-based camera index for a given frame index.

        NeRF synthetic datasets use a single camera, so all frames
        are from camera 0.
        """
        return 0

    def get_frames_per_camera(self) -> list[int]:
        """Return list of frame counts per camera.

        NeRF synthetic datasets use a single camera, so all frames
        are attributed to camera 0. Derived values:
        - num_cameras = len(frames_per_camera) = 1
        - num_frames = sum(frames_per_camera) = len(self)
        """
        return [len(self)]

    def __len__(self):
        return len(self.poses)

    @torch.cuda.nvtx.range("nerf_dataset::_getitem")
    def __getitem__(self, idx) -> dict:
        out_shape = (1, self.image_h, self.image_w, 3)
        img, alpha = NeRFDataset.__read_image(
            self.image_paths[idx],
            self.img_wh,
            return_alpha=True,
            bg_color=self.bg_color,
        )

        output_dict = {
            "data": torch.tensor(img).reshape(out_shape),
            "pose": torch.tensor(self.poses[idx]).unsqueeze(0),
            "camera_idx": self.get_camera_idx(idx),
            "frame_idx": idx,
        }

        if self.load_normals:
            normal_path = self.normal_paths[idx]
            if not os.path.exists(normal_path):
                raise FileNotFoundError(f"Normal path {normal_path} does not exist.")
            normal = NeRFDataset.__read_image(
                normal_path,
                self.img_wh,
                return_alpha=False,
                bg_color=None,
            )
            output_dict["normal"] = torch.tensor(normal).reshape(out_shape)

        if self.load_materials:
            albedo_path = self.material_albedo_paths[idx]
            roughness_path = self.material_roughness_paths[idx]
            if os.path.exists(albedo_path) and os.path.exists(roughness_path):
                is_synthetic4relight = "synthetic4relight" in os.path.normpath(self.root_dir).lower()
                albedo = NeRFDataset.__read_linear_image(
                    albedo_path,
                    self.img_wh,
                    num_channels=3,
                    srgb_to_linear=is_synthetic4relight,
                    apply_alpha=is_synthetic4relight,
                )
                roughness = NeRFDataset.__read_linear_image(
                    roughness_path,
                    self.img_wh,
                    num_channels=1,
                    apply_alpha=is_synthetic4relight,
                )

                output_dict["material_albedo"] = torch.from_numpy(albedo).reshape(out_shape)
                output_dict["material_roughness"] = torch.from_numpy(roughness).reshape(1, self.image_h, self.image_w, 1)
            else:
                missing_paths = [path for path in (albedo_path, roughness_path) if not os.path.exists(path)]
                raise FileNotFoundError(f"Material path(s) do not exist: {missing_paths}")

        if hasattr(self, "prior_normal_paths"):
            prior_normal_path = self.prior_normal_paths[idx]
            if not os.path.exists(prior_normal_path):
                raise FileNotFoundError(f"Diffusion prior normal path {prior_normal_path} does not exist.")
            prior_normal = NeRFDataset.__read_image(
                prior_normal_path,
                self.img_wh,
                return_alpha=False,
                bg_color=None,
            )
            output_dict["prior_normal"] = torch.tensor(prior_normal).reshape(out_shape)

        if hasattr(self, "prior_albedo_paths"):
            prior_albedo_path = self.prior_albedo_paths[idx]
            if not os.path.exists(prior_albedo_path):
                raise FileNotFoundError(f"Diffusion prior albedo path {prior_albedo_path} does not exist.")
            prior_albedo = NeRFDataset.__read_linear_image(prior_albedo_path, self.img_wh, num_channels=3)
            output_dict["prior_albedo"] = torch.from_numpy(prior_albedo).reshape(out_shape)

        if hasattr(self, "prior_roughness_paths"):
            prior_roughness_path = self.prior_roughness_paths[idx]
            if not os.path.exists(prior_roughness_path):
                raise FileNotFoundError(f"Diffusion prior roughness path {prior_roughness_path} does not exist.")
            prior_roughness = NeRFDataset.__read_linear_image(prior_roughness_path, self.img_wh, num_channels=1)
            output_dict["prior_roughness"] = torch.from_numpy(prior_roughness).reshape(
                1,
                self.image_h,
                self.image_w,
                1,
            )

        mask_path = self.mask_paths[idx]
        if os.path.exists(mask_path):
            mask = torch.from_numpy(np.array(Image.open(mask_path).convert("L"))).reshape(1, self.image_h, self.image_w, 1)
            output_dict["mask"] = mask
        elif alpha is not None:
            mask = torch.from_numpy(alpha).reshape(1, self.image_h, self.image_w, 1)
            output_dict["mask"] = mask

        gradient_mask_path = self.gradient_mask_paths[idx]
        if os.path.exists(gradient_mask_path):
            gradient_mask = torch.from_numpy(np.array(Image.open(gradient_mask_path).convert("L"))).reshape(
                1, self.image_h, self.image_w, 1
            )
            output_dict["gradient_mask"] = gradient_mask

        return output_dict

    def get_gpu_batch_with_intrinsics(self, batch):
        """Add the intrinsics to the batch and move data to GPU."""

        data = batch["data"][0].to(self.device, non_blocking=True) / 255.0
        pose = batch["pose"][0].to(self.device, non_blocking=True)
        assert data.dtype == torch.float32
        assert pose.dtype == torch.float32

        # Get ray tensors and pixel coords for current worker (creates them if needed)
        rays_o_cam, rays_d_cam, pixel_coords = self._lazy_worker_ray_tensors_cache()

        sample = {
            "rgb_gt": data,
            "rays_ori": rays_o_cam,
            "rays_dir": rays_d_cam,
            "T_to_world": pose,
            "intrinsics": self.intrinsics,
            "camera_idx": batch["camera_idx"][0].item(),
            "frame_idx": batch["frame_idx"][0].item(),
            "pixel_coords": pixel_coords,
        }

        if "mask" in batch:
            mask = batch["mask"][0].to(self.device, non_blocking=True) / 255.0
            mask = (mask > 0.5).to(torch.float32)
            sample["mask"] = mask

        if "gradient_mask" in batch:
            gradient_mask = batch["gradient_mask"][0].to(self.device, non_blocking=True) / 255.0
            gradient_mask = (gradient_mask > 0.5).to(torch.float32)
            sample["gradient_mask"] = gradient_mask

        if "normal" in batch:
            normal = batch["normal"][0].to(self.device, non_blocking=True) / 255.0
            sample["normal_gt"] = normal * 2.0 - 1.0

        if "material_albedo" in batch:
            material_albedo = batch["material_albedo"][0].to(self.device, non_blocking=True)
            material_roughness = batch["material_roughness"][0].to(self.device, non_blocking=True)
            sample["material_albedo_gt"] = material_albedo
            sample["material_roughness_gt"] = material_roughness

        prior_kwargs = {}
        if "prior_normal" in batch:
            prior_normal = batch["prior_normal"][0].to(self.device, non_blocking=True) / 255.0
            prior_kwargs["normal"] = prior_normal * 2.0 - 1.0
        if "prior_albedo" in batch:
            prior_kwargs["albedo"] = batch["prior_albedo"][0].to(self.device, non_blocking=True)
        if "prior_roughness" in batch:
            prior_kwargs["roughness"] = batch["prior_roughness"][0].to(self.device, non_blocking=True)
        if prior_kwargs:
            sample["prior"] = BatchPrior(**prior_kwargs)

        return Batch(**sample)

    @staticmethod
    def _normal_path_from_image_path(image_path: str) -> str:
        normal_path = image_path.replace("rgba", "normal")
        return normal_path

    @staticmethod
    def _material_paths_from_image_path(image_path: str) -> dict[str, str]:
        image_dir = os.path.dirname(image_path)
        image_stem = os.path.splitext(os.path.basename(image_path))[0]
        if image_stem.endswith("_rgba"):
            image_stem = image_stem[: -len("_rgba")]

        candidate_pairs = (
            (
                os.path.join(image_dir, "albedo.png"),
                os.path.join(image_dir, "roughness.png"),
            ),
            (
                os.path.join(image_dir, f"{image_stem}_albedo.png"),
                os.path.join(image_dir, f"{image_stem}_rough.png"),
            ),
            (
                os.path.join(image_dir, f"{image_stem}_albedo.png"),
                os.path.join(image_dir, f"{image_stem}_roughness.png"),
            ),
        )

        for albedo_path, roughness_path in candidate_pairs:
            if os.path.exists(albedo_path) and os.path.exists(roughness_path):
                return {"albedo": albedo_path, "roughness": roughness_path}

        albedo_path, roughness_path = candidate_pairs[0]
        return {"albedo": albedo_path, "roughness": roughness_path}

    def _resolve_image_path(self, frame_path: str) -> str:
        image_path = os.path.join(self.root_dir, frame_path)
        candidates = (
            image_path,
            image_path + ".png",
            image_path + ".jpg",
            image_path + "_rgba.png",
            image_path + "_rgba.jpg",
        )

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        raise FileNotFoundError(f"Image path {image_path} does not exist.")

    @property
    def image_h(self):
        return self.img_wh[1]

    @property
    def image_w(self):
        return self.img_wh[0]

    def create_dataset_camera_visualization(self):
        # just one global intrinsic mat for now
        intrinsics = self.K

        cam_list = []
        for i_cam, pose in enumerate(self.poses):
            trans_mat = pose
            trans_mat_world_to_camera = np.linalg.inv(trans_mat)

            # these cameras follow the opposite convention from polyscope
            camera_convention_rot = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            trans_mat_world_to_camera = camera_convention_rot @ trans_mat_world_to_camera

            w = self.image_w
            h = self.image_h

            f_w = intrinsics[0, 0]
            f_h = intrinsics[1, 1]

            fov_w = 2.0 * np.arctan(0.5 * w / f_w)
            fov_h = 2.0 * np.arctan(0.5 * h / f_h)

            img = NeRFDataset.__read_image(
                self.image_paths[i_cam],
                self.img_wh,
                return_alpha=False,
                bg_color=self.bg_color,
            )
            rgb = img.reshape(h, w, 3) / np.float32(255.0)

            assert rgb.dtype == np.float32, "RGB image must be of type float32, but got {}".format(rgb.dtype)

            cam_list.append(
                {
                    "ext_mat": trans_mat_world_to_camera,
                    "w": w,
                    "h": h,
                    "fov_w": fov_w,
                    "fov_h": fov_h,
                    "rgb_img": rgb,
                    "split": self.split,
                }
            )

        create_camera_visualization(cam_list)

    @staticmethod
    @torch.cuda.amp.autocast(dtype=torch.float32)
    def __get_ray_directions(H, W, K, device="cpu", ray_jitter=None, return_uv=False, flatten=True):
        """
        Get ray directions for all pixels in camera coordinate [right down front].
        Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
                ray-tracing-generating-camera-rays/standard-coordinate-systems

        Inputs:
            H, W: image height and width
            K: (3, 3) camera intrinsics
            ray_jitter: Optional RayJitter component, for whether the ray passes randomly inside the pixel
            return_uv: whether to return uv image coordinates

        Outputs: (shape depends on @flatten)
            directions: (H, W, 3) or (H*W, 3), the direction of the rays in camera coordinate
            uv: (H, W, 2) or (H*W, 2) image coordinates
        """
        grid = create_meshgrid(H, W, False, device=device)[0]  # (H, W, 2)
        u, v = grid.unbind(-1)

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        if ray_jitter is None:  # pass by the center
            directions = torch.stack([(u - cx + 0.5) / fx, (v - cy + 0.5) / fy, torch.ones_like(u)], -1)
        else:
            jitter = ray_jitter(u.shape)
            directions = torch.stack(
                [
                    ((u + jitter[:, :, 0]) - cx) / fx,
                    ((v + jitter[:, :, 1]) - cy) / fy,
                    torch.ones_like(u),
                ],
                -1,
            )
        if flatten:
            directions = directions.reshape(-1, 3)
            grid = grid.reshape(-1, 2)

        if return_uv:
            return directions, grid

        return torch.nn.functional.normalize(directions, dim=-1)

    @staticmethod
    @torch.cuda.amp.autocast(dtype=torch.float32)
    def __get_rays(directions, c2w):
        """
        Get ray origin and directions in world coordinate for all pixels in one image.
        Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
                ray-tracing-generating-camera-rays/standard-coordinate-systems

        Inputs:
            directions: (N, 3) ray directions in camera coordinate
            c2w: (3, 4) or (N, 3, 4) transformation matrix from camera coordinate to world coordinate

        Outputs:
            rays_o: (N, 3), the origin of the rays in world coordinate
            rays_d: (N, 3), the direction of the rays in world coordinate
        """
        if c2w.ndim == 2:
            # Rotate ray directions from camera coordinate to the world coordinate
            rays_d = directions @ c2w[:, :3].T
        else:
            rays_d = rearrange(directions, "n c -> n 1 c") @ rearrange(c2w[..., :3], "n a b -> n b a")
            rays_d = rearrange(rays_d, "n 1 c -> n c")
        # The origin of all rays is the camera origin in world coordinate
        rays_o = c2w[..., 3].expand_as(rays_d)

        return rays_o, rays_d

    @staticmethod
    def __read_image(img_path, img_wh, return_alpha=False, bg_color=None):
        img = imageio.imread(img_path).astype(np.float32) / 255.0
        alpha = None
        # img[..., :3] = srgb_to_linear(img[..., :3])

        # Below assume image is float32
        if img.shape[2] == 4:  # blend A to RGB
            if return_alpha:
                alpha = img[:, :, -1]
            if bg_color is None:
                img = img[..., :3]
            elif bg_color == "black":
                img = img[..., :3] * img[..., -1:]
            elif bg_color == "white":
                img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
            else:
                assert False, f"{bg_color} is not a supported background color."

        img = cv2.resize(img, img_wh)
        img = rearrange(img, "h w c -> (h w) c")

        # Convert to uint8 again
        img = (img * 255.0).astype(np.uint8)
        assert img.dtype == np.uint8, "Image must be uint8"

        if return_alpha:
            if alpha is not None:
                alpha = cv2.resize(alpha, img_wh)
                alpha = rearrange(alpha, "h w -> (h w)")
                alpha = (alpha * 255.0).astype(np.uint8)
            return img, alpha
        else:
            return img

    @staticmethod
    def __read_linear_image(img_path, img_wh, num_channels, srgb_to_linear=False, apply_alpha=False):
        img = imageio.imread(img_path)
        if img.ndim == 2:
            img = img[..., None]
        if img.ndim != 3:
            raise ValueError(f"Expected image with shape [H, W, C] for {img_path}, got {img.shape}")

        if np.issubdtype(img.dtype, np.integer):
            img = img.astype(np.float32) / float(np.iinfo(img.dtype).max)
        else:
            img = img.astype(np.float32)

        alpha = None
        if img.shape[2] == 4:
            alpha = img[..., 3:4]
            img = img[..., :3]

        if srgb_to_linear:
            img = np.clip(img, 0.0, 1.0)
            img = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)

        img = cv2.resize(img, img_wh)
        if alpha is not None:
            alpha = cv2.resize(alpha, img_wh)
            if alpha.ndim == 2:
                alpha = alpha[..., None]
            if apply_alpha:
                img = img * alpha
        if img.ndim == 2:
            img = img[..., None]

        if num_channels == 1:
            img = img[..., :1]
        elif num_channels == 3:
            if img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            else:
                img = img[..., :3]
        else:
            raise ValueError(f"Unsupported material image channel count: {num_channels}")

        img = np.clip(img, 0.0, 1.0).astype(np.float32)
        return rearrange(img, "h w c -> (h w) c")
