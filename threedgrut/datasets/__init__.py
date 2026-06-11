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

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .dataset_colmap import ColmapDataset
from .dataset_nerf import NeRFDataset
from .dataset_scannetpp import ScannetppDataset
from .datasetNcore import NCoreDataset
from .utils import read_colmap_extrinsics_binary, read_colmap_extrinsics_text


def _load_colmap_exif_exposures(
    dataset_path: str,
    downsample_factor: int,
) -> list[Optional[float]]:
    """Load EXIF exposure data for all COLMAP images.

    Reads COLMAP extrinsics to get all image paths, then loads EXIF exposure
    data and returns mean-normalized values. This is called once and shared
    between train and val datasets.

    Args:
        dataset_path: Path to COLMAP dataset root
        downsample_factor: Downsample factor for images folder suffix

    Returns:
        List of mean-normalized log2 exposure values for all images.
    """
    from threedgrut.utils.exif import load_exif_exposures

    # Read COLMAP extrinsics to get image names
    try:
        cameras_extrinsic_file = os.path.join(dataset_path, "sparse/0", "images.bin")
        cam_extrinsics = read_colmap_extrinsics_binary(cameras_extrinsic_file)
    except Exception:
        cameras_extrinsic_file = os.path.join(dataset_path, "sparse/0", "images.txt")
        cam_extrinsics = read_colmap_extrinsics_text(cameras_extrinsic_file)

    # Build image paths
    downsample_suffix = "" if downsample_factor == 1 else f"_{downsample_factor}"
    images_folder = f"images{downsample_suffix}"

    image_paths: list[Path] = []
    for extr in cam_extrinsics:
        image_path = Path(dataset_path) / images_folder / extr.name
        image_paths.append(image_path)

    return load_exif_exposures(image_paths)


def _maybe_generate_diffusion_priors(config, train_dataset) -> None:
    if not hasattr(train_dataset, "image_paths"):
        return

    needed_aovs = []
    if config.loss.get("use_normal_prior_regularization", False):
        needed_aovs.append("normal")
    if config.loss.get("use_albedo_prior_regularization", False):
        needed_aovs.append("albedo")
    if config.loss.get("use_roughness_prior_regularization", False):
        needed_aovs.append("roughness")
    if not needed_aovs:
        return

    from threedgrut.utils.rgb2x_prior import (
        DEFAULT_RGB2X_CACHE_DIR,
        DEFAULT_RGB2X_MODEL,
        DEFAULT_RGB2X_OUTPUT_DIR,
        rgb2x_prior_paths,
    )

    prior_config = config.get("diffusion_prior", {})
    output_root = prior_config.get("output_dir", DEFAULT_RGB2X_OUTPUT_DIR)
    configured_aovs = tuple(prior_config.get("aovs", needed_aovs))
    aovs = tuple(
        dict.fromkeys(
            [aov for aov in configured_aovs if aov in needed_aovs] + needed_aovs
        )
    )
    if not aovs:
        return

    if torch.cuda.is_available():
        from threedgrut.utils.rgb2x_prior import generate_rgb2x_priors_in_subprocess

        generate_rgb2x_priors_in_subprocess(
            train_dataset.image_paths,
            dataset_root=config.path,
            camera_to_worlds=getattr(train_dataset, "poses", None),
            output_root=output_root,
            model_name_or_path=prior_config.get(
                "model_name_or_path", DEFAULT_RGB2X_MODEL
            ),
            cache_dir=prior_config.get("cache_dir", DEFAULT_RGB2X_CACHE_DIR),
            aovs=aovs,
            input_size=prior_config.get("input_size", 512),
            inference_steps=prior_config.get("inference_steps", 50),
            seed=prior_config.get("seed", 42),
            skip_existing=prior_config.get("skip_existing", True),
            local_files_only=prior_config.get("local_files_only", False),
            batch_size=prior_config.get("batch_size", 1),
        )
    else:
        from threedgrut.utils.logger import logger

        logger.warning(
            "prior regularization is enabled but CUDA is unavailable; skipping rgb2x prior generation."
        )

    for aov in aovs:
        setattr(
            train_dataset,
            f"prior_{aov}_paths",
            np.array(
                [
                    str(rgb2x_prior_paths(path, config.path, output_root, (aov,))[aov])
                    for path in train_dataset.image_paths
                ],
                dtype=str,
            ),
        )


def _dataset_split_enabled(dataset_config, key: str, split: str) -> bool:
    return dataset_config.get(f"{key}_{split}", dataset_config.get(key, False))


def make(name: str, config, ray_jitter):
    match name:
        case "nerf":
            train_dataset = NeRFDataset(
                config.path,
                split="train",
                bg_color=config.model.background.color,
                ray_jitter=ray_jitter,
                load_normals=config.dataset.get("normal", False),
                load_materials=_dataset_split_enabled(
                    config.dataset, "material", "train"
                ),
                mask_from_background=config.dataset.get("mask_from_background", None),
            )
            val_dataset = NeRFDataset(
                config.path,
                split="val",
                bg_color=config.model.background.color,
                load_normals=config.dataset.get("normal", False),
                load_materials=_dataset_split_enabled(
                    config.dataset, "material", "val"
                ),
                mask_from_background=config.dataset.get("mask_from_background", None),
            )
        case "colmap":
            # Load EXIF exposure data if enabled (shared between train and val)
            if config.dataset.get("load_exif", True):
                exif_exposures = _load_colmap_exif_exposures(
                    config.path,
                    config.dataset.downsample_factor,
                )
            else:
                exif_exposures = None

            train_dataset = ColmapDataset(
                config.path,
                split="train",
                downsample_factor=config.dataset.downsample_factor,
                test_split_interval=config.dataset.test_split_interval,
                ray_jitter=ray_jitter,
                exif_exposures=exif_exposures,
                bg_color=config.model.background.color,
            )
            val_dataset = ColmapDataset(
                config.path,
                split="val",
                downsample_factor=config.dataset.downsample_factor,
                test_split_interval=config.dataset.test_split_interval,
                exif_exposures=exif_exposures,
                bg_color=config.model.background.color,
            )
        case "scannetpp":
            train_dataset = ScannetppDataset(
                config.path,
                split="train",
                ray_jitter=ray_jitter,
                downsample_factor=config.dataset.downsample_factor,
                test_split_interval=config.dataset.test_split_interval,
            )
            val_dataset = ScannetppDataset(
                config.path,
                split="val",
                downsample_factor=config.dataset.downsample_factor,
                test_split_interval=config.dataset.test_split_interval,
            )
        case "ncore":
            train_dataset = NCoreDataset(
                datapath=config.path,
                device="cuda",
                split="train",
                camera_ids=config.dataset.get(
                    "camera_ids", None
                ),  # Null = auto-select single camera sensor
                lidar_ids=config.dataset.get(
                    "lidar_ids", None
                ),  # Null = auto-select single lidar sensor
                downsample=config.dataset.get(
                    "downsample", 1.0
                ),  # Training downsample factor
                sample_full_image=config.dataset.train.get("sample_full_image", True),
                window_size=config.dataset.train.get("window_size", 256),
                n_samples_per_epoch=config.dataset.train.get(
                    "n_samples_per_epoch", 1000
                ),
                n_train_sample_timepoints=config.dataset.train.get(
                    "n_train_sample_timepoints", 1
                ),
                n_train_sample_camera_rays=config.dataset.train.get(
                    "n_train_sample_camera_rays", 4096
                ),
                n_val_image_subsample=config.dataset.get("n_val_image_subsample", 1),
                val_frame_interval=config.dataset.get(
                    "val_frame_interval", 8
                ),  # Frame-level split
                seek_offset_sec=config.dataset.train.get("seek_offset_sec", 0.0),
                duration_sec=config.dataset.train.get("duration_sec", None),
                poses_component_group=config.dataset.get(
                    "poses_component_group", "default"
                ),
                intrinsics_component_group=config.dataset.get(
                    "intrinsics_component_group", "default"
                ),
                masks_component_group=config.dataset.get(
                    "masks_component_group", "default"
                ),
                jpeg_backend_cpu=config.dataset.get("jpeg_backend_cpu", "simplejpeg"),
                simplejpeg_fastdct=config.dataset.get("simplejpeg_fastdct", False),
                simplejpeg_fastupsample=config.dataset.get(
                    "simplejpeg_fastupsample", False
                ),
                lidar_color_generic_data_name=config.dataset.get(
                    "lidar_color_generic_data_name", "rgb"
                ),
            )
            # Validation uses same temporal window as training by default
            train_seek_offset = config.dataset.train.get("seek_offset_sec", 0.0)
            train_duration = config.dataset.train.get("duration_sec", None)

            val_config = config.dataset.get("val", {})
            val_seek_offset_cfg = val_config.get("seek_offset_sec", None)
            val_duration_cfg = val_config.get("duration_sec", None)

            # Use training values if validation config is None, -1, or not set
            val_seek_offset = (
                train_seek_offset
                if (val_seek_offset_cfg is None or val_seek_offset_cfg < 0)
                else val_seek_offset_cfg
            )
            val_duration = (
                train_duration
                if (val_duration_cfg is None or val_duration_cfg < 0)
                else val_duration_cfg
            )

            val_dataset = NCoreDataset(
                datapath=config.path,
                device="cuda",
                split="val",
                camera_ids=config.dataset.get(
                    "camera_ids", None
                ),  # Null = auto-select single camera sensor
                lidar_ids=config.dataset.get(
                    "lidar_ids", None
                ),  # Null = auto-select single lidar sensor
                downsample=config.dataset.get("downsample", 1.0),
                sample_full_image=True,
                window_size=config.dataset.get("window_size", 256),
                n_val_image_subsample=config.dataset.get("n_val_image_subsample", 1),
                val_frame_interval=config.dataset.get(
                    "val_frame_interval", 8
                ),  # Frame-level split
                seek_offset_sec=val_seek_offset,
                duration_sec=val_duration,
                poses_component_group=config.dataset.get(
                    "poses_component_group", "default"
                ),
                intrinsics_component_group=config.dataset.get(
                    "intrinsics_component_group", "default"
                ),
                masks_component_group=config.dataset.get(
                    "masks_component_group", "default"
                ),
                jpeg_backend_cpu=config.dataset.get("jpeg_backend_cpu", "simplejpeg"),
                simplejpeg_fastdct=config.dataset.get("simplejpeg_fastdct", False),
                simplejpeg_fastupsample=config.dataset.get(
                    "simplejpeg_fastupsample", False
                ),
                lidar_color_generic_data_name=config.dataset.get(
                    "lidar_color_generic_data_name", "rgb"
                ),
            )
        case _:
            raise ValueError(
                f'Unsupported dataset type: {config.dataset.type}. Choose between: ["colmap", "nerf", "scannetpp", "ncore"].'
            )

    _maybe_generate_diffusion_priors(config, train_dataset)

    return train_dataset, val_dataset


def make_test(name: str, config):
    match name:
        case "nerf":
            dataset = NeRFDataset(
                config.path,
                split="test",
                bg_color=config.model.background.color,
                load_normals=config.dataset.get("normal", False),
                load_materials=_dataset_split_enabled(
                    config.dataset, "material", "test"
                ),
                mask_from_background=config.dataset.get("mask_from_background", None),
            )
        case "colmap":
            # Load EXIF exposure data if enabled
            if config.dataset.get("load_exif", True):
                exif_exposures = _load_colmap_exif_exposures(
                    config.path,
                    config.dataset.downsample_factor,
                )
            else:
                exif_exposures = None

            dataset = ColmapDataset(
                config.path,
                split="val",
                downsample_factor=config.dataset.downsample_factor,
                test_split_interval=config.dataset.test_split_interval,
                exif_exposures=exif_exposures,
            )
        case "scannetpp":
            dataset = ScannetppDataset(
                config.path,
                split="val",
                downsample_factor=config.dataset.downsample_factor,
                test_split_interval=config.dataset.test_split_interval,
            )
        case "ncore":
            # Inherit temporal window from training by default
            train_seek_offset = config.dataset.train.get("seek_offset_sec", 0.0)
            train_duration = config.dataset.train.get("duration_sec", None)

            val_config = config.dataset.get("val", {})
            val_seek_offset_cfg = val_config.get("seek_offset_sec", None)
            val_duration_cfg = val_config.get("duration_sec", None)

            # Use training values if validation config is None, -1, or not set
            test_seek_offset = (
                train_seek_offset
                if (val_seek_offset_cfg is None or val_seek_offset_cfg < 0)
                else val_seek_offset_cfg
            )
            test_duration = (
                train_duration
                if (val_duration_cfg is None or val_duration_cfg < 0)
                else val_duration_cfg
            )

            dataset = NCoreDataset(
                datapath=config.path,
                device="cuda",
                split="val",
                camera_ids=config.dataset.get(
                    "camera_ids", None
                ),  # Null = auto-select single camera sensor
                lidar_ids=config.dataset.get(
                    "lidar_ids", None
                ),  # Null = auto-select single lidar sensor
                downsample=config.dataset.get("downsample", 1.0),
                sample_full_image=True,
                window_size=config.dataset.get("window_size", 256),
                n_val_image_subsample=config.dataset.get("n_val_image_subsample", 1),
                val_frame_interval=config.dataset.get(
                    "val_frame_interval", 8
                ),  # Frame-level split
                seek_offset_sec=test_seek_offset,
                duration_sec=test_duration,
                poses_component_group=config.dataset.get(
                    "poses_component_group", "default"
                ),
                intrinsics_component_group=config.dataset.get(
                    "intrinsics_component_group", "default"
                ),
                masks_component_group=config.dataset.get(
                    "masks_component_group", "default"
                ),
                jpeg_backend_cpu=config.dataset.get("jpeg_backend_cpu", "simplejpeg"),
                simplejpeg_fastdct=config.dataset.get("simplejpeg_fastdct", False),
                simplejpeg_fastupsample=config.dataset.get(
                    "simplejpeg_fastupsample", False
                ),
                lidar_color_generic_data_name=config.dataset.get(
                    "lidar_color_generic_data_name", "rgb"
                ),
            )
        case _:
            raise ValueError(
                f'Unsupported dataset type: {config.dataset.type}. Choose between: ["colmap", "nerf", "scannetpp", "ncore"].'
            )
    return dataset
