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

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from addict import Dict
from omegaconf import DictConfig, OmegaConf
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import threedgrut.datasets as datasets
from threedgrut.datasets.protocols import BoundedMultiViewDataset
from threedgrut.datasets.utils import DEFAULT_DEVICE, MultiEpochsDataLoader, PointCloud
from threedgrut.model.environment import Environment, save_environment_exr
from threedgrut.model.losses import (
    depth_distortion_loss,
    edge_aware_smoothness_loss,
    mask_entropy_loss,
    masked_l2_loss,
    prior_normal_alignment_loss,
    pseudo_normal_loss,
    ssim,
)
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.optimizers import SelectiveAdam
from threedgrut.render import Renderer
from threedgrut.strategy.base import BaseStrategy
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import check_step_condition, create_summary_writer, jet_map
from threedgrut.utils.normal import NormalUtils
from threedgrut.utils.render import apply_post_processing
from threedgrut.utils.timer import CudaTimer
from threedgrut.utils.visualize import TrainingVisualizer


class Trainer3DGRUT:
    """Trainer for paper: "3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes" """

    model: MixtureOfGaussians
    """ Gaussian Model """

    train_dataset: BoundedMultiViewDataset
    val_dataset: BoundedMultiViewDataset

    train_dataloader: torch.utils.data.DataLoader
    val_dataloader: torch.utils.data.DataLoader

    scene_extent: float = 1.0
    """TODO: Add docstring"""

    scene_bbox: tuple[torch.Tensor, torch.Tensor]  # Tuple of vec3 (min,max)
    """TODO: Add docstring"""

    strategy: BaseStrategy
    """ Strategy for optimizing the Gaussian model in terms of densification, pruning, etc. """

    gui = None
    """ If GUI is enabled, references the GUI interface """

    criterions: Dict
    """ Contains functors required to compute evaluation metrics, i.e. psnr, ssim, lpips """

    tracking: Dict
    """ Contains all components used to report progress of training """

    post_processing: Optional[nn.Module] = None
    """ Post-processing module """

    post_processing_optimizers: Optional[list] = None
    """ Optimizers for post-processing module """

    post_processing_schedulers: Optional[list] = None
    """ Schedulers for post-processing module optimizers """

    _distillation_start_step: int = -1
    """ Step at which distillation starts (-1 means disabled) """

    @staticmethod
    def create_from_checkpoint(resume: str, conf: DictConfig):
        """Create a new trainer from a checkpoint file"""

        conf.resume = resume
        conf.import_ply.enabled = False
        return Trainer3DGRUT(conf)

    @staticmethod
    def create_from_ply(ply_path: str, conf: DictConfig):
        """Create a new trainer from a PLY file"""

        conf.resume = ""
        conf.import_ply.enabled = True
        conf.import_ply.path = ply_path
        return Trainer3DGRUT(conf)

    @torch.cuda.nvtx.range("setup-trainer")
    def __init__(self, conf: DictConfig, device=None):
        """Set up a new training session, or continue an existing one based on configuration"""

        # Keep track of useful fields
        self.conf = conf
        """ Global configuration of model, scene, optimization, etc"""
        self.device = device if device is not None else DEFAULT_DEVICE
        """ Device used for training and visualizations """
        self.environment = None
        """ Optional environment map loaded from conf.environment. """
        self.global_step = 0
        """ Current global iteration of the trainer """
        self.n_iterations = conf.n_iterations
        """ Total number of train iterations to take (for multiple passes over the dataset) """
        self.n_epochs = 0
        """ Total number of train epochs / passes, e.g. single pass over the dataset."""
        self.val_frequency = conf.val_frequency
        """ Validation frequency, in terms on global steps """

        # Setup the trainer and components
        logger.log_rule("Load Datasets")
        self.init_dataloaders(conf)
        self.init_scene_extents(self.train_dataset)
        logger.log_rule("Initialize Model")
        self.init_model(conf, self.scene_extent)
        if OmegaConf.select(conf, "environment", default=None) is not None:
            self.init_environment(conf)
        self.init_densification_and_pruning_strategy(conf)
        logger.log_rule("Setup Model Weights & Training")
        self.init_metrics()
        self.setup_training(conf, self.model, self.train_dataset)
        self.init_experiments_tracking(conf)
        self.init_visualizer(conf)
        self.init_normal_utils()
        self.init_post_processing(conf)
        self.init_gui(conf, self.model, self.train_dataset, self.val_dataset, self.scene_bbox)

    def init_dataloaders(self, conf: DictConfig):
        from threedgrut.datasets.utils import configure_dataloader_for_platform

        train_dataset, val_dataset = datasets.make(name=conf.dataset.type, config=conf, ray_jitter=None)
        train_shuffle_generator = torch.Generator()
        train_shuffle_generator.manual_seed(int(conf.get("train_shuffle_seed", conf.seed_initialization)))
        train_dataloader_kwargs = configure_dataloader_for_platform(
            {
                "num_workers": conf.num_workers,
                "batch_size": 1,
                "shuffle": True,
                "generator": train_shuffle_generator,
                "pin_memory": True,
                "persistent_workers": True if conf.num_workers > 0 else False,
            }
        )

        val_dataloader_kwargs = configure_dataloader_for_platform(
            {
                "num_workers": conf.num_workers,
                "batch_size": 1,
                "shuffle": False,
                "pin_memory": True,
                "persistent_workers": True if conf.num_workers > 0 else False,
            }
        )

        train_dataloader = MultiEpochsDataLoader(train_dataset, **train_dataloader_kwargs)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, **val_dataloader_kwargs)

        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.val_dataset = val_dataset
        self.val_dataloader = val_dataloader

    def teardown_dataloaders(self):
        if self.train_dataloader is not None:
            del self.train_dataloader
        if self.val_dataloader is not None:
            del self.val_dataloader
        if self.train_dataset is not None:
            del self.train_dataset
        if self.val_dataset is not None:
            del self.val_dataset

    def init_scene_extents(self, train_dataset: BoundedMultiViewDataset) -> None:
        scene_bbox: tuple[torch.Tensor, torch.Tensor]  # Tuple of vec3 (min,max)
        scene_extent = train_dataset.get_scene_extent()
        scene_bbox = train_dataset.get_scene_bbox()
        self.scene_extent = scene_extent
        self.scene_bbox = scene_bbox

    def init_model(self, conf: DictConfig, scene_extent=None) -> None:
        """Initializes the gaussian model and the optix context"""
        self.model = MixtureOfGaussians(conf, scene_extent=scene_extent)

    def init_environment(self, conf: DictConfig) -> None:
        env_path = OmegaConf.select(conf, "environment.path", default=None)
        env_type = OmegaConf.select(conf, "environment.type", default="2d")
        optimize_environment = bool(OmegaConf.select(conf, "model.optimize_environment", default=False))
        self.environment = Environment(
            path=env_path,
            device=self.device,
            environment_type=env_type,
            optimize_environment=optimize_environment,
        )
        self.model.optimize_environment = self.environment.optimize_environment
        self.model.environment = self.environment.get_environment_parameter()

    def restore_environment_from_checkpoint(self, checkpoint: dict, conf: DictConfig) -> None:
        environment_state = checkpoint.get("environment_state")
        if self.environment is None or environment_state is None:
            return

        self.environment.load_state_dict(environment_state)
        self.environment.configure_optimization(bool(OmegaConf.select(conf, "model.optimize_environment", default=False)))
        self.model.optimize_environment = self.environment.optimize_environment
        self.model.environment = self.environment.get_environment_parameter()

    def init_densification_and_pruning_strategy(self, conf: DictConfig) -> None:
        """Set pre-train / post-train iteration logic. i.e. densification and pruning"""
        assert self.model is not None
        match self.conf.strategy.method:
            case "GSStrategy":
                from threedgrut.strategy.gs import GSStrategy

                self.strategy = GSStrategy(conf, self.model)
                logger.info("🔆 Using GS strategy")
            case "MCMCStrategy":
                from threedgrut.strategy.mcmc import MCMCStrategy

                self.strategy = MCMCStrategy(conf, self.model)
                logger.info("🔆 Using MCMC strategy")
            case _:
                raise ValueError(f"unrecognized model.strategy {conf.strategy.method}")

    def setup_training(
        self,
        conf: DictConfig,
        model: MixtureOfGaussians,
        train_dataset: BoundedMultiViewDataset,
    ):
        """
        Performs required steps to setup the optimization:
        1. Initialize the gaussian model fields: load previous weights from checkpoint, or initialize from scratch.
        2. Build BVH acceleration structure for gaussian model, if not loaded with checkpoint
        3. Set up the optimizer to optimize the gaussian model params
        4. Initialize the densification buffers in the densificaiton strategy
        """

        # Initialize
        if conf.resume:  # Load a checkpoint
            logger.info(f"🤸 Loading a pretrained checkpoint from {conf.resume}!")
            checkpoint = torch.load(conf.resume, weights_only=False)
            model.init_from_checkpoint(checkpoint, setup_optimizer=False)
            self.restore_environment_from_checkpoint(checkpoint, conf)
            model.setup_optimizer(state_dict=checkpoint["optimizer"])
            self.strategy.init_densification_buffer(checkpoint)
            global_step = checkpoint["global_step"]

            # Restore post-processing state
            if "post_processing" in checkpoint and self.post_processing is not None:
                self.post_processing.load_state_dict(checkpoint["post_processing"]["module"])
                for opt, opt_state in zip(
                    self.post_processing_optimizers,
                    checkpoint["post_processing"]["optimizers"],
                ):
                    opt.load_state_dict(opt_state)
                for sched, sched_state in zip(
                    self.post_processing_schedulers,
                    checkpoint["post_processing"]["schedulers"],
                ):
                    sched.load_state_dict(sched_state)
                logger.info("📷 Post-processing state restored from checkpoint")
        elif conf.import_ply.enabled:
            ply_path = (
                conf.import_ply.path
                if conf.import_ply.path
                else f"{conf.out_dir}/{conf.experiment_name}/export_last.ply"
            )
            logger.info(f"Loading a ply model from {ply_path}!")
            model.init_from_ply(ply_path)
            self.strategy.init_densification_buffer()
            model.build_acc()
            global_step = conf.import_ply.init_global_step
        else:
            logger.info(f"🤸 Initiating new 3dgrut training..")
            match conf.initialization.method:
                case "random":
                    model.init_from_random_point_cloud(
                        num_gaussians=conf.initialization.num_gaussians,
                        xyz_max=conf.initialization.xyz_max,
                        xyz_min=conf.initialization.xyz_min,
                    )
                case "colmap":
                    observer_points = torch.tensor(
                        train_dataset.get_observer_points(),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    model.init_from_colmap(conf.path, observer_points)
                case "fused_point_cloud":
                    observer_points = torch.tensor(
                        train_dataset.get_observer_points(),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    ply_path = conf.initialization.fused_point_cloud_path
                    logger.info(f"Initializing from accumulated point cloud: {ply_path}")
                    model.init_from_fused_point_cloud(ply_path, observer_points)
                case "point_cloud":
                    try:
                        ply_path = os.path.join(conf.path, "point_cloud.ply")
                        model.init_from_pretrained_point_cloud(ply_path)
                    except FileNotFoundError as e:
                        logger.error(e)
                        raise e
                case "checkpoint":
                    checkpoint = torch.load(conf.initialization.path, weights_only=False)
                    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
                    self.restore_environment_from_checkpoint(checkpoint, conf)
                case "lidar":
                    assert isinstance(
                        train_dataset, datasets.NCoreDataset
                    ), "can only initialize from lidar with NCoreDataset"
                    pc = PointCloud.from_sequence(
                        list(train_dataset.get_point_clouds(step_frame=1, non_dynamic_points_only=True)),
                        device="cpu",
                    )
                    if conf.initialization.num_points < len(pc.xyz_end):
                        # Deterministically random subsample points if there are more points than the specified number of gaussians
                        rng = torch.Generator().manual_seed(conf.seed_initialization)
                        idxs = torch.randperm(len(pc.xyz_end), generator=rng)[: conf.initialization.num_points]
                        pc = pc.selected_idxs(idxs)
                    observer_points = torch.tensor(
                        train_dataset.get_observer_points(),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    model.init_from_lidar(pc, observer_points)
                case _:
                    raise ValueError(
                        f"unrecognized initialization.method {conf.initialization.method}, choose from [colmap, point_cloud, random, checkpoint, lidar]"
                    )

            self.strategy.init_densification_buffer()

            model.build_acc()
            model.setup_optimizer()
            global_step = 0

        self.global_step = global_step
        self.n_epochs = int((conf.n_iterations + len(train_dataset) - 1) / len(train_dataset))

    def init_gui(
        self,
        conf: DictConfig,
        model: MixtureOfGaussians,
        train_dataset: BoundedMultiViewDataset,
        val_dataset: BoundedMultiViewDataset,
        scene_bbox,
    ):
        gui = None

        if conf.with_gui:
            from threedgrut.utils.gui import GUI

            gui = GUI(conf, model, train_dataset, val_dataset, scene_bbox)

        elif conf.with_viser_gui:
            from threedgrut.utils.viser_gui_util import ViserGUI

            gui = ViserGUI(conf, model, train_dataset, val_dataset, scene_bbox)

        self.gui = gui

    def init_metrics(self):
        self.criterions = Dict(
            psnr=PeakSignalNoiseRatio(data_range=1).to(self.device),
            ssim=StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device),
            lpips=LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(self.device),
        )

    def init_experiments_tracking(self, conf: DictConfig):
        # Initialize the tensorboard writer
        object_name = Path(conf.path).stem
        writer, out_dir, run_name = create_summary_writer(
            conf, object_name, conf.out_dir, conf.experiment_name, conf.use_wandb
        )
        logger.info(f"📊 Training logs & will be saved to: {out_dir}")

        # Store parsed config for reference
        with open(os.path.join(out_dir, "parsed.yaml"), "w") as fp:
            OmegaConf.save(config=conf, f=fp)

        # Pack all components used to track progress of training
        self.tracking = Dict(
            writer=writer,
            run_name=run_name,
            object_name=object_name,
            output_dir=out_dir,
        )

    def init_visualizer(self, conf: DictConfig):
        self.visualizer = TrainingVisualizer(
            output_dir=self.tracking.output_dir,
            frequency=getattr(conf, "visualize_frequency", 0),
            has_normal_gt=conf.dataset.get("normal", False),
            show_pbr_material=conf.render.method == "3dgptir",
        )

    def init_normal_utils(self):
        self.normal_utils = NormalUtils()

    def init_post_processing(self, conf: DictConfig):
        """Initialize post-processing module based on config."""
        method = conf.post_processing.method

        if method is None:
            return

        if method == "ppisp":
            from ppisp import PPISP, PPISPConfig

            frames_per_camera = self.train_dataset.get_frames_per_camera()
            num_cameras = len(frames_per_camera)
            num_frames = sum(frames_per_camera)

            use_controller = conf.post_processing.get("use_controller", True)

            # Distillation mode: controller activates after main training
            # Total iterations = n_iterations, distillation starts at n_iterations - n_distillation_steps
            n_distillation_steps = conf.post_processing.get("n_distillation_steps", 5000)
            if use_controller and n_distillation_steps > 0:
                main_training_steps = conf.n_iterations - n_distillation_steps
                controller_activation_ratio = main_training_steps / conf.n_iterations
                controller_distillation = True
                self._distillation_start_step = main_training_steps
                logger.info(f"📷 PPISP distillation mode: controller activates at step {main_training_steps}")
            elif use_controller:
                controller_activation_ratio = 0.8
                controller_distillation = False
                self._distillation_start_step = -1
            else:
                controller_activation_ratio = 0.0
                controller_distillation = False
                self._distillation_start_step = -1

            ppisp_config = PPISPConfig(
                use_controller=use_controller,
                controller_distillation=controller_distillation,
                controller_activation_ratio=controller_activation_ratio,
            )

            self.post_processing = PPISP(
                num_cameras=num_cameras,
                num_frames=num_frames,
                config=ppisp_config,
            ).to(self.device)

            self.post_processing_optimizers = self.post_processing.create_optimizers()
            self.post_processing_schedulers = self.post_processing.create_schedulers(
                self.post_processing_optimizers,
                max_optimization_iters=conf.n_iterations,
            )

            logger.info(f"📷 {method.upper()} initialized: {num_cameras} cameras, {num_frames} frames")
        else:
            raise ValueError(f"Unknown post-processing method: {method}")

    @torch.cuda.nvtx.range("get_metrics")
    def get_metrics(
        self,
        gpu_batch: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        losses: dict[str, torch.Tensor],
        profilers: dict[str, CudaTimer],
        split: str = "training",
        iteration: Optional[int] = None,
    ) -> dict[str, Union[int, float]]:
        """Computes dictionary of single batch metrics based on current batch output.
        Args:
            gpu_batch: GT data of current batch
            output: model prediction for current batch
            losses: dictionary of loss terms computed for current batch
            split: name of split metrics are computed for - 'training' or 'validation'
            iteration: optional, local iteration number within the current pass, e.g 0 <= iter < len(dataset).
        Returns:
            Dictionary of metrics
        """
        metrics = dict()
        step = self.global_step

        rgb_gt = gpu_batch.rgb_gt
        rgb_pred = outputs["pred_rgb"]

        psnr = self.criterions["psnr"]
        ssim = self.criterions["ssim"]
        lpips = self.criterions["lpips"]

        # Move losses to cpu once
        metrics["losses"] = {k: v.detach().item() for k, v in losses.items()}

        is_compute_train_hit_metrics = (split == "training") and (step % self.conf.writer.hit_stat_frequency == 0)
        is_compute_validation_metrics = split == "validation"

        if is_compute_train_hit_metrics or is_compute_validation_metrics:
            metrics["hits_mean"] = outputs["hits_count"].mean().item()
            metrics["hits_std"] = outputs["hits_count"].std().item()
            metrics["hits_min"] = outputs["hits_count"].min().item()
            metrics["hits_max"] = outputs["hits_count"].max().item()

        if is_compute_validation_metrics:
            with torch.cuda.nvtx.range(f"criterions_psnr"):
                metrics["psnr"] = psnr(rgb_pred, rgb_gt).item()

            rgb_gt_full = rgb_gt.permute(0, 3, 1, 2)
            pred_rgb_full = rgb_pred.permute(0, 3, 1, 2)
            pred_rgb_full_clipped = rgb_pred.clip(0, 1).permute(0, 3, 1, 2)

            with torch.cuda.nvtx.range(f"criterions_ssim"):
                metrics["ssim"] = ssim(pred_rgb_full, rgb_gt_full).item()
            with torch.cuda.nvtx.range(f"criterions_lpips"):
                metrics["lpips"] = lpips(pred_rgb_full_clipped, rgb_gt_full).item()

            if iteration in self.conf.writer.log_image_views:
                metrics["img_hit_counts"] = jet_map(outputs["hits_count"][-1], self.conf.writer.max_num_hits)
                metrics["img_gt"] = gpu_batch.rgb_gt[-1].clip(0, 1.0)
                metrics["img_pred"] = outputs["pred_rgb"][-1].clip(0, 1.0)
                metrics["img_pred_dist"] = jet_map(outputs["pred_dist"][-1], 100)
                metrics["img_pred_opacity"] = jet_map(outputs["pred_opacity"][-1], 1)

        if profilers:
            timings = {}
            for key, timer in profilers.items():
                if timer.enabled:
                    timings[key] = timer.timing()
            if timings:
                metrics["timings"] = timings

        return metrics

    @torch.cuda.nvtx.range("get_losses")
    def get_losses(
        self, gpu_batch: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Computes dictionary of losses for current batch.
        Args:
            gpu_batch: GT data of current batch
            outputs: model prediction for current batch
        Returns:
            losses: dictionary of loss terms computed for current batch.
        """
        rgb_gt = gpu_batch.rgb_gt
        rgb_pred = outputs["pred_rgb"]
        mask = gpu_batch.mask
        gradient_mask = gpu_batch.gradient_mask

        # Mask out the invalid pixels if a gradient mask is provided.
        if gradient_mask is not None:
            rgb_gt = rgb_gt * gradient_mask
            rgb_pred = rgb_pred * gradient_mask

        # L1 loss
        loss_l1 = torch.zeros(1, device=self.device)
        lambda_l1 = 0.0
        if self.conf.loss.use_l1:
            with torch.cuda.nvtx.range(f"loss-l1"):
                loss_l1 = torch.abs(rgb_pred - rgb_gt).mean()
                lambda_l1 = self.conf.loss.lambda_l1

        # L2 loss
        loss_l2 = torch.zeros(1, device=self.device)
        lambda_l2 = 0.0
        if self.conf.loss.use_l2:
            with torch.cuda.nvtx.range(f"loss-l2"):
                loss_l2 = torch.nn.functional.mse_loss(outputs["pred_rgb"], rgb_gt)
                lambda_l2 = self.conf.loss.lambda_l2

        # DSSIM loss
        loss_ssim = torch.zeros(1, device=self.device)
        lambda_ssim = 0.0
        if self.conf.loss.use_ssim:
            with torch.cuda.nvtx.range(f"loss-ssim"):
                rgb_gt_full = torch.permute(rgb_gt, (0, 3, 1, 2))
                pred_rgb_full = torch.permute(rgb_pred, (0, 3, 1, 2))
                loss_ssim = 1.0 - ssim(pred_rgb_full, rgb_gt_full)
                lambda_ssim = self.conf.loss.lambda_ssim

        # Opacity regularization
        loss_opacity = torch.zeros(1, device=self.device)
        lambda_opacity = 0.0
        if self.conf.loss.use_opacity:
            with torch.cuda.nvtx.range(f"loss-opacity"):
                loss_opacity = torch.abs(self.model.get_density()).mean()
                lambda_opacity = self.conf.loss.lambda_opacity

        # Mask entropy loss on rendered opacity
        loss_mask_entropy = torch.zeros(1, device=self.device)
        lambda_mask_entropy = 0.0
        if self.conf.loss.use_mask_entropy:
            pred_opacity = outputs.get("pred_opacity")
            if mask is not None and pred_opacity is not None:
                with torch.cuda.nvtx.range(f"loss-mask-entropy"):
                    loss_mask_entropy = mask_entropy_loss(pred_opacity, mask)
                    lambda_mask_entropy = self.conf.loss.lambda_mask_entropy

        # Scale regularization
        loss_scale = torch.zeros(1, device=self.device)
        lambda_scale = 0.0
        if self.conf.loss.use_scale:
            with torch.cuda.nvtx.range(f"loss-scale"):
                loss_scale = torch.abs(self.model.get_scale()).mean()
                lambda_scale = self.conf.loss.lambda_scale


        # Diffusion prior regularization on shading normals
        loss_priors_regularization = torch.zeros(1, device=self.device)
        lambda_priors_regularization = 0.0
        use_normal_prior_regularization = self.conf.loss.use_normal_prior_regularization
        normal_priors_end_iteration = self.conf.loss.normal_priors_end_iteration
        normal_prior_active = use_normal_prior_regularization and (
            normal_priors_end_iteration < 0 or self.global_step <= normal_priors_end_iteration
        )
        if normal_prior_active:
            pred_shadingnormal = outputs.get("pred_shadingnormal")
            prior = getattr(gpu_batch, "prior", None)
            prior_normal = getattr(prior, "normal", None) if prior is not None else None
            if pred_shadingnormal is not None and prior_normal is not None:
                with torch.cuda.nvtx.range(f"loss-priors-regularization"):
                    loss_priors_regularization = prior_normal_alignment_loss(
                        pred_shadingnormal,
                        prior_normal,
                        valid_mask=mask,
                    )
                    lambda_priors_regularization = self.conf.loss.lambda_normal_priors_regularization

        # Shading normal loss
        loss_shading_normal = torch.zeros(1, device=self.device)
        lambda_shading_normal = 0.0
        if self.conf.loss.use_pseudo_normal_supervision:
            pred_shadingnormal = outputs.get("pred_shadingnormal")
            pseudo_normal = getattr(gpu_batch, "pseudo_normal", None)
            pseudo_normal_mask = getattr(gpu_batch, "pseudo_normal_mask", None)
            if pred_shadingnormal is not None and pseudo_normal is not None and pseudo_normal_mask is not None:
                with torch.cuda.nvtx.range(f"loss-shading-normal"):
                    loss_shading_normal = pseudo_normal_loss(
                        pred_shadingnormal,
                        pseudo_normal,
                        valid_mask=pseudo_normal_mask,
                        detach_pseudo_normal=not normal_prior_active,
                    )
                    lambda_shading_normal = self.conf.loss.lambda_shading_normal

        # Depth distortion loss
        loss_depth_distortion = torch.zeros(1, device=self.device)
        lambda_depth_distortion = 0.0
        if (
            self.conf.loss.use_depth_distortion
            and self.global_step >= self.conf.loss.depth_distortion_start_iteration
        ):
            pred_depth_distortion = outputs.get("pred_depth_distortion")
            if pred_depth_distortion is not None:
                with torch.cuda.nvtx.range(f"loss-depth-distortion"):
                    loss_depth_distortion = depth_distortion_loss(pred_depth_distortion)
                    lambda_depth_distortion = self.conf.loss.lambda_depth_distortion

        # Edge aware smoothness loss
        loss_edge_aware_smoothness = torch.zeros(1, device=self.device)
        lambda_edge_aware_smoothness = 0.0
        if self.conf.loss.use_edge_aware_smoothness:
            with torch.cuda.nvtx.range(f"loss-edge-aware-smoothness"):
                edge_aware_smoothness_scale = self.conf.loss.edge_aware_smoothness_scale
                edge_aware_smoothness_eps = self.conf.loss.edge_aware_smoothness_eps
                edge_aware_smoothness_outputs = self.conf.loss.edge_aware_smoothness_outputs
                if len(edge_aware_smoothness_outputs) == 0:
                    raise ValueError("loss.edge_aware_smoothness_outputs must contain at least one output key")

                edge_aware_terms = []
                for smoothness_output_key in edge_aware_smoothness_outputs:
                    pred_smoothness_map = outputs.get(smoothness_output_key)
                    if pred_smoothness_map is None:
                        raise KeyError(f"Configured edge-aware smoothness output '{smoothness_output_key}' was not rendered")
                    edge_aware_terms.append(
                        edge_aware_smoothness_loss(
                            pred_smoothness_map,
                            rgb_gt,
                            mask=mask,
                            eps=edge_aware_smoothness_eps,
                            scale=edge_aware_smoothness_scale,
                        )
                    )

                loss_edge_aware_smoothness = torch.stack(edge_aware_terms).mean()
                lambda_edge_aware_smoothness = self.conf.loss.lambda_edge_aware_smoothness

        # Total loss
        loss = (
            lambda_l1 * loss_l1
            + lambda_l2 * loss_l2
            + lambda_ssim * loss_ssim
            + lambda_opacity * loss_opacity
            + lambda_mask_entropy * loss_mask_entropy
            + lambda_scale * loss_scale
            + lambda_shading_normal * loss_shading_normal
            + lambda_priors_regularization * loss_priors_regularization
            + lambda_depth_distortion * loss_depth_distortion
            + lambda_edge_aware_smoothness * loss_edge_aware_smoothness
        )
        return dict(
            total_loss=loss,
            l1_loss=lambda_l1 * loss_l1,
            l2_loss=lambda_l2 * loss_l2,
            ssim_loss=lambda_ssim * loss_ssim,
            opacity_loss=lambda_opacity * loss_opacity,
            mask_entropy_loss=lambda_mask_entropy * loss_mask_entropy,
            scale_loss=lambda_scale * loss_scale,
            shading_normal_loss=lambda_shading_normal * loss_shading_normal,
            priors_regularization_loss=lambda_priors_regularization * loss_priors_regularization,
            depth_distortion_loss=lambda_depth_distortion * loss_depth_distortion,
            edge_aware_smoothness_loss=lambda_edge_aware_smoothness * loss_edge_aware_smoothness,
        )

    @torch.cuda.nvtx.range("get_pbr_losses")
    def get_pbr_losses(
        self, gpu_batch: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Computes dictionary of PBR losses for current batch.
        Args:
            gpu_batch: GT data of current batch
            outputs: model prediction for current batch
        Returns:
            losses: dictionary of loss terms computed for current batch.
        """
        rgb_gt_srgb = gpu_batch.rgb_gt
        rgb_gt = torch.where(
            rgb_gt_srgb <= 0.04045,
            rgb_gt_srgb / 12.92,
            ((rgb_gt_srgb + 0.055) / 1.055) ** 2.4,
        )
        rgb_pred = outputs["pred_pbr"]
        mask = gpu_batch.mask
        gradient_mask = gpu_batch.gradient_mask

        # Mask out the invalid pixels if a gradient mask is provided.
        if gradient_mask is not None:
            rgb_gt = rgb_gt * gradient_mask
            rgb_pred = rgb_pred * gradient_mask

        # L1 loss
        loss_l1 = torch.zeros(1, device=self.device)
        lambda_l1 = 0.0
        if self.conf.loss.use_l1:
            with torch.cuda.nvtx.range(f"loss-l1"):
                loss_l1 = torch.abs(rgb_pred - rgb_gt).mean()
                lambda_l1 = self.conf.loss.lambda_l1

        # L2 loss
        loss_l2 = torch.zeros(1, device=self.device)
        lambda_l2 = 0.0
        if self.conf.loss.use_l2:
            with torch.cuda.nvtx.range(f"loss-l2"):
                loss_l2 = torch.nn.functional.mse_loss(rgb_pred, rgb_gt)
                lambda_l2 = self.conf.loss.lambda_l2

        # DSSIM loss
        loss_ssim = torch.zeros(1, device=self.device)
        lambda_ssim = 0.0
        if self.conf.loss.use_ssim:
            with torch.cuda.nvtx.range(f"loss-ssim"):
                rgb_gt_full = torch.permute(rgb_gt, (0, 3, 1, 2))
                pred_rgb_full = torch.permute(rgb_pred, (0, 3, 1, 2))
                loss_ssim = 1.0 - ssim(pred_rgb_full, rgb_gt_full)
                lambda_ssim = self.conf.loss.lambda_ssim

        # Material prior regularization
        loss_albedo_priors_regularization = torch.zeros(1, device=self.device)
        lambda_albedo_priors_regularization = 0.0
        loss_roughness_priors_regularization = torch.zeros(1, device=self.device)
        lambda_roughness_priors_regularization = 0.0

        pred_material = outputs.get("pred_material")
        prior = getattr(gpu_batch, "prior", None)
        material_mask = mask
        if gradient_mask is not None:
            material_mask = gradient_mask if material_mask is None else material_mask * gradient_mask

        use_albedo_prior_regularization = self.conf.loss.get("use_albedo_prior_regularization", False)
        albedo_priors_end_iteration = self.conf.loss.get("albedo_priors_end_iteration", -1)
        albedo_prior_active = use_albedo_prior_regularization and (
            albedo_priors_end_iteration < 0 or self.global_step <= albedo_priors_end_iteration
        )
        if albedo_prior_active and pred_material is not None:
            prior_albedo = getattr(prior, "albedo", None) if prior is not None else None
            if prior_albedo is not None:
                with torch.cuda.nvtx.range(f"loss-albedo-priors-regularization"):
                    pred_albedo = pred_material[..., :3]
                    loss_albedo_priors_regularization = masked_l2_loss(
                        pred_albedo,
                        prior_albedo,
                        material_mask,
                    )
                    lambda_albedo_priors_regularization = self.conf.loss.lambda_albedo_priors_regularization

        use_roughness_prior_regularization = self.conf.loss.get("use_roughness_prior_regularization", False)
        roughness_priors_end_iteration = self.conf.loss.get("roughness_priors_end_iteration", -1)
        roughness_prior_active = use_roughness_prior_regularization and (
            roughness_priors_end_iteration < 0 or self.global_step <= roughness_priors_end_iteration
        )
        if roughness_prior_active and pred_material is not None:
            prior_roughness = getattr(prior, "roughness", None) if prior is not None else None
            if prior_roughness is not None:
                with torch.cuda.nvtx.range(f"loss-roughness-priors-regularization"):
                    pred_roughness = pred_material[..., 3:4]
                    loss_roughness_priors_regularization = masked_l2_loss(
                        pred_roughness,
                        prior_roughness,
                        material_mask,
                    )
                    lambda_roughness_priors_regularization = self.conf.loss.lambda_roughness_priors_regularization

        # Edge aware smoothness loss
        loss_edge_aware_smoothness = torch.zeros(1, device=self.device)
        lambda_edge_aware_smoothness = 0.0
        if self.conf.loss.use_edge_aware_smoothness:
            with torch.cuda.nvtx.range(f"loss-edge-aware-smoothness"):
                edge_aware_smoothness_scale = self.conf.loss.edge_aware_smoothness_scale
                edge_aware_smoothness_eps = self.conf.loss.edge_aware_smoothness_eps
                edge_aware_smoothness_outputs = self.conf.loss.edge_aware_smoothness_outputs
                if len(edge_aware_smoothness_outputs) == 0:
                    raise ValueError("loss.edge_aware_smoothness_outputs must contain at least one output key")

                edge_aware_terms = []
                for smoothness_output_key in edge_aware_smoothness_outputs:
                    pred_smoothness_map = outputs.get(smoothness_output_key)
                    if pred_smoothness_map is None:
                        raise KeyError(f"Configured edge-aware smoothness output '{smoothness_output_key}' was not rendered")
                    edge_aware_terms.append(
                        edge_aware_smoothness_loss(
                            pred_smoothness_map,
                            rgb_gt,
                            mask=mask,
                            eps=edge_aware_smoothness_eps,
                            scale=edge_aware_smoothness_scale,
                        )
                    )

                loss_edge_aware_smoothness = torch.stack(edge_aware_terms).mean()
                lambda_edge_aware_smoothness = self.conf.loss.lambda_edge_aware_smoothness

        # Total loss
        loss = (
            lambda_l1 * loss_l1
            + lambda_l2 * loss_l2
            + lambda_ssim * loss_ssim
            + lambda_albedo_priors_regularization * loss_albedo_priors_regularization
            + lambda_roughness_priors_regularization * loss_roughness_priors_regularization
            + lambda_edge_aware_smoothness * loss_edge_aware_smoothness
        )
        return dict(
            total_loss=loss,
            l1_loss=lambda_l1 * loss_l1,
            l2_loss=lambda_l2 * loss_l2,
            ssim_loss=lambda_ssim * loss_ssim,
            albedo_priors_regularization_loss=lambda_albedo_priors_regularization * loss_albedo_priors_regularization,
            roughness_priors_regularization_loss=lambda_roughness_priors_regularization * loss_roughness_priors_regularization,
            edge_aware_smoothness_loss=lambda_edge_aware_smoothness * loss_edge_aware_smoothness,
        )

    def _compute_losses(
        self, gpu_batch: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if self.conf.render.method == "3dgptir":
            return self.get_pbr_losses(gpu_batch, outputs)
        return self.get_losses(gpu_batch, outputs)

    @torch.cuda.nvtx.range("log_validation_iter")
    def log_validation_iter(
        self,
        gpu_batch: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        batch_metrics: dict[str, Any],
        iteration: Optional[int] = None,
    ) -> None:
        """Log information after a single validation iteration.
        Args:
            gpu_batch: GT data of current batch
            outputs: model prediction for current batch
            batch_metrics: dictionary of metrics computed for current batch
            iteration: optional, local iteration number within the current pass, e.g 0 <= iter < len(dataset).
        """
        logger.log_progress(
            task_name="Validation",
            advance=1,
            iteration=f"{str(iteration)}",
            psnr=batch_metrics["psnr"],
            loss=batch_metrics["losses"]["total_loss"],
        )

    @torch.cuda.nvtx.range("log_validation_pass")
    def log_validation_pass(self, metrics: dict[str, Any]) -> None:
        """Log information after a single validation pass.
        Args:
            metrics: dictionary of aggregated metrics for all batches in current pass.
        """
        writer = self.tracking.writer
        global_step = self.global_step

        if "img_pred" in metrics:
            writer.add_images(
                "image/pred/val",
                torch.stack(metrics["img_pred"]),
                global_step,
                dataformats="NHWC",
            )
        if "img_gt" in metrics:
            writer.add_images(
                "image/gt",
                torch.stack(metrics["img_gt"]),
                global_step,
                dataformats="NHWC",
            )
        if "img_hit_counts" in metrics:
            writer.add_images(
                "image/hit_counts/val",
                torch.stack(metrics["img_hit_counts"]),
                global_step,
                dataformats="NHWC",
            )
        if "img_pred_dist" in metrics:
            writer.add_images(
                "image/dist/val",
                torch.stack(metrics["img_pred_dist"]),
                global_step,
                dataformats="NHWC",
            )
        if "img_pred_opacity" in metrics:
            writer.add_images(
                "image/opacity/val",
                torch.stack(metrics["img_pred_opacity"]),
                global_step,
                dataformats="NHWC",
            )

        mean_timings = {}
        if "timings" in metrics:
            for time_key in metrics["timings"]:
                mean_timings[time_key] = np.mean(metrics["timings"][time_key])
                writer.add_scalar("time/" + time_key + "/val", mean_timings[time_key], global_step)

        writer.add_scalar("num_particles/val", self.model.num_gaussians, self.global_step)

        mean_psnr = np.mean(metrics["psnr"])
        writer.add_scalar("psnr/val", mean_psnr, global_step)
        writer.add_scalar("ssim/val", np.mean(metrics["ssim"]), global_step)
        writer.add_scalar("lpips/val", np.mean(metrics["lpips"]), global_step)
        writer.add_scalar("hits/min/val", np.mean(metrics["hits_min"]), global_step)
        writer.add_scalar("hits/max/val", np.mean(metrics["hits_max"]), global_step)
        writer.add_scalar("hits/mean/val", np.mean(metrics["hits_mean"]), global_step)

        loss = np.mean(metrics["losses"]["total_loss"])
        writer.add_scalar("loss/total/val", loss, global_step)
        if self.conf.loss.use_l1:
            l1_loss = np.mean(metrics["losses"]["l1_loss"])
            writer.add_scalar("loss/l1/val", l1_loss, global_step)
        if self.conf.loss.use_l2:
            l2_loss = np.mean(metrics["losses"]["l2_loss"])
            writer.add_scalar("loss/l2/val", l2_loss, global_step)
        if self.conf.loss.use_ssim:
            ssim_loss = np.mean(metrics["losses"]["ssim_loss"])
            writer.add_scalar("loss/ssim/val", ssim_loss, global_step)

        table = {k: np.mean(v) for k, v in metrics.items() if k in ("psnr", "ssim", "lpips")}
        for time_key in mean_timings:
            table[time_key] = f"{'{:.2f}'.format(mean_timings[time_key])}" + " ms/it"
        logger.log_table(f"📊 Validation Metrics - Step {global_step}", record=table)

    @torch.cuda.nvtx.range(f"log_training_iter")
    def log_training_iter(
        self,
        gpu_batch: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        batch_metrics: dict[str, Any],
        iteration: Optional[int] = None,
    ) -> None:
        """Log information after a single training iteration.
        Args:
            gpu_batch: GT data of current batch
            outputs: model prediction for current batch
            batch_metrics: dictionary of metrics computed for current batch
            iteration: optional, local iteration number within the current pass, e.g 0 <= iter < len(dataset).
        """
        writer = self.tracking.writer
        global_step = self.global_step

        if self.conf.enable_writer and global_step > 0 and global_step % self.conf.log_frequency == 0:
            loss = np.mean(batch_metrics["losses"]["total_loss"])
            writer.add_scalar("loss/total/train", loss, global_step)
            if self.conf.loss.use_l1:
                l1_loss = np.mean(batch_metrics["losses"]["l1_loss"])
                writer.add_scalar("loss/l1/train", l1_loss, global_step)
            if self.conf.loss.use_l2:
                l2_loss = np.mean(batch_metrics["losses"]["l2_loss"])
                writer.add_scalar("loss/l2/train", l2_loss, global_step)
            if self.conf.loss.use_ssim:
                ssim_loss = np.mean(batch_metrics["losses"]["ssim_loss"])
                writer.add_scalar("loss/ssim/train", ssim_loss, global_step)
            if self.conf.loss.use_opacity and "opacity_loss" in batch_metrics["losses"]:
                opacity_loss = np.mean(batch_metrics["losses"]["opacity_loss"])
                writer.add_scalar("loss/opacity/train", opacity_loss, global_step)
            if self.conf.loss.use_mask_entropy and "mask_entropy_loss" in batch_metrics["losses"]:
                mask_entropy = np.mean(batch_metrics["losses"]["mask_entropy_loss"])
                writer.add_scalar("loss/mask_entropy/train", mask_entropy, global_step)
            if self.conf.loss.use_scale and "scale_loss" in batch_metrics["losses"]:
                scale_loss = np.mean(batch_metrics["losses"]["scale_loss"])
                writer.add_scalar("loss/scale/train", scale_loss, global_step)
            if self.conf.loss.use_pseudo_normal_supervision and "shading_normal_loss" in batch_metrics["losses"]:
                shading_normal_loss = np.mean(batch_metrics["losses"]["shading_normal_loss"])
                writer.add_scalar("loss/shading_normal/train", shading_normal_loss, global_step)
            if self.conf.loss.get("use_normal_prior_regularization", False) and "priors_regularization_loss" in batch_metrics["losses"]:
                priors_regularization_loss = np.mean(batch_metrics["losses"]["priors_regularization_loss"])
                writer.add_scalar("loss/priors_regularization/train", priors_regularization_loss, global_step)
            if (
                self.conf.loss.get("use_albedo_prior_regularization", False)
                and "albedo_priors_regularization_loss" in batch_metrics["losses"]
            ):
                albedo_priors_regularization_loss = np.mean(
                    batch_metrics["losses"]["albedo_priors_regularization_loss"]
                )
                writer.add_scalar("loss/albedo_priors_regularization/train", albedo_priors_regularization_loss, global_step)
            if (
                self.conf.loss.get("use_roughness_prior_regularization", False)
                and "roughness_priors_regularization_loss" in batch_metrics["losses"]
            ):
                roughness_priors_regularization_loss = np.mean(
                    batch_metrics["losses"]["roughness_priors_regularization_loss"]
                )
                writer.add_scalar(
                    "loss/roughness_priors_regularization/train",
                    roughness_priors_regularization_loss,
                    global_step,
                )
            if self.conf.loss.use_edge_aware_smoothness:
                edge_aware_smoothness_loss = np.mean(batch_metrics["losses"]["edge_aware_smoothness_loss"])
                writer.add_scalar("loss/edge_aware_smoothness/train", edge_aware_smoothness_loss, global_step)
            if self.post_processing is not None and "post_processing_reg_loss" in batch_metrics["losses"]:
                post_processing_reg_loss = np.mean(batch_metrics["losses"]["post_processing_reg_loss"])
                writer.add_scalar(
                    "loss/post_processing_reg/train",
                    post_processing_reg_loss,
                    global_step,
                )
            if "psnr" in batch_metrics:
                writer.add_scalar("psnr/train", batch_metrics["psnr"], self.global_step)
            if "ssim" in batch_metrics:
                writer.add_scalar("ssim/train", batch_metrics["ssim"], self.global_step)
            if "lpips" in batch_metrics:
                writer.add_scalar("lpips/train", batch_metrics["lpips"], self.global_step)
            if "hits_mean" in batch_metrics:
                writer.add_scalar("hits/mean/train", batch_metrics["hits_mean"], self.global_step)
            if "hits_std" in batch_metrics:
                writer.add_scalar("hits/std/train", batch_metrics["hits_std"], self.global_step)
            if "hits_min" in batch_metrics:
                writer.add_scalar("hits/min/train", batch_metrics["hits_min"], self.global_step)
            if "hits_max" in batch_metrics:
                writer.add_scalar("hits/max/train", batch_metrics["hits_max"], self.global_step)

            if "timings" in batch_metrics:
                for time_key in batch_metrics["timings"]:
                    writer.add_scalar(
                        "time/" + time_key + "/train",
                        batch_metrics["timings"][time_key],
                        self.global_step,
                    )

            writer.add_scalar("num_particles/train", self.model.num_gaussians, self.global_step)
            writer.add_scalar("train/num_GS", self.model.num_gaussians, self.global_step)

            # # NOTE: hack to easily compare with 3DGS
            # writer.add_scalar("train_loss_patches/total_loss", loss, global_step)
            # writer.add_scalar("gaussians/count", self.model.num_gaussians, self.global_step)

        logger.log_progress(
            task_name="Training",
            advance=1,
            step=f"{str(self.global_step)}",
            loss=batch_metrics["losses"]["total_loss"],
        )

    @torch.cuda.nvtx.range(f"log_training_pass")
    def log_training_pass(self, metrics):
        """Log information after a single training pass.
        Args:
            metrics: dictionary of aggregated metrics for all batches in current pass.
        """
        pass

    @torch.cuda.nvtx.range(f"on_training_end")
    def on_training_end(self):
        """Callback that prompts at the end of training."""
        conf = self.conf
        out_dir = self.tracking.output_dir

        # Export the mixture-of-3d-gaussians
        logger.log_rule("Exporting Models")

        if conf.export_ply.enabled:
            from threedgrut.export import PLYExporter

            ply_path = conf.export_ply.path if conf.export_ply.path else os.path.join(out_dir, "export_last.ply")
            exporter = PLYExporter()
            exporter.export(self.model, Path(ply_path), dataset=self.train_dataset, conf=conf)

        if conf.export_usd.enabled:
            from threedgrut.export import USDExporter

            # Determine format for filename suffix
            format_suffix = "lightfield"
            exporter = USDExporter.from_config(conf)

            # Handle path: if not set or relative, put in output directory
            if conf.export_usd.path:
                usdz_path = conf.export_usd.path
                if not os.path.isabs(usdz_path):
                    usdz_path = os.path.join(out_dir, usdz_path)
            else:
                # Default filename includes format suffix
                usdz_path = os.path.join(out_dir, f"export_last_{format_suffix}.usdz")

            exporter.export(
                self.model,
                Path(usdz_path),
                dataset=self.train_dataset,
                conf=conf,
                background=getattr(self, "background", None),
            )

        # Export post-processing report (PPISP-based)
        if self.post_processing is not None and conf.post_processing.method == "ppisp":
            from ppisp.report import export_ppisp_report

            logger.info("📊 Exporting PPISP report...")

            ppisp_report_dir = Path(out_dir) / "ppisp_report"
            frames_per_camera = self.train_dataset.get_frames_per_camera()

            # Get camera names if available
            camera_names = None
            if hasattr(self.train_dataset, "get_camera_names"):
                camera_names = self.train_dataset.get_camera_names()

            export_ppisp_report(
                self.post_processing,
                frames_per_camera=frames_per_camera,
                output_dir=ppisp_report_dir,
                camera_names=camera_names,
            )
            logger.info(f"📊 PPISP report saved to: {ppisp_report_dir}")

        self.teardown_dataloaders()
        self.save_checkpoint(last_checkpoint=True)

        # Evaluate on test set
        if conf.test_last:
            logger.log_rule("Evaluation on Test Set")

            # Renderer test split
            renderer = Renderer.from_preloaded_model(
                model=self.model,
                out_dir=out_dir,
                path=conf.path,
                save_gt=False,
                writer=self.tracking.writer,
                global_step=self.global_step,
                compute_extra_metrics=conf.compute_extra_metrics,
                post_processing=self.post_processing,
            )
            renderer.render_all()

    @torch.cuda.nvtx.range(f"save_checkpoint")
    def save_checkpoint(self, last_checkpoint: bool = False):
        """Saves checkpoint to a path under {conf.out_dir}/{conf.experiment_name}.
        Args:
            last_checkpoint: If true, will update checkpoint title to 'last'.
                             Otherwise uses global step
        """
        global_step = self.global_step
        out_dir = self.tracking.output_dir
        parameters = self.model.get_model_parameters()
        parameters |= {"global_step": self.global_step, "epoch": self.n_epochs - 1}
        if self.environment is not None:
            parameters["environment_state"] = self.environment.state_dict()

        strategy_parameters = self.strategy.get_strategy_parameters()
        parameters = {**parameters, **strategy_parameters}

        # Add post-processing state to checkpoint (module + optimizers + schedulers)
        if self.post_processing is not None:
            parameters["post_processing"] = {
                "module": self.post_processing.state_dict(),
                "optimizers": [opt.state_dict() for opt in self.post_processing_optimizers],
                "schedulers": [sched.state_dict() for sched in self.post_processing_schedulers],
            }

        os.makedirs(os.path.join(out_dir, f"ours_{int(global_step)}"), exist_ok=True)
        if not last_checkpoint:
            ckpt_path = os.path.join(out_dir, f"ours_{int(global_step)}", f"ckpt_{global_step}.pt")
        else:
            ckpt_path = os.path.join(out_dir, "ckpt_last.pt")
        torch.save(parameters, ckpt_path)
        logger.info(f'💾 Saved checkpoint to: "{os.path.abspath(ckpt_path)}"')
        if self.conf.render.method == "3dgptir" and self.environment is not None:
            envmap_path = os.path.splitext(ckpt_path)[0] + "_environment.exr"
            if save_environment_exr(self.environment.get_environment(), envmap_path) is not None:
                logger.info(f'🌐 Saved environment map to: "{os.path.abspath(envmap_path)}"')

    def render_gui(self, scene_updated):
        """Render & refresh a single frame for the gui"""
        gui = self.gui
        if gui is not None:
            import polyscope as ps

            if gui.live_update:
                if scene_updated or self.model.positions.requires_grad:
                    gui.update_cloud_viz()
                gui.update_render_view_viz()

            ps.frame_tick()
            while not gui.viz_do_train:
                ps.frame_tick()

            if ps.window_requests_close():
                logger.warning(
                    "Terminating training from GUI window is not supported. Please terminate it from the terminal."
                )

    def render_gui_viser(self, scene_updated):
        gui = self.gui
        if gui is not None:
            if gui.live_update:
                # update render view
                if scene_updated or self.model.positions.requires_grad:
                    gui.update_point_cloud()
                for client in gui.server.get_clients().values():
                    gui.update_render_view(client, force=True)
                while not gui.viz_do_train:
                    time.sleep(0.0001)

    @torch.cuda.nvtx.range(f"run_train_iter")
    def run_train_iter(
        self,
        global_step: int,
        batch: dict,
        profilers: dict,
        metrics: list,
        conf: DictConfig,
    ):
        # Freeze Gaussians and suspend strategy when distillation starts
        if self._distillation_start_step >= 0 and global_step >= self._distillation_start_step:
            self.model.freeze_gaussians()
            self.strategy.suspend()

        # Access the GPU-cache batch data
        with torch.cuda.nvtx.range(f"train_iter{global_step}_get_gpu_batch"):
            gpu_batch = self.train_dataset.get_gpu_batch_with_intrinsics(batch)

        # Perform validation if required
        is_time_to_validate = (global_step > 0 or conf.validate_first) and (global_step % self.val_frequency == 0)
        if is_time_to_validate:
            self.run_validation_pass(conf)

        # Compute the outputs of a single batch
        with torch.cuda.nvtx.range(f"train_{global_step}_fwd"):
            profilers["inference"].start()
            sh_indirect_iterations = int(conf.sh_indirect_iterations)
            sh_indirect = sh_indirect_iterations > 0 and global_step < sh_indirect_iterations
            outputs = self.model(
                gpu_batch,
                train=True,
                frame_id=global_step,
                sh_indirect=sh_indirect,
            )
            profilers["inference"].end()

        # Apply post-processing to rendered output
        if self.post_processing is not None:
            with torch.cuda.nvtx.range(f"train_{global_step}_post_processing"):
                outputs = apply_post_processing(self.post_processing, outputs, gpu_batch, training=True)

        # Convert depth to normal
        with torch.cuda.nvtx.range(f"train_{global_step}_pseudo_normal"):
            pred_dist = outputs.get("pred_dist")
            if pred_dist is not None:
                valid = pred_dist > 0
                pseudo_normal, pseudo_normal_mask = self.normal_utils.depth_to_pseudo_normal(
                    rays_o=gpu_batch.rays_ori,
                    rays_d=gpu_batch.rays_dir,
                    T_to_world=gpu_batch.T_to_world,
                    pred_dist=pred_dist,
                    valid=valid,
                    pred_opacity=outputs.get("pred_opacity"),
                    foreground_mask=gpu_batch.gradient_mask,
                )
                gpu_batch.pseudo_normal = pseudo_normal
                gpu_batch.pseudo_normal_mask = pseudo_normal_mask

        # Compute the losses of a single batch
        with torch.cuda.nvtx.range(f"train_{global_step}_loss"):
            batch_losses = self._compute_losses(gpu_batch, outputs)
            # Add post-processing regularization loss
            if self.post_processing is not None:
                post_processing_reg_loss = self.post_processing.get_regularization_loss()
                batch_losses["total_loss"] = batch_losses["total_loss"] + post_processing_reg_loss
                batch_losses["post_processing_reg_loss"] = post_processing_reg_loss

        # Backward strategy step
        with torch.cuda.nvtx.range(f"train_{global_step}_pre_bwd"):
            self.strategy.pre_backward(
                step=global_step,
                scene_extent=self.scene_extent,
                train_dataset=self.train_dataset,
                batch=gpu_batch,
                writer=self.tracking.writer,
            )

        # Back-propagate the gradients and update the parameters
        with torch.cuda.nvtx.range(f"train_{global_step}_bwd"):
            profilers["backward"].start()
            batch_losses["total_loss"].backward()
            profilers["backward"].end()

        # Post backward strategy step
        with torch.cuda.nvtx.range(f"train_{global_step}_post_bwd"):
            scene_updated = self.strategy.post_backward(
                step=global_step,
                scene_extent=self.scene_extent,
                train_dataset=self.train_dataset,
                batch=gpu_batch,
                writer=self.tracking.writer,
            )

        # Optimizer step
        with torch.cuda.nvtx.range(f"train_{global_step}_backprop"):
            if isinstance(self.model.optimizer, SelectiveAdam):
                assert (
                    outputs["mog_visibility"].shape == self.model.density.shape
                ), f"Visibility shape {outputs['mog_visibility'].shape} does not match density shape {self.model.density.shape}"
                self.model.optimizer.step(outputs["mog_visibility"])
            else:
                self.model.optimizer.step()
            self.model.optimizer.zero_grad()

        # Scheduler step
        with torch.cuda.nvtx.range(f"train_{global_step}_scheduler"):
            self.model.scheduler_step(global_step)

        # Post-processing optimizer/scheduler step
        if self.post_processing_optimizers is not None:
            with torch.cuda.nvtx.range(f"train_{global_step}_post_processing_opt"):
                for opt in self.post_processing_optimizers:
                    opt.step()
                    opt.zero_grad()
                for sched in self.post_processing_schedulers:
                    sched.step()

        # Post backward strategy step
        with torch.cuda.nvtx.range(f"train_{global_step}_post_opt_step"):
            scene_updated = self.strategy.post_optimizer_step(
                step=global_step,
                scene_extent=self.scene_extent,
                train_dataset=self.train_dataset,
                batch=gpu_batch,
                writer=self.tracking.writer,
            )

        # Update the SH if required
        if self.model.progressive_training and check_step_condition(
            global_step, 0, 1e6, self.model.feature_dim_increase_interval
        ):
            self.model.increase_num_active_features()

        # Update the BVH if required
        if scene_updated or (
            conf.model.bvh_update_frequency > 0 and global_step % conf.model.bvh_update_frequency == 0
        ):
            with torch.cuda.nvtx.range(f"train_{global_step}_bvh"):
                profilers["build_as"].start()
                self.model.build_acc(rebuild=True)
                profilers["build_as"].end()

        # Increment the global step
        global_step += 1
        self.global_step = global_step

        # Compute metrics
        batch_metrics = self.get_metrics(
            gpu_batch,
            outputs,
            batch_losses,
            profilers,
            split="training",
            iteration=iter,
        )
        if "forward_render" in self.model.renderer.timings:
            batch_metrics["timings"]["forward_render_cuda"] = self.model.renderer.timings["forward_render"]
        if "backward_render" in self.model.renderer.timings:
            batch_metrics["timings"]["backward_render_cuda"] = self.model.renderer.timings["backward_render"]
        metrics.append(batch_metrics)

        # !!! Below global step has been incremented !!!
        with torch.cuda.nvtx.range(f"train_{global_step - 1}_log_iter"):
            self.log_training_iter(gpu_batch, outputs, batch_metrics, iter)
        with torch.cuda.nvtx.range(f"train_{global_step - 1}_save_ckpt"):
            if global_step in conf.checkpoint.iterations:
                self.save_checkpoint()

        # Updating the GUI
        with torch.cuda.nvtx.range(f"train_{global_step - 1}_update_gui"):
            if self.conf.with_viser_gui:
                self.render_gui_viser(scene_updated)
            elif self.conf.with_gui:
                self.render_gui(scene_updated)

        # Visualize Training Images
        with torch.cuda.nvtx.range(f"train_{global_step - 1}_visualize"):
            visualization_outputs = outputs
            environment = self.model.get_environment()
            if environment is not None:
                visualization_outputs = dict(outputs)
                visualization_outputs["environment"] = environment
            self.visualizer.save(global_step, visualization_outputs, gpu_batch)

    @torch.cuda.nvtx.range(f"run_train_pass")
    def run_train_pass(self, conf: DictConfig):
        """Runs a single train epoch over the dataset."""
        metrics = []
        profilers = {
            "inference": CudaTimer(enabled=self.conf.enable_frame_timings),
            "backward": CudaTimer(enabled=self.conf.enable_frame_timings),
            "build_as": CudaTimer(enabled=self.conf.enable_frame_timings),
        }

        for iter, batch in enumerate(self.train_dataloader):
            # Check if we have reached the maximum number of iterations
            if self.global_step >= conf.n_iterations:
                return

            # Step for training iteration
            self.run_train_iter(self.global_step, batch, profilers, metrics, conf)

        self.log_training_pass(metrics)

    @torch.cuda.nvtx.range(f"run_validation_pass")
    @torch.no_grad()
    def run_validation_pass(self, conf: DictConfig) -> dict[str, Any]:
        """Runs a single validation epoch over the dataset.
        Returns:
             dictionary of metrics computed and aggregated over validation set.
        """

        profilers = {
            "inference": CudaTimer(),
        }
        metrics = []
        logger.info(f"Step {self.global_step} -- Running validation..")
        logger.start_progress(
            task_name="Validation",
            total_steps=len(self.val_dataloader),
            color="medium_purple3",
        )

        for val_iteration, batch_idx in enumerate(self.val_dataloader):
            # Access the GPU-cache batch data
            gpu_batch = self.val_dataset.get_gpu_batch_with_intrinsics(batch_idx)

            # Compute the outputs of a single batch
            with torch.cuda.nvtx.range(f"train.validation_step_{self.global_step}"):
                profilers["inference"].start()
                outputs = self.model(gpu_batch, train=False)
                # Apply post-processing for validation (novel view mode)
                if self.post_processing is not None:
                    outputs = apply_post_processing(self.post_processing, outputs, gpu_batch, training=False)
                profilers["inference"].end()

                batch_losses = self._compute_losses(gpu_batch, outputs)
                batch_metrics = self.get_metrics(
                    gpu_batch,
                    outputs,
                    batch_losses,
                    profilers,
                    split="validation",
                    iteration=val_iteration,
                )

                self.log_validation_iter(gpu_batch, outputs, batch_metrics, iteration=val_iteration)
                metrics.append(batch_metrics)

        logger.end_progress(task_name="Validation")

        metrics = self._flatten_list_of_dicts(metrics)
        self.log_validation_pass(metrics)
        return metrics

    @staticmethod
    def _flatten_list_of_dicts(list_of_dicts):
        """
        Converts list of dicts -> dict of lists.
        Supports flattening of up to 2 levels of dict hierarchies
        """
        flat_dict = defaultdict(list)
        for d in list_of_dicts:
            for k, v in d.items():
                if isinstance(v, dict):
                    flat_dict[k] = defaultdict(list) if k not in flat_dict else flat_dict[k]
                    for inner_k, inner_v in v.items():
                        flat_dict[k][inner_k].append(inner_v)
                else:
                    flat_dict[k].append(v)
        return flat_dict

    def run_training(self):
        """Initiate training logic for n_epochs.
        Training and validation are controlled by the config.
        """
        assert self.model.optimizer is not None, "Optimizer needs to be initialized before the training can start!"
        conf = self.conf

        logger.log_rule(f"Training {conf.render.method.upper()}")

        # Training loop
        logger.start_progress(task_name="Training", total_steps=conf.n_iterations, color="spring_green1")

        for epoch_idx in range(self.n_epochs):
            self.run_train_pass(conf)

        logger.end_progress(task_name="Training")

        # Report training statistics
        stats = logger.finished_tasks["Training"]
        elapsed = float(stats["elapsed"])
        iteration_speed = self.global_step / elapsed
        table = dict(
            n_steps=f"{self.global_step}",
            n_epochs=f"{self.n_epochs}",
            training_time=f"{elapsed:.2f} s",
            iteration_speed=f"{iteration_speed:.2f} it/s",
        )
        time_json = dict(
            n_steps=int(self.global_step),
            n_epochs=int(self.n_epochs),
            training_time=float(elapsed),
            iteration_speed=float(iteration_speed),
            training_time_text=table["training_time"],
            iteration_speed_text=table["iteration_speed"],
        )
        time_path = os.path.join(self.tracking.output_dir, "time.json")
        with open(time_path, "w") as f:
            json.dump(time_json, f, indent=2)
        logger.info(f"📄 Training time saved to: {time_path}")
        logger.log_table(f"🎊 Training Statistics", record=table)

        # Perform testing
        self.on_training_end()
        logger.info(f"🥳 Training Complete.")

        # Updating the GUI
        if self.gui is not None:
            self.gui.training_done = True
            logger.info(f"🎨 GUI Blocking... Terminate GUI to Stop.")
            self.gui.block_in_rendering_loop(fps=60)
