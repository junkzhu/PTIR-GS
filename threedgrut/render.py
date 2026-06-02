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
from pathlib import Path

import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import threedgrut.datasets as datasets
from threedgrut.model.environment import Environment
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.model.ptir_helper import (
    append_ptir_metrics,
    compute_albedo_rescale_ratio,
    compute_ptir_full_image_metrics,
    rescale_albedo,
    save_scaled_albedo_checkpoint,
    summarize_ptir_metrics,
)
from threedgrut.utils.color_correct import color_correct_affine
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import create_summary_writer
from threedgrut.utils.normal import normal_mae
from threedgrut.utils.render import apply_post_processing


class Renderer:
    def __init__(
        self,
        model,
        conf,
        global_step,
        out_dir,
        path="",
        save_gt=True,
        writer=None,
        compute_extra_metrics=True,
        post_processing=None,
        checkpoint_path=None,
    ) -> None:

        if path:  # Replace the path to the test data
            conf.path = path

        self.model = model
        self.out_dir = out_dir
        self.save_gt = save_gt
        self.path = path
        self.conf = conf
        self.global_step = global_step
        self.dataset, self.dataloader = self.create_test_dataloader(conf)
        self.writer = writer
        self.compute_extra_metrics = compute_extra_metrics
        self.post_processing = post_processing
        self.checkpoint_path = checkpoint_path

        if conf.model.background.color == "black":
            self.bg_color = torch.zeros((3,), dtype=torch.float32, device="cuda")
        elif conf.model.background.color == "white":
            self.bg_color = torch.ones((3,), dtype=torch.float32, device="cuda")
        else:
            assert False, f"{conf.model.background.color} is not a supported background color."

    @staticmethod
    def _linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
        image = image.clip(0.0, 1.0)
        return torch.where(
            image <= 0.0031308,
            12.92 * image,
            1.055 * image ** (1.0 / 2.4) - 0.055,
        )

    @staticmethod
    def _save_nhwc_image(image: torch.Tensor, path: str, linear_to_srgb: bool = False) -> None:
        if linear_to_srgb:
            image = Renderer._linear_to_srgb(image)
        else:
            image = image.clip(0.0, 1.0)
        torchvision.utils.save_image(image.squeeze(0).permute(2, 0, 1), path)

    @staticmethod
    def _restore_environment_from_checkpoint(model, conf, checkpoint: dict) -> None:
        if conf.render.method != "3dgptir" and OmegaConf.select(conf, "environment", default=None) is None:
            return

        environment = Environment(
            path=OmegaConf.select(conf, "environment.path", default=None),
            device=model.device,
            environment_type=OmegaConf.select(conf, "environment.type", default="2d"),
            optimize_environment=bool(OmegaConf.select(conf, "model.optimize_environment", default=False)),
            parameterization=OmegaConf.select(conf, "environment.parameterization", default="linear"),
        )
        environment_state = checkpoint.get("environment_state")
        if environment_state is not None:
            environment.load_state_dict(environment_state)
            environment.configure_optimization(bool(OmegaConf.select(conf, "model.optimize_environment", default=False)))

        model.optimize_environment = environment.optimize_environment
        model.environment_parameterization = environment.environment_parameterization
        model.environment = environment.get_environment_parameter()
        model.environment_alias_table = None
        if conf.render.method == "3dgptir" and conf.render.get("enable_mis", False):
            model.environment_alias_table = environment.build_alias_table()

    def create_test_dataloader(self, conf):
        """Create the test dataloader for the given configuration."""
        from threedgrut.datasets.utils import configure_dataloader_for_platform

        dataset = datasets.make_test(name=conf.dataset.type, config=conf)

        # Configure DataLoader arguments for the current platform
        dataloader_kwargs = configure_dataloader_for_platform(
            {
                "num_workers": 8,
                "batch_size": 1,
                "shuffle": False,
                "collate_fn": None,
            }
        )

        dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
        return dataset, dataloader

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path,
        out_dir,
        path="",
        save_gt=True,
        writer=None,
        model=None,
        computes_extra_metrics=True,
    ):
        """Loads checkpoint for test path.
        If path is stated, it will override the test path in checkpoint.
        If model is None, it will be loaded base on the
        """

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        global_step = checkpoint["global_step"]

        conf = checkpoint["config"]
        # overrides
        if conf["render"]["method"] in ("3dgrt", "3dgptir"):
            conf["render"]["particle_kernel_density_clamping"] = True
            conf["render"]["min_transmittance"] = 0.03
        conf["render"]["enable_kernel_timings"] = True

        object_name = Path(conf.path).stem
        experiment_name = conf["experiment_name"]
        writer, out_dir, run_name = create_summary_writer(conf, object_name, out_dir, experiment_name, use_wandb=False)

        if model is None:
            # Initialize the model and the optix context
            model = MixtureOfGaussians(conf)
            # Initialize the parameters from checkpoint
            model.init_from_checkpoint(checkpoint, setup_optimizer=False)
        cls._restore_environment_from_checkpoint(model, conf, checkpoint)
        model.build_acc()

        # Load post-processing if present in checkpoint
        post_processing = None
        method = conf.post_processing.method
        if "post_processing" in checkpoint and method == "ppisp":
            from ppisp import PPISP, PPISPConfig

            # Derive config from training settings to match trainer.py
            use_controller = conf.post_processing.get("use_controller", True)
            n_distillation_steps = conf.post_processing.get("n_distillation_steps", 5000)
            if use_controller and n_distillation_steps > 0:
                main_training_steps = conf.n_iterations - n_distillation_steps
                controller_activation_ratio = main_training_steps / conf.n_iterations
                controller_distillation = True
            elif use_controller:
                controller_activation_ratio = 0.8
                controller_distillation = False
            else:
                controller_activation_ratio = 0.0
                controller_distillation = False

            ppisp_config = PPISPConfig(
                use_controller=use_controller,
                controller_distillation=controller_distillation,
                controller_activation_ratio=controller_activation_ratio,
            )

            post_processing = PPISP.from_state_dict(checkpoint["post_processing"]["module"], config=ppisp_config)
            post_processing = post_processing.to("cuda")
            num_cameras = post_processing.crf_params.shape[0]
            num_frames = post_processing.exposure_params.shape[0]
            logger.info(f"📷 {method.upper()} loaded from checkpoint: {num_cameras} cameras, {num_frames} frames")

        return Renderer(
            model=model,
            conf=conf,
            global_step=global_step,
            out_dir=out_dir,
            path=path,
            save_gt=save_gt,
            writer=writer,
            compute_extra_metrics=computes_extra_metrics,
            post_processing=post_processing,
            checkpoint_path=checkpoint_path,
        )

    @classmethod
    def from_preloaded_model(
        cls,
        model,
        out_dir,
        path="",
        save_gt=True,
        writer=None,
        global_step=None,
        compute_extra_metrics=False,
        post_processing=None,
        checkpoint_path=None,
    ):
        """Loads checkpoint for test path."""

        conf = model.conf
        if global_step is None:
            global_step = ""
        model.build_acc()
        return Renderer(
            model=model,
            conf=conf,
            global_step=global_step,
            out_dir=out_dir,
            path=path,
            save_gt=save_gt,
            writer=writer,
            compute_extra_metrics=compute_extra_metrics,
            post_processing=post_processing,
            checkpoint_path=checkpoint_path,
        )

    @torch.no_grad()
    def render_all(self):
        """Render all the images in the test dataset and log the metrics."""

        # Criterions that we log during training
        criterions = {"psnr": PeakSignalNoiseRatio(data_range=1).to("cuda")}

        if self.compute_extra_metrics:
            criterions |= {
                "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda"),
                "lpips": LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to("cuda"),
            }

        is_ptir = self.conf.render.method == "3dgptir"
        save_renders = not is_ptir
        output_path_renders = None
        if save_renders:
            output_path_renders = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "renders")
            os.makedirs(output_path_renders, exist_ok=True)

        output_path_normals = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "normals")
        os.makedirs(output_path_normals, exist_ok=True)

        output_path_ptir_aovs = {}
        if is_ptir:
            for aov_name in ("direct", "indirect", "pbr", "albedo", "roughness"):
                output_path_ptir_aovs[aov_name] = os.path.join(
                    self.out_dir, f"ours_{int(self.global_step)}", aov_name
                )
                os.makedirs(output_path_ptir_aovs[aov_name], exist_ok=True)

        albedo_frame_names = []
        albedo_gt_list = []
        albedo_list = []
        albedo_rescale_single = None
        albedo_rescale_rgb = None
        ptir_metric_lists = {}

        if self.save_gt:
            output_path_gt = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "gt")
            os.makedirs(output_path_gt, exist_ok=True)

        psnr = []
        ssim = []
        lpips = []
        cc_psnr = []
        cc_ssim = []
        cc_lpips = []
        normal_maes = []
        inference_time = []
        compute_normal_mae = self.conf.dataset.get("normal", False)

        best_psnr = -1.0
        worst_psnr = 2**16 * 1.0

        best_psnr_img = None
        best_psnr_img_gt = None

        worst_psnr_img = None
        worst_psnr_img_gt = None

        logger.start_progress(task_name="Rendering", total_steps=len(self.dataloader), color="orange1")

        for iteration, batch in enumerate(self.dataloader):
            frame_name = "{0:05d}.png".format(iteration)

            # Get the GPU-cached batch
            gpu_batch = self.dataset.get_gpu_batch_with_intrinsics(batch)

            # Compute the outputs of a single batch
            outputs = self.model(gpu_batch)

            # Apply post-processing
            if self.post_processing is not None:
                outputs = apply_post_processing(self.post_processing, outputs, gpu_batch, training=False)

            pred_rgb_full = outputs["pred_rgb"]
            rgb_gt_full = gpu_batch.rgb_gt
            normal_gt = getattr(gpu_batch, "normal_gt", None)
            material_albedo_gt = getattr(gpu_batch, "material_albedo_gt", None)
            material_roughness_gt = getattr(gpu_batch, "material_roughness_gt", None)
            if is_ptir:
                metric_rgb_full = self._linear_to_srgb(outputs["pred_pbr"])
            else:
                metric_rgb_full = pred_rgb_full

            # The values are already alpha composited with the background
            if output_path_renders is not None:
                self._save_nhwc_image(pred_rgb_full, os.path.join(output_path_renders, frame_name))
            pred_shadingnormal = outputs.get("pred_shadingnormal")
            pred_normals_full = pred_shadingnormal if pred_shadingnormal is not None else outputs.get("pred_normals")
            if pred_normals_full is not None:
                normals_to_write = (0.5 * (pred_normals_full + 1.0)).clip(0, 1.0)
                self._save_nhwc_image(normals_to_write, os.path.join(output_path_normals, frame_name))

            pred_material = outputs.get("pred_material")
            pred_roughness = None
            if pred_material is not None:
                pred_roughness = pred_material[..., 3:4]

            if output_path_ptir_aovs:
                ptir_aovs = {
                    "direct": outputs.get("pred_direct"),
                    "indirect": outputs.get("pred_indirect"),
                    "pbr": outputs.get("pred_pbr"),
                }
                for aov_name, aov_image in ptir_aovs.items():
                    if aov_image is not None:
                        self._save_nhwc_image(
                            aov_image,
                            os.path.join(output_path_ptir_aovs[aov_name], frame_name),
                            linear_to_srgb=True,
                        )

                if pred_material is not None:
                    albedo = pred_material[..., 0:3]
                    roughness = pred_roughness.repeat(1, 1, 1, 3)
                    self._save_nhwc_image(albedo, os.path.join(output_path_ptir_aovs["albedo"], frame_name))
                    self._save_nhwc_image(roughness, os.path.join(output_path_ptir_aovs["roughness"], frame_name))
                    if material_albedo_gt is not None:
                        albedo_frame_names.append(frame_name)
                        albedo_gt_list.append(material_albedo_gt.detach().cpu())
                        albedo_list.append(albedo.detach().cpu())

            pred_img_to_write = metric_rgb_full[-1].clip(0, 1.0)
            gt_img_to_write = rgb_gt_full[-1].clip(0, 1.0)

            if self.save_gt:
                self._save_nhwc_image(rgb_gt_full, os.path.join(output_path_gt, frame_name))

            # Compute the loss
            ptir_frame_metrics = {}
            normal_mask = None
            if normal_gt is not None:
                normal_mask = getattr(gpu_batch, "mask", None)
                if normal_mask is not None:
                    normal_mask = normal_mask[..., 0] > 0.5
                else:
                    normal_mask = normal_gt.abs().sum(dim=-1) > 1e-6
            if is_ptir:
                ptir_frame_metrics = compute_ptir_full_image_metrics(
                    criterions=criterions,
                    pred_pbr=metric_rgb_full,
                    rgb_gt=rgb_gt_full,
                    roughness=pred_roughness if material_roughness_gt is not None else None,
                    roughness_gt=material_roughness_gt,
                    normal=pred_shadingnormal if normal_gt is not None else None,
                    normal_gt=normal_gt,
                    normal_valid_mask=normal_mask,
                )
                append_ptir_metrics(ptir_metric_lists, ptir_frame_metrics)
                psnr_single_img = ptir_frame_metrics["psnr_pbr"]
            else:
                psnr_single_img = criterions["psnr"](metric_rgb_full, rgb_gt_full).item()
            psnr.append(psnr_single_img)  # evaluation on valid rays only
            normal_mae_single_img = None

            if is_ptir:
                normal_mae_single_img = ptir_frame_metrics.get("mae_normal")
                if normal_mae_single_img is not None:
                    normal_maes.append(normal_mae_single_img)
            elif compute_normal_mae:
                if pred_shadingnormal is not None and normal_gt is not None:
                    normal_mae_single_img = normal_mae(
                        pred_shadingnormal,
                        normal_gt,
                        valid_mask=normal_mask,
                        average_full_image=True,
                    ).item()
                    normal_maes.append(normal_mae_single_img)

            log_msg = f"Frame {iteration}, PSNR: {psnr[-1]}"
            if normal_mae_single_img is not None:
                log_msg += f", Normal MAE: {normal_mae_single_img}"
            logger.info(log_msg)

            if psnr_single_img > best_psnr:
                best_psnr = psnr_single_img
                best_psnr_img = pred_img_to_write
                best_psnr_img_gt = gt_img_to_write

            if psnr_single_img < worst_psnr:
                worst_psnr = psnr_single_img
                worst_psnr_img = pred_img_to_write
                worst_psnr_img_gt = gt_img_to_write

            # evaluate on full image
            if is_ptir:
                ssim.append(ptir_frame_metrics["ssim_pbr"])
                lpips.append(ptir_frame_metrics["lpips_pbr"])
            else:
                ssim.append(
                    criterions["ssim"](
                        metric_rgb_full.permute(0, 3, 1, 2),
                        rgb_gt_full.permute(0, 3, 1, 2),
                    ).item()
                )
                lpips.append(
                    criterions["lpips"](
                        metric_rgb_full.clip(0, 1).permute(0, 3, 1, 2),
                        rgb_gt_full.permute(0, 3, 1, 2),
                    ).item()
                )

            # Color-corrected metrics
            pred_rgb_cc = color_correct_affine(metric_rgb_full, rgb_gt_full)
            cc_psnr.append(criterions["psnr"](pred_rgb_cc, rgb_gt_full).item())
            cc_ssim.append(
                criterions["ssim"](
                    pred_rgb_cc.permute(0, 3, 1, 2),
                    rgb_gt_full.permute(0, 3, 1, 2),
                ).item()
            )
            cc_lpips.append(
                criterions["lpips"](
                    pred_rgb_cc.clip(0, 1).permute(0, 3, 1, 2),
                    rgb_gt_full.permute(0, 3, 1, 2),
                ).item()
            )

            # Record the time
            inference_time.append(outputs["frame_time_ms"])

            progress_metrics = {"iteration": f"{str(iteration)}", "psnr": psnr[-1]}
            if normal_mae_single_img is not None:
                progress_metrics["normal_mae"] = normal_mae_single_img
            logger.log_progress(task_name="Rendering", advance=1, **progress_metrics)

        logger.end_progress(task_name="Rendering")

        if output_path_ptir_aovs:
            if albedo_list:
                albedo_rescale_single, albedo_rescale_rgb, albedo_rescale_ratio = compute_albedo_rescale_ratio(
                    albedo_gt_list,
                    albedo_list,
                    selection_context=(
                        self.out_dir,
                        self.conf.get("path", ""),
                        self.conf.get("experiment_name", ""),
                    ),
                )
                output_path_albedo_scaled = os.path.join(
                    self.out_dir, f"ours_{int(self.global_step)}", "albedo_scaled"
                )
                os.makedirs(output_path_albedo_scaled, exist_ok=True)
                selected_values = torch.as_tensor(albedo_rescale_ratio).detach().cpu().reshape(-1).tolist()
                logger.info(
                    "PTIR albedo rescale ratio: "
                    f"single={albedo_rescale_single.item():.6f}, "
                    f"rgb={[round(v, 6) for v in albedo_rescale_rgb.tolist()]}, "
                    f"selected={[round(v, 6) for v in selected_values]}"
                )
                for frame_name, albedo, albedo_gt in zip(albedo_frame_names, albedo_list, albedo_gt_list):
                    albedo_scaled = rescale_albedo(albedo, albedo_rescale_ratio)
                    self._save_nhwc_image(albedo_scaled, os.path.join(output_path_albedo_scaled, frame_name))
                    append_ptir_metrics(
                        ptir_metric_lists,
                        compute_ptir_full_image_metrics(
                            criterions=criterions,
                            albedo_scaled=albedo_scaled,
                            albedo_gt=albedo_gt,
                        ),
                    )
                logger.info(f"Scaled albedo saved to: {output_path_albedo_scaled}")
                scaled_ckpt_path = save_scaled_albedo_checkpoint(
                    self.model,
                    self.out_dir,
                    albedo_rescale_ratio,
                    source_checkpoint_path=self.checkpoint_path,
                )
                logger.info(f'Scaled albedo checkpoint saved to: "{os.path.abspath(scaled_ckpt_path)}"')
            else:
                logger.info("PTIR albedo scaling skipped: no material_albedo_gt was found in the test batches.")

        ptir_metrics = summarize_ptir_metrics(ptir_metric_lists, include_values=False)
        mean_psnr = np.mean(psnr)
        mean_ssim = np.mean(ssim)
        mean_lpips = np.mean(lpips)
        mean_cc_psnr = np.mean(cc_psnr)
        mean_cc_ssim = np.mean(cc_ssim)
        mean_cc_lpips = np.mean(cc_lpips)
        mean_normal_mae = np.mean(normal_maes) if normal_maes else None
        std_psnr = np.std(psnr)
        mean_inference_time = np.mean(inference_time)

        if is_ptir:
            table = {}
        else:
            table = dict(
                mean_psnr=mean_psnr,
                mean_ssim=mean_ssim,
                mean_lpips=mean_lpips,
                mean_cc_psnr=mean_cc_psnr,
                mean_cc_ssim=mean_cc_ssim,
                mean_cc_lpips=mean_cc_lpips,
                std_psnr=std_psnr,
            )
            if mean_normal_mae is not None:
                table["normal_mae"] = mean_normal_mae
        if not is_ptir and albedo_rescale_single is not None and albedo_rescale_rgb is not None:
            table["albedo_rescale_single"] = float(albedo_rescale_single.item())
            table["albedo_rescale_rgb"] = str([round(value, 6) for value in albedo_rescale_rgb.tolist()])
        if is_ptir:
            for key in (
                "psnr_pbr_mean",
                "ssim_pbr_mean",
                "lpips_pbr_mean",
                "psnr_albedo_scaled_mean",
                "ssim_albedo_scaled_mean",
                "lpips_albedo_scaled_mean",
                "mse_roughness_mean",
                "mae_normal_mean",
            ):
                if key in ptir_metrics:
                    table[key] = ptir_metrics[key]
        else:
            for key, value in ptir_metrics.items():
                if key.endswith("_mean"):
                    table[key] = value

        if self.conf.render.enable_kernel_timings:
            table["mean_inference_time"] = f"{'{:.2f}'.format(mean_inference_time)}" + " ms/frame"

        # Save metrics to JSON file
        if is_ptir:
            metrics_json = {}
        else:
            metrics_json = dict(
                mean_psnr=float(mean_psnr),
                mean_ssim=float(mean_ssim),
                mean_lpips=float(mean_lpips),
                mean_cc_psnr=float(mean_cc_psnr),
                mean_cc_ssim=float(mean_cc_ssim),
                mean_cc_lpips=float(mean_cc_lpips),
            )
            if mean_normal_mae is not None:
                metrics_json["normal_mae"] = float(mean_normal_mae)
        if albedo_rescale_single is not None and albedo_rescale_rgb is not None:
            metrics_json["albedo_rescale_single"] = float(albedo_rescale_single.item())
            metrics_json["albedo_rescale_rgb"] = [float(value) for value in albedo_rescale_rgb.tolist()]
        if is_ptir:
            for key in (
                "psnr_pbr_mean",
                "ssim_pbr_mean",
                "lpips_pbr_mean",
                "psnr_albedo_scaled_mean",
                "ssim_albedo_scaled_mean",
                "lpips_albedo_scaled_mean",
                "mse_roughness_mean",
                "mae_normal_mean",
            ):
                if key in ptir_metrics:
                    metrics_json[key] = ptir_metrics[key]
        else:
            metrics_json.update(ptir_metrics)
        metrics_path = os.path.join(self.out_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_json, f, indent=2)
        logger.info(f"📄 Metrics saved to: {metrics_path}")

        logger.log_table(f"⭐ Test Metrics - Step {self.global_step}", record=table)

        if self.writer is not None:
            self.writer.add_scalar("psnr/test", mean_psnr, self.global_step)
            self.writer.add_scalar("ssim/test", mean_ssim, self.global_step)
            self.writer.add_scalar("lpips/test", mean_lpips, self.global_step)
            self.writer.add_scalar("cc_psnr/test", mean_cc_psnr, self.global_step)
            self.writer.add_scalar("cc_ssim/test", mean_cc_ssim, self.global_step)
            self.writer.add_scalar("cc_lpips/test", mean_cc_lpips, self.global_step)
            if mean_normal_mae is not None:
                self.writer.add_scalar("normal_mae/test", mean_normal_mae, self.global_step)
            self.writer.add_scalar("time/inference/test", mean_inference_time, self.global_step)

            if best_psnr_img is not None:
                self.writer.add_images(
                    "image/best_psnr/test",
                    torch.stack([best_psnr_img, best_psnr_img_gt]),
                    self.global_step,
                    dataformats="NHWC",
                )

            if worst_psnr_img is not None:
                self.writer.add_images(
                    "image/worst_psnr/test",
                    torch.stack([worst_psnr_img, worst_psnr_img_gt]),
                    self.global_step,
                    dataformats="NHWC",
                )

        return mean_psnr, std_psnr, mean_inference_time
