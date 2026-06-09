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

import logging
import math
import os
from enum import IntEnum

import torch
import torch.utils.cpp_extension

from threedgrut.datasets.protocols import Batch
from threedgrut.model.filters import Filter
from threedgrut.model.ptir_helper import post_processing
from threedgrut.utils.logger import logger as rich_logger
from threedgrut.utils.timer import CudaTimer

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
#

_threedgptir_plugin = None


def load_threedgptir_plugin(conf):
    global _threedgptir_plugin
    if _threedgptir_plugin is None:
        try:
            from . import libthreedgptir_cc as threedgptir  # type: ignore
        except ImportError:
            from .setup_threedgptir import setup_threedgptir

            threedgptir = setup_threedgptir(conf)
        _threedgptir_plugin = threedgptir


# ----------------------------------------------------------------------------
#
class Tracer:
    class _Autograd(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            tracer_wrapper,
            frame_id,
            ray_to_world,
            ray_ori,
            ray_dir,
            mog_pos,
            mog_rot,
            mog_scl,
            mog_dns,
            mog_sph,
            mog_snrm,
            mog_malb,
            mog_mrgh,
            mog_mmet,
            environment,
            environment_alias_table,
            render_opts,
            sph_degree,
            min_transmittance,
            max_bounces,
        ):
            particle_density = torch.concat(
                [mog_pos, mog_dns, mog_rot, mog_scl, torch.zeros_like(mog_dns)], dim=1
            )
            particle_material = torch.concat([mog_malb, mog_mrgh, mog_mmet], dim=1)
            (
                ray_radiance,
                ray_density,
                ray_hit_distance,
                ray_hit_distance_second_moment,
                ray_depth_distortion,
                ray_normals,
                ray_shadingnormal,
                ray_material,
                hits_count,
                mog_visibility,
                ray_pbr,
                ray_light,
                pbr_components,
            ) = tracer_wrapper.trace(
                frame_id,
                ray_to_world,
                ray_ori,
                ray_dir,
                particle_density,
                particle_material,
                mog_sph,
                mog_snrm,
                environment,
                environment_alias_table,
                render_opts,
                sph_degree,
                min_transmittance,
                max_bounces,
            )
            ctx.save_for_backward(
                ray_to_world,
                ray_ori,
                ray_dir,
                ray_radiance,
                ray_density,
                ray_hit_distance,
                ray_hit_distance_second_moment,
                ray_depth_distortion,
                ray_normals,
                ray_shadingnormal,
                ray_material,
                ray_pbr,
                ray_light,
                particle_density,
                particle_material,
                mog_sph,
                mog_snrm,
                environment,
                environment_alias_table,
            )
            ctx.frame_id = frame_id
            ctx.render_opts = render_opts
            ctx.sph_degree = sph_degree
            ctx.min_transmittance = min_transmittance
            ctx.max_bounces = max_bounces
            ctx.tracer_wrapper = tracer_wrapper
            return (
                ray_radiance,
                ray_density,
                ray_hit_distance[:, :, :, 0:1],  # return only the hit distance
                ray_hit_distance_second_moment,
                ray_depth_distortion,
                ray_normals,
                ray_shadingnormal,
                ray_material,
                hits_count,
                mog_visibility,
                ray_pbr,
                ray_light,
                pbr_components,
            )

        @staticmethod
        def backward(
            ctx,
            ray_radiance_grd,
            ray_density_grd,
            ray_hit_distance_grd,
            ray_hit_distance_second_moment_grd,
            ray_depth_distortion_grd,
            ray_normals_grd,
            ray_shadingnormal_grd,
            ray_material_grd,
            ray_hits_count_grd_UNUSED,
            mog_visibility_grd_UNUSED,
            ray_pbr_grd,
            ray_light_grd,
            pbr_components_grd_UNUSED,
        ):
            (
                ray_to_world,
                ray_ori,
                ray_dir,
                ray_radiance,
                ray_density,
                ray_hit_distance,
                ray_hit_distance_second_moment,
                ray_depth_distortion,
                ray_normals,
                ray_shadingnormal,
                ray_material,
                ray_pbr,
                ray_light,
                particle_density,
                particle_material,
                mog_sph,
                mog_snrm,
                environment,
                environment_alias_table,
            ) = ctx.saved_variables
            frame_id = ctx.frame_id
            if ray_light_grd is None:
                ray_light_grd = torch.zeros_like(ray_light)
            (
                particle_density_grd,
                particle_material_grd,
                mog_sph_grd,
                mog_sn_grd,
                environment_grd,
            ) = ctx.tracer_wrapper.trace_bwd(
                frame_id,
                ray_to_world,
                ray_ori,
                ray_dir,
                ray_radiance,
                ray_density,
                ray_hit_distance,
                ray_hit_distance_second_moment,
                ray_depth_distortion,
                ray_normals,
                ray_shadingnormal,
                ray_material,
                ray_pbr,
                ray_light,
                particle_density,
                particle_material,
                mog_sph,
                mog_snrm,
                environment,
                environment_alias_table,
                ray_radiance_grd,
                ray_density_grd,
                ray_hit_distance_grd,
                ray_hit_distance_second_moment_grd,
                ray_depth_distortion_grd,
                ray_normals_grd,
                ray_shadingnormal_grd,
                ray_material_grd,
                ray_pbr_grd,
                ray_light_grd,
                ctx.render_opts,
                ctx.sph_degree,
                ctx.min_transmittance,
                ctx.max_bounces,
            )
            mog_pos_grd, mog_dns_grd, mog_rot_grd, mog_scl_grd, _ = torch.split(
                particle_density_grd, [3, 1, 4, 3, 1], dim=1
            )
            mog_malb_grd, mog_mrgh_grd, mog_mmet_grd = torch.split(
                particle_material_grd, [3, 1, 1], dim=1
            )
            return (
                None,
                None,
                None,
                None,
                None,
                mog_pos_grd,
                mog_rot_grd,
                mog_scl_grd,
                mog_dns_grd,
                mog_sph_grd,
                mog_sn_grd,
                mog_malb_grd,
                mog_mrgh_grd,
                mog_mmet_grd,
                environment_grd,
                None,
                None,
                None,
                None,
                None,
            )

    class RenderOpts(IntEnum):
        NONE = 0
        INDIRECT = 1
        DEFAULT = NONE

    _MULTISPP_PATTERNS = {
        1: ((0.5000, 0.5000),),
        2: ((0.2500, 0.2500), (0.7500, 0.7500)),
        4: ((0.3750, 0.1250), (0.8750, 0.3750), (0.6250, 0.8750), (0.1250, 0.6250)),
        8: (
            (0.5625, 0.6875),
            (0.4375, 0.3125),
            (0.8125, 0.4375),
            (0.3125, 0.8125),
            (0.1875, 0.1875),
            (0.0625, 0.5625),
            (0.6875, 0.0625),
            (0.9375, 0.9375),
        ),
        16: (
            (0.5625, 0.4375),
            (0.4375, 0.6875),
            (0.3125, 0.3750),
            (0.7500, 0.5625),
            (0.1875, 0.6250),
            (0.6250, 0.1875),
            (0.1875, 0.3125),
            (0.6875, 0.8125),
            (0.3750, 0.1250),
            (0.5000, 0.9375),
            (0.2500, 0.8750),
            (0.1250, 0.2500),
            (0.0000, 0.5000),
            (0.9375, 0.7500),
            (0.8750, 0.0625),
            (0.0625, 0.0000),
        ),
        32: tuple(
            ((x + 0.5) / 8.0, (y + 0.5) / 4.0) for y in range(4) for x in range(8)
        ),
    }

    def __init__(self, conf):

        self.device = "cuda"
        self.conf = conf
        self.num_update_bvh = 0
        self._warned_spp_fallback = False
        self._logged_spp_configs = set()
        self.pred_pbr_filter = Filter(self.conf.render.get("filter_type", "none"))

        logger.info(
            f'🔆 Creating threedgptir Optix tracing pipeline.. Using CUDA path: "{torch.utils.cpp_extension.CUDA_HOME}"'
        )
        torch.zeros(
            1, device=self.device
        )  # Create a dummy tensor to force cuda context init
        load_threedgptir_plugin(conf)

        self.tracer_wrapper = _threedgptir_plugin.OptixTracer(
            os.path.dirname(__file__),
            torch.utils.cpp_extension.CUDA_HOME,
            self.conf.render.pipeline_type,
            self.conf.render.backward_pipeline_type,
            self.conf.render.primitive_type,
            self.conf.render.particle_kernel_degree,
            self.conf.render.particle_kernel_min_response,
            self.conf.render.particle_kernel_density_clamping,
            self.conf.render.particle_radiance_sph_degree,
            self.conf.render.enable_normals,
            self.conf.render.enable_hitcounts,
            self.conf.render.enable_mis,
            self.conf.render.get(
                "enable_metallic",
                self.conf.model.get("optimize_material_metallic", False),
            ),
            self.conf.render.visualize_environment,
        )

        self.frame_timer = (
            CudaTimer() if self.conf.render.enable_kernel_timings else None
        )
        self.timings = {}

    def _get_spp(self, train: bool) -> int:
        if train:
            spp = self.conf.render.inversion_spp
        else:
            spp = self.conf.render.render_spp
        return max(1, int(spp))

    def _get_spp_chunk(self, spp: int) -> int:
        return min(spp, max(1, int(self.conf.render.get("spp_chunk", spp))))

    def _make_spp_jitter(
        self,
        spp: int,
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype,
        frame_id: int,
    ):
        if spp in self._MULTISPP_PATTERNS:
            return torch.tensor(
                self._MULTISPP_PATTERNS[spp], dtype=dtype, device=device
            ).view(spp, 1, 1, 2)

        grid_w = int(math.ceil(math.sqrt(spp)))
        grid_h = int(math.ceil(spp / grid_w))
        sample_idx = torch.arange(spp, dtype=dtype, device=device)
        jitter = torch.stack(
            (
                (torch.remainder(sample_idx, grid_w) + 0.5) / grid_w,
                (torch.div(sample_idx, grid_w, rounding_mode="floor") + 0.5) / grid_h,
            ),
            dim=-1,
        )

        generator = torch.Generator(device=device)
        generator.manual_seed((int(frame_id) + 1) * 1315423911 + spp * 2654435761)
        shift = torch.rand((1, 2), dtype=dtype, device=device, generator=generator)
        return torch.remainder(jitter + shift, 1.0).view(spp, 1, 1, 2)

    def _expand_rays_for_spp(
        self,
        gpu_batch: Batch,
        spp: int,
        frame_id: int,
        jitter: torch.Tensor | None = None,
    ):
        rays_ori = gpu_batch.rays_ori.contiguous()
        rays_dir = gpu_batch.rays_dir.contiguous()
        if spp == 1 and jitter is None:
            return rays_ori, rays_dir, 1

        intrinsics = getattr(gpu_batch, "intrinsics", None)
        pixel_coords = getattr(gpu_batch, "pixel_coords", None)
        if intrinsics is None or pixel_coords is None:
            if not self._warned_spp_fallback:
                rich_logger.warning(
                    "PTIR SPP requested but batch has no pinhole intrinsics/pixel_coords; using spp=1."
                )
                self._warned_spp_fallback = True
            return rays_ori, rays_dir, 1

        base_batch, h, w, _ = rays_dir.shape
        fx, fy, cx, cy = [
            torch.as_tensor(v, dtype=rays_dir.dtype, device=rays_dir.device)
            for v in intrinsics
        ]
        pixel_origin = (
            pixel_coords.to(dtype=rays_dir.dtype, device=rays_dir.device) - 0.5
        )
        if jitter is None:
            jitter = self._make_spp_jitter(
                spp, h, w, rays_dir.device, rays_dir.dtype, frame_id
            )

        pixel_origin = pixel_origin.unsqueeze(0)
        jitter = (
            jitter.view(spp, 1, 1, 1, 2)
            if jitter.shape[1:3] == (1, 1)
            else jitter.view(spp, 1, h, w, 2)
        )
        dirs = torch.stack(
            (
                (pixel_origin[..., 0] + jitter[..., 0] - cx) / fx,
                (pixel_origin[..., 1] + jitter[..., 1] - cy) / fy,
                torch.ones(
                    (spp, base_batch, h, w),
                    dtype=rays_dir.dtype,
                    device=rays_dir.device,
                ),
            ),
            dim=-1,
        )
        dirs = (
            torch.nn.functional.normalize(dirs, dim=-1)
            .reshape(spp * base_batch, h, w, 3)
            .contiguous()
        )
        origins = (
            rays_ori.unsqueeze(0)
            .expand(spp, *rays_ori.shape)
            .reshape(spp * base_batch, h, w, 3)
            .contiguous()
        )
        return origins, dirs, spp

    @staticmethod
    def _average_spp_output(
        value: torch.Tensor, spp: int, base_batch: int
    ) -> torch.Tensor:
        if spp == 1:
            return value
        return value.reshape(spp, base_batch, *value.shape[1:]).mean(dim=0)

    def build_acc(self, gaussians, rebuild=True):
        with torch.cuda.nvtx.range(f"build-bvh-full-build-{rebuild}"):
            allow_bvh_update = (
                self.conf.render.max_consecutive_bvh_update > 1
            ) and not self.conf.render.particle_kernel_density_clamping
            rebuild_bvh = (
                rebuild
                or self.conf.render.particle_kernel_density_clamping
                or self.num_update_bvh >= self.conf.render.max_consecutive_bvh_update
            )
            self.tracer_wrapper.build_bvh(
                gaussians.positions.view(-1, 3).contiguous(),
                gaussians.rotation_activation(gaussians.rotation)
                .view(-1, 4)
                .contiguous(),
                gaussians.scale_activation(gaussians.scale).view(-1, 3).contiguous(),
                gaussians.density_activation(gaussians.density)
                .view(-1, 1)
                .contiguous(),
                rebuild_bvh,
                allow_bvh_update,
            )
            self.num_update_bvh = 0 if rebuild_bvh else self.num_update_bvh + 1

    def render(
        self,
        gaussians,
        gpu_batch: Batch,
        train=False,
        frame_id=0,
        sh_indirect: bool = False,
    ):
        num_gaussians = gaussians.num_gaussians
        with torch.cuda.nvtx.range(f"model.forward({num_gaussians} gaussians)"):
            if self.frame_timer is not None:
                self.frame_timer.start()

            base_batch = gpu_batch.rays_dir.shape[0]
            spp = self._get_spp(train)
            if spp > 1 and (
                getattr(gpu_batch, "intrinsics", None) is None
                or getattr(gpu_batch, "pixel_coords", None) is None
            ):
                if not self._warned_spp_fallback:
                    rich_logger.warning(
                        "PTIR SPP requested but batch has no pinhole intrinsics/pixel_coords; using spp=1."
                    )
                    self._warned_spp_fallback = True
                spp = 1

            spp_chunk = self._get_spp_chunk(spp)
            h, w = gpu_batch.rays_dir.shape[1:3]
            spp_jitter = None
            if spp > 1:
                spp_jitter = self._make_spp_jitter(
                    spp,
                    h,
                    w,
                    gpu_batch.rays_dir.device,
                    gpu_batch.rays_dir.dtype,
                    frame_id,
                )
            spp_config = (
                bool(train),
                int(spp),
                int(spp_chunk),
                int(base_batch),
                int(h),
                int(w),
            )
            if spp_config not in self._logged_spp_configs:
                num_chunks = (spp + spp_chunk - 1) // spp_chunk
                rich_logger.warning(
                    f"PTIR effective SPP: train={train} spp={spp} spp_chunk={spp_chunk} "
                    f"chunks={num_chunks} base_batch={base_batch} chunk_ray_batch<={spp_chunk * base_batch} "
                    f"resolution={w}x{h}"
                )
                self._logged_spp_configs.add(spp_config)

            accumulated_outputs = None
            mog_visibility = None
            total_spp = 0
            for spp_start in range(0, spp, spp_chunk):
                chunk_spp = min(spp_chunk, spp - spp_start)
                chunk_jitter = (
                    None
                    if spp_jitter is None
                    else spp_jitter[spp_start : spp_start + chunk_spp]
                )
                rays_ori, rays_dir, chunk_spp = self._expand_rays_for_spp(
                    gpu_batch,
                    chunk_spp,
                    frame_id + spp_start,
                    jitter=chunk_jitter,
                )
                environment = gaussians.get_environment()
                alias_table = getattr(gaussians, "environment_alias_table", None)
                if alias_table is None:
                    environment_alias_table = torch.empty(
                        0, dtype=torch.float32, device=rays_dir.device
                    )
                else:
                    environment_alias_table = (
                        torch.concat(
                            [
                                alias_table.prob.reshape(
                                    1, alias_table.height, alias_table.width
                                ),
                                alias_table.alias.reshape(
                                    1, alias_table.height, alias_table.width
                                ).to(dtype=torch.float32),
                                alias_table.pdf.reshape(
                                    1, alias_table.height, alias_table.width
                                ),
                            ],
                            dim=0,
                        )
                        .to(device=rays_dir.device)
                        .contiguous()
                    )

                (
                    chunk_pred_rgb,
                    chunk_pred_opacity,
                    chunk_pred_dist,
                    chunk_pred_dist_second_moment,
                    chunk_pred_distortion,
                    chunk_pred_normals,
                    chunk_pred_shadingnormal,
                    chunk_pred_material,
                    chunk_hits_count,
                    chunk_mog_visibility,
                    chunk_pred_pbr,
                    chunk_pred_light,
                    chunk_pbr_components,
                ) = Tracer._Autograd.apply(
                    self.tracer_wrapper,
                    frame_id + spp_start,
                    gpu_batch.T_to_world.contiguous(),
                    rays_ori,
                    rays_dir,
                    gaussians.positions.contiguous(),
                    gaussians.get_rotation().contiguous(),
                    gaussians.get_scale().contiguous(),
                    gaussians.get_density().contiguous(),
                    gaussians.get_features().contiguous(),
                    gaussians.get_shading_normal().contiguous(),
                    gaussians.get_material_albedo().contiguous(),
                    gaussians.get_material_roughness().contiguous(),
                    gaussians.get_material_metallic().contiguous(),
                    environment,
                    environment_alias_table,
                    int(
                        Tracer.RenderOpts.INDIRECT
                        if sh_indirect
                        else Tracer.RenderOpts.DEFAULT
                    ),
                    gaussians.n_active_features,
                    self.conf.render.min_transmittance,
                    self.conf.render.get("max_bounces", 3),
                )

                chunk_pred_rgb, chunk_pred_opacity = gaussians.background(
                    gpu_batch.T_to_world.contiguous(),
                    rays_dir,
                    chunk_pred_rgb,
                    chunk_pred_opacity,
                    train,
                )

                chunk_outputs = (
                    self._average_spp_output(chunk_pred_rgb, chunk_spp, base_batch),
                    self._average_spp_output(chunk_pred_opacity, chunk_spp, base_batch),
                    self._average_spp_output(chunk_pred_dist, chunk_spp, base_batch),
                    self._average_spp_output(
                        chunk_pred_dist_second_moment, chunk_spp, base_batch
                    ),
                    self._average_spp_output(
                        chunk_pred_distortion, chunk_spp, base_batch
                    ),
                    self._average_spp_output(chunk_pred_normals, chunk_spp, base_batch),
                    self._average_spp_output(
                        chunk_pred_shadingnormal, chunk_spp, base_batch
                    ),
                    self._average_spp_output(
                        chunk_pred_material, chunk_spp, base_batch
                    ),
                    self._average_spp_output(chunk_hits_count, chunk_spp, base_batch),
                    self._average_spp_output(chunk_pred_pbr, chunk_spp, base_batch),
                    self._average_spp_output(chunk_pred_light, chunk_spp, base_batch),
                    self._average_spp_output(
                        chunk_pbr_components.detach(), chunk_spp, base_batch
                    ),
                )
                weighted_chunk_outputs = tuple(
                    output * chunk_spp for output in chunk_outputs
                )
                if accumulated_outputs is None:
                    accumulated_outputs = weighted_chunk_outputs
                else:
                    accumulated_outputs = tuple(
                        accumulated + weighted
                        for accumulated, weighted in zip(
                            accumulated_outputs, weighted_chunk_outputs
                        )
                    )

                mog_visibility = (
                    chunk_mog_visibility
                    if mog_visibility is None
                    else torch.maximum(mog_visibility, chunk_mog_visibility)
                )
                total_spp += chunk_spp

            if self.frame_timer is not None:
                self.frame_timer.end()

            (
                pred_rgb,
                pred_opacity,
                pred_dist,
                pred_dist_second_moment,
                pred_distortion,
                pred_normals,
                pred_shadingnormal,
                pred_material,
                hits_count,
                pred_pbr,
                pred_light,
                pbr_components,
            ) = tuple(accumulated / total_spp for accumulated in accumulated_outputs)
            pred_direct = pbr_components[..., 0, :]
            pred_indirect = pbr_components[..., 1, :]

            pred_dist = pred_dist / pred_opacity
            pred_dist = torch.nan_to_num(pred_dist, 0.0, 0.0)

            pred_pbr = self.pred_pbr_filter(pred_pbr)
            pred_light = self.pred_pbr_filter(pred_light)
            pred_direct = self.pred_pbr_filter(pred_direct)
            pred_indirect = self.pred_pbr_filter(pred_indirect)

        if self.frame_timer is not None:
            self.timings["forward_render"] = self.frame_timer.timing()

        outputs = {
            "pred_rgb": pred_rgb,
            "pred_opacity": pred_opacity,
            "pred_dist": pred_dist,
            "pred_depth_second_moment": pred_dist_second_moment,
            "pred_depth_distortion": pred_distortion,
            "pred_normals": torch.nn.functional.normalize(pred_normals, dim=3),
            "pred_shadingnormal": pred_shadingnormal,
            "pred_material": pred_material,
            "pred_pbr": pred_pbr,
            "pred_light": pred_light,
            "pred_direct": pred_direct,
            "pred_indirect": pred_indirect,
            "hits_count": hits_count,
            "frame_time_ms": self.frame_timer.timing()
            if self.frame_timer is not None
            else 0.0,
            "mog_visibility": mog_visibility,
        }
        return post_processing(outputs, gpu_batch, self.conf.render.visualize_environment)
