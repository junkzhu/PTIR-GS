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
from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.utils.cpp_extension
from omegaconf import OmegaConf

from threedgrut.datasets.protocols import Batch
from threedgrut.model.filters import Filter
from threedgrut.model.ptir_helper import post_processing
from threedgrut.utils.logger import logger as rich_logger
from threedgrut.utils.timer import CudaTimer

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
#

_threedgptir_plugin = None


@dataclass(frozen=True)
class LightBuffers:
    lights: torch.Tensor
    alias_table: torch.Tensor
    mesh_vertices: torch.Tensor
    mesh_triangles: torch.Tensor
    mesh_lights: torch.Tensor
    mesh_triangle_alias_table: torch.Tensor

    @classmethod
    def empty(cls, device: torch.device | str) -> "LightBuffers":
        return cls(
            lights=torch.empty((0, 9), dtype=torch.float32, device=device),
            alias_table=torch.empty((5, 0), dtype=torch.float32, device=device),
            mesh_vertices=torch.empty((0, 3), dtype=torch.float32, device=device),
            mesh_triangles=torch.empty((0, 3), dtype=torch.int32, device=device),
            mesh_lights=torch.empty((0, 14), dtype=torch.float32, device=device),
            mesh_triangle_alias_table=torch.empty(
                (3, 0), dtype=torch.float32, device=device
            ),
        )

    @classmethod
    def from_batch(cls, batch: Batch, device: torch.device | str) -> "LightBuffers":
        empty = cls.empty(device)

        def value_or_empty(name: str, fallback: torch.Tensor) -> torch.Tensor:
            value = getattr(batch, name, None)
            return fallback if value is None else value

        return cls(
            lights=value_or_empty("lights", empty.lights),
            alias_table=value_or_empty("light_alias_table", empty.alias_table),
            mesh_vertices=value_or_empty("mesh_light_vertices", empty.mesh_vertices),
            mesh_triangles=value_or_empty("mesh_light_triangles", empty.mesh_triangles),
            mesh_lights=value_or_empty("mesh_lights", empty.mesh_lights),
            mesh_triangle_alias_table=value_or_empty(
                "mesh_light_triangle_alias_table",
                empty.mesh_triangle_alias_table,
            ),
        ).to(device)

    def to(self, device: torch.device | str) -> "LightBuffers":
        return LightBuffers(
            lights=self.lights.to(device=device, dtype=torch.float32).contiguous(),
            alias_table=self.alias_table.to(
                device=device, dtype=torch.float32
            ).contiguous(),
            mesh_vertices=self.mesh_vertices.to(
                device=device, dtype=torch.float32
            ).contiguous(),
            mesh_triangles=self.mesh_triangles.to(
                device=device, dtype=torch.int32
            ).contiguous(),
            mesh_lights=self.mesh_lights.to(
                device=device, dtype=torch.float32
            ).contiguous(),
            mesh_triangle_alias_table=self.mesh_triangle_alias_table.to(
                device=device, dtype=torch.float32
            ).contiguous(),
        )

    def tensors(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        return (
            self.lights,
            self.alias_table,
            self.mesh_vertices,
            self.mesh_triangles,
            self.mesh_lights,
            self.mesh_triangle_alias_table,
        )


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
            sg_environment,
            sg_sampling_distribution,
            light_buffers,
            render_opts,
            sph_degree,
            min_transmittance,
            max_bounces,
            enable_secondary_nee,
        ):
            particle_density = torch.concat(
                [mog_pos, mog_dns, mog_rot, mog_scl, torch.zeros_like(mog_dns)], dim=1
            )
            particle_material = torch.concat([mog_malb, mog_mrgh, mog_mmet], dim=1)
            light_buffers = light_buffers.to(ray_dir.device)
            (
                lights,
                light_alias_table,
                mesh_light_vertices,
                mesh_light_triangles,
                mesh_lights,
                mesh_light_triangle_alias_table,
            ) = light_buffers.tensors()
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
                sg_environment,
                sg_sampling_distribution,
                lights,
                light_alias_table,
                mesh_light_vertices,
                mesh_light_triangles,
                mesh_lights,
                mesh_light_triangle_alias_table,
                render_opts,
                sph_degree,
                min_transmittance,
                max_bounces,
                int(enable_secondary_nee),
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
                sg_environment,
                sg_sampling_distribution,
                lights,
                light_alias_table,
                mesh_light_vertices,
                mesh_light_triangles,
                mesh_lights,
                mesh_light_triangle_alias_table,
            )
            ctx.frame_id = frame_id
            ctx.render_opts = render_opts
            ctx.sph_degree = sph_degree
            ctx.min_transmittance = min_transmittance
            ctx.max_bounces = max_bounces
            ctx.enable_secondary_nee = bool(enable_secondary_nee)
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
                sg_environment,
                sg_sampling_distribution,
                lights,
                light_alias_table,
                mesh_light_vertices,
                mesh_light_triangles,
                mesh_lights,
                mesh_light_triangle_alias_table,
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
                sg_environment_grd,
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
                sg_environment,
                sg_sampling_distribution,
                lights,
                light_alias_table,
                mesh_light_vertices,
                mesh_light_triangles,
                mesh_lights,
                mesh_light_triangle_alias_table,
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
                int(ctx.enable_secondary_nee),
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
                sg_environment_grd,
                None,
                None,
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
        if OmegaConf.select(
            self.conf, "render.enable_secondary_nee", default=None
        ) is None:
            OmegaConf.update(
                self.conf,
                "render.enable_secondary_nee",
                True,
                force_add=True,
            )
        self.num_update_bvh = 0
        self._scene_mesh_active = False
        self._warned_spp_fallback = False
        self._logged_spp_configs = set()
        self.pred_pbr_filter = Filter(self.conf.render.get("filter_type", "none"))
        self.visualize_lights = bool(self.conf.render.get("visualize_lights", False))
        self.max_self_occlusion_offset = float(
            self.conf.render.get("max_self_occlusion_offset", 0.1)
        )
        if (
            not math.isfinite(self.max_self_occlusion_offset)
            or self.max_self_occlusion_offset < 0.0
        ):
            raise ValueError(
                "render.max_self_occlusion_offset must be a finite non-negative value, "
                f"got {self.max_self_occlusion_offset}"
            )
        self.max_bounces = int(self.conf.render.max_bounces)
        if self.max_bounces < 0:
            raise ValueError(
                f"render.max_bounces must be >= 0, got {self.max_bounces}"
            )
        self.render_bounces = int(
            self.conf.render.get("render_bounces", self.max_bounces)
        )
        if self.render_bounces < 0:
            raise ValueError(
                "render.render_bounces must be >= 0, "
                f"got {self.render_bounces}"
            )
        # The native tracer counts the primary ray as the first path segment,
        # while the configured bounce limits count only reflection/path bounces.
        self.environment_type = str(
            self.conf.get("environment", {}).get("type", "2d")
        ).lower()

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
            self.max_self_occlusion_offset,
            self.conf.render.particle_radiance_sph_degree,
            self.conf.render.enable_normals,
            self.conf.render.enable_hitcounts,
            self.conf.render.enable_mis,
            self.conf.render.get(
                "enable_metallic",
                self.conf.model.get("optimize_material_metallic", False),
            ),
            self.visualize_lights,
            self.conf.render.russian_roulette,
            self.conf.render.discrete_model,
        )

        self.frame_timer = (
            CudaTimer() if self.conf.render.enable_kernel_timings else None
        )
        self.sg_distribution_timer = (
            CudaTimer() if self.conf.render.enable_kernel_timings else None
        )
        self.timings = {}

    def _uses_native_sg(self, gaussians) -> bool:
        environment = getattr(gaussians, "environment", None)
        return self.environment_type == "spherical_gaussian" and hasattr(
            environment, "native_parameters"
        )

    def _prepare_native_sg(self, gaussians, device: torch.device):
        environment = getattr(gaussians, "environment", None)
        if not self._uses_native_sg(gaussians):
            return (
                torch.empty((0, 7), dtype=torch.float32, device=device),
                torch.empty((0, 2), dtype=torch.float32, device=device),
            )

        parameters = environment.native_parameters().to(
            device=device, dtype=torch.float32
        ).contiguous()
        if self.sg_distribution_timer is not None:
            self.sg_distribution_timer.start()
        sampling_distribution = environment.native_sampling_distribution(parameters)
        if self.sg_distribution_timer is not None:
            self.sg_distribution_timer.end()
        return parameters, sampling_distribution

    def _required_train_output_keys(self) -> set[str]:
        keys = {
            "pred_pbr",
            "pred_material",
        }

        lambda_light = float(self.conf.loss.get("lambda_light", 0.0))
        if (
            bool(self.conf.loss.get("use_light_consistency", False))
            or lambda_light > 0.0
        ):
            keys.add("pred_light")

        if bool(self.conf.loss.get("use_edge_aware_smoothness", False)):
            keys.update(self.conf.loss.get("edge_aware_smoothness_outputs", []))

        return keys

    def _train_output_key_sets(
        self, train: bool
    ) -> tuple[set[str] | None, set[str] | None]:
        if not train or not bool(
            self.conf.render.get("hide_intermediate_outputs", False)
        ):
            return None, None

        final_output_keys = self._required_train_output_keys()
        accumulation_keys = set(final_output_keys)
        if "pred_pbr" in accumulation_keys:
            accumulation_keys.add("pred_opacity")
        if "pred_direct" in accumulation_keys or "pred_indirect" in accumulation_keys:
            accumulation_keys.update(("pbr_components", "pred_opacity"))
        return final_output_keys, accumulation_keys

    def _postprocess_pbr_outputs(self, outputs: dict[str, torch.Tensor]) -> None:
        pred_opacity = outputs.get("pred_opacity")
        opacity_weighted_keys = ("pred_pbr", "pred_direct", "pred_indirect")
        if pred_opacity is not None:
            for key in opacity_weighted_keys:
                value = outputs.get(key)
                if value is not None:
                    outputs[key] = value * pred_opacity

        for key in (*opacity_weighted_keys, "pred_light"):
            value = outputs.get(key)
            if value is not None:
                outputs[key] = self.pred_pbr_filter(value)

    @staticmethod
    def _normalize_depth_output(outputs: dict[str, torch.Tensor]) -> None:
        pred_dist = outputs.get("pred_dist")
        pred_opacity = outputs.get("pred_opacity")
        if pred_dist is not None and pred_opacity is not None:
            outputs["pred_dist"] = torch.nan_to_num(
                pred_dist / pred_opacity, 0.0, 0.0
            )

    def _get_spp(self, train: bool) -> int:
        if train:
            spp = self.conf.render.inversion_spp
        else:
            spp = self.conf.render.render_spp
        return max(1, int(spp))

    def _get_native_max_bounces(self, train: bool) -> int:
        max_bounces = self.max_bounces if train else self.render_bounces
        return max_bounces + 1

    def _get_spp_chunk(self, spp: int) -> int:
        return min(spp, max(1, int(self.conf.render.get("spp_chunk", spp))))

    def _warn_spp_fallback(self, reason: str):
        if not self._warned_spp_fallback:
            rich_logger.warning(f"PTIR SPP fallback to spp=1: {reason}.")
            self._warned_spp_fallback = True

    @staticmethod
    def _flat(value, dtype: torch.dtype, device: torch.device):
        if value is None:
            return None
        return torch.as_tensor(value, dtype=dtype, device=device).flatten()

    def _extract_pinhole_intrinsics(
        self,
        gpu_batch: Batch,
        dtype: torch.dtype,
        device: torch.device,
        warn: bool = True,
    ):
        # Case 1: direct intrinsics, expected as [fx, fy, cx, cy] or 3x3 K.
        intrinsics = self._flat(getattr(gpu_batch, "intrinsics", None), dtype, device)
        if intrinsics is not None:
            if intrinsics.numel() == 4:
                return tuple(intrinsics)
            if intrinsics.numel() == 9:
                K = intrinsics.reshape(3, 3)
                return K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            if warn:
                self._warn_spp_fallback("unsupported intrinsics format")
            return None

        # Case 2: nerfstudio OpenCV pinhole camera parameters.
        params = getattr(
            gpu_batch, "intrinsics_OpenCVPinholeCameraModelParameters", None
        )
        if params is None:
            if warn:
                self._warn_spp_fallback("missing pinhole intrinsics")
            return None

        def get(name):
            if isinstance(params, dict):
                return params.get(name)
            return getattr(params, name, None)

        # SPP ray expansion assumes distortion-free pinhole rays.
        for name in ("radial_coeffs", "tangential_coeffs", "thin_prism_coeffs"):
            coeffs = self._flat(get(name), dtype, device)
            if coeffs is not None and coeffs.numel() > 0:
                if torch.any(torch.abs(coeffs) > 1e-8).item():
                    if warn:
                        self._warn_spp_fallback("distorted camera model")
                    return None

        focal = self._flat(get("focal_length"), dtype, device)
        principal = self._flat(get("principal_point"), dtype, device)

        if focal is None or principal is None or focal.numel() < 1 or principal.numel() < 2:
            if warn:
                self._warn_spp_fallback("incomplete pinhole intrinsics")
            return None

        fx = focal[0]
        fy = focal[1] if focal.numel() > 1 else focal[0]
        cx, cy = principal[:2]
        return fx, fy, cx, cy

    def _can_expand_spp(self, gpu_batch: Batch) -> bool:
        return (
            getattr(gpu_batch, "pixel_coords", None) is not None
            and self._extract_pinhole_intrinsics(
                gpu_batch,
                gpu_batch.rays_dir.dtype,
                gpu_batch.rays_dir.device,
                warn=False,
            )
            is not None
        )

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

        pixel_coords = getattr(gpu_batch, "pixel_coords", None)
        intrinsics = self._extract_pinhole_intrinsics(
            gpu_batch, rays_dir.dtype, rays_dir.device
        )
        if intrinsics is None or pixel_coords is None:
            if pixel_coords is None:
                self._warn_spp_fallback("batch has no pixel_coords")
            return rays_ori, rays_dir, 1

        base_batch, h, w, _ = rays_dir.shape
        fx, fy, cx, cy = intrinsics
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
            native_max_bounces = self._get_native_max_bounces(train)
            if spp > 1 and not self._can_expand_spp(gpu_batch):
                self._warn_spp_fallback(
                    "batch has no supported pinhole intrinsics/pixel_coords"
                )
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

            final_output_keys, accumulation_keys = self._train_output_key_sets(train)

            accumulated_outputs = {}
            mog_visibility = None
            total_spp = 0

            native_sg_enabled = self._uses_native_sg(gaussians)
            self.timings.pop("sg_sampling_distribution_update", None)
            sg_environment, sg_sampling_distribution = self._prepare_native_sg(
                gaussians, gpu_batch.rays_dir.device
            )
            if native_sg_enabled:
                # Native rendering must not rasterize SGs into an envmap.
                environment = torch.empty(
                    (0, 0, 4), dtype=torch.float32, device=gpu_batch.rays_dir.device
                )
                environment_alias_table = torch.empty(
                    0, dtype=torch.float32, device=gpu_batch.rays_dir.device
                )
            else:
                environment = gaussians.get_environment()
                if environment is None:
                    environment = torch.empty(
                        (0, 0, 4),
                        dtype=torch.float32,
                        device=gpu_batch.rays_dir.device,
                    )
                alias_table = getattr(gaussians, "environment_alias_table", None)
                if alias_table is None:
                    environment_alias_table = torch.empty(
                        0, dtype=torch.float32, device=gpu_batch.rays_dir.device
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
                        .to(device=gpu_batch.rays_dir.device)
                        .contiguous()
                    )
            light_buffers = LightBuffers.from_batch(
                gpu_batch,
                device=gpu_batch.rays_dir.device,
            )
            has_scene_mesh = light_buffers.mesh_triangles.numel() > 0
            if has_scene_mesh:
                self.tracer_wrapper.build_scene_mesh_bvh(
                    light_buffers.mesh_vertices,
                    light_buffers.mesh_triangles,
                )
                self._scene_mesh_active = True
            elif self._scene_mesh_active:
                self.tracer_wrapper.build_scene_mesh_bvh(
                    light_buffers.mesh_vertices,
                    light_buffers.mesh_triangles,
                )
                self._scene_mesh_active = False
            enable_secondary_nee = self.conf.render.enable_secondary_nee

            def accumulate_output(name: str, value: torch.Tensor) -> None:
                if accumulation_keys is not None and name not in accumulation_keys:
                    return
                averaged = self._average_spp_output(value, chunk_spp, base_batch)
                weighted = averaged * chunk_spp
                previous = accumulated_outputs.get(name)
                accumulated_outputs[name] = (
                    weighted if previous is None else previous + weighted
                )

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
                    sg_environment,
                    sg_sampling_distribution,
                    light_buffers,
                    int(
                        Tracer.RenderOpts.INDIRECT
                        if sh_indirect
                        else Tracer.RenderOpts.DEFAULT
                    ),
                    gaussians.n_active_features,
                    self.conf.render.min_transmittance,
                    native_max_bounces,
                    enable_secondary_nee,
                )

                chunk_pred_rgb, chunk_pred_opacity = gaussians.background(
                    gpu_batch.T_to_world.contiguous(),
                    rays_dir,
                    chunk_pred_rgb,
                    chunk_pred_opacity,
                    train,
                )

                for name, value in (
                    ("pred_rgb", chunk_pred_rgb),
                    ("pred_opacity", chunk_pred_opacity),
                    ("pred_dist", chunk_pred_dist),
                    ("pred_depth_second_moment", chunk_pred_dist_second_moment),
                    ("pred_depth_distortion", chunk_pred_distortion),
                    ("pred_normals", chunk_pred_normals),
                    ("pred_shadingnormal", chunk_pred_shadingnormal),
                    ("pred_material", chunk_pred_material),
                    ("hits_count", chunk_hits_count),
                    ("pred_pbr", chunk_pred_pbr),
                    ("pred_light", chunk_pred_light),
                    ("pbr_components", chunk_pbr_components.detach()),
                ):
                    accumulate_output(name, value)

                mog_visibility = (
                    chunk_mog_visibility
                    if mog_visibility is None
                    else torch.maximum(mog_visibility, chunk_mog_visibility)
                )
                total_spp += chunk_spp

            if self.frame_timer is not None:
                self.frame_timer.end()

            accumulated_outputs = {
                name: accumulated / total_spp
                for name, accumulated in accumulated_outputs.items()
            }
            pbr_components = accumulated_outputs.pop("pbr_components", None)
            if pbr_components is not None:
                accumulated_outputs["pred_direct"] = pbr_components[..., 0, :]
                accumulated_outputs["pred_indirect"] = pbr_components[..., 1, :]

            self._normalize_depth_output(accumulated_outputs)
            # https://github.com/fudan-zvg/IRGS/blob/main/gaussian_renderer/__init__.py#L233
            self._postprocess_pbr_outputs(accumulated_outputs)

        if self.frame_timer is not None:
            self.timings["forward_render"] = self.frame_timer.timing()
        if native_sg_enabled and self.sg_distribution_timer is not None:
            self.timings["sg_sampling_distribution_update"] = (
                self.sg_distribution_timer.timing()
            )

        output_keys = (
            "pred_rgb",
            "pred_opacity",
            "pred_dist",
            "pred_depth_second_moment",
            "pred_depth_distortion",
            "pred_normals",
            "pred_shadingnormal",
            "pred_material",
            "pred_pbr",
            "pred_light",
            "pred_direct",
            "pred_indirect",
            "hits_count",
        )
        output_values = {key: accumulated_outputs.get(key) for key in output_keys}
        if output_values["pred_normals"] is not None:
            output_values["pred_normals"] = torch.nn.functional.normalize(
                output_values["pred_normals"], dim=3
            )
        outputs = {
            key: value
            for key, value in output_values.items()
            if value is not None
            and (final_output_keys is None or key in final_output_keys)
        }
        outputs["frame_time_ms"] = (
            self.frame_timer.timing() if self.frame_timer is not None else 0.0
        )
        outputs["mog_visibility"] = mog_visibility
        return post_processing(outputs, gpu_batch, self.visualize_lights)
