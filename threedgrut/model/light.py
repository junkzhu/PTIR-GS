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

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional

import imageio
import imageio.plugins.freeimage as fi
import numpy as np
import torch
import torch.nn.functional as F

from threedgrut.model.aliastable import (
    DEFAULT_ALIAS_TABLE_SIZE,
    ENVIRONMENT_TYPE_CUBE,
    ENVIRONMENT_TYPE_SPHERICAL_GAUSSIAN,
    EnvAliasTable,
    build_alias_table,
    build_environment_alias_table,
    environment_importance_weights,
)


TensorLike = torch.Tensor | Sequence[float]
LIGHT_TYPE_ENV = 0
LIGHT_TYPE_POINT = 1
LIGHT_TYPE_SPHERE = 2
LIGHT_TYPE_MESH = 3
PACKED_LIGHT_SIZE = 9
PACKED_MESH_LIGHT_SIZE = 14
_DEFAULT_ENVIRONMENT_SIZE = (64, 128)


@dataclass(frozen=True)
class LightAliasTable:
    prob: torch.Tensor
    alias: torch.Tensor
    light_type: torch.Tensor
    light_index: torch.Tensor
    light_select_pdf: torch.Tensor

    @property
    def num_entries(self) -> int:
        return int(self.prob.numel())

    def pack(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if self.num_entries == 0:
            return torch.empty((5, 0), dtype=dtype, device=device)
        return torch.stack(
            [
                self.prob.to(device=device, dtype=dtype).reshape(-1),
                self.alias.to(device=device, dtype=dtype).reshape(-1),
                self.light_type.to(device=device, dtype=dtype).reshape(-1),
                self.light_index.to(device=device, dtype=dtype).reshape(-1),
                self.light_select_pdf.to(device=device, dtype=dtype).reshape(-1),
            ],
            dim=0,
        ).contiguous()


@dataclass(frozen=True)
class MeshLightTriangleAliasTable:
    """Packed triangle sampling data for a mesh light.

    Triangle alias entries are packed across all mesh lights. Each mesh stores a
    triangle offset/count into the packed arrays.
    """

    prob: torch.Tensor
    alias: torch.Tensor
    triangle_pdf: torch.Tensor
    total_area: torch.Tensor

    @property
    def num_entries(self) -> int:
        return int(self.prob.numel())

    def pack(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if self.num_entries == 0:
            return torch.empty((3, 0), dtype=dtype, device=device)
        return torch.stack(
            [
                self.prob.to(device=device, dtype=dtype).reshape(-1),
                self.alias.to(device=device, dtype=dtype).reshape(-1),
                self.triangle_pdf.to(device=device, dtype=dtype).reshape(-1),
            ],
            dim=0,
        ).contiguous()


@dataclass(frozen=True)
class MeshLightPack:
    vertices: torch.Tensor
    triangles: torch.Tensor
    params: torch.Tensor
    triangle_alias_table: torch.Tensor
    powers: torch.Tensor

    @property
    def num_lights(self) -> int:
        return int(self.params.shape[0])


class LightType(IntEnum):
    ENVIRONMENT = LIGHT_TYPE_ENV
    POINT = LIGHT_TYPE_POINT
    SPHERE = LIGHT_TYPE_SPHERE
    MESH = LIGHT_TYPE_MESH

    Environment = ENVIRONMENT
    Sphere = SPHERE
    Mesh = MESH
    Point = POINT

    @classmethod
    def normalize(cls, value: "LightType | str | int") -> "LightType":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.lower().replace("-", "_")
            aliases = {
                "env": cls.ENVIRONMENT,
                "envmap": cls.ENVIRONMENT,
                "environment": cls.ENVIRONMENT,
                "environmentlight": cls.ENVIRONMENT,
                "environment_light": cls.ENVIRONMENT,
                "sphere": cls.SPHERE,
                "spherelight": cls.SPHERE,
                "sphere_light": cls.SPHERE,
                "mesh": cls.MESH,
                "meshlight": cls.MESH,
                "mesh_light": cls.MESH,
                "point": cls.POINT,
                "pointlight": cls.POINT,
                "point_light": cls.POINT,
            }
            if normalized in aliases:
                return aliases[normalized]
            raise ValueError(f"Unsupported light type '{value}'.")
        return cls(int(value))


class Light(ABC):
    """Base class for scene lights.

    Subclasses own their light-specific parameters and expose a compact tensor
    pack for renderer-side upload.
    """

    @property
    @abstractmethod
    def type(self) -> LightType:
        raise NotImplementedError("Light subclasses must expose their type.")

    @abstractmethod
    def pack(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        raise NotImplementedError("Light subclasses must implement pack().")

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError("Light subclasses must implement to_dict().")

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Light":
        light_type = LightType.normalize(data.get("type", "sphere"))
        if light_type == LightType.ENVIRONMENT:
            return EnvironmentLight(
                environment=data.get("environment"),
                environment_type=data.get("environment_type", "2d"),
                light_index=int(data.get("light_index", 0)),
            )
        if light_type == LightType.SPHERE:
            return SphereLight(
                center=data.get("center", (0.0, 0.0, 0.0)),
                radius=data.get("radius", 1.0),
                radiance=data.get("radiance", (1.0, 1.0, 1.0)),
                two_sided=bool(data.get("two_sided", False)),
            )
        if light_type == LightType.POINT:
            return PointLight(
                position=data.get("position", data.get("center", (0.0, 0.0, 0.0))),
                intensity=data.get(
                    "intensity", data.get("radiance", (1.0, 1.0, 1.0))
                ),
            )
        if light_type == LightType.MESH:
            return MeshLight(
                vertices=data.get("vertices", ()),
                triangles=data.get("triangles", data.get("faces", ())),
                radiance=data.get("radiance", (1.0, 1.0, 1.0)),
                two_sided=bool(data.get("two_sided", False)),
                albedo=data.get("albedo", (0.8, 0.8, 0.8)),
                roughness=data.get("roughness", 0.5),
                metallic=data.get("metallic", 0.0),
                surface=bool(data.get("surface", False)),
            )
        raise NotImplementedError(f"Light type '{light_type.name}' is not implemented yet.")

    @staticmethod
    def pack_lights(
        lights: torch.Tensor | Sequence["Light | dict[str, Any]"] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if lights is None:
            return torch.empty((0, PACKED_LIGHT_SIZE), dtype=dtype, device=device)

        if isinstance(lights, torch.Tensor):
            packed = lights.to(device=device, dtype=dtype)
            if packed.numel() == 0:
                return torch.empty((0, PACKED_LIGHT_SIZE), dtype=dtype, device=device)
            if packed.ndim != 2 or packed.shape[1] != PACKED_LIGHT_SIZE:
                raise ValueError(
                    f"packed lights must have shape [N, {PACKED_LIGHT_SIZE}], "
                    f"got {tuple(packed.shape)}."
                )
            return packed.contiguous()

        packed_lights = []
        for light in lights:
            if isinstance(light, dict):
                light = Light.from_dict(light)
            if not isinstance(light, Light):
                raise TypeError(
                    f"lights must be Light instances or dictionaries, got {type(light).__name__}."
                )
            if light.type not in (LightType.POINT, LightType.SPHERE):
                raise NotImplementedError(
                    f"Packed light upload only supports point and sphere lights, got {light.type.name}."
                )
            light_type = torch.tensor(
                [float(light.type)], dtype=dtype, device=device or light.device
            )
            packed_lights.append(
                torch.cat([light_type, light.pack(device=device, dtype=dtype)], dim=0)
            )

        if not packed_lights:
            return torch.empty((0, PACKED_LIGHT_SIZE), dtype=dtype, device=device)
        return torch.stack(packed_lights, dim=0).contiguous()


def _as_vec3(
    value: TensorLike,
    name: str,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device).reshape(-1)
    if tensor.numel() != 3:
        raise ValueError(f"{name} must contain exactly 3 values, got shape {tuple(torch.as_tensor(value).shape)}.")
    if not bool(torch.isfinite(tensor.detach()).all().item()):
        raise ValueError(f"{name} must contain finite values.")
    return tensor.contiguous()


def _as_scalar(
    value: torch.Tensor | float,
    name: str,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device).reshape(-1)
    if tensor.numel() != 1:
        raise ValueError(f"{name} must be a scalar, got shape {tuple(torch.as_tensor(value).shape)}.")
    if not bool(torch.isfinite(tensor.detach()).all().item()):
        raise ValueError(f"{name} must be finite.")
    return tensor.contiguous()


def _as_vertices(
    value: torch.Tensor | Sequence[Sequence[float]],
    name: str = "vertices",
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.ndim != 2 or tensor.shape[1] != 3:
        raise ValueError(f"{name} must have shape [V, 3], got {tuple(tensor.shape)}.")
    if not bool(torch.isfinite(tensor.detach()).all().item()):
        raise ValueError(f"{name} must contain finite values.")
    return tensor.contiguous()


def _as_triangles(
    value: torch.Tensor | Sequence[Sequence[int]],
    num_vertices: int,
    name: str = "triangles",
    device: torch.device | str | None = None,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.int64, device=device)
    if tensor.ndim != 2 or tensor.shape[1] != 3:
        raise ValueError(f"{name} must have shape [T, 3], got {tuple(tensor.shape)}.")
    if tensor.numel() == 0:
        return tensor.to(dtype=torch.int32).contiguous()
    if int(tensor.min().detach().cpu().item()) < 0:
        raise ValueError(f"{name} must not contain negative vertex indices.")
    if int(tensor.max().detach().cpu().item()) >= num_vertices:
        raise ValueError(
            f"{name} contains an index outside the vertex range [0, {num_vertices})."
        )
    return tensor.to(dtype=torch.int32).contiguous()


def _rgb_luminance(rgb: torch.Tensor) -> torch.Tensor:
    weights = rgb.new_tensor((0.2126, 0.7152, 0.0722))
    return torch.sum(rgb[..., :3] * weights, dim=-1)


def _triangle_areas(vertices: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    if triangles.numel() == 0:
        return torch.empty((0,), dtype=vertices.dtype, device=vertices.device)
    tri_vertices = vertices[triangles.to(dtype=torch.long)]
    edge0 = tri_vertices[:, 1] - tri_vertices[:, 0]
    edge1 = tri_vertices[:, 2] - tri_vertices[:, 0]
    areas = 0.5 * torch.linalg.cross(edge0, edge1, dim=-1).norm(dim=-1)
    return torch.where(
        torch.isfinite(areas) & (areas > 0.0),
        areas,
        torch.zeros_like(areas),
    )


def _as_environment_tensor(
    environment: torch.Tensor | None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor | None:
    if environment is None:
        return None
    tensor = torch.as_tensor(environment, dtype=dtype, device=device)
    if tensor.numel() == 0:
        return torch.empty((0, 0, 4), dtype=dtype, device=device)
    if tensor.ndim != 3 or tensor.shape[-1] != 4:
        raise ValueError(
            f"environment light must have shape [H, W, 4], got {tuple(tensor.shape)}."
        )
    return tensor.contiguous()


class EnvironmentLight(Light):
    """Environment map entry for the top-level light sampler.

    The Environment class in this module still owns image loading and
    alias-table construction. This wrapper represents that loaded texture as a
    Light entry alongside sphere and mesh lights.
    """

    def __init__(
        self,
        environment: torch.Tensor | None = None,
        environment_type: str = "2d",
        light_index: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.environment = _as_environment_tensor(
            environment, device=device, dtype=dtype
        )
        self.environment_type = str(environment_type).lower()
        self.light_index = int(light_index)

    @property
    def type(self) -> LightType:
        return LightType.ENVIRONMENT

    @property
    def device(self) -> torch.device | None:
        if self.environment is None:
            return None
        return self.environment.device

    @property
    def dtype(self) -> torch.dtype:
        if self.environment is None:
            return torch.float32
        return self.environment.dtype

    def pack(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        environment = _as_environment_tensor(
            self.environment, device=device, dtype=dtype
        )
        if environment is None:
            return torch.empty((0, 0, 4), dtype=dtype, device=device)
        return environment

    def estimate_power(self) -> torch.Tensor | None:
        if self.environment is None or self.environment.numel() == 0:
            return None

        weights = environment_importance_weights(
            self.environment,
            environment_type=self.environment_type,
            include_solid_angle=True,
        )
        if weights is None:
            return None
        power = torch.as_tensor(
            weights, dtype=torch.float32, device=weights.device
        ).sum()
        if (
            not bool(torch.isfinite(power).item())
            or float(power.detach().cpu().item()) <= 0.0
        ):
            return None
        return power

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "EnvironmentLight":
        dtype = self.dtype if dtype is None else dtype
        return EnvironmentLight(
            environment=self.environment,
            environment_type=self.environment_type,
            light_index=self.light_index,
            device=device,
            dtype=dtype,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "environment",
            "environment": None
            if self.environment is None
            else self.environment.detach().clone(),
            "environment_type": self.environment_type,
            "light_index": self.light_index,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.environment = _as_environment_tensor(
            state_dict.get("environment", self.environment),
            device=self.device,
            dtype=self.dtype,
        )
        self.environment_type = str(
            state_dict.get("environment_type", self.environment_type)
        ).lower()
        self.light_index = int(state_dict.get("light_index", self.light_index))

    def to_dict(self) -> dict[str, Any]:
        shape = None if self.environment is None else list(self.environment.shape)
        return {
            "type": "environment",
            "environment_type": self.environment_type,
            "light_index": self.light_index,
            "shape": shape,
        }


def environment_tensor_to_rgb_numpy(environment: torch.Tensor) -> np.ndarray:
    environment = environment.detach()
    if environment.ndim != 3 or environment.shape[-1] < 3:
        raise ValueError(
            f"Environment must have shape [H, W, C>=3], got {tuple(environment.shape)}"
        )

    rgb = environment[..., :3].detach().cpu().numpy()
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    return rgb.astype(np.float32, copy=False)


def save_environment_exr(
    environment: Optional[torch.Tensor], output_path: str
) -> Optional[str]:
    if environment is None:
        return None

    rgb = environment_tensor_to_rgb_numpy(environment)
    imageio.v2.imwrite(output_path, rgb)
    return output_path


def _softplus_inverse(value: torch.Tensor) -> torch.Tensor:
    tiny = torch.finfo(value.dtype).tiny
    return torch.log(torch.expm1(value).clamp_min(tiny))


def _radiance_to_sg_raw(radiance: torch.Tensor, gamma: float) -> torch.Tensor:
    scaled = float(gamma) * torch.log1p(torch.clamp(radiance, min=0.0))
    return _softplus_inverse(scaled)


def _sg_raw_to_radiance(raw: torch.Tensor, gamma: float) -> torch.Tensor:
    # 80 is below float32 overflow while leaving any practical HDR amplitude
    # unchanged. The clamp also stops invalid optimizer excursions cleanly.
    log_radiance = torch.clamp(F.softplus(raw) / float(gamma), max=80.0)
    return torch.nan_to_num(
        torch.expm1(log_radiance), nan=0.0, posinf=5.54062238439351e34
    )


def _fibonacci_sphere_directions(
    n_lobes: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if n_lobes <= 0:
        raise ValueError(f"n_lobes must be positive, got {n_lobes}.")

    indices = torch.arange(n_lobes, dtype=dtype, device=device)
    if n_lobes == 1:
        return torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype, device=device)

    y = 1.0 - indices / float(n_lobes - 1) * 2.0
    radius = torch.sqrt(torch.clamp(1.0 - y * y, min=0.0))
    golden_angle = torch.pi * (
        3.0 - torch.sqrt(torch.as_tensor(5.0, dtype=dtype, device=device))
    )
    theta = indices * golden_angle
    return torch.stack(
        [torch.cos(theta) * radius, y, torch.sin(theta) * radius],
        dim=-1,
    ).contiguous()


def _equirectangular_texel_directions(
    height: int,
    width: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if height <= 0 or width <= 0:
        raise ValueError(f"resolution entries must be positive, got {(height, width)}.")

    y, x = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    u = (x + 0.5) / float(width)
    v = (y + 0.5) / float(height)
    theta = u * 2.0 * torch.pi - torch.pi
    phi = (v - 0.5) * torch.pi
    return torch.stack(
        [
            torch.sin(theta) * torch.cos(phi),
            torch.cos(theta) * torch.cos(phi),
            -torch.sin(phi),
        ],
        dim=-1,
    ).contiguous()


def _as_resolution(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, str):
        parts = value.lower().replace("x", ",").split(",")
        value = [part.strip() for part in parts if part.strip()]
    if not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"resolution must be a (height, width) pair, got {value}.")
    height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"resolution entries must be positive, got {value}.")
    return height, width


def _as_rgb_tensor(
    value: float | Sequence[float] | torch.Tensor,
    device: torch.device | str | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == 0:
        tensor = tensor.expand(3)
    tensor = tensor.reshape(-1)
    if tensor.numel() != 3:
        raise ValueError(f"RGB value must have one or three entries, got {value}.")
    return tensor


def _alternating_sg_sharpness(
    n_lobes: int,
    device: torch.device | str | None,
    dtype: torch.dtype,
    low: float,
    high: float,
) -> torch.Tensor:
    indices = torch.arange(n_lobes, device=device)
    sharpnesses = torch.empty(n_lobes, device=device, dtype=dtype)
    sharpnesses[indices % 2 == 0] = float(low)
    sharpnesses[indices % 2 == 1] = float(high)
    return sharpnesses


class SphericalGaussianEnvironment(torch.nn.Module):
    """Learnable SG lighting with native evaluation and optional envmap export."""

    DEFAULT_NUM_LOBES = 24
    DEFAULT_GAMMA = 0.3
    DEFAULT_SHARPNESS_LOW = 1.0
    DEFAULT_SHARPNESS_HIGH = 20.0
    DEFAULT_RAW_AMPLITUDE = -3.5
    DEFAULT_MIN_SHARPNESS = 1.0e-4
    DEFAULT_MAX_SHARPNESS = 1.0e4
    NATIVE_PARAMETER_SIZE = 7
    NATIVE_SAMPLING_SIZE = 2
    LUMINANCE_WEIGHTS = (0.2126, 0.7152, 0.0722)

    def __init__(
        self,
        n_lobes: int = DEFAULT_NUM_LOBES,
        resolution: tuple[int, int] = _DEFAULT_ENVIRONMENT_SIZE,
        gamma: float = DEFAULT_GAMMA,
        sharpness: float | Sequence[float] | torch.Tensor | None = None,
        init_radiance: float | Sequence[float] | torch.Tensor = 0.5,
        min_sharpness: float = DEFAULT_MIN_SHARPNESS,
        max_sharpness: float = DEFAULT_MAX_SHARPNESS,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        raw_amplitudes: torch.Tensor | None = None,
        directions: torch.Tensor | None = None,
        sharpnesses: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        self.n_lobes = int(n_lobes)
        self.height, self.width = _as_resolution(resolution, _DEFAULT_ENVIRONMENT_SIZE)
        if self.height == 6 * self.width:
            raise ValueError(
                "spherical_gaussian resolution is evaluated as a 2D environment map "
                f"and must not use cubemap strip shape [6*N, N], got {(self.height, self.width)}."
            )
        self.gamma = float(gamma)
        self.min_sharpness = float(min_sharpness)
        self.max_sharpness = float(max_sharpness)

        if self.gamma <= 0.0:
            raise ValueError(f"gamma must be positive, got {gamma}.")
        if self.min_sharpness <= 0.0:
            raise ValueError(
                f"min_sharpness must be positive, got {min_sharpness}."
            )
        if self.max_sharpness <= self.min_sharpness:
            raise ValueError(
                "max_sharpness must be greater than min_sharpness, got "
                f"{self.max_sharpness} <= {self.min_sharpness}."
            )

        if directions is None:
            directions = _fibonacci_sphere_directions(
                self.n_lobes, device=device, dtype=dtype
            )
        else:
            directions = torch.as_tensor(directions, device=device, dtype=dtype)
            if directions.shape != (self.n_lobes, 3):
                raise ValueError(
                    f"directions must have shape [{self.n_lobes}, 3], got {tuple(directions.shape)}."
                )
        directions = F.normalize(directions, dim=-1)
        self.directions = torch.nn.Parameter(directions.detach().clone())

        texel_directions = _equirectangular_texel_directions(
            self.height, self.width, device=device, dtype=dtype
        )
        self.register_buffer("texel_directions", texel_directions)

        if sharpnesses is None:
            if sharpness is None:
                sharpnesses = _alternating_sg_sharpness(
                    self.n_lobes,
                    device=device,
                    dtype=dtype,
                    low=self.DEFAULT_SHARPNESS_LOW,
                    high=self.DEFAULT_SHARPNESS_HIGH,
                )
            else:
                sharpnesses = torch.as_tensor(sharpness, device=device, dtype=dtype)
                if sharpnesses.ndim == 0:
                    sharpnesses = sharpnesses.expand(self.n_lobes)
                sharpnesses = sharpnesses.reshape(-1)
        else:
            sharpnesses = torch.as_tensor(
                sharpnesses, device=device, dtype=dtype
            ).reshape(-1)
        if sharpnesses.numel() != self.n_lobes:
            raise ValueError(
                f"sharpness must have one or {self.n_lobes} entries, got {sharpnesses.numel()}."
            )
        sharpnesses = torch.clamp(sharpnesses, min=self.min_sharpness)
        raw_sharpness = _softplus_inverse(sharpnesses - self.min_sharpness)
        self.raw_sharpness = torch.nn.Parameter(raw_sharpness.detach().clone())

        if raw_amplitudes is None:
            init_rgb = _as_rgb_tensor(init_radiance, device, dtype)
            default_rgb = torch.full((3,), 0.5, device=device, dtype=dtype)
            if torch.allclose(init_rgb, default_rgb):
                raw_amplitudes = torch.full(
                    (self.n_lobes, 3),
                    self.DEFAULT_RAW_AMPLITUDE,
                    device=device,
                    dtype=dtype,
                )
            else:
                kernel_mean = (1.0 - torch.exp(-2.0 * sharpnesses)) / (
                    2.0 * sharpnesses
                )
                amplitudes = init_rgb.reshape(1, 3) / torch.clamp(
                    float(self.n_lobes) * kernel_mean.reshape(self.n_lobes, 1),
                    min=torch.finfo(dtype).tiny,
                )
                raw_amplitudes = _radiance_to_sg_raw(amplitudes, self.gamma)
        else:
            raw_amplitudes = torch.as_tensor(
                raw_amplitudes, device=device, dtype=dtype
            )
            if raw_amplitudes.shape != (self.n_lobes, 3):
                raise ValueError(
                    f"raw_amplitudes must have shape [{self.n_lobes}, 3], got {tuple(raw_amplitudes.shape)}."
                )
        self.raw_amplitudes = torch.nn.Parameter(raw_amplitudes.detach().clone())

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        init_radiance: float | Sequence[float] | torch.Tensor = 0.5,
    ) -> "SphericalGaussianEnvironment":
        config = {} if config is None else dict(config)
        resolution = _as_resolution(
            config.get("resolution", config.get("size", None)),
            _DEFAULT_ENVIRONMENT_SIZE,
        )
        return cls(
            n_lobes=int(
                config.get("n_lobes", config.get("num_lobes", cls.DEFAULT_NUM_LOBES))
            ),
            resolution=resolution,
            gamma=float(config.get("gamma", cls.DEFAULT_GAMMA)),
            sharpness=config.get("sharpness", None),
            init_radiance=init_radiance,
            min_sharpness=float(
                config.get("min_sharpness", cls.DEFAULT_MIN_SHARPNESS)
            ),
            max_sharpness=float(
                config.get("max_sharpness", cls.DEFAULT_MAX_SHARPNESS)
            ),
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, torch.Tensor],
        config: Mapping[str, Any] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "SphericalGaussianEnvironment":
        config = {} if config is None else dict(config)
        raw_amplitudes = torch.as_tensor(
            state["raw_amplitudes"], device=device, dtype=dtype
        )
        n_lobes = int(raw_amplitudes.shape[0])

        directions = state.get("directions", None)
        if directions is not None:
            directions = torch.as_tensor(directions, device=device, dtype=dtype)

        sharpnesses = None
        if "raw_sharpness" not in state and "sharpnesses" in state:
            sharpnesses = torch.as_tensor(
                state["sharpnesses"], device=device, dtype=dtype
            )

        texel_directions = state.get("texel_directions", None)
        if texel_directions is not None:
            texel_directions = torch.as_tensor(
                texel_directions, device=device, dtype=dtype
            )
            resolution = tuple(int(v) for v in texel_directions.shape[:2])
        else:
            resolution = _as_resolution(
                config.get("resolution", config.get("size", None)),
                _DEFAULT_ENVIRONMENT_SIZE,
            )

        module = cls(
            n_lobes=n_lobes,
            resolution=resolution,
            gamma=float(config.get("gamma", cls.DEFAULT_GAMMA)),
            sharpness=None if sharpnesses is None else sharpnesses,
            init_radiance=0.5,
            min_sharpness=float(
                config.get("min_sharpness", cls.DEFAULT_MIN_SHARPNESS)
            ),
            max_sharpness=float(
                config.get("max_sharpness", cls.DEFAULT_MAX_SHARPNESS)
            ),
            device=device,
            dtype=dtype,
            raw_amplitudes=raw_amplitudes,
            directions=directions,
            sharpnesses=sharpnesses,
        )
        module.load_state_dict(dict(state), strict=False)
        return module

    def config_dict(self) -> dict[str, Any]:
        return {
            "n_lobes": self.n_lobes,
            "resolution": [self.height, self.width],
            "gamma": self.gamma,
            "min_sharpness": self.min_sharpness,
            "max_sharpness": self.max_sharpness,
        }

    def amplitudes(self) -> torch.Tensor:
        return _sg_raw_to_radiance(self.raw_amplitudes, self.gamma)

    def sharpness(self) -> torch.Tensor:
        sharpness = F.softplus(self.raw_sharpness) + self.min_sharpness
        sharpness = torch.nan_to_num(
            sharpness,
            nan=self.min_sharpness,
            posinf=self.max_sharpness,
            neginf=self.min_sharpness,
        )
        return torch.clamp(
            sharpness,
            min=self.min_sharpness,
            max=self.max_sharpness,
        )

    def normalized_directions(self) -> torch.Tensor:
        directions = torch.nan_to_num(
            self.directions, nan=0.0, posinf=0.0, neginf=0.0
        )
        lengths = torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
        normalized = directions / torch.clamp(lengths, min=1.0e-8)
        fallback = torch.zeros_like(normalized)
        fallback[..., 1] = 1.0
        return torch.where(lengths > 1.0e-8, normalized, fallback)

    def native_parameters(self) -> torch.Tensor:
        """Pack current differentiable SGs as [axis.xyz, lambda, amplitude.rgb]."""
        return torch.cat(
            [
                self.normalized_directions(),
                self.sharpness().reshape(-1, 1),
                self.amplitudes(),
            ],
            dim=-1,
        ).contiguous()

    @staticmethod
    def normalization_from_sharpness(sharpness: torch.Tensor) -> torch.Tensor:
        """Analytical integral of exp(lambda * (mu dot omega - 1))."""
        sharpness = torch.clamp(
            torch.nan_to_num(sharpness, nan=1.0e-8, posinf=1.0e4, neginf=1.0e-8),
            min=1.0e-8,
            max=1.0e4,
        )
        return 2.0 * torch.pi * (-torch.expm1(-2.0 * sharpness)) / sharpness

    def native_sampling_distribution(
        self, parameters: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Build the current detached [Z_k, alpha_k] SG proposal snapshot.

        This intentionally does not cache values: optimized amplitudes and
        sharpnesses must produce a fresh proposal for every renderer forward.
        Sampling/PDF terms remain detached, matching the existing replay
        estimator's treatment of its envmap alias table.
        """
        if parameters is None:
            parameters = self.native_parameters()
        if parameters.ndim != 2 or parameters.shape[-1] != self.NATIVE_PARAMETER_SIZE:
            raise ValueError(
                "native SG parameters must have shape [K, 7], got "
                f"{tuple(parameters.shape)}."
            )

        with torch.no_grad():
            snapshot = parameters.detach()
            normalization = self.normalization_from_sharpness(snapshot[:, 3])
            luminance_weights = snapshot.new_tensor(self.LUMINANCE_WEIGHTS)
            luminance = torch.sum(
                torch.clamp(torch.nan_to_num(snapshot[:, 4:7]), min=0.0)
                * luminance_weights,
                dim=-1,
            )
            lobe_weights = torch.nan_to_num(
                luminance * normalization, nan=0.0, posinf=0.0, neginf=0.0
            )
            total_weight = torch.sum(lobe_weights)
            tiny = torch.finfo(snapshot.dtype).tiny
            safe_total = torch.clamp(total_weight, min=tiny)
            weighted_alpha = lobe_weights / safe_total
            uniform_alpha = torch.full_like(
                weighted_alpha, 1.0 / float(max(1, snapshot.shape[0]))
            )
            valid_total = torch.isfinite(total_weight) & (total_weight > tiny)
            alpha = torch.where(valid_total, weighted_alpha, uniform_alpha)
            return torch.stack([normalization, alpha], dim=-1).contiguous()

    def evaluate_directions(self, directions: torch.Tensor) -> torch.Tensor:
        """Evaluate the SG sum directly for arbitrary directions."""
        directions = F.normalize(torch.nan_to_num(directions), dim=-1)
        parameters = self.native_parameters()
        dots = torch.matmul(directions, parameters[:, :3].transpose(0, 1))
        basis = torch.exp(parameters[:, 3] * (torch.clamp(dots, -1.0, 1.0) - 1.0))
        return torch.matmul(basis, parameters[:, 4:7])

    def mixture_pdf(
        self,
        directions: torch.Tensor,
        parameters: torch.Tensor | None = None,
        sampling_distribution: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate the exact full SG-mixture density for validation/debugging."""
        parameters = self.native_parameters() if parameters is None else parameters
        sampling_distribution = (
            self.native_sampling_distribution(parameters)
            if sampling_distribution is None
            else sampling_distribution
        )
        directions = F.normalize(torch.nan_to_num(directions), dim=-1)
        dots = torch.matmul(directions, parameters[:, :3].transpose(0, 1))
        basis = torch.exp(parameters[:, 3] * (torch.clamp(dots, -1.0, 1.0) - 1.0))
        component_pdf = basis / torch.clamp(
            sampling_distribution[:, 0], min=torch.finfo(parameters.dtype).tiny
        )
        return torch.matmul(component_pdf, sampling_distribution[:, 1])

    @torch.no_grad()
    def sample_directions(
        self,
        num_samples: int,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference native SG sampler used by numerical validation/debugging."""
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")
        parameters = self.native_parameters().detach()
        sampling = self.native_sampling_distribution(parameters)
        random_values = torch.rand(
            (num_samples, 3),
            dtype=parameters.dtype,
            device=parameters.device,
            generator=generator,
        )
        cumulative = torch.cumsum(sampling[:, 1], dim=0)
        cumulative[-1] = 1.0
        selected = torch.searchsorted(
            cumulative, random_values[:, 0].contiguous()
        ).clamp_max(parameters.shape[0] - 1)
        axis = parameters[selected, :3]
        sharpness = parameters[selected, 3]

        one_minus_exp_negative_two_lambda = -torch.expm1(-2.0 * sharpness)
        one_minus_inverse_cdf = torch.clamp(
            (1.0 - random_values[:, 1]) * one_minus_exp_negative_two_lambda,
            max=1.0 - torch.finfo(parameters.dtype).eps,
        )
        log_inverse_cdf = torch.log1p(-one_minus_inverse_cdf)
        concentrated_cos_theta = 1.0 + log_inverse_cdf / torch.clamp(
            sharpness, min=1.0e-8
        )
        uniform_cos_theta = -1.0 + 2.0 * random_values[:, 1]
        cos_theta = torch.where(
            sharpness < 1.0e-6, uniform_cos_theta, concentrated_cos_theta
        ).clamp(-1.0, 1.0)
        sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta.square(), min=0.0))
        phi = 2.0 * torch.pi * random_values[:, 2]

        helper_z = torch.zeros_like(axis)
        helper_z[:, 2] = 1.0
        helper_x = torch.zeros_like(axis)
        helper_x[:, 0] = 1.0
        helper = torch.where((axis[:, 2].abs() < 0.999).unsqueeze(-1), helper_z, helper_x)
        tangent = F.normalize(torch.cross(helper, axis, dim=-1), dim=-1)
        bitangent = torch.cross(axis, tangent, dim=-1)
        directions = F.normalize(
            tangent * (sin_theta * torch.cos(phi)).unsqueeze(-1)
            + bitangent * (sin_theta * torch.sin(phi)).unsqueeze(-1)
            + axis * cos_theta.unsqueeze(-1),
            dim=-1,
        )
        return directions, self.mixture_pdf(directions, parameters, sampling)

    def estimate_power(self) -> torch.Tensor:
        """Analytical luminance integral used by the top-level light sampler."""
        parameters = self.native_parameters()
        luminance_weights = parameters.new_tensor(self.LUMINANCE_WEIGHTS)
        luminance = torch.sum(parameters[:, 4:7] * luminance_weights, dim=-1)
        return torch.sum(
            luminance * self.normalization_from_sharpness(parameters[:, 3])
        )

    def forward(self) -> torch.Tensor:
        # Optional visualization/debug/export path. Native PTIR rendering calls
        # native_parameters() and never needs this rasterization.
        rgb = self.evaluate_directions(
            self.texel_directions.reshape(-1, 3)
        ).reshape(self.height, self.width, 3)
        alpha = torch.ones(
            self.height,
            self.width,
            1,
            dtype=rgb.dtype,
            device=rgb.device,
        )
        return torch.cat([rgb, alpha], dim=-1).contiguous()


class Environment:
    """Load environment maps and expose them as 4-channel torch tensors."""

    FIXED_ENVIRONMENT_OPTIONS = ["Model-Background", "Black", "White"]
    ENVIRONMENT_EXTENSIONS = (".hdr", ".exr", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
    CUBEMAP_FACE_NAMES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
    CUBEMAP_FACE_ALIASES = (
        ("+x", "posx", "px", "right"),
        ("-x", "negx", "nx", "left"),
        ("+y", "posy", "py", "top", "up"),
        ("-y", "negy", "ny", "bottom", "down"),
        ("+z", "posz", "pz", "front"),
        ("-z", "negz", "nz", "back"),
    )
    DEFAULT_ENVIRONMENT_SIZE = _DEFAULT_ENVIRONMENT_SIZE
    DEFAULT_CUBEMAP_FACE_SIZE = 64
    LOG_ENVIRONMENT_MIN = 1.0e-6
    LOG_ENVIRONMENT_PARAMETERIZATION = "log_exp"
    LINEAR_ENVIRONMENT_PARAMETERIZATION = "linear"

    def __init__(
        self,
        path: Optional[str] = None,
        device: Optional[torch.device | str] = None,
        environment_type: str = "2d",
        optimize_environment: bool = False,
        parameterization: str = LINEAR_ENVIRONMENT_PARAMETERIZATION,
    ):
        self.device = device
        self.path = path
        self.folder = None
        self.environment_type = str(environment_type).lower()
        self.environment_parameterization = (
            self._normalize_environment_parameterization(parameterization)
        )
        self.optimize_environment = bool(optimize_environment)
        self.intensity = 1.0

        self.current_name = "Model-Background"
        self.environment = None
        self._hdr_data = None
        self.environment_offset = [0.0, 0.0]

        self.available_environments = [
            option for option in self.FIXED_ENVIRONMENT_OPTIONS
        ]
        if path is None:
            self.init_environment()
        else:
            self.load_path(path)

    @classmethod
    def _actual_to_internal(cls, environment: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(environment, min=cls.LOG_ENVIRONMENT_MIN))

    @staticmethod
    def _internal_to_actual(environment: torch.Tensor) -> torch.Tensor:
        return torch.exp(environment)

    def _uses_log_parameterization(self) -> bool:
        return (
            self.environment_parameterization == self.LOG_ENVIRONMENT_PARAMETERIZATION
        )

    def _as_environment_tensor(self, environment: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(
            environment, dtype=torch.float32, device=self.device
        ).contiguous()
        if tensor.dim() != 3 or tensor.size(-1) != 4:
            raise ValueError(
                f"environment must have shape [H, W, 4], got {tuple(tensor.shape)}"
            )
        return tensor

    def _set_environment_parameter(
        self, environment: torch.Tensor, parameterization: Optional[str] = None
    ) -> None:
        if parameterization is not None:
            self.environment_parameterization = (
                self._normalize_environment_parameterization(parameterization)
            )
        tensor = self._as_environment_tensor(environment)
        if self.optimize_environment:
            self.environment = torch.nn.Parameter(
                tensor.detach().clone(), requires_grad=True
            )
        elif self._uses_log_parameterization():
            self.environment = self._internal_to_actual(tensor).detach()
        else:
            self.environment = tensor.detach()

    def _set_environment_tensor(self, environment: Optional[torch.Tensor]) -> None:
        if environment is None:
            self.environment = None
            return

        tensor = self._as_environment_tensor(environment)
        if self.optimize_environment and self._uses_log_parameterization():
            tensor = self._actual_to_internal(tensor)
            self.environment = torch.nn.Parameter(
                tensor.detach().clone(), requires_grad=True
            )
        elif self.optimize_environment:
            self.environment = torch.nn.Parameter(
                tensor.detach().clone(), requires_grad=True
            )
        else:
            self.environment = tensor.detach()

    def configure_optimization(self, enabled: bool) -> None:
        self.optimize_environment = bool(enabled)
        environment = self.get_environment()
        if environment is not None:
            self._set_environment_tensor(environment)

    @classmethod
    def _normalize_environment_parameterization(cls, parameterization: str) -> str:
        normalized = str(parameterization).lower()
        options = (
            cls.LINEAR_ENVIRONMENT_PARAMETERIZATION,
            cls.LOG_ENVIRONMENT_PARAMETERIZATION,
        )
        if normalized not in options:
            raise ValueError(
                f"environment.parameterization must be one of {options}, got '{parameterization}'."
            )
        return normalized

    @classmethod
    def _list_environments(cls, folder: str) -> list[str]:
        return [
            name
            for name in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, name))
            or name.lower().endswith(cls.ENVIRONMENT_EXTENSIONS)
        ]

    def _read_environment_file(self, environment_path: str) -> np.ndarray:
        suffix = os.path.splitext(environment_path)[1].lower()
        if suffix == ".hdr":
            try:
                return imageio.v2.imread(environment_path, format="HDR-FI")
            except RuntimeError:
                # HDR loading requires the FreeImage plugin library.
                fi.download()
                return imageio.v2.imread(environment_path, format="HDR-FI")
        return imageio.v2.imread(environment_path)

    @staticmethod
    def _prepare_rgb(data: np.ndarray) -> np.ndarray:
        rgb = np.asarray(data)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=-1)
        if rgb.ndim != 3:
            raise ValueError(
                f"Environment map must have shape HxW or HxWxC, got {rgb.shape}."
            )
        if rgb.shape[-1] == 1:
            rgb = np.repeat(rgb, 3, axis=-1)
        elif rgb.shape[-1] > 3:
            rgb = rgb[..., :3]
        elif rgb.shape[-1] != 3:
            raise ValueError(
                f"Environment map must have 1, 3, or 4 channels, got {rgb.shape[-1]}."
            )

        if np.issubdtype(rgb.dtype, np.integer):
            rgb = rgb.astype(np.float32) / np.iinfo(rgb.dtype).max
        else:
            rgb = rgb.astype(np.float32, copy=False)

        return np.maximum(np.nan_to_num(rgb, nan=0.0, neginf=0.0), 0.0)

    @classmethod
    def _prepare_cubemap(cls, rgb: np.ndarray) -> np.ndarray:
        if rgb.ndim == 4:
            if rgb.shape[0] != 6 or rgb.shape[1] != rgb.shape[2]:
                raise ValueError(
                    f"Cubemap array must have shape 6xNxNxC, got {rgb.shape}."
                )
            return np.concatenate([rgb[face] for face in range(6)], axis=0)

        height, width, _ = rgb.shape
        if height == 6 * width:
            return rgb
        if width == 6 * height:
            return np.concatenate(
                [rgb[:, face * height : (face + 1) * height] for face in range(6)],
                axis=0,
            )

        raise ValueError(
            "Cubemap must be a vertical strip [6*N, N, C], a horizontal strip [N, 6*N, C], "
            f"or six square faces; got {rgb.shape}."
        )

    def _prepare_environment_data(self, data: np.ndarray) -> np.ndarray:
        rgb = self._prepare_rgb(data)
        if self.environment_type == ENVIRONMENT_TYPE_CUBE:
            rgb = self._prepare_cubemap(rgb)
        return rgb

    @classmethod
    def _find_cubemap_face_paths(cls, folder: str) -> list[str]:
        files = [
            name
            for name in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, name))
            and name.lower().endswith(cls.ENVIRONMENT_EXTENSIONS)
        ]
        lowered = {os.path.splitext(name)[0].lower(): name for name in files}

        face_paths = []
        for aliases in cls.CUBEMAP_FACE_ALIASES:
            match = None
            for alias in aliases:
                if alias in lowered:
                    match = lowered[alias]
                    break
            if match is None:
                raise FileNotFoundError(
                    f"Could not find cubemap face {cls.CUBEMAP_FACE_NAMES[len(face_paths)]} in {folder}. "
                    f"Expected one of: {aliases}."
                )
            face_paths.append(os.path.join(folder, match))

        return face_paths

    def load_cubemap_files(
        self, face_paths: Sequence[str] | Mapping[str, str]
    ) -> torch.Tensor:
        """Load six square cubemap face files in +X, -X, +Y, -Y, +Z, -Z order."""
        if isinstance(face_paths, Mapping):
            face_paths = [face_paths[name] for name in self.CUBEMAP_FACE_NAMES]
        if len(face_paths) != 6:
            raise ValueError(
                f"Cubemap loading requires six face files, got {len(face_paths)}."
            )

        faces = [
            self._prepare_rgb(self._read_environment_file(path)) for path in face_paths
        ]
        face_size = faces[0].shape[0]
        for face_name, face in zip(self.CUBEMAP_FACE_NAMES, faces):
            if face.shape[0] != face.shape[1]:
                raise ValueError(
                    f"Cubemap face {face_name} must be square, got {face.shape}."
                )
            if face.shape[:2] != (face_size, face_size):
                raise ValueError(
                    f"Cubemap face {face_name} shape {face.shape[:2]} does not match {face_size}x{face_size}."
                )

        self.environment_type = ENVIRONMENT_TYPE_CUBE
        self.path = None
        self.folder = os.path.commonpath(
            [os.path.dirname(os.path.abspath(path)) for path in face_paths]
        )
        self.current_name = "Cubemap-Faces"
        self._hdr_data = self._prepare_cubemap(np.stack(faces, axis=0))
        self._update()
        return self.get_environment()

    def load_path(self, environment_path: str) -> torch.Tensor:
        """Load an environment map from an explicit file path."""
        if os.path.isdir(environment_path):
            if self.environment_type != ENVIRONMENT_TYPE_CUBE:
                raise ValueError("Directory loading is only supported for cubemaps.")
            environment = self.load_cubemap_files(
                self._find_cubemap_face_paths(environment_path)
            )
            self.path = environment_path
            self.current_name = os.path.basename(os.path.normpath(environment_path))
            return environment

        if not os.path.isfile(environment_path):
            raise FileNotFoundError(f"Environment map not found: {environment_path}")

        self.path = environment_path
        self.folder = os.path.dirname(environment_path)
        self.current_name = os.path.basename(environment_path)
        self._hdr_data = self._prepare_environment_data(
            self._read_environment_file(environment_path)
        )
        self._update()
        return self.get_environment()

    def load_file(self, environment_path: str) -> torch.Tensor:
        return self.load_path(environment_path)

    def _load_hdr(
        self, environment_name: Optional[str] = None
    ) -> Optional[torch.Tensor]:
        """Load an environment map by name from ``self.folder``."""
        if (
            not self.available_environments
            or environment_name in self.FIXED_ENVIRONMENT_OPTIONS
        ):
            self.environment = None
            return None

        if environment_name not in self.available_environments:
            raise ValueError(
                f"Environment map {self.folder}{os.path.sep}{environment_name} not found."
            )

        if environment_name != self.current_name:
            environment_path = os.path.join(self.folder, environment_name)
            if os.path.isdir(environment_path):
                self.load_path(environment_path)
            else:
                self._hdr_data = self._prepare_environment_data(
                    self._read_environment_file(environment_path)
                )
                self._update()

        return self.get_environment()

    def _constant_environment(self, value: float) -> torch.Tensor:
        if self.environment_type == ENVIRONMENT_TYPE_CUBE:
            height = 6 * self.DEFAULT_CUBEMAP_FACE_SIZE
            width = self.DEFAULT_CUBEMAP_FACE_SIZE
        else:
            height, width = self.DEFAULT_ENVIRONMENT_SIZE
        environment = torch.full(
            [height, width, 4],
            value * self.intensity,
            dtype=torch.float32,
            device=self.device,
        )
        environment[..., 3] = 1.0
        return environment

    def init_environment(self, value: float = 0.5) -> torch.Tensor:
        self.path = None
        self.folder = None
        self.current_name = "Initialized"
        self._hdr_data = None
        self._set_environment_tensor(self._constant_environment(value))
        return self.get_environment()

    def set_env(self, env_name: Optional[str] = None) -> None:
        if env_name in ("Model-Background", "Black"):
            self._hdr_data = None
            self._set_environment_tensor(self._constant_environment(0.0))
        elif env_name == "White":
            self._hdr_data = None
            self._set_environment_tensor(self._constant_environment(1.0))
        else:
            self._load_hdr(env_name)
        self.current_name = env_name

    def _update(self) -> None:
        if self._hdr_data is None:
            return
        environment = torch.as_tensor(
            self._hdr_data, dtype=torch.float32, device=self.device
        ).contiguous()
        environment = environment * self.intensity
        pad = environment.new_ones(environment.shape[0], environment.shape[1], 1)
        self._set_environment_tensor(torch.cat([environment, pad], dim=-1))

    def get_environment_parameter(
        self,
    ) -> torch.Tensor | torch.nn.Parameter | torch.nn.Module | None:
        return self.environment

    def get_environment(self) -> Optional[torch.Tensor]:
        if self.environment is None:
            return None
        if self.optimize_environment and self._uses_log_parameterization():
            return self._internal_to_actual(self.environment)
        return self.environment

    def get_environment_offset(self) -> torch.Tensor:
        return torch.tensor(
            self.environment_offset, dtype=torch.float32, device=self.device
        )

    def build_alias_table(
        self,
        environment: Optional[torch.Tensor] = None,
        target_size: Optional[tuple[int, int]] = DEFAULT_ALIAS_TABLE_SIZE,
        eps: float = 0.0,
    ) -> Optional[EnvAliasTable]:
        if environment is None:
            environment = self.get_environment()
        return build_environment_alias_table(
            environment,
            environment_type=self.environment_type,
            target_size=target_size,
            eps=eps,
        )

    def is_ignore_environment(self) -> bool:
        return self.current_name == "Model-Background"

    def state_dict(self) -> dict:
        return {
            "current_name": self.current_name,
            "path": self.path,
            "environment_offset": list(self.environment_offset),
            "environment": None
            if self.environment is None
            else self.environment.detach().clone(),
            "environment_parameterization": (
                self.environment_parameterization
                if self.optimize_environment
                else self.LINEAR_ENVIRONMENT_PARAMETERIZATION
            ),
            "environment_type": self.environment_type,
            "optimize_environment": self.optimize_environment,
            "intensity": self.intensity,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.current_name = state_dict.get("current_name", self.current_name)
        self.path = state_dict.get("path", self.path)
        self.environment_offset = list(
            state_dict.get("environment_offset", self.environment_offset)
        )
        self.environment_type = str(
            state_dict.get("environment_type", self.environment_type)
        ).lower()
        self.optimize_environment = bool(
            state_dict.get("optimize_environment", self.optimize_environment)
        )
        self.intensity = float(state_dict.get("intensity", self.intensity))
        environment = state_dict.get("environment")
        parameterization = state_dict.get(
            "environment_parameterization", self.LINEAR_ENVIRONMENT_PARAMETERIZATION
        )
        if environment is None:
            self.environment_parameterization = (
                self._normalize_environment_parameterization(parameterization)
            )
            self.environment = None
        else:
            self._set_environment_parameter(environment, parameterization)
        self._hdr_data = None


class SGEnvironment(Environment):
    """Environment wrapper for learnable spherical Gaussian lighting."""

    def __init__(
        self,
        path: Optional[str] = None,
        device: Optional[torch.device | str] = None,
        optimize_environment: bool = False,
        spherical_gaussian: Mapping[str, Any] | None = None,
    ) -> None:
        self.spherical_gaussian_config = (
            {} if spherical_gaussian is None else dict(spherical_gaussian)
        )
        super().__init__(
            path=path,
            device=device,
            environment_type=ENVIRONMENT_TYPE_SPHERICAL_GAUSSIAN,
            optimize_environment=optimize_environment,
            parameterization=self.LINEAR_ENVIRONMENT_PARAMETERIZATION,
        )

    def _set_spherical_gaussian_module(
        self, module: SphericalGaussianEnvironment
    ) -> None:
        if self.device is not None:
            module = module.to(device=self.device)
        module.requires_grad_(self.optimize_environment)
        self.environment = module
        self.environment_parameterization = self.LINEAR_ENVIRONMENT_PARAMETERIZATION
        self.spherical_gaussian_config = module.config_dict()

    def _set_spherical_gaussian_environment(
        self,
        init_radiance: float | Sequence[float] | torch.Tensor = 0.5,
        state: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        if state is None:
            module = SphericalGaussianEnvironment.from_config(
                self.spherical_gaussian_config,
                device=self.device,
                dtype=torch.float32,
                init_radiance=init_radiance,
            )
        else:
            module = SphericalGaussianEnvironment.from_state(
                state,
                config=self.spherical_gaussian_config,
                device=self.device,
                dtype=torch.float32,
            )
        self._set_spherical_gaussian_module(module)

    def _set_spherical_gaussian_from_tensor(self, environment: torch.Tensor) -> None:
        tensor = self._as_environment_tensor(environment)
        mean_rgb = tensor[..., :3].mean(dim=(0, 1)).detach()
        self._set_spherical_gaussian_environment(init_radiance=mean_rgb)

    def _set_environment_parameter(
        self, environment: torch.Tensor, parameterization: Optional[str] = None
    ) -> None:
        self.environment_parameterization = self.LINEAR_ENVIRONMENT_PARAMETERIZATION
        self._set_spherical_gaussian_from_tensor(environment)

    def _set_environment_tensor(self, environment: Optional[torch.Tensor]) -> None:
        if environment is None:
            self.environment = None
            return
        self._set_spherical_gaussian_from_tensor(environment)

    def configure_optimization(self, enabled: bool) -> None:
        self.optimize_environment = bool(enabled)
        if isinstance(self.environment, SphericalGaussianEnvironment):
            self.environment.requires_grad_(self.optimize_environment)

    def init_environment(self, value: float = 0.5) -> torch.Tensor:
        self.path = None
        self.folder = None
        self.current_name = "Initialized"
        self._hdr_data = None
        self._set_spherical_gaussian_environment(init_radiance=value)
        return self.get_environment()

    def set_env(self, env_name: Optional[str] = None) -> None:
        if env_name in ("Model-Background", "Black"):
            self._hdr_data = None
            self._set_spherical_gaussian_environment(init_radiance=0.0)
        elif env_name == "White":
            self._hdr_data = None
            self._set_spherical_gaussian_environment(init_radiance=1.0)
        else:
            self._load_hdr(env_name)
        self.current_name = env_name

    def get_environment_parameter(
        self,
    ) -> torch.nn.Module | None:
        return self.environment

    def get_environment(self) -> Optional[torch.Tensor]:
        if self.environment is None:
            return None
        return self.environment()

    def state_dict(self) -> dict:
        return {
            "current_name": self.current_name,
            "path": self.path,
            "environment_offset": list(self.environment_offset),
            "environment": None,
            "spherical_gaussian_state": None
            if self.environment is None
            else {
                name: value.detach().clone()
                for name, value in self.environment.state_dict().items()
            },
            "spherical_gaussian_config": self.spherical_gaussian_config,
            "environment_parameterization": self.LINEAR_ENVIRONMENT_PARAMETERIZATION,
            "environment_type": self.environment_type,
            "optimize_environment": self.optimize_environment,
            "intensity": self.intensity,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.current_name = state_dict.get("current_name", self.current_name)
        self.path = state_dict.get("path", self.path)
        self.environment_offset = list(
            state_dict.get("environment_offset", self.environment_offset)
        )
        self.environment_type = ENVIRONMENT_TYPE_SPHERICAL_GAUSSIAN
        self.optimize_environment = bool(
            state_dict.get("optimize_environment", self.optimize_environment)
        )
        self.intensity = float(state_dict.get("intensity", self.intensity))
        self.spherical_gaussian_config = dict(
            state_dict.get(
                "spherical_gaussian_config", self.spherical_gaussian_config
            )
            or {}
        )

        spherical_gaussian_state = state_dict.get("spherical_gaussian_state")
        if spherical_gaussian_state is not None:
            self._set_spherical_gaussian_environment(state=spherical_gaussian_state)
        else:
            environment = state_dict.get("environment")
            if environment is None:
                self._set_spherical_gaussian_environment()
            else:
                self._set_environment_parameter(environment)
        self._hdr_data = None


def create_environment(
    path: Optional[str] = None,
    device: Optional[torch.device | str] = None,
    environment_type: str = "2d",
    optimize_environment: bool = False,
    parameterization: str = Environment.LINEAR_ENVIRONMENT_PARAMETERIZATION,
    spherical_gaussian: Mapping[str, Any] | None = None,
) -> Environment:
    if str(environment_type).lower() == ENVIRONMENT_TYPE_SPHERICAL_GAUSSIAN:
        return SGEnvironment(
            path=path,
            device=device,
            optimize_environment=optimize_environment,
            spherical_gaussian=spherical_gaussian,
        )
    return Environment(
        path=path,
        device=device,
        environment_type=environment_type,
        optimize_environment=optimize_environment,
        parameterization=parameterization,
    )


def estimate_environment_power(
    environment: torch.Tensor | EnvironmentLight | SphericalGaussianEnvironment | None,
    environment_type: str = "2d",
) -> torch.Tensor | None:
    if isinstance(environment, SphericalGaussianEnvironment):
        return environment.estimate_power()
    if isinstance(environment, EnvironmentLight):
        return environment.estimate_power()
    if environment is None:
        return None

    environment_light = EnvironmentLight(
        environment=environment,
        environment_type=environment_type,
    )
    return environment_light.estimate_power()


def estimate_light_power(packed_lights: torch.Tensor) -> torch.Tensor:
    packed = torch.as_tensor(packed_lights, dtype=torch.float32)
    if packed.numel() == 0:
        return packed.new_empty((0,))
    if packed.ndim != 2 or packed.shape[1] != PACKED_LIGHT_SIZE:
        raise ValueError(
            f"packed lights must have shape [N, {PACKED_LIGHT_SIZE}], "
            f"got {tuple(packed.shape)}."
        )

    powers = packed.new_zeros((packed.shape[0],))
    light_types = packed[:, 0].round().to(dtype=torch.int64)

    point_mask = light_types == LIGHT_TYPE_POINT
    if bool(point_mask.any().item()):
        powers[point_mask] = _rgb_luminance(torch.clamp(packed[point_mask, 5:8], min=0.0))

    sphere_mask = light_types == LIGHT_TYPE_SPHERE
    if bool(sphere_mask.any().item()):
        radius = torch.clamp(packed[sphere_mask, 4], min=0.0)
        radiance = torch.clamp(packed[sphere_mask, 5:8], min=0.0)
        powers[sphere_mask] = _rgb_luminance(radiance) * (4.0 * torch.pi * radius * radius)

    return powers.contiguous()


def build_light_alias_table(
    environment: torch.Tensor
    | EnvironmentLight
    | SphericalGaussianEnvironment
    | None = None,
    environment_type: str = "2d",
    lights: torch.Tensor | None = None,
    mesh_powers: torch.Tensor | Sequence[float] | None = None,
    device: torch.device | str | None = None,
) -> LightAliasTable | None:
    entries_type = []
    entries_index = []
    entries_weight = []

    if isinstance(environment, SphericalGaussianEnvironment):
        environment_light_index = 0
        env_power = environment.estimate_power().detach()
    else:
        environment_light = (
            environment
            if isinstance(environment, EnvironmentLight)
            else EnvironmentLight(
                environment=environment,
                environment_type=environment_type,
            )
        )
        environment_light_index = environment_light.light_index
        env_power = environment_light.estimate_power()
    if env_power is not None:
        entries_type.append(LIGHT_TYPE_ENV)
        entries_index.append(environment_light_index)
        entries_weight.append(env_power)

    packed_lights = (
        torch.empty((0, PACKED_LIGHT_SIZE), dtype=torch.float32, device=device)
        if lights is None
        else torch.as_tensor(lights, dtype=torch.float32, device=device).contiguous()
    )
    light_powers = estimate_light_power(packed_lights)
    for index, power in enumerate(light_powers):
        if (
            bool(torch.isfinite(power).item())
            and float(power.detach().cpu().item()) > 0.0
        ):
            light_type = int(packed_lights[index, 0].detach().cpu().item() + 0.5)
            entries_type.append(light_type)
            entries_index.append(index)
            entries_weight.append(power)

    if mesh_powers is not None:
        mesh_powers = torch.as_tensor(
            mesh_powers,
            dtype=torch.float32,
            device=device,
        ).reshape(-1)
        for index, power in enumerate(mesh_powers):
            if (
                bool(torch.isfinite(power).item())
                and float(power.detach().cpu().item()) > 0.0
            ):
                entries_type.append(LIGHT_TYPE_MESH)
                entries_index.append(index)
                entries_weight.append(power)

    if not entries_weight:
        return None

    weights = torch.stack(
        [w.to(dtype=torch.float32, device=device) for w in entries_weight]
    )
    total = weights.sum()
    if (
        not bool(torch.isfinite(total).item())
        or float(total.detach().cpu().item()) <= 0.0
    ):
        return None

    prob, alias = build_alias_table(weights)
    light_select_pdf = (weights / total).to(dtype=torch.float32).contiguous()
    return LightAliasTable(
        prob=prob,
        alias=alias,
        light_type=torch.tensor(entries_type, dtype=torch.int32, device=prob.device),
        light_index=torch.tensor(entries_index, dtype=torch.int32, device=prob.device),
        light_select_pdf=light_select_pdf.to(device=prob.device),
    )


def build_mesh_light_triangle_alias_table(
    vertices: torch.Tensor,
    triangles: torch.Tensor,
) -> MeshLightTriangleAliasTable | None:
    vertices = _as_vertices(vertices)
    triangles = _as_triangles(triangles, vertices.shape[0], device=vertices.device)
    if triangles.numel() == 0:
        return None

    areas = _triangle_areas(vertices, triangles)
    total_area = areas.sum()
    if not bool(torch.isfinite(total_area).item()) or float(total_area.detach().cpu().item()) <= 0.0:
        return None

    prob, alias = build_alias_table(areas)
    area_pdf = torch.where(
        areas > 0.0,
        (areas / total_area) / torch.clamp(areas, min=torch.finfo(areas.dtype).tiny),
        torch.zeros_like(areas),
    )
    return MeshLightTriangleAliasTable(
        prob=prob,
        alias=alias,
        triangle_pdf=area_pdf.to(dtype=torch.float32).contiguous(),
        total_area=total_area.reshape(1).to(dtype=torch.float32).contiguous(),
    )


class MeshLight(Light):
    """Triangle mesh emitter, optionally with a uniform PBR surface.

    Packed mesh-light layout:
        [triangle_offset, triangle_count, vertex_offset, vertex_count,
         radiance.rgb, two_sided, albedo.rgb, roughness, metallic, surface]
    """

    PACKED_SIZE = PACKED_MESH_LIGHT_SIZE

    def __init__(
        self,
        vertices: torch.Tensor | Sequence[Sequence[float]],
        triangles: torch.Tensor | Sequence[Sequence[int]],
        radiance: TensorLike = (1.0, 1.0, 1.0),
        two_sided: bool = False,
        albedo: TensorLike = (0.8, 0.8, 0.8),
        roughness: torch.Tensor | float = 0.5,
        metallic: torch.Tensor | float = 0.0,
        surface: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.vertices = _as_vertices(vertices, device=device, dtype=dtype)
        self.triangles = _as_triangles(
            triangles,
            self.vertices.shape[0],
            device=self.vertices.device,
        )
        self.radiance = _as_vec3(
            radiance, "radiance", device=self.vertices.device, dtype=dtype
        )
        self.two_sided = bool(two_sided)
        self.albedo = _as_vec3(
            albedo, "albedo", device=self.vertices.device, dtype=dtype
        )
        self.roughness = _as_scalar(
            roughness, "roughness", device=self.vertices.device, dtype=dtype
        )
        self.metallic = _as_scalar(
            metallic, "metallic", device=self.vertices.device, dtype=dtype
        )
        self.surface = bool(surface)
        self._validate()

    @property
    def type(self) -> LightType:
        return LightType.MESH

    @property
    def device(self) -> torch.device:
        return self.vertices.device

    @property
    def dtype(self) -> torch.dtype:
        return self.vertices.dtype

    @property
    def num_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def num_triangles(self) -> int:
        return int(self.triangles.shape[0])

    def _validate(self) -> None:
        if self.num_vertices == 0:
            raise ValueError("vertices must contain at least one vertex.")
        if self.num_triangles == 0:
            raise ValueError("triangles must contain at least one triangle.")
        if bool((self.radiance.detach() < 0.0).any().item()):
            raise ValueError("radiance must be non-negative.")
        for name, value in (
            ("albedo", self.albedo),
            ("roughness", self.roughness),
            ("metallic", self.metallic),
        ):
            if bool(((value.detach() < 0.0) | (value.detach() > 1.0)).any().item()):
                raise ValueError(f"{name} must be in [0, 1].")
        if self.total_area() <= 0.0:
            raise ValueError("mesh light must contain at least one non-degenerate triangle.")

    def triangle_areas(self) -> torch.Tensor:
        return _triangle_areas(self.vertices, self.triangles)

    def total_area_tensor(self) -> torch.Tensor:
        return self.triangle_areas().sum().reshape(1).to(dtype=torch.float32)

    def total_area(self) -> float:
        return float(self.total_area_tensor().detach().cpu().item())

    def estimate_power(self) -> torch.Tensor:
        area = self.total_area_tensor().to(device=self.radiance.device)
        two_sided_scale = 2.0 if self.two_sided else 1.0
        return _rgb_luminance(torch.clamp(self.radiance, min=0.0)) * area[0] * two_sided_scale

    def build_triangle_alias_table(self) -> MeshLightTriangleAliasTable | None:
        return build_mesh_light_triangle_alias_table(self.vertices, self.triangles)

    def pack(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        triangle_offset: int = 0,
        vertex_offset: int = 0,
    ) -> torch.Tensor:
        device = self.device if device is None else device
        packed = torch.tensor(
            [
                float(triangle_offset),
                float(self.num_triangles),
                float(vertex_offset),
                float(self.num_vertices),
                float(self.radiance[0].detach().cpu().item()),
                float(self.radiance[1].detach().cpu().item()),
                float(self.radiance[2].detach().cpu().item()),
                1.0 if self.two_sided else 0.0,
                float(self.albedo[0].detach().cpu().item()),
                float(self.albedo[1].detach().cpu().item()),
                float(self.albedo[2].detach().cpu().item()),
                float(self.roughness[0].detach().cpu().item()),
                float(self.metallic[0].detach().cpu().item()),
                1.0 if self.surface else 0.0,
            ],
            dtype=dtype,
            device=device,
        )
        return packed.contiguous()

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "MeshLight":
        dtype = self.dtype if dtype is None else dtype
        return MeshLight(
            vertices=self.vertices.to(device=device, dtype=dtype),
            triangles=self.triangles.to(device=device),
            radiance=self.radiance.to(device=device, dtype=dtype),
            two_sided=self.two_sided,
            albedo=self.albedo.to(device=device, dtype=dtype),
            roughness=self.roughness.to(device=device, dtype=dtype),
            metallic=self.metallic.to(device=device, dtype=dtype),
            surface=self.surface,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.name.lower(),
            "vertices": self.vertices.detach().clone(),
            "triangles": self.triangles.detach().clone(),
            "radiance": self.radiance.detach().clone(),
            "two_sided": self.two_sided,
            "albedo": self.albedo.detach().clone(),
            "roughness": self.roughness.detach().clone(),
            "metallic": self.metallic.detach().clone(),
            "surface": self.surface,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.vertices = _as_vertices(
            state_dict.get("vertices", self.vertices),
            device=self.device,
            dtype=self.dtype,
        )
        self.triangles = _as_triangles(
            state_dict.get("triangles", self.triangles),
            self.vertices.shape[0],
            device=self.device,
        )
        self.radiance = _as_vec3(
            state_dict.get("radiance", self.radiance),
            "radiance",
            device=self.device,
            dtype=self.dtype,
        )
        self.two_sided = bool(state_dict.get("two_sided", self.two_sided))
        self.albedo = _as_vec3(
            state_dict.get("albedo", self.albedo),
            "albedo",
            device=self.device,
            dtype=self.dtype,
        )
        self.roughness = _as_scalar(
            state_dict.get("roughness", self.roughness),
            "roughness",
            device=self.device,
            dtype=self.dtype,
        )
        self.metallic = _as_scalar(
            state_dict.get("metallic", self.metallic),
            "metallic",
            device=self.device,
            dtype=self.dtype,
        )
        self.surface = bool(state_dict.get("surface", self.surface))
        self._validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "mesh",
            "vertices": self.vertices.detach().cpu().tolist(),
            "triangles": self.triangles.detach().cpu().tolist(),
            "radiance": self.radiance.detach().cpu().tolist(),
            "two_sided": self.two_sided,
            "albedo": self.albedo.detach().cpu().tolist(),
            "roughness": float(self.roughness.detach().cpu().item()),
            "metallic": float(self.metallic.detach().cpu().item()),
            "surface": self.surface,
        }


def pack_mesh_lights(
    mesh_lights: MeshLightPack
    | MeshLight
    | dict[str, Any]
    | Sequence[MeshLight | dict[str, Any]]
    | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> MeshLightPack:
    if isinstance(mesh_lights, MeshLightPack):
        return MeshLightPack(
            vertices=mesh_lights.vertices.to(device=device, dtype=dtype).contiguous(),
            triangles=mesh_lights.triangles.to(device=device, dtype=torch.int32).contiguous(),
            params=mesh_lights.params.to(device=device, dtype=dtype).contiguous(),
            triangle_alias_table=mesh_lights.triangle_alias_table.to(
                device=device, dtype=dtype
            ).contiguous(),
            powers=mesh_lights.powers.to(device=device, dtype=dtype).contiguous(),
        )

    if mesh_lights is None:
        return MeshLightPack(
            vertices=torch.empty((0, 3), dtype=dtype, device=device),
            triangles=torch.empty((0, 3), dtype=torch.int32, device=device),
            params=torch.empty((0, PACKED_MESH_LIGHT_SIZE), dtype=dtype, device=device),
            triangle_alias_table=torch.empty((3, 0), dtype=dtype, device=device),
            powers=torch.empty((0,), dtype=dtype, device=device),
        )

    if isinstance(mesh_lights, (MeshLight, dict)):
        mesh_lights = [mesh_lights]

    vertices_list = []
    triangles_list = []
    params_list = []
    alias_prob_list = []
    alias_index_list = []
    triangle_pdf_list = []
    powers_list = []
    vertex_offset = 0
    triangle_offset = 0

    for light in mesh_lights:
        if isinstance(light, dict):
            light = Light.from_dict(light)
        if not isinstance(light, MeshLight):
            raise TypeError(
                f"mesh_lights must be MeshLight instances or dictionaries, got {type(light).__name__}."
            )

        mesh_light = light.to(device=device, dtype=dtype)
        alias_table = mesh_light.build_triangle_alias_table()
        if alias_table is None:
            continue

        vertices_list.append(mesh_light.vertices)
        triangles_list.append(mesh_light.triangles + vertex_offset)
        params_list.append(
            mesh_light.pack(
                device=device,
                dtype=dtype,
                triangle_offset=triangle_offset,
                vertex_offset=vertex_offset,
            )
        )
        alias_prob_list.append(alias_table.prob.to(device=device, dtype=dtype))
        alias_index_list.append(alias_table.alias.to(device=device, dtype=dtype))
        triangle_pdf_list.append(alias_table.triangle_pdf.to(device=device, dtype=dtype))
        powers_list.append(mesh_light.estimate_power().to(device=device, dtype=dtype))

        vertex_offset += mesh_light.num_vertices
        triangle_offset += mesh_light.num_triangles

    if not params_list:
        return pack_mesh_lights(None, device=device, dtype=dtype)

    triangle_alias_table = torch.stack(
        [
            torch.cat(alias_prob_list, dim=0),
            torch.cat(alias_index_list, dim=0),
            torch.cat(triangle_pdf_list, dim=0),
        ],
        dim=0,
    ).contiguous()
    return MeshLightPack(
        vertices=torch.cat(vertices_list, dim=0).contiguous(),
        triangles=torch.cat(triangles_list, dim=0).to(dtype=torch.int32).contiguous(),
        params=torch.stack(params_list, dim=0).contiguous(),
        triangle_alias_table=triangle_alias_table,
        powers=torch.stack(powers_list, dim=0).reshape(-1).contiguous(),
    )


class PointLight(Light):
    """Mathematical point light.

    Packed layout:
        [position.xyz, unused, intensity.rgb, unused]
    """

    PACKED_SIZE = 8

    def __init__(
        self,
        position: TensorLike = (0.0, 0.0, 0.0),
        intensity: TensorLike | None = None,
        radiance: TensorLike | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if intensity is None:
            intensity = (1.0, 1.0, 1.0) if radiance is None else radiance
        self.position = _as_vec3(position, "position", device=device, dtype=dtype)
        self.intensity = _as_vec3(intensity, "intensity", device=device, dtype=dtype)
        self._validate()

    @property
    def type(self) -> LightType:
        return LightType.POINT

    @property
    def device(self) -> torch.device:
        return self.position.device

    @property
    def dtype(self) -> torch.dtype:
        return self.position.dtype

    @property
    def radiance(self) -> torch.Tensor:
        return self.intensity

    def _validate(self) -> None:
        if bool((self.intensity.detach() < 0.0).any().item()):
            raise ValueError("intensity must be non-negative.")

    def pack(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        device = self.device if device is None else device
        unused = torch.zeros((1,), dtype=dtype, device=device)
        packed = torch.cat(
            [
                self.position.to(device=device, dtype=dtype),
                unused,
                self.intensity.to(device=device, dtype=dtype),
                unused,
            ],
            dim=0,
        )
        return packed.contiguous()

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "PointLight":
        dtype = self.dtype if dtype is None else dtype
        return PointLight(
            position=self.position.to(device=device, dtype=dtype),
            intensity=self.intensity.to(device=device, dtype=dtype),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.name.lower(),
            "position": self.position.detach().clone(),
            "intensity": self.intensity.detach().clone(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.position = _as_vec3(
            state_dict.get("position", self.position),
            "position",
            device=self.device,
            dtype=self.dtype,
        )
        self.intensity = _as_vec3(
            state_dict.get("intensity", self.intensity),
            "intensity",
            device=self.device,
            dtype=self.dtype,
        )
        self._validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "point",
            "position": self.position.detach().cpu().tolist(),
            "intensity": self.intensity.detach().cpu().tolist(),
        }


class SphereLight(Light):
    """Spherical area light.

    Packed layout:
        [center.xyz, radius, radiance.rgb, two_sided]
    """

    PACKED_SIZE = 8

    def __init__(
        self,
        center: TensorLike = (0.0, 0.0, 0.0),
        radius: torch.Tensor | float = 1.0,
        radiance: TensorLike = (1.0, 1.0, 1.0),
        two_sided: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.center = _as_vec3(center, "center", device=device, dtype=dtype)
        self.radius = _as_scalar(radius, "radius", device=device, dtype=dtype)
        self.radiance = _as_vec3(radiance, "radiance", device=device, dtype=dtype)
        self.two_sided = bool(two_sided)
        self._validate()

    @property
    def type(self) -> LightType:
        return LightType.SPHERE

    @property
    def device(self) -> torch.device:
        return self.center.device

    @property
    def dtype(self) -> torch.dtype:
        return self.center.dtype

    def _validate(self) -> None:
        if float(self.radius.detach().cpu().item()) <= 0.0:
            raise ValueError("radius must be positive.")
        if bool((self.radiance.detach() < 0.0).any().item()):
            raise ValueError("radiance must be non-negative.")

    def pack(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        device = self.device if device is None else device
        two_sided = torch.tensor(
            [1.0 if self.two_sided else 0.0], dtype=dtype, device=device
        )
        packed = torch.cat(
            [
                self.center.to(device=device, dtype=dtype),
                self.radius.to(device=device, dtype=dtype),
                self.radiance.to(device=device, dtype=dtype),
                two_sided,
            ],
            dim=0,
        )
        return packed.contiguous()

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "SphereLight":
        dtype = self.dtype if dtype is None else dtype
        return SphereLight(
            center=self.center.to(device=device, dtype=dtype),
            radius=self.radius.to(device=device, dtype=dtype),
            radiance=self.radiance.to(device=device, dtype=dtype),
            two_sided=self.two_sided,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.name.lower(),
            "center": self.center.detach().clone(),
            "radius": self.radius.detach().clone(),
            "radiance": self.radiance.detach().clone(),
            "two_sided": self.two_sided,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.center = _as_vec3(
            state_dict.get("center", self.center),
            "center",
            device=self.device,
            dtype=self.dtype,
        )
        self.radius = _as_scalar(
            state_dict.get("radius", self.radius),
            "radius",
            device=self.device,
            dtype=self.dtype,
        )
        self.radiance = _as_vec3(
            state_dict.get("radiance", self.radiance),
            "radiance",
            device=self.device,
            dtype=self.dtype,
        )
        self.two_sided = bool(state_dict.get("two_sided", self.two_sided))
        self._validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "sphere",
            "center": self.center.detach().cpu().tolist(),
            "radius": float(self.radius.detach().cpu().item()),
            "radiance": self.radiance.detach().cpu().tolist(),
            "two_sided": self.two_sided,
        }


__all__ = [
    "LIGHT_TYPE_ENV",
    "LIGHT_TYPE_MESH",
    "LIGHT_TYPE_POINT",
    "LIGHT_TYPE_SPHERE",
    "PACKED_LIGHT_SIZE",
    "PACKED_MESH_LIGHT_SIZE",
    "Environment",
    "EnvironmentLight",
    "Light",
    "LightAliasTable",
    "LightType",
    "MeshLight",
    "MeshLightPack",
    "MeshLightTriangleAliasTable",
    "PointLight",
    "SGEnvironment",
    "SphereLight",
    "SphericalGaussianEnvironment",
    "build_light_alias_table",
    "build_mesh_light_triangle_alias_table",
    "create_environment",
    "environment_tensor_to_rgb_numpy",
    "estimate_light_power",
    "pack_mesh_lights",
    "save_environment_exr",
]
