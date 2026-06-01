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

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

import imageio
import imageio.plugins.freeimage as fi
import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_ALIAS_TABLE_SIZE = (64, 128)


@dataclass(frozen=True)
class EnvAliasTable:
    width: int
    height: int
    numCells: int
    prob: torch.Tensor
    alias: torch.Tensor
    pdf: torch.Tensor


def build_alias_table(weights: torch.Tensor, eps: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a Vose alias table from non-negative weights.

    Returns flattened ``prob`` and ``alias`` tensors suitable for the C-side
    EnvAliasTable fields.
    """
    weights_tensor = torch.as_tensor(weights)
    if weights_tensor.numel() == 0:
        raise ValueError("weights must contain at least one element.")
    if eps < 0.0:
        raise ValueError(f"eps must be non-negative, got {eps}.")

    device = weights_tensor.device
    flat_weights = weights_tensor.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    flat_weights = torch.where(
        torch.isfinite(flat_weights) & (flat_weights > 0.0),
        flat_weights,
        torch.zeros_like(flat_weights),
    )
    if eps > 0.0:
        flat_weights = flat_weights + eps

    num_entries = flat_weights.numel()
    total_weight = float(flat_weights.sum().item())
    if total_weight <= 0.0:
        probabilities = np.ones(num_entries, dtype=np.float32)
        aliases = np.arange(num_entries, dtype=np.int64)
    else:
        scaled_weights = flat_weights.numpy() * (float(num_entries) / total_weight)
        probabilities = np.empty(num_entries, dtype=np.float32)
        aliases = np.arange(num_entries, dtype=np.int64)

        small = [int(index) for index in np.nonzero(scaled_weights < 1.0)[0]]
        large = [int(index) for index in np.nonzero(scaled_weights >= 1.0)[0]]

        while small and large:
            small_index = small.pop()
            large_index = large.pop()

            probabilities[small_index] = np.float32(np.clip(scaled_weights[small_index], 0.0, 1.0))
            aliases[small_index] = large_index

            scaled_weights[large_index] -= 1.0 - scaled_weights[small_index]
            if scaled_weights[large_index] < 1.0:
                small.append(large_index)
            else:
                large.append(large_index)

        for index in small:
            probabilities[index] = 1.0
            aliases[index] = index
        for index in large:
            probabilities[index] = 1.0
            aliases[index] = index

    prob = torch.from_numpy(probabilities).to(device=device, dtype=torch.float32).contiguous()
    alias = torch.from_numpy(aliases.astype(np.int32)).to(device=device, dtype=torch.int32).contiguous()
    return prob, alias


def _environment_luminance(
    environment: torch.Tensor,
    luminance_weights: tuple[float, float, float],
) -> torch.Tensor:
    tensor = torch.as_tensor(environment)
    if tensor.ndim != 3 or tensor.shape[-1] < 3:
        raise ValueError(f"Environment must have shape [H, W, C>=3], got {tuple(tensor.shape)}")

    rgb = tensor[..., :3].detach().to(dtype=torch.float32)
    rgb = torch.where(torch.isfinite(rgb) & (rgb > 0.0), rgb, torch.zeros_like(rgb))
    weights = rgb.new_tensor(luminance_weights)
    return torch.sum(rgb * weights, dim=-1)


def _resize_2d_environment_for_alias_table(
    environment: torch.Tensor,
    target_size: Optional[tuple[int, int]],
) -> torch.Tensor:
    tensor = torch.as_tensor(environment).detach()
    if target_size is None:
        return tensor
    if len(target_size) != 2:
        raise ValueError(f"target_size must be a (height, width) pair, got {target_size}.")

    target_height, target_width = int(target_size[0]), int(target_size[1])
    if target_height <= 0 or target_width <= 0:
        raise ValueError(f"target_size entries must be positive, got {target_size}.")
    if tensor.ndim != 3 or tensor.shape[-1] < 3:
        raise ValueError(f"Environment must have shape [H, W, C>=3], got {tuple(tensor.shape)}")
    if tuple(tensor.shape[:2]) == (target_height, target_width):
        return tensor

    tensor = tensor.to(dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(tensor, size=(target_height, target_width), mode="area")
    return resized.squeeze(0).permute(1, 2, 0).contiguous()


def _equirect_solid_angles(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    row_edges = (torch.arange(height + 1, dtype=dtype, device=device) / float(height) - 0.5) * torch.pi
    row_solid_angles = (2.0 * torch.pi / float(width)) * (torch.sin(row_edges[1:]) - torch.sin(row_edges[:-1]))
    return torch.clamp(row_solid_angles, min=0.0).reshape(height, 1).expand(height, width).contiguous()


def _cubemap_solid_angles(face_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    edges = torch.linspace(-1.0, 1.0, face_size + 1, dtype=dtype, device=device)
    u0 = edges[:-1].reshape(1, face_size)
    u1 = edges[1:].reshape(1, face_size)
    v0 = edges[:-1].reshape(face_size, 1)
    v1 = edges[1:].reshape(face_size, 1)

    def area_element(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.atan2(u * v, torch.sqrt(u * u + v * v + 1.0))

    solid_angle = (
        area_element(u1, v1)
        - area_element(u0, v1)
        - area_element(u1, v0)
        + area_element(u0, v0)
    )
    return torch.clamp(solid_angle, min=0.0).repeat(6, 1).contiguous()


def _environment_solid_angles(
    height: int,
    width: int,
    environment_type: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if environment_type == "cube":
        if height != 6 * width:
            raise ValueError(f"Cubemap environment must have shape [6*N, N, C], got H={height}, W={width}.")
        return _cubemap_solid_angles(width, device, dtype)
    return _equirect_solid_angles(height, width, device, dtype)


def environment_importance_weights(
    environment: Optional[torch.Tensor],
    environment_type: str = "2d",
    include_solid_angle: bool = True,
    luminance_weights: tuple[float, float, float] = (0.2126, 0.7152, 0.0722),
) -> Optional[torch.Tensor]:
    """Build per-texel importance weights from environment luminance."""
    if environment is None:
        return None

    normalized_type = str(environment_type).lower()
    if normalized_type not in ("2d", "cube"):
        raise ValueError(f"environment_type must be one of ['2d', 'cube'], got '{environment_type}'.")

    weights = _environment_luminance(environment, luminance_weights)
    if not include_solid_angle:
        return weights.contiguous()

    height, width = weights.shape
    solid_angles = _environment_solid_angles(height, width, normalized_type, weights.device, weights.dtype)
    return (weights * solid_angles).contiguous()


def build_environment_alias_table(
    environment: Optional[torch.Tensor],
    environment_type: str = "2d",
    target_size: Optional[tuple[int, int]] = DEFAULT_ALIAS_TABLE_SIZE,
    luminance_weights: tuple[float, float, float] = (0.2126, 0.7152, 0.0722),
    eps: float = 0.0,
) -> Optional[EnvAliasTable]:
    """Build an EnvAliasTable for environment-map importance sampling."""
    if environment is None:
        return None
    if eps < 0.0:
        raise ValueError(f"eps must be non-negative, got {eps}.")

    normalized_type = str(environment_type).lower()
    if normalized_type not in ("2d", "cube"):
        raise ValueError(f"environment_type must be one of ['2d', 'cube'], got '{environment_type}'.")

    if normalized_type == "2d":
        environment = _resize_2d_environment_for_alias_table(environment, target_size)

    luminance = _environment_luminance(environment, luminance_weights)
    if eps > 0.0:
        luminance = luminance + eps

    height, width = luminance.shape
    solid_angles = _environment_solid_angles(height, width, normalized_type, luminance.device, luminance.dtype)
    sample_weights = (luminance * solid_angles).contiguous()

    total_weight = sample_weights.sum()
    if not bool(torch.isfinite(total_weight)) or float(total_weight.item()) <= 0.0:
        sample_weights = solid_angles
        total_weight = sample_weights.sum()

    pdf = torch.where(
        solid_angles > 0.0,
        sample_weights / torch.clamp(total_weight * solid_angles, min=torch.finfo(sample_weights.dtype).tiny),
        torch.zeros_like(sample_weights),
    )
    prob, alias = build_alias_table(sample_weights)
    return EnvAliasTable(
        width=int(width),
        height=int(height),
        numCells=int(height * width),
        prob=prob,
        alias=alias,
        pdf=pdf.reshape(-1).to(dtype=torch.float32).contiguous(),
    )


def environment_tensor_to_rgb_numpy(environment: torch.Tensor) -> np.ndarray:
    environment = environment.detach()
    if environment.ndim != 3 or environment.shape[-1] < 3:
        raise ValueError(f"Environment must have shape [H, W, C>=3], got {tuple(environment.shape)}")

    rgb = environment[..., :3].detach().cpu().numpy()
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    return rgb.astype(np.float32, copy=False)


def save_environment_exr(environment: Optional[torch.Tensor], output_path: str) -> Optional[str]:
    if environment is None:
        return None

    rgb = environment_tensor_to_rgb_numpy(environment)
    imageio.v2.imwrite(output_path, rgb)
    return output_path


class Environment:
    """Load environment maps and expose them as 4-channel torch tensors.

    This is the lightweight model-side version of the playground environment
    helper. It intentionally does not do tonemapping; loaded HDR/EXR values are
    kept linear and only padded with an alpha channel for CUDA texture upload.
    When log-exp optimization is enabled, ``self.environment`` stores
    log-radiance parameters and ``get_environment()`` returns exp-activated
    radiance. Linear optimization stores and returns the texels directly.
    """

    FIXED_ENVIRONMENT_OPTIONS = ["Model-Background", "Black", "White"]
    ENVIRONMENT_EXTENSIONS = (".hdr", ".exr", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
    ENVIRONMENT_TYPE_OPTIONS = ["2d", "cube"]
    CUBEMAP_FACE_NAMES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
    CUBEMAP_FACE_ALIASES = (
        ("+x", "posx", "px", "right"),
        ("-x", "negx", "nx", "left"),
        ("+y", "posy", "py", "top", "up"),
        ("-y", "negy", "ny", "bottom", "down"),
        ("+z", "posz", "pz", "front"),
        ("-z", "negz", "nz", "back"),
    )
    DEFAULT_ENVIRONMENT_SIZE = (64, 128)
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
        self.environment_type = self._normalize_environment_type(environment_type)
        self.environment_parameterization = self._normalize_environment_parameterization(parameterization)
        self.optimize_environment = bool(optimize_environment)
        self.intensity = 1.0

        self.current_name = "Model-Background"
        self.environment = None
        self._hdr_data = None
        self.environment_offset = [0.0, 0.0]

        self.available_environments = [option for option in self.FIXED_ENVIRONMENT_OPTIONS]
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
        return self.environment_parameterization == self.LOG_ENVIRONMENT_PARAMETERIZATION

    def _as_environment_tensor(self, environment: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(environment, dtype=torch.float32, device=self.device).contiguous()
        if tensor.dim() != 3 or tensor.size(-1) != 4:
            raise ValueError(f"environment must have shape [H, W, 4], got {tuple(tensor.shape)}")
        return tensor

    def _set_environment_parameter(self, environment: torch.Tensor, parameterization: Optional[str] = None) -> None:
        if parameterization is not None:
            self.environment_parameterization = self._normalize_environment_parameterization(parameterization)
        tensor = self._as_environment_tensor(environment)
        if self.optimize_environment:
            self.environment = torch.nn.Parameter(tensor.detach().clone(), requires_grad=True)
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
            self.environment = torch.nn.Parameter(tensor.detach().clone(), requires_grad=True)
        elif self.optimize_environment:
            self.environment = torch.nn.Parameter(tensor.detach().clone(), requires_grad=True)
        else:
            self.environment = tensor.detach()

    def configure_optimization(self, enabled: bool) -> None:
        environment = self.get_environment()
        self.optimize_environment = bool(enabled)
        if environment is not None:
            self._set_environment_tensor(environment)

    @classmethod
    def _normalize_environment_type(cls, environment_type: str) -> str:
        normalized = str(environment_type).lower()
        if normalized not in cls.ENVIRONMENT_TYPE_OPTIONS:
            raise ValueError(
                f"environment_type must be one of {cls.ENVIRONMENT_TYPE_OPTIONS}, got '{environment_type}'."
            )
        return normalized

    @classmethod
    def _normalize_environment_parameterization(cls, parameterization: str) -> str:
        normalized = str(parameterization).lower()
        options = (cls.LINEAR_ENVIRONMENT_PARAMETERIZATION, cls.LOG_ENVIRONMENT_PARAMETERIZATION)
        if normalized not in options:
            raise ValueError(f"environment.parameterization must be one of {options}, got '{parameterization}'.")
        return normalized

    @classmethod
    def _list_environments(cls, folder: str) -> list[str]:
        return [
            name
            for name in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, name)) or name.lower().endswith(cls.ENVIRONMENT_EXTENSIONS)
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
            raise ValueError(f"Environment map must have shape HxW or HxWxC, got {rgb.shape}.")
        if rgb.shape[-1] == 1:
            rgb = np.repeat(rgb, 3, axis=-1)
        elif rgb.shape[-1] > 3:
            rgb = rgb[..., :3]
        elif rgb.shape[-1] != 3:
            raise ValueError(f"Environment map must have 1, 3, or 4 channels, got {rgb.shape[-1]}.")

        if np.issubdtype(rgb.dtype, np.integer):
            rgb = rgb.astype(np.float32) / np.iinfo(rgb.dtype).max
        else:
            rgb = rgb.astype(np.float32, copy=False)

        return np.maximum(np.nan_to_num(rgb, nan=0.0, neginf=0.0), 0.0)

    @classmethod
    def _prepare_cubemap(cls, rgb: np.ndarray) -> np.ndarray:
        if rgb.ndim == 4:
            if rgb.shape[0] != 6 or rgb.shape[1] != rgb.shape[2]:
                raise ValueError(f"Cubemap array must have shape 6xNxNxC, got {rgb.shape}.")
            return np.concatenate([rgb[face] for face in range(6)], axis=0)

        height, width, _ = rgb.shape
        if height == 6 * width:
            return rgb
        if width == 6 * height:
            return np.concatenate([rgb[:, face * height : (face + 1) * height] for face in range(6)], axis=0)

        raise ValueError(
            "Cubemap must be a vertical strip [6*N, N, C], a horizontal strip [N, 6*N, C], "
            f"or six square faces; got {rgb.shape}."
        )

    def _prepare_environment_data(self, data: np.ndarray) -> np.ndarray:
        rgb = self._prepare_rgb(data)
        if self.environment_type == "cube":
            rgb = self._prepare_cubemap(rgb)
        return rgb

    @classmethod
    def _find_cubemap_face_paths(cls, folder: str) -> list[str]:
        files = [
            name
            for name in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, name)) and name.lower().endswith(cls.ENVIRONMENT_EXTENSIONS)
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

    def load_cubemap_files(self, face_paths: Sequence[str] | Mapping[str, str]) -> torch.Tensor:
        """Load six square cubemap face files in +X, -X, +Y, -Y, +Z, -Z order."""
        if isinstance(face_paths, Mapping):
            face_paths = [face_paths[name] for name in self.CUBEMAP_FACE_NAMES]
        if len(face_paths) != 6:
            raise ValueError(f"Cubemap loading requires six face files, got {len(face_paths)}.")

        faces = [self._prepare_rgb(self._read_environment_file(path)) for path in face_paths]
        face_size = faces[0].shape[0]
        for face_name, face in zip(self.CUBEMAP_FACE_NAMES, faces):
            if face.shape[0] != face.shape[1]:
                raise ValueError(f"Cubemap face {face_name} must be square, got {face.shape}.")
            if face.shape[:2] != (face_size, face_size):
                raise ValueError(
                    f"Cubemap face {face_name} shape {face.shape[:2]} does not match {face_size}x{face_size}."
                )

        self.environment_type = "cube"
        self.path = None
        self.folder = os.path.commonpath([os.path.dirname(os.path.abspath(path)) for path in face_paths])
        self.current_name = "Cubemap-Faces"
        self._hdr_data = self._prepare_cubemap(np.stack(faces, axis=0))
        self._update()
        return self.get_environment()

    def load_path(self, environment_path: str) -> torch.Tensor:
        """Load an environment map from an explicit file path."""
        if os.path.isdir(environment_path):
            if self.environment_type != "cube":
                raise ValueError("Directory loading is only supported for cubemaps.")
            environment = self.load_cubemap_files(self._find_cubemap_face_paths(environment_path))
            self.path = environment_path
            self.current_name = os.path.basename(os.path.normpath(environment_path))
            return environment

        if not os.path.isfile(environment_path):
            raise FileNotFoundError(f"Environment map not found: {environment_path}")

        self.path = environment_path
        self.folder = os.path.dirname(environment_path)
        self.current_name = os.path.basename(environment_path)
        self._hdr_data = self._prepare_environment_data(self._read_environment_file(environment_path))
        self._update()
        return self.get_environment()

    def load_file(self, environment_path: str) -> torch.Tensor:
        return self.load_path(environment_path)

    def _load_hdr(self, environment_name: Optional[str] = None) -> Optional[torch.Tensor]:
        """Load an environment map by name from ``self.folder``."""
        if not self.available_environments or environment_name in self.FIXED_ENVIRONMENT_OPTIONS:
            self.environment = None
            return None

        if environment_name not in self.available_environments:
            raise ValueError(f"Environment map {self.folder}{os.path.sep}{environment_name} not found.")

        if environment_name != self.current_name:
            environment_path = os.path.join(self.folder, environment_name)
            if os.path.isdir(environment_path):
                self.load_path(environment_path)
            else:
                self._hdr_data = self._prepare_environment_data(self._read_environment_file(environment_path))
                self._update()

        return self.get_environment()

    def _constant_environment(self, value: float) -> torch.Tensor:
        if self.environment_type == "cube":
            height = 6 * self.DEFAULT_CUBEMAP_FACE_SIZE
            width = self.DEFAULT_CUBEMAP_FACE_SIZE
        else:
            height, width = self.DEFAULT_ENVIRONMENT_SIZE
        environment = torch.full([height, width, 4], value * self.intensity, dtype=torch.float32, device=self.device)
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
        environment = torch.as_tensor(self._hdr_data, dtype=torch.float32, device=self.device).contiguous()
        environment = environment * self.intensity
        pad = environment.new_ones(environment.shape[0], environment.shape[1], 1)
        self._set_environment_tensor(torch.cat([environment, pad], dim=-1))

    def get_environment_parameter(self) -> Optional[torch.Tensor]:
        return self.environment

    def get_environment(self) -> Optional[torch.Tensor]:
        if self.environment is None:
            return None
        if self.optimize_environment and self._uses_log_parameterization():
            return self._internal_to_actual(self.environment)
        return self.environment

    def get_environment_offset(self) -> torch.Tensor:
        return torch.tensor(self.environment_offset, dtype=torch.float32, device=self.device)

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
            "environment": None if self.environment is None else self.environment.detach().clone(),
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
        self.environment_offset = list(state_dict.get("environment_offset", self.environment_offset))
        self.environment_type = self._normalize_environment_type(
            state_dict.get("environment_type", self.environment_type)
        )
        self.optimize_environment = bool(state_dict.get("optimize_environment", self.optimize_environment))
        self.intensity = float(state_dict.get("intensity", self.intensity))
        environment = state_dict.get("environment")
        parameterization = state_dict.get("environment_parameterization", self.LINEAR_ENVIRONMENT_PARAMETERIZATION)
        if environment is None:
            self.environment_parameterization = self._normalize_environment_parameterization(parameterization)
            self.environment = None
        else:
            self._set_environment_parameter(environment, parameterization)
        self._hdr_data = None
