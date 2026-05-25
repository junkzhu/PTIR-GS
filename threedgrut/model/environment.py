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
from typing import Optional

import imageio
import imageio.plugins.freeimage as fi
import numpy as np
import torch


def environment_tensor_to_rgb_numpy(environment: torch.Tensor) -> np.ndarray:
    environment = environment.detach()
    if environment.ndim != 3 or environment.shape[-1] < 3:
        raise ValueError(f"Environment must have shape [H, W, C>=3], got {tuple(environment.shape)}")

    rgb = environment[..., :3].detach().cpu().numpy()
    rgb = np.maximum(np.nan_to_num(rgb, nan=0.0, neginf=0.0), 0.0)
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
    When optimization is enabled, ``self.environment`` stores log-radiance
    parameters and ``get_environment()`` returns the exp-activated radiance.
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
    ):
        self.device = device
        self.path = path
        self.folder = None
        self.environment_type = self._normalize_environment_type(environment_type)
        self.optimize_environment = bool(optimize_environment)

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

    def _as_environment_tensor(self, environment: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(environment, dtype=torch.float32, device=self.device).contiguous()
        if tensor.dim() != 3 or tensor.size(-1) != 4:
            raise ValueError(f"environment must have shape [H, W, 4], got {tuple(tensor.shape)}")
        return tensor

    def _set_environment_parameter(self, environment: torch.Tensor) -> None:
        tensor = self._as_environment_tensor(environment)
        if self.optimize_environment:
            self.environment = torch.nn.Parameter(tensor.detach().clone(), requires_grad=True)
        else:
            self.environment = self._internal_to_actual(tensor).detach()

    def _set_environment_tensor(self, environment: Optional[torch.Tensor]) -> None:
        if environment is None:
            self.environment = None
            return

        tensor = self._as_environment_tensor(environment)
        if self.optimize_environment:
            tensor = self._actual_to_internal(tensor)
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
        environment = torch.full([height, width, 4], value, dtype=torch.float32, device=self.device)
        environment[..., 3] = 1.0
        return environment

    def init_environment(self, value: float = 1.5) -> torch.Tensor:
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
        pad = environment.new_ones(environment.shape[0], environment.shape[1], 1)
        self._set_environment_tensor(torch.cat([environment, pad], dim=-1))

    def get_environment_parameter(self) -> Optional[torch.Tensor]:
        return self.environment

    def get_environment(self) -> Optional[torch.Tensor]:
        if self.environment is None:
            return None
        if self.optimize_environment:
            return self._internal_to_actual(self.environment)
        return self.environment

    def get_environment_offset(self) -> torch.Tensor:
        return torch.tensor(self.environment_offset, dtype=torch.float32, device=self.device)

    def is_ignore_environment(self) -> bool:
        return self.current_name == "Model-Background"

    def state_dict(self) -> dict:
        return {
            "current_name": self.current_name,
            "path": self.path,
            "environment_offset": list(self.environment_offset),
            "environment": None if self.environment is None else self.environment.detach().clone(),
            "environment_parameterization": (
                self.LOG_ENVIRONMENT_PARAMETERIZATION
                if self.optimize_environment
                else self.LINEAR_ENVIRONMENT_PARAMETERIZATION
            ),
            "environment_type": self.environment_type,
            "optimize_environment": self.optimize_environment,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.current_name = state_dict.get("current_name", self.current_name)
        self.path = state_dict.get("path", self.path)
        self.environment_offset = list(state_dict.get("environment_offset", self.environment_offset))
        self.environment_type = self._normalize_environment_type(
            state_dict.get("environment_type", self.environment_type)
        )
        self.optimize_environment = bool(state_dict.get("optimize_environment", self.optimize_environment))
        environment = state_dict.get("environment")
        parameterization = state_dict.get("environment_parameterization", self.LINEAR_ENVIRONMENT_PARAMETERIZATION)
        if environment is None:
            self.environment = None
        elif parameterization == self.LOG_ENVIRONMENT_PARAMETERIZATION:
            self._set_environment_parameter(environment)
        elif parameterization == self.LINEAR_ENVIRONMENT_PARAMETERIZATION:
            self._set_environment_tensor(environment)
        else:
            raise ValueError(f"Unknown environment parameterization '{parameterization}'.")
        self._hdr_data = None
