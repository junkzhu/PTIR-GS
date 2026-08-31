# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for placing one Gaussian checkpoint inside another scene.

The renderer stores geometry and PBR attributes directly on every Gaussian.  A
scene composition is therefore an inference checkpoint whose fields are the
concatenation of a base scene and one or more transformed object checkpoints.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


GAUSSIAN_FIELDS = (
    "positions",
    "rotation",
    "scale",
    "density",
    "features_albedo",
    "features_specular",
    "shading_normal",
    "material_albedo",
    "material_roughness",
    "material_metallic",
)

_ROUGHNESS_MIN = 0.09
_ROUGHNESS_RANGE = 0.9
_MATERIAL_EPS = 1.0e-6


@dataclass(frozen=True)
class Placement:
    """Similarity transform that places an object's bottom anchor on a surface."""

    contact_point: Sequence[float]
    up: Sequence[float] = (0.0, 0.0, 1.0)
    scale: float = 1.0
    yaw_degrees: float = 0.0
    lift: float = 0.0
    anchor_quantile: float = 0.002

    def validate(self) -> None:
        if len(self.contact_point) != 3 or len(self.up) != 3:
            raise ValueError("contact_point and up must be 3D vectors")
        values = [*self.contact_point, *self.up, self.scale, self.yaw_degrees, self.lift]
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("placement values must be finite")
        if self.scale <= 0.0:
            raise ValueError(f"scale must be positive, got {self.scale}")
        if not 0.0 <= self.anchor_quantile < 0.5:
            raise ValueError(
                f"anchor_quantile must be in [0, 0.5), got {self.anchor_quantile}"
            )
        if math.sqrt(sum(float(value) ** 2 for value in self.up)) <= 1.0e-12:
            raise ValueError("up must be non-zero")


@dataclass(frozen=True)
class MaterialOverride:
    """Optional constant/tint overrides in activated PBR material space."""

    metallic: float | None = None
    roughness: float | None = None
    albedo: Sequence[float] | None = None
    albedo_tint: Sequence[float] | None = None

    def validate(self) -> None:
        if self.metallic is not None and not 0.0 <= self.metallic <= 1.0:
            raise ValueError(f"metallic must be in [0, 1], got {self.metallic}")
        if self.roughness is not None and not _ROUGHNESS_MIN < self.roughness < 0.99:
            raise ValueError(
                f"roughness must be in ({_ROUGHNESS_MIN}, 0.99), got {self.roughness}"
            )
        for name, value in (("albedo", self.albedo), ("albedo_tint", self.albedo_tint)):
            if value is None:
                continue
            if len(value) != 3 or not all(math.isfinite(float(channel)) for channel in value):
                raise ValueError(f"{name} must be a finite RGB triplet")
            if name == "albedo" and not all(0.0 <= float(channel) <= 1.0 for channel in value):
                raise ValueError("albedo channels must be in [0, 1]")
            if name == "albedo_tint" and not all(float(channel) >= 0.0 for channel in value):
                raise ValueError("albedo_tint channels must be non-negative")


def _as_vector(
    value: Sequence[float], *, dtype: torch.dtype, device: torch.device | str
) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device)


def quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Hamilton product for wxyz quaternions, with normal PyTorch broadcasting."""

    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert one wxyz quaternion to a 3x3 rotation matrix."""

    quaternion = quaternion / torch.linalg.vector_norm(quaternion)
    w, x, y, z = quaternion.unbind()
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        )
    ).reshape(3, 3)


def quaternion_align_vectors(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the shortest-arc wxyz rotation from ``source`` to ``target``."""

    source = source / torch.linalg.vector_norm(source)
    target = target / torch.linalg.vector_norm(target)
    dot = torch.dot(source, target).clamp(-1.0, 1.0)
    if float(dot) < -1.0 + 1.0e-6:
        # Pick the basis axis least parallel to source for a stable 180-degree turn.
        basis = torch.zeros_like(source)
        basis[int(torch.argmin(source.abs()))] = 1.0
        axis = torch.linalg.cross(source, basis)
        axis = axis / torch.linalg.vector_norm(axis)
        return torch.cat((torch.zeros(1, dtype=source.dtype, device=source.device), axis))
    cross = torch.linalg.cross(source, target)
    quaternion = torch.cat(((1.0 + dot).reshape(1), cross))
    return quaternion / torch.linalg.vector_norm(quaternion)


def placement_rotation(
    up: Sequence[float], yaw_degrees: float, *, dtype: torch.dtype, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return quaternion/matrix mapping local +Z to the requested surface normal."""

    local_up = torch.tensor((0.0, 0.0, 1.0), dtype=dtype, device=device)
    target_up = _as_vector(up, dtype=dtype, device=device)
    target_up = target_up / torch.linalg.vector_norm(target_up)
    align = quaternion_align_vectors(local_up, target_up)
    yaw_radians = math.radians(yaw_degrees)
    yaw = torch.tensor(
        (math.cos(yaw_radians / 2.0), 0.0, 0.0, math.sin(yaw_radians / 2.0)),
        dtype=dtype,
        device=device,
    )
    rotation = quaternion_multiply(align, yaw)
    rotation = rotation / torch.linalg.vector_norm(rotation)
    return rotation, quaternion_to_matrix(rotation)


def robust_bottom_anchor(positions: torch.Tensor, quantile: float = 0.002) -> torch.Tensor:
    """Estimate bottom-center while ignoring a small fraction of floaters."""

    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        raise ValueError(f"positions must have shape [N, 3], got {tuple(positions.shape)}")
    if not 0.0 <= quantile < 0.5:
        raise ValueError(f"quantile must be in [0, 0.5), got {quantile}")
    lower = torch.quantile(positions, quantile, dim=0)
    upper = torch.quantile(positions, 1.0 - quantile, dim=0)
    return torch.stack(((lower[0] + upper[0]) * 0.5, (lower[1] + upper[1]) * 0.5, lower[2]))


def _logit(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(_MATERIAL_EPS, 1.0 - _MATERIAL_EPS)
    return torch.log(value / (1.0 - value))


def _roughness_inverse(value: torch.Tensor) -> torch.Tensor:
    scaled = ((value - _ROUGHNESS_MIN) / _ROUGHNESS_RANGE).clamp(
        _MATERIAL_EPS, 1.0 - _MATERIAL_EPS
    )
    return torch.log(scaled / (1.0 - scaled))


def apply_material_override(
    checkpoint: dict[str, Any], override: MaterialOverride
) -> dict[str, Any]:
    """Apply material values without mutating ``checkpoint``."""

    override.validate()
    result = dict(checkpoint)
    reference = checkpoint["material_albedo"]
    dtype, device = reference.dtype, reference.device
    count = reference.shape[0]

    if override.albedo is not None:
        albedo = _as_vector(override.albedo, dtype=dtype, device=device).expand(count, 3)
        result["material_albedo"] = torch.nn.Parameter(_logit(albedo), requires_grad=False)
    elif override.albedo_tint is not None:
        tint = _as_vector(override.albedo_tint, dtype=dtype, device=device)
        albedo = torch.sigmoid(reference.detach()) * tint
        result["material_albedo"] = torch.nn.Parameter(_logit(albedo), requires_grad=False)

    if override.roughness is not None:
        roughness = torch.full(
            (count, 1), override.roughness, dtype=dtype, device=device
        )
        result["material_roughness"] = torch.nn.Parameter(
            _roughness_inverse(roughness), requires_grad=False
        )
    if override.metallic is not None:
        metallic = torch.full((count, 1), override.metallic, dtype=dtype, device=device)
        result["material_metallic"] = torch.nn.Parameter(
            _logit(metallic), requires_grad=False
        )
    return result


def transform_checkpoint(
    checkpoint: Mapping[str, Any], placement: Placement
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Transform all geometry-bearing fields of an object checkpoint on CPU."""

    placement.validate()
    _validate_gaussian_fields(checkpoint, "object")
    result = dict(checkpoint)
    positions = checkpoint["positions"].detach()
    dtype, device = positions.dtype, positions.device
    anchor = robust_bottom_anchor(positions, placement.anchor_quantile)
    rotation_quaternion, rotation_matrix = placement_rotation(
        placement.up, placement.yaw_degrees, dtype=dtype, device=device
    )
    up = _as_vector(placement.up, dtype=dtype, device=device)
    up = up / torch.linalg.vector_norm(up)
    contact = _as_vector(placement.contact_point, dtype=dtype, device=device)
    translation = contact + placement.lift * up - placement.scale * (rotation_matrix @ anchor)

    transformed_positions = placement.scale * (positions @ rotation_matrix.T) + translation
    transformed_rotation = quaternion_multiply(
        rotation_quaternion.expand_as(checkpoint["rotation"]), checkpoint["rotation"].detach()
    )
    transformed_rotation = torch.nn.functional.normalize(transformed_rotation, dim=-1)
    transformed_normals = checkpoint["shading_normal"].detach() @ rotation_matrix.T

    result["positions"] = torch.nn.Parameter(transformed_positions, requires_grad=False)
    result["rotation"] = torch.nn.Parameter(transformed_rotation, requires_grad=False)
    result["scale"] = torch.nn.Parameter(
        checkpoint["scale"].detach() + math.log(placement.scale), requires_grad=False
    )
    result["shading_normal"] = torch.nn.Parameter(transformed_normals, requires_grad=False)

    similarity = torch.eye(4, dtype=dtype, device=device)
    similarity[:3, :3] = placement.scale * rotation_matrix
    similarity[:3, 3] = translation
    metadata = {
        "contact_point": [float(value) for value in contact.cpu()],
        "surface_up": [float(value) for value in up.cpu()],
        "bottom_anchor_local": [float(value) for value in anchor.cpu()],
        "scale": float(placement.scale),
        "yaw_degrees": float(placement.yaw_degrees),
        "lift": float(placement.lift),
        "object_to_scene": similarity.cpu().tolist(),
        "bounds_scene": {
            "min": [float(value) for value in transformed_positions.amin(dim=0).cpu()],
            "max": [float(value) for value in transformed_positions.amax(dim=0).cpu()],
        },
    }
    return result, metadata


def _validate_gaussian_fields(checkpoint: Mapping[str, Any], label: str) -> None:
    missing = [name for name in GAUSSIAN_FIELDS if not torch.is_tensor(checkpoint.get(name))]
    if missing:
        raise ValueError(f"{label} checkpoint is missing tensor field(s): {', '.join(missing)}")
    count = checkpoint["positions"].shape[0]
    mismatched = [
        name for name in GAUSSIAN_FIELDS if checkpoint[name].ndim == 0 or checkpoint[name].shape[0] != count
    ]
    if mismatched:
        raise ValueError(
            f"{label} checkpoint fields do not share Gaussian count {count}: {', '.join(mismatched)}"
        )


def _compatible_features(base: Mapping[str, Any], obj: Mapping[str, Any]) -> None:
    for name in GAUSSIAN_FIELDS:
        if base[name].shape[1:] != obj[name].shape[1:]:
            raise ValueError(
                f"incompatible field {name}: base {tuple(base[name].shape)}, "
                f"object {tuple(obj[name].shape)}"
            )
    for name in ("max_n_features", "n_active_features"):
        if int(base[name]) != int(obj[name]):
            raise ValueError(
                f"incompatible {name}: base {int(base[name])}, object {int(obj[name])}"
            )


def compose_checkpoints(
    base: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
    *,
    name: str,
    composition_metadata: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Concatenate a base checkpoint with already transformed object checkpoints."""

    if not objects:
        raise ValueError("at least one object checkpoint is required")
    _validate_gaussian_fields(base, "base")
    for index, obj in enumerate(objects):
        _validate_gaussian_fields(obj, f"object {index}")
        _compatible_features(base, obj)

    result: dict[str, Any] = {}
    for field in GAUSSIAN_FIELDS:
        tensors = [base[field].detach(), *(obj[field].detach() for obj in objects)]
        result[field] = torch.nn.Parameter(torch.cat(tensors, dim=0), requires_grad=False)

    for field in (
        "background",
        "n_active_features",
        "max_n_features",
        "progressive_training",
        "scene_extent",
        "feature_dim_increase_interval",
        "feature_dim_increase_step",
        "global_step",
        "epoch",
        "environment_state",
        "albedo_rescale_rgb",
    ):
        if field in base:
            result[field] = copy.deepcopy(base[field])

    conf = copy.deepcopy(base["config"])
    conf.experiment_name = name
    conf.render.enable_metallic = True
    # This is an inference artifact. Keeping the training switch disabled avoids
    # accidentally optimizing the constant metallic override on resume.
    conf.model.optimize_material_metallic = False
    result["config"] = conf
    result["composition"] = {
        "version": 1,
        "name": name,
        "base_gaussians": int(base["positions"].shape[0]),
        "object_gaussians": [int(obj["positions"].shape[0]) for obj in objects],
        "total_gaussians": int(result["positions"].shape[0]),
        "objects": list(composition_metadata or ()),
    }
    return result


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint on CPU, using mmap when supported by PyTorch."""

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    try:
        return torch.load(
            checkpoint_path, map_location="cpu", weights_only=False, mmap=True
        )
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def save_checkpoint(checkpoint: Mapping[str, Any], path: str | Path, *, overwrite: bool = False) -> Path:
    """Atomically publish a composed checkpoint after writing beside its target."""

    output_path = Path(path).expanduser().resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists (pass overwrite=True): {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    try:
        torch.save(dict(checkpoint), temporary_path)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path


__all__ = [
    "GAUSSIAN_FIELDS",
    "MaterialOverride",
    "Placement",
    "apply_material_override",
    "compose_checkpoints",
    "load_checkpoint",
    "placement_rotation",
    "quaternion_align_vectors",
    "quaternion_multiply",
    "quaternion_to_matrix",
    "robust_bottom_anchor",
    "save_checkpoint",
    "transform_checkpoint",
]
