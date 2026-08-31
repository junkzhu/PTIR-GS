# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load camera and emissive-light animation exported for relighting demos."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from threedgrut.datasets.protocols import Batch
from threedgrut.model.light import Environment, MeshLight, SphereLight


DEFAULT_VIDEO_FPS = 30


def encode_aov_videos(
    output_paths: dict[str, Path], fps: int = DEFAULT_VIDEO_FPS
) -> dict[str, Path]:
    """Encode every non-empty AOV image directory as an H.264 video."""

    if fps <= 0:
        raise ValueError(f"Video FPS must be positive, got {fps}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode relighting AOV videos")

    videos = {}
    for name, frames_dir in output_paths.items():
        if not any(frames_dir.glob("*.png")):
            continue
        video_path = frames_dir.parent / f"{name}.mp4"
        command = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-pattern_type",
            "glob",
            "-i",
            str(frames_dir / "*.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as error:
            message = error.stderr.strip() or f"ffmpeg exited with {error.returncode}"
            raise RuntimeError(f"Failed to encode {name} AOV video: {message}") from error
        videos[name] = video_path
    return videos


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Relighting sequence file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _matrix4(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-5):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    return matrix


def _frame_map(frames: Any, name: str) -> dict[int, dict[str, Any]]:
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{name} must contain at least one frame")
    result = {}
    for entry in frames:
        if not isinstance(entry, dict) or "frame" not in entry or "transform" not in entry:
            raise ValueError(f"Each entry in {name} must contain frame and transform")
        frame_id = int(entry["frame"])
        if frame_id in result:
            raise ValueError(f"Duplicate frame {frame_id} in {name}")
        _matrix4(entry["transform"], f"{name}[{frame_id}].transform")
        result[frame_id] = entry
    return result


def _parse_hex_color(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, str) or not value.startswith("#"):
        raise ValueError(f"Expected an RGB hex color, got {value!r}")
    digits = value[1:]
    if len(digits) == 3:
        digits = "".join(character * 2 for character in digits)
    if len(digits) != 6:
        raise ValueError(f"Expected #RGB or #RRGGBB, got {value!r}")
    try:
        return tuple(int(digits[offset : offset + 2], 16) / 255.0 for offset in (0, 2, 4))
    except ValueError as error:
        raise ValueError(f"Invalid RGB hex color: {value!r}") from error


def _mesh_is_closed(faces: np.ndarray) -> bool:
    """Return true when every undirected edge has exactly two incident faces."""

    if faces.shape[0] == 0:
        return False
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(counts.size > 0 and np.all(counts == 2))


def _mesh_arrays(geometry: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(geometry, dict) or geometry.get("type") != "mesh":
        raise ValueError(f"{name}.geometry must be a mesh")
    vertices = np.asarray(geometry.get("vertices"), dtype=np.float64)
    faces = np.asarray(geometry.get("faces"), dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,):
        raise ValueError(f"{name} mesh vertices must have shape [V, 3]")
    if faces.ndim != 2 or faces.shape[1:] != (3,):
        raise ValueError(f"{name} mesh faces must have shape [T, 3]")
    return vertices, faces


def _surface_material(value: Any, name: str) -> tuple[np.ndarray, float, float]:
    if not isinstance(value, dict):
        raise ValueError(f"{name}.material must be a JSON object")
    albedo = np.asarray(value.get("albedo", (0.8, 0.8, 0.8)), dtype=np.float64)
    roughness = float(value.get("roughness", 0.5))
    metallic = float(value.get("metallic", 0.0))
    if albedo.shape != (3,) or not np.isfinite(albedo).all() or np.any((albedo < 0.0) | (albedo > 1.0)):
        raise ValueError(f"{name}.material.albedo must be an RGB triplet in [0, 1]")
    if not np.isfinite([roughness, metallic]).all() or not (0.0 <= roughness <= 1.0 and 0.0 <= metallic <= 1.0):
        raise ValueError(f"{name} roughness and metallic must be in [0, 1]")
    return albedo, roughness, metallic


@dataclass(frozen=True)
class RelightingIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def values(self) -> list[float]:
        return [self.fx, self.fy, self.cx, self.cy]


class RelightingSequence:
    """Validated relighting sequence converted to 3DGRT's RDF world space.

    ``gaussians_transform.json`` describes Gaussian placement in Blender and is
    intentionally not part of this camera/light rendering path.
    """

    CAMERA_FILE = "camera_seq.json"
    LIGHT_FILE = "light_seq.json"
    MESH_FILE = "mesh_seq.json"
    GAUSSIANS_TRANSFORM_FILE = "gaussians_transform.json"

    def __init__(self, directory: str | Path, light_scale: float = 1.0) -> None:
        self.directory = Path(directory).expanduser().resolve()
        if not self.directory.is_dir():
            raise NotADirectoryError(f"Relighting sequence directory not found: {self.directory}")
        self.light_scale = float(light_scale)
        if not np.isfinite(self.light_scale) or self.light_scale < 0.0:
            raise ValueError(f"light_scale must be a finite non-negative number, got {light_scale}")

        self.camera_data = _load_json_object(self.directory / self.CAMERA_FILE)
        self.light_data = _load_json_object(self.directory / self.LIGHT_FILE)
        mesh_path = self.directory / self.MESH_FILE
        self.mesh_data = (
            _load_json_object(mesh_path) if mesh_path.is_file() else self.light_data
        )
        self._validate_headers()

        cameras = self.camera_data.get("cameras")
        if not isinstance(cameras, list) or len(cameras) != 1:
            count = len(cameras) if isinstance(cameras, list) else 0
            raise ValueError(f"Exactly one camera is currently supported, found {count}")
        self.camera = cameras[0]
        if not isinstance(self.camera, dict):
            raise ValueError("Camera entry must be a JSON object")
        self.camera_frames = _frame_map(self.camera.get("frames"), "camera.frames")
        self.frame_ids = sorted(self.camera_frames)
        self.intrinsics = self._load_intrinsics(self.camera.get("intrinsics"))

        lights = self.light_data.get("lights")
        if not isinstance(lights, list):
            raise ValueError("light_seq.json must contain a lights array")
        self.light_entries = lights
        self.light_frames = []
        for index, light in enumerate(lights):
            if not isinstance(light, dict):
                raise ValueError(f"lights[{index}] must be a JSON object")
            geometry = light.get("geometry")
            if isinstance(geometry, dict) and geometry.get("type") == "none":
                self.light_frames.append({})
                continue
            self.light_frames.append(
                _frame_map(light.get("frames"), f"lights[{index}].frames")
            )

        meshes = self.mesh_data.get("meshes", [])
        if not isinstance(meshes, list):
            raise ValueError("mesh_seq.json meshes must be an array")
        self.mesh_entries = meshes
        self.mesh_frames = []
        for index, mesh in enumerate(meshes):
            if not isinstance(mesh, dict):
                raise ValueError(f"meshes[{index}] must be a JSON object")
            _mesh_arrays(mesh.get("geometry"), f"meshes[{index}]")
            _surface_material(mesh.get("material", {}), f"meshes[{index}]")
            self.mesh_frames.append(
                _frame_map(mesh.get("frames"), f"meshes[{index}].frames")
            )

        envmap = self.light_data.get("envmap", {"type": "color", "color": "#000000"})
        if not isinstance(envmap, dict):
            raise ValueError("envmap must be a JSON object")
        self.environment_type = envmap.get("type", "color")
        self.environment_color = None
        self.environment_path = None
        if self.environment_type == "color":
            self.environment_color = _parse_hex_color(envmap.get("color", "#000000"))
        elif self.environment_type == "file":
            path_value = envmap.get("path")
            if not isinstance(path_value, str) or not path_value:
                raise ValueError("A file envmap must contain a non-empty path")
            self.environment_path = Path(path_value).expanduser()
            if not self.environment_path.is_absolute():
                raise ValueError(f"Environment map path must be absolute: {path_value}")
            if not self.environment_path.is_file():
                raise FileNotFoundError(f"Environment map not found: {self.environment_path}")
        else:
            raise ValueError(f"Unsupported envmap type: {self.environment_type!r}")

        transform_path = self.directory / self.GAUSSIANS_TRANSFORM_FILE
        self.has_gaussians_transform = transform_path.is_file()
        if self.coordinate_space == "blender":
            self.rdf_from_world = np.linalg.inv(
                _matrix4(self.camera_data.get("world_from_rdf"), "world_from_rdf")
            )
            self.rdf_from_camera = np.linalg.inv(
                _matrix4(self.camera_data.get("camera_from_rdf"), "camera_from_rdf")
            )
        else:
            self.rdf_from_world = np.eye(4, dtype=np.float64)
            self.rdf_from_camera = np.eye(4, dtype=np.float64)

    def _validate_headers(self) -> None:
        for name, data in ((self.CAMERA_FILE, self.camera_data), (self.LIGHT_FILE, self.light_data)):
            if data.get("version") != 1:
                raise ValueError(f"Unsupported {name} version: {data.get('version')!r}")
        self.coordinate_space = self.camera_data.get("coordinate_space")
        if self.coordinate_space not in ("blender", "rdf"):
            raise ValueError(
                f"Unsupported relighting coordinate space: {self.coordinate_space!r}"
            )
        if self.light_data.get("coordinate_space") != self.coordinate_space:
            raise ValueError("camera_seq.json and light_seq.json use different coordinate spaces")
        if self.mesh_data.get("version") != 1 or self.mesh_data.get("coordinate_space") != self.coordinate_space:
            raise ValueError("mesh_seq.json uses an incompatible version or coordinate space")
        if self.coordinate_space == "blender":
            camera_world_from_rdf = _matrix4(
                self.camera_data.get("world_from_rdf"), "camera.world_from_rdf"
            )
            light_world_from_rdf = _matrix4(
                self.light_data.get("world_from_rdf"), "light.world_from_rdf"
            )
            if not np.allclose(camera_world_from_rdf, light_world_from_rdf):
                raise ValueError("camera and light world_from_rdf matrices do not match")

    @staticmethod
    def _load_intrinsics(value: Any) -> RelightingIntrinsics:
        if not isinstance(value, dict):
            raise ValueError("Camera intrinsics must be a JSON object")
        if value.get("model") != "pinhole":
            raise ValueError(f"Only pinhole cameras are supported, got {value.get('model')!r}")
        intrinsics = RelightingIntrinsics(
            width=int(value["width"]),
            height=int(value["height"]),
            fx=float(value["fx"]),
            fy=float(value["fy"]),
            cx=float(value["cx"]),
            cy=float(value["cy"]),
        )
        if intrinsics.width <= 0 or intrinsics.height <= 0:
            raise ValueError("Camera width and height must be positive")
        if intrinsics.fx <= 0.0 or intrinsics.fy <= 0.0:
            raise ValueError("Camera focal lengths must be positive")
        return intrinsics

    @staticmethod
    def _select_frame(frames: dict[int, dict[str, Any]], frame_id: int) -> dict[str, Any]:
        if frame_id in frames:
            return frames[frame_id]
        frame_ids = sorted(frames)
        if not frame_ids:
            raise KeyError(f"No object transform for frame {frame_id}")
        right = int(np.searchsorted(frame_ids, frame_id))
        if right == 0 or right == len(frame_ids):
            return frames[frame_ids[0 if right == 0 else -1]]
        left_id, right_id = frame_ids[right - 1], frame_ids[right]
        alpha = (frame_id - left_id) / (right_id - left_id)
        left = _matrix4(frames[left_id]["transform"], "left keyframe")
        right = _matrix4(frames[right_id]["transform"], "right keyframe")
        transform = (1.0 - alpha) * left + alpha * right
        scale = (1.0 - alpha) * np.linalg.norm(left[:3, :3], axis=0)
        scale += alpha * np.linalg.norm(right[:3, :3], axis=0)
        u, _, vh = np.linalg.svd(transform[:3, :3])
        if np.linalg.det(u @ vh) < 0.0:
            u[:, -1] *= -1.0
        transform[:3, :3] = (u @ vh) * scale
        return {"frame": frame_id, "transform": transform.tolist()}

    def camera_to_world(self, frame_id: int) -> np.ndarray:
        frame = self.camera_frames[frame_id]
        scene_from_camera = _matrix4(
            frame["transform"], f"camera.frames[{frame_id}].transform"
        )
        if self.coordinate_space == "rdf":
            return scene_from_camera
        # Convert both the Blender world basis and the Blender camera-local
        # basis. The right-side inverse must not be used for non-camera objects.
        return self.rdf_from_world @ scene_from_camera @ self.rdf_from_camera

    def _object_to_rdf(self, transform: Any, name: str) -> np.ndarray:
        scene_from_object = _matrix4(transform, name)
        return (
            scene_from_object
            if self.coordinate_space == "rdf"
            else self.rdf_from_world @ scene_from_object
        )

    def make_lights(
        self, frame_id: int, device: torch.device | str
    ) -> list[SphereLight | MeshLight]:
        result = []
        for index, (light, frames) in enumerate(zip(self.light_entries, self.light_frames)):
            geometry = light.get("geometry")
            if isinstance(geometry, dict) and geometry.get("type") == "none":
                continue
            frame = self._select_frame(frames, frame_id)
            rdf_from_light = self._object_to_rdf(
                frame["transform"], f"lights[{index}].frames[{frame_id}].transform"
            )
            color = np.asarray(light.get("color"), dtype=np.float64)
            if color.shape != (3,) or not np.isfinite(color).all() or np.any(color < 0.0):
                raise ValueError(f"lights[{index}].color must be a non-negative RGB triplet")
            strength = float(light.get("strength", 1.0))
            if not np.isfinite(strength) or strength < 0.0:
                raise ValueError(f"lights[{index}].strength must be non-negative")
            radiance = color * strength * self.light_scale

            if not isinstance(geometry, dict):
                raise ValueError(f"lights[{index}].geometry must be a JSON object")
            geometry_type = geometry.get("type")
            if geometry_type == "sphere":
                center = rdf_from_light[:3, 3]
                # The export stores the final world-space radius. Applying the
                # object transform scale here would scale it a second time.
                radius = float(geometry["radius"])
                result.append(
                    SphereLight(
                        center=center,
                        radius=radius,
                        radiance=radiance,
                        two_sided=False,
                        device=device,
                    )
                )
            elif geometry_type == "mesh":
                vertices, faces = _mesh_arrays(geometry, f"lights[{index}]")
                vertices_h = np.concatenate(
                    [vertices, np.ones((vertices.shape[0], 1), dtype=np.float64)], axis=1
                )
                model_vertices = (rdf_from_light @ vertices_h.T).T[:, :3]
                result.append(
                    MeshLight(
                        vertices=model_vertices,
                        triangles=faces,
                        radiance=radiance,
                        # Closed emitters use their outward winding. Open area
                        # lights are two-sided because Blender exports in this
                        # dataset do not consistently encode emission direction
                        # in the triangle winding.
                        two_sided=not _mesh_is_closed(faces),
                        device=device,
                    )
                )
            else:
                raise ValueError(
                    f"Unsupported lights[{index}] geometry type: {geometry_type!r}"
                )

        for index, (mesh, frames) in enumerate(zip(self.mesh_entries, self.mesh_frames)):
            frame = self._select_frame(frames, frame_id)
            rdf_from_mesh = self._object_to_rdf(
                frame["transform"], f"meshes[{index}].frames[{frame_id}].transform"
            )
            vertices, faces = _mesh_arrays(mesh.get("geometry"), f"meshes[{index}]")
            vertices_h = np.concatenate(
                [vertices, np.ones((vertices.shape[0], 1), dtype=np.float64)], axis=1
            )
            albedo, roughness, metallic = _surface_material(
                mesh.get("material", {}), f"meshes[{index}]"
            )
            result.append(
                MeshLight(
                    vertices=(rdf_from_mesh @ vertices_h.T).T[:, :3],
                    triangles=faces,
                    radiance=(0.0, 0.0, 0.0),
                    albedo=albedo,
                    roughness=roughness,
                    metallic=metallic,
                    surface=True,
                    device=device,
                )
            )
        return result

    def make_environment(self, device: torch.device | str) -> torch.Tensor:
        if self.environment_path is not None:
            environment = Environment(
                path=str(self.environment_path),
                device=device,
                environment_type="2d",
                optimize_environment=False,
            ).get_environment()
            if environment is None:
                raise ValueError(f"Failed to load environment map: {self.environment_path}")
            return environment
        environment = torch.empty(
            (64, 128, 4), dtype=torch.float32, device=device
        )
        environment[..., :3] = torch.as_tensor(
            self.environment_color, dtype=torch.float32, device=device
        )
        environment[..., 3] = 1.0
        return environment.contiguous()

    def make_batch(self, frame_id: int, device: torch.device | str) -> Batch:
        intrinsics = self.intrinsics
        u, v = torch.meshgrid(
            torch.arange(intrinsics.width, device=device, dtype=torch.float32),
            torch.arange(intrinsics.height, device=device, dtype=torch.float32),
            indexing="xy",
        )
        rays_dir = torch.stack(
            [
                (u + 0.5 - intrinsics.cx) / intrinsics.fx,
                (v + 0.5 - intrinsics.cy) / intrinsics.fy,
                torch.ones_like(u),
            ],
            dim=-1,
        )
        rays_dir = torch.nn.functional.normalize(rays_dir, dim=-1).unsqueeze(0)
        rays_ori = torch.zeros_like(rays_dir)
        pixel_coords = torch.stack([u + 0.5, v + 0.5], dim=-1).unsqueeze(0)
        pose = torch.as_tensor(
            self.camera_to_world(frame_id), dtype=torch.float32, device=device
        ).unsqueeze(0)
        return Batch(
            rays_ori=rays_ori,
            rays_dir=rays_dir,
            T_to_world=pose,
            intrinsics=intrinsics.values,
            camera_idx=0,
            frame_idx=frame_id - 1,
            pixel_coords=pixel_coords,
        )


__all__ = [
    "DEFAULT_VIDEO_FPS",
    "RelightingIntrinsics",
    "RelightingSequence",
    "encode_aov_videos",
]
