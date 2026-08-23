# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import multiprocessing as mp
import os
import traceback
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image

from threedgrut.utils.logger import logger


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RGB2X_DIR = PROJECT_ROOT / "rgb2x"
DEFAULT_RGB2X_OUTPUT_DIR = DEFAULT_RGB2X_DIR / "outputs"
DEFAULT_RGB2X_MODEL = "zheng95z/rgb-to-x"
DEFAULT_RGB2X_CACHE_DIR = DEFAULT_RGB2X_DIR / "model_cache"

RGB2X_PROMPTS = {
    "albedo": "Albedo (diffuse basecolor)",
    "roughness": "Roughness",
    "metallic": "Metallicness",
    "normal": "Camera-space Normal",
    "irradiance": "Irradiance (diffuse lighting)",
}


def _dataset_output_path(dataset_root: str | os.PathLike[str]) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    parts = [part for part in root.parts if part not in (root.anchor, "")]
    if len(parts) >= 2:
        return Path(parts[-2]) / parts[-1]
    if parts:
        return Path(parts[-1])
    return Path("dataset")


def _resolve_project_path(path: str | os.PathLike[str]) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def rgb2x_prior_paths(
    image_path: str | os.PathLike[str],
    dataset_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str] = DEFAULT_RGB2X_OUTPUT_DIR,
    aovs: Sequence[str] = ("normal",),
) -> dict[str, Path]:
    image = Path(image_path).expanduser().resolve()
    root = Path(dataset_root).expanduser().resolve()

    try:
        relative = image.relative_to(root)
    except ValueError:
        relative = Path(image.name)

    base_dir = (
        _resolve_project_path(output_root)
        / _dataset_output_path(root)
        / relative.parent
    )
    stem = relative.stem
    return {aov: base_dir / f"{stem}_prior_{aov}.png" for aov in aovs}


def _load_image_as_tensor(image_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    img = Image.open(image_path)
    has_alpha = img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    )
    img = img.convert("RGBA" if has_alpha else "RGB")

    tensor = TF.to_tensor(img)
    rgb = torch.pow(tensor[:3], 2.2)
    alpha = tensor[3:4] if has_alpha else torch.ones_like(rgb[:1])
    return rgb, alpha


def _save_tensor_png(
    img_chw: torch.Tensor, alpha_chw: torch.Tensor, out_path: Path
) -> None:
    if img_chw.ndim == 4:
        img_chw = img_chw[0]
    if alpha_chw.ndim == 4:
        alpha_chw = alpha_chw[0]

    img_chw = img_chw.clamp(0, 1).detach().cpu()
    alpha_chw = alpha_chw.clamp(0, 1).detach().cpu()

    img_hwc = (img_chw * 255).byte().permute(1, 2, 0).numpy()
    alpha_hw = (alpha_chw[0] * 255).byte().numpy()

    img_rgba = Image.fromarray(img_hwc).convert("RGBA")
    img_rgba.putalpha(Image.fromarray(alpha_hw))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img_rgba.save(out_path)


def _srgb_to_linear(image_chw: torch.Tensor) -> torch.Tensor:
    return image_chw.clamp(0.0, 1.0).pow(2.2)


def _camera_normal_to_world(
    normal_chw: torch.Tensor,
    c2w: torch.Tensor,
    *,
    c2w_uses_opengl_axes: bool,
) -> torch.Tensor:
    """Transform an RGB2X OpenGL camera normal into world space.

    Raw NeRF/Blender camera-to-world matrices use the same right/up/back axes
    as RGB2X, while dataset poses standardized by 3DGRUT use OpenCV-style
    right/down/front axes. Only the latter require a Y/Z flip.
    """
    normal_chw = normal_chw * 2.0 - 1.0
    if not c2w_uses_opengl_axes:
        opengl_to_opencv = normal_chw.new_tensor((1.0, -1.0, -1.0)).view(3, 1, 1)
        normal_chw = normal_chw * opengl_to_opencv
    rotation = c2w[:3, :3].to(device=normal_chw.device, dtype=normal_chw.dtype)
    normal_hwc = normal_chw.permute(1, 2, 0)
    normal_world = torch.einsum("ij,hwj->hwi", rotation, normal_hwc)
    normal_world = torch.nn.functional.normalize(normal_world, dim=-1, eps=1e-6)
    return (normal_world.permute(2, 0, 1) + 1.0) * 0.5


def _load_nerf_raw_camera_to_worlds(
    image_paths: Sequence[Path],
    dataset_root: str | os.PathLike[str],
) -> list[torch.Tensor | None] | None:
    root = Path(dataset_root).expanduser().resolve()
    transform_files = [
        root / "transforms_train.json",
        root / "transforms_val.json",
        root / "transforms_test.json",
    ]
    if not any(path.exists() for path in transform_files):
        return None

    pose_by_image: dict[Path, torch.Tensor] = {}
    for transform_file in transform_files:
        if not transform_file.exists():
            continue
        with open(transform_file, "r") as f:
            frames = json.load(f).get("frames", [])
        for frame in frames:
            frame_path = (root / f"{frame['file_path']}").resolve()
            c2w = torch.as_tensor(frame["transform_matrix"], dtype=torch.float32)
            pose_by_image[frame_path] = c2w
            if frame_path.suffix == "":
                for ext in (".png", ".jpg", ".jpeg"):
                    pose_by_image[frame_path.with_suffix(ext)] = c2w

    raw_c2ws = [pose_by_image.get(path) for path in image_paths]
    return raw_c2ws if any(c2w is not None for c2w in raw_c2ws) else None


def _import_rgb2x_pipeline():
    from diffusers import DDIMScheduler
    from rgb2x.pipeline_rgb2x import StableDiffusionAOVMatEstPipeline

    return DDIMScheduler, StableDiffusionAOVMatEstPipeline


def _as_serializable_camera_to_worlds(
    camera_to_worlds: Iterable[torch.Tensor] | None,
) -> list | None:
    if camera_to_worlds is None:
        return None

    serializable = []
    for c2w in camera_to_worlds:
        if c2w is None:
            serializable.append(None)
        elif isinstance(c2w, torch.Tensor):
            serializable.append(c2w.detach().cpu().tolist())
        else:
            serializable.append(
                torch.as_tensor(c2w, dtype=torch.float32).cpu().tolist()
            )
    return serializable


def _generate_rgb2x_priors_worker(kwargs: dict, queue) -> None:
    try:
        stats = generate_rgb2x_priors(**kwargs)
        queue.put(("ok", stats))
    except Exception:
        queue.put(("error", traceback.format_exc()))


def generate_rgb2x_priors_in_subprocess(
    image_paths: Iterable[str | os.PathLike[str]],
    dataset_root: str | os.PathLike[str],
    camera_to_worlds: Iterable[torch.Tensor] | None = None,
    **kwargs,
) -> dict[str, int]:
    """Generate priors in a fresh process so diffusion CUDA state cannot poison training."""
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    worker_kwargs = {
        **kwargs,
        "image_paths": [str(Path(path).expanduser().resolve()) for path in image_paths],
        "dataset_root": str(Path(dataset_root).expanduser().resolve()),
        "camera_to_worlds": _as_serializable_camera_to_worlds(camera_to_worlds),
    }
    process = ctx.Process(
        target=_generate_rgb2x_priors_worker, args=(worker_kwargs, queue)
    )
    process.start()
    process.join()

    if queue.empty():
        raise RuntimeError(
            f"rgb2x prior subprocess exited without a result; exitcode={process.exitcode}"
        )

    status, payload = queue.get()
    if status != "ok":
        raise RuntimeError(f"rgb2x prior subprocess failed:\n{payload}")
    if process.exitcode not in (0, None):
        raise RuntimeError(
            f"rgb2x prior subprocess exited with code {process.exitcode}"
        )
    return payload


@torch.no_grad()
def generate_rgb2x_priors(
    image_paths: Iterable[str | os.PathLike[str]],
    dataset_root: str | os.PathLike[str],
    camera_to_worlds: Iterable[torch.Tensor] | None = None,
    output_root: str | os.PathLike[str] = DEFAULT_RGB2X_OUTPUT_DIR,
    model_name_or_path: str | os.PathLike[str] = DEFAULT_RGB2X_MODEL,
    cache_dir: str | os.PathLike[str] = DEFAULT_RGB2X_CACHE_DIR,
    aovs: Sequence[str] = ("normal",),
    input_size: int = 512,
    inference_steps: int = 50,
    seed: int = 42,
    skip_existing: bool = True,
    local_files_only: bool = False,
    batch_size: int = 1,
) -> dict[str, int]:
    aovs = tuple(aovs)
    invalid_aovs = [aov for aov in aovs if aov not in RGB2X_PROMPTS]
    if invalid_aovs:
        raise ValueError(
            f"Unsupported rgb2x AOVs: {invalid_aovs}; choose from {sorted(RGB2X_PROMPTS)}"
        )

    image_paths = [Path(path).expanduser().resolve() for path in image_paths]
    raw_nerf_c2ws = _load_nerf_raw_camera_to_worlds(image_paths, dataset_root)
    raw_nerf_image_paths = {
        image_path
        for image_path, raw_c2w in zip(image_paths, raw_nerf_c2ws or [])
        if raw_c2w is not None
    }
    if camera_to_worlds is None:
        c2w_list = [None] * len(image_paths)
    else:
        c2w_list = [
            torch.as_tensor(c2w, dtype=torch.float32) for c2w in camera_to_worlds
        ]
        if len(c2w_list) != len(image_paths):
            raise ValueError(
                "camera_to_worlds must have the same length as image_paths"
            )
    if raw_nerf_c2ws is not None:
        c2w_list = [
            raw if raw is not None else c2w for raw, c2w in zip(raw_nerf_c2ws, c2w_list)
        ]

    image_items = list(dict(zip(image_paths, c2w_list)).items())
    if not image_items:
        return {"saved": 0, "skipped": 0, "failed": 0}

    pending: list[tuple[Path, torch.Tensor | None]] = []
    skipped = 0
    for image_path, c2w in image_items:
        out_paths = rgb2x_prior_paths(image_path, dataset_root, output_root, aovs)
        missing = [path for path in out_paths.values() if not path.exists()]
        if skip_existing and not missing:
            skipped += len(aovs)
        else:
            pending.append((image_path, c2w))

    if not pending:
        logger.info(
            f"rgb2x priors already exist for {len(image_items)} image(s); output={output_root}"
        )
        return {"saved": 0, "skipped": skipped, "failed": 0}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    DDIMScheduler, StableDiffusionAOVMatEstPipeline = _import_rgb2x_pipeline()

    logger.info(
        f"Generating rgb2x priors for {len(pending)} image(s), aovs={list(aovs)}, output={output_root}"
    )
    pipe = StableDiffusionAOVMatEstPipeline.from_pretrained(
        str(model_name_or_path),
        torch_dtype=torch_dtype,
        cache_dir=str(_resolve_project_path(cache_dir)),
        local_files_only=local_files_only,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing",
    )
    pipe.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=device).manual_seed(seed)

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    saved = 0
    failed = 0
    batch_starts = range(0, len(pending), batch_size)
    for start in logger.track(
        batch_starts, description="rgb2x priors", color="salmon1", transient=True
    ):
        batch_items = pending[start : start + batch_size]
        valid_items: list[
            tuple[Path, torch.Tensor | None, torch.Tensor, torch.Tensor, int, int]
        ] = []
        for image_path, c2w in batch_items:
            try:
                rgb, alpha = _load_image_as_tensor(image_path)
            except Exception as exc:
                logger.warning(f"rgb2x failed to read {image_path}: {exc}")
                failed += len(aovs)
                continue
            valid_items.append(
                (image_path, c2w, rgb, alpha, rgb.shape[1], rgb.shape[2])
            )

        if not valid_items:
            continue

        rgb_in = torch.stack([rgb for _, _, rgb, _, _, _ in valid_items], dim=0)
        alpha_in = torch.stack([alpha for _, _, _, alpha, _, _ in valid_items], dim=0)
        rgb_in = F.interpolate(
            rgb_in, size=(input_size, input_size), mode="bilinear", align_corners=False
        ).to(device)
        alpha_in = F.interpolate(
            alpha_in,
            size=(input_size, input_size),
            mode="bilinear",
            align_corners=False,
        ).to(device)
        rgb_in = rgb_in * alpha_in

        for aov in aovs:
            batch_out_paths: list[Path] = []
            batch_c2ws: list[torch.Tensor | None] = []
            batch_sizes_for_aov: list[tuple[int, int]] = []
            batch_indices: list[int] = []
            for idx, (image_path, c2w, _, _, height, width) in enumerate(valid_items):
                out_path = rgb2x_prior_paths(
                    image_path, dataset_root, output_root, (aov,)
                )[aov]
                if skip_existing and out_path.exists():
                    skipped += 1
                    continue
                batch_out_paths.append(out_path)
                batch_c2ws.append(c2w)
                batch_sizes_for_aov.append((height, width))
                batch_indices.append(idx)

            if not batch_out_paths:
                continue

            try:
                result = pipe(
                    prompt=[RGB2X_PROMPTS[aov]] * len(batch_out_paths),
                    photo=rgb_in[batch_indices],
                    num_inference_steps=inference_steps,
                    height=input_size,
                    width=input_size,
                    generator=generator,
                    required_aovs=[aov],
                    output_type="pt",
                )
                aov_batch = result.images[0]
                for i, (out_path, c2w, (height, width)) in enumerate(
                    zip(batch_out_paths, batch_c2ws, batch_sizes_for_aov)
                ):
                    img_tensor = aov_batch[i]
                    if (height, width) != (input_size, input_size):
                        img_tensor = TF.resize(img_tensor, [height, width])
                    if aov == "normal" and c2w is not None:
                        image_path = valid_items[batch_indices[i]][0]
                        uses_raw_nerf_axes = image_path in raw_nerf_image_paths
                        img_tensor = _camera_normal_to_world(
                            img_tensor,
                            c2w,
                            c2w_uses_opengl_axes=uses_raw_nerf_axes,
                        )
                    if aov == "albedo":
                        img_tensor = _srgb_to_linear(img_tensor)
                    alpha_orig = (
                        valid_items[batch_indices[i]][3].unsqueeze(0).to(device)
                    )
                    _save_tensor_png(img_tensor * alpha_orig, alpha_orig, out_path)
                    saved += 1
            except Exception as exc:
                logger.warning(f"rgb2x inference failed for batch aov [{aov}]: {exc}")
                failed += len(batch_out_paths)

    del pipe
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    logger.info(f"rgb2x priors done: saved={saved}, skipped={skipped}, failed={failed}")
    return {"saved": saved, "skipped": skipped, "failed": failed}
