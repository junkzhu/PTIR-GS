#!/usr/bin/env python3
"""Compose a new chair 3DGS object into a kitchen scene and relight it.

The relighting sequence supplies animated analytic lights, mirror geometry, and
the environment color. The script creates an inference-only composed
checkpoint, then calls the normal sequence renderer to produce all AOV videos.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.playground.composition import (  # noqa: E402
    GAUSSIAN_FIELDS,
    Placement,
    compose_checkpoints,
    load_checkpoint,
    save_checkpoint,
    transform_checkpoint,
)


DEFAULT_BASE_CHECKPOINT = Path("path/to/data/kitchen_ckpt_last.pt")
DEFAULT_OBJECT_CHECKPOINT = Path("path/to/data/chair_ckpt_last_scaled.pt")
DEFAULT_SEQUENCE_DIR = SCRIPT_DIR / "demo_seq"
DEFAULT_DATASET_DIR = Path("path/to/data/kitchen")
DEFAULT_OUTPUT_DIR = Path("path/to/data/demo_seq")
DEFAULT_COMPOSED_CHECKPOINT = DEFAULT_OUTPUT_DIR / "kitchen_chair_composed.pt"

# A measured tabletop contact plane in the kitchen RDF scene.
CONTACT_POINT = (-1.20, 2.37, 0.90)
TABLE_UP = (0.0, -0.786, -0.618)
OBJECT_SCALE = 0.30
OBJECT_YAW_DEGREES = 225.0


def _finite_object(checkpoint: dict) -> tuple[dict, int]:
    count = int(checkpoint["positions"].shape[0])
    valid = torch.ones(count, dtype=torch.bool)
    for field in GAUSSIAN_FIELDS:
        values = checkpoint.get(field)
        if not torch.is_tensor(values) or values.ndim == 0 or values.shape[0] != count:
            raise ValueError(f"object checkpoint has invalid field: {field}")
        valid &= torch.isfinite(values).all(dim=1)
    if bool(valid.all()):
        return checkpoint, count
    result = dict(checkpoint)
    for field in GAUSSIAN_FIELDS:
        result[field] = torch.nn.Parameter(
            checkpoint[field].detach()[valid].contiguous(), requires_grad=False
        )
    return result, int(valid.sum())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--object-checkpoint", type=Path, default=DEFAULT_OBJECT_CHECKPOINT)
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--composed-checkpoint", type=Path, default=DEFAULT_COMPOSED_CHECKPOINT)
    parser.add_argument("--spp", type=int, default=1024)
    parser.add_argument("--light-scale", type=float, default=1.0)
    parser.add_argument(
        "--reuse-composed",
        action="store_true",
        help="Reuse --composed-checkpoint instead of composing it again",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.spp < 1 or args.light_scale < 0.0:
        raise SystemExit("--spp must be positive and --light-scale must be non-negative")
    base = args.base_checkpoint.expanduser().resolve()
    object_path = args.object_checkpoint.expanduser().resolve()
    sequence = args.sequence.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    composed_path = args.composed_checkpoint.expanduser().resolve()
    for path in (base, object_path, sequence, dataset):
        if not path.exists():
            raise FileNotFoundError(f"Demo asset not found: {path}")
    for name in ("camera_seq.json", "light_seq.json", "mesh_seq.json"):
        if not (sequence / name).is_file():
            raise FileNotFoundError(f"Demo sequence is missing {name}: {sequence}")

    if not args.reuse_composed or not composed_path.is_file():
        base_checkpoint = load_checkpoint(base)
        object_checkpoint, object_count = _finite_object(load_checkpoint(object_path))
        placed_object, placement_metadata = transform_checkpoint(
            object_checkpoint,
            Placement(
                contact_point=CONTACT_POINT,
                up=TABLE_UP,
                scale=OBJECT_SCALE,
                yaw_degrees=OBJECT_YAW_DEGREES,
                anchor_quantile=0.002,
            ),
        )
        composed = compose_checkpoints(
            base_checkpoint,
            [placed_object],
            name="demo_seq",
            composition_metadata=[
                {
                    "name": "chair",
                    "source_checkpoint": str(object_path),
                    "gaussians": object_count,
                    "placement": placement_metadata,
                }
            ],
        )
        save_checkpoint(composed, composed_path, overwrite=True)
        print(f"Composed checkpoint: {composed_path}")
    else:
        print(f"Reusing composed checkpoint: {composed_path}")

    light_data = json.loads((sequence / "light_seq.json").read_text(encoding="utf-8"))
    print(f"Lights: {len(light_data.get('lights', []))}; environment: {light_data.get('envmap', {})}")
    command = [
        sys.executable,
        str(ROOT / "render.py"),
        "--checkpoint",
        str(composed_path),
        "--path",
        str(dataset),
        "--out-dir",
        str(out_dir),
        "--relighting-sequence",
        str(sequence),
        "--render_frame_stride",
        "1",
        "--relighting-light-scale",
        str(args.light_scale),
        "--override",
        f"render.render_spp={args.spp}",
        "--override",
        "render.enable_metallic=true",
    ]
    subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
