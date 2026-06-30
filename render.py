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

import argparse
from pathlib import Path
from threedgrut.model.light import MeshLight, PointLight, SphereLight
from threedgrut.render import Renderer

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="path to the pretrained checkpoint",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="",
        help="Path to the training data, if not provided taken from ckpt",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        type=str,
        help="Output path. Required unless --environment-relight is set; --environment-relight defaults to the checkpoint run directory.",
    )
    parser.add_argument(
        "--save-gt",
        action="store_false",
        help="If set, the GT images will not be saved [True by default]",
    )
    parser.add_argument(
        "--compute-extra-metrics",
        action="store_false",
        help="If set, extra image metrics will not be computed [True by default]",
    )
    relight_group = parser.add_mutually_exclusive_group()
    relight_group.add_argument(
        "--environment-relight",
        action="store_true",
        help="If set, render the scaled-albedo checkpoint under every environment map in --environment-dir.",
    )
    relight_group.add_argument(
        "--lights-relight",
        action="store_true",
        help="If set, render with the demo point/sphere/mesh lights.",
    )
    parser.add_argument(
        "--environment-dir",
        type=str,
        default=None,
        help="Folder containing environment maps used by --environment-relight.",
    )
    parser.add_argument(
        "--environment-path",
        type=str,
        default=None,
        help="Explicit environment map path. For --lights-relight, no environment is loaded unless this is set.",
    )
    parser.add_argument(
        "--visualize-lights",
        action="store_true",
        help="If set, make environment and visible lights contribute on escaped rays.",
    )
    parser.add_argument(
        "--render_frame_stride",
        type=int,
        default=1,
        help=(
            "Render every Nth test frame. Applies to normal, lights, and "
            "environment relight rendering. Default: 1."
        ),
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Hydra/OmegaConf-style checkpoint config override. May be passed "
            "multiple times, e.g. --override render.max_bounces=10."
        ),
    )
    args = parser.parse_args()

    if args.environment_relight and not args.environment_dir:
        parser.error("--environment-dir is required when --environment-relight is set")
    if args.environment_relight and args.environment_path:
        parser.error("--environment-path cannot be used with --environment-relight")
    if not args.environment_relight and not args.out_dir:
        parser.error("--out-dir is required unless --environment-relight is set")
    if args.render_frame_stride < 1:
        parser.error("--render_frame_stride must be >= 1")

    out_dir = args.out_dir
    if args.environment_relight and out_dir is None:
        out_dir = str(Path(args.checkpoint).resolve().parent)
    visualize_lights = args.visualize_lights

    if args.environment_relight:
        renderer = Renderer.from_checkpoint(
            checkpoint_path=str(Path(out_dir) / "ckpt_last_scaled.pt"),
            path=args.path,
            out_dir=out_dir,
            save_gt=False,
            computes_extra_metrics=False,
            create_run_dir=False,
            visualize_lights=visualize_lights,
            config_overrides=args.override,
        )
        renderer.render_relight_all(
            environment_dir=args.environment_dir,
            frame_stride=args.render_frame_stride,
        )
    elif args.lights_relight:
        renderer = Renderer.from_checkpoint(
            checkpoint_path=args.checkpoint,
            path=args.path,
            out_dir=out_dir,
            save_gt=False,
            computes_extra_metrics=False,
            visualize_lights=visualize_lights,
            restore_environment=args.environment_path is not None,
            environment_path=args.environment_path,
            config_overrides=args.override,
        )

        # Lights relight has no paired GT target, so metrics are always disabled.
        renderer.compute_metrics = False
        renderer.model.lights = [
            PointLight(
                position=(0.0, 0.0, 1.0),
                intensity=(20.0, 20.0, 20.0),
                device=renderer.model.device,
            ),
            MeshLight(
                vertices=[
                    [0.3, -0.1, 0.9],
                    [0.5, -0.1, 0.9],
                    [0.5, 0.1, 0.9],
                    [0.3, 0.1, 0.9],
                    [0.3, -0.1, 1.1],
                    [0.5, -0.1, 1.1],
                    [0.5, 0.1, 1.1],
                    [0.3, 0.1, 1.1],
                ],
                triangles=[
                    [0, 3, 2],
                    [0, 2, 1],
                    [4, 5, 6],
                    [4, 6, 7],
                    [0, 1, 5],
                    [0, 5, 4],
                    [3, 7, 6],
                    [3, 6, 2],
                    [0, 4, 7],
                    [0, 7, 3],
                    [1, 2, 6],
                    [1, 6, 5],
                ],
                radiance=(0.0, 0.0, 20.0),
                two_sided=False,
                device=renderer.model.device,
            ),
            SphereLight(
                center=(0.0, 1.0, 0.0),
                radius=0.1,
                radiance=(20.0, 0.0, 0.0),
                device=renderer.model.device,
            ),
        ]
        renderer.render_all(frame_stride=args.render_frame_stride)
    else:
        renderer = Renderer.from_checkpoint(
            checkpoint_path=args.checkpoint,
            path=args.path,
            out_dir=out_dir,
            save_gt=args.save_gt,
            computes_extra_metrics=args.compute_extra_metrics,
            visualize_lights=visualize_lights,
            environment_path=args.environment_path,
            config_overrides=args.override,
        )
        renderer.render_all(frame_stride=args.render_frame_stride)
