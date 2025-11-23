"""
Render script for 3D Gaussian Splatting.
Renders RGB images from trained models.
"""

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.gaussian_model import GaussianModel


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    """Render a set of views and save images."""
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_result = render(view, gaussians, pipeline, background)
        rendering = render_result["render"]
        gt = view.original_image[0:3, :, :]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams,
                skip_train: bool, skip_test: bool):
    """Load model and render train/test sets."""
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            print(f"Rendering {len(scene.getTrainCameras())} training views...")
            render_set(dataset.model_path, "train", scene.loaded_iter,
                      scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
            print(f"Rendering {len(scene.getTestCameras())} test views...")
            render_set(dataset.model_path, "test", scene.loaded_iter,
                      scene.getTestCameras(), gaussians, pipeline, background)


if __name__ == "__main__":
    parser = ArgumentParser(description="Render script for 3D Gaussian Splatting")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int,
                        help="Model iteration to render (-1 = latest)")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip rendering training views")
    parser.add_argument("--skip_test", action="store_true",
                        help="Skip rendering test views")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")

    args = get_combined_args(parser)

    print("=" * 60)
    print("3DGS Rendering")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Source: {args.source_path}")
    print(f"Iteration: {args.iteration}")
    print(f"Resolution: {args.resolution} (-1=auto, 1/2/4/8=divide, >8=target width)")
    print("=" * 60)

    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test
    )

    print("=" * 60)
    print("Rendering Complete!")
    print("=" * 60)
