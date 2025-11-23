"""
Render script for Event-LangSplat
Supports rendering RGB images, language features, and event features.
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
from arguments import ModelParams, PipelineParams, EventParams, LanguageParams, get_combined_args
from scene.gaussian_model import GaussianModel
import numpy as np
from PIL import Image


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, 
               render_rgb=True, render_language=False, render_event=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    
    if render_language:
        language_path = os.path.join(model_path, name, "ours_{}".format(iteration), "language_features")
        makedirs(language_path, exist_ok=True)
    
    if render_event:
        event_path = os.path.join(model_path, name, "ours_{}".format(iteration), "event_features")
        makedirs(event_path, exist_ok=True)

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_result = render(view, gaussians, pipeline, background)
        rendering = render_result["render"]
        gt = view.original_image[0:3, :, :]
        
        # Save RGB render
        if render_rgb:
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
        # Save language features if requested
        if render_language and "language_feature_image" in render_result:
            language_features = render_result["language_feature_image"]
            # Normalize and convert to RGB for visualization
            language_vis = normalize_features_for_visualization(language_features)
            torchvision.utils.save_image(language_vis, 
                                       os.path.join(language_path, '{0:05d}'.format(idx) + ".png"))
            # Also save raw features as numpy
            np.save(os.path.join(language_path, '{0:05d}'.format(idx) + ".npy"), 
                   language_features.cpu().numpy())
        
        # Save event features if requested
        if render_event:
            if "event_feature_image" in render_result:
                event_features = render_result["event_feature_image"]
                # Normalize and convert to RGB for visualization
                event_vis = normalize_features_for_visualization(event_features)
                torchvision.utils.save_image(event_vis, 
                                           os.path.join(event_path, '{0:05d}'.format(idx) + ".png"))
                # Also save raw features as numpy
                np.save(os.path.join(event_path, '{0:05d}'.format(idx) + ".npy"), 
                       event_features.cpu().numpy())
            else:
                # Create placeholder event feature visualization from Gaussian event features
                # This is a fallback when event_feature_image is not available
                dummy_features = torch.zeros((3, rendering.shape[1], rendering.shape[2]))
                torchvision.utils.save_image(dummy_features, 
                                           os.path.join(event_path, '{0:05d}'.format(idx) + ".png"))


def normalize_features_for_visualization(features):
    """
    Normalize high-dimensional features for RGB visualization.
    Uses PCA or similar dimensionality reduction to 3 channels.
    """
    # Simple approach: take first 3 channels and normalize
    if features.shape[0] >= 3:
        vis_features = features[:3]
    else:
        # Pad with zeros if less than 3 channels
        vis_features = torch.nn.functional.pad(features, (0, 0, 0, 0, 0, 3 - features.shape[0]))
    
    # Normalize to [0, 1]
    vis_features = (vis_features - vis_features.min()) / (vis_features.max() - vis_features.min() + 1e-8)
    
    return vis_features


def render_sets(dataset : ModelParams, event_params : EventParams, language_params : LanguageParams,
                iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
                render_rgb : bool, render_language : bool, render_event : bool):
    with torch.no_grad():
        # Initialize Gaussian model with feature dimensions
        gaussians = GaussianModel(
            dataset.sh_degree,
            language_feature_dim=language_params.language_feature_dim if language_params.use_language else 0,
            event_feature_dim=event_params.event_feature_dim if event_params.use_events else 0
        )
        
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), 
                      gaussians, pipeline, background, render_rgb, render_language, render_event)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), 
                      gaussians, pipeline, background, render_rgb, render_language, render_event)


def create_feature_comparison_visualization(model_path, iteration, views):
    """
    Create side-by-side comparison of RGB, language features, and event features.
    """
    comparison_path = os.path.join(model_path, "comparisons", "ours_{}".format(iteration))
    makedirs(comparison_path, exist_ok=True)
    
    rgb_path = os.path.join(model_path, "test", "ours_{}".format(iteration), "renders")
    language_path = os.path.join(model_path, "test", "ours_{}".format(iteration), "language_features")
    event_path = os.path.join(model_path, "test", "ours_{}".format(iteration), "event_features")
    
    for idx in range(len(views)):
        rgb_img = Image.open(os.path.join(rgb_path, '{0:05d}'.format(idx) + ".png"))
        
        comparison_imgs = [rgb_img]
        
        if os.path.exists(language_path):
            lang_img = Image.open(os.path.join(language_path, '{0:05d}'.format(idx) + ".png"))
            comparison_imgs.append(lang_img)
        
        if os.path.exists(event_path):
            event_img = Image.open(os.path.join(event_path, '{0:05d}'.format(idx) + ".png"))
            comparison_imgs.append(event_img)
        
        # Create horizontal concatenation
        widths, heights = zip(*(i.size for i in comparison_imgs))
        total_width = sum(widths)
        max_height = max(heights)
        
        comparison = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in comparison_imgs:
            comparison.paste(img, (x_offset, 0))
            x_offset += img.size[0]
        
        comparison.save(os.path.join(comparison_path, '{0:05d}'.format(idx) + "_comparison.png"))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script for Event-LangSplat")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    event_params = EventParams(parser)
    language_params = LanguageParams(parser)
    
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_rgb", action="store_true", default=True, help="Render RGB images")
    parser.add_argument("--render_language", action="store_true", help="Render language features")
    parser.add_argument("--render_event", action="store_true", help="Render event features")
    parser.add_argument("--create_comparison", action="store_true", help="Create feature comparison visualization")
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args), 
        event_params.extract(args),
        language_params.extract(args),
        args.iteration, 
        pipeline.extract(args), 
        args.skip_train, 
        args.skip_test,
        args.render_rgb,
        args.render_language,
        args.render_event
    )
    
    if args.create_comparison and not args.skip_test:
        # Load scene to get test views
        with torch.no_grad():
            gaussians = GaussianModel(3)
            scene = Scene(model.extract(args), gaussians, load_iteration=args.iteration, shuffle=False)
            create_feature_comparison_visualization(args.model_path, args.iteration, scene.getTestCameras())