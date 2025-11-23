"""
Minimal 3DGS Training - No event processing, pure RGB only
Memory optimized to avoid stack smashing
"""

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from scene import Scene, GaussianModel
from tqdm import tqdm
from arguments import ModelParams, PipelineParams, OptimizationParams
import argparse


def training(args):
    """Minimal training function."""

    # Get resolution (default to -1 if not specified)
    resolution = getattr(args, 'resolution', -1)

    # Output directory
    output_dir = f"output/minimal_3dgs_{os.path.basename(args.source_path)}_{args.iterations}iter"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "point_cloud", f"iteration_{args.iterations}"), exist_ok=True)

    print(f"🚀 Minimal 3DGS Training")
    print(f"   📁 Dataset: {args.source_path}")
    print(f"   💾 Output: {output_dir}")
    print(f"   🔄 Iterations: {args.iterations}")
    print(f"   📐 Resolution: {resolution} (-1=auto, 1/2/4/8=divide, >8=target width)")

    # Parse arguments using standard classes
    parser = argparse.ArgumentParser()
    model_params = ModelParams(parser)
    opt_params = OptimizationParams(parser)
    pipe_params = PipelineParams(parser)

    # Override with our settings
    model_args = model_params.extract(argparse.Namespace(
        source_path=args.source_path,
        model_path=output_dir,
        images="renders",
        depths="",
        resolution=resolution,
        white_background=False,
        data_device="cuda",
        eval=True,
        train_test_exp=False,
        sh_degree=3
    ))

    opt_args = opt_params.extract(argparse.Namespace(
        iterations=args.iterations,
        position_lr_init=0.00016,
        position_lr_final=0.0000016,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=args.iterations,
        feature_lr=0.0025,
        opacity_lr=0.05,
        scaling_lr=0.005,
        rotation_lr=0.001,
        exposure_lr_init=0.01,
        exposure_lr_final=0.001,
        exposure_lr_delay_steps=0,
        exposure_lr_delay_mult=0.0,
        percent_dense=0.01,
        lambda_dssim=0.2,
        densification_interval=100,
        opacity_reset_interval=3000,
        densify_from_iter=500,
        densify_until_iter=int(args.iterations * 0.75),
        densify_grad_threshold=0.0002,
        depth_l1_weight_init=1.0,
        depth_l1_weight_final=0.01,
        random_background=False,
        optimizer_type="default"
    ))

    pipe_args = pipe_params.extract(argparse.Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False
    ))

    # Save cfg_args for rendering
    cfg_args = argparse.Namespace(
        sh_degree=model_args.sh_degree,
        source_path=model_args.source_path,
        model_path=model_args.model_path,
        images=model_args.images,
        depths=model_args.depths,
        resolution=model_args.resolution,
        white_background=model_args.white_background,
        data_device=model_args.data_device,
        eval=model_args.eval,
        train_test_exp=model_args.train_test_exp,
        convert_SHs_python=pipe_args.convert_SHs_python,
        compute_cov3D_python=pipe_args.compute_cov3D_python,
        debug=pipe_args.debug,
        antialiasing=pipe_args.antialiasing
    )
    with open(os.path.join(output_dir, "cfg_args"), 'w') as f:
        f.write(str(cfg_args))
    print(f"✅ Saved cfg_args to {output_dir}/cfg_args")

    # Initialize model
    print(f"\n🏗️ Initializing Gaussian model...")
    gaussians = GaussianModel(model_args.sh_degree)
    scene = Scene(model_args, gaussians)
    gaussians.training_setup(opt_args)

    print(f"✅ Initialized {len(gaussians.get_xyz):,} Gaussians")

    # Training loop
    print(f"\n🎯 Training...")
    bg_color = [1, 1, 1] if model_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = scene.getTrainCameras().copy()
    best_loss = float('inf')

    progress_bar = tqdm(range(1, args.iterations + 1), desc="Training")

    for iteration in progress_bar:
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Select camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        render_pkg = render(viewpoint_cam, gaussians, pipe_args, background)
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()

        # Loss
        l1_loss_val = l1_loss(image, gt_image)
        loss = (1.0 - opt_args.lambda_dssim) * l1_loss_val

        if image.shape[0] == gt_image.shape[0] == 3:
            try:
                ssim_val = 1.0 - ssim(image, gt_image)
                loss += opt_args.lambda_dssim * ssim_val
            except:
                pass

        loss.backward()

        current_loss = l1_loss_val.item()
        if current_loss < best_loss:
            best_loss = current_loss

        # Densification
        with torch.no_grad():
            if iteration < opt_args.densify_until_iter:
                gaussians.max_radii2D[render_pkg['visibility_filter']] = torch.max(
                    gaussians.max_radii2D[render_pkg['visibility_filter']], render_pkg['radii'][render_pkg['visibility_filter']]
                )
                gaussians.add_densification_stats(render_pkg["viewspace_points"], render_pkg['visibility_filter'])

                if iteration > opt_args.densify_from_iter and iteration % opt_args.densification_interval == 0:
                    size_threshold = 20 if iteration > opt_args.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt_args.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, gaussians.max_radii2D)

                if iteration % opt_args.opacity_reset_interval == 0:
                    gaussians.reset_opacity()

        # Optimizer step
        if iteration < args.iterations:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        # Progress
        if iteration % 10 == 0:
            progress_bar.set_postfix({'L1': f"{current_loss:.4f}", 'Best': f"{best_loss:.4f}", 'G': f"{len(gaussians.get_xyz):,}"})

        if iteration % 500 == 0:
            print(f"\n[{iteration}/{args.iterations}] L1: {current_loss:.4f}, Best: {best_loss:.4f}, Gaussians: {len(gaussians.get_xyz):,}")

    # Save
    print(f"\n💾 Saving...")
    point_cloud_path = os.path.join(output_dir, "point_cloud", f"iteration_{args.iterations}", "point_cloud.ply")
    gaussians.save_ply(point_cloud_path)

    print(f"✅ Training complete!")
    print(f"   📊 Final L1 Loss: {current_loss:.4f}")
    print(f"   🏆 Best L1 Loss: {best_loss:.4f}")
    print(f"   🏗️ Final Gaussians: {len(gaussians.get_xyz):,}")
    print(f"   💾 Saved to: {output_dir}")

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal 3DGS Training")
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("-r", "--resolution", type=int, default=-1,
                        help="Resolution: -1=auto (max 1600), 1/2/4/8=divide factor, or target width")
    args = parser.parse_args()

    training(args)
