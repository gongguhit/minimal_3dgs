"""
COLMAP Preprocessing Script
Converts raw images or video to COLMAP format for 3DGS training.

Usage:
    # From images
    python colmap_preprocess.py --input ./data/lego_raw --output ./data/lego_colmap

    # From video
    python colmap_preprocess.py --input ./data/video.mp4 --output ./data/scene --fps 2
"""

import os
import sys
import shutil
import argparse
import tempfile
from pathlib import Path
from glob import glob

import cv2
import numpy as np
import pycolmap
from PIL import Image
from plyfile import PlyData, PlyElement


# Supported video extensions
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}


def _is_video_file(path):
    """Check if path is a video file."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def _extract_frames_from_video(video_path, output_dir, fps=None, skip_frames=None,
                                max_frames=None, resize_width=None, resize_height=None,
                                resize_max=None, verbose=True):
    """
    Extract frames from video file.

    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        fps: Target frames per second (mutually exclusive with skip_frames)
        skip_frames: Extract every Nth frame (mutually exclusive with fps)
        max_frames: Maximum number of frames to extract
        resize_width: Resize frames to this width
        resize_height: Resize frames to this height
        resize_max: Resize so largest dimension equals this value
        verbose: Print progress messages

    Returns:
        dict with extraction statistics
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if verbose:
        print(f"Video: {video_path.name}")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {video_fps:.2f}")
        print(f"   Total frames: {total_frames}")
        print(f"   Duration: {total_frames/video_fps:.1f}s")

    # Determine frame extraction interval
    if fps is not None and skip_frames is not None:
        raise ValueError("Cannot specify both --fps and --skip_frames")

    if fps is not None:
        # Calculate skip based on target fps
        if fps > video_fps:
            print(f"Warning: Requested FPS ({fps}) > video FPS ({video_fps:.2f}), using all frames")
            frame_interval = 1
        else:
            frame_interval = max(1, int(video_fps / fps))
    elif skip_frames is not None:
        frame_interval = skip_frames
    else:
        # Default: extract at ~2 FPS for reasonable coverage
        frame_interval = max(1, int(video_fps / 2))

    # Calculate resize dimensions
    new_width, new_height = width, height
    if resize_width is not None:
        scale = resize_width / width
        new_width = resize_width
        new_height = int(height * scale)
    elif resize_height is not None:
        scale = resize_height / height
        new_width = int(width * scale)
        new_height = resize_height
    elif resize_max is not None:
        if width >= height:
            scale = resize_max / width
        else:
            scale = resize_max / height
        new_width = int(width * scale)
        new_height = int(height * scale)

    do_resize = (new_width != width or new_height != height)

    if verbose:
        print(f"   Frame interval: every {frame_interval} frames")
        expected_frames = (total_frames + frame_interval - 1) // frame_interval
        if max_frames:
            expected_frames = min(expected_frames, max_frames)
        print(f"   Expected output: ~{expected_frames} frames")
        if do_resize:
            print(f"   Output size: {new_width}x{new_height}")

    # Extract frames
    frame_idx = 0
    saved_count = 0

    if verbose:
        print(f"\nExtracting frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Resize if needed
            if do_resize:
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

            # Save frame
            frame_path = output_dir / f"{saved_count:05d}.png"
            cv2.imwrite(str(frame_path), frame)
            saved_count += 1

            if verbose and saved_count % 50 == 0:
                print(f"   Extracted {saved_count} frames...")

            if max_frames and saved_count >= max_frames:
                break

        frame_idx += 1

    cap.release()

    if verbose:
        print(f"   Extracted {saved_count} frames to {output_dir}")

    return {
        "video_path": str(video_path),
        "total_video_frames": total_frames,
        "video_fps": video_fps,
        "original_size": (width, height),
        "output_size": (new_width, new_height),
        "frame_interval": frame_interval,
        "extracted_frames": saved_count,
    }


def _resize_image(src_path, dst_path, target_width=None, target_height=None, max_dimension=None):
    """
    Resize image with various options.

    Args:
        src_path: Source image path
        dst_path: Destination path
        target_width: Resize to specific width (maintains aspect ratio)
        target_height: Resize to specific height (maintains aspect ratio)
        max_dimension: Resize so largest dimension equals this value

    Returns:
        (new_width, new_height) tuple
    """
    img = Image.open(src_path)
    orig_w, orig_h = img.size

    if target_width is not None:
        # Resize to specific width
        scale = target_width / orig_w
        new_w = target_width
        new_h = int(orig_h * scale)
    elif target_height is not None:
        # Resize to specific height
        scale = target_height / orig_h
        new_w = int(orig_w * scale)
        new_h = target_height
    elif max_dimension is not None:
        # Resize so largest dimension matches
        if orig_w >= orig_h:
            scale = max_dimension / orig_w
        else:
            scale = max_dimension / orig_h
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
    else:
        # No resizing, just copy
        new_w, new_h = orig_w, orig_h

    if new_w != orig_w or new_h != orig_h:
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # Convert to RGB if necessary (handle RGBA, grayscale, etc.)
    if img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')

    img.save(dst_path, quality=95)
    return new_w, new_h


def _add_normals_to_ply(ply_path):
    """Add zero normals to PLY file (required by 3DGS)."""
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']

    # Check if normals already exist
    if 'nx' in vertices.data.dtype.names:
        return

    # Get existing data
    x = vertices['x']
    y = vertices['y']
    z = vertices['z']
    red = vertices['red']
    green = vertices['green']
    blue = vertices['blue']

    # Create new structured array with normals
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    new_vertices = np.empty(len(x), dtype=dtype)
    new_vertices['x'] = x
    new_vertices['y'] = y
    new_vertices['z'] = z
    new_vertices['nx'] = 0.0
    new_vertices['ny'] = 0.0
    new_vertices['nz'] = 0.0
    new_vertices['red'] = red
    new_vertices['green'] = green
    new_vertices['blue'] = blue

    # Save new PLY
    el = PlyElement.describe(new_vertices, 'vertex')
    PlyData([el], text=False).write(ply_path)


def run_colmap_reconstruction(input_dir: str, output_dir: str,
                               camera_model: str = "SIMPLE_PINHOLE",
                               single_camera: bool = True,
                               verbose: bool = True,
                               resize_width: int = None,
                               resize_height: int = None,
                               resize_max: int = None):
    """
    Run COLMAP reconstruction pipeline on input images.

    Args:
        input_dir: Directory containing input images
        output_dir: Output directory for COLMAP dataset
        camera_model: Camera model (SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV)
        single_camera: Whether all images use the same camera
        verbose: Print progress messages
        resize_width: Resize images to this width (maintains aspect ratio)
        resize_height: Resize images to this height (maintains aspect ratio)
        resize_max: Resize images so largest dimension equals this value
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Find images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    images = []
    for ext in image_extensions:
        images.extend(glob(str(input_path / ext)))

    if not images:
        # Check subdirectories
        for ext in image_extensions:
            images.extend(glob(str(input_path / '**' / ext), recursive=True))

    if not images:
        print(f"Error: No images found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(images)} images in {input_dir}")

    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    images_dir = output_path / "images"
    sparse_dir = output_path / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Copy/link images to output directory
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Determine if resizing is needed
    do_resize = resize_width is not None or resize_height is not None or resize_max is not None
    action = "Resizing" if do_resize else "Copying"
    print(f"{action} images to {images_dir}...")

    final_size = None
    for i, img_path in enumerate(sorted(images)):
        src = Path(img_path)
        # Rename to sequential format (always use .png for consistency)
        dst = images_dir / f"{i:05d}.png"

        if do_resize:
            new_w, new_h = _resize_image(
                src, dst,
                target_width=resize_width,
                target_height=resize_height,
                max_dimension=resize_max
            )
            if final_size is None:
                final_size = (new_w, new_h)
        else:
            # Just copy and convert to PNG
            img = Image.open(src)
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            img.save(dst, quality=95)
            if final_size is None:
                final_size = img.size

    resize_info = f" (resized to {final_size[0]}x{final_size[1]})" if do_resize else ""
    print(f"Processed {len(images)} images{resize_info}")

    # Database path
    database_path = output_path / "database.db"
    if database_path.exists():
        database_path.unlink()

    # Step 1: Feature extraction
    if verbose:
        print("\n[1/3] Extracting features...")

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.sift.max_num_features = 8192

    pycolmap.extract_features(
        database_path=str(database_path),
        image_path=str(images_dir),
        camera_mode=pycolmap.CameraMode.SINGLE if single_camera else pycolmap.CameraMode.AUTO,
        camera_model=camera_model,
        extraction_options=extraction_options,
    )

    if verbose:
        print("   Features extracted successfully")

    # Step 2: Feature matching
    if verbose:
        print("\n[2/3] Matching features...")

    matching_options = pycolmap.FeatureMatchingOptions()

    pycolmap.match_exhaustive(
        database_path=str(database_path),
        matching_options=matching_options,
    )

    if verbose:
        print("   Features matched successfully")

    # Step 3: Sparse reconstruction (mapping)
    if verbose:
        print("\n[3/3] Running sparse reconstruction...")

    maps = pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=str(images_dir),
        output_path=str(sparse_dir.parent),
        options=pycolmap.IncrementalPipelineOptions(
            min_num_matches=15,
        )
    )

    if not maps:
        print("Error: Reconstruction failed - no maps generated")
        sys.exit(1)

    # Find the largest reconstruction
    largest_map = max(maps.values(), key=lambda m: m.num_reg_images())

    if verbose:
        print(f"   Reconstruction complete!")
        print(f"   Registered images: {largest_map.num_reg_images()}/{len(images)}")
        print(f"   3D points: {largest_map.num_points3D()}")

    # Save the reconstruction
    largest_map.write(sparse_dir)

    # Also export as PLY for visualization
    ply_path = sparse_dir / "points3D.ply"
    largest_map.export_PLY(str(ply_path))

    # Add normals to PLY (required by 3DGS)
    _add_normals_to_ply(ply_path)

    # Create symlink: renders -> images (for compatibility)
    renders_link = output_path / "renders"
    if renders_link.exists() or renders_link.is_symlink():
        renders_link.unlink()
    renders_link.symlink_to("images")

    if verbose:
        print(f"\n   Saved reconstruction to {sparse_dir}")
        print(f"   Exported point cloud to {ply_path} (with normals)")

    # Cleanup - remove other reconstructions if any
    for item in (sparse_dir.parent).iterdir():
        if item.is_dir() and item.name != "0":
            shutil.rmtree(item)

    return {
        "num_images": len(images),
        "num_registered": largest_map.num_reg_images(),
        "num_points": largest_map.num_points3D(),
        "output_dir": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw images or video to COLMAP format for 3DGS training"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input: directory of images OR video file (mp4/mov/avi/mkv/etc.)"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output directory for COLMAP dataset"
    )
    parser.add_argument(
        "--camera_model", default="SIMPLE_PINHOLE",
        choices=["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV"],
        help="Camera model to use (default: SIMPLE_PINHOLE)"
    )
    parser.add_argument(
        "--multi_camera", action="store_true",
        help="Use multiple cameras (default: single camera for all images)"
    )
    parser.add_argument(
        "--resize_width", type=int, default=None,
        help="Resize images to this width (maintains aspect ratio)"
    )
    parser.add_argument(
        "--resize_height", type=int, default=None,
        help="Resize images to this height (maintains aspect ratio)"
    )
    parser.add_argument(
        "--resize_max", type=int, default=None,
        help="Resize images so largest dimension equals this value"
    )
    # Video-specific options
    parser.add_argument(
        "--fps", type=float, default=None,
        help="[Video] Extract frames at this FPS (default: 2 FPS)"
    )
    parser.add_argument(
        "--skip_frames", type=int, default=None,
        help="[Video] Extract every Nth frame (alternative to --fps)"
    )
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="[Video] Maximum number of frames to extract"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress messages"
    )

    args = parser.parse_args()

    # Detect if input is video or image directory
    is_video = _is_video_file(args.input)

    print("=" * 60)
    print("COLMAP Preprocessing for 3DGS")
    print("=" * 60)
    print(f"Input:  {args.input} ({'video' if is_video else 'images'})")
    print(f"Output: {args.output}")
    print(f"Camera: {args.camera_model} ({'multi' if args.multi_camera else 'single'})")

    # Show resize info
    if args.resize_width:
        print(f"Resize: width={args.resize_width}")
    elif args.resize_height:
        print(f"Resize: height={args.resize_height}")
    elif args.resize_max:
        print(f"Resize: max_dimension={args.resize_max}")
    else:
        print(f"Resize: none (original size)")

    # Show video options if applicable
    if is_video:
        if args.fps:
            print(f"Video:  extract at {args.fps} FPS")
        elif args.skip_frames:
            print(f"Video:  extract every {args.skip_frames} frames")
        else:
            print(f"Video:  extract at ~2 FPS (default)")
        if args.max_frames:
            print(f"        max {args.max_frames} frames")

    print("=" * 60)

    # If video, extract frames first
    video_result = None
    if is_video:
        # Create temporary directory for extracted frames
        temp_frames_dir = Path(args.output) / "_temp_frames"

        print("\n[0/3] Extracting frames from video...")
        video_result = _extract_frames_from_video(
            video_path=args.input,
            output_dir=temp_frames_dir,
            fps=args.fps,
            skip_frames=args.skip_frames,
            max_frames=args.max_frames,
            resize_width=args.resize_width,
            resize_height=args.resize_height,
            resize_max=args.resize_max,
            verbose=not args.quiet
        )

        # Use extracted frames as input, and disable resize (already done)
        input_dir = str(temp_frames_dir)
        resize_width = None
        resize_height = None
        resize_max = None
    else:
        input_dir = args.input
        resize_width = args.resize_width
        resize_height = args.resize_height
        resize_max = args.resize_max

    result = run_colmap_reconstruction(
        input_dir=input_dir,
        output_dir=args.output,
        camera_model=args.camera_model,
        single_camera=not args.multi_camera,
        verbose=not args.quiet,
        resize_width=resize_width,
        resize_height=resize_height,
        resize_max=resize_max
    )

    # Cleanup temporary frames if video input
    if is_video:
        temp_frames_dir = Path(args.output) / "_temp_frames"
        if temp_frames_dir.exists():
            shutil.rmtree(temp_frames_dir)

    print("\n" + "=" * 60)
    print("COLMAP Preprocessing Complete!")
    print("=" * 60)
    if video_result:
        print(f"Video:      {video_result['extracted_frames']} frames extracted")
    print(f"Images:     {result['num_registered']}/{result['num_images']} registered")
    print(f"3D Points:  {result['num_points']}")
    print(f"Output:     {result['output_dir']}")
    print("\nTo train 3DGS:")
    print(f"  python train_minimal.py -s {result['output_dir']} --iterations 5000")
    print("=" * 60)


if __name__ == "__main__":
    main()
