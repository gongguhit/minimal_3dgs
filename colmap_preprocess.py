"""
COLMAP Preprocessing Script
Converts raw images to COLMAP format for 3DGS training.

Usage:
    python colmap_preprocess.py --input ./data/lego_raw --output ./data/lego_colmap
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from glob import glob

import numpy as np
import pycolmap
from plyfile import PlyData, PlyElement


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
                               verbose: bool = True):
    """
    Run COLMAP reconstruction pipeline on input images.

    Args:
        input_dir: Directory containing input images
        output_dir: Output directory for COLMAP dataset
        camera_model: Camera model (SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV)
        single_camera: Whether all images use the same camera
        verbose: Print progress messages
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

    print(f"Copying images to {images_dir}...")
    for i, img_path in enumerate(sorted(images)):
        src = Path(img_path)
        # Rename to sequential format
        dst = images_dir / f"{i:05d}{src.suffix.lower()}"
        shutil.copy2(src, dst)

    print(f"Copied {len(images)} images")

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
        description="Convert raw images to COLMAP format for 3DGS training"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input directory containing images"
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
        "--quiet", "-q", action="store_true",
        help="Suppress progress messages"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("COLMAP Preprocessing for 3DGS")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Camera: {args.camera_model} ({'multi' if args.multi_camera else 'single'})")
    print("=" * 60)

    result = run_colmap_reconstruction(
        input_dir=args.input,
        output_dir=args.output,
        camera_model=args.camera_model,
        single_camera=not args.multi_camera,
        verbose=not args.quiet
    )

    print("\n" + "=" * 60)
    print("COLMAP Preprocessing Complete!")
    print("=" * 60)
    print(f"Images:     {result['num_registered']}/{result['num_images']} registered")
    print(f"3D Points:  {result['num_points']}")
    print(f"Output:     {result['output_dir']}")
    print("\nTo train 3DGS:")
    print(f"  python train_minimal.py -s {result['output_dir']} --iterations 5000")
    print("=" * 60)


if __name__ == "__main__":
    main()
