"""
Convert 3DGS PLY to SuperSplat-compatible format.

SuperSplat expects specific property names that differ from the original 3DGS format.
"""

import numpy as np
from plyfile import PlyData, PlyElement
import argparse
from pathlib import Path


def convert_ply_to_supersplat(input_path, output_path):
    """
    Convert 3DGS PLY format to SuperSplat-compatible format.

    Main differences:
    - SuperSplat expects 'red', 'green', 'blue' instead of f_dc_* for colors
    - Or it expects the standard 3DGS format with proper headers
    """
    print(f"Loading: {input_path}")
    plydata = PlyData.read(input_path)
    vertex = plydata['vertex']

    # Get all properties
    properties = [p.name for p in vertex.properties]
    print(f"Properties: {len(properties)}")
    print(f"Vertices: {len(vertex.data)}")

    # Extract data
    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
    normals = np.stack([vertex['nx'], vertex['ny'], vertex['nz']], axis=1)

    # DC colors (convert from SH to RGB)
    # f_dc values are in SH space, need to convert: color = (f_dc * C0 + 0.5)
    C0 = 0.28209479177387814  # SH constant
    f_dc = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=1)
    colors = np.clip(f_dc * C0 + 0.5, 0, 1)  # Convert SH to RGB [0,1]
    colors_uint8 = (colors * 255).astype(np.uint8)

    # Opacity (stored as inverse sigmoid)
    opacity_raw = vertex['opacity']
    opacity = 1 / (1 + np.exp(-opacity_raw))  # Sigmoid
    opacity_uint8 = (np.clip(opacity, 0, 1) * 255).astype(np.uint8)

    # Scales (stored as log)
    scales = np.exp(np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=1))

    # Rotations (quaternion wxyz)
    rot = np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']], axis=1)
    # Normalize quaternions
    rot = rot / (np.linalg.norm(rot, axis=1, keepdims=True) + 1e-8)

    # Create SuperSplat-compatible PLY
    # SuperSplat format uses: x,y,z, f_dc_0-2, opacity, scale_0-2, rot_0-3
    # But also supports a simpler format with RGB colors

    # Method 1: Create standard Gaussian Splat format (what SuperSplat actually expects)
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ]

    data = np.empty(len(vertex.data), dtype=dtype)
    data['x'] = vertex['x']
    data['y'] = vertex['y']
    data['z'] = vertex['z']
    data['nx'] = vertex['nx']
    data['ny'] = vertex['ny']
    data['nz'] = vertex['nz']
    data['f_dc_0'] = vertex['f_dc_0']
    data['f_dc_1'] = vertex['f_dc_1']
    data['f_dc_2'] = vertex['f_dc_2']
    data['opacity'] = vertex['opacity']
    data['scale_0'] = vertex['scale_0']
    data['scale_1'] = vertex['scale_1']
    data['scale_2'] = vertex['scale_2']
    data['rot_0'] = vertex['rot_0']
    data['rot_1'] = vertex['rot_1']
    data['rot_2'] = vertex['rot_2']
    data['rot_3'] = vertex['rot_3']

    # Create PLY element and save
    el = PlyElement.describe(data, 'vertex')

    # Save as binary little endian (standard format)
    PlyData([el], text=False, byte_order='<').write(output_path)

    print(f"Saved: {output_path}")
    print(f"  Vertices: {len(data)}")
    print(f"  Properties: {len(dtype)} (SH degree 0 only)")
    print(f"  Format: binary_little_endian")


def main():
    parser = argparse.ArgumentParser(description="Convert 3DGS PLY to SuperSplat format")
    parser.add_argument("input", help="Input PLY file (3DGS format)")
    parser.add_argument("-o", "--output", help="Output PLY file (default: input_supersplat.ply)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_supersplat.ply"

    convert_ply_to_supersplat(input_path, output_path)


if __name__ == "__main__":
    main()
