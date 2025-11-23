# Pure 3D Gaussian Splatting

A minimal 3D Gaussian Splatting pipeline for novel view synthesis, from raw images or video to evaluation.

## Requirements

- PyTorch with CUDA support
- pycolmap
- ~8-12GB GPU memory

## Setup

```bash
source setup_environment.sh
```

## Pipeline

### Step 1: COLMAP Preprocessing

Convert raw images or video to COLMAP format:

```bash
# From images
python colmap_preprocess.py -i ./data/my_scene_raw -o ./data/my_scene

# From video
python colmap_preprocess.py -i ./data/my_video.mp4 -o ./data/my_scene --fps 2 --resize_max 800
```

**Input:** Folder with images OR video file (mp4/mov/avi/mkv/webm/etc.)
```
data/my_scene_raw/
├── IMG_001.jpg
├── IMG_002.jpg
├── photo_003.png
└── ...
```

**Output:** COLMAP-formatted dataset
```
data/my_scene/
├── images/          # Renamed to 00000.png, 00001.png, ...
├── renders/         # Symlink to images/
├── sparse/0/
│   ├── cameras.bin
│   ├── images.bin
│   ├── points3D.bin
│   └── points3D.ply
└── database.db
```

**Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `-i, --input` | Input: image folder OR video file | Required |
| `-o, --output` | Output COLMAP dataset folder | Required |
| `--camera_model` | SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV | SIMPLE_PINHOLE |
| `--multi_camera` | Use different camera per image | False (single camera) |
| `--resize_width` | Resize to this width (maintains aspect ratio) | None |
| `--resize_height` | Resize to this height (maintains aspect ratio) | None |
| `--resize_max` | Resize so largest dimension equals this value | None |
| `--fps` | [Video] Extract frames at this FPS | 2 |
| `--skip_frames` | [Video] Extract every Nth frame (alternative to --fps) | None |
| `--max_frames` | [Video] Maximum number of frames to extract | None |

**Video Examples:**
```bash
# Extract at 2 FPS, resize to max 800px
python colmap_preprocess.py -i ./data/video.mp4 -o ./data/scene --fps 2 --resize_max 800

# Extract every 30th frame, limit to 100 frames
python colmap_preprocess.py -i ./data/video.mov -o ./data/scene --skip_frames 30 --max_frames 100

# Portrait video (2160x3840) → 450x800 output
python colmap_preprocess.py -i ./data/portrait.mov -o ./data/scene --fps 1 --resize_max 800
```

**Resize Examples:**
```bash
# Resize to 800px width (maintains aspect ratio)
python colmap_preprocess.py -i ./data/my_scene_raw -o ./data/my_scene --resize_width 800

# Resize so largest dimension is 1024px
python colmap_preprocess.py -i ./data/my_scene_raw -o ./data/my_scene --resize_max 1024
```

### Step 2: Training

Train 3D Gaussian Splatting model:

```bash
python train_minimal.py -s ./data/my_scene --iterations 5000
```

**Output:** Trained model in `output/minimal_3dgs_my_scene_5000iter/`

**Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `-s, --source_path` | Path to COLMAP dataset | Required |
| `--iterations` | Training iterations | 30000 |
| `-r, --resolution` | Image resolution (see below) | -1 (auto) |
| `--model_path` | Custom output path | Auto-generated |

**Resolution Options:**
| Value | Behavior |
|-------|----------|
| `-1` | Auto: scale to max 1600px width if larger |
| `1, 2, 4, 8` | Divide original size by this factor |
| `>8` | Target width in pixels |

**Iteration Guidelines:**
| Iterations | Use Case | Time |
|------------|----------|------|
| 2000 | Quick test | ~30s |
| 5000 | Standard | ~1min |
| 30000 | Production | ~5min |

### Step 3: Rendering

Render images from trained model:

```bash
python render.py \
    -m ./output/minimal_3dgs_my_scene_5000iter \
    -s ./data/my_scene \
    --iteration 5000 \
    --skip_train
```

**Output:** Rendered images in `output/minimal_3dgs_my_scene_5000iter/test/ours_5000/`

**Options:**
| Flag | Description |
|------|-------------|
| `-m, --model_path` | Path to trained model |
| `-s, --source_path` | Path to source dataset |
| `--iteration` | Model iteration to render |
| `-r, --resolution` | Image resolution (-1=auto, 1/2/4/8=divide, >8=width) |
| `--skip_train` | Only render test views |
| `--skip_test` | Only render training views |

### Step 4: Evaluation

Compute quality metrics:

```bash
python metrics_eval.py \
    -m ./output/minimal_3dgs_my_scene_5000iter \
    --iteration 5000
```

**Output:** PSNR, SSIM, LPIPS metrics

**Options:**
| Flag | Description |
|------|-------------|
| `-m, --model_path` | Path to trained model |
| `--iteration` | Model iteration to evaluate |

### Step 5: Visualization (Optional)

Convert PLY for web viewers like [SuperSplat](https://playcanvas.com/supersplat):

```bash
python convert_to_supersplat.py ./output/minimal_3dgs_my_scene_5000iter/point_cloud/iteration_5000/point_cloud.ply
```

**Output:** `point_cloud_supersplat.ply` (smaller file, compatible with web viewers)

**Why convert?** The original PLY uses SH degree 3 (62 properties, ~80MB), while web viewers expect SH degree 0 (17 properties, ~20MB).

**Options:**
| Flag | Description |
|------|-------------|
| `input` | Input PLY file (3DGS format) |
| `-o, --output` | Output file path (default: `input_supersplat.ply`) |

## Complete Examples

### From Images
```bash
source setup_environment.sh

# 1. Preprocess raw images
python colmap_preprocess.py -i ./data/my_scene_raw -o ./data/my_scene

# 2. Train (5000 iterations)
python train_minimal.py -s ./data/my_scene --iterations 5000

# 3. Render test views
python render.py -m ./output/minimal_3dgs_my_scene_5000iter -s ./data/my_scene --iteration 5000 --skip_train

# 4. Evaluate
python metrics_eval.py -m ./output/minimal_3dgs_my_scene_5000iter --iteration 5000
```

### From Video
```bash
source setup_environment.sh

# 1. Preprocess video (extract frames at 2 FPS, resize to max 800px)
python colmap_preprocess.py -i ./data/my_video.mp4 -o ./data/my_scene --fps 2 --resize_max 800

# 2-4. Same as above
python train_minimal.py -s ./data/my_scene --iterations 5000
python render.py -m ./output/minimal_3dgs_my_scene_5000iter -s ./data/my_scene --iteration 5000 --skip_train
python metrics_eval.py -m ./output/minimal_3dgs_my_scene_5000iter --iteration 5000
```

## Example Results

### Lego Dataset (103 images, 2000 iterations)

| Stage | Result |
|-------|--------|
| COLMAP | 103/103 registered, 17,198 points |
| Training | 56,288 Gaussians, L1=0.0045 |
| **PSNR** | **32.54 dB** |
| **SSIM** | **0.9827** |
| **LPIPS** | **0.0345** |

### Materials Dataset (103 images, 2000 iterations)

| Stage | Result |
|-------|--------|
| **PSNR** | **29.38 dB** |
| **SSIM** | **0.9749** |
| **LPIPS** | **0.0651** |

### Sofa Video (29 frames from video, 5000 iterations)

| Stage | Result |
|-------|--------|
| Video | 2160x3840 @ 60fps, 28.8s |
| Extraction | 29 frames @ 1 FPS, resized to 450x800 |
| COLMAP | 29/29 registered, 4,243 points |
| Training | 347,495 Gaussians |
| **PSNR** | **20.59 dB** |
| **SSIM** | **0.7427** |
| **LPIPS** | **0.2753** |

*Note: Real-world video captures typically have lower metrics than synthetic datasets due to motion blur and lighting variations.*

## Project Structure

```
├── colmap_preprocess.py  # Step 1: Images/Video → COLMAP
├── train_minimal.py      # Step 2: Train 3DGS
├── render.py             # Step 3: Render views
├── metrics_eval.py       # Step 4: Evaluate metrics
├── scene/                # Data loading modules
├── gaussian_renderer/    # Rendering modules
├── utils/                # Utility functions
├── arguments/            # CLI argument parsing
├── submodules/           # CUDA extensions
├── setup_environment.sh  # Environment setup
└── environment.yml       # Conda environment
```

## Troubleshooting

**COLMAP fails to register images:**
- Ensure sufficient overlap between views (>60%)
- Check image quality (not blurry)
- Try `--camera_model OPENCV` for lens distortion

**Out of memory:**
- Reduce `--iterations`
- Use smaller images

**Training diverges:**
- Check COLMAP reconstruction quality
- Ensure points3D.ply has sufficient points (>1000)
