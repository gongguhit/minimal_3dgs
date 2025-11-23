# Pure 3D Gaussian Splatting

A minimal 3D Gaussian Splatting pipeline for novel view synthesis, from raw images to evaluation.

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

Convert raw images to COLMAP format:

```bash
python colmap_preprocess.py -i ./data/my_scene_raw -o ./data/my_scene
```

**Input:** Folder with images (any naming, any format: jpg/jpeg/png)
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
| `-i, --input` | Input folder with raw images | Required |
| `-o, --output` | Output COLMAP dataset folder | Required |
| `--camera_model` | SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV | SIMPLE_PINHOLE |
| `--multi_camera` | Use different camera per image | False (single camera) |

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
| `--model_path` | Custom output path | Auto-generated |

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

## Complete Example

```bash
# Setup environment
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

## Project Structure

```
├── colmap_preprocess.py  # Step 1: Raw images → COLMAP
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
