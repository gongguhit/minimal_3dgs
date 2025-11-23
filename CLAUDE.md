# Pure 3D Gaussian Splatting

**CRITICAL**: Always source environment before work:
```bash
source setup_environment.sh
```

## Pipeline

```bash
# 1. COLMAP preprocessing (raw images → COLMAP format)
python colmap_preprocess.py -i ./data/my_scene_raw -o ./data/my_scene

# 2. Train 3DGS
python train_minimal.py -s ./data/my_scene --iterations 5000

# 3. Render test views
python render.py -m ./output/minimal_3dgs_my_scene_5000iter -s ./data/my_scene --iteration 5000 --skip_train

# 4. Evaluate metrics
python metrics_eval.py -m ./output/minimal_3dgs_my_scene_5000iter --iteration 5000
```

## Scripts

| Script | Description |
|--------|-------------|
| `colmap_preprocess.py` | Raw images → COLMAP format |
| `train_minimal.py` | Train 3DGS model |
| `render.py` | Render views |
| `metrics_eval.py` | Evaluate PSNR/SSIM/LPIPS |

## Data Format

```
data/my_scene/
├── images/           # RGB images (00000.png, 00001.png, ...)
├── renders/          # Symlink to images/
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    ├── points3D.bin
    └── points3D.ply
```

## Training Options

| Iterations | Use Case |
|------------|----------|
| 2000 | Quick test |
| 5000 | Standard |
| 30000 | Production |

## Troubleshooting

1. **Environment**: Always run `source setup_environment.sh` first
2. **CUDA OOM**: Reduce iterations or image size
3. **COLMAP fails**: Ensure >60% overlap between views
4. **Training diverges**: Check points3D.ply has >1000 points
