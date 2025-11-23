#!/usr/bin/env python3
"""
Simple metrics evaluation script for Event-LangSplat
Computes PSNR, SSIM, and LPIPS from rendered images.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import os
from pathlib import Path
import argparse
from tqdm import tqdm

# Import metrics
from utils.image_utils import psnr
from utils.loss_utils import ssim

# Try importing LPIPS, fallback if not available
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available. Install with: pip install lpips")

def load_image(image_path):
    """Load and convert image to tensor."""
    image = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    return transform(image)

def compute_metrics(rendered_path, gt_path, lpips_fn=None):
    """Compute PSNR, SSIM, and LPIPS between rendered and ground truth images."""
    
    # Load images
    rendered = load_image(rendered_path).cuda()
    gt = load_image(gt_path).cuda()
    
    # Ensure same dimensions
    if rendered.shape != gt.shape:
        # Resize rendered to match GT
        rendered = torch.nn.functional.interpolate(
            rendered.unsqueeze(0), size=gt.shape[-2:], mode='bilinear', align_corners=False
        ).squeeze(0)
    
    # Compute metrics
    psnr_val = psnr(rendered, gt)
    if torch.is_tensor(psnr_val):
        psnr_val = psnr_val.mean().item()
    
    ssim_val = ssim(rendered, gt)
    if torch.is_tensor(ssim_val):
        ssim_val = ssim_val.mean().item()
    
    lpips_val = None
    if lpips_fn is not None and LPIPS_AVAILABLE:
        # LPIPS expects values in [-1, 1], but our images are in [0, 1]
        rendered_lpips = rendered * 2.0 - 1.0
        gt_lpips = gt * 2.0 - 1.0
        lpips_val = lpips_fn(rendered_lpips.unsqueeze(0), gt_lpips.unsqueeze(0)).item()
    
    return {
        'psnr': psnr_val,
        'ssim': ssim_val,
        'lpips': lpips_val
    }

def evaluate_model(model_path, iteration=3000):
    """Evaluate model using rendered images."""
    
    print(f"🧪 Evaluating model: {model_path}")
    print(f"📊 Iteration: {iteration}")
    
    # Initialize LPIPS
    lpips_fn = None
    if LPIPS_AVAILABLE:
        lpips_fn = lpips.LPIPS(net='alex').cuda()
        print("✅ LPIPS initialized")
    else:
        print("⚠️  LPIPS not available")
    
    results = {}
    
    # Evaluate test and train sets
    for split in ['test', 'train']:
        renders_dir = Path(model_path) / split / f"ours_{iteration}" / "renders"
        gt_dir = Path(model_path) / split / f"ours_{iteration}" / "gt"
        
        if not renders_dir.exists() or not gt_dir.exists():
            print(f"⚠️  {split} directories not found, skipping...")
            continue
        
        print(f"\n📊 Evaluating {split} set...")
        
        # Get all rendered images
        render_files = sorted(list(renders_dir.glob("*.png")))
        split_metrics = []
        
        for render_file in tqdm(render_files, desc=f"Processing {split}"):
            gt_file = gt_dir / render_file.name
            
            if not gt_file.exists():
                print(f"⚠️  GT file not found: {gt_file}")
                continue
            
            try:
                metrics = compute_metrics(render_file, gt_file, lpips_fn)
                metrics['filename'] = render_file.name
                split_metrics.append(metrics)
            except Exception as e:
                print(f"❌ Error processing {render_file.name}: {e}")
                continue
        
        if split_metrics:
            # Compute summary statistics
            psnr_values = [m['psnr'] for m in split_metrics]
            ssim_values = [m['ssim'] for m in split_metrics]
            lpips_values = [m['lpips'] for m in split_metrics if m['lpips'] is not None]
            
            summary = {
                'count': len(split_metrics),
                'psnr_mean': np.mean(psnr_values),
                'psnr_std': np.std(psnr_values),
                'psnr_min': np.min(psnr_values),
                'psnr_max': np.max(psnr_values),
                'ssim_mean': np.mean(ssim_values),
                'ssim_std': np.std(ssim_values),
                'ssim_min': np.min(ssim_values),
                'ssim_max': np.max(ssim_values)
            }
            
            if lpips_values:
                summary.update({
                    'lpips_mean': np.mean(lpips_values),
                    'lpips_std': np.std(lpips_values),
                    'lpips_min': np.min(lpips_values),
                    'lpips_max': np.max(lpips_values)
                })
            
            results[split] = {
                'summary': summary,
                'per_image': split_metrics
            }
            
            # Print results
            print(f"\n📊 {split.upper()} SET RESULTS:")
            print(f"   📈 PSNR: {summary['psnr_mean']:.2f} ± {summary['ssim_std']:.2f} dB")
            print(f"   📈 SSIM: {summary['ssim_mean']:.4f} ± {summary['ssim_std']:.4f}")
            if 'lpips_mean' in summary:
                print(f"   📈 LPIPS: {summary['lpips_mean']:.4f} ± {summary['lpips_std']:.4f}")
            print(f"   📊 Images evaluated: {summary['count']}")
    
    # Save results
    results_file = Path(model_path) / f"metrics_evaluation_iter_{iteration}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Event-LangSplat metrics")
    parser.add_argument("-m", "--model_path", required=True, help="Path to model directory")
    parser.add_argument("--iteration", type=int, default=3000, help="Iteration to evaluate")
    
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.iteration)