"""
notebooks/colab_execution_guide.py

Complete step-by-step Google Colab execution guide.
Each step has: exact commands, expected outputs, expected runtime, troubleshooting.

HOW TO USE:
  Copy each ## STEP block into a separate Colab cell and run sequentially.
  Monitor GPU usage: Runtime → Manage Sessions → GPU usage indicator.
  If session disconnects, resume from the last completed step.

TARGET ENVIRONMENT:
  Google Colab free tier with T4 GPU (12-16 GB VRAM, 12-14 GB usable)
  Python 3.10+, CUDA 11.8+
"""

# ============================================================
# STEP 0: Mount Google Drive (for persistent checkpoints)
# ============================================================
"""
# Expected runtime: 30 seconds
# Expected output: Drive mounted at /content/drive

from google.colab import drive
drive.mount('/content/drive')

import os
os.makedirs('/content/drive/MyDrive/lung_seg', exist_ok=True)
print("Drive mounted. All checkpoints will save to /content/drive/MyDrive/lung_seg")
"""

# ============================================================
# STEP 1: Clone project and install dependencies
# ============================================================
"""
# Expected runtime: 2-3 minutes
# Expected output: All packages installed, no errors

# Option A: from local upload
import os
os.makedirs('/content/lung_seg', exist_ok=True)
# Upload your zip via Files panel, then:
# !unzip /content/lung_seg_project.zip -d /content/lung_seg/

# Option B: create the project directory structure (copy all .py files manually)
# (assumes you've uploaded all files to /content/lung_seg/)

os.chdir('/content/lung_seg')
!pip install -q -r requirements.txt

# Verify critical packages
import torch
import transformers
import albumentations
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram_gb:.1f} GB")
    if vram_gb < 10:
        print("WARNING: Less than 10 GB VRAM detected. LoRA path may OOM.")
print(f"Transformers: {transformers.__version__}")
"""

# ============================================================
# STEP 2: Configure Kaggle credentials
# ============================================================
"""
# Expected runtime: 10 seconds
# Expected output: "Kaggle credentials set"

# METHOD 1 (recommended): Use Colab Secrets
# Go to: Settings (gear icon) → Secrets → Add new secret
# Add: KAGGLE_USERNAME = your_username
#       KAGGLE_KEY = your_api_key

from google.colab import userdata
import os

try:
    os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
    os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')
    print("Kaggle credentials set from Colab Secrets.")
except Exception:
    # METHOD 2: Upload kaggle.json manually
    from google.colab import files
    print("Upload your kaggle.json file:")
    uploaded = files.upload()
    import json, os
    kaggle_dir = os.path.expanduser('~/.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)
    with open(os.path.join(kaggle_dir, 'kaggle.json'), 'w') as f:
        f.write(list(uploaded.values())[0].decode())
    os.chmod(os.path.join(kaggle_dir, 'kaggle.json'), 0o600)
    print("kaggle.json installed.")
"""

# ============================================================
# STEP 3: Download dataset (Stage 0)
# ============================================================
"""
# Expected runtime: 3-5 minutes
# Expected output: "Dataset extraction complete."
# Disk usage: ~1.5 GB

import sys
sys.path.insert(0, '/content/lung_seg')

from utils.logging_utils import setup_logging
from utils.reproducibility import seed_everything

setup_logging('./logs', 'colab_run')
seed_everything(42)

from data.downloader import download_dataset
download_dataset(output_dir='./data/raw')

# TROUBLESHOOTING:
# If "403 Forbidden": Your Kaggle credentials are invalid. Re-check username and key.
# If "Connection timeout": Colab's network is slow. Retry in 60 seconds.
# If already downloaded: Shows "Dataset already extracted" — no problem.
"""

# ============================================================
# STEP 4: Validate, clean, split, compute statistics (Stage 1)
# ============================================================
"""
# Expected runtime: 2-5 minutes
# Expected output: Split counts, normalization stats, fallback box stats

from data.validator import validate_dataset
from data.cleaner import build_clean_csv, compute_normalization_stats, compute_fallback_box_stats
from data.split import assign_splits

# Validate
report = validate_dataset(
    data_root='./data/raw',
    montgomery_prefix='MCUCXR_',
    shenzhen_prefix='CHNCXR_',
)
print(report.summary())

# Clean
from pathlib import Path
Path('./data/processed').mkdir(parents=True, exist_ok=True)
df = build_clean_csv(report, './data/processed/dataset.csv')

# Split
split_df = assign_splits(df, splits_dir='./data/splits')
print("Split counts:")
print(split_df.groupby(['source', 'split']).size())

# Statistics (MUST run AFTER split, uses only Shenzhen-train)
norm_stats = compute_normalization_stats(split_df, output_json_path='./data/processed/norm_stats.json')
fallback_box = compute_fallback_box_stats(split_df, output_json_path='./data/processed/fallback_box_stats.json')

# TROUBLESHOOTING:
# "No Shenzhen training images found": Split wasn't assigned. Check split.csv.
# "DATA LEAKAGE DETECTED": A bug in split logic. STOP and report.
"""

# ============================================================
# STEP 5: Train U-Net (5-fold CV + final model) (Stage 2)
# ============================================================
"""
# Expected runtime: 45-90 minutes
# Expected output: Per-fold Dice scores, best final model checkpoint
# Expected VRAM: ~2.5 GB

import torch
device = torch.device('cuda')

from omegaconf import OmegaConf
unet_cfg = OmegaConf.to_container(OmegaConf.merge(
    OmegaConf.load('configs/base.yaml'),
    OmegaConf.load('configs/unet.yaml')
), resolve=True)

import json
norm_stats = json.load(open('./data/processed/norm_stats.json'))

from training.unet_trainer import train_unet_kfold, train_unet_final

# 5-fold CV
fold_results = train_unet_kfold(
    df=split_df, cfg=unet_cfg, norm_stats=norm_stats,
    checkpoint_base_dir='./checkpoints/unet_cv',
    log_dir='./logs', device=device, seed=42
)
import json
json.dump(fold_results, open('./reports/output/unet_cv_results.json', 'w'), indent=2)
print(f"CV Dice: {sum(fold_results['dice'])/len(fold_results['dice']):.4f}")

# Final model
unet_ckpt = train_unet_final(
    df=split_df, cfg=unet_cfg, norm_stats=norm_stats,
    checkpoint_dir='./checkpoints/unet_final',
    log_dir='./logs', device=device, seed=42
)
print(f"Final U-Net checkpoint: {unet_ckpt}")

# Copy to Drive
import shutil
shutil.copy(unet_ckpt, '/content/drive/MyDrive/lung_seg/unet_best.pth')
print("Saved to Drive.")

# TROUBLESHOOTING:
# OOM: Reduce unet.yaml batch_size from 8 to 4.
# Dice not improving: Check normalization stats are correct.
"""

# ============================================================
# STEP 6: Generate prompt boxes (Stage 3)
# ============================================================
"""
# Expected runtime: 5-10 minutes
# Expected output: prompt_boxes.csv with fallback_rate < 10%

from models.unet import build_unet
from utils.checkpoint import load_checkpoint
from prompt_generation.box_extractor import (
    generate_prompt_cache, generate_oracle_prompt_cache,
    load_fallback_box_stats
)
from utils.memory_monitor import free_model_memory

unet_model = build_unet(features=[32, 64, 128, 256]).to(device)
load_checkpoint('./checkpoints/unet_final/best_model.pth', unet_model, device=device)

fallback_stats = load_fallback_box_stats('./data/processed/fallback_box_stats.json')

box_df = generate_prompt_cache(
    df=split_df, unet_model=unet_model, fallback_box_stats=fallback_stats,
    device=device, output_csv_path='./data/processed/prompt_boxes.csv'
)
generate_oracle_prompt_cache(
    df=split_df, output_csv_path='./data/processed/oracle_boxes.csv'
)

free_model_memory(unet_model)
print("U-Net freed from GPU.")
print(f"Fallback rate: {box_df['used_fallback'].mean()*100:.1f}%")

# TROUBLESHOOTING:
# fallback_rate > 30%: U-Net is predicting empty masks. Check U-Net training converged.
# Memory error: Restart runtime (Runtime → Restart runtime) and reload checkpoint.
"""

# ============================================================
# STEP 7: Train SAM Freeze-Encoder (Stage 4)
# ============================================================
"""
# Expected runtime: 60-120 minutes
# Expected VRAM: ~6-8 GB
# Expected output: best_model.pth in checkpoints/sam_freeze/

import pandas as pd
from omegaconf import OmegaConf
box_df = pd.read_csv('./data/processed/prompt_boxes.csv')

sam_cfg = OmegaConf.to_container(OmegaConf.merge(
    OmegaConf.load('configs/base.yaml'),
    OmegaConf.load('configs/sam_freeze.yaml')
), resolve=True)
sam_cfg['native_size'] = 512

from training.sam_trainer import train_sam
freeze_ckpt = train_sam(
    df=split_df, box_cache_df=box_df, cfg=sam_cfg,
    checkpoint_dir='./checkpoints/sam_freeze',
    log_dir='./logs', device=device, seed=42
)
import shutil
shutil.copy(freeze_ckpt, '/content/drive/MyDrive/lung_seg/sam_freeze_best.pth')
print(f"SAM Freeze checkpoint: {freeze_ckpt}")

# TROUBLESHOOTING:
# OOM: Reduce sam_freeze.yaml batch_size to 2.
# Session disconnects: Restart and reload from latest checkpoint (auto-resume).
"""

# ============================================================
# STEP 8: (Optional) Train SAM LoRA (Stage 5)
# ============================================================
"""
# Expected runtime: 90-180 minutes
# Expected VRAM: ~8-10 GB (RISKY on 12 GB — may OOM)
# Expected output: best_model.pth in checkpoints/sam_lora/ (if successful)

sam_lora_cfg = OmegaConf.to_container(OmegaConf.merge(
    OmegaConf.load('configs/base.yaml'),
    OmegaConf.load('configs/sam_lora.yaml')
), resolve=True)
sam_lora_cfg['native_size'] = 512

try:
    lora_ckpt = train_sam(
        df=split_df, box_cache_df=box_df, cfg=sam_lora_cfg,
        checkpoint_dir='./checkpoints/sam_lora',
        log_dir='./logs', device=device, seed=42
    )
    shutil.copy(lora_ckpt, '/content/drive/MyDrive/lung_seg/sam_lora_best.pth')
    print(f"SAM LoRA checkpoint: {lora_ckpt}")
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    lora_ckpt = None
    print("OOM during LoRA training. Ablations A5/A6 will be marked FAILED.")
    print("Primary results (A1-A4, A7, A8) are unaffected.")
"""

# ============================================================
# STEP 9: Run all ablation evaluations (Stage 6)
# ============================================================
"""
# Expected runtime: 30-60 minutes
# Expected output: ablation_comparison.csv, rq_answers.json

# This step runs all 8 ablation configurations and computes
# in-domain and cross-domain metrics for each.
# It automatically skips unavailable ablations (e.g. if LoRA failed).

# Load and run evaluations via run_experiment.py
!python run_experiment.py --config configs/base.yaml --start_stage 6

# OR run manually:
from evaluation.cross_domain_eval import build_ablation_comparison_table, answer_research_questions
import json
with open('./reports/output/ablation_results_raw.json') as f:
    all_metrics = json.load(f)

# The comparison table shows all ablations side-by-side
import pandas as pd
table = pd.read_csv('./reports/output/ablation_comparison.csv')
print(table[['ablation_id', 'ablation_label', 'dice_in', 'dice_xd', 'delta_dice']].to_string())

# TROUBLESHOOTING:
# KeyError 'A5': LoRA results not available (expected if Stage 5 failed)
# OOM during evaluation: Reduce evaluator batch_size to 1
"""

# ============================================================
# STEP 10: Generate final report (Stage 8)
# ============================================================
"""
# Expected runtime: 2 minutes
# Expected output: final_report.md in reports/output/

!python run_experiment.py --config configs/base.yaml --start_stage 8

# Download report
from google.colab import files
files.download('./reports/output/final_report.md')
files.download('./reports/output/ablation_comparison.csv')
files.download('./reports/output/rq_answers.json')

print("Report generation complete!")
print("Files downloaded to your local machine.")
"""

# ============================================================
# RESUME GUIDE (after Colab disconnects)
# ============================================================
"""
# After disconnection, run these cells in order:

# Cell 1: Remount Drive
from google.colab import drive
drive.mount('/content/drive')
import os; os.chdir('/content/lung_seg')

# Cell 2: Reinstall packages (always needed after disconnect)
!pip install -q -r requirements.txt

# Cell 3: Set credentials
import os
from google.colab import userdata
os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')

# Cell 4: Load split_df and box_df from CSV (already computed)
import pandas as pd
split_df = pd.read_csv('./data/splits/splits.csv')
box_df = pd.read_csv('./data/processed/prompt_boxes.csv')
print(f"Loaded: {len(split_df)} records")

# Cell 5: Resume from last completed stage
import torch
device = torch.device('cuda')

# Copy checkpoints back from Drive (if needed)
import shutil
shutil.copy('/content/drive/MyDrive/lung_seg/unet_best.pth', './checkpoints/unet_final/best_model.pth')
shutil.copy('/content/drive/MyDrive/lung_seg/sam_freeze_best.pth', './checkpoints/sam_freeze/best_model.pth')

# Resume from SAM evaluation
!python run_experiment.py --config configs/base.yaml --start_stage 6
"""

print("Colab execution guide loaded.")
print("Copy each ## STEP block into a separate Colab cell and run sequentially.")
