"""
visualization/plots.py

Visualization utilities for segmentation results and ablation analysis.

Generates:
  - Mask overlay plots (prediction vs ground truth on CXR image)
  - Ablation comparison bar charts
  - Performance Degradation (Δ) comparison bar charts
  - Cross-domain metric heatmaps

Audit fix: LOW-24 — batch-save visualizations to avoid RAM overflow.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Color scheme (colorblind-friendly)
COLORS = {
    "gt_mask": (0.2, 0.8, 0.2, 0.4),        # transparent green
    "pred_mask": (0.9, 0.1, 0.1, 0.4),        # transparent red
    "overlap": (0.2, 0.2, 0.9, 0.5),           # transparent blue
    "unet": "#2196F3",
    "sam_freeze": "#4CAF50",
    "sam_lora": "#FF9800",
}


def save_mask_overlays(
    image_paths: List[str],
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    output_dir: str,
    prefix: str = "overlay",
    max_images: int = 20,
    native_size: int = 512,
) -> None:
    """
    Save overlay images: CXR background + GT mask + predicted mask.

    Green = GT only, Red = Prediction only, Blue = Overlap (both).

    Processes images one at a time to avoid RAM overflow.

    Audit fix: LOW-24 — batch file writing, not in-memory accumulation.

    Args:
        image_paths: List of CXR image file paths.
        pred_masks: List of (H, W) binary float32 predicted masks.
        gt_masks: List of (H, W) binary float32 ground-truth masks.
        output_dir: Directory to save PNG files.
        prefix: Filename prefix.
        max_images: Maximum number of overlays to save.
        native_size: Resolution for display.
    """
    from PIL import Image as PILImage
    from data.transforms import resize_image_pil

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    n_save = min(max_images, len(image_paths))

    for i in range(n_save):
        img_rgb = resize_image_pil(image_paths[i], native_size, to_rgb=True)
        pred = pred_masks[i]
        gt = gt_masks[i]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"{Path(image_paths[i]).stem}", fontsize=10)

        # Original image
        axes[0].imshow(img_rgb)
        axes[0].set_title("Input CXR", fontsize=9)
        axes[0].axis("off")

        # GT mask overlay
        axes[1].imshow(img_rgb)
        gt_rgba = np.zeros((*gt.shape, 4), dtype=np.float32)
        gt_rgba[gt > 0.5] = [0.2, 0.8, 0.2, 0.5]
        axes[1].imshow(gt_rgba)
        axes[1].set_title("Ground Truth", fontsize=9)
        axes[1].axis("off")

        # Prediction overlay with color-coded regions
        axes[2].imshow(img_rgb)
        overlay = _make_three_color_overlay(pred, gt)
        axes[2].imshow(overlay, alpha=0.5)
        axes[2].set_title("Prediction (Red=FP, Green=FN, Blue=TP)", fontsize=9)
        axes[2].axis("off")

        plt.tight_layout()
        out_path = Path(output_dir) / f"{prefix}_{i:04d}_{Path(image_paths[i]).stem}.png"
        plt.savefig(str(out_path), dpi=100, bbox_inches="tight")
        plt.close(fig)

    logger.info(f"Saved {n_save} overlay images to {output_dir}")


def plot_ablation_comparison(
    comparison_table: pd.DataFrame,
    metric: str = "dice_in",
    output_path: str = "./reports/output/ablation_comparison.png",
    title: str = "Ablation Study: In-Domain Dice",
) -> None:
    """
    Bar chart comparing all ablations on a specified metric.

    Args:
        comparison_table: Output of cross_domain_eval.build_ablation_comparison_table().
        metric: Column name to plot.
        output_path: Where to save the figure.
        title: Plot title.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df = comparison_table.dropna(subset=[metric]).copy()
    df = df.sort_values("ablation_id")

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(df)))

    bars = ax.bar(
        df["ablation_label"],
        df[metric],
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85,
    )

    # Annotate bars with values
    for bar, val in zip(bars, df[metric]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.002,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xlabel("Ablation Configuration", fontsize=11)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, min(1.05, df[metric].max() + 0.05))
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Ablation comparison chart saved: {output_path}")


def plot_degradation_comparison(
    comparison_table: pd.DataFrame,
    output_path: str = "./reports/output/degradation_comparison.png",
) -> None:
    """
    Bar chart showing Performance Degradation Δ for each ablation.

    Smaller bar = less degradation = better cross-domain robustness.
    Positive Δ = worse on Montgomery (expected). Negative Δ = surprising.

    Args:
        comparison_table: Output of build_ablation_comparison_table().
        output_path: Where to save figure.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df = comparison_table.dropna(subset=["delta_dice"]).copy()
    df = df.sort_values("ablation_id")

    fig, ax = plt.subplots(figsize=(12, 5))

    bar_colors = [
        "#F44336" if val > 0.05 else "#FF9800" if val > 0.02 else "#4CAF50"
        for val in df["delta_dice"]
    ]
    bars = ax.bar(
        df["ablation_label"],
        df["delta_dice"],
        color=bar_colors,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    for bar, val in zip(bars, df["delta_dice"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.001,
            f"{val:+.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xlabel("Ablation Configuration", fontsize=11)
    ax.set_ylabel("Δ Dice (In-Domain − Cross-Domain)", fontsize=11)
    ax.set_title(
        "Performance Degradation under Domain Shift\n"
        "(smaller = better cross-domain robustness)",
        fontsize=12, fontweight="bold",
    )

    # Legend
    patches = [
        mpatches.Patch(color="#4CAF50", label="Low degradation (Δ < 0.02)"),
        mpatches.Patch(color="#FF9800", label="Moderate degradation (0.02–0.05)"),
        mpatches.Patch(color="#F44336", label="High degradation (Δ > 0.05)"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=9)

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Degradation comparison chart saved: {output_path}")


def plot_training_curves(
    metrics_csv_path: str,
    output_path: str,
    experiment_name: str = "Training",
) -> None:
    """
    Plot training loss and validation Dice over epochs.

    Args:
        metrics_csv_path: Path to experiment logger CSV.
        output_path: Where to save the figure.
        experiment_name: Title label.
    """
    df = pd.read_csv(metrics_csv_path)
    if df.empty:
        logger.warning(f"Empty metrics CSV: {metrics_csv_path}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{experiment_name} Training Curves", fontsize=12)

    # Loss
    if "total_loss" in df.columns:
        axes[0].plot(df["epoch"], df["total_loss"], label="Total Loss", color="#1976D2")
        if "dice_loss" in df.columns:
            axes[0].plot(df["epoch"], df["dice_loss"], "--", label="Dice Loss", alpha=0.7)
        if "bce_loss" in df.columns:
            axes[0].plot(df["epoch"], df["bce_loss"], ":", label="BCE Loss", alpha=0.7)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

    # Validation Dice
    for col, label, color in [
        ("val_dice", "Val Dice", "#4CAF50"),
        ("val_iou", "Val IoU", "#FF9800"),
    ]:
        if col in df.columns:
            axes[1].plot(df["epoch"], df[col], label=label, color=color)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Metrics")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Training curves saved: {output_path}")


def _make_three_color_overlay(
    pred: np.ndarray, gt: np.ndarray
) -> np.ndarray:
    """
    Create RGBA overlay:
        Blue  = True Positive (both pred and gt are 1)
        Red   = False Positive (pred=1, gt=0)
        Green = False Negative (pred=0, gt=1)
    """
    h, w = pred.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)

    tp = (pred > 0.5) & (gt > 0.5)
    fp = (pred > 0.5) & (gt <= 0.5)
    fn = (pred <= 0.5) & (gt > 0.5)

    overlay[tp] = [0.2, 0.2, 1.0, 0.6]   # Blue: TP
    overlay[fp] = [1.0, 0.2, 0.2, 0.6]   # Red:  FP
    overlay[fn] = [0.2, 1.0, 0.2, 0.6]   # Green: FN
    return overlay
