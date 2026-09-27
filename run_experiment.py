"""
run_experiment.py

Main orchestrator for the cross-domain lung segmentation pipeline.

Executes all pipeline stages in order with checkpointing at each stage:
  Stage 0: Download and validate dataset
  Stage 1: Clean, split, compute normalization stats
  Stage 2: Train U-Net (5-fold CV + final model)
  Stage 3: Generate prompt boxes (U-Net inference on all images)
  Stage 4: Train SAM Freeze-Encoder
  Stage 5: (Optional) Train SAM LoRA
  Stage 6: Run all ablation evaluations (A1–A8)
  Stage 7: Generate visualizations
  Stage 8: Generate final report

Usage:
    python run_experiment.py --config configs/base.yaml --start_stage 0
    python run_experiment.py --config configs/base.yaml --start_stage 4  # resume from SAM

Each stage writes a .done marker to allow resuming from a specific stage.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Stage completion markers
# ─────────────────────────────────────────────────────────────────────────────

STAGE_MARKERS = {
    0: "download",
    1: "split_and_stats",
    2: "unet_training",
    3: "prompt_generation",
    4: "sam_freeze_training",
    5: "sam_lora_training",
    6: "ablation_evaluation",
    7: "visualization",
    8: "report",
}


def stage_done(marker_dir: str, stage: int) -> bool:
    """Return True iff this stage's .done marker file exists."""
    return (Path(marker_dir) / f".{STAGE_MARKERS[stage]}.done").exists()


def mark_done(marker_dir: str, stage: int) -> None:
    Path(marker_dir).mkdir(parents=True, exist_ok=True)
    (Path(marker_dir) / f".{STAGE_MARKERS[stage]}.done").touch()


def load_configs(base_config_path: str) -> Dict[str, Any]:
    """Load and merge all YAML configs."""
    from omegaconf import OmegaConf
    base = OmegaConf.load(base_config_path)
    unet_cfg = OmegaConf.load("configs/unet.yaml")
    freeze_cfg = OmegaConf.load("configs/sam_freeze.yaml")
    lora_cfg = OmegaConf.load("configs/sam_lora.yaml")
    return {
        "base": OmegaConf.to_container(base, resolve=True),
        "unet": OmegaConf.to_container(OmegaConf.merge(base, unet_cfg), resolve=True),
        "sam_freeze": OmegaConf.to_container(OmegaConf.merge(base, freeze_cfg), resolve=True),
        "sam_lora": OmegaConf.to_container(OmegaConf.merge(base, lora_cfg), resolve=True),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage implementations
# ─────────────────────────────────────────────────────────────────────────────

def run_stage_0_download(cfg: Dict, marker_dir: str) -> None:
    """Download and validate dataset."""
    from data.downloader import download_dataset
    from data.validator import validate_dataset

    logger = logging.getLogger("stage_0")
    data_root = cfg["base"]["paths"]["data_root"]

    if stage_done(marker_dir, 0):
        logger.info("Stage 0 already done. Skipping.")
        return

    logger.info("=== STAGE 0: Download & Validate ===")
    download_dataset(
        output_dir=data_root,
        kaggle_dataset=cfg["base"]["dataset"]["kaggle_dataset"],
    )

    report = validate_dataset(
        data_root=data_root,
        montgomery_prefix=cfg["base"]["dataset"]["montgomery_prefix"],
        shenzhen_prefix=cfg["base"]["dataset"]["shenzhen_prefix"],
        min_foreground_fraction=cfg["base"]["dataset"]["min_mask_foreground_fraction"],
    )

    if report.valid_images == 0:
        raise RuntimeError("No valid images found after validation!")

    logger.info(f"Validation complete: {report.valid_images} valid images.")
    mark_done(marker_dir, 0)


def run_stage_1_split_and_stats(cfg: Dict, marker_dir: str) -> pd.DataFrame:
    """Clean, split, compute norm stats and fallback box."""
    from data.cleaner import (
        build_clean_csv,
        compute_normalization_stats,
        compute_fallback_box_stats,
    )
    from data.split import assign_splits
    from data.validator import validate_dataset

    logger = logging.getLogger("stage_1")
    paths = cfg["base"]["paths"]
    dataset_cfg = cfg["base"]["dataset"]
    split_cfg = cfg["base"]["split"]

    splits_csv = Path(paths["splits_root"]) / "splits.csv"

    if stage_done(marker_dir, 1) and splits_csv.exists():
        logger.info("Stage 1 already done. Loading splits.")
        from data.split import load_splits
        return load_splits(str(splits_csv))

    logger.info("=== STAGE 1: Split & Statistics ===")

    report = validate_dataset(
        data_root=paths["data_root"],
        montgomery_prefix=dataset_cfg["montgomery_prefix"],
        shenzhen_prefix=dataset_cfg["shenzhen_prefix"],
    )

    raw_csv = Path(paths["processed_root"]) / "dataset.csv"
    df = build_clean_csv(report, str(raw_csv))

    split_df = assign_splits(
        df,
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
        split_seed=split_cfg["split_seed"],
        splits_dir=paths["splits_root"],
    )

    # In-domain source is normally "Shenzhen"; in degraded single-domain runs,
    # split.py records the effective in-domain source in split_meta.json.
    in_domain_source = "Shenzhen"
    _meta_path = Path(paths["splits_root"]) / "split_meta.json"
    if _meta_path.exists():
        try:
            _meta = json.loads(_meta_path.read_text())
            in_domain_source = _meta.get("in_domain_source", in_domain_source)
            if _meta.get("degraded_single_domain"):
                logger.warning(
                    "Degraded single-domain run: in-domain source = %s "
                    "(cross-domain evaluation will be unavailable).",
                    in_domain_source,
                )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Could not read split_meta.json (%s); using 'Shenzhen'.", e)

    compute_normalization_stats(
        split_df,
        native_size=dataset_cfg["native_size"],
        output_json_path=paths["norm_stats"],
        in_domain_source=in_domain_source,
    )

    compute_fallback_box_stats(
        split_df,
        unet_size=dataset_cfg["unet_size"],
        box_margin_fraction=cfg["base"]["prompt"]["box_margin_fraction"],
        output_json_path=paths["fallback_box_stats"],
        in_domain_source=in_domain_source,
    )

    mark_done(marker_dir, 1)
    return split_df


def run_stage_2_unet_training(
    cfg: Dict, split_df: pd.DataFrame, marker_dir: str, device: torch.device
) -> str:
    """Train U-Net: 5-fold CV + final model."""
    from training.unet_trainer import train_unet_kfold, train_unet_final

    logger = logging.getLogger("stage_2")
    paths = cfg["base"]["paths"]
    norm_stats = _load_json(paths["norm_stats"])
    seed = cfg["base"]["project"]["seed"]

    unet_final_ckpt = Path(paths["checkpoints_root"]) / "unet_final" / "best_model.pth"

    if stage_done(marker_dir, 2) and unet_final_ckpt.exists():
        logger.info("Stage 2 already done. Skipping U-Net training.")
        return str(unet_final_ckpt)

    logger.info("=== STAGE 2: U-Net Training ===")

    # 5-fold CV for robustness assessment
    fold_results = train_unet_kfold(
        df=split_df,
        cfg=cfg["unet"],
        norm_stats=norm_stats,
        checkpoint_base_dir=str(Path(paths["checkpoints_root"]) / "unet_cv"),
        log_dir=paths["logs_root"],
        device=device,
        seed=seed,
    )

    # Save CV results
    _save_json(
        fold_results,
        str(Path(paths["reports_root"]) / "unet_cv_results.json"),
    )

    # Final model (train+val)
    final_ckpt = train_unet_final(
        df=split_df,
        cfg=cfg["unet"],
        norm_stats=norm_stats,
        checkpoint_dir=str(Path(paths["checkpoints_root"]) / "unet_final"),
        log_dir=paths["logs_root"],
        device=device,
        seed=seed,
    )

    mark_done(marker_dir, 2)
    return final_ckpt


def run_stage_3_prompt_generation(
    cfg: Dict, split_df: pd.DataFrame, unet_ckpt: str,
    marker_dir: str, device: torch.device
) -> pd.DataFrame:
    """Run U-Net inference, extract boxes for all images."""
    from models.unet import build_unet
    from prompt_generation.box_extractor import (
        generate_prompt_cache,
        generate_oracle_prompt_cache,
        load_fallback_box_stats,
    )
    from utils.checkpoint import load_checkpoint

    logger = logging.getLogger("stage_3")
    paths = cfg["base"]["paths"]
    dataset_cfg = cfg["base"]["dataset"]

    prompt_csv = paths["prompt_cache"]
    oracle_csv = str(Path(paths["processed_root"]) / "oracle_boxes.csv")

    if stage_done(marker_dir, 3) and Path(prompt_csv).exists():
        logger.info("Stage 3 already done. Loading cached boxes.")
        box_df = pd.read_csv(prompt_csv)
        return box_df

    logger.info("=== STAGE 3: Prompt Box Generation ===")

    # Load final U-Net
    unet_cfg = cfg["unet"]
    model = build_unet(
        in_channels=unet_cfg["model"]["in_channels"],
        out_channels=unet_cfg["model"]["out_channels"],
        features=unet_cfg["model"]["features"],
        dropout=0.0,  # no dropout at inference
    ).to(device)
    load_checkpoint(unet_ckpt, model, device=device)
    model.eval()

    fallback_stats = load_fallback_box_stats(paths["fallback_box_stats"])

    # Pipeline boxes (from U-Net predictions)
    box_df = generate_prompt_cache(
        df=split_df,
        unet_model=model,
        fallback_box_stats=fallback_stats,
        device=device,
        unet_size=dataset_cfg["unet_size"],
        native_size=dataset_cfg["native_size"],
        margin_fraction=cfg["base"]["prompt"]["box_margin_fraction"],
        min_component_area_fraction=dataset_cfg["min_component_area_fraction"],
        output_csv_path=prompt_csv,
        # Fix #3: feed U-Net the SAME normalization used at train/val time.
        norm_stats_json_path=paths["norm_stats"],
    )

    # Oracle boxes (from GT masks) — for Ablation A7
    generate_oracle_prompt_cache(
        df=split_df,
        unet_size=dataset_cfg["unet_size"],
        native_size=dataset_cfg["native_size"],
        margin_fraction=cfg["base"]["prompt"]["box_margin_fraction"],
        output_csv_path=oracle_csv,
    )

    # Free U-Net from GPU before SAM training
    from utils.memory_monitor import free_model_memory
    free_model_memory(model)
    logger.info("U-Net freed from GPU.")

    mark_done(marker_dir, 3)
    return box_df


def run_stage_4_sam_freeze(
    cfg: Dict, split_df: pd.DataFrame, box_df: pd.DataFrame,
    marker_dir: str, device: torch.device
) -> str:
    """Fine-tune SAM with Freeze-Encoder."""
    from training.sam_trainer import train_sam

    logger = logging.getLogger("stage_4")
    paths = cfg["base"]["paths"]
    checkpoint_dir = str(Path(paths["checkpoints_root"]) / "sam_freeze")
    best_ckpt = str(Path(checkpoint_dir) / "best_model.pth")

    if stage_done(marker_dir, 4) and Path(best_ckpt).exists():
        logger.info("Stage 4 already done. Skipping SAM Freeze training.")
        return best_ckpt

    logger.info("=== STAGE 4: SAM Freeze-Encoder Training ===")

    # Add native_size to SAM config
    sam_cfg = cfg["sam_freeze"]
    sam_cfg["native_size"] = cfg["base"]["dataset"]["native_size"]

    best_ckpt = train_sam(
        df=split_df,
        box_cache_df=box_df,
        cfg=sam_cfg,
        checkpoint_dir=checkpoint_dir,
        log_dir=paths["logs_root"],
        device=device,
        seed=cfg["base"]["project"]["seed"],
    )

    mark_done(marker_dir, 4)
    return best_ckpt


def run_stage_5_sam_lora(
    cfg: Dict, split_df: pd.DataFrame, box_df: pd.DataFrame,
    marker_dir: str, device: torch.device
) -> Optional[str]:
    """Fine-tune SAM with LoRA (optional — bail on OOM)."""
    from training.sam_trainer import train_sam

    logger = logging.getLogger("stage_5")
    paths = cfg["base"]["paths"]
    checkpoint_dir = str(Path(paths["checkpoints_root"]) / "sam_lora")
    best_ckpt = str(Path(checkpoint_dir) / "best_model.pth")

    if stage_done(marker_dir, 5) and Path(best_ckpt).exists():
        logger.info("Stage 5 already done. Skipping SAM LoRA training.")
        return best_ckpt

    logger.info("=== STAGE 5: SAM LoRA Training (OPTIONAL) ===")
    logger.warning(
        "SAM LoRA requires ~8-10 GB VRAM. Will bail gracefully on OOM."
    )

    try:
        sam_cfg = cfg["sam_lora"]
        sam_cfg["native_size"] = cfg["base"]["dataset"]["native_size"]

        best_ckpt = train_sam(
            df=split_df,
            box_cache_df=box_df,
            cfg=sam_cfg,
            checkpoint_dir=checkpoint_dir,
            log_dir=paths["logs_root"],
            device=device,
            seed=cfg["base"]["project"]["seed"],
        )
        mark_done(marker_dir, 5)
        return best_ckpt
    except torch.cuda.OutOfMemoryError:
        logger.error(
            "OOM during SAM LoRA training. Stage 5 skipped. "
            "Primary results (Freeze-Encoder) are unaffected."
        )
        torch.cuda.empty_cache()
        return None


def run_stage_6_evaluation(
    cfg: Dict,
    split_df: pd.DataFrame,
    box_df: pd.DataFrame,
    unet_ckpt: str,
    freeze_ckpt: str,
    lora_ckpt: Optional[str],
    marker_dir: str,
    device: torch.device,
) -> Dict:
    """Run all ablation evaluations (A1–A8)."""
    from evaluation.evaluator import evaluate_unet, evaluate_sam
    from evaluation.cross_domain_eval import (
        build_ablation_comparison_table,
        answer_research_questions,
        save_full_results,
    )
    from models.unet import build_unet
    from models.sam_wrapper import SAMFineTuner, load_sam, configure_sam_freeze_encoder
    from prompt_generation.box_extractor import generate_oracle_prompt_cache
    from utils.checkpoint import load_checkpoint
    from utils.memory_monitor import free_model_memory

    logger = logging.getLogger("stage_6")
    paths = cfg["base"]["paths"]
    dataset_cfg = cfg["base"]["dataset"]
    norm_stats = _load_json(paths["norm_stats"])
    native_size = dataset_cfg["native_size"]

    results_path = Path(paths["reports_root"]) / "ablation_results_raw.json"

    if stage_done(marker_dir, 6) and results_path.exists():
        logger.info("Stage 6 already done.")
        with open(str(results_path)) as f:
            return json.load(f)

    logger.info("=== STAGE 6: Ablation Evaluation ===")

    all_metrics: Dict[str, Any] = {}

    # ── A1: U-Net baseline ───────────────────────────────────────────────────
    unet_model = build_unet(
        in_channels=cfg["unet"]["model"]["in_channels"],
        out_channels=cfg["unet"]["model"]["out_channels"],
        features=cfg["unet"]["model"]["features"],
        dropout=0.0,
    ).to(device)
    load_checkpoint(unet_ckpt, unet_model, device=device)
    unet_model.eval()

    all_metrics["A1"] = {
        "in_domain": evaluate_unet(
            unet_model, split_df, norm_stats, device,
            split_name="test", source_filter="Shenzhen",
        ),
        "cross_domain": evaluate_unet(
            unet_model, split_df, norm_stats, device,
            split_name="cross_domain_test", source_filter="Montgomery",
        ),
    }
    free_model_memory(unet_model)

    # ── A2: SAM Zero-Shot ────────────────────────────────────────────────────
    sam_model, processor = load_sam(
        model_id=cfg["sam_freeze"]["model"]["sam_model_id"], device=device
    )
    zero_shot_finetuner = SAMFineTuner(
        model=sam_model, processor=processor, device=device,
        use_cache=False, multimask_output=False,
    )
    all_metrics["A2"] = {
        "in_domain": evaluate_sam(
            zero_shot_finetuner, split_df, box_df, device, native_size,
            use_ensemble=False, split_name="test",
            source_filter="Shenzhen", ablation_id="A2",
        ),
        "cross_domain": evaluate_sam(
            zero_shot_finetuner, split_df, box_df, device, native_size,
            use_ensemble=False, split_name="cross_domain_test",
            source_filter="Montgomery", ablation_id="A2",
        ),
    }
    free_model_memory(sam_model)

    # ── A3/A4: SAM Freeze-Encoder ────────────────────────────────────────────
    sam_model_f, processor_f = load_sam(
        model_id=cfg["sam_freeze"]["model"]["sam_model_id"], device=device
    )
    configure_sam_freeze_encoder(sam_model_f)
    load_checkpoint(freeze_ckpt, sam_model_f, device=device, strict=False)
    sam_model_f.eval()

    freeze_finetuner = SAMFineTuner(
        model=sam_model_f, processor=processor_f, device=device,
        use_cache=False, multimask_output=False,
    )

    for ablation_id, use_ens in [("A3", False), ("A4", True)]:
        all_metrics[ablation_id] = {
            "in_domain": evaluate_sam(
                freeze_finetuner, split_df, box_df, device, native_size,
                use_ensemble=use_ens, split_name="test",
                source_filter="Shenzhen", ablation_id=ablation_id,
            ),
            "cross_domain": evaluate_sam(
                freeze_finetuner, split_df, box_df, device, native_size,
                use_ensemble=use_ens, split_name="cross_domain_test",
                source_filter="Montgomery", ablation_id=ablation_id,
            ),
        }

    # ── A7: Oracle Bound ─────────────────────────────────────────────────────
    oracle_csv = str(Path(paths["processed_root"]) / "oracle_boxes.csv")
    if Path(oracle_csv).exists():
        oracle_box_df = pd.read_csv(oracle_csv)
        all_metrics["A7"] = {
            "in_domain": evaluate_sam(
                freeze_finetuner, split_df, box_df, device, native_size,
                use_ensemble=False, split_name="test",
                source_filter="Shenzhen", ablation_id="A7",
                oracle_box_df=oracle_box_df,
            ),
            "cross_domain": evaluate_sam(
                freeze_finetuner, split_df, box_df, device, native_size,
                use_ensemble=False, split_name="cross_domain_test",
                source_filter="Montgomery", ablation_id="A7",
                oracle_box_df=oracle_box_df,
            ),
        }

    free_model_memory(sam_model_f)
    # ── A8: Stress-test box (sensitivity analysis) ──────────────────────────
    # Re-load freeze model for A8 (already freed above; need to reload)
    sam_model_a8, proc_a8 = load_sam(
        model_id=cfg["sam_freeze"]["model"]["sam_model_id"], device=device
    )
    configure_sam_freeze_encoder(sam_model_a8)
    load_checkpoint(freeze_ckpt, sam_model_a8, device=device, strict=False)
    sam_model_a8.eval()

    stress_finetuner = SAMFineTuner(
        model=sam_model_a8, processor=proc_a8, device=device,
        use_cache=False, multimask_output=False,
    )

    from prompt_generation.box_extractor import get_stress_test_box
    from data.transforms import resize_mask_pil

    def _make_stress_df(target_df, native_size=512, unet_size=256, seed=42):
        """Build a box_df with stress-test (corrupted) boxes from GT masks."""
        rows = []
        for _, row in target_df.iterrows():
            gt_mask = resize_mask_pil(row["mask_path"], unet_size)
            stress_box = get_stress_test_box(gt_mask, error_scale=0.25,
                                              native_size=native_size, seed=seed)
            rows.append({
                "image_path": row["image_path"],
                "x1_native": float(stress_box[0]),
                "y1_native": float(stress_box[1]),
                "x2_native": float(stress_box[2]),
                "y2_native": float(stress_box[3]),
            })
        return pd.DataFrame(rows)

    stress_in_df = _make_stress_df(
        split_df[split_df["split"] == "test"].reset_index(drop=True)
    )
    stress_xd_df = _make_stress_df(
        split_df[split_df["split"] == "cross_domain_test"].reset_index(drop=True)
    )

    all_metrics["A8"] = {
        "in_domain": evaluate_sam(
            stress_finetuner, split_df, stress_in_df, device, native_size,
            use_ensemble=False, split_name="test",
            source_filter="Shenzhen", ablation_id="A8",
        ),
        "cross_domain": evaluate_sam(
            stress_finetuner, split_df, stress_xd_df, device, native_size,
            use_ensemble=False, split_name="cross_domain_test",
            source_filter="Montgomery", ablation_id="A8",
        ),
    }
    free_model_memory(sam_model_a8)


    # ── A5/A6: SAM LoRA ──────────────────────────────────────────────────────
    if lora_ckpt and Path(lora_ckpt).exists():
        from models.lora_sam import inject_lora_into_sam, enable_gradient_checkpointing_sam
        sam_lora, proc_lora = load_sam(
            model_id=cfg["sam_lora"]["model"]["sam_model_id"], device=device
        )
        lora_cfg_m = cfg["sam_lora"]["model"]["lora"]
        sam_lora = inject_lora_into_sam(
            sam_lora, rank=lora_cfg_m["rank"], alpha=lora_cfg_m["alpha"]
        )
        load_checkpoint(lora_ckpt, sam_lora, device=device, strict=False)
        sam_lora.eval()
        lora_finetuner = SAMFineTuner(
            model=sam_lora, processor=proc_lora, device=device,
            use_cache=False, multimask_output=False,
        )
        for ablation_id, use_ens in [("A5", False), ("A6", True)]:
            all_metrics[ablation_id] = {
                "in_domain": evaluate_sam(
                    lora_finetuner, split_df, box_df, device, native_size,
                    use_ensemble=use_ens, split_name="test",
                    source_filter="Shenzhen", ablation_id=ablation_id,
                ),
                "cross_domain": evaluate_sam(
                    lora_finetuner, split_df, box_df, device, native_size,
                    use_ensemble=use_ens, split_name="cross_domain_test",
                    source_filter="Montgomery", ablation_id=ablation_id,
                ),
            }
        free_model_memory(sam_lora)
    else:
        logger.warning("LoRA checkpoint not available. A5/A6 skipped.")

    # Save raw metrics (serializing per-image lists separately)
    serializable = {
        k: {
            domain: {
                mk: mv for mk, mv in dm.items()
                if not isinstance(mv, list)  # exclude per-image lists from JSON
            }
            for domain, dm in v.items()
        }
        for k, v in all_metrics.items()
    }
    _save_json(serializable, str(results_path))

    # Build comparison table and answer RQs
    from evaluation.cross_domain_eval import build_ablation_comparison_table, answer_research_questions, save_full_results
    comparison_table = build_ablation_comparison_table(all_metrics)
    rq_answers = answer_research_questions(comparison_table, all_metrics)
    save_full_results(comparison_table, rq_answers, all_metrics, paths["reports_root"])

    mark_done(marker_dir, 6)
    return all_metrics


def run_stage_8_report(
    cfg: Dict,
    all_metrics: Dict,
    memory_report: Optional[Dict],
    marker_dir: str,
) -> None:
    """Generate final Markdown report."""
    from evaluation.cross_domain_eval import (
        build_ablation_comparison_table,
        answer_research_questions,
    )
    from reports.report_generator import generate_report

    logger = logging.getLogger("stage_8")
    paths = cfg["base"]["paths"]

    if stage_done(marker_dir, 8):
        logger.info("Stage 8 already done.")
        return

    logger.info("=== STAGE 8: Report Generation ===")

    comparison_table = build_ablation_comparison_table(all_metrics)
    rq_answers = answer_research_questions(comparison_table, all_metrics)

    report_path = generate_report(
        comparison_table=comparison_table,
        rq_answers=rq_answers,
        all_metrics=all_metrics,
        memory_report=memory_report,
        output_dir=paths["reports_root"],
    )
    logger.info(f"Final report: {report_path}")
    mark_done(marker_dir, 8)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run lung segmentation pipeline")
    parser.add_argument(
        "--config", default="configs/base.yaml", help="Base config YAML path"
    )
    parser.add_argument(
        "--start_stage", type=int, default=0,
        help="Stage to start from (0=download, skip already-done stages)"
    )
    parser.add_argument(
        "--skip_lora", action="store_true",
        help="Skip SAM LoRA training (Stage 5)"
    )
    args = parser.parse_args()

    cfg = load_configs(args.config)
    paths = cfg["base"]["paths"]
    marker_dir = paths["checkpoints_root"]

    from utils.logging_utils import setup_logging
    from utils.reproducibility import seed_everything
    from utils.memory_monitor import MemoryMonitor

    setup_logging(
        log_dir=paths["logs_root"],
        experiment_name="run_experiment",
        level=cfg["base"]["logging"]["level"],
    )
    logger = logging.getLogger("main")
    seed_everything(cfg["base"]["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(
            f"GPU: {torch.cuda.get_device_name(0)} | "
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    memory_monitor = MemoryMonitor(limit_gb=14.0)

    # Run pipeline stages
    if args.start_stage <= 0:
        run_stage_0_download(cfg, marker_dir)

    split_df = run_stage_1_split_and_stats(cfg, marker_dir)

    unet_ckpt = None
    if args.start_stage <= 2:
        with memory_monitor.track("unet_training"):
            unet_ckpt = run_stage_2_unet_training(cfg, split_df, marker_dir, device)
    else:
        unet_ckpt = str(
            Path(paths["checkpoints_root"]) / "unet_final" / "best_model.pth"
        )

    box_df = None
    if args.start_stage <= 3:
        with memory_monitor.track("prompt_generation"):
            box_df = run_stage_3_prompt_generation(
                cfg, split_df, unet_ckpt, marker_dir, device
            )
    else:
        box_df = pd.read_csv(paths["prompt_cache"])

    freeze_ckpt = None
    if args.start_stage <= 4:
        with memory_monitor.track("sam_freeze_training"):
            freeze_ckpt = run_stage_4_sam_freeze(
                cfg, split_df, box_df, marker_dir, device
            )
    else:
        freeze_ckpt = str(
            Path(paths["checkpoints_root"]) / "sam_freeze" / "best_model.pth"
        )

    lora_ckpt = None
    if not args.skip_lora and args.start_stage <= 5:
        with memory_monitor.track("sam_lora_training"):
            lora_ckpt = run_stage_5_sam_lora(
                cfg, split_df, box_df, marker_dir, device
            )

    all_metrics = {}
    if args.start_stage <= 6:
        with memory_monitor.track("ablation_evaluation"):
            all_metrics = run_stage_6_evaluation(
                cfg, split_df, box_df,
                unet_ckpt, freeze_ckpt, lora_ckpt,
                marker_dir, device,
            )

    memory_report = memory_monitor.report()

    if args.start_stage <= 8:
        run_stage_8_report(cfg, all_metrics, memory_report, marker_dir)

    logger.info("Pipeline complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _save_json(data: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
