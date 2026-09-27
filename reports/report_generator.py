"""
reports/report_generator.py

Generates the final project report from all evaluation results.

Report format:
  - Markdown with tables and section headers (readable in any text viewer)
  - ablation_comparison.csv (machine-readable, importable to pandas)
  - rq_answers.json (structured RQ responses)
  - summary_metrics.json (key numbers for quick lookup)

Audit fix: LOW-25 — graceful partial reporting if some ablations failed.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_report(
    comparison_table: pd.DataFrame,
    rq_answers: Dict[str, str],
    all_metrics: Dict[str, Any],
    memory_report: Optional[Dict] = None,
    output_dir: str = "./reports/output",
    experiment_name: str = "cross_domain_lung_segmentation",
) -> str:
    """
    Generate the complete final research report in Markdown format.

    Handles missing ablations gracefully (marks them as FAILED/OOM
    rather than crashing).

    Audit fix: LOW-25 — partial results handling.

    Args:
        comparison_table: Ablation comparison DataFrame.
        rq_answers: Dict of RQ label → answer string.
        all_metrics: Raw metrics dict for all ablations.
        memory_report: Optional MemoryMonitor.report() dict.
        output_dir: Directory to write report files.
        experiment_name: Label for the report.

    Returns:
        Path to the generated Markdown report.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md_path = Path(output_dir) / "final_report.md"

    lines = []

    # ── Header ──────────────────────────────────────────────────────────────
    lines += [
        f"# Research Report: {experiment_name}",
        f"",
        f"**Generated:** {timestamp}",
        f"",
        "---",
        "",
    ]

    # ── Executive Summary ────────────────────────────────────────────────────
    lines += [
        "## Executive Summary",
        "",
        "This report presents the complete results of the cross-domain lung "
        "segmentation study comparing U-Net (trained from scratch) with "
        "SAM (ViT-B) fine-tuned via Parameter-Efficient Fine-Tuning (PEFT) "
        "on the Montgomery and Shenzhen chest X-ray datasets.",
        "",
    ]

    # ── Ablation Comparison Table ────────────────────────────────────────────
    lines += [
        "## Ablation Study Results",
        "",
        "### All Ablation Configurations",
        "",
    ]

    if comparison_table.empty:
        lines.append("*No ablation results available.*")
    else:
        lines.append(_df_to_markdown(
            comparison_table[[
                "ablation_id", "ablation_label",
                "dice_in", "iou_in", "hd95_in",
                "dice_xd", "iou_xd", "hd95_xd",
                "delta_dice",
            ]].round(4),
            headers={
                "ablation_id": "ID",
                "ablation_label": "Configuration",
                "dice_in": "Dice (In-Domain)",
                "iou_in": "IoU (In-Domain)",
                "hd95_in": "HD95 (In)",
                "dice_xd": "Dice (Montgomery)",
                "iou_xd": "IoU (Montgomery)",
                "hd95_xd": "HD95 (XD)",
                "delta_dice": "Δ Dice ↓",
            }
        ))
    lines.append("")

    # ── Performance Degradation Section ─────────────────────────────────────
    lines += [
        "### Performance Degradation Summary",
        "",
        "> **Δ Dice** = In-Domain Dice − Cross-Domain Dice.",
        "> Positive Δ = worse on Montgomery. Smaller Δ = better robustness.",
        "",
    ]

    if not comparison_table.empty:
        best_robustness = comparison_table.loc[
            comparison_table["delta_dice"].abs().idxmin()
        ]
        lines.append(
            f"**Best cross-domain robustness:** {best_robustness['ablation_label']} "
            f"(Δ = {best_robustness['delta_dice']:+.4f})"
        )
    lines.append("")

    # ── Research Questions ───────────────────────────────────────────────────
    lines += ["## Research Question Answers", ""]
    for rq, answer in rq_answers.items():
        lines += [f"### {rq}", "", answer, ""]

    # ── Memory Usage ─────────────────────────────────────────────────────────
    lines += ["## Memory Usage", ""]
    if memory_report:
        lines.append("| Stage | Peak VRAM (GB) | Time (s) |")
        lines.append("|---|---|---|")
        for stage, stats in memory_report.items():
            if hasattr(stats, "peak_allocated_gb"):
                lines.append(
                    f"| {stage} | {stats.peak_allocated_gb:.2f} | "
                    f"{stats.timestamp:.1f} |"
                )
    else:
        lines.append("*Memory report not available.*")
    lines.append("")

    # ── Notes on Failed Ablations ────────────────────────────────────────────
    defined_ablations = {"A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"}
    if not comparison_table.empty:
        completed = set(comparison_table["ablation_id"].tolist())
    else:
        completed = set()
    failed = defined_ablations - completed

    if failed:
        lines += [
            "## Failed / Unavailable Ablations",
            "",
            "The following ablations were not completed (likely OOM or skipped):",
            "",
        ]
        for ab in sorted(failed):
            lines.append(
                f"- **{ab}**: Not completed. "
                f"Check logs for OOM or configuration errors."
            )
        lines.append("")

    # ── Footer ───────────────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "*Report generated by reports/report_generator.py*",
        f"*Timestamp: {timestamp}*",
    ]

    report_text = "\n".join(lines)

    # Write Markdown
    with open(str(md_path), "w") as f:
        f.write(report_text)
    logger.info(f"Markdown report saved: {md_path}")

    # Save summary JSON
    summary = {}
    if not comparison_table.empty:
        for _, row in comparison_table.iterrows():
            ab_id = row["ablation_id"]
            summary[ab_id] = {
                "dice_in": float(row.get("dice_in", float("nan"))),
                "dice_xd": float(row.get("dice_xd", float("nan"))),
                "delta_dice": float(row.get("delta_dice", float("nan"))),
            }

    summary_path = Path(output_dir) / "summary_metrics.json"
    with open(str(summary_path), "w") as f:
        json.dump({"ablations": summary, "rq_answers": rq_answers}, f, indent=2)
    logger.info(f"Summary metrics saved: {summary_path}")

    return str(md_path)


def _df_to_markdown(df: pd.DataFrame, headers: Optional[Dict] = None) -> str:
    """Convert a DataFrame to a Markdown table string."""
    if headers:
        col_names = [headers.get(c, c) for c in df.columns]
    else:
        col_names = list(df.columns)

    rows = [
        "| " + " | ".join(str(v) for v in col_names) + " |",
        "| " + " | ".join(["---"] * len(col_names)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for v in row:
            if isinstance(v, float):
                values.append(f"{v:.4f}" if not np.isnan(v) else "N/A")
            else:
                values.append(str(v))
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join(rows)
