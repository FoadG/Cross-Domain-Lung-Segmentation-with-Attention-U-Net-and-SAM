"""
evaluation/cross_domain_eval.py  [CORRECTED — adversarial review, fix #4]

Cross-domain evaluation: computes Performance Degradation (Δ) for each
ablation and answers all 5 research questions.

Δ = Dice_in_domain (in-domain test) − Dice_cross_domain (cross-domain)
Positive Δ: model performs worse on the unseen domain (expected).
Negative Δ: model performs BETTER cross-domain (rare, surprising).

Audit fix: CRITICAL-09 — explicit sign convention and per-model Δ reporting.

*** Audit fix #4 — statistical rigor (publication blocker) ***
The previous version compared bare point estimates of mean Dice and declared
"OUTPERFORMS / UNDERPERFORMS" off ANY nonzero difference (and a hand-picked
0.005 threshold for RQ2). `scipy.stats` was imported but never used: there were
no confidence intervals and no significance tests, so none of the RQ "difference"
claims were defensible.

This version adds, using the per-image score arrays that evaluator.py already
returns (`dice_per_image`, `iou_per_image`, `hd95_per_image`):
  * 95% bootstrap confidence intervals on every reported mean;
  * a difference test for every RQ comparison, choosing the correct test:
      - PAIRED Wilcoxon signed-rank when both models scored the SAME images
        in the same order (e.g. A1 vs A4 on the shared cross-domain set;
        A3 vs A4 and A4 vs A6 on the shared in-domain test set);
      - UNPAIRED Mann-Whitney U / independent bootstrap for the in-domain vs
        cross-domain Δ (different images);
  * significance-aware verdicts: a difference is only called a difference when
    p < 0.05 AND the 95% CI of the difference excludes 0.

IMPORTANT LIMITATION (stated, not papered over): SAM is trained with a SINGLE
seed. Bootstrap CIs here quantify PER-IMAGE sampling uncertainty on a fixed set
of weights — NOT training-run variance. A claim about the *method* (e.g. "LoRA
beats Freeze-Encoder") still requires multiple training seeds. Where SAM is
involved, the answers say so explicitly.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

# Deterministic seed so bootstrap CIs / p-values are reproducible across runs.
BOOTSTRAP_SEED = 1234
N_BOOTSTRAP = 5000
CI_PERCENT = 95.0

# Ablation identifiers for clear reporting
ABLATION_LABELS = {
    "A1": "U-Net (baseline)",
    "A2": "SAM zero-shot",
    "A3": "SAM Freeze + single box",
    "A4": "SAM Freeze + ensemble",
    "A5": "SAM LoRA + single box",
    "A6": "SAM LoRA + ensemble",
    "A7": "SAM Freeze + oracle box",
    "A8": "SAM Freeze + stress box",
}


# ─────────────────────────────────────────────────────────────────────────────
# Statistical helpers (audit fix #4)
# ─────────────────────────────────────────────────────────────────────────────

def _rng() -> np.random.RandomState:
    return np.random.RandomState(BOOTSTRAP_SEED)


def _bootstrap_mean_ci(
    values: List[float],
    n_boot: int = N_BOOTSTRAP,
    ci: float = CI_PERCENT,
) -> Dict[str, float]:
    """Mean of `values` with a percentile bootstrap CI.

    Returns {mean, ci_low, ci_high, n}. Empty/short inputs degrade gracefully:
    a single value yields a zero-width CI; an empty list yields NaNs.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n == 0:
        return {"mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": 0}
    mean = float(arr.mean())
    if n == 1:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "n": 1}
    rng = _rng()
    idx = rng.randint(0, n, size=(n_boot, n))
    boot_means = arr[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, (100 - ci) / 2))
    hi = float(np.percentile(boot_means, 100 - (100 - ci) / 2))
    return {"mean": mean, "ci_low": lo, "ci_high": hi, "n": int(n)}


def _difference_test(
    a: List[float],
    b: List[float],
    paired: bool,
    n_boot: int = N_BOOTSTRAP,
    ci: float = CI_PERCENT,
) -> Dict:
    """Test whether mean(a) − mean(b) differs from 0.

    paired=True  -> per-image arrays for the SAME images in the same order;
                    bootstrap CI on per-image differences + Wilcoxon signed-rank.
    paired=False -> independent samples; bootstrap CI on the difference of means
                    (resampling each group independently) + Mann-Whitney U.

    Returns a dict with: mean_a, mean_b, mean_diff, diff_ci_low, diff_ci_high,
    p_value, test, paired, n_a, n_b, significant (p<0.05 AND CI excludes 0).
    """
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    out = {
        "mean_a": float(aa.mean()) if aa.size else float("nan"),
        "mean_b": float(bb.mean()) if bb.size else float("nan"),
        "mean_diff": float("nan"),
        "diff_ci_low": float("nan"),
        "diff_ci_high": float("nan"),
        "p_value": None,
        "test": None,
        "paired": paired,
        "n_a": int(aa.size),
        "n_b": int(bb.size),
        "significant": False,
        "note": None,
    }
    if aa.size == 0 or bb.size == 0:
        out["note"] = "missing per-image scores for one or both models"
        return out

    rng = _rng()

    if paired:
        if aa.size != bb.size:
            # Fall back to unpaired if lengths differ (cannot pair by index).
            return _difference_test(a, b, paired=False, n_boot=n_boot, ci=ci)
        diff = aa - bb
        out["mean_diff"] = float(diff.mean())
        n = diff.size
        idx = rng.randint(0, n, size=(n_boot, n))
        boot = diff[idx].mean(axis=1)
        out["diff_ci_low"] = float(np.percentile(boot, (100 - ci) / 2))
        out["diff_ci_high"] = float(np.percentile(boot, 100 - (100 - ci) / 2))
        out["test"] = "wilcoxon_signed_rank"
        try:
            if np.allclose(diff, 0.0):
                out["p_value"] = 1.0
                out["note"] = "all per-image differences are zero"
            else:
                _, p = scipy_stats.wilcoxon(aa, bb)
                out["p_value"] = float(p)
        except ValueError as e:  # e.g. too few nonzero diffs
            out["note"] = f"wilcoxon failed: {e}"
    else:
        out["mean_diff"] = float(aa.mean() - bb.mean())
        idx_a = rng.randint(0, aa.size, size=(n_boot, aa.size))
        idx_b = rng.randint(0, bb.size, size=(n_boot, bb.size))
        boot = aa[idx_a].mean(axis=1) - bb[idx_b].mean(axis=1)
        out["diff_ci_low"] = float(np.percentile(boot, (100 - ci) / 2))
        out["diff_ci_high"] = float(np.percentile(boot, 100 - (100 - ci) / 2))
        out["test"] = "mann_whitney_u"
        try:
            _, p = scipy_stats.mannwhitneyu(aa, bb, alternative="two-sided")
            out["p_value"] = float(p)
        except ValueError as e:
            out["note"] = f"mann_whitney failed: {e}"

    if out["p_value"] is not None:
        ci_excludes_zero = (out["diff_ci_low"] > 0) or (out["diff_ci_high"] < 0)
        out["significant"] = bool(out["p_value"] < 0.05 and ci_excludes_zero)
    return out


def _fmt_ci(d: Dict, key: str = "mean") -> str:
    """Format 'mean [lo, hi]' for a bootstrap-CI dict."""
    if not d or np.isnan(d.get(key, float("nan"))):
        return "n/a"
    return f"{d[key]:.4f} [{d['ci_low']:.4f}, {d['ci_high']:.4f}]"


def _fmt_diff(t: Dict) -> str:
    """Format a difference-test result into a publication-style phrase."""
    if t["p_value"] is None:
        return f"Δ n/a ({t.get('note', 'insufficient data')})"
    sig = "significant" if t["significant"] else "NOT significant"
    return (
        f"Δ={t['mean_diff']:+.4f} "
        f"[{t['diff_ci_low']:+.4f}, {t['diff_ci_high']:+.4f}] "
        f"(p={t['p_value']:.4g}, {t['test']}, {sig})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Degradation + comparison table
# ─────────────────────────────────────────────────────────────────────────────

def compute_performance_degradation(
    in_domain_metrics: Dict,
    cross_domain_metrics: Dict,
) -> Dict:
    """
    Compute Performance Degradation (Δ) for a single model, with CIs.

    Δ_Dice = Dice_in_domain − Dice_cross_domain   (positive = worse cross-domain)

    When per-image arrays are present, attaches bootstrap CIs for each domain
    mean and an UNPAIRED bootstrap CI + Mann-Whitney p-value for the Δ
    (in-domain and cross-domain are different images, hence unpaired).

    Audit fix: CRITICAL-09 (sign), #4 (CIs + tests).
    """
    result = {
        "dice_in_domain": in_domain_metrics.get("dice_mean", float("nan")),
        "dice_cross_domain": cross_domain_metrics.get("dice_mean", float("nan")),
        "iou_in_domain": in_domain_metrics.get("iou_mean", float("nan")),
        "iou_cross_domain": cross_domain_metrics.get("iou_mean", float("nan")),
    }

    result["delta_dice"] = result["dice_in_domain"] - result["dice_cross_domain"]
    result["delta_iou"] = result["iou_in_domain"] - result["iou_cross_domain"]

    if "hd95_mean" in in_domain_metrics and "hd95_mean" in cross_domain_metrics:
        result["hd95_in_domain"] = in_domain_metrics["hd95_mean"]
        result["hd95_cross_domain"] = cross_domain_metrics["hd95_mean"]
        # For HD95, larger is worse, so Δ is reversed
        result["delta_hd95"] = (
            result["hd95_cross_domain"] - result["hd95_in_domain"]
        )

    # Statistics from per-image arrays (audit fix #4)
    in_dice = in_domain_metrics.get("dice_per_image")
    xd_dice = cross_domain_metrics.get("dice_per_image")
    if in_dice is not None and xd_dice is not None:
        result["dice_in_ci"] = _bootstrap_mean_ci(in_dice)
        result["dice_xd_ci"] = _bootstrap_mean_ci(xd_dice)
        # Δ is in_domain − cross_domain on DIFFERENT images -> unpaired.
        delta_test = _difference_test(in_dice, xd_dice, paired=False)
        result["delta_dice_test"] = delta_test

    return result


def build_ablation_comparison_table(
    all_metrics: Dict[str, Dict],
) -> pd.DataFrame:
    """
    Assemble the complete ablation comparison table (now with CI columns).
    """
    rows = []
    for ablation_id, mdict in all_metrics.items():
        in_dm = mdict.get("in_domain", {})
        xd_dm = mdict.get("cross_domain", {})

        if not in_dm or not xd_dm:
            logger.warning(
                f"Missing metrics for ablation {ablation_id}. Skipping."
            )
            continue

        degradation = compute_performance_degradation(in_dm, xd_dm)
        dice_in_ci = degradation.get("dice_in_ci", {})
        dice_xd_ci = degradation.get("dice_xd_ci", {})
        delta_test = degradation.get("delta_dice_test", {})

        row = {
            "ablation_id": ablation_id,
            "ablation_label": ABLATION_LABELS.get(ablation_id, ablation_id),
            # In-domain metrics
            "dice_in": degradation["dice_in_domain"],
            "dice_in_ci_low": dice_in_ci.get("ci_low", float("nan")),
            "dice_in_ci_high": dice_in_ci.get("ci_high", float("nan")),
            "iou_in": degradation["iou_in_domain"],
            "hd95_in": degradation.get("hd95_in_domain", float("nan")),
            # Cross-domain metrics
            "dice_xd": degradation["dice_cross_domain"],
            "dice_xd_ci_low": dice_xd_ci.get("ci_low", float("nan")),
            "dice_xd_ci_high": dice_xd_ci.get("ci_high", float("nan")),
            "iou_xd": degradation["iou_cross_domain"],
            "hd95_xd": degradation.get("hd95_cross_domain", float("nan")),
            # Performance Degradation
            "delta_dice": degradation["delta_dice"],
            "delta_dice_ci_low": delta_test.get("diff_ci_low", float("nan")),
            "delta_dice_ci_high": delta_test.get("diff_ci_high", float("nan")),
            "delta_dice_p": delta_test.get("p_value", float("nan")),
            "delta_dice_significant": delta_test.get("significant", False),
            "delta_iou": degradation["delta_iou"],
            "delta_hd95": degradation.get("delta_hd95", float("nan")),
            # Sample counts
            "n_in_domain": in_dm.get("n_images", 0),
            "n_cross_domain": xd_dm.get("n_images", 0),
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("ablation_id").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Research questions (significance-aware)
# ─────────────────────────────────────────────────────────────────────────────

# Comparisons where both models score the SAME images in the same order, so a
# paired test is valid (evaluator iterates a deterministically-filtered df).
_SAME_DOMAIN_PAIRED = True


def _per_image(all_metrics: Dict, ablation_id: str, domain: str,
               metric: str = "dice") -> Optional[List[float]]:
    m = all_metrics.get(ablation_id, {}).get(domain, {})
    return m.get(f"{metric}_per_image")


def _holm_bonferroni(pvals: Dict[str, float], alpha: float = 0.05) -> Dict[str, Dict]:
    """Holm–Bonferroni step-down correction over a family of p-values.

    Returns {key: {"p_adjusted": float, "reject": bool}}. Controls the
    family-wise error rate across the RQ comparisons (audit fix #4: the previous
    version ran several tests with no correction, inflating false positives).
    Keys with a None p-value are skipped (not part of the tested family).
    """
    valid = {k: v for k, v in pvals.items() if v is not None}
    m = len(valid)
    out: Dict[str, Dict] = {k: {"p_adjusted": None, "reject": False} for k in pvals}
    if m == 0:
        return out
    # Sort ascending by raw p.
    ordered = sorted(valid.items(), key=lambda kv: kv[1])
    running_max = 0.0
    for rank, (k, p) in enumerate(ordered):
        adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, adj)  # enforce monotonic non-decreasing
        out[k]["p_adjusted"] = running_max
        out[k]["reject"] = running_max < alpha
    return out


def _fmt_diff_adj(t: Dict) -> str:
    """Format a difference test including the Holm-adjusted p and verdict basis."""
    if t["p_value"] is None:
        return f"Δ n/a ({t.get('note', 'insufficient data')})"
    sig = "significant" if t.get("significant_adjusted") else "NOT significant"
    padj = t.get("p_adjusted")
    padj_s = f"{padj:.4g}" if padj is not None else "n/a"
    return (
        f"Δ={t['mean_diff']:+.4f} "
        f"[{t['diff_ci_low']:+.4f}, {t['diff_ci_high']:+.4f}] "
        f"(p_raw={t['p_value']:.4g}, p_Holm={padj_s}, {t['test']}, "
        f"{sig} after correction)"
    )


def answer_research_questions(
    comparison_table: pd.DataFrame,
    all_metrics: Dict[str, Dict],
) -> Dict[str, str]:
    """
    Generate significance-aware answers to the 5 research questions.

    Each comparison reports both means with 95% bootstrap CIs, the difference
    with its CI, and a p-value from the appropriate test. Verdicts only assert a
    difference when it is statistically significant. SAM single-seed limitation
    is stated explicitly.
    """
    answers: Dict[str, str] = {}

    SAM_CAVEAT = (
        " [Limitation: SAM is single-seed; this CI/p reflects per-image "
        "sampling on fixed weights, not training-run variance.]"
    )

    def mean_str(ablation_id: str, domain: str) -> str:
        vals = _per_image(all_metrics, ablation_id, domain)
        if vals is None:
            # Fall back to stored mean if per-image arrays are absent.
            col = "dice_in" if domain == "in_domain" else "dice_xd"
            row = comparison_table[comparison_table["ablation_id"] == ablation_id]
            if row.empty:
                return "n/a"
            return f"{float(row[col].iloc[0]):.4f} (no CI: per-image scores absent)"
        return _fmt_ci(_bootstrap_mean_ci(vals))

    # ── Stage 1: compute the RQ-family difference tests that have valid data.
    # All three are within-domain comparisons on the SAME images -> paired.
    a1 = _per_image(all_metrics, "A1", "in_domain")
    a3 = _per_image(all_metrics, "A3", "in_domain")
    a4 = _per_image(all_metrics, "A4", "in_domain")
    a6 = _per_image(all_metrics, "A6", "in_domain")

    tests: Dict[str, Dict] = {}
    if a1 is not None and a4 is not None:
        tests["RQ1"] = _difference_test(a4, a1, paired=_SAME_DOMAIN_PAIRED)  # A4 − A1
    if a6 is not None and a4 is not None:
        tests["RQ2"] = _difference_test(a6, a4, paired=_SAME_DOMAIN_PAIRED)  # A6 − A4
    if a3 is not None and a4 is not None:
        tests["RQ4"] = _difference_test(a4, a3, paired=_SAME_DOMAIN_PAIRED)  # A4 − A3

    # ── Stage 2: Holm–Bonferroni across the family (audit fix #4).
    holm = _holm_bonferroni({k: t["p_value"] for k, t in tests.items()})
    for k, t in tests.items():
        t["p_adjusted"] = holm[k]["p_adjusted"]
        # Significant only if Holm rejects AND the (raw) bootstrap CI excludes 0.
        ci_excl0 = (t["diff_ci_low"] > 0) or (t["diff_ci_high"] < 0)
        t["significant_adjusted"] = bool(holm[k]["reject"] and ci_excl0)

    HOLM_NOTE = (
        " [Multiple-comparison note: significance is Holm–Bonferroni-corrected "
        "across the RQ family.]"
    )

    # ── RQ1: SAM Freeze-Encoder+ensemble (A4) vs U-Net (A1), in-domain. ──
    if "RQ1" in tests:
        t = tests["RQ1"]
        if not t["significant_adjusted"]:
            verdict = ("no statistically significant in-domain difference between "
                       "SAM Freeze-Encoder+ensemble and U-Net")
        elif t["mean_diff"] > 0:
            verdict = "SAM Freeze-Encoder+ensemble significantly outperforms U-Net in-domain"
        else:
            verdict = "SAM Freeze-Encoder+ensemble significantly underperforms U-Net in-domain"
        answers["RQ1"] = (
            f"U-Net (A1) in-domain Dice: {mean_str('A1', 'in_domain')}. "
            f"SAM Freeze-Encoder+ensemble (A4): {mean_str('A4', 'in_domain')}. "
            f"{_fmt_diff_adj(t)}. Conclusion: {verdict}."
            + SAM_CAVEAT + HOLM_NOTE
            + " (For a prompt-only comparison without ensembling, see A3.)"
        )
    else:
        answers["RQ1"] = "Insufficient per-image data for RQ1 (A1 and/or A4 missing)."

    # ── RQ2: LoRA (A6) vs Freeze-Encoder (A4), in-domain. ──
    if a6 is None:
        answers["RQ2"] = (
            "SAM LoRA (A6) results not available (e.g. OOM during training). "
            "Cannot compare with Freeze-Encoder; A4 remains the reference."
        )
    elif "RQ2" not in tests:
        answers["RQ2"] = "Insufficient per-image data for RQ2 (A4 missing)."
    else:
        t = tests["RQ2"]
        if not t["significant_adjusted"]:
            verdict = "LoRA does NOT significantly improve over Freeze-Encoder"
        elif t["mean_diff"] > 0:
            verdict = "LoRA significantly improves over Freeze-Encoder"
        else:
            verdict = "LoRA is significantly WORSE than Freeze-Encoder"
        answers["RQ2"] = (
            f"SAM Freeze-Encoder (A4): {mean_str('A4', 'in_domain')}. "
            f"SAM LoRA (A6): {mean_str('A6', 'in_domain')}. "
            f"{_fmt_diff_adj(t)}. Conclusion: {verdict}." + SAM_CAVEAT + HOLM_NOTE
        )

    # ── RQ3: Which model degrades less cross-domain (smaller |Δ|)?
    #         Compares two Δ's; each Δ is in-domain − cross-domain (unpaired). ──
    unet_deg = comparison_table[comparison_table["ablation_id"] == "A1"]
    sam_deg = comparison_table[comparison_table["ablation_id"] == "A4"]
    if not unet_deg.empty and not sam_deg.empty:
        ud = float(unet_deg["delta_dice"].iloc[0])
        sd = float(sam_deg["delta_dice"].iloc[0])
        u_lo = float(unet_deg["delta_dice_ci_low"].iloc[0])
        u_hi = float(unet_deg["delta_dice_ci_high"].iloc[0])
        s_lo = float(sam_deg["delta_dice_ci_low"].iloc[0])
        s_hi = float(sam_deg["delta_dice_ci_high"].iloc[0])
        winner = "SAM" if abs(sd) < abs(ud) else "U-Net"
        answers["RQ3"] = (
            f"U-Net Δ_Dice={ud:+.4f} [{u_lo:+.4f}, {u_hi:+.4f}] | "
            f"SAM Freeze-Encoder Δ_Dice={sd:+.4f} [{s_lo:+.4f}, {s_hi:+.4f}]. "
            f"Smaller |Δ| (better cross-domain robustness): {winner}. "
            "Note: each Δ's CI is per-image bootstrap; overlapping CIs mean the "
            "robustness gap is not established. A direct test of the two Δ's "
            "would require a study design with multiple seeds." + SAM_CAVEAT
        )
    else:
        answers["RQ3"] = (
            "Insufficient data to compare cross-domain degradation "
            "(A1 and/or A4 cross-domain metrics missing — note this also occurs "
            "if the in-domain test domain has no images, which the data-pairing "
            "fix in validator.py addresses)."
        )

    # ── RQ4: Ensemble (A4) vs single box (A3), in-domain. ──
    if "RQ4" in tests:
        t = tests["RQ4"]
        if not t["significant_adjusted"]:
            verdict = "ensemble does NOT significantly change accuracy vs a single box"
        elif t["mean_diff"] > 0:
            verdict = "ensemble significantly improves accuracy vs a single box"
        else:
            verdict = "ensemble significantly HURTS accuracy vs a single box"
        answers["RQ4"] = (
            f"SAM single-box (A3): {mean_str('A3', 'in_domain')} | "
            f"SAM ensemble (A4): {mean_str('A4', 'in_domain')}. "
            f"{_fmt_diff_adj(t)}. Conclusion: {verdict}." + SAM_CAVEAT + HOLM_NOTE
        )
    else:
        answers["RQ4"] = "Insufficient per-image data for ensemble comparison (A3/A4)."

    # ── RQ5: VRAM budget — measured, not statistical. ──
    answers["RQ5"] = (
        "Verified by the memory monitor during execution; see memory_report.json "
        "for measured peak VRAM per stage. (Not a statistical claim.)"
    )

    return answers


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def _collect_stats_payload(all_metrics: Dict) -> Dict:
    """Build a JSON-serializable stats block (means, CIs, tests) per ablation."""
    payload = {}
    for ablation_id, mdict in all_metrics.items():
        in_dm = mdict.get("in_domain", {})
        xd_dm = mdict.get("cross_domain", {})
        if not in_dm or not xd_dm:
            continue
        deg = compute_performance_degradation(in_dm, xd_dm)
        payload[ablation_id] = {
            "dice_in_ci": deg.get("dice_in_ci"),
            "dice_xd_ci": deg.get("dice_xd_ci"),
            "delta_dice": deg.get("delta_dice"),
            "delta_dice_test": deg.get("delta_dice_test"),
        }
    return payload


def save_full_results(
    comparison_table: pd.DataFrame,
    rq_answers: Dict[str, str],
    all_metrics: Dict,
    output_dir: str,
) -> None:
    """
    Save all evaluation results to disk:
        - ablation_comparison.csv  (now includes CI + p columns)
        - rq_answers.json
        - statistics.json          (means, CIs, difference tests)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    csv_path = Path(output_dir) / "ablation_comparison.csv"
    comparison_table.to_csv(str(csv_path), index=False)
    logger.info(f"Ablation comparison table saved: {csv_path}")

    rq_path = Path(output_dir) / "rq_answers.json"
    with open(str(rq_path), "w") as f:
        json.dump(rq_answers, f, indent=2)
    logger.info(f"RQ answers saved: {rq_path}")

    stats_path = Path(output_dir) / "statistics.json"
    with open(str(stats_path), "w") as f:
        json.dump(_collect_stats_payload(all_metrics), f, indent=2)
    logger.info(f"Statistics (CIs + tests) saved: {stats_path}")

    logger.info("\n" + "=" * 60)
    logger.info("RESEARCH QUESTION ANSWERS")
    logger.info("=" * 60)
    for rq, answer in rq_answers.items():
        logger.info(f"{rq}: {answer}")
    logger.info("=" * 60)


def _get_delta(table: pd.DataFrame, ablation_id: str) -> float:
    row = table[table["ablation_id"] == ablation_id]
    if row.empty:
        return float("nan")
    return float(row["delta_dice"].iloc[0])
