"""
data/split.py  [ADVERSARIAL-REVIEW CORRECTED — v2]

Assigns train/val/test/cross_domain_test splits.

Design (cross-domain protocol):
  - The IN-DOMAIN source (default 'Shenzhen') is split into train/val/test.
  - The CROSS-DOMAIN source (default 'Montgomery') is held out entirely as
    'cross_domain_test' and never seen during training.

Robustness changes vs the original:
  * Source-agnostic: in/cross domains are parameters, not hard-coded literals.
  * Graceful degradation: if the in-domain source is absent (the exact cause
    of the original `No Shenzhen images found` crash), instead of dying we
    either (a) raise an actionable error explaining the likely mask-pairing
    cause, or (b) if `degrade_if_missing=True`, fall back to single-domain
    in-domain-only evaluation using whichever domain IS present, with a loud
    warning that cross-domain RQs (e.g. degradation Δ) become unanswerable.
  * The leakage guard is generalized: NO cross-domain (or any non-in-domain)
    row may appear in train/val, regardless of which source is in-domain.
  * A `split_meta.json` sidecar records exactly which source was in-domain,
    which was cross-domain, and whether the run degraded — so downstream
    code and the final report can state the protocol honestly.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

logger = logging.getLogger(__name__)


def assign_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    split_seed: int = 42,
    splits_dir: str = "./data/splits",
    in_domain_source: str = "Shenzhen",
    cross_domain_source: str = "Montgomery",
    degrade_if_missing: bool = True,
) -> pd.DataFrame:
    """Assign split labels. Returns df with a populated 'split' column."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got "
            f"{train_ratio + val_ratio + test_ratio:.4f}"
        )

    df = df.copy()
    df["split"] = ""

    counts = df["source"].value_counts().to_dict()
    present = {s for s, n in counts.items() if n > 0}
    logger.info(f"Source counts available for splitting: {counts}")

    effective_in, effective_cross, degraded = _resolve_domains(
        present, in_domain_source, cross_domain_source, degrade_if_missing, counts
    )

    # Cross-domain source (if any) -> held-out cross_domain_test.
    if effective_cross is not None:
        cross_mask = df["source"] == effective_cross
        df.loc[cross_mask, "split"] = "cross_domain_test"
        logger.info(
            f"Assigned {int(cross_mask.sum())} '{effective_cross}' images "
            f"-> cross_domain_test"
        )

    # In-domain source -> stratified train/val/test.
    in_df = df[df["source"] == effective_in].copy()
    if len(in_df) == 0:
        raise ValueError(
            f"In-domain source '{effective_in}' has zero rows even after "
            f"resolution. This should not happen; check the DataFrame."
        )

    in_indices = in_df.index.tolist()
    fg = in_df["mask_foreground_fraction"].fillna(0.0).values
    strat = _make_strat_bins(fg)

    train_val_idx, test_idx = train_test_split(
        np.arange(len(in_indices)),
        test_size=test_ratio,
        random_state=split_seed,
        stratify=_safe_strat(strat),
    )
    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val_ratio,
        random_state=split_seed,
        stratify=_safe_strat(strat[train_val_idx]),
    )

    df.loc[[in_indices[i] for i in train_idx], "split"] = "train"
    df.loc[[in_indices[i] for i in val_idx], "split"] = "val"
    df.loc[[in_indices[i] for i in test_idx], "split"] = "test"

    for split_name in ["train", "val", "test", "cross_domain_test"]:
        n = int((df["split"] == split_name).sum())
        logger.info(f"  Split '{split_name}': {n} images")
        if n == 0 and split_name in ("train", "val", "test"):
            raise ValueError(
                f"Split '{split_name}' is empty after assigning the in-domain "
                f"source '{effective_in}' ({len(in_df)} images). With these "
                f"ratios you need at least ~{_min_images_needed(test_ratio, val_ratio)} "
                f"in-domain images."
            )

    _assert_no_nonindomain_in_training(df, effective_in)

    out_dir = Path(splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out_dir / "splits.csv"), index=False)
    meta = {
        "in_domain_source": effective_in,
        "cross_domain_source": effective_cross,
        "degraded_single_domain": degraded,
        "split_seed": split_seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "counts": {k: int(v) for k, v in counts.items()},
    }
    with open(out_dir / "split_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Split CSV + meta saved to: {out_dir}")
    if degraded:
        logger.warning(
            "DEGRADED SINGLE-DOMAIN MODE: cross-domain source unavailable. "
            "Cross-domain degradation research questions CANNOT be answered "
            "from this run; report only in-domain results."
        )
    return df


def _resolve_domains(present, in_src, cross_src, degrade, counts):
    """Decide which source is in-domain vs cross-domain, degrading if needed.

    Returns (effective_in, effective_cross_or_None, degraded_bool).
    """
    if in_src in present:
        effective_cross = cross_src if cross_src in present else None
        if effective_cross is None and cross_src is not None:
            logger.warning(
                "Cross-domain source '%s' not present (%s). Proceeding "
                "in-domain only; cross-domain Δ will be unavailable.",
                cross_src, sorted(present),
            )
        return in_src, effective_cross, False

    # In-domain source missing -> this is the original crash condition.
    msg = (
        f"In-domain source '{in_src}' has 0 valid images. Present sources: "
        f"{counts}. The most common cause is mask/image pairing failure "
        f"(e.g. Shenzhen masks named '<stem>_mask.png' not matching the image "
        f"stem). Verify data/validator.py pairing first."
    )
    if not degrade:
        raise ValueError(msg)

    if not present:
        raise ValueError("No images of any source were found; cannot split.")

    # Degrade: use the LARGEST present source as in-domain.
    fallback_in = max(present, key=lambda s: counts.get(s, 0))
    remaining = present - {fallback_in}
    fallback_cross = max(remaining, key=lambda s: counts.get(s, 0)) if remaining else None
    logger.warning(msg)
    logger.warning(
        "degrade_if_missing=True -> using '%s' as in-domain%s.",
        fallback_in,
        f" and '{fallback_cross}' as cross-domain" if fallback_cross else " (single domain)",
    )
    return fallback_in, fallback_cross, True


def load_splits(splits_csv_path: str, in_domain_source: Optional[str] = None) -> pd.DataFrame:
    """Load a saved split CSV and re-check the leakage guard."""
    if not Path(splits_csv_path).exists():
        raise FileNotFoundError(
            f"Split CSV not found: {splits_csv_path}. Run assign_splits() first."
        )
    df = pd.read_csv(splits_csv_path)
    if in_domain_source is None:
        meta_path = Path(splits_csv_path).parent / "split_meta.json"
        if meta_path.exists():
            in_domain_source = json.loads(meta_path.read_text()).get("in_domain_source")
    if in_domain_source:
        _assert_no_nonindomain_in_training(df, in_domain_source)
    return df


def get_fold_indices(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
    in_domain_source: str = "Shenzhen",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """K-fold CV indices over the in-domain train+val subset."""
    sub = df[
        (df["source"] == in_domain_source) & (df["split"].isin(["train", "val"]))
    ].reset_index(drop=True)
    if len(sub) == 0:
        raise ValueError(
            f"No '{in_domain_source}' train/val images for cross-validation."
        )
    fg = sub["mask_foreground_fraction"].fillna(0.0).values
    strat = _make_strat_bins(fg)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(skf.split(np.arange(len(sub)), strat))
    logger.info(f"Generated {n_folds} CV folds from {len(sub)} images.")
    return folds


def _make_strat_bins(fg_fractions: np.ndarray, n_bins: int = 4) -> np.ndarray:
    n = len(fg_fractions)
    if n == 0:
        return np.array([], dtype=int)
    max_safe_bins = max(1, int(n * 0.15))
    n_bins_actual = min(n_bins, max_safe_bins)
    percentiles = np.percentile(fg_fractions, np.linspace(0, 100, n_bins_actual + 1))
    labels = np.digitize(fg_fractions, percentiles[1:-1])
    return labels.astype(int)


def _safe_strat(strat: np.ndarray):
    """Return strat labels for stratification, or None if any bin has <2 members
    (sklearn would otherwise raise). Falls back to a random (non-stratified)
    split rather than crashing on tiny / degenerate sets."""
    if strat is None or len(strat) == 0:
        return None
    _, cnts = np.unique(strat, return_counts=True)
    return strat if cnts.min() >= 2 else None


def _min_images_needed(test_ratio: float, val_ratio: float) -> int:
    # rough lower bound so each of train/val/test gets >=1 sample
    smallest = min(test_ratio, val_ratio, 1 - test_ratio - val_ratio)
    return int(np.ceil(1.0 / max(smallest, 1e-6)))


def _assert_no_nonindomain_in_training(df: pd.DataFrame, in_domain_source: str) -> None:
    """No row whose source != in_domain_source may appear in train/val."""
    leakage = df[
        (df["source"] != in_domain_source) & (df["split"].isin(["train", "val"]))
    ]
    if len(leakage) > 0:
        cols = [c for c in ["image_path", "source", "split"] if c in leakage.columns]
        raise AssertionError(
            f"DATA LEAKAGE: {len(leakage)} non-in-domain rows in train/val "
            f"(in_domain_source='{in_domain_source}').\n{leakage[cols].to_string()}"
        )
