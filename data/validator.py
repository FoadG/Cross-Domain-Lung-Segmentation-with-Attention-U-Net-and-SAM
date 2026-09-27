"""
data/validator.py  [ADVERSARIAL-REVIEW CORRECTED — v2]

Root-cause fix for: "ValueError: No Shenzhen images found in DataFrame."

The original validator paired images to masks by EXACT filename stem and
classified a file as a mask only if the substring "mask" appeared anywhere
in its path. In the `nikhilpandey360/chest-xray-masks-and-labels` dataset:

  * Montgomery masks are named   MCUCXR_xxxx_0.png        (NO `_mask` suffix)
  * Shenzhen   masks are named   CHNCXR_xxxx_0_mask.png   (WITH `_mask` suffix)

Exact-stem matching therefore paired every Montgomery image but ZERO Shenzhen
images (their mask stem `..._mask` never equals the image stem). Result:
Shenzhen_count == 0, and split.py crashed.

This version:
  1. Detects masks by directory name OR filename suffix (robust to both layouts).
  2. Pairs by a NORMALIZED stem (strips `_mask` / `_combined` / `_segmentation`),
     so `CHNCXR_0001_0_mask` pairs with image `CHNCXR_0001_0`.
  3. Explicitly EXCLUDES the combined-mask output directory from scanning, so
     re-runs do not pick up generated masks as new "images".
  4. Emits a per-directory + per-domain diagnostic so a missing domain is
     immediately visible in the logs instead of failing silently downstream.
  5. Keeps the original public API (validate_dataset -> ValidationReport) and
     all existing fields, so cleaner.py / run_experiment.py are unaffected.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Suffixes that indicate a file is a mask rather than an image. Order matters:
# longer / more specific suffixes are tried first when normalizing stems.
MASK_STEM_SUFFIXES: Tuple[str, ...] = ("_combined", "_segmentation", "_mask")


@dataclass
class FileRecord:
    image_path: str
    mask_path: Optional[str]            # path to (combined) mask ready for use
    all_mask_paths: List[str]           # all raw mask files (1 for Shenzhen, >=1 for Montgomery)
    source: str
    mask_exists: bool
    image_readable: bool
    mask_readable: bool
    image_shape: Optional[Tuple[int, int]]
    mask_shape: Optional[Tuple[int, int]]
    mask_foreground_fraction: Optional[float]
    issues: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return (self.mask_exists and self.image_readable
                and self.mask_readable and len(self.issues) == 0)


@dataclass
class ValidationReport:
    total_images: int
    valid_images: int
    excluded_no_mask: int
    excluded_unreadable: int
    montgomery_count: int
    shenzhen_count: int
    records: List[FileRecord]
    unknown_count: int = 0              # images matching neither prefix (new, defaulted)

    def summary(self) -> str:
        lines = [
            "=" * 60, "DATASET VALIDATION REPORT", "=" * 60,
            f"  Total image files found:  {self.total_images}",
            f"  Valid pairs (usable):     {self.valid_images}",
            f"  Excluded (no mask):       {self.excluded_no_mask}",
            f"  Excluded (unreadable):    {self.excluded_unreadable}",
            f"  Montgomery (valid):       {self.montgomery_count}",
            f"  Shenzhen   (valid):       {self.shenzhen_count}",
            f"  Unknown-prefix (valid):   {self.unknown_count}",
            "=" * 60,
        ]
        return "\n".join(lines)


def _is_mask_path(path: Path, combined_mask_dir: Path) -> bool:
    """A file is a mask if it lives under a *directory* whose name contains
    'mask', OR if its filename stem ends with a known mask suffix. Files inside
    the generated combined-mask directory are handled separately (excluded)."""
    # any parent directory name mentions mask (e.g. masks/, ManualMask/leftMask/)
    for part in path.parts[:-1]:
        if "mask" in part.lower():
            return True
    return path.stem.lower().endswith(MASK_STEM_SUFFIXES)


def _normalize_stem(stem: str) -> str:
    """Strip a trailing mask suffix so a mask filename maps to its image stem."""
    low = stem.lower()
    for suf in MASK_STEM_SUFFIXES:
        if low.endswith(suf):
            return stem[: -len(suf)]
    return stem


def validate_dataset(
    data_root: str,
    montgomery_prefix: str = "MCUCXR_",
    shenzhen_prefix: str = "CHNCXR_",
    min_foreground_fraction: float = 0.005,
    combined_mask_dir: str = "./data/processed/combined_masks",
    max_workers: int = 4,
) -> ValidationReport:
    """
    Validate all image/mask pairs. Montgomery masks (left+right) are unioned
    and written to `combined_mask_dir`; Shenzhen masks are used directly.

    Robust to the two real-world mask-naming conventions in this dataset.
    """
    data_path = Path(data_root)
    if not data_path.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    combined_dir = Path(combined_mask_dir).resolve()
    combined_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Scanning dataset at: {data_root}")

    # Scan everything, but NEVER treat the generated combined-mask dir as input.
    all_png = [
        p for p in sorted(data_path.rglob("*.png"))
        if combined_dir not in p.resolve().parents and p.resolve().parent != combined_dir
    ]

    image_files = [p for p in all_png if not _is_mask_path(p, combined_dir)]
    mask_files = [p for p in all_png if _is_mask_path(p, combined_dir)]

    # Diagnostic: where did files come from? Surfaces duplicate dirs / odd counts.
    _log_directory_breakdown(image_files, mask_files, data_path)

    mask_by_norm: Dict[str, List[Path]] = {}
    for mf in mask_files:
        mask_by_norm.setdefault(_normalize_stem(mf.stem), []).append(mf)

    logger.info(f"Found {len(image_files)} images, {len(mask_files)} masks.")

    records: List[FileRecord] = []
    excluded_no_mask = 0
    excluded_unreadable = 0
    # Per-domain "image present" tally (independent of pairing success) for diagnosis
    prefix_seen = {"Montgomery": 0, "Shenzhen": 0, "Unknown": 0}

    for image_path in image_files:
        stem = image_path.stem
        filename = image_path.name

        if filename.startswith(montgomery_prefix):
            source = "Montgomery"
        elif filename.startswith(shenzhen_prefix):
            source = "Shenzhen"
        else:
            source = "Unknown"
        prefix_seen[source] += 1

        candidate_masks = mask_by_norm.get(stem, [])
        mask_exists = len(candidate_masks) > 0
        if not mask_exists:
            excluded_no_mask += 1

        record = FileRecord(
            image_path=str(image_path),
            mask_path=None,
            all_mask_paths=[str(p) for p in candidate_masks],
            source=source,
            mask_exists=mask_exists,
            image_readable=False,
            mask_readable=False,
            image_shape=None,
            mask_shape=None,
            mask_foreground_fraction=None,
        )

        try:
            with Image.open(str(image_path)) as img:
                record.image_shape = (img.size[1], img.size[0])
                record.image_readable = True
        except Exception as e:
            record.issues.append(f"Image unreadable: {e}")
            excluded_unreadable += 1

        if mask_exists and record.image_readable:
            # Multiple mask files (e.g. Montgomery left+right) -> union them.
            # A single mask file is just used directly. This is decided by the
            # number of paired masks, NOT by the (often unreliable) source label.
            try:
                if len(candidate_masks) > 1:
                    combined = _combine_masks(candidate_masks)
                    combined_path = combined_dir / f"{stem}_combined.png"
                    if not combined_path.exists():
                        Image.fromarray((combined * 255).astype(np.uint8)).save(str(combined_path))
                    record.mask_path = str(combined_path)
                    record.mask_shape = combined.shape
                    record.mask_foreground_fraction = float(combined.mean())
                    record.mask_readable = True
                else:
                    mask_path = candidate_masks[0]
                    with Image.open(str(mask_path)) as mimg:
                        mask_arr = np.array(mimg.convert("L"), dtype=np.float32)
                    if mask_arr.max() > 1:
                        mask_arr = (mask_arr > 127).astype(np.float32)
                    record.mask_path = str(mask_path)
                    record.mask_shape = mask_arr.shape
                    record.mask_foreground_fraction = float(mask_arr.mean())
                    record.mask_readable = True

                if (record.mask_foreground_fraction is not None
                        and record.mask_foreground_fraction < min_foreground_fraction):
                    record.issues.append(
                        f"Mask foreground too low: {record.mask_foreground_fraction:.4f}"
                    )
            except Exception as e:
                record.issues.append(f"Mask processing failed: {e}")

        records.append(record)

    valid_records = [r for r in records if r.is_valid]
    montgomery_count = sum(1 for r in valid_records if r.source == "Montgomery")
    shenzhen_count = sum(1 for r in valid_records if r.source == "Shenzhen")
    unknown_count = sum(1 for r in valid_records if r.source == "Unknown")

    report = ValidationReport(
        total_images=len(image_files),
        valid_images=len(valid_records),
        excluded_no_mask=excluded_no_mask,
        excluded_unreadable=excluded_unreadable,
        montgomery_count=montgomery_count,
        shenzhen_count=shenzhen_count,
        unknown_count=unknown_count,
        records=records,
    )
    logger.info(report.summary())

    # Loud, actionable diagnostics for the exact failure that was reported.
    _diagnose_missing_domains(prefix_seen, montgomery_count, shenzhen_count,
                              mask_files, montgomery_prefix, shenzhen_prefix)
    return report


def _combine_masks(mask_paths: List[Path]) -> np.ndarray:
    """Union of multiple binary masks (e.g. left + right lung for Montgomery)."""
    combined = None
    for mp in mask_paths:
        with Image.open(str(mp)) as mimg:
            arr = np.array(mimg.convert("L"), dtype=np.float32)
        if arr.max() > 1:
            arr = (arr > 127).astype(np.float32)
        combined = arr if combined is None else np.maximum(combined, arr)
    return combined.astype(np.float32)


def _log_directory_breakdown(image_files, mask_files, data_path) -> None:
    """Surface per-directory counts so duplicate folders (a common cause of
    inflated counts like '276 Montgomery') are immediately visible."""
    def by_dir(files):
        counts: Dict[str, int] = {}
        for f in files:
            rel = str(Path(f).parent.relative_to(data_path)) if data_path in Path(f).parents else str(Path(f).parent)
            counts[rel] = counts.get(rel, 0) + 1
        return counts
    logger.info(f"Image files by directory: {by_dir(image_files)}")
    logger.info(f"Mask  files by directory: {by_dir(mask_files)}")


def _diagnose_missing_domains(prefix_seen, montgomery_count, shenzhen_count,
                              mask_files, montgomery_prefix, shenzhen_prefix) -> None:
    if shenzhen_count == 0 and prefix_seen.get("Shenzhen", 0) > 0:
        sample = [m.name for m in mask_files if m.name.startswith(shenzhen_prefix)][:5]
        logger.error(
            "Shenzhen images were found (%d with prefix '%s') but NONE paired to a "
            "mask. Sample Shenzhen mask filenames present: %s. Check that mask "
            "naming is handled by MASK_STEM_SUFFIXES.",
            prefix_seen["Shenzhen"], shenzhen_prefix, sample,
        )
    if montgomery_count == 0 and prefix_seen.get("Montgomery", 0) > 0:
        logger.error(
            "Montgomery images found (%d) but none paired. Check mask layout.",
            prefix_seen["Montgomery"],
        )
    if prefix_seen.get("Unknown", 0) > 0:
        logger.warning(
            "%d image(s) matched neither prefix '%s' nor '%s' and are labelled "
            "Unknown. They will be excluded from cross-domain logic.",
            prefix_seen["Unknown"], montgomery_prefix, shenzhen_prefix,
        )
