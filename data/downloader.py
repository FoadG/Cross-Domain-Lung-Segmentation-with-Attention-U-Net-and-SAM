"""
data/downloader.py

Downloads the Chest X-ray Masks and Labels dataset from Kaggle.
Supports both kaggle.json credentials file and environment variables.

Fix C1: Optional import moved to top of file.
"""

import logging
import os
import zipfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def setup_kaggle_credentials(kaggle_json_path: Optional[str] = None) -> None:
    import json
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        logger.info("Using Kaggle credentials from environment variables.")
        return
    if kaggle_json_path and os.path.exists(kaggle_json_path):
        with open(kaggle_json_path) as f:
            creds = json.load(f)
        os.environ["KAGGLE_USERNAME"] = creds["username"]
        os.environ["KAGGLE_KEY"] = creds["key"]
        logger.info(f"Kaggle credentials loaded from: {kaggle_json_path}")
        return
    default_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(default_path):
        with open(default_path) as f:
            creds = json.load(f)
        os.environ["KAGGLE_USERNAME"] = creds["username"]
        os.environ["KAGGLE_KEY"] = creds["key"]
        logger.info(f"Kaggle credentials loaded from: {default_path}")
        return
    raise EnvironmentError(
        "No Kaggle credentials found. Provide one of:\n"
        "  1. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables\n"
        "  2. Place kaggle.json at ~/.kaggle/kaggle.json\n"
        "  3. Pass kaggle_json_path argument to download_dataset()"
    )


def download_dataset(
    output_dir: str,
    kaggle_dataset: str = "nikhilpandey360/chest-xray-masks-and-labels",
    kaggle_json_path: Optional[str] = None,
    force_download: bool = False,
) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    extracted_marker = output_path / ".extracted"

    if extracted_marker.exists() and not force_download:
        logger.info(f"Dataset already extracted at {output_dir}.")
        return str(output_path)

    setup_kaggle_credentials(kaggle_json_path)

    try:
        import kaggle
    except ImportError:
        raise RuntimeError("kaggle package not installed. Run: pip install kaggle")

    logger.info(f"Downloading dataset: {kaggle_dataset} → {output_dir}")
    try:
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(kaggle_dataset, path=str(output_path),
                                           quiet=False, unzip=False)
    except Exception as e:
        raise RuntimeError(f"Kaggle download failed: {e}") from e

    zip_files = list(output_path.glob("*.zip"))
    if not zip_files:
        raise RuntimeError(f"No zip file found in {output_dir} after download.")
    zip_path = zip_files[0]

    logger.info(f"Extracting {zip_path} → {output_dir}")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(output_path))
    except zipfile.BadZipFile as e:
        raise RuntimeError(f"Extraction failed: {e}") from e

    extracted_marker.touch()
    logger.info("Dataset extraction complete.")
    return str(output_path)
