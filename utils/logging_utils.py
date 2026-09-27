"""
utils/logging_utils.py

Structured logging for the entire project.
Provides a unified logger that writes to:
  1. Console (human-readable)
  2. File log (persistent across Colab session)
  3. CSV experiment log (for easy pandas analysis later)
  4. TensorBoard (optional)
"""

import csv
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logging(
    log_dir: str,
    experiment_name: str,
    level: str = "INFO",
) -> logging.Logger:
    """
    Configure the root logger for the project.

    Creates:
        - Console handler (INFO level)
        - File handler (DEBUG level, writes to log_dir/experiment_name.log)

    Args:
        log_dir: Directory where log file will be written.
        experiment_name: Name used for the log file and in log prefixes.
        level: Console log level string (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured root logger.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, f"{experiment_name}.log")

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplicates on re-setup
    root_logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(fmt)
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)

    root_logger.info(f"Logging initialized. File: {log_file}")
    return root_logger


class ExperimentLogger:
    """
    Records per-epoch and per-step metrics to a CSV file.

    Each call to log_metrics() appends one row to the CSV. This allows
    the full training history to be loaded with pandas for analysis.

    Usage:
        logger = ExperimentLogger("./logs", "unet_fold0")
        logger.log_metrics(epoch=1, split="train", dice=0.82, loss=0.34)
    """

    def __init__(self, log_dir: str, experiment_name: str) -> None:
        """
        Args:
            log_dir: Directory for CSV file.
            experiment_name: Prefix for the CSV filename.
        """
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self._csv_path = os.path.join(log_dir, f"{experiment_name}_metrics.csv")
        self._file = open(self._csv_path, "a", newline="")
        self._writer: Optional[csv.DictWriter] = None
        self._logger = logging.getLogger(self.__class__.__name__)

    def log_metrics(self, **kwargs: Any) -> None:
        """
        Log an arbitrary set of key-value metrics.

        A 'timestamp' field is automatically added.

        Args:
            **kwargs: Metric names and values. Should include at minimum
                      'epoch' or 'step' for traceability.
        """
        kwargs["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        if self._writer is None:
            # Initialize writer with column names from first call
            fieldnames = list(kwargs.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
            self._writer.writeheader()

        self._writer.writerow(kwargs)
        self._file.flush()

        # Also emit to standard logging
        metric_str = " | ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in kwargs.items()
            if k != "timestamp"
        )
        self._logger.info(f"Metrics: {metric_str}")

    def close(self) -> None:
        """Flush and close the CSV file."""
        self._file.flush()
        self._file.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def log_config(config: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Log a configuration dictionary in a human-readable format.

    Args:
        config: Configuration as a nested dictionary (from OmegaConf).
        logger: Logger to emit to.
    """
    logger.info("=" * 50)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("=" * 50)
    _log_dict_recursive(config, logger, indent=0)
    logger.info("=" * 50)


def _log_dict_recursive(
    d: Dict[str, Any], logger: logging.Logger, indent: int
) -> None:
    prefix = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            logger.info(f"{prefix}{k}:")
            _log_dict_recursive(v, logger, indent + 1)
        else:
            logger.info(f"{prefix}{k}: {v}")
