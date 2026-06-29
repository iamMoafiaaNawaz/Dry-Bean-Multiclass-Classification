"""
utils.py
--------
Shared utilities for the Dry Bean Classification project.

Provides:
- Centralised logging factory
- Step-progress printer  ([1/7] Loading Dataset …)
- Execution-time decorator
- Artefact-directory initialisation
- Reproducibility seed helper
"""

from __future__ import annotations

import logging
import os
import random
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import numpy as np

from config import LOG_DATE_FORMAT, LOG_FORMAT, LOG_LEVEL, MODELS_DIR, DATA_DIR


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger configured with the project-wide format.

    All loggers share the same StreamHandler so output is consistent
    regardless of which module emits the message.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        )
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    return logger


# Module-level logger used by the helpers below
_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Step-progress printer
# ---------------------------------------------------------------------------

def print_step(step: int, total: int, description: str) -> None:
    """
    Print a formatted pipeline-step banner.

    Example output::

        ============================================================
        [3/7] Scaling Features
        ============================================================

    Parameters
    ----------
    step:
        Current step number (1-based).
    total:
        Total number of steps in the pipeline.
    description:
        Human-readable label for this step.
    """
    banner = "=" * 60
    print(f"\n{banner}")
    print(f"  [{step}/{total}] {description}")
    print(f"{banner}")
    _log.info("Step [%d/%d] — %s", step, total, description)


# ---------------------------------------------------------------------------
# Timing decorator
# ---------------------------------------------------------------------------

def timer(func: Callable) -> Callable:
    """
    Decorator that logs how long a function takes to execute.

    Usage::

        @timer
        def train_model(...):
            ...

    Parameters
    ----------
    func:
        The function to wrap.

    Returns
    -------
    Callable
        Wrapped function with timing side-effect.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        _log.info("'%s' completed in %.2f seconds.", func.__name__, elapsed)
        return result

    return wrapper


# ---------------------------------------------------------------------------
# Directory initialisation
# ---------------------------------------------------------------------------

def ensure_directories() -> None:
    """
    Create required project directories if they do not already exist.

    Called once at the start of ``train.py`` so the rest of the pipeline
    can assume ``data/`` and ``models/`` are always present.
    """
    for directory in (DATA_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        _log.debug("Directory ready: %s", directory)

    _log.info("Project directories verified.")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and the OS environment.

    Ensures that shuffling, train/test splits, and model initialisations
    are deterministic across runs given the same seed.

    Parameters
    ----------
    seed:
        Integer seed value (use ``config.RANDOM_STATE``).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    _log.info("Global random seed set to %d.", seed)


# ---------------------------------------------------------------------------
# Metrics pretty-printer
# ---------------------------------------------------------------------------

def print_metrics_table(metrics: dict[str, dict[str, float]]) -> None:
    """
    Print a formatted comparison table for multiple models.

    Parameters
    ----------
    metrics:
        Mapping of model name → dict of metric name → value.
        Example::

            {
                "Random Forest": {"accuracy": 0.974, "f1": 0.973},
                "XGBoost":       {"accuracy": 0.981, "f1": 0.980},
            }
    """
    if not metrics:
        _log.warning("No metrics to display.")
        return

    col_width = 18
    metric_keys = list(next(iter(metrics.values())).keys())

    header = f"{'Model':<20}" + "".join(f"{k:>{col_width}}" for k in metric_keys)
    separator = "-" * len(header)

    print(f"\n{separator}")
    print(header)
    print(separator)

    for model_name, values in metrics.items():
        row = f"{model_name:<20}" + "".join(
            f"{v:>{col_width}.4f}" for v in values.values()
        )
        print(row)

    print(separator)


# ---------------------------------------------------------------------------
# Dataset download helper
# ---------------------------------------------------------------------------

def download_dataset(destination: Path) -> None:
    """
    Download the Dry Bean Dataset CSV from a public mirror if it is absent.

    Uses only the Python standard library (``urllib``) to avoid adding an
    extra dependency.  The function is idempotent — if the file already
    exists it returns immediately without making a network request.

    Parameters
    ----------
    destination:
        Full path where the CSV should be saved
        (typically ``config.DATASET_PATH``).

    Raises
    ------
    RuntimeError
        If the download fails and no local copy exists.
    """
    if destination.exists():
        _log.info("Dataset already present at '%s'. Skipping download.", destination)
        return

    import urllib.request

    # Public CSV mirror hosted on GitHub
    url = (
        "https://raw.githubusercontent.com/"
        "dsaks/dry-bean-dataset/main/Dry_Bean_Dataset.csv"
    )

    _log.info("Dataset not found locally. Downloading from remote …")
    print(f"  ↳ Source : {url}")
    print(f"  ↳ Target : {destination}")

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, destination)
        _log.info("Download complete — %s", destination)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download dataset from '{url}'. "
            "Please place 'Dry_Bean_Dataset.csv' inside the 'data/' directory manually."
        ) from exc