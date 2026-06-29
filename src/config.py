"""
config.py
---------
Centralised configuration for the Dry Bean Classification project.

All paths, constants, model hyperparameters, and pipeline settings live here
so that every other module stays free of hard-coded magic values.  Changing a
training parameter, output path, or dataset URL therefore requires touching
exactly one file.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

# Repository root is two levels above this file: project/src/config.py → project/
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = ROOT_DIR / "data"
MODELS_DIR: Path = ROOT_DIR / "models"
SRC_DIR: Path = ROOT_DIR / "src"

# Raw dataset file expected inside data/
DATASET_FILENAME: str = "Dry_Bean_Dataset.csv"
DATASET_PATH: Path = DATA_DIR / DATASET_FILENAME

# Persisted artefacts written by train.py and consumed by predict.py
MODEL_PATH: Path = MODELS_DIR / "model.joblib"
SCALER_PATH: Path = MODELS_DIR / "scaler.joblib"
LABEL_ENCODER_PATH: Path = MODELS_DIR / "label_encoder.joblib"

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------

# Seven bean varieties the model must distinguish
TARGET_COLUMN: str = "Class"

BEAN_CLASSES: list[str] = [
    "SEKER",
    "BARBUNYA",
    "BOMBAY",
    "CALI",
    "DERMASON",
    "HOROZ",
    "SIRA",
]

# All 16 morphological features provided by the dataset
FEATURE_COLUMNS: list[str] = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

TEST_SIZE: float = 0.20          # 80 / 20 split
RANDOM_STATE: int = 42           # seed for reproducibility everywhere

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------

# Random Forest — tuned for this dataset via prior grid search
RANDOM_FOREST_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": None,           # grow full trees; rely on min_samples_leaf
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": "balanced",  # handles mild class imbalance gracefully
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

# XGBoost — gradient-boosted trees with early-stopping-friendly settings
XGBOOST_PARAMS: dict = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric": "mlogloss",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

# train.py raises an error if the winning model falls below this threshold
MINIMUM_ACCURACY: float = 0.90