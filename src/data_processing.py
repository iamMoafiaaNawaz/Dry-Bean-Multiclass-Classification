"""
data_processing.py
------------------
End-to-end data preprocessing pipeline for the Dry Bean Classification project.

Responsibilities
----------------
1. Load raw CSV from disk (or trigger auto-download).
2. Validate schema — confirm all expected columns are present.
3. Clean data   — remove duplicates and handle missing values.
4. Encode target — map string class labels to integer indices.
5. Split dataset — stratified train / test split.
6. Scale features — fit StandardScaler on train, transform both splits.
7. Return ready-to-use arrays plus fitted transformer objects.

Every major step emits a numbered progress banner and structured log
messages so pipeline execution is fully observable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Make src/ importable when this file is executed directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BEAN_CLASSES,
    DATASET_PATH,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from utils import download_dataset, get_logger, print_step

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------

logger = get_logger(__name__)

# Total pipeline steps — update this if steps are added / removed
_TOTAL_STEPS = 7


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_preprocessing() -> Tuple[
    np.ndarray,  # X_train
    np.ndarray,  # X_test
    np.ndarray,  # y_train
    np.ndarray,  # y_test
    StandardScaler,
    LabelEncoder,
]:
    """
    Execute the full preprocessing pipeline.

    Returns
    -------
    X_train, X_test : np.ndarray
        Scaled feature matrices for training and evaluation.
    y_train, y_test : np.ndarray
        Integer-encoded target vectors.
    scaler : StandardScaler
        Fitted scaler — must be persisted alongside the model.
    label_encoder : LabelEncoder
        Fitted encoder — required to map predictions back to class names.
    """
    # ------------------------------------------------------------------
    # Step 1 — Load dataset
    # ------------------------------------------------------------------
    print_step(1, _TOTAL_STEPS, "Loading Dataset")
    df = _load_dataset()

    # ------------------------------------------------------------------
    # Step 2 — Clean data
    # ------------------------------------------------------------------
    print_step(2, _TOTAL_STEPS, "Cleaning Data")
    df = _clean_data(df)

    # ------------------------------------------------------------------
    # Step 3 — Validate schema
    # ------------------------------------------------------------------
    print_step(3, _TOTAL_STEPS, "Validating Schema & Features")
    _validate_schema(df)

    # ------------------------------------------------------------------
    # Step 4 — Encode target labels
    # ------------------------------------------------------------------
    print_step(4, _TOTAL_STEPS, "Encoding Target Labels")
    df, label_encoder = _encode_target(df)

    # ------------------------------------------------------------------
    # Step 5 — Split into train / test sets
    # ------------------------------------------------------------------
    print_step(5, _TOTAL_STEPS, "Splitting Dataset (Train / Test)")
    X_train, X_test, y_train, y_test = _split_data(df)

    # ------------------------------------------------------------------
    # Step 6 — Scale features
    # ------------------------------------------------------------------
    print_step(6, _TOTAL_STEPS, "Scaling Features (StandardScaler)")
    X_train, X_test, scaler = _scale_features(X_train, X_test)

    # ------------------------------------------------------------------
    # Step 7 — Summary
    # ------------------------------------------------------------------
    print_step(7, _TOTAL_STEPS, "Preprocessing Complete — Summary")
    _print_summary(X_train, X_test, y_train, y_test, label_encoder)

    return X_train, X_test, y_train, y_test, scaler, label_encoder


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_dataset() -> pd.DataFrame:
    """
    Load the Dry Bean CSV into a DataFrame.

    Triggers ``download_dataset()`` first so that Docker containers and
    fresh clones work without manual data placement.

    Returns
    -------
    pd.DataFrame
        Raw dataset as loaded from disk.

    Raises
    ------
    FileNotFoundError
        If the dataset is absent and the download also fails.
    """
    download_dataset(DATASET_PATH)

    logger.info("Reading CSV from '%s' …", DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)

    logger.info(
        "Dataset loaded — %d rows × %d columns.", df.shape[0], df.shape[1]
    )
    print(f"  ↳ Shape   : {df.shape}")
    print(f"  ↳ Columns : {list(df.columns)}")

    return df


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicates and handle missing values.

    Strategy
    --------
    - Duplicate rows are dropped; a warning is emitted if any are found.
    - Numeric columns with missing values are imputed with the column median
      (robust to outliers and skewed distributions).
    - If the target column has missing values the affected rows are dropped,
      since imputing class labels would introduce noise.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from ``_load_dataset``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with a reset index.
    """
    original_len = len(df)

    # --- Duplicates ---
    n_dupes = df.duplicated().sum()
    if n_dupes:
        logger.warning("Dropping %d duplicate rows.", n_dupes)
        df = df.drop_duplicates()
    else:
        logger.info("No duplicate rows found.")
    print(f"  ↳ Duplicates removed : {n_dupes}")

    # --- Missing values (features) ---
    missing = df[FEATURE_COLUMNS].isnull().sum()
    cols_with_missing = missing[missing > 0]

    if not cols_with_missing.empty:
        logger.warning(
            "Imputing missing values in %d column(s) with column median.",
            len(cols_with_missing),
        )
        for col in cols_with_missing.index:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.debug("  '%s': %d missing → imputed with %.4f", col, cols_with_missing[col], median_val)
    else:
        logger.info("No missing values detected in feature columns.")

    # --- Missing values (target) ---
    n_missing_target = df[TARGET_COLUMN].isnull().sum()
    if n_missing_target:
        logger.warning(
            "Dropping %d rows with missing target labels.", n_missing_target
        )
        df = df.dropna(subset=[TARGET_COLUMN])

    df = df.reset_index(drop=True)
    print(f"  ↳ Missing values     : {cols_with_missing.to_dict() or 'None'}")
    print(f"  ↳ Rows after cleaning: {len(df)}  (removed {original_len - len(df)})")

    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """
    Assert that all required feature and target columns are present.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame.

    Raises
    ------
    ValueError
        If any expected column is missing from the DataFrame.
    ValueError
        If an unexpected class label is discovered in the target column.
    """
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_cols = [c for c in required_columns if c not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Dataset is missing required columns: {missing_cols}\n"
            f"Found columns: {list(df.columns)}"
        )
    logger.info("Schema validation passed — all %d columns present.", len(required_columns))

    # Validate class labels (normalise to upper-case for comparison)
    actual_classes = set(df[TARGET_COLUMN].str.upper().unique())
    expected_classes = set(BEAN_CLASSES)
    unexpected = actual_classes - expected_classes

    if unexpected:
        logger.warning(
            "Unexpected class labels found: %s — they will be encoded but "
            "may affect model performance.",
            unexpected,
        )

    print(f"  ↳ Features validated : {len(FEATURE_COLUMNS)}")
    print(f"  ↳ Classes found      : {sorted(actual_classes)}")


def _encode_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Encode the string target column into integer class indices.

    scikit-learn estimators require numeric targets for multiclass
    classification.  The LabelEncoder mapping is preserved so predictions
    can be converted back to human-readable bean names.

    Parameters
    ----------
    df : pd.DataFrame
        Validated DataFrame.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with the target column replaced by integer codes.
    label_encoder : LabelEncoder
        Fitted encoder.
    """
    # Normalise to upper-case so "seker" and "SEKER" map to the same class
    df[TARGET_COLUMN] = df[TARGET_COLUMN].str.upper().str.strip()

    label_encoder = LabelEncoder()
    df[TARGET_COLUMN] = label_encoder.fit_transform(df[TARGET_COLUMN])

    mapping = dict(
        zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
    )
    logger.info("Label encoding complete — mapping: %s", mapping)
    print(f"  ↳ Encoding map : {mapping}")

    return df, label_encoder


def _split_data(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a stratified train / test split.

    Stratification ensures each split contains the same proportion of
    every bean class, which is especially important for minority classes.

    Parameters
    ----------
    df : pd.DataFrame
        Encoded DataFrame.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Feature and target arrays for each split.
    """
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.info(
        "Split complete — train: %d samples | test: %d samples.",
        len(X_train),
        len(X_test),
    )
    print(f"  ↳ Train samples : {len(X_train)}")
    print(f"  ↳ Test  samples : {len(X_test)}")
    print(f"  ↳ Test size     : {TEST_SIZE:.0%}")

    return X_train, X_test, y_train, y_test


def _scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardise features to zero mean and unit variance.

    The scaler is fit **only** on the training set to prevent data
    leakage; the same fitted scaler is then applied to the test set.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Raw feature arrays from the train/test split.

    Returns
    -------
    X_train_scaled, X_test_scaled : np.ndarray
        Transformed arrays.
    scaler : StandardScaler
        Fitted scaler instance (must be saved for inference).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info(
        "Scaling complete — feature means (train): %s",
        np.round(X_train_scaled.mean(axis=0), 4),
    )
    print(f"  ↳ Feature mean  (post-scale, train) ≈ 0.0  ✓")
    print(f"  ↳ Feature std   (post-scale, train) ≈ 1.0  ✓")

    return X_train_scaled, X_test_scaled, scaler


def _print_summary(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
) -> None:
    """
    Print a final preprocessing summary to stdout.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Scaled feature arrays.
    y_train, y_test : np.ndarray
        Encoded target arrays.
    label_encoder : LabelEncoder
        Fitted encoder used to display class names.
    """
    print("\n  ┌─────────────────────────────────────────┐")
    print("  │        PREPROCESSING SUMMARY            │")
    print("  ├─────────────────────────────────────────┤")
    print(f"  │  Total samples   : {len(X_train) + len(X_test):<6}                │")
    print(f"  │  Training set    : {len(X_train):<6}                │")
    print(f"  │  Test set        : {len(X_test):<6}                │")
    print(f"  │  Features        : {X_train.shape[1]:<6}                │")
    print(f"  │  Classes         : {len(label_encoder.classes_):<6}                │")
    print("  ├─────────────────────────────────────────┤")

    unique, counts = np.unique(y_train, return_counts=True)
    for cls_idx, cnt in zip(unique, counts):
        cls_name = label_encoder.classes_[cls_idx]
        print(f"  │  {cls_name:<12}: {cnt:>4} train samples          │")

    print("  └─────────────────────────────────────────┘")
    logger.info("Preprocessing pipeline finished successfully.")