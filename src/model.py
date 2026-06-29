"""
model.py
--------
Model definitions, training, evaluation, and selection logic for the
Dry Bean Classification project.

Responsibilities
----------------
- Build Random Forest and XGBoost classifier pipelines.
- Train each model and capture per-class and aggregate metrics.
- Generate confusion matrices and classification reports.
- Automatically select the best model by weighted F1-score.
- Persist the winning model, scaler, and label encoder to disk.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    LABEL_ENCODER_PATH,
    MINIMUM_ACCURACY,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_FOREST_PARAMS,
    RANDOM_STATE,
    SCALER_PATH,
    XGBOOST_PARAMS,
    MINIMUM_ACCURACY,
)
from utils import get_logger, timer

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_random_forest() -> RandomForestClassifier:
    """
    Instantiate a Random Forest classifier with project hyperparameters.

    Returns
    -------
    RandomForestClassifier
        Untrained classifier ready for ``.fit()``.
    """
    logger.info("Building Random Forest with params: %s", RANDOM_FOREST_PARAMS)
    return RandomForestClassifier(**RANDOM_FOREST_PARAMS)


def build_xgboost() -> XGBClassifier:
    """
    Instantiate an XGBoost classifier with project hyperparameters.

    Returns
    -------
    XGBClassifier
        Untrained classifier ready for ``.fit()``.
    """
    logger.info("Building XGBoost with params: %s", XGBOOST_PARAMS)
    return XGBClassifier(**XGBOOST_PARAMS)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@timer
def train_model(model, X_train: np.ndarray, y_train: np.ndarray):
    """
    Fit a classifier on the training data.

    Parameters
    ----------
    model :
        Any scikit-learn compatible estimator.
    X_train : np.ndarray
        Scaled training features.
    y_train : np.ndarray
        Encoded training labels.

    Returns
    -------
    model
        The same estimator after calling ``.fit()``.
    """
    model_name = type(model).__name__
    logger.info("Training %s on %d samples …", model_name, len(X_train))
    model.fit(X_train, y_train)
    logger.info("%s training complete.", model_name)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
) -> Dict[str, float]:
    """
    Compute classification metrics and print a full report.

    Metrics computed
    ----------------
    - Accuracy
    - Weighted Precision
    - Weighted Recall
    - Weighted F1-score

    Parameters
    ----------
    model :
        Trained classifier.
    X_test : np.ndarray
        Scaled test features.
    y_test : np.ndarray
        True encoded labels.
    label_encoder : LabelEncoder
        Maps integer codes → bean class names for the report.

    Returns
    -------
    dict
        ``{"accuracy": float, "precision": float, "recall": float, "f1": float}``
    """
    model_name = type(model).__name__
    y_pred = model.predict(X_test)
    class_names = label_encoder.classes_

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n  {'─'*52}")
    print(f"  Results — {model_name}")
    print(f"  {'─'*52}")
    print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  {'─'*52}")

    print(f"\n  Classification Report — {model_name}")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            digits=4,
        )
    )

    logger.info(
        "%s → Accuracy=%.4f | Precision=%.4f | Recall=%.4f | F1=%.4f",
        model_name,
        accuracy,
        precision,
        recall,
        f1,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def plot_confusion_matrix(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
) -> None:
    """
    Generate and save a confusion matrix heatmap for a trained model.

    The figure is saved to ``models/<ModelName>_confusion_matrix.png``.

    Parameters
    ----------
    model :
        Trained classifier.
    X_test : np.ndarray
        Scaled test features.
    y_test : np.ndarray
        True encoded labels.
    label_encoder : LabelEncoder
        Provides human-readable class names for axis labels.
    """
    model_name = type(model).__name__
    y_pred = model.predict(X_test)
    class_names = label_encoder.classes_

    cm = confusion_matrix(y_test, y_pred)
    cm_normalised = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")

    # Raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
    )
    axes[0].set_title("Raw Counts")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # Normalised
    sns.heatmap(
        cm_normalised,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
    )
    axes[1].set_title("Normalised (Row %)")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.tight_layout()

    output_path = MODELS_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Confusion matrix saved → %s", output_path)
    print(f"  ↳ Confusion matrix saved : {output_path}")


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def select_best_model(
    models: Dict[str, Tuple[object, Dict[str, float]]]
) -> Tuple[str, object]:
    """
    Choose the best model based on weighted F1-score.

    Parameters
    ----------
    models : dict
        ``{ "Model Name": (fitted_model, metrics_dict), … }``

    Returns
    -------
    best_name : str
        Name of the winning model.
    best_model :
        The winning fitted estimator.

    Raises
    ------
    ValueError
        If ``models`` is empty.
    """
    if not models:
        raise ValueError("No models provided for selection.")

    best_name = max(models, key=lambda name: models[name][1]["f1"])
    best_model, best_metrics = models[best_name]

    print(f"\n  {'='*52}")
    print(f"  🏆  Best Model : {best_name}")
    print(f"       Accuracy  : {best_metrics['accuracy']:.4f}")
    print(f"       F1-Score  : {best_metrics['f1']:.4f}")
    print(f"  {'='*52}")

    logger.info("Best model selected: %s (F1=%.4f)", best_name, best_metrics["f1"])
    return best_name, best_model


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_artefacts(
    model,
    scaler: StandardScaler,
    label_encoder: LabelEncoder,
) -> None:
    """
    Persist the winning model, scaler, and label encoder to disk.

    All three artefacts are required at inference time:
    - ``model.joblib``         — the fitted classifier
    - ``scaler.joblib``        — to standardise new samples
    - ``label_encoder.joblib`` — to convert predictions → class names

    Parameters
    ----------
    model :
        Best fitted estimator selected by ``select_best_model``.
    scaler : StandardScaler
        Fitted scaler from the preprocessing pipeline.
    label_encoder : LabelEncoder
        Fitted label encoder from the preprocessing pipeline.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model,         MODEL_PATH)
    joblib.dump(scaler,        SCALER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    logger.info("Model saved        → %s", MODEL_PATH)
    logger.info("Scaler saved       → %s", SCALER_PATH)
    logger.info("Label encoder saved→ %s", LABEL_ENCODER_PATH)

    print(f"\n  ↳ Model saved         : {MODEL_PATH}")
    print(f"  ↳ Scaler saved        : {SCALER_PATH}")
    print(f"  ↳ Label encoder saved : {LABEL_ENCODER_PATH}")


def load_artefacts() -> Tuple[object, StandardScaler, LabelEncoder]:
    """
    Load all persisted artefacts required for inference.

    Returns
    -------
    model, scaler, label_encoder
        The same objects that were saved by ``save_artefacts``.

    Raises
    ------
    FileNotFoundError
        If any artefact file is missing — run ``train.py`` first.
    """
    for path in (MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Artefact not found: '{path}'. "
                "Run 'python src/train.py' to generate model files."
            )

    model         = joblib.load(MODEL_PATH)
    scaler        = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    logger.info(
        "Artefacts loaded — model: %s", type(model).__name__
    )
    return model, scaler, label_encoder


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

def assert_minimum_accuracy(metrics: Dict[str, float], model_name: str) -> None:
    """
    Raise an error if the best model's accuracy falls below the threshold.

    Parameters
    ----------
    metrics : dict
        Metrics dict returned by ``evaluate_model``.
    model_name : str
        Name used in the error message.

    Raises
    ------
    RuntimeError
        If ``accuracy < config.MINIMUM_ACCURACY``.
    """
    acc = metrics["accuracy"]
    if acc < MINIMUM_ACCURACY:
        raise RuntimeError(
            f"Quality gate failed: {model_name} achieved accuracy={acc:.4f}, "
            f"which is below the required threshold of {MINIMUM_ACCURACY:.4f}. "
            "Review feature engineering or hyperparameters."
        )
    logger.info(
        "Quality gate passed: accuracy=%.4f ≥ threshold=%.4f",
        acc,
        MINIMUM_ACCURACY,
    )
    print(f"\n  ✅ Quality gate passed — accuracy {acc:.4f} ≥ {MINIMUM_ACCURACY:.4f}")