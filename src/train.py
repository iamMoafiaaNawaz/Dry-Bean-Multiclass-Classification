"""
train.py
--------
Main training entry-point for the Dry Bean Classification project.

Execution flow
--------------
1.  Initialise directories and global random seed.
2.  Run the full preprocessing pipeline (load → clean → encode → split → scale).
3.  Train Random Forest and XGBoost classifiers.
4.  Evaluate both models and print metrics + confusion matrices.
5.  Automatically select the best model by weighted F1-score.
6.  Assert the quality gate (accuracy ≥ 95 %).
7.  Persist the winning model, scaler, and label encoder to disk.

Usage
-----
    # From the project root:
    python src/train.py

    # Inside Docker:
    docker run dry-bean-classifier
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure src/ is on the path regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RANDOM_STATE
from data_processing import run_preprocessing
from model import (
    assert_minimum_accuracy,
    build_random_forest,
    build_xgboost,
    evaluate_model,
    plot_confusion_matrix,
    save_artefacts,
    select_best_model,
    train_model,
)
from utils import (
    ensure_directories,
    get_logger,
    print_step,
    print_metrics_table,
    set_global_seed,
)

logger = get_logger(__name__)

_TRAIN_STEPS = 5   # steps specific to the training phase (after preprocessing)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrate the end-to-end training pipeline.

    All side effects (directory creation, file writes, stdout output) happen
    inside this function so the module is importable without side effects.
    """
    pipeline_start = time.perf_counter()

    _print_header()

    # ------------------------------------------------------------------
    # Initialise environment
    # ------------------------------------------------------------------
    ensure_directories()
    set_global_seed(RANDOM_STATE)

    # ------------------------------------------------------------------
    # Preprocessing  (steps 1-7 printed inside run_preprocessing)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test, scaler, label_encoder = run_preprocessing()

    # ------------------------------------------------------------------
    # Step 1/5 — Train Random Forest
    # ------------------------------------------------------------------
    print_step(1, _TRAIN_STEPS, "Training Random Forest")
    rf_model = build_random_forest()
    rf_model  = train_model(rf_model, X_train, y_train)

    # ------------------------------------------------------------------
    # Step 2/5 — Train XGBoost
    # ------------------------------------------------------------------
    print_step(2, _TRAIN_STEPS, "Training XGBoost")
    xgb_model = build_xgboost()
    xgb_model  = train_model(xgb_model, X_train, y_train)

    # ------------------------------------------------------------------
    # Step 3/5 — Evaluate both models
    # ------------------------------------------------------------------
    print_step(3, _TRAIN_STEPS, "Evaluating Models")

    rf_metrics  = evaluate_model(rf_model,  X_test, y_test, label_encoder)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, label_encoder)

    # Side-by-side comparison table
    print_metrics_table(
        {
            "Random Forest": rf_metrics,
            "XGBoost":       xgb_metrics,
        }
    )

    # Confusion matrices (saved as PNG files)
    plot_confusion_matrix(rf_model,  X_test, y_test, label_encoder)
    plot_confusion_matrix(xgb_model, X_test, y_test, label_encoder)

    # ------------------------------------------------------------------
    # Step 4/5 — Select best model
    # ------------------------------------------------------------------
    print_step(4, _TRAIN_STEPS, "Selecting Best Model")

    best_name, best_model = select_best_model(
        {
            "Random Forest": (rf_model,  rf_metrics),
            "XGBoost":       (xgb_model, xgb_metrics),
        }
    )

    best_metrics = rf_metrics if best_name == "Random Forest" else xgb_metrics
    assert_minimum_accuracy(best_metrics, best_name)

    # ------------------------------------------------------------------
    # Step 5/5 — Save artefacts
    # ------------------------------------------------------------------
    print_step(5, _TRAIN_STEPS, "Saving Model Artefacts")
    save_artefacts(best_model, scaler, label_encoder)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - pipeline_start
    _print_footer(best_name, best_metrics, elapsed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_header() -> None:
    """Print a startup banner to stdout."""
    print("\n")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       DRY BEAN MULTICLASS CLASSIFICATION PIPELINE       ║")
    print("║                   Training  Mode                        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    logger.info("Training pipeline started.")


def _print_footer(
    best_name: str,
    metrics: dict,
    elapsed: float,
) -> None:
    """Print a completion summary banner."""
    print("\n")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║                  TRAINING COMPLETE ✅                   ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Best Model  : {best_name:<42}║")
    print(f"║  Accuracy    : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)                      ║")
    print(f"║  F1-Score    : {metrics['f1']:.4f}                                   ║")
    print(f"║  Total time  : {elapsed:>6.1f}s                                  ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  Artefacts saved:                                        ║")
    print("║    models/model.joblib                                   ║")
    print("║    models/scaler.joblib                                  ║")
    print("║    models/label_encoder.joblib                           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    logger.info(
        "Pipeline complete — best=%s | acc=%.4f | f1=%.4f | time=%.1fs",
        best_name,
        metrics["accuracy"],
        metrics["f1"],
        elapsed,
    )


# ---------------------------------------------------------------------------
# Script guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()