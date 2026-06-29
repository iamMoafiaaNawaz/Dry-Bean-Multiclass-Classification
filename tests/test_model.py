"""
test_model.py
-------------
Pytest test suite for the Dry Bean Classification project.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from config import (
    BEAN_CLASSES,
    FEATURE_COLUMNS,
    LABEL_ENCODER_PATH,
    MINIMUM_ACCURACY,
    MODEL_PATH,
    SCALER_PATH,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def artefacts():
    from model import load_artefacts
    return load_artefacts()


@pytest.fixture(scope="session")
def sample_input() -> list[float]:
    return [
        28395.0, 610.291, 208.178, 173.888,
        1.197,   0.549,  28715.0, 190.141,
        0.763,   0.989,  0.958,   0.913,
        0.007,   0.003,  0.834,   0.998,
    ]


@pytest.fixture(scope="session")
def batch_inputs() -> list[list[float]]:
    return [
        [28395.0, 610.291, 208.178, 173.888, 1.197, 0.549,
         28715.0, 190.141, 0.763,   0.989,   0.958, 0.913,
         0.007,   0.003,   0.834,   0.998],
        [33263.0, 673.698, 249.481, 170.170, 1.466, 0.734,
         33756.0, 205.743, 0.781,   0.985,   0.861, 0.931,
         0.007,   0.003,   0.718,   0.996],
        [49887.0, 820.932, 326.098, 195.932, 1.664, 0.796,
         51109.0, 252.177, 0.761,   0.976,   0.931, 0.774,
         0.005,   0.002,   0.716,   0.998],
        [99456.0, 1139.60, 478.459, 266.365, 1.796, 0.831,
        100832.0, 355.949, 0.748,   0.986,   0.961, 0.728,
         0.003,   0.001,   0.649,   0.998],
        [53078.0, 878.429, 360.869, 190.214, 1.896, 0.849,
         55103.0, 259.950, 0.737,   0.963,   0.862, 0.726,
         0.005,   0.002,   0.608,   0.997],
    ]


# ---------------------------------------------------------------------------
# 1. Artefact existence tests
# ---------------------------------------------------------------------------

class TestArtefactExistence:

    def test_model_file_exists(self):
        assert MODEL_PATH.exists(), f"Model file not found at '{MODEL_PATH}'."

    def test_scaler_file_exists(self):
        assert SCALER_PATH.exists(), f"Scaler file not found at '{SCALER_PATH}'."

    def test_label_encoder_file_exists(self):
        assert LABEL_ENCODER_PATH.exists(), f"Label encoder not found at '{LABEL_ENCODER_PATH}'."


# ---------------------------------------------------------------------------
# 2. Artefact loading tests
# ---------------------------------------------------------------------------

class TestArtefactLoading:

    def test_model_loads_successfully(self, artefacts):
        model, _, _ = artefacts
        assert hasattr(model, "predict")

    def test_scaler_loads_successfully(self, artefacts):
        _, scaler, _ = artefacts
        assert hasattr(scaler, "transform")

    def test_label_encoder_loads_successfully(self, artefacts):
        _, _, label_encoder = artefacts
        assert hasattr(label_encoder, "inverse_transform")

    def test_label_encoder_knows_all_classes(self, artefacts):
        _, _, label_encoder = artefacts
        assert set(label_encoder.classes_) == set(BEAN_CLASSES)


# ---------------------------------------------------------------------------
# 3. Single prediction tests
# ---------------------------------------------------------------------------

class TestSinglePrediction:

    def test_prediction_returns_dict(self, sample_input):
        from predict import predict
        result = predict(sample_input)
        assert isinstance(result, dict)

    def test_prediction_has_required_keys(self, sample_input):
        from predict import predict
        result = predict(sample_input)
        required = {"predicted_class", "predicted_index", "confidence", "all_probabilities"}
        assert required.issubset(result.keys())

    def test_predicted_class_is_valid_bean(self, sample_input):
        from predict import predict
        result = predict(sample_input)
        assert result["predicted_class"] in BEAN_CLASSES

    def test_predicted_index_is_integer(self, sample_input):
        from predict import predict
        result = predict(sample_input)
        assert isinstance(result["predicted_index"], int)
        assert result["predicted_index"] >= 0

    def test_confidence_in_valid_range(self, sample_input):
        from predict import predict
        result = predict(sample_input)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_all_probabilities_sum_to_one(self, sample_input):
        from predict import predict
        result = predict(sample_input)
        total = sum(result["all_probabilities"].values())
        assert abs(total - 1.0) < 1e-4

    def test_all_probabilities_cover_all_classes(self, sample_input):
        from predict import predict
        result = predict(sample_input)
        assert set(result["all_probabilities"].keys()) == set(BEAN_CLASSES)

    def test_all_probabilities_non_negative(self, sample_input):
        from predict import predict
        result = predict(sample_input)
        for cls, prob in result["all_probabilities"].items():
            assert prob >= 0.0


# ---------------------------------------------------------------------------
# 4. Input validation tests
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_too_few_features_raises_value_error(self):
        from predict import predict
        with pytest.raises(ValueError, match="Expected 16 feature values"):
            predict([1.0, 2.0, 3.0])

    def test_too_many_features_raises_value_error(self):
        from predict import predict
        with pytest.raises(ValueError, match="Expected 16 feature values"):
            predict([1.0] * 20)

    def test_non_numeric_feature_raises_value_error(self, sample_input):
        from predict import predict
        bad_input = list(sample_input)
        bad_input[0] = "not_a_number"
        with pytest.raises(ValueError):
            predict(bad_input)


# ---------------------------------------------------------------------------
# 5. Batch prediction tests
# ---------------------------------------------------------------------------

class TestBatchPrediction:

    def test_batch_output_length_matches_input(self, batch_inputs):
        from predict import predict_batch
        results = predict_batch(batch_inputs)
        assert len(results) == len(batch_inputs)

    def test_no_missing_predictions(self, batch_inputs):
        from predict import predict_batch
        results = predict_batch(batch_inputs)
        for i, result in enumerate(results):
            assert result is not None
            assert "predicted_class" in result

    def test_all_batch_classes_are_valid(self, batch_inputs):
        from predict import predict_batch
        results = predict_batch(batch_inputs)
        for result in results:
            assert result["predicted_class"] in BEAN_CLASSES


# ---------------------------------------------------------------------------
# 6. Accuracy gate tests
# ---------------------------------------------------------------------------

class TestAccuracyGate:

    def test_model_accuracy_above_threshold(self, artefacts):
        from data_processing import run_preprocessing
        from sklearn.metrics import accuracy_score

        model, scaler, label_encoder = artefacts
        _, X_test, _, y_test, _, _ = run_preprocessing()

        y_pred   = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        assert accuracy >= MINIMUM_ACCURACY, (
            f"Accuracy {accuracy:.4f} below threshold {MINIMUM_ACCURACY:.4f}"
        )

    def test_prediction_output_shape(self, artefacts, batch_inputs):
        model, scaler, _ = artefacts
        X        = np.array(batch_inputs, dtype=float)
        X_scaled = scaler.transform(X)
        y_pred   = model.predict(X_scaled)
        assert y_pred.shape == (len(batch_inputs),)