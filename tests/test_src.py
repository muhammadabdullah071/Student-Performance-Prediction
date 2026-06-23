"""Tests for the Student Performance Prediction project."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd

from src.data_preprocessing import DataPreprocessor
from src.evaluate import ModelEvaluator, RegressionMetrics
from src.utils import validate_data, set_seed


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""

    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            "Student_ID": range(1, n + 1),
            "Gender": np.random.choice(["Male", "Female"], n),
            "Parent_Education": np.random.choice(["High School", "College", "Postgraduate"], n),
            "Access_to_Resources": np.random.choice(["Low", "Medium", "High"], n),
            "Motivation_Level": np.random.choice(["Low", "Medium", "High"], n),
            "Study_Hours": np.random.uniform(0, 40, n),
            "Attendance": np.random.uniform(40, 100, n),
            "Sleep_Hours": np.random.uniform(3, 11, n),
            "Previous_Grade": np.random.uniform(30, 100, n),
            "Assignments_Completed": np.random.randint(0, 20, n),
            "Participation_Score": np.random.uniform(0, 100, n),
            "Exam_Score": np.random.uniform(0, 100, n),
        })

    def test_initialization(self):
        dp = DataPreprocessor()
        assert dp.test_size == 0.2
        assert dp.val_size == 0.1
        assert dp.random_state == 42
        assert dp.target_column == "Exam_Score"

    def test_custom_initialization(self):
        dp = DataPreprocessor(test_size=0.3, val_size=0.15, random_state=123, target_column="Score")
        assert dp.test_size == 0.3
        assert dp.val_size == 0.15
        assert dp.random_state == 123
        assert dp.target_column == "Score"

    def test_clean_data_drops_student_id(self, sample_df):
        dp = DataPreprocessor()
        cleaned = dp.clean_data(sample_df)
        assert "Student_ID" not in cleaned.columns

    def test_clean_data_removes_duplicates(self, sample_df):
        dp = DataPreprocessor()
        duplicated = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
        cleaned = dp.clean_data(duplicated)
        assert len(cleaned) == len(sample_df)

    def test_clean_data_imputes_missing(self):
        dp = DataPreprocessor()
        n = 50
        df = pd.DataFrame({
            "Gender": ["Male"] * n,
            "Parent_Education": ["College"] * n,
            "Access_to_Resources": ["Medium"] * n,
            "Motivation_Level": ["Medium"] * n,
            "Study_Hours": [10.0] * n,
            "Attendance": [80.0] * n,
            "Sleep_Hours": [7.0] * n,
            "Previous_Grade": [70.0] * n,
            "Assignments_Completed": [10] * n,
            "Participation_Score": [60.0] * n,
            "Exam_Score": [75.0] * n,
        })
        df.loc[0, "Study_Hours"] = np.nan
        cleaned = dp.clean_data(df)
        assert cleaned["Study_Hours"].isnull().sum() == 0

    def test_split_data_shapes(self, sample_df):
        dp = DataPreprocessor()
        cleaned = dp.clean_data(sample_df)
        X_train, X_val, X_test, y_train, y_val, y_test = dp.split_data(cleaned)

        total = len(cleaned)
        test_size = int(total * 0.2)
        remaining = total - test_size
        val_size = int(remaining * (0.1 / 0.8))

        assert len(X_test) == test_size or len(X_test) == test_size + 1
        assert len(X_train) + len(X_val) + len(X_test) == total

    def test_fit_transform_output_shape(self, sample_df):
        dp = DataPreprocessor()
        cleaned = dp.clean_data(sample_df)
        X_train, X_val, X_test, y_train, y_val, y_test = dp.split_data(cleaned)

        X_train_t, X_val_t, X_test_t = dp.fit_transform(X_train, X_val, X_test)

        assert X_train_t.shape[1] == X_val_t.shape[1] == X_test_t.shape[1]
        assert X_train_t.shape[0] == len(X_train)
        assert X_val_t.shape[0] == len(X_val)
        assert X_test_t.shape[0] == len(X_test)


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""

    @pytest.fixture
    def y_true(self):
        return np.array([80, 75, 90, 65, 70])

    @pytest.fixture
    def y_pred_perfect(self):
        return np.array([80, 75, 90, 65, 70])

    @pytest.fixture
    def y_pred_imperfect(self):
        return np.array([78, 72, 88, 68, 72])

    def test_calculate_metrics_perfect(self, y_true, y_pred_perfect):
        metrics = ModelEvaluator.calculate_metrics(y_true, y_pred_perfect)
        assert metrics.mae == 0.0
        assert metrics.mse == 0.0
        assert metrics.rmse == 0.0
        assert metrics.r2 == 1.0

    def test_calculate_metrics_imperfect(self, y_true, y_pred_imperfect):
        metrics = ModelEvaluator.calculate_metrics(y_true, y_pred_imperfect)
        assert metrics.mae > 0
        assert metrics.mse > 0
        assert metrics.rmse > 0
        assert 0 < metrics.r2 < 1

    def test_metrics_types(self, y_true, y_pred_imperfect):
        metrics = ModelEvaluator.calculate_metrics(y_true, y_pred_imperfect)
        assert isinstance(metrics, RegressionMetrics)
        assert isinstance(metrics.mae, float)
        assert isinstance(metrics.mse, float)
        assert isinstance(metrics.rmse, float)
        assert isinstance(metrics.r2, float)

    def test_get_best_model(self):
        metrics = {
            "Model_A": RegressionMetrics(mae=5.0, mse=30.0, rmse=5.48, r2=0.75),
            "Model_B": RegressionMetrics(mae=3.0, mse=15.0, rmse=3.87, r2=0.88),
            "Model_C": RegressionMetrics(mae=4.0, mse=22.0, rmse=4.69, r2=0.81),
        }
        best_name, best_metrics = ModelEvaluator.get_best_model(metrics)
        assert best_name == "Model_B"
        assert best_metrics.r2 == 0.88


class TestUtils:
    """Tests for utility functions."""

    def test_validate_data_structure(self):
        df = pd.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50],
            "C": ["x", "y", "z", "x", "y"],
        })
        results = validate_data(df)
        assert "shape" in results
        assert "missing_values" in results
        assert "duplicate_rows" in results
        assert "outliers" in results
        assert results["shape"] == (5, 3)

    def test_validate_data_with_missing(self):
        df = pd.DataFrame({
            "A": [1, 2, np.nan, 4, 5],
            "B": [10, 20, 30, np.nan, 50],
        })
        results = validate_data(df)
        assert results["total_missing"] == 2

    def test_set_seed(self):
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        assert np.array_equal(a, b)