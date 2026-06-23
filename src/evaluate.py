"""Model evaluation utilities for regression tasks."""

from typing import Dict, Tuple, Any

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from pydantic import BaseModel, Field


class RegressionMetrics(BaseModel):
    """Container for regression evaluation metrics."""
    mae: float = Field(description="Mean Absolute Error")
    mse: float = Field(description="Mean Squared Error")
    rmse: float = Field(description="Root Mean Squared Error")
    r2: float = Field(description="R-squared Score")

    class Config:
        frozen = True


class ModelEvaluator:
    """Evaluates regression models using multiple metrics."""

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
        """Calculate all regression metrics.

        Args:
            y_true: Ground truth target values
            y_pred: Predicted target values

        Returns:
            RegressionMetrics object with all metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        return RegressionMetrics(mae=mae, mse=mse, rmse=rmse, r2=r2)

    @staticmethod
    def evaluate(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> RegressionMetrics:
        """Evaluate a model on test data.

        Args:
            model: Trained model with predict method
            X_test: Test features
            y_test: Ground truth test targets

        Returns:
            RegressionMetrics object with all metrics

        Raises:
            ValueError: If model doesn't have a predict method
        """
        if not hasattr(model, "predict"):
            raise ValueError("Model must have a predict method")

        y_pred = model.predict(X_test)
        return ModelEvaluator.calculate_metrics(y_test, y_pred)

    @staticmethod
    def get_best_model(
        metrics_dict: Dict[str, RegressionMetrics]
    ) -> Tuple[str, RegressionMetrics]:
        """Select the best model based on R² score.

        Args:
            metrics_dict: Dictionary mapping model names to their metrics

        Returns:
            Tuple of (best model name, best model metrics)
        """
        best_name = max(metrics_dict, key=lambda name: metrics_dict[name].r2)
        return best_name, metrics_dict[best_name]

    @staticmethod
    def print_comparison_table(metrics_dict: Dict[str, RegressionMetrics]) -> str:
        """Format a comparison table of model metrics as a string.

        Args:
            metrics_dict: Dictionary mapping model names to their metrics

        Returns:
            Formatted table string
        """
        header = f"{'Model':<25} {'MAE':<10} {'MSE':<10} {'RMSE':<10} {'R²':<10}"
        separator = "-" * len(header)
        lines = [header, separator]

        for name, metrics in metrics_dict.items():
            lines.append(
                f"{name:<25} {metrics.mae:<10.4f} {metrics.mse:<10.4f} "
                f"{metrics.rmse:<10.4f} {metrics.r2:<10.4f}"
            )

        return "\n".join(lines)