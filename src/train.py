"""Model training and comparison pipeline for Student Performance Prediction."""

import warnings
from typing import Dict, Any, Tuple
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from src.data_preprocessing import DataPreprocessor
from src.evaluate import ModelEvaluator, RegressionMetrics
from src.utils import set_seed, logger, save_model

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not installed. Skipping XGBoost model.")

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    logger.warning("LightGBM not installed. Skipping LightGBM model.")


def get_models() -> Dict[str, Any]:
    """Initialize all regression models with hyperparameters.

    Returns:
        Dictionary mapping model names to model instances
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
    }

    if XGB_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    if LGBM_AVAILABLE:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    return models


def train_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Any:
    """Train a single model on given data.

    Args:
        model: Unfitted model instance
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target

    Returns:
        Fitted model
    """
    logger.info(f"Training {model.__class__.__name__}...")
    model.fit(X_train, y_train)
    return model


def train_and_evaluate_all(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Dict[str, RegressionMetrics], str, Any]:
    """Train all models, evaluate them, and return results.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        X_test: Test features
        y_test: Test target

    Returns:
        Tuple containing:
        - Dictionary mapping model name to evaluation metrics
        - Name of the best performing model
        - The best performing model instance
    """
    models = get_models()
    evaluator = ModelEvaluator()

    all_metrics: Dict[str, RegressionMetrics] = {}

    for name, model in models.items():
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Training: {name}")
        logger.info(f"{'=' * 40}")

        trained_model = train_model(model, X_train, y_train, X_val, y_val)

        metrics = evaluator.evaluate(trained_model, X_test, y_test)
        all_metrics[name] = metrics

        logger.info(f"{name} - Test Set Performance:")
        for metric_name, value in metrics.model_dump().items():
            logger.info(f"  {metric_name.upper()}: {value:.4f}")

    best_model_name, best_metrics = evaluator.get_best_model(all_metrics)
    logger.info(f"\n{'=' * 40}")
    logger.info(f"BEST MODEL: {best_model_name}")
    logger.info(f"{'=' * 40}")
    for metric_name, value in best_metrics.model_dump().items():
        logger.info(f"  {metric_name.upper()}: {value:.4f}")

    return all_metrics, best_model_name, models[best_model_name]


def main():
    """Run the full training pipeline."""
    set_seed(42)

    import subprocess
    import sys

    dataset_path = "data/raw/student_performance.csv"
    if not Path(dataset_path).exists():
        logger.info("Dataset not found. Running download_dataset.py...")
        subprocess.run([sys.executable, "src/download_dataset.py"], check=True)

    preprocessor = DataPreprocessor(
        test_size=0.2,
        val_size=0.1,
        target_column="Exam_Score",
    )

    (X_train, X_val, X_test,
     y_train, y_val, y_test, df_cleaned) = preprocessor.preprocess(dataset_path)

    logger.info(f"Training set size: {X_train.shape[0]}")
    logger.info(f"Validation set size: {X_val.shape[0]}")
    logger.info(f"Test set size: {X_test.shape[0]}")

    all_metrics, best_model_name, best_model = train_and_evaluate_all(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    save_model(best_model, "models/best_model.pkl")
    save_model(preprocessor.preprocessor, "models/preprocessor.pkl")

    metrics_dir = Path("reports")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    import json
    metrics_json = {}
    for name, m in all_metrics.items():
        metrics_json[name] = m.model_dump()
    with open(metrics_dir / "model_metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=4)

    logger.info(f"\nBest model '{best_model_name}' saved to models/best_model.pkl")
    logger.info("Preprocessor saved to models/preprocessor.pkl")
    logger.info("Metrics saved to reports/model_metrics.json")


if __name__ == "__main__":
    main()