"""Utility functions for the Student Performance Prediction project."""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_directories() -> None:
    """Create all necessary project directories if they don't exist."""
    dirs = [
        "data/raw",
        "data/processed",
        "models",
        "reports/figures",
        "notebooks",
        "screenshots",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    logger.info("Project directories verified/created.")


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except (ImportError, ModuleNotFoundError):
        pass
    logger.info(f"Random seed set to {seed}")


def save_model(model, filepath: str) -> None:
    """Save trained model using joblib.

    Args:
        model: Trained model object
        filepath: Path to save the model
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")


def load_model(filepath: str):
    """Load a saved model from disk.

    Args:
        filepath: Path to the saved model file

    Returns:
        Loaded model object
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    model = joblib.load(filepath)
    logger.info(f"Model loaded from {filepath}")
    return model


def save_metrics(metrics: Dict[str, float], filepath: str) -> None:
    """Save evaluation metrics to a JSON file.

    Args:
        metrics: Dictionary of metric names to values
        filepath: Path to save the metrics JSON file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {filepath}")


def load_metrics(filepath: str) -> Dict[str, float]:
    """Load evaluation metrics from a JSON file.

    Args:
        filepath: Path to the metrics JSON file

    Returns:
        Dictionary of metric names to values
    """
    with open(filepath, "r") as f:
        metrics = json.load(f)
    return metrics


def save_figure(fig, filename: str, dpi: int = 150) -> None:
    """Save matplotlib figure to reports/figures directory.

    Args:
        fig: Matplotlib figure object
        filename: Output filename (should include extension)
        dpi: Resolution in dots per inch
    """
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved to {filepath}")


def create_correlation_heatmap(df: pd.DataFrame, numeric_cols: list) -> None:
    """Create and save correlation heatmap.

    Args:
        df: DataFrame containing numeric columns
        numeric_cols: List of numeric column names
    """
    corr_matrix = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_figure(fig, "correlation_heatmap.png", dpi=200)


def create_histograms(df: pd.DataFrame, numeric_cols: list) -> None:
    """Create and save histogram grid for numeric features.

    Args:
        df: DataFrame containing numeric columns
        numeric_cols: List of numeric column names
    """
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, bins=30, ax=axes[i])
        axes[i].set_title(f"Distribution of {col}", fontsize=12, fontweight="bold")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_figure(fig, "histograms.png", dpi=200)


def create_boxplots(df: pd.DataFrame, numeric_cols: list) -> None:
    """Create and save boxplot grid for outlier analysis.

    Args:
        df: DataFrame containing numeric columns
        numeric_cols: List of numeric column names
    """
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.boxplot(x=df[col].dropna(), ax=axes[i])
        axes[i].set_title(f"Boxplot of {col}", fontsize=12, fontweight="bold")
        axes[i].set_xlabel(col)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Outlier Analysis - Boxplots", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_figure(fig, "boxplots.png", dpi=200)


def create_scatter_plots(df: pd.DataFrame, features: list, target: str) -> None:
    """Create and save scatter plots of features vs target.

    Args:
        df: DataFrame containing the data
        features: List of feature column names
        target: Target column name
    """
    n_cols = 3
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        axes[i].scatter(df[feat], df[target], alpha=0.5, s=30)
        axes[i].set_xlabel(feat, fontsize=10)
        axes[i].set_ylabel(target, fontsize=10)
        axes[i].set_title(f"{feat} vs {target}", fontsize=12, fontweight="bold")

        z = np.polyfit(df[feat].dropna(), df[target].dropna(), 1)
        p = np.poly1d(z)
        x_sorted = np.sort(df[feat].dropna())
        axes[i].plot(x_sorted, p(x_sorted), "r--", alpha=0.8, linewidth=2)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Relationships with Target", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_figure(fig, "scatter_plots.png", dpi=200)


def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform data validation checks.

    Args:
        df: Input DataFrame to validate

    Returns:
        Dictionary containing validation results
    """
    results = {}

    results["shape"] = df.shape
    results["total_rows"] = len(df)
    results["total_columns"] = len(df.columns)
    results["memory_usage_mb"] = df.memory_usage(deep=True).sum() / (1024 * 1024)

    results["missing_values"] = df.isnull().sum().to_dict()
    results["total_missing"] = int(df.isnull().sum().sum())
    results["missing_percentage"] = round(
        (results["total_missing"] / (results["total_rows"] * results["total_columns"])) * 100, 2
    )

    results["duplicate_rows"] = int(df.duplicated().sum())
    results["duplicate_percentage"] = round(
        (results["duplicate_rows"] / results["total_rows"]) * 100, 2
    )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outliers = {}
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((df[col] < lower) | (df[col] > upper)).sum())
        outliers[col] = {
            "count": outlier_count,
            "percentage": round((outlier_count / len(df)) * 100, 2),
            "lower_bound": round(lower, 2),
            "upper_bound": round(upper, 2),
        }
    results["outliers"] = outliers

    return results


def print_validation_report(results: Dict[str, Any]) -> None:
    """Pretty-print data validation report.

    Args:
        results: Dictionary from validate_data()
    """
    print("\n" + "=" * 60)
    print("DATA VALIDATION REPORT".center(60))
    print("=" * 60)

    print(f"\nDataset Shape: {results['shape']}")
    print(f"Memory Usage: {results['memory_usage_mb']:.2f} MB")

    print(f"\n--- Missing Values ---")
    print(f"Total Missing: {results['total_missing']} ({results['missing_percentage']}%)")
    missing_cols = {k: v for k, v in results["missing_values"].items() if v > 0}
    if missing_cols:
        for col, count in missing_cols.items():
            print(f"  {col}: {count} missing")

    print(f"\n--- Duplicate Rows ---")
    print(f"Total Duplicates: {results['duplicate_rows']} ({results['duplicate_percentage']}%)")

    print(f"\n--- Outliers (IQR method) ---")
    for col, stats in results["outliers"].items():
        if stats["count"] > 0:
            print(f"  {col}: {stats['count']} outliers ({stats['percentage']}%)")
    print("=" * 60 + "\n")