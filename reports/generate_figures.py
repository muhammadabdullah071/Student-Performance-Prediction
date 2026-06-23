"""Generate all visualizations for the project report."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', palette='muted')

OUTPUT = Path("reports/figures")
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/raw/student_performance.csv")

NUM_COLS = ['Study_Hours', 'Attendance', 'Sleep_Hours', 'Previous_Grade',
            'Assignments_Completed', 'Participation_Score', 'Exam_Score']
FEATURES = ['Study_Hours', 'Attendance', 'Sleep_Hours', 'Previous_Grade',
            'Assignments_Completed', 'Participation_Score']

def save(fig, name):
    fig.savefig(OUTPUT / name, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  {name}")

print("Generating figures...")

corr = df[NUM_COLS].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax)
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
save(fig, 'correlation_heatmap.png')

target_corr = corr['Exam_Score'].drop('Exam_Score').sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in target_corr.values]
ax.barh(target_corr.index, target_corr.values, color=colors, edgecolor='black')
for i, v in enumerate(target_corr.values):
    ax.text(v + 0.01 if v > 0 else v - 0.06, i, f'{v:.3f}', va='center', fontweight='bold')
ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_title('Feature Correlation with Exam Score', fontweight='bold')
plt.tight_layout()
save(fig, 'target_correlation.png')

fig, axes = plt.subplots(3, 3, figsize=(14, 11))
for i, col in enumerate(NUM_COLS):
    sns.histplot(df[col].dropna(), kde=True, bins=35, color='steelblue', ax=axes[i//3][i%3])
    axes[i//3][i%3].set_title(col, fontweight='bold')
for j in range(i+1, 9):
    axes[j//3][j%3].set_visible(False)
plt.suptitle('Feature Distributions with KDE', fontsize=14, fontweight='bold')
plt.tight_layout()
save(fig, 'histograms.png')

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for i, col in enumerate(NUM_COLS):
    sns.boxplot(x=df[col].dropna(), color='lightcoral', ax=axes[i//4][i%4])
    axes[i//4][i%4].set_title(col, fontweight='bold')
axes[1][3].set_visible(False)
plt.suptitle('Outlier Analysis - Boxplots', fontweight='bold')
plt.tight_layout()
save(fig, 'boxplots.png')

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, feat in enumerate(FEATURES):
    ax = axes[i//3][i%3]
    valid = df[[feat, 'Exam_Score']].dropna()
    ax.scatter(valid[feat], valid['Exam_Score'], alpha=0.3, s=10, c='steelblue')
    z = np.polyfit(valid[feat], valid['Exam_Score'], 1)
    x_s = np.sort(valid[feat])
    ax.plot(x_s, np.poly1d(z)(x_s), 'r--', lw=2)
    corr_val = valid[feat].corr(valid['Exam_Score'])
    ax.text(0.05, 0.95, f'r = {corr_val:.3f}', transform=ax.transAxes, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_xlabel(feat); ax.set_ylabel('Exam Score')
    ax.set_title(f'{feat} vs Exam Score', fontweight='bold')
plt.suptitle('Feature Relationships with Target Variable', fontsize=14, fontweight='bold')
plt.tight_layout()
save(fig, 'scatter_plots.png')

import json
metrics_path = Path("reports/model_metrics.json")
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics_data = json.load(f)

    rf = pd.DataFrame([
        {"Model": k, "RMSE": v["rmse"], "R2": v["r2"]}
        for k, v in metrics_data.items()
    ]).sort_values("R2", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    best_rmse = rf["RMSE"].min()
    colors = ["#e74c3c" if v == best_rmse else "#3498db" for v in rf["RMSE"]]
    axes[0].barh(rf["Model"], rf["RMSE"], color=colors, edgecolor="black")
    for _, row in rf.iterrows():
        axes[0].text(row["RMSE"] + 0.1, list(rf["Model"]).index(row["Model"]),
                     f"{row['RMSE']:.2f}", va="center", fontweight="bold")
    axes[0].set_xlabel("RMSE (lower is better)")
    axes[0].set_title("RMSE Comparison", fontweight="bold")

    best_r2 = rf["R2"].max()
    colors = ["#2ecc71" if v == best_r2 else "#3498db" for v in rf["R2"]]
    axes[1].barh(rf["Model"], rf["R2"], color=colors, edgecolor="black")
    for _, row in rf.iterrows():
        axes[1].text(row["R2"] + 0.005, list(rf["Model"]).index(row["Model"]),
                     f"{row['R2']:.4f}", va="center", fontweight="bold")
    axes[1].set_xlabel("R2 Score (higher is better)")
    axes[1].set_title("R2 Score Comparison", fontweight="bold")
    plt.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save(fig, "model_comparison.png")

    from sklearn.ensemble import RandomForestRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    best_name = max(metrics_data, key=lambda n: metrics_data[n]["r2"])
    if best_name in ["Random Forest", "XGBoost", "LightGBM"]:
        import joblib
        model = joblib.load("models/best_model.pkl")
        if hasattr(model, "feature_importances_"):
            from sklearn.preprocessing import OneHotEncoder
            preproc = joblib.load("models/preprocessor.pkl")
            cat_names = preproc.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(
                ["Gender", "Parent_Education", "Access_to_Resources", "Motivation_Level"]
            )
            all_names = ["Study_Hours", "Attendance", "Sleep_Hours", "Previous_Grade",
                         "Assignments_Completed", "Participation_Score"] + list(cat_names)
            imp = pd.DataFrame({"Feature": all_names, "Importance": model.feature_importances_})\
                .sort_values("Importance", ascending=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(imp["Feature"], imp["Importance"], color=sns.color_palette("viridis", len(imp)))
            ax.set_title(f"Feature Importance ({best_name})", fontweight="bold")
            plt.tight_layout()
            save(fig, "feature_importance.png")

print("\nAll figures generated successfully.")