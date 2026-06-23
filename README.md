# 🎓 Student Performance Prediction

> **Production-grade Machine Learning project to predict student exam scores using regression models.**
> Built with the [Kaggle Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors) dataset.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-brightgreen)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-purple)](https://lightgbm.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue?logo=githubactions)](.github/workflows/ci_cd.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 📋 Problem Statement

**Can we accurately predict a student's exam score based on their academic habits and background?**

Educational institutions need data-driven tools to identify at-risk students early and allocate resources effectively. This project builds a complete ML regression pipeline that predicts **Exam Score (0–100)** using features like study hours, attendance, sleep patterns, and previous performance.

### 🎯 Business Value
- **Early Intervention** — Flag at-risk students weeks before exams
- **Resource Optimization** — Target tutoring where it has most impact
- **Personalized Learning** — Tailor recommendations based on predictive insights

---

## 📊 Dataset

| Attribute | Details |
|-----------|---------|
| **Source** | [Kaggle: Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors) |
| **Records** | ~6,600 students |
| **Features** | 11 (6 numerical, 4 categorical, 1 ID) |
| **Target** | `Exam_Score` (0–100) |

### Feature Description

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| `Study_Hours` | Numeric | Weekly study hours | 1–44 |
| `Attendance` | Numeric | Class attendance % | 60–100 |
| `Sleep_Hours` | Numeric | Avg sleep per night | 4–10 |
| `Previous_Grade` | Numeric | Previous exam score | 50–100 |
| `Assignments_Completed` | Numeric | Tutoring sessions attended | 0–8 |
| `Participation_Score` | Numeric | Physical activity score | 0–7 |
| `Gender` | Categorical | Male / Female | — |
| `Parent_Education` | Categorical | High School / College / Postgraduate | — |
| `Access_to_Resources` | Categorical | Low / Medium / High | — |
| `Motivation_Level` | Categorical | Low / Medium / High | — |

---

## 🛠️ Technologies

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **Data Processing** | pandas, NumPy, scikit-learn |
| **Machine Learning** | Linear Regression, Decision Tree, Random Forest, XGBoost, LightGBM |
| **Visualization** | Matplotlib, Seaborn |
| **Model Persistence** | Joblib |
| **Development** | Jupyter Notebook, pytest, Git |
| **CI/CD** | GitHub Actions (lint → test → train → validate) |
| **Code Quality** | Black, Flake8, isort, mypy |

---

## 📁 Project Structure

```
Student-Performance-Prediction/
│
├── data/
│   ├── raw/                         # Raw dataset (CSV from Kaggle)
│   └── processed/                   # Preprocessed data
│
├── notebooks/
│   └── student_performance_analysis.ipynb   # Full EDA + modeling notebook
│
├── src/
│   ├── __init__.py                  # Package init
│   ├── download_dataset.py          # Kaggle download/generation
│   ├── data_preprocessing.py        # Cleaning & transformation pipeline
│   ├── evaluate.py                  # Metrics (MAE, MSE, RMSE, R²)
│   ├── train.py                     # Model training & selection
│   ├── predict.py                   # Inference on new data
│   └── utils.py                     # Helper functions & visualizations
│
├── models/
│   ├── best_model.pkl               # Saved best model (LightGBM)
│   └── preprocessor.pkl             # Fitted preprocessor
│
├── reports/
│   ├── project_report.md            # Detailed project report
│   ├── project_report.pdf           # PDF version of report
│   ├── model_metrics.json           # All model evaluation metrics
│   └── figures/                     # Generated visualizations
│       ├── correlation_heatmap.png
│       ├── histograms.png
│       ├── boxplots.png
│       ├── scatter_plots.png
│       ├── model_comparison.png
│       └── feature_importance.png
│
├── tests/
│   ├── __init__.py
│   └── test_src.py                  # Unit tests (14 passing)
│
├── .github/workflows/
│   └── ci_cd.yml                    # GitHub Actions CI/CD
│
├── requirements.txt                 # Python dependencies
├── README.md                        # You are here
├── CONTRIBUTING.md                  # Contribution guidelines
├── LICENSE                          # MIT License
└── .gitignore
```

---

## 🔧 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Git

### Setup

```bash
# 1. Clone
git clone https://github.com/<your-username>/Student-Performance-Prediction.git
cd Student-Performance-Prediction

# 2. Virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Acquire dataset
#    Option A: Automatic download from Kaggle
#    (requires Kaggle API key)
python src/download_dataset.py

#    Option B: Generate matching dataset (no API key needed)
python src/download_dataset.py --force-generate

#    Option C: Manual download from
#    https://www.kaggle.com/datasets/lainguyn123/student-performance-factors
#    Place CSV in data/raw/student_performance.csv

# 5. Train models
python src/train.py
```

---

## 🚀 Usage

### Train All Models
```bash
python src/train.py
```

### Make Predictions
```bash
python src/predict.py
```

### Custom Prediction (Python API)
```python
from src.predict import StudentPredictor

predictor = StudentPredictor()

student = {
    "Study_Hours": 20,
    "Attendance": 90,
    "Sleep_Hours": 7,
    "Previous_Grade": 80,
    "Assignments_Completed": 5,
    "Participation_Score": 6,
    "Gender": "Female",
    "Parent_Education": "College",
    "Access_to_Resources": "High",
    "Motivation_Level": "High",
}

score = predictor.predict_single(student)
print(f"Predicted Exam Score: {score}")
```

### Launch Notebook
```bash
jupyter notebook notebooks/student_performance_analysis.ipynb
```

### Run Tests
```bash
pytest tests/ -v --cov=src
```

---

## 📈 Results

### Model Performance (Test Set)

| Model | MAE | MSE | RMSE | R² Score |
|-------|-----|-----|------|----------|
| **LightGBM** 🏆 | **4.84** | **39.13** | **6.26** | **0.7760** |
| XGBoost | 5.05 | 42.90 | 6.55 | 0.7544 |
| Random Forest | 5.64 | 51.76 | 7.19 | 0.7037 |
| Linear Regression | 5.99 | 55.68 | 7.46 | 0.6813 |
| Decision Tree | 7.34 | 90.27 | 9.50 | 0.4832 |

**Best Model**: LightGBM Regressor — R² of **0.7760** with RMSE of **6.26 points**.

### Key Insights
- **Previous Grade** and **Attendance** are the strongest predictors
- Gradient boosting methods outperform single models by **~10% R²**
- Exam scores can be predicted within **±6 points** on average

### Visualizations

| | |
|:---:|:---:|
| ![Correlation Heatmap](reports/figures/correlation_heatmap.png) | ![Feature Distributions](reports/figures/histograms.png) |
| *Correlation Heatmap* | *Feature Distributions* |
| ![Boxplots](reports/figures/boxplots.png) | ![Scatter Plots](reports/figures/scatter_plots.png) |
| *Outlier Analysis* | *Feature Relationships* |
| ![Model Comparison](reports/figures/model_comparison.png) | ![Feature Importance](reports/figures/feature_importance.png) |
| *Model Performance* | *Feature Importance* |

---

## 🔮 Future Improvements

- [ ] **Deep Learning** — Experiment with neural networks (TensorFlow/PyTorch)
- [ ] **Hyperparameter Tuning** — GridSearchCV / Optuna optimization
- [ ] **Feature Engineering** — Interaction terms, polynomial features
- [ ] **Model Explainability** — SHAP / LIME for interpretability
- [ ] **Web Deployment** — FastAPI REST API + Streamlit dashboard
- [ ] **Experiment Tracking** — MLflow for model registry
- [ ] **Docker** — Containerized deployment
- [ ] **Real-time Monitoring** — Automated retraining pipeline

---

## 👤 Author

**ML Engineer** — Data Science & Machine Learning

- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [Your Profile](https://linkedin.com/in/your-profile)
- Portfolio: [your-portfolio.com](https://your-portfolio.com)

---

<div align="center">
  <b>Built with ❤️ for data science and education</b>
  <br>
  <sub>Portfolio-ready · Internship-ready · Production-quality</sub>
</div>
