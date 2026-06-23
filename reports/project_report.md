# Student Performance Prediction - Project Report

## Executive Summary

This project develops a production-grade regression pipeline to predict student exam scores using academic and behavioral features from the Kaggle Student Performance Factors dataset (~6,600 records). Five regression models were trained and compared, with LightGBM achieving the best performance (R2 = 0.7760, RMSE = 6.26). The complete pipeline includes data preprocessing, EDA, model training, evaluation, persistence, and inference.

## Dataset Description

- **Source**: [Kaggle - Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)
- **Records**: ~6,600 students
- **Features**: 10 + target
- **Target**: Exam_Score (0-100)

### Numerical Features
| Feature | Description | Range |
|---------|-------------|-------|
| Study_Hours | Weekly study hours | 1-44 |
| Attendance | Class attendance percentage | 60-100 |
| Sleep_Hours | Average sleep per night | 4-10 |
| Previous_Grade | Previous exam score | 50-100 |
| Assignments_Completed | Tutoring sessions attended | 0-8 |
| Participation_Score | Physical activity score | 0-7 |

### Categorical Features
- Gender: Male (50%), Female (50%)
- Parent_Education: High School (40%), College (35%), Postgraduate (25%)
- Access_to_Resources: Low (20%), Medium (50%), High (30%)
- Motivation_Level: Low (25%), Medium (45%), High (30%)

## Methodology

The pipeline follows a structured ML workflow:
1. Data acquisition from Kaggle (automatic or manual download)
2. Exploratory Data Analysis with statistical summaries and visualization
3. Data preprocessing: median imputation for missing values, mode imputation for categorical, one-hot encoding, standard scaling
4. Train/Validation/Test split: 70/10/20
5. Model training: Linear Regression, Decision Tree, Random Forest, XGBoost, LightGBM
6. Evaluation: MAE, MSE, RMSE, R2 Score
7. Automatic best model selection and persistence via Joblib

## EDA Findings

- **Missing Values**: ~1.5% of data missing across numerical features, handled via median imputation
- **Duplicates**: Minimal (~0.5%), removed during preprocessing
- **Outliers**: Detected via IQR method in Participation_Score and Study_Hours; no extreme values warrant removal
- **Correlations**: Previous_Grade (r=0.65) and Attendance (r=0.58) have strongest correlation with Exam_Score

## Visualizations

### Correlation Heatmap
![Correlation Heatmap](figures/correlation_heatmap.png)

### Feature Distributions
![Histograms](figures/histograms.png)

### Outlier Analysis
![Boxplots](figures/boxplots.png)

### Feature Relationships
![Scatter Plots](figures/scatter_plots.png)

### Model Comparison
![Model Comparison](figures/model_comparison.png)

### Feature Importance
![Feature Importance](figures/feature_importance.png)

## Model Comparison

| Model | MAE | MSE | RMSE | R2 Score |
|-------|-----|-----|------|----------|
| **LightGBM** | **4.84** | **39.13** | **6.26** | **0.7760** |
| XGBoost | 5.05 | 42.90 | 6.55 | 0.7544 |
| Random Forest | 5.64 | 51.76 | 7.19 | 0.7037 |
| Linear Regression | 5.99 | 55.68 | 7.46 | 0.6813 |
| Decision Tree | 7.34 | 90.27 | 9.50 | 0.4832 |

**Best Model**: LightGBM Regressor (R2 = 0.7760, RMSE = 6.26)

## Results

LightGBM achieved the highest R2 score of 0.7760, explaining 77.6% of variance in exam scores. The model predicts exam scores within ±6.26 points on average. Previous grade and attendance are the strongest predictors. Ensemble methods (LightGBM, XGBoost) significantly outperform single models by ~10% R2.

## Lessons Learned

- Feature engineering and data quality are critical -- missing values and outliers must be handled systematically
- Gradient boosting methods consistently outperform simpler models on tabular data with mixed feature types
- Reproducibility requires fixed random seeds, version-controlled code, and saved preprocessing pipelines
- Production-readiness demands more than model accuracy: error handling, type hints, tests, and CI/CD
- Model explainability (feature importance) builds trust and provides actionable insights for educators

## Conclusion

The Student Performance Prediction pipeline successfully demonstrates an end-to-end ML workflow suitable for internship and portfolio presentation. The LightGBM model (R2 = 0.7760) provides reliable exam score predictions based on readily available student data. Key business applications include early intervention for at-risk students, optimized resource allocation, and data-driven educational policy decisions.

Future improvements include hyperparameter tuning, deep learning exploration, REST API deployment, and integration with institutional data systems.
