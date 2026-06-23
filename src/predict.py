"""Inference script for Student Performance Prediction."""

import sys
from typing import Dict, Union, Optional

import numpy as np
import pandas as pd

from src.utils import load_model, logger
from src.data_preprocessing import DataPreprocessor


class StudentPredictor:
    """Handles prediction of student exam scores using trained model."""

    def __init__(
        self,
        model_path: str = "models/best_model.pkl",
        preprocessor_path: str = "models/preprocessor.pkl",
    ):
        self.model = load_model(model_path)
        self.preprocessor = load_model(preprocessor_path)
        self._feature_spec = DataPreprocessor()

    def _validate_input(self, data: Dict[str, Union[float, int, str]]) -> pd.DataFrame:
        """Validate and convert input dictionary to DataFrame.

        Args:
            data: Dictionary of student features

        Returns:
            DataFrame with validated features

        Raises:
            ValueError: If required features are missing or invalid
        """
        required_features = {
            "Study_Hours": (float, int),
            "Attendance": (float, int),
            "Sleep_Hours": (float, int),
            "Previous_Grade": (float, int),
            "Assignments_Completed": (float, int),
            "Participation_Score": (float, int),
        }

        optional_features = {
            "Gender": str,
            "Parent_Education": str,
            "Access_to_Resources": str,
            "Motivation_Level": str,
        }

        for feature, types in required_features.items():
            if feature not in data:
                raise ValueError(f"Missing required feature: {feature}")
            if not isinstance(data[feature], types):
                raise TypeError(
                    f"Feature '{feature}' must be one of {types}, got {type(data[feature]).__name__}"
                )

        for feature in required_features:
            if data[feature] < 0:
                raise ValueError(f"Feature '{feature}' must be non-negative")

        row = {}
        for feat in required_features:
            row[feat] = data[feat]

        for feat in optional_features:
            row[feat] = data.get(feat, "Unknown")

        return pd.DataFrame([row])

    def predict_single(self, student_data: Dict[str, Union[float, int, str]]) -> float:
        """Predict exam score for a single student.

        Args:
            student_data: Dictionary of student features

        Returns:
            Predicted exam score
        """
        df = self._validate_input(student_data)

        df = self._feature_spec.clean_data(df)

        X_transformed = self.preprocessor.transform(df)
        prediction = float(self.model.predict(X_transformed)[0])
        prediction = max(0.0, min(100.0, prediction))

        return round(prediction, 2)

    def predict_batch(self, students_data: list) -> np.ndarray:
        """Predict exam scores for multiple students.

        Args:
            students_data: List of student data dictionaries

        Returns:
            Array of predicted exam scores
        """
        predictions = [self.predict_single(data) for data in students_data]
        return np.array(predictions)


def get_sample_student_data() -> list:
    """Return a list of sample student data for testing.

    Returns:
        List of dictionaries with sample student features
    """
    return [
        {
            "Study_Hours": 15.0,
            "Attendance": 95.0,
            "Sleep_Hours": 7.5,
            "Previous_Grade": 85.0,
            "Assignments_Completed": 18,
            "Participation_Score": 90.0,
            "Gender": "Female",
            "Parent_Education": "College",
            "Access_to_Resources": "High",
            "Motivation_Level": "High",
        },
        {
            "Study_Hours": 5.0,
            "Attendance": 60.0,
            "Sleep_Hours": 6.0,
            "Previous_Grade": 55.0,
            "Assignments_Completed": 6,
            "Participation_Score": 40.0,
            "Gender": "Male",
            "Parent_Education": "High School",
            "Access_to_Resources": "Low",
            "Motivation_Level": "Low",
        },
        {
            "Study_Hours": 10.0,
            "Attendance": 80.0,
            "Sleep_Hours": 8.0,
            "Previous_Grade": 70.0,
            "Assignments_Completed": 12,
            "Participation_Score": 65.0,
            "Gender": "Male",
            "Parent_Education": "College",
            "Access_to_Resources": "Medium",
            "Motivation_Level": "Medium",
        },
    ]


def main():
    """Run a sample prediction to demonstrate the predict module."""
    predictor = StudentPredictor()
    sample_students = get_sample_student_data()

    print("\n" + "=" * 65)
    print("STUDENT PERFORMANCE PREDICTIONS".center(65))
    print("=" * 65)

    for i, student in enumerate(sample_students, 1):
        try:
            score = predictor.predict_single(student)
            print(f"\nStudent {i}:")
            for key, val in student.items():
                print(f"  {key}: {val}")
            print(f"  >>> Predicted Exam Score: {score:.2f}")
        except (ValueError, TypeError) as e:
            print(f"\nStudent {i}: ERROR - {e}")

    print("\n" + "=" * 65)

    try:
        predictor.predict_single({})
    except ValueError as e:
        print(f"\nGraceful error handling: {e}")


if __name__ == "__main__":
    main()