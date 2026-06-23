"""Data preprocessing pipeline for Student Performance Prediction."""

from typing import Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path


class DataPreprocessor:
    """Handles data loading, cleaning, preprocessing, and splitting."""

    def __init__(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        target_column: str = "Exam_Score",
    ):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.target_column = target_column
        self.numeric_features = [
            "Study_Hours",
            "Attendance",
            "Sleep_Hours",
            "Previous_Grade",
            "Assignments_Completed",
            "Participation_Score",
        ]
        self.categorical_features = [
            "Gender",
            "Parent_Education",
            "Access_to_Resources",
            "Motivation_Level",
        ]
        self._drop_columns = ["Student_ID"]
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_names: Optional[list] = None

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load dataset from CSV file.

        Args:
            filepath: Path to the CSV file

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Dataset not found at {filepath}")

        df = pd.read_csv(filepath)
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataset by handling missing values and removing duplicates.

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        df = df.drop(columns=[c for c in self._drop_columns if c in df.columns], errors="ignore")

        df = df.drop_duplicates(keep="first").reset_index(drop=True)

        self._impute_missing_values(df)

        return df

    def _impute_missing_values(self, df: pd.DataFrame) -> None:
        """Impute missing values in-place using median for numeric and mode for categorical.

        Args:
            df: DataFrame to impute (modified in-place)
        """
        for col in self.numeric_features:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        for col in self.categorical_features:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode().iloc[0])

    def _build_preprocessor(self) -> ColumnTransformer:
        """Build a ColumnTransformer for numeric scaling and categorical encoding.

        Returns:
            Configured ColumnTransformer
        """
        numeric_transformer = Pipeline(steps=[
            ("scaler", StandardScaler()),
        ])

        categorical_transformer = Pipeline(steps=[
            ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_features),
                ("cat", categorical_transformer, self.categorical_features),
            ],
            remainder="passthrough",
        )

        return preprocessor

    def _get_feature_names(self, df: pd.DataFrame) -> list:
        """Get feature names after transformation.

        Args:
            df: Input DataFrame (used to infer categorical levels)

        Returns:
            List of transformed feature names
        """
        names = list(self.numeric_features)

        for cat_col in self.categorical_features:
            categories = df[cat_col].dropna().unique()
            for category in sorted(categories):
                if category != sorted(categories)[0]:
                    names.append(f"{cat_col}_{category}")

        return names

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train, validation, and test sets.

        Args:
            df: Preprocessed DataFrame

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        y = df[self.target_column].values
        X = df.drop(columns=[self.target_column])

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        val_ratio = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=self.random_state
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def fit_transform(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit preprocessor on training data and transform all splits.

        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features

        Returns:
            Tuple of transformed (X_train, X_val, X_test) as numpy arrays
        """
        self.preprocessor = self._build_preprocessor()

        X_train_transformed = self.preprocessor.fit_transform(X_train)
        X_val_transformed = self.preprocessor.transform(X_val)
        X_test_transformed = self.preprocessor.transform(X_test)

        self.feature_names = self._get_feature_names(
            pd.concat([X_train, X_val, X_test], axis=0)
        )

        return X_train_transformed, X_val_transformed, X_test_transformed

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted preprocessor.

        Args:
            X: Feature DataFrame

        Returns:
            Transformed feature array
        """
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")

        return self.preprocessor.transform(X)

    def preprocess(
        self, filepath: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """End-to-end preprocessing: load, clean, split, transform.

        Args:
            filepath: Path to raw CSV dataset

        Returns:
            Tuple of (X_train_t, X_val_t, X_test_t, y_train, y_val, y_test, cleaned_df)
        """
        df = self.load_data(filepath)
        df_cleaned = self.clean_data(df)
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(df_cleaned)
        X_train_t, X_val_t, X_test_t = self.fit_transform(X_train, X_val, X_test)

        return X_train_t, X_val_t, X_test_t, y_train, y_val, y_test, df_cleaned