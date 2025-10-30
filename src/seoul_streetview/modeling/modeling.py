"""
Modeling Module

This module provides functionality for predicting Urban Vitality Index (UVI)
using machine learning models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, Tuple, Optional, List
import joblib


class UVIPredictor:
    """Class for predicting Urban Vitality Index using ML models."""

    def __init__(self):
        """Initialize the UVIPredictor."""
        self.models = {
            'Decision Tree': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(),
            'Gradient Boosting': GradientBoostingRegressor(),
            'Support Vector Machine': SVR(),
            'K-Nearest Neighbors': KNeighborsRegressor()
        }

        self.param_grids = {
            'Decision Tree': {'max_depth': list(range(1, 11))},
            'Random Forest': {'n_estimators': list(range(100, 501, 100))},
            'Gradient Boosting': {'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.09, 0.1]},
            'Support Vector Machine': {'kernel': ['linear', 'poly', 'rbf']},
            'K-Nearest Neighbors': {'n_neighbors': list(range(10, 101, 10))}
        }

        self.best_models = {}
        self.results = {}

    def prepare_data(
        self,
        segmentation_csv: str,
        uvi_excel: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for modeling.

        Args:
            segmentation_csv: Path to segmentation results CSV
            uvi_excel: Path to UVI Excel file
            test_size: Proportion of test set (default: 0.2)
            random_state: Random state for reproducibility (default: 42)

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Load segmentation data
        prop = pd.read_csv(segmentation_csv)

        # Load UVI data
        uvi = pd.read_excel(uvi_excel, header=None)

        # Combine data
        uvprop = pd.concat([prop, uvi], axis=1)
        uvprop.rename(columns={0: 'UVI'}, inplace=True)

        # Remove any rows with NaN values
        uvprop = uvprop.dropna()

        # Reset index after dropping rows
        uvprop = uvprop.reset_index(drop=True)

        print(f"Loaded {len(uvprop)} samples after removing NaN values")

        # Prepare features and target
        X = uvprop.drop(['filename', 'UVI'], axis=1, errors='ignore')
        y = uvprop['UVI']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        return X_train, X_test, y_train, y_test

    def train_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_folds: int = 5
    ) -> None:
        """
        Train all models using grid search.

        Args:
            X_train: Training features
            y_train: Training targets
            cv_folds: Number of cross-validation folds (default: 5)
        """
        print("Training models...")

        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")

            grid_search = GridSearchCV(
                model,
                self.param_grids[model_name],
                scoring='r2',
                cv=cv_folds,
                n_jobs=-1
            )

            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            self.best_models[model_name] = best_model

            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")

    def evaluate_models(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict:
        """
        Evaluate all trained models.

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets

        Returns:
            Dictionary with evaluation results
        """
        print("\nEvaluating models...")

        for model_name, model in self.best_models.items():
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)

            self.results[model_name] = {
                'Train R2': train_r2,
                'Test R2': test_r2,
                'Train RMSE': train_rmse,
                'Test RMSE': test_rmse,
                'Train MAE': train_mae,
                'Test MAE': test_mae
            }

        return self.results

    def print_results(self) -> None:
        """Print evaluation results in a formatted table."""
        print("\n" + "=" * 80)
        print("MODEL PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"{'Model':<25} {'Train R2':<12} {'Test R2':<12} {'Test RMSE':<12} {'Test MAE':<12}")
        print("-" * 80)

        for model_name, metrics in self.results.items():
            print(f"{model_name:<25} "
                  f"{metrics['Train R2']:<12.4f} "
                  f"{metrics['Test R2']:<12.4f} "
                  f"{metrics['Test RMSE']:<12.4f} "
                  f"{metrics['Test MAE']:<12.4f}")

        print("=" * 80)

    def plot_feature_importance(
        self,
        feature_names: List[str],
        model_name: str = 'Gradient Boosting',
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot feature importance for tree-based models.

        Args:
            feature_names: List of feature names
            model_name: Name of the model (default: 'Gradient Boosting')
            save_path: Path to save the plot (optional)
        """
        if model_name not in self.best_models:
            print(f"Model '{model_name}' not found.")
            return

        model = self.best_models[model_name]

        if not hasattr(model, 'feature_importances_'):
            print(f"Model '{model_name}' does not have feature importances.")
            return

        importance = model.feature_importances_

        plt.figure(figsize=(10, 6))
        plt.bar(feature_names, importance)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title(f'Feature Importance - {model_name}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def predict(
        self,
        X: pd.DataFrame,
        model_name: str = 'Gradient Boosting'
    ) -> np.ndarray:
        """
        Make predictions using a trained model.

        Args:
            X: Features to predict
            model_name: Name of the model to use (default: 'Gradient Boosting')

        Returns:
            Array of predictions
        """
        if model_name not in self.best_models:
            raise ValueError(f"Model '{model_name}' not found. Train models first.")

        return self.best_models[model_name].predict(X)

    def save_model(
        self,
        model_name: str,
        file_path: str
    ) -> None:
        """
        Save a trained model to disk.

        Args:
            model_name: Name of the model to save
            file_path: Path to save the model
        """
        if model_name not in self.best_models:
            raise ValueError(f"Model '{model_name}' not found.")

        joblib.dump(self.best_models[model_name], file_path)
        print(f"Model saved to {file_path}")

    def load_model(
        self,
        model_name: str,
        file_path: str
    ) -> None:
        """
        Load a trained model from disk.

        Args:
            model_name: Name to assign to the loaded model
            file_path: Path to the saved model
        """
        self.best_models[model_name] = joblib.load(file_path)
        print(f"Model loaded from {file_path}")


def main():
    """Main function to demonstrate usage."""
    predictor = UVIPredictor()

    # Check if data files exist
    import os

    segmentation_file = "class_percentages.csv"
    uvi_file = "Urban_vitality_index.xlsx"

    if not os.path.exists(segmentation_file) or not os.path.exists(uvi_file):
        print("Data files not found. Please ensure you have:")
        print(f"  - {segmentation_file}")
        print(f"  - {uvi_file}")
        return

    # Prepare data
    X_train, X_test, y_train, y_test = predictor.prepare_data(
        segmentation_csv=segmentation_file,
        uvi_excel=uvi_file
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Features: {list(X_train.columns)}")

    # Train models
    predictor.train_models(X_train, y_train)

    # Evaluate models
    predictor.evaluate_models(X_train, X_test, y_train, y_test)

    # Print results
    predictor.print_results()

    # Plot feature importance
    predictor.plot_feature_importance(
        feature_names=list(X_train.columns),
        save_path="data/processed/feature_importance.png"
    )


if __name__ == "__main__":
    main()
