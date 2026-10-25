"""
Automated Unit Tests for AutoML Studio ML Engine
Tests classification, regression, preprocessing pipelines, hyperparameter tuning,
Joblib persistence, and prediction inference.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes
from ml_engine import DatasetManager, MLPipelineManager


class TestAutoMLPipeline(unittest.TestCase):

    def setUp(self):
        self.dataset_mgr = DatasetManager()
        self.pipeline_mgr = MLPipelineManager()

    def test_dataset_manager_summary(self):
        """Test dataset loading and summary generation."""
        iris = load_iris(as_frame=True)
        df = iris.frame
        df["species"] = iris.target_names[iris.target]
        df.drop(columns=["target"], inplace=True)

        summary = self.dataset_mgr.load_dataframe(df, "iris_test.csv")
        self.assertEqual(summary["total_rows_cleaned"], 149) # 1 duplicate row in standard iris
        self.assertEqual(summary["total_columns"], 5)
        self.assertIn("sepal length (cm)", summary["numerical_columns"])
        self.assertIn("species", summary["categorical_columns"])

    def test_classification_pipeline(self):
        """Test classification configuration, training, and metrics."""
        iris = load_iris(as_frame=True)
        df = iris.frame
        df["species"] = iris.target_names[iris.target]
        df.drop(columns=["target"], inplace=True)

        self.dataset_mgr.load_dataframe(df, "iris.csv")
        features = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
        config_res = self.pipeline_mgr.configure(
            df=self.dataset_mgr.df,
            target_column="species",
            feature_columns=features,
            task_type="classification",
            test_size=0.2,
            random_seed=42
        )
        self.assertEqual(config_res["task_type"], "classification")
        self.assertEqual(len(config_res["classes"]), 3)

        # Train Random Forest
        metrics = self.pipeline_mgr.train_baseline("Random Forest Classifier")
        self.assertIn("accuracy", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.8)
        self.assertIn("confusion_matrix", metrics)

        # Test Prediction
        pred = self.pipeline_mgr.predict_single({
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2
        })
        self.assertEqual(pred["prediction"], "setosa")
        self.assertIn("confidence", pred)

    def test_regression_pipeline(self):
        """Test regression configuration, training, and metrics."""
        diab = load_diabetes(as_frame=True)
        df = diab.frame.rename(columns={"target": "progression"})

        self.dataset_mgr.load_dataframe(df, "diabetes.csv")
        features = [c for c in df.columns if c != "progression"]
        self.pipeline_mgr.configure(
            df=self.dataset_mgr.df,
            target_column="progression",
            feature_columns=features,
            task_type="regression",
            test_size=0.2,
            random_seed=42
        )

        metrics = self.pipeline_mgr.train_baseline("Random Forest Regressor")
        self.assertIn("r2", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)

        # Single prediction
        sample = {col: float(df[col].iloc[0]) for col in features}
        pred = self.pipeline_mgr.predict_single(sample)
        self.assertIsInstance(pred["prediction"], (int, float))

    def test_tuning_and_persistence(self):
        """Test hyperparameter tuning and model saving/reloading."""
        iris = load_iris(as_frame=True)
        df = iris.frame
        df["species"] = iris.target_names[iris.target]
        df.drop(columns=["target"], inplace=True)

        self.dataset_mgr.load_dataframe(df, "iris.csv")
        features = ["sepal length (cm)", "petal length (cm)"]
        self.pipeline_mgr.configure(
            df=self.dataset_mgr.df,
            target_column="species",
            feature_columns=features,
            task_type="classification",
            test_size=0.2,
            random_seed=42
        )

        self.pipeline_mgr.train_baseline("K-Nearest Neighbors Classifier")
        tuned = self.pipeline_mgr.fine_tune("K-Nearest Neighbors Classifier", search_type="grid", cv_folds=2)
        self.assertTrue(tuned["is_tuned"])
        self.assertIn("best_params", tuned)

        # Save model
        save_res = self.pipeline_mgr.save_model("Iris_KNN_Test", "iris.csv", 150, 5)
        model_id = save_res["model_id"]

        # Re-load model into a fresh manager
        fresh_mgr = MLPipelineManager()
        load_res = fresh_mgr.load_model(model_id)
        self.assertEqual(load_res["name"], "Iris_KNN_Test")

        # Predict with re-loaded model
        pred = fresh_mgr.predict_single({"sepal length (cm)": 5.0, "petal length (cm)": 1.5})
        self.assertEqual(pred["prediction"], "setosa")

        # Clean up
        self.pipeline_mgr.delete_model(model_id)


if __name__ == "__main__":
    unittest.main()
