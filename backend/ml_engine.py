"""
AutoML Studio ML Engine
Handles dataset preprocessing, model training, hyperparameter tuning,
evaluation, persistence (Joblib + SQLite), visualization, and prediction.
"""

import os
import io
import time
import json
import base64
import sqlite3
import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    import joblib
except ImportError:
    import pickle
    class FallbackJoblib:
        @staticmethod
        def dump(value, filename):
            with open(filename, 'wb') as f:
                pickle.dump(value, f)
        @staticmethod
        def load(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
    joblib = FallbackJoblib()

# Set matplotlib backend to non-interactive Agg before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Scikit-learn components
try:
    from sklearn.model_selection import (
        train_test_split,
        GridSearchCV,
        RandomizedSearchCV,
        StratifiedShuffleSplit,
        KFold,
        StratifiedKFold,
        cross_val_score
    )
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures

    # Classifiers
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC, SVR

    # Regressors
    from sklearn.linear_model import LinearRegression

    # Metrics
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        roc_curve,
        auc,
        roc_auc_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("\n[AutoML Studio Alert] Scikit-learn is not installed in this Python environment.")
    print("[AutoML Studio Alert] To train models, please run: pip install -r requirements.txt\n")
    Pipeline = Any
    ColumnTransformer = Any

# Constants & Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
DB_PATH = os.path.join(BASE_DIR, "automl.db")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def init_db():
    """Initializes the SQLite database for model metadata and experiments."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            algorithm TEXT NOT NULL,
            task_type TEXT NOT NULL,
            target_column TEXT NOT NULL,
            feature_names TEXT NOT NULL,
            hyperparameters TEXT NOT NULL,
            metrics TEXT NOT NULL,
            created_at TEXT NOT NULL,
            file_path TEXT NOT NULL,
            dataset_name TEXT,
            dataset_rows INTEGER,
            dataset_cols INTEGER
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY,
            model_id TEXT,
            algorithm TEXT NOT NULL,
            task_type TEXT NOT NULL,
            is_tuned INTEGER DEFAULT 0,
            search_type TEXT,
            train_score REAL,
            test_score REAL,
            metric_name TEXT,
            duration_sec REAL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


init_db()


def figure_to_base64(fig) -> str:
    """Converts a matplotlib figure to base64 encoded PNG string."""
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode('utf-8')


class DatasetManager:
    """Manages active dataset loading, summary, validation, and statistics."""
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.raw_df: Optional[pd.DataFrame] = None
        self.filename: str = ""
        self.duplicates_removed: int = 0

    def load_csv(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Loads and parses a CSV from raw bytes."""
        try:
            # Try utf-8 first, fallback to latin1
            try:
                df = pd.read_csv(io.BytesIO(file_content), encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(file_content), encoding='latin1')
        except Exception as e:
            raise ValueError(f"Failed to parse CSV file: {str(e)}")

        if df.empty:
            raise ValueError("The uploaded CSV file is empty.")
        if df.shape[1] < 2:
            raise ValueError("Dataset must have at least 2 columns (1 feature + 1 target).")

        self.filename = filename
        self.raw_df = df.copy()
        
        # Check duplicates
        initial_rows = len(df)
        df_cleaned = df.drop_duplicates()
        self.duplicates_removed = initial_rows - len(df_cleaned)
        self.df = df_cleaned.reset_index(drop=True)

        return self.get_summary()

    def load_dataframe(self, df: pd.DataFrame, name: str) -> Dict[str, Any]:
        """Loads a pre-existing pandas dataframe."""
        self.filename = name
        self.raw_df = df.copy()
        initial_rows = len(df)
        df_cleaned = df.drop_duplicates()
        self.duplicates_removed = initial_rows - len(df_cleaned)
        self.df = df_cleaned.reset_index(drop=True)
        return self.get_summary()

    def get_summary(self) -> Dict[str, Any]:
        """Generates comprehensive dataset statistics and column metadata."""
        if self.df is None:
            raise ValueError("No dataset loaded.")

        df = self.df
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        column_info = []
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            missing = int(df[col].isnull().sum())
            missing_pct = round((missing / len(df)) * 100, 2)
            unique_count = int(df[col].nunique())
            is_num = col in num_cols

            col_stat = {
                "name": col,
                "dtype": dtype_str,
                "is_numerical": is_num,
                "missing_count": missing,
                "missing_pct": missing_pct,
                "unique_count": unique_count,
            }
            if is_num:
                col_stat["mean"] = round(float(df[col].mean()), 4) if not pd.isna(df[col].mean()) else None
                col_stat["std"] = round(float(df[col].std()), 4) if not pd.isna(df[col].std()) else None
                col_stat["min"] = round(float(df[col].min()), 4) if not pd.isna(df[col].min()) else None
                col_stat["max"] = round(float(df[col].max()), 4) if not pd.isna(df[col].max()) else None
            else:
                top_val = df[col].mode()
                col_stat["top_value"] = str(top_val.iloc[0]) if not top_val.empty else None

            column_info.append(col_stat)

        # First 10 rows as preview
        preview_rows = df.head(10).replace({np.nan: None}).to_dict(orient="records")

        # Descriptive statistics for numerical features
        desc_stats = {}
        if num_cols:
            desc_df = df[num_cols].describe().round(4).replace({np.nan: None})
            desc_stats = desc_df.to_dict()

        return {
            "filename": self.filename,
            "total_rows_raw": len(self.raw_df) if self.raw_df is not None else len(df),
            "total_rows_cleaned": len(df),
            "duplicates_removed": self.duplicates_removed,
            "total_columns": int(df.shape[1]),
            "numerical_columns": num_cols,
            "categorical_columns": cat_cols,
            "columns": column_info,
            "preview": preview_rows,
            "descriptive_stats": desc_stats
        }

    def generate_visualizations(self, target_col: Optional[str] = None) -> Dict[str, str]:
        """Generates distribution, correlation, and missing-value plots."""
        if self.df is None:
            raise ValueError("No dataset loaded.")

        df = self.df
        charts = {}

        # 1. Missing Values Bar Chart
        missing = df.isnull().sum()
        if missing.sum() > 0:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            missing_cols = missing[missing > 0]
            missing_pct = (missing_cols / len(df)) * 100
            ax.bar(missing_pct.index, missing_pct.values, color='#ef4444', alpha=0.85, edgecolor='#b91c1c')
            ax.set_title("Missing Values Percentage (%) by Column", fontsize=12, fontweight='bold', pad=10)
            ax.set_ylabel("Missing (%)", fontsize=10)
            ax.set_ylim(0, max(100, float(missing_pct.max()) * 1.15))
            plt.xticks(rotation=45, ha='right', fontsize=9)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            charts["missing_values"] = figure_to_base64(fig)

        # 2. Target Variable Distribution (if configured)
        if target_col and target_col in df.columns:
            fig, ax = plt.subplots(figsize=(6.5, 3.8))
            is_num = target_col in df.select_dtypes(include=[np.number]).columns
            nunique = df[target_col].nunique()

            if nunique <= 20 or not is_num:
                # Categorical or discrete target
                counts = df[target_col].value_counts().head(15)
                ax.bar([str(x) for x in counts.index], counts.values, color='#3b82f6', alpha=0.85, edgecolor='#1d4ed8')
                ax.set_title(f"Target Distribution: '{target_col}' (Class Frequencies)", fontsize=12, fontweight='bold', pad=10)
                ax.set_ylabel("Count", fontsize=10)
                plt.xticks(rotation=30, ha='right', fontsize=9)
            else:
                # Continuous target
                data = df[target_col].dropna()
                ax.hist(data, bins=25, color='#10b981', alpha=0.75, edgecolor='#047857', density=True)
                ax.set_title(f"Target Distribution: '{target_col}' (Density Histogram)", fontsize=12, fontweight='bold', pad=10)
                ax.set_ylabel("Density", fontsize=10)
                ax.set_xlabel(target_col, fontsize=10)

            ax.grid(axis='y', linestyle='--', alpha=0.5)
            charts["target_distribution"] = figure_to_base64(fig)

        # 3. Correlation Heatmap (for numerical features)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) >= 2:
            # Limit to top 12 numerical features for legibility if large
            heatmap_cols = num_cols[:12]
            corr = df[heatmap_cols].corr()
            
            fig, ax = plt.subplots(figsize=(7, 5.5))
            cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
            fig.colorbar(cax, shrink=0.8)

            ax.set_xticks(range(len(heatmap_cols)))
            ax.set_yticks(range(len(heatmap_cols)))
            ax.set_xticklabels(heatmap_cols, rotation=45, ha='left', fontsize=8)
            ax.set_yticklabels(heatmap_cols, fontsize=8)

            # Annotate values
            for i in range(len(heatmap_cols)):
                for j in range(len(heatmap_cols)):
                    val = corr.iloc[i, j]
                    ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                            color='black' if abs(val) < 0.6 else 'white', fontsize=7.5)

            ax.set_title(f"Feature Correlation Heatmap ({len(heatmap_cols)} numeric features)",
                         fontsize=11, fontweight='bold', pad=15)
            charts["correlation_heatmap"] = figure_to_base64(fig)

        # 4. Feature Distributions (Top 4 numeric features)
        if num_cols:
            cols_to_plot = [c for c in num_cols if c != target_col][:4]
            if cols_to_plot:
                fig, axes = plt.subplots(1, len(cols_to_plot), figsize=(3.8 * len(cols_to_plot), 3.2))
                if len(cols_to_plot) == 1:
                    axes = [axes]
                for ax, col in zip(axes, cols_to_plot):
                    clean_data = df[col].dropna()
                    ax.hist(clean_data, bins=20, color='#6366f1', alpha=0.75, edgecolor='#4338ca')
                    ax.set_title(col, fontsize=9.5, fontweight='semibold')
                    ax.set_ylabel("Count", fontsize=8)
                    ax.grid(axis='y', linestyle='--', alpha=0.4)
                charts["feature_distributions"] = figure_to_base64(fig)

        return charts


class MLPipelineManager:
    """Builds, trains, tunes, evaluates, and persists Scikit-learn pipelines."""

    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.fitted_pipeline: Optional[Pipeline] = None
        self.baseline_pipeline: Optional[Pipeline] = None
        self.tuned_pipeline: Optional[Pipeline] = None
        
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.classes_: Optional[List[Any]] = None

        self.last_metrics: Dict[str, Any] = {}
        self.tuning_results: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Any] = {}

    def configure(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        task_type: str,
        test_size: float = 0.2,
        random_seed: int = 42,
        imputation_strategy: str = "median",
        scaling: bool = True
    ) -> Dict[str, Any]:
        """Validates inputs, sets up train-test split, and validates task alignment."""
        if not HAS_SKLEARN:
            raise ValueError(
                "Scikit-learn (sklearn) is not installed in this Python environment. "
                "Please run in your terminal: pip install -r requirements.txt"
            )
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' does not exist in dataset.")
        if target_column in feature_columns:
            raise ValueError("Target column cannot be included in input features.")
        if not feature_columns:
            raise ValueError("At least one input feature must be selected.")
        for col in feature_columns:
            if col not in df.columns:
                raise ValueError(f"Feature column '{col}' does not exist in dataset.")

        # Clean target: remove rows where target is null
        valid_mask = ~df[target_column].isnull()
        if valid_mask.sum() < 10:
            raise ValueError("Target column has too many missing values (fewer than 10 valid rows).")

        df_subset = df[valid_mask].copy()

        # Task type validation
        y_raw = df_subset[target_column]
        nunique = y_raw.nunique()
        is_num = np.issubdtype(y_raw.dtype, np.number)

        if task_type == "classification":
            if nunique > 50 and is_num:
                raise ValueError(
                    f"Selected target has {nunique} unique numerical values. "
                    "This looks like a continuous regression target. Please select 'Regression' or pick a categorical target."
                )
            if nunique < 2:
                raise ValueError("Classification target must contain at least 2 distinct classes.")
        elif task_type == "regression":
            if not is_num:
                raise ValueError(
                    f"Target column '{target_column}' is non-numeric ({y_raw.dtype}). "
                    "Regression requires a continuous numerical target. Please select 'Classification' instead."
                )
            if nunique < 5:
                raise ValueError(
                    f"Target column '{target_column}' has only {nunique} unique values. "
                    "A regression target requires continuous values. Consider choosing 'Classification'."
                )

        X = df_subset[feature_columns].copy()
        y = y_raw.copy()

        # Check for constant features (zero variance)
        constant_features = [col for col in feature_columns if X[col].nunique() <= 1]
        if len(constant_features) == len(feature_columns):
            raise ValueError("All selected features have constant values. Cannot train a model.")

        # Train-test split
        stratify = None
        if task_type == "classification":
            # Stratify only if each class has at least 2 samples
            class_counts = y.value_counts()
            if (class_counts >= 2).all() and len(class_counts) <= 20:
                stratify = y

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_seed, stratify=stratify
            )
        except Exception:
            # Fallback without stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_seed
            )

        self.config = {
            "target_column": target_column,
            "feature_columns": feature_columns,
            "task_type": task_type,
            "test_size": test_size,
            "random_seed": random_seed,
            "imputation_strategy": imputation_strategy,
            "scaling": scaling,
            "constant_features_flagged": constant_features
        }

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        if task_type == "classification":
            self.classes_ = sorted(list(y.unique()))
        else:
            self.classes_ = None

        return {
            "status": "configured",
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_count": len(feature_columns),
            "task_type": task_type,
            "target_column": target_column,
            "classes": [str(c) for c in self.classes_] if self.classes_ is not None else None,
            "constant_features": constant_features
        }

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Constructs a leak-free ColumnTransformer for numerical and categorical features."""
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        transformers = []

        if num_cols:
            num_steps = [
                ("imputer", SimpleImputer(strategy=self.config.get("imputation_strategy", "median")))
            ]
            if self.config.get("scaling", True):
                num_steps.append(("scaler", StandardScaler()))
            transformers.append(("num", Pipeline(num_steps), num_cols))

        if cat_cols:
            cat_steps = [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]
            transformers.append(("cat", Pipeline(cat_steps), cat_cols))

        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _get_estimator_and_params(self, algorithm: str, random_seed: int) -> Tuple[Any, Dict[str, List[Any]]]:
        """Returns the Scikit-learn estimator instance and hyperparameter search grid."""
        task = self.config.get("task_type", "classification")

        if task == "classification":
            if algorithm == "Logistic Regression":
                estimator = LogisticRegression(max_iter=1000, random_state=random_seed)
                param_grid = {
                    "clf__C": [0.01, 0.1, 1.0, 10.0],
                    "clf__solver": ["lbfgs"],
                }
            elif algorithm == "Decision Tree Classifier":
                estimator = DecisionTreeClassifier(random_state=random_seed)
                param_grid = {
                    "clf__max_depth": [3, 5, 8, None],
                    "clf__min_samples_split": [2, 5, 10],
                    "clf__criterion": ["gini", "entropy"]
                }
            elif algorithm == "Random Forest Classifier":
                estimator = RandomForestClassifier(n_estimators=50, random_state=random_seed)
                param_grid = {
                    "clf__n_estimators": [30, 50, 100],
                    "clf__max_depth": [4, 8, None],
                    "clf__min_samples_split": [2, 5]
                }
            elif algorithm == "K-Nearest Neighbors Classifier":
                estimator = KNeighborsClassifier(n_neighbors=5)
                param_grid = {
                    "clf__n_neighbors": [3, 5, 7, 9],
                    "clf__weights": ["uniform", "distance"]
                }
            elif algorithm == "Naive Bayes":
                estimator = GaussianNB()
                param_grid = {
                    "clf__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
                }
            elif algorithm == "Support Vector Machine Classifier":
                estimator = SVC(probability=True, random_state=random_seed)
                param_grid = {
                    "clf__C": [0.1, 1.0, 10.0],
                    "clf__kernel": ["linear", "rbf"]
                }
            else:
                raise ValueError(f"Unsupported classification algorithm: {algorithm}")

        else: # Regression
            if algorithm == "Linear Regression":
                estimator = LinearRegression()
                param_grid = {
                    "reg__fit_intercept": [True, False]
                }
            elif algorithm == "Polynomial Regression":
                # Pipeline step inside regressor
                estimator = Pipeline([
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("lin", LinearRegression())
                ])
                param_grid = {
                    "reg__poly__degree": [1, 2, 3]
                }
            elif algorithm == "Decision Tree Regressor":
                estimator = DecisionTreeRegressor(random_state=random_seed)
                param_grid = {
                    "reg__max_depth": [3, 5, 8, None],
                    "reg__min_samples_split": [2, 5, 10]
                }
            elif algorithm == "Random Forest Regressor":
                estimator = RandomForestRegressor(n_estimators=50, random_state=random_seed)
                param_grid = {
                    "reg__n_estimators": [30, 50, 100],
                    "reg__max_depth": [4, 8, None],
                    "reg__min_samples_split": [2, 5]
                }
            elif algorithm == "Support Vector Regression":
                estimator = SVR()
                param_grid = {
                    "reg__C": [0.1, 1.0, 10.0],
                    "reg__kernel": ["linear", "rbf"],
                    "reg__epsilon": [0.01, 0.1, 0.2]
                }
            else:
                raise ValueError(f"Unsupported regression algorithm: {algorithm}")

        return estimator, param_grid

    def train_baseline(self, algorithm: str) -> Dict[str, Any]:
        """Trains baseline model with full Scikit-learn pipeline."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Dataset not configured. Please run configuration step first.")

        seed = self.config.get("random_seed", 42)
        task = self.config.get("task_type", "classification")
        preprocessor = self._build_preprocessor(self.X_train)
        estimator, _ = self._get_estimator_and_params(algorithm, seed)

        step_name = "clf" if task == "classification" else "reg"
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            (step_name, estimator)
        ])

        start_time = time.time()
        pipeline.fit(self.X_train, self.y_train)
        elapsed_sec = round(time.time() - start_time, 4)

        self.baseline_pipeline = pipeline
        self.fitted_pipeline = pipeline

        # Evaluate on test set
        metrics = self._evaluate_pipeline(pipeline, self.X_test, self.y_test, task)
        metrics["training_time_sec"] = elapsed_sec
        metrics["algorithm"] = algorithm
        metrics["is_tuned"] = False
        metrics["hyperparameters"] = self._extract_params(pipeline, step_name)

        # Cross-validation score on training set
        cv_metric = "accuracy" if task == "classification" else "r2"
        try:
            cv_scores = cross_val_score(pipeline, self.X_train, self.y_train, cv=3, scoring=cv_metric)
            metrics["cv_score_mean"] = round(float(np.mean(cv_scores)), 4)
            metrics["cv_score_std"] = round(float(np.std(cv_scores)), 4)
        except Exception:
            metrics["cv_score_mean"] = None
            metrics["cv_score_std"] = None

        self.last_metrics = metrics
        
        # Visualizations
        charts = self._generate_evaluation_charts(pipeline, self.X_test, self.y_test, task)
        metrics["charts"] = charts

        return metrics

    def fine_tune(self, algorithm: str, search_type: str = "grid", cv_folds: int = 3, n_iter: int = 10) -> Dict[str, Any]:
        """Performs hyperparameter fine-tuning using GridSearchCV or RandomizedSearchCV."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Dataset not configured.")

        seed = self.config.get("random_seed", 42)
        task = self.config.get("task_type", "classification")
        preprocessor = self._build_preprocessor(self.X_train)
        estimator, param_grid = self._get_estimator_and_params(algorithm, seed)

        step_name = "clf" if task == "classification" else "reg"
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            (step_name, estimator)
        ])

        scoring = "accuracy" if task == "classification" else "r2"
        
        # Adapt folds if train size is small
        n_samples = len(self.X_train)
        cv = min(cv_folds, max(2, n_samples // 5))

        start_time = time.time()
        if search_type == "random":
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=min(n_iter, 12),
                scoring=scoring,
                cv=cv,
                random_state=seed,
                n_jobs=-1
            )
        else:
            search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                scoring=scoring,
                cv=cv,
                n_jobs=-1
            )

        search.fit(self.X_train, self.y_train)
        elapsed_sec = round(time.time() - start_time, 4)

        best_pipeline = search.best_estimator_
        self.tuned_pipeline = best_pipeline
        self.fitted_pipeline = best_pipeline

        # Evaluate tuned pipeline on test set
        tuned_metrics = self._evaluate_pipeline(best_pipeline, self.X_test, self.y_test, task)
        tuned_metrics["training_time_sec"] = elapsed_sec
        tuned_metrics["algorithm"] = algorithm
        tuned_metrics["is_tuned"] = True
        tuned_metrics["search_type"] = search_type
        tuned_metrics["best_cv_score"] = round(float(search.best_score_), 4)
        tuned_metrics["best_params"] = {k.replace(f"{step_name}__", ""): v for k, v in search.best_params_.items()}
        tuned_metrics["hyperparameters"] = self._extract_params(best_pipeline, step_name)

        # Baseline comparison
        baseline_score = self.last_metrics.get("accuracy" if task == "classification" else "r2")
        tuned_score = tuned_metrics.get("accuracy" if task == "classification" else "r2")
        improvement = round(tuned_score - baseline_score, 4) if baseline_score is not None and tuned_score is not None else 0.0

        tuned_metrics["baseline_score"] = baseline_score
        tuned_metrics["tuned_score"] = tuned_score
        tuned_metrics["improvement"] = improvement

        charts = self._generate_evaluation_charts(best_pipeline, self.X_test, self.y_test, task)
        tuned_metrics["charts"] = charts

        self.last_metrics = tuned_metrics
        return tuned_metrics

    def _extract_params(self, pipeline: Pipeline, step_name: str) -> Dict[str, Any]:
        """Extracts readable hyperparameters from the estimator."""
        estimator = pipeline.named_steps.get(step_name)
        if not estimator:
            return {}
        try:
            params = estimator.get_params()
            clean_params = {}
            for k, v in params.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    clean_params[k] = v
            return clean_params
        except Exception:
            return {}

    def _evaluate_pipeline(self, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, task: str) -> Dict[str, Any]:
        """Evaluates pipeline predictions on test set with authentic metrics."""
        y_pred = pipeline.predict(X_test)

        if task == "classification":
            acc = float(accuracy_score(y_test, y_pred))
            prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
            prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
            rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
            rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
            f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
            f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

            # Confusion matrix
            labels = sorted(list(set(y_test).union(set(y_pred))))
            cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()

            # ROC-AUC calculation
            roc_auc_val = None
            roc_note = None
            try:
                if hasattr(pipeline, "predict_proba"):
                    probs = pipeline.predict_proba(X_test)
                    if len(labels) == 2:
                        roc_auc_val = float(roc_auc_score(y_test, probs[:, 1]))
                    else:
                        roc_auc_val = float(roc_auc_score(y_test, probs, multi_class="ovr", average="weighted"))
                else:
                    roc_note = "Model does not output probability estimates (ROC-AUC requires predict_proba)."
            except Exception as e:
                roc_note = f"ROC-AUC unavailable: {str(e)}"

            return {
                "task_type": "classification",
                "accuracy": round(acc, 4),
                "precision": round(prec_weighted, 4),
                "precision_macro": round(prec_macro, 4),
                "recall": round(rec_weighted, 4),
                "recall_macro": round(rec_macro, 4),
                "f1_score": round(f1_weighted, 4),
                "f1_macro": round(f1_macro, 4),
                "confusion_matrix": cm,
                "labels": [str(l) for l in labels],
                "roc_auc": round(roc_auc_val, 4) if roc_auc_val is not None else None,
                "roc_note": roc_note
            }

        else: # Regression
            mae = float(mean_absolute_error(y_test, y_pred))
            mse = float(mean_squared_error(y_test, y_pred))
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(y_test, y_pred))

            return {
                "task_type": "regression",
                "mae": round(mae, 4),
                "mse": round(mse, 4),
                "rmse": round(rmse, 4),
                "r2": round(r2, 4)
            }

    def _generate_evaluation_charts(self, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, task: str) -> Dict[str, str]:
        """Generates visual evaluation plots (confusion matrix, ROC, residuals, actual vs predicted)."""
        charts = {}
        y_pred = pipeline.predict(X_test)

        if task == "classification":
            labels = sorted(list(set(y_test).union(set(y_pred))))
            cm = confusion_matrix(y_test, y_pred, labels=labels)

            # 1. Confusion Matrix Heatmap
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            cax = ax.matshow(cm, cmap='Blues', alpha=0.85)
            fig.colorbar(cax, shrink=0.8)

            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels([str(l) for l in labels], rotation=30, ha='left', fontsize=8.5)
            ax.set_yticklabels([str(l) for l in labels], fontsize=8.5)
            ax.set_xlabel("Predicted Label", fontsize=10, labelpad=10)
            ax.set_ylabel("True Label", fontsize=10)
            ax.set_title("Confusion Matrix (Test Set)", fontsize=11, fontweight='bold', pad=15)

            for i in range(len(labels)):
                for j in range(len(labels)):
                    val = cm[i, j]
                    ax.text(j, i, str(val), ha='center', va='center',
                            color='white' if val > cm.max() / 2 else 'black',
                            fontweight='bold', fontsize=10)

            charts["confusion_matrix"] = figure_to_base64(fig)

            # 2. ROC Curve (if binary and predict_proba available)
            if hasattr(pipeline, "predict_proba") and len(labels) == 2:
                try:
                    probs = pipeline.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, probs, pos_label=labels[1])
                    roc_auc_val = auc(fpr, tpr)

                    fig, ax = plt.subplots(figsize=(5.5, 4.2))
                    ax.plot(fpr, tpr, color='#2563eb', lw=2.2, label=f'ROC curve (AUC = {roc_auc_val:.3f})')
                    ax.plot([0, 1], [0, 1], color='#94a3b8', lw=1.5, linestyle='--', label='Random Chance')
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate (FPR)', fontsize=9.5)
                    ax.set_ylabel('True Positive Rate (TPR)', fontsize=9.5)
                    ax.set_title('Receiver Operating Characteristic (ROC)', fontsize=11, fontweight='bold', pad=10)
                    ax.legend(loc="lower right", fontsize=9)
                    ax.grid(True, linestyle=':', alpha=0.6)
                    charts["roc_curve"] = figure_to_base64(fig)
                except Exception:
                    pass

        else: # Regression
            # 1. Actual vs Predicted Plot
            fig, ax = plt.subplots(figsize=(5.5, 4.2))
            ax.scatter(y_test, y_pred, color='#2563eb', alpha=0.7, edgecolors='none', s=35)
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], color='#ef4444', linestyle='--', lw=2, label='Perfect Fit (y = x)')
            ax.set_xlabel("Actual Values", fontsize=9.5)
            ax.set_ylabel("Predicted Values", fontsize=9.5)
            ax.set_title("Actual vs. Predicted", fontsize=11, fontweight='bold', pad=10)
            ax.legend(fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.5)
            charts["actual_vs_predicted"] = figure_to_base64(fig)

            # 2. Residual Plot
            residuals = y_test - y_pred
            fig, ax = plt.subplots(figsize=(5.5, 4.2))
            ax.scatter(y_pred, residuals, color='#7c3aed', alpha=0.7, edgecolors='none', s=35)
            ax.axhline(0, color='#ef4444', linestyle='--', lw=1.8)
            ax.set_xlabel("Predicted Values", fontsize=9.5)
            ax.set_ylabel("Residuals (Actual - Predicted)", fontsize=9.5)
            ax.set_title("Residual Plot", fontsize=11, fontweight='bold', pad=10)
            ax.grid(True, linestyle=':', alpha=0.5)
            charts["residuals_plot"] = figure_to_base64(fig)

            # 3. Residual Distribution
            fig, ax = plt.subplots(figsize=(5.5, 4.2))
            ax.hist(residuals, bins=25, color='#059669', alpha=0.75, edgecolor='#047857')
            ax.axvline(0, color='#ef4444', linestyle='--', lw=1.8)
            ax.set_xlabel("Residual Value", fontsize=9.5)
            ax.set_ylabel("Frequency", fontsize=9.5)
            ax.set_title("Residual Distribution (Error Analysis)", fontsize=11, fontweight='bold', pad=10)
            ax.grid(axis='y', linestyle=':', alpha=0.5)
            charts["residual_distribution"] = figure_to_base64(fig)

        return charts

    def predict_single(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Makes real prediction for a single data point using the fitted pipeline."""
        if self.fitted_pipeline is None:
            raise ValueError("No trained model available. Please train or load a model first.")

        # Reconstruct DataFrame matching feature columns exactly
        feature_cols = self.config.get("feature_columns", [])
        row = {}
        for col in feature_cols:
            val = input_dict.get(col)
            if val is None or val == "":
                # Imputation in pipeline handles NaN
                row[col] = np.nan
            else:
                # Try converting to float if numeric
                try:
                    row[col] = float(val)
                except (ValueError, TypeError):
                    row[col] = str(val)

        df_input = pd.DataFrame([row], columns=feature_cols)
        pred = self.fitted_pipeline.predict(df_input)[0]

        result: Dict[str, Any] = {
            "prediction": int(pred) if isinstance(pred, (np.integer, int)) else (
                round(float(pred), 4) if isinstance(pred, (np.floating, float)) else str(pred)
            ),
            "task_type": self.config.get("task_type", "classification")
        }

        # Check for class probabilities
        if self.config.get("task_type") == "classification" and hasattr(self.fitted_pipeline, "predict_proba"):
            try:
                probs = self.fitted_pipeline.predict_proba(df_input)[0]
                classes = self.fitted_pipeline.classes_
                prob_dict = {str(cls): round(float(p) * 100, 2) for cls, p in zip(classes, probs)}
                result["probabilities"] = prob_dict
                result["confidence"] = round(float(np.max(probs)) * 100, 2)
            except Exception:
                result["probabilities"] = None

        return result

    def batch_predict(self, file_content: bytes) -> Dict[str, Any]:
        """Runs predictions on an uploaded CSV file using the active fitted pipeline."""
        if self.fitted_pipeline is None:
            raise ValueError("No trained model available. Please train or load a model first.")

        try:
            df = pd.read_csv(io.BytesIO(file_content))
        except Exception as e:
            raise ValueError(f"Failed to parse batch CSV: {str(e)}")

        feature_cols = self.config.get("feature_columns", [])
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            raise ValueError(f"Batch dataset missing required features: {', '.join(missing_features)}")

        X_batch = df[feature_cols].copy()
        preds = self.fitted_pipeline.predict(X_batch)

        df_result = df.copy()
        df_result["Predicted_" + str(self.config.get("target_column", "Output"))] = preds

        # If classification and proba available
        if self.config.get("task_type") == "classification" and hasattr(self.fitted_pipeline, "predict_proba"):
            try:
                probs = self.fitted_pipeline.predict_proba(X_batch)
                df_result["Confidence_Pct"] = np.round(np.max(probs, axis=1) * 100, 2)
            except Exception:
                pass

        # Convert back to CSV string
        csv_buffer = io.StringIO()
        df_result.to_csv(csv_buffer, index=False)
        csv_string = csv_buffer.getvalue()

        # Preview of first 15 predictions
        preview = df_result.head(15).replace({np.nan: None}).to_dict(orient="records")

        return {
            "total_samples": len(df),
            "preview": preview,
            "csv_data": csv_string
        }

    def save_model(self, model_name: str, dataset_name: str = "", rows: int = 0, cols: int = 0) -> Dict[str, Any]:
        """Saves fitted pipeline with Joblib and registers metadata in SQLite."""
        if self.fitted_pipeline is None:
            raise ValueError("No model is currently fitted to save.")

        model_id = f"model_{int(time.time())}_{np.random.randint(1000, 9999)}"
        file_name = f"{model_id}.joblib"
        file_path = os.path.join(MODELS_DIR, file_name)

        # Save artifact with Joblib
        payload = {
            "pipeline": self.fitted_pipeline,
            "config": self.config,
            "metrics": self.last_metrics,
            "classes": self.classes_
        }
        joblib.dump(payload, file_path)

        # Register in SQLite
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        created_at = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        cursor.execute("""
            INSERT INTO models (
                id, name, algorithm, task_type, target_column, feature_names,
                hyperparameters, metrics, created_at, file_path, dataset_name, dataset_rows, dataset_cols
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_id,
            model_name or f"Model-{model_id[:8]}",
            self.last_metrics.get("algorithm", "Unknown"),
            self.config.get("task_type", "classification"),
            self.config.get("target_column", "Target"),
            json.dumps(self.config.get("feature_columns", [])),
            json.dumps(self.last_metrics.get("hyperparameters", {})),
            json.dumps(self.last_metrics),
            created_at,
            file_path,
            dataset_name,
            rows,
            cols
        ))
        conn.commit()
        conn.close()

        return {
            "status": "saved",
            "model_id": model_id,
            "name": model_name,
            "created_at": created_at
        }

    def load_model(self, model_id: str) -> Dict[str, Any]:
        """Loads a model from Joblib file and SQLite metadata into active state."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM models WHERE id = ?", (model_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise ValueError(f"Model ID '{model_id}' not found.")

        file_path = row[9]
        if not os.path.exists(file_path):
            raise ValueError(f"Model file '{file_path}' no longer exists on disk.")

        payload = joblib.load(file_path)
        self.fitted_pipeline = payload["pipeline"]
        self.config = payload["config"]
        self.last_metrics = payload.get("metrics", {})
        self.classes_ = payload.get("classes")

        return {
            "status": "loaded",
            "id": row[0],
            "name": row[1],
            "algorithm": row[2],
            "task_type": row[3],
            "target_column": row[4],
            "feature_names": json.loads(row[5]),
            "hyperparameters": json.loads(row[6]),
            "metrics": json.loads(row[7]),
            "created_at": row[8]
        }

    def list_saved_models(self) -> List[Dict[str, Any]]:
        """Lists all saved models with performance metrics and details."""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM models ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()

        models = []
        for r in rows:
            models.append({
                "id": r["id"],
                "name": r["name"],
                "algorithm": r["algorithm"],
                "task_type": r["task_type"],
                "target_column": r["target_column"],
                "feature_names": json.loads(r["feature_names"]),
                "hyperparameters": json.loads(r["hyperparameters"]),
                "metrics": json.loads(r["metrics"]),
                "created_at": r["created_at"],
                "dataset_name": r["dataset_name"],
                "dataset_rows": r["dataset_rows"],
                "dataset_cols": r["dataset_cols"]
            })
        return models

    def delete_model(self, model_id: str) -> bool:
        """Deletes a saved model file and its SQLite record."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM models WHERE id = ?", (model_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return False

        file_path = row[0]
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

        cursor.execute("DELETE FROM models WHERE id = ?", (model_id,))
        conn.commit()
        conn.close()
        return True
