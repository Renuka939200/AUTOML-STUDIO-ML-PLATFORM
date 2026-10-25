# AutoML Studio – End-to-End Machine Learning Platform

AutoML Studio is a full-stack, automated machine learning application designed for academic projects, data practitioners, and researchers. It provides a complete workflow from raw CSV ingestion to hyperparameter fine-tuning, statistical evaluation, Joblib persistence, and real-time inference.

---

## 🌟 Key Features

1. **Dataset Ingestion & Exploration**:
   - Upload any CSV file or load 1-click built-in benchmark datasets (Iris Classification, Wine Recognition, Diabetes Progression, California Housing).
   - Automated duplicate detection and removal with transparent row reporting.
   - Schema type inference (continuous numerical vs categorical text/factors).
   - Nullity matrix / missing value statistics per column.
   - Descriptive summary statistics (mean, standard deviation, min, 25%, median, 75%, max).
   - Matplotlib & Seaborn visualizations: Correlation heatmaps, feature histograms with KDE, missing value bars, and target distributions.

2. **Data Preprocessing & Leakage Prevention**:
   - Built with Scikit-learn's `ColumnTransformer` and `Pipeline`.
   - Missing-value imputation: Median or Mean for numeric columns, Most-Frequent (mode) for categorical features.
   - Categorical Encoding: `OneHotEncoder(handle_unknown='ignore')` prevents pipeline failures when new categories appear at test or prediction time.
   - Numeric Feature Scaling: Optional `StandardScaler` (zero-mean, unit-variance).
   - **Zero Data Leakage Guarantee**: Train-test splitting precedes all transformer fitting. Preprocessors are strictly fitted on the training split and applied to test and inference sets.

3. **Supported Machine Learning Algorithms**:
   - **Classification (6 Models)**:
     1. Logistic Regression (L-BFGS optimizer, L2 regularization)
     2. Decision Tree Classifier (Gini impurity / entropy)
     3. Random Forest Classifier (Bagging ensemble with randomized subspace features)
     4. K-Nearest Neighbors Classifier (Euclidean neighborhood voting)
     5. Naive Bayes (GaussianNB with feature variance smoothing)
     6. Support Vector Machine Classifier (SVC with RBF kernel and calibrated probabilities)
   - **Regression (5 Models)**:
     1. Linear Regression (Ordinary Least Squares)
     2. Polynomial Regression (`PolynomialFeatures(degree=2, 3)` + Linear Regression)
     3. Decision Tree Regressor (Piecewise constant mean squared error minimization)
     4. Random Forest Regressor (Ensembled variance reduction over decorrelated trees)
     5. Support Vector Regression (SVR with epsilon-insensitive tube)

4. **Hyperparameter Fine-Tuning**:
   - User-selectable `GridSearchCV` (exhaustive) and `RandomizedSearchCV` (stochastic).
   - Configurable K-Fold cross-validation (2, 3, or 5 folds) performed exclusively on the training partition.
   - Algorithm-specific hyperparameter search grids.
   - Baseline vs. tuned model performance comparison with delta improvement scores.

5. **Statistical Model Evaluation**:
   - **Classification Metrics**: Test Accuracy, Macro/Weighted Precision, Macro/Weighted Recall, Weighted F1-Score, Confusion Matrix Heatmap, and ROC-AUC curve (with probability calibration).
   - **Regression Metrics**: R² Score (Coefficient of Determination), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Mean Squared Error (MSE).
   - Visual plots: Confusion matrix, ROC curve, Actual vs. Predicted scatter plot (with ideal 45° reference line), Residual plot, and Residual error distribution histogram.

6. **Interactive Inference & Batch Predictions**:
   - Dynamic form dynamically generated for all active input features.
   - Real-time single-record predictions with confidence score and class probability distributions.
   - High-throughput batch CSV scoring with tabular preview and one-click scored CSV download.

7. **Model Persistence & Registry**:
   - Entire fitted pipelines (preprocessor + estimator) serialized using **Joblib**.
   - SQLite database (`automl.db`) registers metadata, hyperparameters, evaluation metrics, timestamps, and dataset schemas.
   - Instant model reloading from disk directly into active memory.

---

## 🛠️ Technology Stack

- **Frontend**: React 19, Vite, TypeScript, Tailwind CSS, Lucide React.
- **Full-Stack Server**: Express.js reverse-proxy bridge with `vite.middlewares`.
- **Backend**: Python 3 (Flask, Flask-CORS).
- **Machine Learning**: Scikit-learn, Pandas, NumPy, SciPy.
- **Visualization**: Matplotlib, Seaborn (server-side non-interactive rendering to Base64 PNG).
- **Persistence**: Joblib & SQLite 3.

---

## 🚀 Installation & Execution Instructions

### Prerequisites
- Node.js (v18 or higher) & npm
- Python (v3.10 or higher) & pip

### 1. Install Node Dependencies
```bash
npm install
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```
*(Or inside a virtualenv: `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`)*

### 3. Run Application

#### Combined Full-Stack Execution (Development Server)
```bash
npm run dev
```
This starts the full-stack server on `http://localhost:3000`, automatically managing the Python Flask service on port 5001 and mounting Vite dev middleware.

#### Running Frontend & Backend Separately (Alternative)
**Terminal 1 (Backend):**
```bash
cd backend
python3 app.py --port 5001
```

**Terminal 2 (Frontend):**
```bash
npm run dev
```

---

## 🧪 Automated Testing

Automated unit tests for the ML pipeline, data preprocessing, classifiers, regressors, tuning, and Joblib persistence are included in `backend/test_ml_pipeline.py`.

Run the test suite:
```bash
python3 backend/test_ml_pipeline.py
```

Expected output:
```text
....
----------------------------------------------------------------------
Ran 4 tests in 4.609s

OK
```

---

## 📡 REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check probe |
| `POST` | `/api/upload` | Upload CSV dataset (multipart/form-data) |
| `GET` | `/api/dataset/summary` | Retrieve active dataset schema and statistics |
| `GET` | `/api/dataset/samples` | List benchmark sample datasets |
| `POST` | `/api/dataset/load-sample` | Load benchmark dataset (Iris, Wine, Diabetes, Housing) |
| `POST` | `/api/dataset/configure` | Configure target, features, task type, split, and imputer |
| `GET` | `/api/dataset/visualizations` | Retrieve correlation and distribution plots |
| `POST` | `/api/model/train` | Train baseline Scikit-learn pipeline |
| `POST` | `/api/model/tune` | Run GridSearchCV / RandomizedSearchCV tuning |
| `GET` | `/api/model/metrics` | Retrieve evaluation metrics for active fitted model |
| `GET` | `/api/model/visualizations` | Retrieve model evaluation plots (confusion matrix, ROC, residuals) |
| `POST` | `/api/model/predict` | Single-record inference |
| `POST` | `/api/model/batch-predict` | Batch CSV inference and scoring |
| `POST` | `/api/model/save` | Persist fitted pipeline via Joblib and register in SQLite |
| `GET` | `/api/models` | List all saved models in SQLite registry |
| `POST` | `/api/model/load` | Load a persisted model into active memory |
| `DELETE` | `/api/models/<id>` | Delete model artifact and database record |

---

## ⚠️ Known Limitations & Future Work

1. **High-Cardinality Features**: Extremely high cardinality categorical columns (e.g. unique user IDs or raw free text) can cause memory inflation with OneHotEncoder. Future versions will support Target Encoding and CatBoost-style frequency encoding.
2. **Extreme Class Imbalance**: Handled via stratified sampling; future work will incorporate SMOTE oversampling.
3. **Automated Stacking & Ensembling**: Future integration of `StackingClassifier` and `VotingClassifier` over top-performing models.
