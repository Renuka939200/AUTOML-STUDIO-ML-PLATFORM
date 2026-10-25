"""
AutoML Studio – Flask REST API Server
Provides endpoints for dataset upload, summary, configuration,
model training, hyperparameter fine-tuning, evaluation, predictions,
and persistence.
"""

import os
import io
import argparse
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file

try:
    from flask_cors import CORS
    has_cors_ext = True
except ImportError:
    has_cors_ext = False

from ml_engine import DatasetManager, MLPipelineManager

app = Flask(__name__)

if has_cors_ext:
    CORS(app)
else:
    @app.after_request
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

    @app.before_request
    def handle_preflight():
        if request.method == 'OPTIONS':
            res = app.make_default_options_response()
            res.headers['Access-Control-Allow-Origin'] = '*'
            res.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
            res.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
            return res

# Global in-memory managers for the active session
dataset_mgr = DatasetManager()
pipeline_mgr = MLPipelineManager()

# Load built-in sample datasets
def generate_sample_datasets():
    samples_dir = os.path.join(os.path.dirname(__file__), "samples")
    os.makedirs(samples_dir, exist_ok=True)

    try:
        from sklearn.datasets import load_iris, load_wine, load_diabetes
    except ImportError:
        print("[AutoML Backend] Note: Scikit-learn not yet installed. Run: pip install -r requirements.txt")
        return

    try:
        iris_path = os.path.join(samples_dir, "iris_classification.csv")
        if not os.path.exists(iris_path):
            iris = load_iris(as_frame=True)
            df_iris = iris.frame
            df_iris["species"] = iris.target_names[iris.target]
            df_iris = df_iris.drop(columns=["target"])
            df_iris.to_csv(iris_path, index=False)
    except Exception as e:
        print(f"Iris sample dataset error: {e}")

    # 2. Diabetes Disease Progression (Regression)
    try:
        diabetes_path = os.path.join(samples_dir, "diabetes_progression_regression.csv")
        if not os.path.exists(diabetes_path):
            diab = load_diabetes(as_frame=True)
            df_diab = diab.frame
            df_diab.rename(columns={"target": "disease_progression"}, inplace=True)
            df_diab.to_csv(diabetes_path, index=False)
    except Exception as e:
        print(f"Diabetes sample dataset error: {e}")

    # 3. Wine Recognition (Classification)
    try:
        wine_path = os.path.join(samples_dir, "wine_recognition_classification.csv")
        if not os.path.exists(wine_path):
            wine = load_wine(as_frame=True)
            df_wine = wine.frame
            df_wine["wine_class"] = [f"Class_{i}" for i in wine.target]
            df_wine = df_wine.drop(columns=["target"])
            df_wine.to_csv(wine_path, index=False)
    except Exception as e:
        print(f"Wine sample dataset error: {e}")

    # 4. California Housing Regression (100% offline generation, no HTTP requests)
    try:
        housing_path = os.path.join(samples_dir, "housing_prices_regression.csv")
        if not os.path.exists(housing_path):
            np.random.seed(42)
            n = 500
            med_inc = np.round(np.random.uniform(1.5, 9.5, n), 2)
            house_age = np.random.randint(1, 52, n)
            ave_rooms = np.round(np.random.uniform(3.0, 9.0, n), 2)
            ave_bedrooms = np.round(np.random.uniform(0.8, 2.5, n), 2)
            population = np.random.randint(200, 4500, n)
            ave_occup = np.round(np.random.uniform(1.8, 5.0, n), 2)
            latitude = np.round(np.random.uniform(32.5, 42.0, n), 2)
            longitude = np.round(np.random.uniform(-124.3, -114.3, n), 2)
            # Realistic synthetic house value ($100,000s) based on socio-economic predictors
            house_val = np.round(0.42 * med_inc + 0.015 * house_age + 0.12 * ave_rooms - 0.08 * ave_bedrooms + np.random.normal(0, 0.3, n), 2)
            house_val = np.clip(house_val, 0.5, 5.0)

            df_housing = pd.DataFrame({
                "MedInc": med_inc,
                "HouseAge": house_age,
                "AveRooms": ave_rooms,
                "AveBedrms": ave_bedrooms,
                "Population": population,
                "AveOccup": ave_occup,
                "Latitude": latitude,
                "Longitude": longitude,
                "HouseValue": house_val
            })
            df_housing.to_csv(housing_path, index=False)
    except Exception as e:
        print(f"Housing sample dataset error: {e}")


@app.route("/", methods=["GET"])
def index():
    return """
    <!DOCTYPE html>
    <html>
      <head>
        <title>AutoML Studio Backend</title>
        <meta charset="utf-8" />
      </head>
      <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #090d16; color: #f1f5f9; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0;">
        <div style="background: #111827; border: 1px solid #1f2937; padding: 40px; border-radius: 20px; text-align: center; max-width: 500px; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);">
          <div style="width: 50px; height: 50px; background: #1e3a8a; border-radius: 12px; margin: 0 auto 20px; display: flex; align-items: center; justify-content: center; font-size: 24px;">⚡</div>
          <h2 style="margin: 0 0 10px; font-size: 22px;">AutoML Studio Backend is Active!</h2>
          <p style="color: #94a3b8; font-size: 14px; margin-bottom: 25px; line-height: 1.5;">This is the Python Flask REST API server (port 5001). To use the interactive web app, open the main frontend:</p>
          <a href="http://localhost:3000" style="background: #2563eb; color: #ffffff; padding: 12px 28px; border-radius: 10px; text-decoration: none; font-weight: bold; font-size: 15px; display: inline-block;">Open AutoML Studio (Port 3000) →</a>
        </div>
      </body>
    </html>
    """, 200


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "AutoML Studio Backend"})


@app.route("/api/upload", methods=["POST"])
def upload_dataset():
    """Uploads a CSV file and returns dataset summary & preview."""
    if "file" not in request.files:
        return jsonify({"error": "No file part in upload request."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected."}), 400

    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV files (.csv) are currently supported."}), 400

    try:
        content = file.read()
        if len(content) > 50 * 1024 * 1024:
            return jsonify({"error": "File size exceeds maximum limit of 50MB."}), 400

        summary = dataset_mgr.load_csv(content, file.filename)
        return jsonify(summary), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/dataset/samples", methods=["GET"])
def list_sample_datasets():
    """Lists available built-in sample datasets."""
    samples = [
        {
            "id": "iris",
            "name": "Iris Flower Classification",
            "task": "classification",
            "target": "species",
            "description": "150 samples, 4 measurements (sepal & petal length/width) to classify 3 iris species."
        },
        {
            "id": "wine",
            "name": "Wine Recognition",
            "task": "classification",
            "target": "wine_class",
            "description": "178 samples, 13 chemical constituents to classify wines from 3 cultivars."
        },
        {
            "id": "diabetes",
            "name": "Diabetes Disease Progression",
            "task": "regression",
            "target": "disease_progression",
            "description": "442 patients, 10 baseline metrics (age, sex, bmi, bp, serum) to predict disease progression."
        },
        {
            "id": "housing",
            "name": "California Housing Prices",
            "task": "regression",
            "target": "HouseValue",
            "description": "500 districts, 8 census indicators to predict median home value ($100,000s)."
        }
    ]
    return jsonify(samples), 200


@app.route("/api/dataset/load-sample", methods=["POST"])
def load_sample_dataset():
    """Loads a built-in sample dataset by id."""
    data = request.get_json() or {}
    sample_id = data.get("sample_id", "iris")
    samples_dir = os.path.join(os.path.dirname(__file__), "samples")

    file_mapping = {
        "iris": ("iris_classification.csv", "iris_classification.csv"),
        "wine": ("wine_recognition_classification.csv", "wine_recognition_classification.csv"),
        "diabetes": ("diabetes_progression_regression.csv", "diabetes_progression_regression.csv"),
        "housing": ("housing_prices_regression.csv", "housing_prices_regression.csv")
    }

    if sample_id not in file_mapping:
        return jsonify({"error": f"Unknown sample dataset '{sample_id}'."}), 400

    filename, display_name = file_mapping[sample_id]
    filepath = os.path.join(samples_dir, filename)

    if not os.path.exists(filepath):
        generate_sample_datasets()

    try:
        df = pd.read_csv(filepath)
        summary = dataset_mgr.load_dataframe(df, display_name)
        return jsonify(summary), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dataset/summary", methods=["GET"])
def get_dataset_summary():
    """Gets current dataset summary and column stats."""
    try:
        summary = dataset_mgr.get_summary()
        return jsonify(summary), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/dataset/configure", methods=["POST"])
def configure_dataset():
    """Configures target, features, task type, split ratio, and random state."""
    if dataset_mgr.df is None:
        return jsonify({"error": "No dataset currently loaded. Please upload a CSV first."}), 400

    data = request.get_json() or {}
    target = data.get("target_column")
    features = data.get("feature_columns", [])
    task_type = data.get("task_type", "classification")
    test_size = float(data.get("test_size", 0.2))
    random_seed = int(data.get("random_seed", 42))
    imputation = data.get("imputation_strategy", "median")
    scaling = bool(data.get("scaling", True))

    try:
        result = pipeline_mgr.configure(
            df=dataset_mgr.df,
            target_column=target,
            feature_columns=features,
            task_type=task_type,
            test_size=test_size,
            random_seed=random_seed,
            imputation_strategy=imputation,
            scaling=scaling
        )
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/dataset/visualizations", methods=["GET"])
def get_dataset_visualizations():
    """Generates and returns dataset exploratory charts (base64 PNGs)."""
    target = request.args.get("target") or pipeline_mgr.config.get("target_column")
    try:
        charts = dataset_mgr.generate_visualizations(target_col=target)
        return jsonify(charts), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/model/train", methods=["POST"])
def train_model():
    """Trains a baseline model for the configured algorithm."""
    data = request.get_json() or {}
    algorithm = data.get("algorithm")

    if not algorithm:
        return jsonify({"error": "No algorithm specified."}), 400

    try:
        metrics = pipeline_mgr.train_baseline(algorithm)
        return jsonify(metrics), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/model/tune", methods=["POST"])
def fine_tune_model():
    """Fine-tunes model hyperparameters using GridSearchCV or RandomizedSearchCV."""
    data = request.get_json() or {}
    algorithm = data.get("algorithm")
    search_type = data.get("search_type", "grid")
    cv_folds = int(data.get("cv_folds", 3))
    n_iter = int(data.get("n_iter", 10))

    if not algorithm:
        return jsonify({"error": "No algorithm specified for tuning."}), 400

    try:
        result = pipeline_mgr.fine_tune(
            algorithm=algorithm,
            search_type=search_type,
            cv_folds=cv_folds,
            n_iter=n_iter
        )
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/model/metrics", methods=["GET"])
def get_model_metrics():
    """Retrieves metrics for the currently active fitted model."""
    if not pipeline_mgr.last_metrics:
        return jsonify({"error": "No model has been trained yet."}), 400
    return jsonify(pipeline_mgr.last_metrics), 200


@app.route("/api/model/visualizations", methods=["GET"])
def get_model_visualizations():
    """Retrieves evaluation charts for the active model."""
    charts = pipeline_mgr.last_metrics.get("charts", {})
    return jsonify(charts), 200


@app.route("/api/model/predict", methods=["POST"])
def predict():
    """Predicts output for a single input record."""
    data = request.get_json() or {}
    inputs = data.get("inputs", {})

    try:
        res = pipeline_mgr.predict_single(inputs)
        return jsonify(res), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/model/batch-predict", methods=["POST"])
def batch_predict():
    """Runs batch predictions for an uploaded CSV file."""
    if "file" not in request.files:
        return jsonify({"error": "No batch CSV file provided."}), 400

    file = request.files["file"]
    try:
        content = file.read()
        res = pipeline_mgr.batch_predict(content)
        return jsonify(res), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/model/save", methods=["POST"])
def save_model():
    """Saves the current pipeline to Joblib and registers in SQLite."""
    data = request.get_json() or {}
    name = data.get("name", "My Model")
    dataset_name = dataset_mgr.filename
    rows = dataset_mgr.df.shape[0] if dataset_mgr.df is not None else 0
    cols = dataset_mgr.df.shape[1] if dataset_mgr.df is not None else 0

    try:
        res = pipeline_mgr.save_model(name, dataset_name, rows, cols)
        return jsonify(res), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/models", methods=["GET"])
def list_models():
    """Lists all saved models with metadata and metrics."""
    try:
        models = pipeline_mgr.list_saved_models()
        return jsonify(models), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/model/load", methods=["POST"])
def load_model():
    """Loads a previously saved model into active memory."""
    data = request.get_json() or {}
    model_id = data.get("model_id")

    if not model_id:
        return jsonify({"error": "No model_id provided."}), 400

    try:
        res = pipeline_mgr.load_model(model_id)
        return jsonify(res), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/models/<model_id>", methods=["DELETE"])
def delete_model(model_id):
    """Deletes a saved model."""
    success = pipeline_mgr.delete_model(model_id)
    if success:
        return jsonify({"status": "deleted", "model_id": model_id}), 200
    else:
        return jsonify({"error": "Model not found."}), 404


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5001, help="Port to run Flask app on")
    args = parser.parse_args()

    # Pre-generate sample datasets on startup
    try:
        generate_sample_datasets()
    except Exception as e:
        print(f"Sample dataset initialization note: {e}")

    print(f"AutoML Studio Flask backend starting on http://127.0.0.1:{args.port}...")
    app.run(host="127.0.0.1", port=args.port, debug=False)
