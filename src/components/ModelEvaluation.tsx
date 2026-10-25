import React, { useState } from 'react';
import {
  BarChart3,
  Award,
  TrendingUp,
  Save,
  CheckCircle2,
  AlertTriangle,
  Info,
  ArrowRight,
  Sparkles,
  PieChart,
  Activity,
  Layers,
  Loader2
} from 'lucide-react';
import { ModelMetrics } from '../types/automl';
import { saveModel } from '../services/api';

interface ModelEvaluationProps {
  metrics: ModelMetrics;
  onProceedToPredict: () => void;
  onModelSaved?: (modelId: string, name: string) => void;
}

export const ModelEvaluation: React.FC<ModelEvaluationProps> = ({
  metrics,
  onProceedToPredict,
  onModelSaved,
}) => {
  const [modelName, setModelName] = useState<string>(
    `${metrics.algorithm.replace(/\s+/g, '_')}_${metrics.is_tuned ? 'Tuned' : 'Baseline'}`
  );
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!modelName.trim()) return;

    setIsSaving(true);
    setSaveError(null);
    try {
      const res = await saveModel(modelName.trim());
      setSaveSuccess(`Pipeline and estimator successfully persisted via Joblib (ID: ${res.model_id}).`);
      if (onModelSaved) {
        onModelSaved(res.model_id, modelName.trim());
      }
    } catch (err: any) {
      setSaveError(err.message || 'Failed to save model artifact.');
    } finally {
      setIsSaving(false);
    }
  };

  const isClassification = metrics.task_type === 'classification';

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Save Notification Alerts */}
      {saveSuccess && (
        <div className="p-4 rounded-xl bg-emerald-950/40 border border-emerald-800/60 text-emerald-200 text-sm flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0" />
            <span>{saveSuccess}</span>
          </div>
          <button
            onClick={() => setSaveSuccess(null)}
            className="text-xs text-emerald-300 font-bold ml-3"
          >
            Dismiss
          </button>
        </div>
      )}

      {saveError && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800/60 text-rose-200 text-sm flex items-start space-x-3">
          <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div className="flex-1">
            <span className="font-semibold text-rose-100">Save Error: </span>
            {saveError}
          </div>
        </div>
      )}

      {/* Model Header Card with Quick Save & Predict buttons */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-xl font-bold text-white tracking-tight">{metrics.algorithm}</h2>
            <span
              className={`text-xs px-2.5 py-0.5 rounded-full font-bold uppercase ${
                metrics.is_tuned
                  ? 'bg-amber-950 text-amber-300 border border-amber-800/60'
                  : 'bg-blue-950 text-blue-300 border border-blue-800/60'
              }`}
            >
              {metrics.is_tuned ? `Tuned (${metrics.search_type?.toUpperCase()})` : 'Baseline Fit'}
            </span>
            <span className="text-xs px-2 py-0.5 rounded font-mono bg-slate-800 text-slate-300">
              {metrics.training_time_sec}s compute
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Evaluated rigorously on held-out test split (unseen during preprocessing fit and hyperparameter cross-validation).
          </p>
        </div>

        <div className="flex items-center space-x-3">
          <button
            onClick={onProceedToPredict}
            className="px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-md shadow-indigo-600/20 transition flex items-center space-x-2 shrink-0"
          >
            <span>Test Inference / Predict</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Main Metric Cards */}
      {isClassification ? (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <span className="text-xs text-slate-400 font-semibold uppercase tracking-wider block">
              Test Accuracy
            </span>
            <div className="text-2xl font-black text-emerald-400 mt-1">
              {metrics.accuracy !== undefined ? `${(metrics.accuracy * 100).toFixed(2)}%` : 'N/A'}
            </div>
            <span className="text-[11px] text-slate-500 mt-0.5 block">
              Fraction of correct predictions
            </span>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <span className="text-xs text-slate-400 font-semibold uppercase tracking-wider block">
              Weighted F1-Score
            </span>
            <div className="text-2xl font-black text-blue-400 mt-1">
              {metrics.f1_score !== undefined ? `${(metrics.f1_score * 100).toFixed(2)}%` : 'N/A'}
            </div>
            <span className="text-[11px] text-slate-500 mt-0.5 block">
              Harmonic mean (macro: {((metrics.f1_macro ?? 0) * 100).toFixed(1)}%)
            </span>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <span className="text-xs text-slate-400 font-semibold uppercase tracking-wider block">
              Weighted Precision
            </span>
            <div className="text-2xl font-black text-purple-400 mt-1">
              {metrics.precision !== undefined ? `${(metrics.precision * 100).toFixed(2)}%` : 'N/A'}
            </div>
            <span className="text-[11px] text-slate-500 mt-0.5 block">
              Low false alarm rate
            </span>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <span className="text-xs text-slate-400 font-semibold uppercase tracking-wider block">
              ROC-AUC Score
            </span>
            <div className="text-2xl font-black text-indigo-400 mt-1">
              {metrics.roc_auc !== null && metrics.roc_auc !== undefined
                ? metrics.roc_auc.toFixed(3)
                : 'N/A'}
            </div>
            <span className="text-[11px] text-slate-500 mt-0.5 block">
              {metrics.roc_note ? 'Discrete score only' : 'Area under ROC curve'}
            </span>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <span className="text-xs text-slate-400 font-semibold uppercase tracking-wider block">
              R² Score (Determination)
            </span>
            <div className="text-2xl font-black text-emerald-400 mt-1">
              {metrics.r2 !== undefined ? metrics.r2.toFixed(4) : 'N/A'}
            </div>
            <span className="text-[11px] text-slate-500 mt-0.5 block">
              Variance explained (1.0 = ideal)
            </span>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <span className="text-xs text-slate-400 font-semibold uppercase tracking-wider block">
              RMSE (Root Mean Squared)
            </span>
            <div className="text-2xl font-black text-blue-400 mt-1">
              {metrics.rmse !== undefined ? metrics.rmse.toFixed(4) : 'N/A'}
            </div>
            <span className="text-[11px] text-slate-500 mt-0.5 block">
              Error in target units
            </span>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <span className="text-xs text-slate-400 font-semibold uppercase tracking-wider block">
              MAE (Mean Absolute)
            </span>
            <div className="text-2xl font-black text-purple-400 mt-1">
              {metrics.mae !== undefined ? metrics.mae.toFixed(4) : 'N/A'}
            </div>
            <span className="text-[11px] text-slate-500 mt-0.5 block">
              Mean absolute deviation
            </span>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <span className="text-xs text-slate-400 font-semibold uppercase tracking-wider block">
              MSE (Mean Squared)
            </span>
            <div className="text-2xl font-black text-indigo-400 mt-1">
              {metrics.mse !== undefined ? metrics.mse.toFixed(4) : 'N/A'}
            </div>
            <span className="text-[11px] text-slate-500 mt-0.5 block">
              Squared loss penalty
            </span>
          </div>
        </div>
      )}

      {/* Cross-Validation Distinction Banner */}
      <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-xs text-slate-300 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Info className="w-4 h-4 text-blue-400 shrink-0" />
          <span>
            <strong>Validation Methodology: </strong>
            {metrics.is_tuned ? (
              <>
                Hyperparameters were selected via{' '}
                <strong>{metrics.search_type?.toUpperCase()} cross-validation</strong> on the training set (Best CV Score:{' '}
                <strong className="text-amber-400">{metrics.best_cv_score}</strong>). Final displayed metrics above were evaluated on the held-out test split.
              </>
            ) : (
              <>
                Baseline model was trained on the training partition and validated across folds (Mean CV Score:{' '}
                <strong className="text-blue-400">{metrics.cv_score_mean ?? 'N/A'}</strong>).
              </>
            )}
          </span>
        </div>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Classification Charts */}
        {isClassification && (
          <>
            {metrics.charts?.confusion_matrix && (
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 flex flex-col items-center">
                <div className="flex items-center justify-between w-full mb-3">
                  <h4 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                    <Layers className="w-4 h-4 text-blue-400" />
                    Confusion Matrix Heatmap
                  </h4>
                  <span className="text-[10px] text-slate-500">True vs. Predicted</span>
                </div>
                <img
                  src={metrics.charts.confusion_matrix}
                  alt="Confusion Matrix"
                  className="rounded-xl max-h-80 object-contain w-full"
                />
              </div>
            )}

            {metrics.charts?.roc_curve ? (
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 flex flex-col items-center">
                <div className="flex items-center justify-between w-full mb-3">
                  <h4 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                    <Activity className="w-4 h-4 text-indigo-400" />
                    Receiver Operating Characteristic (ROC)
                  </h4>
                  <span className="text-[10px] text-slate-500">Sensitivity vs 1-Specificity</span>
                </div>
                <img
                  src={metrics.charts.roc_curve}
                  alt="ROC Curve"
                  className="rounded-xl max-h-80 object-contain w-full"
                />
              </div>
            ) : (
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 flex flex-col justify-center text-center">
                <Info className="w-8 h-8 text-slate-500 mx-auto mb-2" />
                <h4 className="text-sm font-semibold text-slate-300">ROC Curve Note</h4>
                <p className="text-xs text-slate-400 mt-1 max-w-sm mx-auto">
                  {metrics.roc_note ||
                    'ROC curve is plotted for binary classification when calibrated probability estimates are available from the estimator.'}
                </p>
              </div>
            )}
          </>
        )}

        {/* Regression Charts */}
        {!isClassification && (
          <>
            {metrics.charts?.actual_vs_predicted && (
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 flex flex-col items-center">
                <div className="flex items-center justify-between w-full mb-3">
                  <h4 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-emerald-400" />
                    Actual vs. Predicted Scatter
                  </h4>
                  <span className="text-[10px] text-slate-500">y = x Target Line</span>
                </div>
                <img
                  src={metrics.charts.actual_vs_predicted}
                  alt="Actual vs Predicted"
                  className="rounded-xl max-h-80 object-contain w-full"
                />
              </div>
            )}

            {metrics.charts?.residuals_plot && (
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 flex flex-col items-center">
                <div className="flex items-center justify-between w-full mb-3">
                  <h4 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                    <Activity className="w-4 h-4 text-purple-400" />
                    Residual Error Plot
                  </h4>
                  <span className="text-[10px] text-slate-500">Error vs. Fitted</span>
                </div>
                <img
                  src={metrics.charts.residuals_plot}
                  alt="Residual Plot"
                  className="rounded-xl max-h-80 object-contain w-full"
                />
              </div>
            )}

            {metrics.charts?.residual_distribution && (
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 flex flex-col items-center md:col-span-2">
                <div className="flex items-center justify-between w-full mb-3">
                  <h4 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                    <BarChart3 className="w-4 h-4 text-emerald-400" />
                    Residual Error Distribution (Normality Check)
                  </h4>
                  <span className="text-[10px] text-slate-500">Zero-centered Bell Curve</span>
                </div>
                <img
                  src={metrics.charts.residual_distribution}
                  alt="Residual Distribution"
                  className="rounded-xl max-h-72 object-contain"
                />
              </div>
            )}
          </>
        )}
      </div>

      {/* Persistence / Model Save Form */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <Save className="w-5 h-5 text-emerald-400" />
            Persist Pipeline Artifact (Joblib + SQLite)
          </h3>
          <span className="text-xs text-slate-400 font-mono">Serialized Artifact</span>
        </div>
        <p className="text-xs text-slate-400 mb-4">
          Save the entire fitted pipeline—including column imputation, OneHotEncoder, StandardScaler, and the trained estimator—so it can be reloaded for predictions and compared across experiments.
        </p>

        <form onSubmit={handleSave} className="flex flex-col sm:flex-row items-center gap-3">
          <input
            type="text"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            placeholder="Assign a descriptive model name..."
            className="flex-1 w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-emerald-500"
          />
          <button
            type="submit"
            disabled={isSaving || !modelName.trim()}
            className="w-full sm:w-auto px-6 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-xs font-bold uppercase tracking-wider shadow-md shadow-emerald-600/20 transition flex items-center justify-center space-x-2 shrink-0"
          >
            {isSaving ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Serializing Artifact...</span>
              </>
            ) : (
              <>
                <Save className="w-4 h-4" />
                <span>Save Model Artifact</span>
              </>
            )}
          </button>
        </form>
      </div>
    </div>
  );
};
