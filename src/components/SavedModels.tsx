import React, { useState, useEffect } from 'react';
import {
  Database,
  Trash2,
  FolderOpen,
  Calendar,
  Layers,
  Award,
  Sparkles,
  TrendingUp,
  AlertCircle,
  CheckCircle2,
  Download,
  Loader2,
  ArrowRight
} from 'lucide-react';
import { SavedModelRecord } from '../types/automl';
import { fetchSavedModels, loadSavedModel, deleteSavedModel } from '../services/api';

interface SavedModelsProps {
  onModelLoaded: (loadedData: any) => void;
  onProceedToPredict: () => void;
  activeModelId?: string;
}

export const SavedModels: React.FC<SavedModelsProps> = ({
  onModelLoaded,
  onProceedToPredict,
  activeModelId,
}) => {
  const [models, setModels] = useState<SavedModelRecord[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    setIsLoading(true);
    setErrorMessage(null);
    try {
      const data = await fetchSavedModels();
      setModels(data);
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to fetch saved models from SQLite.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleLoad = async (model: SavedModelRecord) => {
    setLoadingId(model.id);
    setErrorMessage(null);
    try {
      const res = await loadSavedModel(model.id);
      setStatusMessage(`Model '${model.name}' loaded successfully into active memory!`);
      onModelLoaded(res);
    } catch (err: any) {
      setErrorMessage(err.message || `Failed to load model '${model.name}'.`);
    } finally {
      setLoadingId(null);
    }
  };

  const handleDelete = async (id: string, name: string) => {
    if (!confirm(`Are you sure you want to delete model '${name}'?`)) return;

    setDeletingId(id);
    try {
      await deleteSavedModel(id);
      setModels((prev) => prev.filter((m) => m.id !== id));
      setStatusMessage(`Model '${name}' deleted.`);
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to delete model.');
    } finally {
      setDeletingId(null);
    }
  };

  const exportSummaryJSON = () => {
    const jsonStr = JSON.stringify(models, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `automl_experiment_history_${Date.now()}.json`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {statusMessage && (
        <div className="p-4 rounded-xl bg-emerald-950/40 border border-emerald-800/60 text-emerald-200 text-sm flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0" />
            <span>{statusMessage}</span>
          </div>
          <button
            onClick={() => setStatusMessage(null)}
            className="text-xs text-emerald-300 font-bold"
          >
            Dismiss
          </button>
        </div>
      )}

      {errorMessage && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800/60 text-rose-200 text-sm flex items-start space-x-3">
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div className="flex-1">{errorMessage}</div>
        </div>
      )}

      {/* Header card with export button */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <Database className="w-5 h-5 text-blue-400" />
            Model Registry & Experiment Store
          </h3>
          <p className="text-xs text-slate-400 mt-1">
            Persisted Joblib pipeline artifacts and experiment metadata stored in local SQLite database.
          </p>
        </div>

        {models.length > 0 && (
          <button
            onClick={exportSummaryJSON}
            className="flex items-center space-x-2 px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700 transition"
          >
            <Download className="w-4 h-4" />
            <span>Export Registry JSON</span>
          </button>
        )}
      </div>

      {/* Models List */}
      {isLoading ? (
        <div className="p-16 text-center text-slate-400 space-y-3">
          <Loader2 className="w-8 h-8 text-blue-500 animate-spin mx-auto" />
          <p className="text-xs font-medium">Loading saved model registry from SQLite...</p>
        </div>
      ) : models.length === 0 ? (
        <div className="p-16 text-center bg-slate-900 border border-slate-800 rounded-2xl space-y-3">
          <Database className="w-10 h-10 text-slate-700 mx-auto" />
          <h4 className="text-sm font-semibold text-slate-300">No Saved Models in Registry</h4>
          <p className="text-xs text-slate-500 max-w-sm mx-auto">
            Train and evaluate a Scikit-learn model, then click "Save Model Artifact" to store it permanently for predictions and benchmarking.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {models.map((m) => {
            const isActive = activeModelId === m.id;
            const isClf = m.task_type === 'classification';

            return (
              <div
                key={m.id}
                className={`p-6 rounded-2xl border transition-all ${
                  isActive
                    ? 'bg-slate-900 border-blue-500/80 shadow-md shadow-blue-500/10'
                    : 'bg-slate-900/90 border-slate-800 hover:border-slate-700'
                }`}
              >
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                  <div className="space-y-1.5">
                    <div className="flex items-center space-x-2.5">
                      <h4 className="text-base font-bold text-white tracking-tight">{m.name}</h4>
                      {isActive && (
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-blue-950 text-blue-400 border border-blue-800/80 uppercase">
                          Active in Memory
                        </span>
                      )}
                      <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-slate-800 text-slate-300 uppercase">
                        {m.task_type}
                      </span>
                    </div>

                    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-400">
                      <span>Algorithm: <strong className="text-slate-200">{m.algorithm}</strong></span>
                      <span>Target: <strong className="text-indigo-400">{m.target_column}</strong></span>
                      <span>Features: <strong className="text-slate-200">{m.feature_names.length}</strong></span>
                      {m.dataset_name && <span>Dataset: {m.dataset_name}</span>}
                      <span className="flex items-center gap-1 text-slate-500">
                        <Calendar className="w-3.5 h-3.5" />
                        {m.created_at}
                      </span>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center space-x-2 self-start md:self-center shrink-0">
                    <button
                      onClick={() => handleLoad(m)}
                      disabled={loadingId !== null}
                      className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center space-x-1.5 transition ${
                        isActive
                          ? 'bg-slate-800 text-slate-400 cursor-default'
                          : 'bg-blue-600 hover:bg-blue-500 text-white shadow-sm'
                      }`}
                    >
                      {loadingId === m.id ? (
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      ) : (
                        <FolderOpen className="w-3.5 h-3.5" />
                      )}
                      <span>{isActive ? 'Active' : 'Load Pipeline'}</span>
                    </button>

                    <button
                      onClick={() => handleDelete(m.id, m.name)}
                      disabled={deletingId === m.id}
                      title="Delete model from disk"
                      className="p-2 rounded-lg bg-slate-800 hover:bg-rose-950 hover:text-rose-400 text-slate-400 transition"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>

                {/* Score Badges */}
                <div className="mt-4 pt-4 border-t border-slate-800/80 grid grid-cols-2 sm:grid-cols-4 gap-3">
                  {isClf ? (
                    <>
                      <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800/80">
                        <span className="text-[10px] text-slate-500 uppercase font-semibold">Test Accuracy</span>
                        <div className="text-sm font-bold text-emerald-400">
                          {m.metrics?.accuracy !== undefined ? `${(m.metrics.accuracy * 100).toFixed(2)}%` : 'N/A'}
                        </div>
                      </div>
                      <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800/80">
                        <span className="text-[10px] text-slate-500 uppercase font-semibold">Weighted F1</span>
                        <div className="text-sm font-bold text-blue-400">
                          {m.metrics?.f1_score !== undefined ? `${(m.metrics.f1_score * 100).toFixed(2)}%` : 'N/A'}
                        </div>
                      </div>
                      <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800/80">
                        <span className="text-[10px] text-slate-500 uppercase font-semibold">ROC-AUC</span>
                        <div className="text-sm font-bold text-purple-400">
                          {m.metrics?.roc_auc ? m.metrics.roc_auc.toFixed(3) : 'Discrete'}
                        </div>
                      </div>
                      <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800/80">
                        <span className="text-[10px] text-slate-500 uppercase font-semibold">Training Time</span>
                        <div className="text-sm font-bold text-slate-300">
                          {m.metrics?.training_time_sec}s
                        </div>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800/80">
                        <span className="text-[10px] text-slate-500 uppercase font-semibold">R² Score</span>
                        <div className="text-sm font-bold text-emerald-400">
                          {m.metrics?.r2 !== undefined ? m.metrics.r2.toFixed(4) : 'N/A'}
                        </div>
                      </div>
                      <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800/80">
                        <span className="text-[10px] text-slate-500 uppercase font-semibold">RMSE</span>
                        <div className="text-sm font-bold text-blue-400">
                          {m.metrics?.rmse !== undefined ? m.metrics.rmse.toFixed(4) : 'N/A'}
                        </div>
                      </div>
                      <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800/80">
                        <span className="text-[10px] text-slate-500 uppercase font-semibold">MAE</span>
                        <div className="text-sm font-bold text-purple-400">
                          {m.metrics?.mae !== undefined ? m.metrics.mae.toFixed(4) : 'N/A'}
                        </div>
                      </div>
                      <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800/80">
                        <span className="text-[10px] text-slate-500 uppercase font-semibold">Training Time</span>
                        <div className="text-sm font-bold text-slate-300">
                          {m.metrics?.training_time_sec}s
                        </div>
                      </div>
                    </>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
