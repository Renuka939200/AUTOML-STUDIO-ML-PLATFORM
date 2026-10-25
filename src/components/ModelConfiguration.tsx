import React, { useState, useEffect } from 'react';
import {
  SlidersHorizontal,
  Target,
  CheckSquare,
  Square,
  ShieldCheck,
  AlertCircle,
  HelpCircle,
  Layers,
  ArrowRight,
  Loader2,
  CheckCircle2
} from 'lucide-react';
import { DatasetSummary, ModelConfiguration as IModelConfig, ConfigureResponse } from '../types/automl';
import { configureDataset } from '../services/api';

interface ModelConfigurationProps {
  dataset: DatasetSummary;
  initialConfig?: IModelConfig | null;
  onConfigured: (config: IModelConfig, response: ConfigureResponse) => void;
  onProceedToTrain: () => void;
}

export const ModelConfiguration: React.FC<ModelConfigurationProps> = ({
  dataset,
  initialConfig,
  onConfigured,
  onProceedToTrain,
}) => {
  // Heuristic default target: last column or column named 'target', 'label', 'class', 'species', etc.
  const defaultTarget = () => {
    const candidateNames = ['target', 'label', 'class', 'species', 'outcome', 'disease_progression', 'housevalue', 'price', 'charges'];
    for (const c of dataset.columns) {
      if (candidateNames.includes(c.name.toLowerCase())) {
        return c.name;
      }
    }
    return dataset.columns[dataset.columns.length - 1].name;
  };

  const [targetColumn, setTargetColumn] = useState<string>(initialConfig?.target_column || defaultTarget());
  const [featureColumns, setFeatureColumns] = useState<string[]>(
    initialConfig?.feature_columns || dataset.columns.map((c) => c.name).filter((n) => n !== targetColumn)
  );
  
  // Heuristic default task type based on target
  const defaultTask = (targetName: string): 'classification' | 'regression' => {
    const colStat = dataset.columns.find((c) => c.name === targetName);
    if (!colStat) return 'classification';
    if (!colStat.is_numerical || colStat.unique_count <= 10) {
      return 'classification';
    }
    return 'regression';
  };

  const [taskType, setTaskType] = useState<'classification' | 'regression'>(
    initialConfig?.task_type || defaultTask(targetColumn)
  );
  const [testSize, setTestSize] = useState<number>(initialConfig?.test_size || 0.2);
  const [randomSeed, setRandomSeed] = useState<number>(initialConfig?.random_seed || 42);
  const [imputationStrategy, setImputationStrategy] = useState<'median' | 'mean'>(
    (initialConfig?.imputation_strategy as any) || 'median'
  );
  const [scaling, setScaling] = useState<boolean>(initialConfig?.scaling ?? true);

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [configResult, setConfigResult] = useState<ConfigureResponse | null>(null);

  // Update features and task when target column changes
  const handleTargetChange = (newTarget: string) => {
    setTargetColumn(newTarget);
    // Remove new target from selected features
    setFeatureColumns((prev) => prev.filter((col) => col !== newTarget));
    setTaskType(defaultTask(newTarget));
    setConfigResult(null);
  };

  const handleToggleFeature = (colName: string) => {
    if (colName === targetColumn) return;
    setFeatureColumns((prev) =>
      prev.includes(colName) ? prev.filter((c) => c !== colName) : [...prev, colName]
    );
    setConfigResult(null);
  };

  const handleSelectAllFeatures = () => {
    setFeatureColumns(dataset.columns.map((c) => c.name).filter((c) => c !== targetColumn));
    setConfigResult(null);
  };

  const handleDeselectAllFeatures = () => {
    setFeatureColumns([]);
    setConfigResult(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (featureColumns.length === 0) {
      setErrorMessage('You must select at least one input feature column.');
      return;
    }
    if (featureColumns.includes(targetColumn)) {
      setErrorMessage('Target variable cannot be included as an input feature.');
      return;
    }

    setIsSubmitting(true);
    setErrorMessage(null);

    const config: IModelConfig = {
      target_column: targetColumn,
      feature_columns: featureColumns,
      task_type: taskType,
      test_size: testSize,
      random_seed: randomSeed,
      imputation_strategy: imputationStrategy,
      scaling: scaling,
    };

    try {
      const res = await configureDataset(config);
      setConfigResult(res);
      onConfigured(config, res);
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to configure ML pipeline.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const selectedTargetStat = dataset.columns.find((c) => c.name === targetColumn);

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {errorMessage && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800/60 text-rose-200 text-sm flex items-start space-x-3">
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div className="flex-1">
            <span className="font-semibold text-rose-100">Configuration Check: </span>
            {errorMessage}
          </div>
        </div>
      )}

      {configResult && (
        <div className="p-4 rounded-xl bg-emerald-950/40 border border-emerald-800/60 text-emerald-200 text-sm flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0" />
            <div>
              <span className="font-semibold text-white">Pipeline Preprocessing Configured: </span>
              {configResult.train_samples} training rows, {configResult.test_samples} test rows ({Math.round(testSize * 100)}% split). {configResult.feature_count} features ready with leak-free transformers.
            </div>
          </div>
          <button
            onClick={onProceedToTrain}
            className="flex items-center space-x-1.5 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold shadow-sm transition"
          >
            <span>Proceed to Training</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: Target & Task Type */}
          <div className="lg:col-span-5 space-y-6">
            {/* Target Column Selector */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <Target className="w-5 h-5 text-blue-400" />
                  Target Variable (y)
                </h3>
                <span className="text-xs text-slate-400 font-mono">Dependent Column</span>
              </div>

              <div className="space-y-3">
                <label className="text-xs text-slate-400 font-medium block">
                  Select the column you want the machine learning model to predict:
                </label>
                <select
                  value={targetColumn}
                  onChange={(e) => handleTargetChange(e.target.value)}
                  className="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm text-white font-medium focus:outline-none focus:border-blue-500"
                >
                  {dataset.columns.map((c) => (
                    <option key={c.name} value={c.name}>
                      {c.name} ({c.is_numerical ? 'Numeric' : 'Categorical'} • {c.unique_count} unique)
                    </option>
                  ))}
                </select>

                {selectedTargetStat && (
                  <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80 text-xs space-y-1.5 mt-3">
                    <div className="flex justify-between text-slate-400">
                      <span>Type:</span>
                      <span className="text-white font-mono">{selectedTargetStat.dtype}</span>
                    </div>
                    <div className="flex justify-between text-slate-400">
                      <span>Distinct Classes / Values:</span>
                      <span className="text-white font-semibold">{selectedTargetStat.unique_count}</span>
                    </div>
                    <div className="flex justify-between text-slate-400">
                      <span>Missing Cells:</span>
                      <span className={selectedTargetStat.missing_count > 0 ? 'text-rose-400' : 'text-emerald-400'}>
                        {selectedTargetStat.missing_count} (rows auto-filtered)
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Task Type Card */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <Layers className="w-5 h-5 text-purple-400" />
                  Machine Learning Task
                </h3>
                <span className="text-xs text-slate-400">Problem Type</span>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div
                  onClick={() => {
                    setTaskType('classification');
                    setConfigResult(null);
                  }}
                  className={`p-4 rounded-xl border cursor-pointer transition flex flex-col justify-between ${
                    taskType === 'classification'
                      ? 'bg-purple-950/40 border-purple-500 text-purple-200'
                      : 'bg-slate-800/40 border-slate-700/60 hover:bg-slate-800 text-slate-400'
                  }`}
                >
                  <div>
                    <span className="text-xs font-bold uppercase tracking-wider block text-white mb-1">
                      Classification
                    </span>
                    <p className="text-[11px] leading-relaxed">
                      Predict discrete classes or labels (e.g., categories, disease status, churn).
                    </p>
                  </div>
                  <div className="mt-3 text-[10px] font-semibold text-purple-400">
                    Metrics: Accuracy, F1, ROC
                  </div>
                </div>

                <div
                  onClick={() => {
                    setTaskType('regression');
                    setConfigResult(null);
                  }}
                  className={`p-4 rounded-xl border cursor-pointer transition flex flex-col justify-between ${
                    taskType === 'regression'
                      ? 'bg-emerald-950/40 border-emerald-500 text-emerald-200'
                      : 'bg-slate-800/40 border-slate-700/60 hover:bg-slate-800 text-slate-400'
                  }`}
                >
                  <div>
                    <span className="text-xs font-bold uppercase tracking-wider block text-white mb-1">
                      Regression
                    </span>
                    <p className="text-[11px] leading-relaxed">
                      Predict continuous numerical values (e.g., prices, scores, quantities).
                    </p>
                  </div>
                  <div className="mt-3 text-[10px] font-semibold text-emerald-400">
                    Metrics: R², RMSE, MAE
                  </div>
                </div>
              </div>
            </div>

            {/* Split & Preprocessing Settings */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <SlidersHorizontal className="w-5 h-5 text-indigo-400" />
                Validation Split & Preprocessing
              </h3>

              {/* Train / Test Split Slider */}
              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400 font-medium">Train-Test Split:</span>
                  <span className="text-white font-mono font-semibold">
                    {Math.round((1 - testSize) * 100)}% Train / {Math.round(testSize * 100)}% Test
                  </span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="0.4"
                  step="0.05"
                  value={testSize}
                  onChange={(e) => {
                    setTestSize(parseFloat(e.target.value));
                    setConfigResult(null);
                  }}
                  className="w-full accent-blue-500 bg-slate-800 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-500">
                  <span>90% Train (10% Test)</span>
                  <span>80% Train (20% Test)</span>
                  <span>60% Train (40% Test)</span>
                </div>
              </div>

              {/* Random Seed */}
              <div className="space-y-1.5">
                <label className="text-xs text-slate-400 font-medium block">
                  Random Seed (Reproducibility):
                </label>
                <input
                  type="number"
                  value={randomSeed}
                  onChange={(e) => {
                    setRandomSeed(parseInt(e.target.value, 10) || 42);
                    setConfigResult(null);
                  }}
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white font-mono focus:outline-none focus:border-blue-500"
                />
              </div>

              {/* Imputation Strategy */}
              <div className="space-y-1.5">
                <label className="text-xs text-slate-400 font-medium block">
                  Numeric Missing Imputer:
                </label>
                <select
                  value={imputationStrategy}
                  onChange={(e) => {
                    setImputationStrategy(e.target.value as any);
                    setConfigResult(null);
                  }}
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
                >
                  <option value="median">Median (robust to statistical outliers)</option>
                  <option value="mean">Mean (standard Gaussian distribution)</option>
                </select>
                <span className="text-[10px] text-slate-500 block">
                  * Categorical columns automatically use most-frequent (mode) imputation.
                </span>
              </div>

              {/* Feature Scaling Toggle */}
              <div className="pt-2 flex items-center justify-between">
                <div>
                  <span className="text-xs font-semibold text-slate-200 block">
                    StandardScaler (Zero-Mean, Unit Variance)
                  </span>
                  <span className="text-[11px] text-slate-400">
                    Recommended for SVM, KNN, Logistic Regression, SVR.
                  </span>
                </div>
                <input
                  type="checkbox"
                  checked={scaling}
                  onChange={(e) => {
                    setScaling(e.target.checked);
                    setConfigResult(null);
                  }}
                  className="w-4 h-4 accent-blue-600 rounded cursor-pointer"
                />
              </div>
            </div>
          </div>

          {/* Right Column: Feature Selection */}
          <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl p-6 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <CheckSquare className="w-5 h-5 text-emerald-400" />
                  Select Input Features (X)
                </h3>
                <div className="flex space-x-2 text-xs">
                  <button
                    type="button"
                    onClick={handleSelectAllFeatures}
                    className="text-blue-400 hover:text-blue-300 font-semibold"
                  >
                    Select All
                  </button>
                  <span className="text-slate-600">•</span>
                  <button
                    type="button"
                    onClick={handleDeselectAllFeatures}
                    className="text-slate-400 hover:text-slate-300"
                  >
                    Clear All
                  </button>
                </div>
              </div>

              <p className="text-xs text-slate-400 mb-4">
                Choose the predictor columns. Target column is automatically isolated to prevent data leakage.
              </p>

              <div className="border border-slate-800 rounded-xl overflow-hidden divide-y divide-slate-800/60 max-h-[460px] overflow-y-auto">
                {dataset.columns.map((col) => {
                  const isTarget = col.name === targetColumn;
                  const isSelected = featureColumns.includes(col.name);

                  return (
                    <div
                      key={col.name}
                      onClick={() => !isTarget && handleToggleFeature(col.name)}
                      className={`p-3.5 flex items-center justify-between transition ${
                        isTarget
                          ? 'bg-slate-950/60 opacity-50 cursor-not-allowed'
                          : isSelected
                          ? 'bg-blue-950/20 hover:bg-blue-950/30 cursor-pointer'
                          : 'hover:bg-slate-800/40 cursor-pointer'
                      }`}
                    >
                      <div className="flex items-center space-x-3">
                        {isTarget ? (
                          <div className="w-4 h-4 rounded bg-slate-800 border border-slate-700 flex items-center justify-center text-[10px] text-slate-500">
                            -
                          </div>
                        ) : isSelected ? (
                          <CheckSquare className="w-4 h-4 text-blue-400" />
                        ) : (
                          <Square className="w-4 h-4 text-slate-600" />
                        )}

                        <div>
                          <div className="flex items-center space-x-2">
                            <span className={`text-xs font-semibold ${isSelected ? 'text-white' : 'text-slate-300'}`}>
                              {col.name}
                            </span>
                            {isTarget && (
                              <span className="px-1.5 py-0.2 rounded text-[10px] font-bold bg-amber-950 text-amber-400 border border-amber-800/60">
                                TARGET
                              </span>
                            )}
                          </div>
                          <span className="text-[11px] text-slate-500 font-mono">
                            {col.dtype} • {col.unique_count} distinct • {col.missing_count} nulls
                          </span>
                        </div>
                      </div>

                      <div className="text-right">
                        {col.is_numerical ? (
                          <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-blue-950 text-blue-300 border border-blue-800/60">
                            Numerical
                          </span>
                        ) : (
                          <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-purple-950 text-purple-300 border border-purple-800/60">
                            One-Hot Encoded
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Data Leakage Note & Submit Button */}
            <div className="mt-6 pt-4 border-t border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-4">
              <div className="flex items-center space-x-2 text-xs text-slate-400">
                <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
                <span>
                  Transformers are fitted strictly on training data to prevent lookahead data leakage.
                </span>
              </div>

              <button
                type="submit"
                disabled={isSubmitting || featureColumns.length === 0}
                className="w-full sm:w-auto px-6 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-xs font-semibold shadow-md shadow-blue-600/20 transition flex items-center justify-center space-x-2 shrink-0"
              >
                {isSubmitting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Validating Pipeline...</span>
                  </>
                ) : (
                  <>
                    <span>Confirm & Preprocess Pipeline</span>
                    <ArrowRight className="w-4 h-4" />
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </form>
    </div>
  );
};
