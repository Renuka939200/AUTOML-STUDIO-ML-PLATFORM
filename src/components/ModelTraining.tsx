import React, { useState } from 'react';
import {
  Cpu,
  Zap,
  Sliders,
  CheckCircle2,
  Clock,
  Sparkles,
  ArrowRight,
  Loader2,
  AlertTriangle,
  HelpCircle,
  TrendingUp,
  BarChart2
} from 'lucide-react';
import { ModelMetrics, ModelConfiguration } from '../types/automl';
import { trainModel, tuneModel } from '../services/api';

interface ModelTrainingProps {
  config: ModelConfiguration;
  onTrainingComplete: (metrics: ModelMetrics) => void;
  onProceedToEvaluate: () => void;
  lastMetrics?: ModelMetrics | null;
}

export const ModelTraining: React.FC<ModelTrainingProps> = ({
  config,
  onTrainingComplete,
  onProceedToEvaluate,
  lastMetrics,
}) => {
  const classificationAlgorithms = [
    {
      name: 'Random Forest Classifier',
      badge: 'Ensemble',
      desc: 'Robust ensemble of decision trees with bagging. Handles non-linear boundaries and feature interactions gracefully.',
    },
    {
      name: 'Logistic Regression',
      badge: 'Linear',
      desc: 'Standard probabilistic linear classifier with L2 regularization and L-BFGS optimization.',
    },
    {
      name: 'Decision Tree Classifier',
      badge: 'Tree',
      desc: 'Interpretable hierarchical decision rules optimized via Gini impurity or entropy information gain.',
    },
    {
      name: 'K-Nearest Neighbors Classifier',
      badge: 'Instance',
      desc: 'Non-parametric lazy learner classifying points based on Euclidean distance to nearest neighbors.',
    },
    {
      name: 'Naive Bayes',
      badge: 'Bayesian',
      desc: 'Gaussian Naive Bayes applying Bayes theorem with conditional feature independence assumptions.',
    },
    {
      name: 'Support Vector Machine Classifier',
      badge: 'Kernel',
      desc: 'Maximum-margin hyperplane classifier with Radial Basis Function (RBF) kernel and probability calibration.',
    },
  ];

  const regressionAlgorithms = [
    {
      name: 'Random Forest Regressor',
      badge: 'Ensemble',
      desc: 'Non-linear ensemble reducing variance through bagging and random feature subsampling.',
    },
    {
      name: 'Linear Regression',
      badge: 'Linear',
      desc: 'Ordinary least squares linear regression estimating coefficients that minimize residual sum of squares.',
    },
    {
      name: 'Polynomial Regression',
      badge: 'Non-Linear',
      desc: 'Linear model equipped with PolynomialFeatures expansion to capture polynomial interaction terms.',
    },
    {
      name: 'Decision Tree Regressor',
      badge: 'Tree',
      desc: 'Piecewise constant approximation partitioning the feature space into homogeneous response regions.',
    },
    {
      name: 'Support Vector Regression',
      badge: 'Kernel',
      desc: 'Epsilon-insensitive loss function SVR fitting an optimal tube around continuous observations.',
    },
  ];

  const availableAlgorithms =
    config.task_type === 'classification' ? classificationAlgorithms : regressionAlgorithms;

  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>(
    lastMetrics?.algorithm || availableAlgorithms[0].name
  );
  const [searchType, setSearchType] = useState<'grid' | 'random'>('grid');
  const [cvFolds, setCvFolds] = useState<number>(3);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [isTuning, setIsTuning] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [currentMetrics, setCurrentMetrics] = useState<ModelMetrics | null>(lastMetrics || null);

  const handleTrainBaseline = async () => {
    setIsTraining(true);
    setErrorMessage(null);
    try {
      const res = await trainModel(selectedAlgorithm);
      setCurrentMetrics(res);
      onTrainingComplete(res);
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to train baseline model.');
    } finally {
      setIsTraining(false);
    }
  };

  const handleFineTune = async () => {
    setIsTuning(true);
    setErrorMessage(null);
    try {
      const res = await tuneModel(selectedAlgorithm, searchType, cvFolds);
      setCurrentMetrics(res);
      onTrainingComplete(res);
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to fine-tune model hyperparameters.');
    } finally {
      setIsTuning(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {errorMessage && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800/60 text-rose-200 text-sm flex items-start space-x-3">
          <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div className="flex-1">
            <span className="font-semibold text-rose-100">Training Error: </span>
            {errorMessage}
          </div>
        </div>
      )}

      {/* Model status success banner */}
      {currentMetrics && (
        <div className="p-4 rounded-xl bg-slate-900 border border-indigo-500/40 shadow-sm flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-lg bg-indigo-950 border border-indigo-800 flex items-center justify-center text-indigo-400 shrink-0">
              <CheckCircle2 className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-white text-sm">{currentMetrics.algorithm}</span>
                <span
                  className={`text-[10px] px-2 py-0.2 rounded font-semibold uppercase ${
                    currentMetrics.is_tuned
                      ? 'bg-amber-950 text-amber-300 border border-amber-800/60'
                      : 'bg-blue-950 text-blue-300 border border-blue-800/60'
                  }`}
                >
                  {currentMetrics.is_tuned ? `Tuned (${currentMetrics.search_type?.toUpperCase()} CV)` : 'Baseline Model'}
                </span>
                <span className="text-xs text-slate-400 font-mono flex items-center gap-1">
                  <Clock className="w-3.5 h-3.5" />
                  {currentMetrics.training_time_sec}s
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                {currentMetrics.task_type === 'classification' ? (
                  <>
                    Test Accuracy: <strong className="text-emerald-400">{((currentMetrics.accuracy ?? 0) * 100).toFixed(2)}%</strong> • Weighted F1: <strong className="text-white">{((currentMetrics.f1_score ?? 0) * 100).toFixed(2)}%</strong>
                    {currentMetrics.cv_score_mean && ` • ${cvFolds}-Fold CV: ${(currentMetrics.cv_score_mean * 100).toFixed(2)}%`}
                  </>
                ) : (
                  <>
                    Test R² Score: <strong className="text-emerald-400">{(currentMetrics.r2 ?? 0).toFixed(4)}</strong> • RMSE: <strong className="text-white">{(currentMetrics.rmse ?? 0).toFixed(4)}</strong> • MAE: <strong className="text-white">{(currentMetrics.mae ?? 0).toFixed(4)}</strong>
                    {currentMetrics.cv_score_mean && ` • ${cvFolds}-Fold CV R²: ${currentMetrics.cv_score_mean.toFixed(4)}`}
                  </>
                )}
              </p>
            </div>
          </div>

          <button
            onClick={onProceedToEvaluate}
            className="flex items-center space-x-1.5 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-sm transition shrink-0"
          >
            <span>View Full Evaluation & Plots</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Algorithm Selector */}
        <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl p-6">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Cpu className="w-5 h-5 text-blue-400" />
              Select Machine Learning Algorithm
            </h3>
            <span className="text-[11px] px-2.5 py-0.5 rounded-full font-bold uppercase tracking-wider bg-slate-800 text-slate-300">
              {config.task_type}
            </span>
          </div>
          <p className="text-xs text-slate-400 mb-4">
            Scikit-learn algorithms ready to train on the preprocessed pipeline:
          </p>

          <div className="space-y-3">
            {availableAlgorithms.map((algo) => {
              const isSelected = selectedAlgorithm === algo.name;

              return (
                <div
                  key={algo.name}
                  onClick={() => setSelectedAlgorithm(algo.name)}
                  className={`p-4 rounded-xl border cursor-pointer transition flex items-start justify-between ${
                    isSelected
                      ? 'bg-blue-950/30 border-blue-500 text-white shadow-sm shadow-blue-500/10'
                      : 'bg-slate-800/40 border-slate-800 hover:bg-slate-800/80 hover:border-slate-700 text-slate-300'
                  }`}
                >
                  <div className="space-y-1 pr-4">
                    <div className="flex items-center space-x-2">
                      <span className="text-sm font-bold text-white">{algo.name}</span>
                      <span className="text-[10px] px-2 py-0.2 rounded-full font-semibold uppercase bg-slate-800 text-slate-400 border border-slate-700">
                        {algo.badge}
                      </span>
                    </div>
                    <p className="text-xs text-slate-400 leading-relaxed">{algo.desc}</p>
                  </div>

                  <div className="shrink-0 mt-1">
                    <div
                      className={`w-4 h-4 rounded-full border flex items-center justify-center ${
                        isSelected ? 'border-blue-500 bg-blue-500' : 'border-slate-600'
                      }`}
                    >
                      {isSelected && <div className="w-1.5 h-1.5 rounded-full bg-white" />}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right Column: Training & Tuning Actions */}
        <div className="lg:col-span-5 space-y-6">
          {/* Action 1: Train Baseline Model */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Zap className="w-4 h-4 text-emerald-400" />
                Action 1: Train Baseline Model
              </h3>
              <span className="text-[11px] text-slate-400 font-mono">Standard Fit</span>
            </div>
            <p className="text-xs text-slate-400 mb-4">
              Fits the pipeline with default Scikit-learn hyperparameters and computes cross-validation on training data.
            </p>

            <button
              onClick={handleTrainBaseline}
              disabled={isTraining || isTuning}
              className="w-full py-3 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-xs font-bold uppercase tracking-wider shadow-md shadow-emerald-600/20 transition flex items-center justify-center space-x-2"
            >
              {isTraining ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Fitting Scikit-Learn Pipeline...</span>
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  <span>Train {selectedAlgorithm}</span>
                </>
              )}
            </button>
          </div>

          {/* Action 2: Fine-Tune Hyperparameters */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Sliders className="w-4 h-4 text-amber-400" />
                Action 2: Hyperparameter Fine-Tuning
              </h3>
              <span className="text-[11px] text-slate-400 font-mono">CV Search</span>
            </div>
            <p className="text-xs text-slate-400">
              Performs exhaustive GridSearchCV or RandomizedSearchCV over algorithm-specific parameter grids using only the training split.
            </p>

            {/* Tuning Search Method */}
            <div className="space-y-1.5">
              <label className="text-xs text-slate-400 font-medium block">Search Strategy:</label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => setSearchType('grid')}
                  className={`py-2 px-3 rounded-lg text-xs font-semibold border transition ${
                    searchType === 'grid'
                      ? 'bg-amber-950/40 border-amber-500 text-amber-200'
                      : 'bg-slate-800/60 border-slate-700 text-slate-400 hover:bg-slate-800'
                  }`}
                >
                  GridSearchCV (Exhaustive)
                </button>
                <button
                  type="button"
                  onClick={() => setSearchType('random')}
                  className={`py-2 px-3 rounded-lg text-xs font-semibold border transition ${
                    searchType === 'random'
                      ? 'bg-amber-950/40 border-amber-500 text-amber-200'
                      : 'bg-slate-800/60 border-slate-700 text-slate-400 hover:bg-slate-800'
                  }`}
                >
                  RandomizedSearchCV
                </button>
              </div>
            </div>

            {/* Cross-Validation Folds */}
            <div className="space-y-1.5">
              <label className="text-xs text-slate-400 font-medium block">
                Cross-Validation Folds (K-Fold):
              </label>
              <select
                value={cvFolds}
                onChange={(e) => setCvFolds(parseInt(e.target.value, 10))}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-amber-500"
              >
                <option value={2}>2 Folds (Fast)</option>
                <option value={3}>3 Folds (Standard)</option>
                <option value={5}>5 Folds (Rigorous)</option>
              </select>
            </div>

            <button
              onClick={handleFineTune}
              disabled={isTraining || isTuning}
              className="w-full py-3 rounded-xl bg-amber-600 hover:bg-amber-500 disabled:opacity-50 text-white text-xs font-bold uppercase tracking-wider shadow-md shadow-amber-600/20 transition flex items-center justify-center space-x-2"
            >
              {isTuning ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Searching Hyperparameter Space...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Execute {searchType === 'grid' ? 'Grid Search' : 'Random Search'}</span>
                </>
              )}
            </button>
          </div>

          {/* Hyperparameter Summary if Available */}
          {currentMetrics?.hyperparameters && (
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 text-xs">
              <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2 flex items-center justify-between">
                <span>Model Hyperparameters</span>
                <span className="text-[10px] text-indigo-400 font-mono">Fitted Spec</span>
              </div>
              <div className="max-h-36 overflow-y-auto space-y-1 bg-slate-950/60 p-3 rounded-xl border border-slate-800 font-mono text-[11px]">
                {Object.entries(currentMetrics.hyperparameters).map(([k, v]) => (
                  <div key={k} className="flex justify-between text-slate-400">
                    <span className="text-slate-300">{k}:</span>
                    <span className="text-blue-400 font-bold">{String(v)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
