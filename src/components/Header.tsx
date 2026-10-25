import React from 'react';
import { Database, Cpu, CheckCircle2, RotateCw } from 'lucide-react';
import { NavTab } from './Sidebar';

interface HeaderProps {
  currentTab: NavTab;
  activeDatasetName?: string;
  activeAlgorithm?: string;
  taskType?: 'classification' | 'regression';
  onQuickUpload: () => void;
  onRefresh: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  currentTab,
  activeDatasetName,
  activeAlgorithm,
  taskType,
  onQuickUpload,
  onRefresh,
}) => {
  const titles: Record<NavTab, { title: string; subtitle: string }> = {
    overview: {
      title: 'AutoML Studio Overview',
      subtitle: 'Complete end-to-end automated machine learning platform with real Scikit-learn pipelines.',
    },
    upload: {
      title: 'Dataset Ingestion & Exploration',
      subtitle: 'Upload CSV datasets, remove duplicates, inspect schema types, and review summary statistics.',
    },
    configure: {
      title: 'Target & Feature Preprocessing',
      subtitle: 'Configure target variable, select input features, choose task type, and set data split ratio.',
    },
    train: {
      title: 'Model Training & Hyperparameter Tuning',
      subtitle: 'Fit baseline Scikit-learn models or run GridSearchCV / RandomizedSearchCV cross-validation.',
    },
    evaluate: {
      title: 'Model Evaluation & Statistical Plots',
      subtitle: 'Review held-out test set metrics, confusion matrix, ROC-AUC, actual vs predicted, and residual plots.',
    },
    predict: {
      title: 'Interactive Inference & Batch Predictions',
      subtitle: 'Generate predictions for individual records or perform high-throughput batch CSV scoring.',
    },
    models: {
      title: 'Saved Models & Experiment History',
      subtitle: 'Manage persistent Joblib model artifacts and compare evaluation scores across historical runs.',
    },
    report: {
      title: 'Academic Report & Technical Documentation',
      subtitle: 'Detailed mathematical formulations, preprocessing pipelines, algorithm benchmarks, and limitations.',
    },
  };

  const currentInfo = titles[currentTab] || { title: 'AutoML Studio', subtitle: '' };

  return (
    <header className="bg-slate-900 border-b border-slate-800 px-6 py-4 flex items-center justify-between shrink-0">
      <div>
        <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
          {currentInfo.title}
        </h2>
        <p className="text-xs text-slate-400 mt-0.5 max-w-2xl">{currentInfo.subtitle}</p>
      </div>

      <div className="flex items-center space-x-3">
        {activeDatasetName && (
          <div className="hidden md:flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60 text-xs text-slate-300">
            <Database className="w-3.5 h-3.5 text-blue-400" />
            <span className="font-medium truncate max-w-[140px]">{activeDatasetName}</span>
          </div>
        )}

        {activeAlgorithm && (
          <div className="hidden lg:flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-indigo-950/60 border border-indigo-800/60 text-xs text-indigo-300">
            <Cpu className="w-3.5 h-3.5 text-indigo-400" />
            <span className="font-semibold">{activeAlgorithm}</span>
            {taskType && (
              <span className="text-[10px] px-1.5 py-0.2 rounded bg-indigo-800/80 text-white uppercase ml-1">
                {taskType}
              </span>
            )}
          </div>
        )}

        <button
          onClick={onRefresh}
          title="Refresh view"
          className="p-2 rounded-lg bg-slate-800 text-slate-400 hover:text-white hover:bg-slate-700 transition"
        >
          <RotateCw className="w-4 h-4" />
        </button>

        <button
          onClick={onQuickUpload}
          className="px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold shadow-sm transition"
        >
          Upload CSV
        </button>
      </div>
    </header>
  );
};
