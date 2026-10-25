import React, { useState } from 'react';
import {
  Sparkles,
  Play,
  Upload,
  Download,
  AlertTriangle,
  CheckCircle2,
  FileSpreadsheet,
  Layers,
  ArrowRight,
  Loader2,
  Table as TableIcon
} from 'lucide-react';
import { ModelConfiguration, PredictionResult, BatchPredictionResult } from '../types/automl';
import { predictSingle, predictBatch } from '../services/api';

interface PredictionInterfaceProps {
  config: ModelConfiguration;
  activeModelName?: string;
  columnStats?: any[];
}

export const PredictionInterface: React.FC<PredictionInterfaceProps> = ({
  config,
  activeModelName,
  columnStats = [],
}) => {
  const [activeTab, setActiveTab] = useState<'single' | 'batch'>('single');

  // Single Prediction Form State
  const initialFormValues = () => {
    const vals: Record<string, string> = {};
    config.feature_columns.forEach((col) => {
      const stat = columnStats.find((s) => s.name === col);
      if (stat?.is_numerical && stat.mean !== null && stat.mean !== undefined) {
        vals[col] = String(stat.mean);
      } else if (stat?.top_value) {
        vals[col] = String(stat.top_value);
      } else {
        vals[col] = '';
      }
    });
    return vals;
  };

  const [formInputs, setFormInputs] = useState<Record<string, string>>(initialFormValues);
  const [isPredicting, setIsPredicting] = useState(false);
  const [singleResult, setSingleResult] = useState<PredictionResult | null>(null);
  const [singleError, setSingleError] = useState<string | null>(null);

  // Batch Prediction State
  const [batchFile, setBatchFile] = useState<File | null>(null);
  const [isBatchProcessing, setIsBatchProcessing] = useState(false);
  const [batchResult, setBatchResult] = useState<BatchPredictionResult | null>(null);
  const [batchError, setBatchError] = useState<string | null>(null);

  const handleInputChange = (col: string, val: string) => {
    setFormInputs((prev) => ({ ...prev, [col]: val }));
  };

  const handleSingleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsPredicting(true);
    setSingleError(null);

    try {
      const res = await predictSingle(formInputs);
      setSingleResult(res);
    } catch (err: any) {
      setSingleError(err.message || 'Inference error on single record.');
    } finally {
      setIsPredicting(false);
    }
  };

  const handleBatchFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setBatchFile(e.target.files[0]);
      setBatchResult(null);
      setBatchError(null);
    }
  };

  const handleBatchSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!batchFile) return;

    setIsBatchProcessing(true);
    setBatchError(null);

    try {
      const res = await predictBatch(batchFile);
      setBatchResult(res);
    } catch (err: any) {
      setBatchError(err.message || 'Failed to score batch CSV.');
    } finally {
      setIsBatchProcessing(false);
    }
  };

  const downloadBatchCSV = () => {
    if (!batchResult?.csv_data) return;
    const blob = new Blob([batchResult.csv_data], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `automl_predictions_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Mode Switch Tabs */}
      <div className="flex border-b border-slate-800 bg-slate-900 rounded-t-2xl px-6 pt-3">
        <button
          onClick={() => setActiveTab('single')}
          className={`py-3 px-4 text-xs font-semibold border-b-2 flex items-center space-x-2 transition ${
            activeTab === 'single'
              ? 'border-blue-500 text-blue-400'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <Sparkles className="w-4 h-4" />
          <span>Interactive Single-Sample Inference</span>
        </button>

        <button
          onClick={() => setActiveTab('batch')}
          className={`py-3 px-4 text-xs font-semibold border-b-2 flex items-center space-x-2 transition ${
            activeTab === 'batch'
              ? 'border-blue-500 text-blue-400'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <FileSpreadsheet className="w-4 h-4" />
          <span>High-Throughput Batch CSV Scoring</span>
        </button>
      </div>

      {/* Tab 1: Single Prediction */}
      {activeTab === 'single' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Inputs Form */}
          <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl p-6">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <Play className="w-5 h-5 text-blue-400" />
                Input Feature Values
              </h3>
              <span className="text-xs text-slate-400 font-mono">
                Predicting: <strong className="text-white">{config.target_column}</strong>
              </span>
            </div>
            <p className="text-xs text-slate-400 mb-6">
              Dynamically generated fields for all {config.feature_columns.length} selected input features. Pre-filled with dataset means/modes for rapid testing:
            </p>

            {singleError && (
              <div className="p-4 mb-4 rounded-xl bg-rose-950/40 border border-rose-800/60 text-rose-200 text-xs flex items-center space-x-2">
                <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
                <span>{singleError}</span>
              </div>
            )}

            <form onSubmit={handleSingleSubmit} className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-h-[420px] overflow-y-auto pr-1">
                {config.feature_columns.map((col) => {
                  const stat = columnStats.find((s) => s.name === col);
                  return (
                    <div key={col} className="space-y-1">
                      <div className="flex justify-between items-center text-xs">
                        <label className="font-semibold text-slate-200 truncate" title={col}>
                          {col}
                        </label>
                        <span className="text-[10px] text-slate-500 font-mono">
                          {stat?.is_numerical ? 'numeric' : 'categorical'}
                        </span>
                      </div>
                      <input
                        type={stat?.is_numerical ? 'number' : 'text'}
                        step="any"
                        value={formInputs[col] ?? ''}
                        onChange={(e) => handleInputChange(col, e.target.value)}
                        placeholder={`Enter ${col}...`}
                        className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white font-mono focus:outline-none focus:border-blue-500"
                      />
                    </div>
                  );
                })}
              </div>

              <div className="pt-4 border-t border-slate-800 flex items-center justify-between">
                <button
                  type="button"
                  onClick={() => setFormInputs(initialFormValues())}
                  className="text-xs text-slate-400 hover:text-slate-300"
                >
                  Reset Defaults
                </button>

                <button
                  type="submit"
                  disabled={isPredicting}
                  className="px-6 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-xs font-bold uppercase tracking-wider shadow-md shadow-blue-600/20 transition flex items-center space-x-2"
                >
                  {isPredicting ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Computing Inference...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      <span>Generate Prediction</span>
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>

          {/* Results Card */}
          <div className="lg:col-span-5 space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 flex flex-col justify-between min-h-[360px]">
              <div>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-base font-bold text-white flex items-center gap-2">
                    <Sparkles className="w-5 h-5 text-indigo-400" />
                    Inference Output
                  </h3>
                  <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">
                    Real-time
                  </span>
                </div>

                {singleResult ? (
                  <div className="space-y-6">
                    <div className="p-6 rounded-2xl bg-gradient-to-br from-indigo-950/60 to-slate-950 border border-indigo-800/80 text-center shadow-inner">
                      <span className="text-xs text-indigo-300 font-semibold uppercase tracking-wider block">
                        Predicted {config.target_column}
                      </span>
                      <div className="text-3xl font-black text-white mt-2 tracking-tight break-words">
                        {singleResult.prediction}
                      </div>

                      {singleResult.confidence !== undefined && (
                        <div className="mt-3 inline-flex items-center space-x-1.5 px-3 py-1 rounded-full bg-emerald-950 text-emerald-300 border border-emerald-800/60 text-xs font-bold">
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          <span>{singleResult.confidence}% Confidence</span>
                        </div>
                      )}
                    </div>

                    {/* Class probabilities breakdown if classification */}
                    {singleResult.probabilities && (
                      <div className="space-y-2">
                        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">
                          Calibrated Class Probabilities
                        </span>
                        <div className="space-y-1.5 max-h-48 overflow-y-auto">
                          {Object.entries(singleResult.probabilities).map(([cls, prob]) => (
                            <div key={cls} className="space-y-1">
                              <div className="flex justify-between text-xs">
                                <span className="text-slate-300 font-medium">{cls}</span>
                                <span className="text-slate-400 font-mono font-semibold">{prob}%</span>
                              </div>
                              <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                                <div
                                  className={`h-full rounded-full ${
                                    String(singleResult.prediction) === cls
                                      ? 'bg-blue-500'
                                      : 'bg-slate-600'
                                  }`}
                                  style={{ width: `${prob}%` }}
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="py-16 text-center text-slate-500 space-y-2">
                    <Play className="w-10 h-10 text-slate-700 mx-auto" />
                    <p className="text-sm font-semibold text-slate-400">Awaiting Input Record</p>
                    <p className="text-xs text-slate-500 max-w-xs mx-auto">
                      Fill out the feature parameters on the left and click 'Generate Prediction' to evaluate using the fitted Scikit-learn pipeline.
                    </p>
                  </div>
                )}
              </div>

              <div className="pt-4 border-t border-slate-800/80 text-[11px] text-slate-500 flex items-center justify-between">
                <span>Active Model: {activeModelName || 'Fitted Pipeline'}</span>
                <span>Leak-Free Pipeline Preprocessing</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Batch CSV Prediction */}
      {activeTab === 'batch' && (
        <div className="space-y-6">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
            <h3 className="text-base font-bold text-white flex items-center gap-2 mb-2">
              <FileSpreadsheet className="w-5 h-5 text-emerald-400" />
              Batch CSV Inference
            </h3>
            <p className="text-xs text-slate-400 mb-6">
              Upload a CSV file containing the required feature columns ({config.feature_columns.join(', ')}). The platform will pass each record through the fitted pipeline and append the predicted values.
            </p>

            {batchError && (
              <div className="p-4 mb-4 rounded-xl bg-rose-950/40 border border-rose-800/60 text-rose-200 text-xs flex items-center space-x-2">
                <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
                <span>{batchError}</span>
              </div>
            )}

            <form onSubmit={handleBatchSubmit} className="space-y-4">
              <div className="border-2 border-dashed border-slate-700 hover:border-slate-600 rounded-xl p-8 text-center bg-slate-950/30">
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleBatchFileChange}
                  className="hidden"
                  id="batch-csv-upload"
                />
                <label
                  htmlFor="batch-csv-upload"
                  className="cursor-pointer flex flex-col items-center justify-center space-y-2"
                >
                  <Upload className="w-8 h-8 text-slate-400" />
                  <span className="text-xs font-semibold text-white">
                    {batchFile ? batchFile.name : 'Select or drop CSV file for batch inference'}
                  </span>
                  <span className="text-[11px] text-slate-500">
                    {batchFile
                      ? `${(batchFile.size / 1024).toFixed(1)} KB selected`
                      : 'Must include all selected training input features'}
                  </span>
                </label>
              </div>

              <div className="flex justify-end">
                <button
                  type="submit"
                  disabled={isBatchProcessing || !batchFile}
                  className="px-6 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-xs font-bold uppercase tracking-wider shadow-md shadow-emerald-600/20 transition flex items-center space-x-2"
                >
                  {isBatchProcessing ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Processing Batch Pipeline...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      <span>Run Batch Inference</span>
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>

          {/* Batch Results Preview Table */}
          {batchResult && (
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-sm font-bold text-white flex items-center gap-2">
                    <TableIcon className="w-4 h-4 text-blue-400" />
                    Batch Predictions Preview ({batchResult.total_samples.toLocaleString()} samples scored)
                  </h4>
                  <span className="text-xs text-slate-400">First 15 scored rows shown below</span>
                </div>

                <button
                  onClick={downloadBatchCSV}
                  className="flex items-center space-x-2 px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold shadow-sm transition"
                >
                  <Download className="w-4 h-4" />
                  <span>Download Complete Scored CSV</span>
                </button>
              </div>

              <div className="overflow-x-auto rounded-xl border border-slate-800 max-h-96">
                <table className="w-full text-left text-xs text-slate-300 font-mono">
                  <thead className="bg-slate-950/80 uppercase text-[11px] text-slate-400 font-semibold border-b border-slate-800 sticky top-0">
                    <tr>
                      {batchResult.preview.length > 0 &&
                        Object.keys(batchResult.preview[0]).map((col) => (
                          <th key={col} className="px-4 py-3 whitespace-nowrap text-white font-semibold">
                            {col}
                          </th>
                        ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60">
                    {batchResult.preview.map((row, idx) => (
                      <tr key={idx} className="hover:bg-slate-800/30">
                        {Object.entries(row).map(([k, val]) => (
                          <td
                            key={k}
                            className={`px-4 py-2.5 whitespace-nowrap ${
                              k.startsWith('Predicted_') ? 'font-bold text-emerald-400' : 'text-slate-300'
                            }`}
                          >
                            {String(val ?? '')}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
