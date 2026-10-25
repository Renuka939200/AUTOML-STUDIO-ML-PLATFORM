import React, { useState, useRef } from 'react';
import {
  UploadCloud,
  FileSpreadsheet,
  AlertTriangle,
  CheckCircle2,
  Sparkles,
  Layers,
  ArrowRight,
  Loader2
} from 'lucide-react';
import { uploadDataset, loadSampleDataset, fetchSampleDatasets } from '../services/api';
import { DatasetSummary, SampleDatasetInfo } from '../types/automl';

interface DatasetUploadProps {
  onDatasetLoaded: (summary: DatasetSummary) => void;
  onProceedToConfigure: () => void;
  currentDataset?: DatasetSummary | null;
}

export const DatasetUpload: React.FC<DatasetUploadProps> = ({
  onDatasetLoaded,
  onProceedToConfigure,
  currentDataset,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isLoadingSample, setIsLoadingSample] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [sampleDatasets, setSampleDatasets] = useState<SampleDatasetInfo[]>([
    {
      id: 'iris',
      name: 'Iris Flower Dataset',
      task: 'classification',
      target: 'species',
      description: '150 samples with 4 measurements to classify 3 iris species (Setosa, Versicolor, Virginica).',
    },
    {
      id: 'wine',
      name: 'Wine Recognition Dataset',
      task: 'classification',
      target: 'wine_class',
      description: '178 samples with 13 continuous chemical constituents to classify 3 wine cultivars.',
    },
    {
      id: 'diabetes',
      name: 'Diabetes Progression',
      task: 'regression',
      target: 'disease_progression',
      description: '442 patient records with 10 physiological metrics to predict disease progression.',
    },
    {
      id: 'housing',
      name: 'California Housing Prices',
      task: 'regression',
      target: 'HouseValue',
      description: '500 census block groups with 8 socioeconomic features to predict median house values.',
    },
  ]);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileProcess = async (file: File) => {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setErrorMessage('Unsupported file type. Please upload a standard CSV file (.csv).');
      return;
    }

    if (file.size > 50 * 1024 * 1024) {
      setErrorMessage('File size exceeds the 50MB limit.');
      return;
    }

    setIsUploading(true);
    setErrorMessage(null);

    try {
      const summary = await uploadDataset(file);
      onDatasetLoaded(summary);
    } catch (err: any) {
      setErrorMessage(err.message || 'An error occurred while uploading and analyzing the CSV.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileProcess(e.dataTransfer.files[0]);
    }
  };

  const handleSampleSelect = async (sample: SampleDatasetInfo) => {
    setIsLoadingSample(sample.id);
    setErrorMessage(null);
    try {
      const summary = await loadSampleDataset(sample.id);
      onDatasetLoaded(summary);
    } catch (err: any) {
      setErrorMessage(err.message || `Failed to load sample dataset '${sample.name}'.`);
    } finally {
      setIsLoadingSample(null);
    }
  };

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Active dataset banner if already loaded */}
      {currentDataset && (
        <div className="p-4 rounded-xl bg-slate-900 border border-emerald-500/30 flex items-center justify-between shadow-sm">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-lg bg-emerald-950/80 border border-emerald-800 flex items-center justify-center text-emerald-400">
              <CheckCircle2 className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-semibold text-white text-sm">{currentDataset.filename}</span>
                <span className="px-2 py-0.5 rounded text-[11px] font-medium bg-emerald-900/60 text-emerald-300 border border-emerald-800/60">
                  Active Dataset
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                {currentDataset.total_rows_cleaned.toLocaleString()} rows cleaned ({currentDataset.duplicates_removed} duplicates removed) • {currentDataset.total_columns} columns ({currentDataset.numerical_columns.length} numeric, {currentDataset.categorical_columns.length} categorical)
              </p>
            </div>
          </div>
          <button
            onClick={onProceedToConfigure}
            className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold shadow-sm transition"
          >
            <span>Proceed to Configuration</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Error alert */}
      {errorMessage && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800/60 text-rose-200 text-sm flex items-start space-x-3">
          <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div className="flex-1">
            <span className="font-semibold text-rose-100">Upload Validation Error: </span>
            {errorMessage}
          </div>
          <button
            onClick={() => setErrorMessage(null)}
            className="text-xs text-rose-400 hover:text-rose-200 font-bold ml-2"
          >
            Dismiss
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Upload Zone */}
        <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <UploadCloud className="w-5 h-5 text-blue-400" />
                Upload CSV Dataset
              </h3>
              <span className="text-xs text-slate-400">Max size 50 MB</span>
            </div>

            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all flex flex-col items-center justify-center min-h-[220px] ${
                isDragging
                  ? 'border-blue-500 bg-blue-950/20'
                  : 'border-slate-700 hover:border-slate-600 hover:bg-slate-800/30'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files && e.target.files.length > 0) {
                    handleFileProcess(e.target.files[0]);
                  }
                }}
              />

              {isUploading ? (
                <div className="flex flex-col items-center space-y-3">
                  <Loader2 className="w-10 h-10 text-blue-400 animate-spin" />
                  <p className="text-sm font-semibold text-white">Parsing and validating CSV...</p>
                  <p className="text-xs text-slate-400">Extracting schema, detecting nulls & computing statistics</p>
                </div>
              ) : (
                <>
                  <div className="w-14 h-14 rounded-2xl bg-blue-950/80 border border-blue-800/60 flex items-center justify-center text-blue-400 mb-4 shadow-inner">
                    <FileSpreadsheet className="w-7 h-7" />
                  </div>
                  <h4 className="text-sm font-semibold text-white mb-1">
                    Click to browse or drag and drop your CSV file here
                  </h4>
                  <p className="text-xs text-slate-400 max-w-sm">
                    Supports numerical & categorical features. Automated deduplication, missing-value imputation, and type inference applied.
                  </p>
                </>
              )}
            </div>
          </div>

          <div className="mt-4 pt-4 border-t border-slate-800 flex items-center justify-between text-xs text-slate-500">
            <span>Encoding: UTF-8 / Latin-1 auto-detect</span>
            <span>Comma-separated RFC 4180</span>
          </div>
        </div>

        {/* Sample Datasets Selector */}
        <div className="lg:col-span-5 bg-slate-900 border border-slate-800 rounded-2xl p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-amber-400" />
              Instant Benchmark Datasets
            </h3>
            <span className="text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-medium">
              1-Click Load
            </span>
          </div>
          <p className="text-xs text-slate-400 mb-4">
            Don't have a dataset ready? Select one of these benchmark datasets to test classification or regression right away:
          </p>

          <div className="space-y-3">
            {sampleDatasets.map((sample) => (
              <div
                key={sample.id}
                onClick={() => handleSampleSelect(sample)}
                className="p-3.5 rounded-xl bg-slate-800/60 hover:bg-slate-800 border border-slate-700/60 hover:border-slate-600 transition cursor-pointer flex items-start justify-between group"
              >
                <div className="space-y-1 pr-2">
                  <div className="flex items-center space-x-2">
                    <span className="text-xs font-semibold text-white group-hover:text-blue-400 transition">
                      {sample.name}
                    </span>
                    <span
                      className={`text-[10px] px-2 py-0.2 rounded-full font-semibold uppercase ${
                        sample.task === 'classification'
                          ? 'bg-purple-950 text-purple-300 border border-purple-800/60'
                          : 'bg-emerald-950 text-emerald-300 border border-emerald-800/60'
                      }`}
                    >
                      {sample.task}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400 leading-relaxed">
                    {sample.description}
                  </p>
                </div>

                <button
                  disabled={isLoadingSample !== null}
                  className="px-2.5 py-1.5 rounded-lg bg-slate-700/80 group-hover:bg-blue-600 text-slate-200 group-hover:text-white text-xs font-medium transition shrink-0 mt-0.5 flex items-center space-x-1"
                >
                  {isLoadingSample === sample.id ? (
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <span>Load</span>
                  )}
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
