import React, { useState, useEffect } from 'react';
import {
  Table,
  BarChart,
  FileText,
  Activity,
  Layers,
  Sparkles,
  HelpCircle,
  Hash,
  Type,
  PieChart,
  Image as ImageIcon
} from 'lucide-react';
import { DatasetSummary } from '../types/automl';
import { getDatasetVisualizations } from '../services/api';

interface DatasetExplorerProps {
  summary: DatasetSummary;
}

export const DatasetExplorer: React.FC<DatasetExplorerProps> = ({ summary }) => {
  const [activeTab, setActiveTab] = useState<'preview' | 'schema' | 'stats' | 'charts'>('preview');
  const [charts, setCharts] = useState<Record<string, string>>({});
  const [isLoadingCharts, setIsLoadingCharts] = useState(false);
  const [chartsLoaded, setChartsLoaded] = useState(false);

  useEffect(() => {
    if (activeTab === 'charts' && !chartsLoaded) {
      loadVisualizations();
    }
  }, [activeTab]);

  const loadVisualizations = async () => {
    setIsLoadingCharts(true);
    try {
      const data = await getDatasetVisualizations();
      setCharts(data);
      setChartsLoaded(true);
    } catch (err) {
      console.error('Failed to load dataset charts:', err);
    } finally {
      setIsLoadingCharts(false);
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-sm">
      {/* Overview Stat Badges */}
      <div className="p-6 border-b border-slate-800 bg-slate-950/30">
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
          <div className="bg-slate-900/90 border border-slate-800 p-3 rounded-xl">
            <span className="text-[11px] text-slate-400 font-medium">Dataset Rows</span>
            <div className="text-lg font-bold text-white mt-0.5">
              {summary.total_rows_cleaned.toLocaleString()}
            </div>
            <span className="text-[10px] text-slate-500">
              {summary.total_rows_raw} original
            </span>
          </div>

          <div className="bg-slate-900/90 border border-slate-800 p-3 rounded-xl">
            <span className="text-[11px] text-slate-400 font-medium">Duplicates Removed</span>
            <div className={`text-lg font-bold mt-0.5 ${summary.duplicates_removed > 0 ? 'text-amber-400' : 'text-emerald-400'}`}>
              {summary.duplicates_removed}
            </div>
            <span className="text-[10px] text-slate-500">Auto-deduplicated</span>
          </div>

          <div className="bg-slate-900/90 border border-slate-800 p-3 rounded-xl">
            <span className="text-[11px] text-slate-400 font-medium">Total Columns</span>
            <div className="text-lg font-bold text-white mt-0.5">
              {summary.total_columns}
            </div>
            <span className="text-[10px] text-slate-500">Schema width</span>
          </div>

          <div className="bg-slate-900/90 border border-slate-800 p-3 rounded-xl">
            <span className="text-[11px] text-slate-400 font-medium">Numerical Features</span>
            <div className="text-lg font-bold text-blue-400 mt-0.5">
              {summary.numerical_columns.length}
            </div>
            <span className="text-[10px] text-slate-500">Continuous / int</span>
          </div>

          <div className="bg-slate-900/90 border border-slate-800 p-3 rounded-xl">
            <span className="text-[11px] text-slate-400 font-medium">Categorical Features</span>
            <div className="text-lg font-bold text-purple-400 mt-0.5">
              {summary.categorical_columns.length}
            </div>
            <span className="text-[10px] text-slate-500">Strings / discrete</span>
          </div>

          <div className="bg-slate-900/90 border border-slate-800 p-3 rounded-xl">
            <span className="text-[11px] text-slate-400 font-medium">Missing Values</span>
            <div className="text-lg font-bold text-slate-200 mt-0.5">
              {summary.columns.reduce((acc, c) => acc + c.missing_count, 0)}
            </div>
            <span className="text-[10px] text-slate-500">Across all cells</span>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-slate-800 px-6 bg-slate-900">
        <button
          onClick={() => setActiveTab('preview')}
          className={`py-3 px-4 text-xs font-semibold border-b-2 flex items-center space-x-2 transition ${
            activeTab === 'preview'
              ? 'border-blue-500 text-blue-400'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <Table className="w-4 h-4" />
          <span>Data Preview (First 10 Rows)</span>
        </button>

        <button
          onClick={() => setActiveTab('schema')}
          className={`py-3 px-4 text-xs font-semibold border-b-2 flex items-center space-x-2 transition ${
            activeTab === 'schema'
              ? 'border-blue-500 text-blue-400'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <Layers className="w-4 h-4" />
          <span>Schema & Nullity Report</span>
        </button>

        <button
          onClick={() => setActiveTab('stats')}
          className={`py-3 px-4 text-xs font-semibold border-b-2 flex items-center space-x-2 transition ${
            activeTab === 'stats'
              ? 'border-blue-500 text-blue-400'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <Activity className="w-4 h-4" />
          <span>Descriptive Statistics</span>
        </button>

        <button
          onClick={() => setActiveTab('charts')}
          className={`py-3 px-4 text-xs font-semibold border-b-2 flex items-center space-x-2 transition ${
            activeTab === 'charts'
              ? 'border-blue-500 text-blue-400'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <BarChart className="w-4 h-4" />
          <span>Exploratory Charts</span>
        </button>
      </div>

      {/* Content */}
      <div className="p-6">
        {/* 1. Preview Table */}
        {activeTab === 'preview' && (
          <div className="overflow-x-auto rounded-xl border border-slate-800">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="bg-slate-950/80 uppercase text-[11px] text-slate-400 font-semibold border-b border-slate-800">
                <tr>
                  <th className="px-4 py-3 text-slate-500 w-12">#</th>
                  {summary.columns.map((col) => (
                    <th key={col.name} className="px-4 py-3 font-semibold whitespace-nowrap">
                      <div className="flex items-center space-x-1.5">
                        {col.is_numerical ? (
                          <Hash className="w-3.5 h-3.5 text-blue-400" />
                        ) : (
                          <Type className="w-3.5 h-3.5 text-purple-400" />
                        )}
                        <span className="text-white">{col.name}</span>
                        <span className="text-[10px] text-slate-500 font-normal">({col.dtype})</span>
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {summary.preview.map((row, idx) => (
                  <tr key={idx} className="hover:bg-slate-800/30 transition">
                    <td className="px-4 py-2.5 text-slate-500">{idx + 1}</td>
                    {summary.columns.map((col) => {
                      const val = row[col.name];
                      const isNull = val === null || val === undefined;
                      return (
                        <td key={col.name} className="px-4 py-2.5 whitespace-nowrap">
                          {isNull ? (
                            <span className="px-1.5 py-0.5 rounded bg-rose-950/60 text-rose-400 text-[10px] font-sans font-medium">
                              null
                            </span>
                          ) : (
                            <span className={col.is_numerical ? 'text-slate-200' : 'text-purple-200'}>
                              {typeof val === 'number' ? val.toLocaleString() : String(val)}
                            </span>
                          )}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* 2. Schema Breakdown */}
        {activeTab === 'schema' && (
          <div className="overflow-x-auto rounded-xl border border-slate-800">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="bg-slate-950/80 uppercase text-[11px] text-slate-400 font-semibold border-b border-slate-800">
                <tr>
                  <th className="px-4 py-3">Column Name</th>
                  <th className="px-4 py-3">Inferred Type</th>
                  <th className="px-4 py-3">Data Category</th>
                  <th className="px-4 py-3">Missing Cells</th>
                  <th className="px-4 py-3">Unique Values</th>
                  <th className="px-4 py-3">Sample Value / Range</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {summary.columns.map((col) => (
                  <tr key={col.name} className="hover:bg-slate-800/30">
                    <td className="px-4 py-3 font-semibold text-white">{col.name}</td>
                    <td className="px-4 py-3 font-mono text-slate-400">{col.dtype}</td>
                    <td className="px-4 py-3">
                      {col.is_numerical ? (
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-blue-950 text-blue-300 border border-blue-800/60">
                          Numerical
                        </span>
                      ) : (
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-purple-950 text-purple-300 border border-purple-800/60">
                          Categorical
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      {col.missing_count > 0 ? (
                        <span className="text-rose-400 font-medium">
                          {col.missing_count} ({col.missing_pct}%)
                        </span>
                      ) : (
                        <span className="text-emerald-400 font-medium">0 (0.00%)</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-slate-300">{col.unique_count.toLocaleString()}</td>
                    <td className="px-4 py-3 text-slate-400 font-mono text-[11px]">
                      {col.is_numerical
                        ? `[${col.min ?? 'N/A'}, ${col.max ?? 'N/A'}] (μ = ${col.mean ?? 'N/A'})`
                        : `Top: "${col.top_value ?? 'N/A'}"`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* 3. Descriptive Stats */}
        {activeTab === 'stats' && (
          <div className="overflow-x-auto rounded-xl border border-slate-800">
            {Object.keys(summary.descriptive_stats).length > 0 ? (
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-950/80 uppercase text-[11px] text-slate-400 font-semibold border-b border-slate-800">
                  <tr>
                    <th className="px-4 py-3">Metric</th>
                    {Object.keys(summary.descriptive_stats).map((col) => (
                      <th key={col} className="px-4 py-3 whitespace-nowrap text-white font-semibold">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'].map((metric) => (
                    <tr key={metric} className="hover:bg-slate-800/30">
                      <td className="px-4 py-2.5 font-bold uppercase text-[11px] text-slate-400 font-sans">
                        {metric}
                      </td>
                      {Object.keys(summary.descriptive_stats).map((col) => {
                        const val = summary.descriptive_stats[col][metric];
                        return (
                          <td key={col} className="px-4 py-2.5 whitespace-nowrap text-slate-200">
                            {val !== undefined && val !== null ? Number(val).toLocaleString() : 'N/A'}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="p-8 text-center text-slate-500 text-sm">
                No numerical features available for descriptive statistics.
              </div>
            )}
          </div>
        )}

        {/* 4. Visualizations */}
        {activeTab === 'charts' && (
          <div className="space-y-6">
            {isLoadingCharts ? (
              <div className="p-12 text-center text-slate-400 space-y-3">
                <div className="inline-block w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                <p className="text-sm font-medium">Generating statistical charts with Matplotlib & Seaborn...</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {charts.correlation_heatmap && (
                  <div className="bg-slate-950/50 border border-slate-800 rounded-xl p-4 flex flex-col items-center">
                    <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2 self-start">
                      Feature Correlation Heatmap
                    </h4>
                    <img
                      src={charts.correlation_heatmap}
                      alt="Correlation Heatmap"
                      className="rounded-lg max-h-80 object-contain w-full"
                    />
                  </div>
                )}

                {charts.feature_distributions && (
                  <div className="bg-slate-950/50 border border-slate-800 rounded-xl p-4 flex flex-col items-center">
                    <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2 self-start">
                      Numerical Feature Distributions
                    </h4>
                    <img
                      src={charts.feature_distributions}
                      alt="Feature Distributions"
                      className="rounded-lg max-h-80 object-contain w-full"
                    />
                  </div>
                )}

                {charts.missing_values && (
                  <div className="bg-slate-950/50 border border-slate-800 rounded-xl p-4 flex flex-col items-center">
                    <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2 self-start">
                      Missing Values Percentage
                    </h4>
                    <img
                      src={charts.missing_values}
                      alt="Missing Values"
                      className="rounded-lg max-h-80 object-contain w-full"
                    />
                  </div>
                )}

                {charts.target_distribution && (
                  <div className="bg-slate-950/50 border border-slate-800 rounded-xl p-4 flex flex-col items-center">
                    <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2 self-start">
                      Target Variable Distribution
                    </h4>
                    <img
                      src={charts.target_distribution}
                      alt="Target Distribution"
                      className="rounded-lg max-h-80 object-contain w-full"
                    />
                  </div>
                )}

                {Object.keys(charts).length === 0 && !isLoadingCharts && (
                  <div className="col-span-2 p-8 text-center text-slate-400">
                    No visual charts generated for this dataset schema.
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
