export interface ColumnStat {
  name: string;
  dtype: string;
  is_numerical: boolean;
  missing_count: number;
  missing_pct: number;
  unique_count: number;
  mean?: number | null;
  std?: number | null;
  min?: number | null;
  max?: number | null;
  top_value?: string | null;
}

export interface DatasetSummary {
  filename: string;
  total_rows_raw: number;
  total_rows_cleaned: number;
  duplicates_removed: number;
  total_columns: number;
  numerical_columns: string[];
  categorical_columns: string[];
  columns: ColumnStat[];
  preview: Record<string, any>[];
  descriptive_stats: Record<string, Record<string, number>>;
}

export interface SampleDatasetInfo {
  id: string;
  name: string;
  task: 'classification' | 'regression';
  target: string;
  description: string;
}

export interface ModelConfiguration {
  target_column: string;
  feature_columns: string[];
  task_type: 'classification' | 'regression';
  test_size: number;
  random_seed: number;
  imputation_strategy: 'mean' | 'median' | 'most_frequent';
  scaling: boolean;
}

export interface ConfigureResponse {
  status: string;
  train_samples: number;
  test_samples: number;
  feature_count: number;
  task_type: 'classification' | 'regression';
  target_column: string;
  classes?: string[] | null;
  constant_features?: string[];
}

export interface ModelMetrics {
  algorithm: string;
  task_type: 'classification' | 'regression';
  is_tuned: boolean;
  training_time_sec: number;
  hyperparameters?: Record<string, any>;
  cv_score_mean?: number | null;
  cv_score_std?: number | null;
  // Classification
  accuracy?: number;
  precision?: number;
  precision_macro?: number;
  recall?: number;
  recall_macro?: number;
  f1_score?: number;
  f1_macro?: number;
  confusion_matrix?: number[][];
  labels?: string[];
  roc_auc?: number | null;
  roc_note?: string | null;
  // Regression
  mae?: number;
  mse?: number;
  rmse?: number;
  r2?: number;
  // Tuning details
  search_type?: string;
  best_cv_score?: number;
  best_params?: Record<string, any>;
  baseline_score?: number;
  tuned_score?: number;
  improvement?: number;
  charts?: Record<string, string>;
}

export interface PredictionResult {
  prediction: string | number;
  task_type: 'classification' | 'regression';
  probabilities?: Record<string, number> | null;
  confidence?: number;
}

export interface BatchPredictionResult {
  total_samples: number;
  preview: Record<string, any>[];
  csv_data: string;
}

export interface SavedModelRecord {
  id: string;
  name: string;
  algorithm: string;
  task_type: 'classification' | 'regression';
  target_column: string;
  feature_names: string[];
  hyperparameters: Record<string, any>;
  metrics: ModelMetrics;
  created_at: string;
  dataset_name: string;
  dataset_rows: number;
  dataset_cols: number;
}
