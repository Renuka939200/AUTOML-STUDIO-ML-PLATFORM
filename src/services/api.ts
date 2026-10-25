import {
  DatasetSummary,
  SampleDatasetInfo,
  ModelConfiguration,
  ConfigureResponse,
  ModelMetrics,
  PredictionResult,
  BatchPredictionResult,
  SavedModelRecord,
} from '../types/automl';

const BASE_URL = '/api';

/**
 * Safely parses responses, detecting HTML or proxy error pages (e.g. <!doctype ...)
 * and providing human-readable diagnostic messages instead of JSON parse crashes.
 */
async function safeParseResponse<T = any>(res: Response, fallbackError = 'Request failed'): Promise<T> {
  const contentType = res.headers.get('content-type') || '';

  if (!contentType.includes('application/json')) {
    const rawText = await res.text();
    if (!res.ok) {
      if (res.status === 502 || res.status === 504 || rawText.includes('ECONNREFUSED') || rawText.includes('502 Bad Gateway')) {
        throw new Error(
          'Python ML backend (Flask) is not responding on port 5001. Please make sure the Python server is running (python backend/app.py --port 5001).'
        );
      }
      if (rawText.toLowerCase().includes('<!doctype')) {
        throw new Error(
          `Backend returned HTML instead of JSON (HTTP ${res.status}). Ensure the Python Flask server is running on port 5001.`
        );
      }
      throw new Error(`Server returned HTTP ${res.status}: ${rawText.slice(0, 150)}`);
    }
    throw new Error(
      'Unexpected non-JSON response from server. Please verify the Python Flask backend is running on port 5001.'
    );
  }

  let data: any;
  try {
    data = await res.json();
  } catch (err: any) {
    throw new Error(`Failed to parse server response as JSON: ${err.message}`);
  }

  if (!res.ok) {
    throw new Error(data?.error || `${fallbackError} (HTTP ${res.status})`);
  }

  return data as T;
}

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE_URL}/health`);
    const contentType = res.headers.get('content-type') || '';
    if (!res.ok || !contentType.includes('application/json')) return false;
    const data = await res.json();
    return data?.status === 'healthy';
  } catch {
    return false;
  }
}

export async function uploadDataset(file: File): Promise<DatasetSummary> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${BASE_URL}/upload`, {
    method: 'POST',
    body: formData,
  });

  return safeParseResponse<DatasetSummary>(res, 'Failed to upload dataset');
}

export async function fetchSampleDatasets(): Promise<SampleDatasetInfo[]> {
  const res = await fetch(`${BASE_URL}/dataset/samples`);
  return safeParseResponse<SampleDatasetInfo[]>(res, 'Failed to fetch sample datasets');
}

export async function loadSampleDataset(sampleId: string): Promise<DatasetSummary> {
  const res = await fetch(`${BASE_URL}/dataset/load-sample`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sample_id: sampleId }),
  });

  return safeParseResponse<DatasetSummary>(res, 'Failed to load sample dataset');
}

export async function getDatasetSummary(): Promise<DatasetSummary> {
  const res = await fetch(`${BASE_URL}/dataset/summary`);
  return safeParseResponse<DatasetSummary>(res, 'Failed to get dataset summary');
}

export async function configureDataset(config: ModelConfiguration): Promise<ConfigureResponse> {
  const res = await fetch(`${BASE_URL}/dataset/configure`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });

  return safeParseResponse<ConfigureResponse>(res, 'Failed to configure dataset');
}

export async function getDatasetConfig(): Promise<{
  configured: boolean;
  config: ModelConfiguration | null;
  train_samples?: number;
  test_samples?: number;
  classes?: string[] | null;
}> {
  try {
    const res = await fetch(`${BASE_URL}/dataset/config`);
    if (!res.ok) return { configured: false, config: null };
    return await safeParseResponse(res);
  } catch {
    return { configured: false, config: null };
  }
}

export async function getDatasetVisualizations(targetCol?: string): Promise<Record<string, string>> {
  const url = targetCol
    ? `${BASE_URL}/dataset/visualizations?target=${encodeURIComponent(targetCol)}`
    : `${BASE_URL}/dataset/visualizations`;
  const res = await fetch(url);
  return safeParseResponse<Record<string, string>>(res, 'Failed to load dataset visualizations');
}

export async function trainModel(algorithm: string): Promise<ModelMetrics> {
  const res = await fetch(`${BASE_URL}/model/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ algorithm }),
  });

  return safeParseResponse<ModelMetrics>(res, 'Failed to train model');
}

export async function tuneModel(
  algorithm: string,
  searchType: 'grid' | 'random' = 'grid',
  cvFolds: number = 3,
  nIter: number = 10
): Promise<ModelMetrics> {
  const res = await fetch(`${BASE_URL}/model/tune`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      algorithm,
      search_type: searchType,
      cv_folds: cvFolds,
      n_iter: nIter,
    }),
  });

  return safeParseResponse<ModelMetrics>(res, 'Failed to fine-tune model');
}

export async function getModelMetrics(): Promise<ModelMetrics> {
  const res = await fetch(`${BASE_URL}/model/metrics`);
  return safeParseResponse<ModelMetrics>(res, 'Failed to fetch model metrics');
}

export async function getModelVisualizations(): Promise<Record<string, string>> {
  const res = await fetch(`${BASE_URL}/model/visualizations`);
  return safeParseResponse<Record<string, string>>(res, 'Failed to fetch model visualizations');
}

export async function predictSingle(inputs: Record<string, any>): Promise<PredictionResult> {
  const res = await fetch(`${BASE_URL}/model/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ inputs }),
  });

  return safeParseResponse<PredictionResult>(res, 'Prediction failed');
}

export async function predictBatch(file: File): Promise<BatchPredictionResult> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${BASE_URL}/model/batch-predict`, {
    method: 'POST',
    body: formData,
  });

  return safeParseResponse<BatchPredictionResult>(res, 'Batch prediction failed');
}

export async function saveModel(name: string): Promise<{ status: string; model_id: string; created_at: string }> {
  const res = await fetch(`${BASE_URL}/model/save`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });

  return safeParseResponse<{ status: string; model_id: string; created_at: string }>(res, 'Failed to save model');
}

export async function fetchSavedModels(): Promise<SavedModelRecord[]> {
  const res = await fetch(`${BASE_URL}/models`);
  return safeParseResponse<SavedModelRecord[]>(res, 'Failed to fetch saved models');
}

export async function loadSavedModel(modelId: string): Promise<any> {
  const res = await fetch(`${BASE_URL}/model/load`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId }),
  });

  return safeParseResponse<any>(res, 'Failed to load model');
}

export async function deleteSavedModel(modelId: string): Promise<void> {
  const res = await fetch(`${BASE_URL}/models/${modelId}`, {
    method: 'DELETE',
  });

  return safeParseResponse<void>(res, 'Failed to delete model');
}
