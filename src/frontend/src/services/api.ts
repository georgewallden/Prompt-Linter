import type { AnalysisData } from '../types';

const API_BASE_URL = 'http://127.0.0.1:8000';

export const analyzePrompt = async (prompt: string): Promise<AnalysisData> => {
  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || 'An unknown error occurred.');
  }

  const result = await response.json();
  return result.data as AnalysisData;
};