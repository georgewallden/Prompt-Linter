import React, { useState } from 'react';
import { analyzePrompt } from './services/api';
import type { AnalysisData } from './types';
import { ResultsDisplay } from './components/ResultsDisplay';
import './App.css';

function App() {
  const [prompt, setPrompt] = useState<string>('');
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyzeClick = async () => {
    if (!prompt) return;
    setIsLoading(true);
    setError(null);
    setAnalysisData(null);
    try {
      const data = await analyzePrompt(prompt);
      setAnalysisData(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="header">
        <h1>Prompt Linter</h1>
      </div>
      
      <div className="input-card">
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter a prompt to analyze..."
          disabled={isLoading}
        />
        <button onClick={handleAnalyzeClick} disabled={isLoading}>
          {isLoading ? 'Analyzing...' : 'Analyze'}
        </button>
      </div>

      {error && <p className="error-message">Error: {error}</p>}
      {analysisData && <ResultsDisplay data={analysisData} />}
    </div>
  );
}

export default App;