import React, { useState } from 'react';
import { analyzePrompt } from './services/api';
import type { AnalysisData } from './types';
import { ResultsDisplay } from './components/ResultsDisplay'; // Import the new component
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
      <h1>Prompt Linter</h1>
      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="Enter your prompt here..."
        disabled={isLoading}
      />
      <button onClick={handleAnalyzeClick} disabled={isLoading}>
        {isLoading ? 'Analyzing...' : 'Analyze'}
      </button>

      {/* --- THIS IS THE NEW LOGIC --- */}
      <div className="results-area">
        {error && <p className="error-message">Error: {error}</p>}
        
        {/* If we have data, render the ResultsDisplay component and pass the data to it */}
        {analysisData && <ResultsDisplay data={analysisData} />}
      </div>
    </div>
  );
}

export default App;