import React, { useState } from 'react';
import type { AnalysisData, ReplayFrame } from '../types';
import './ResultsDisplay.css'; // We'll create this for component-specific styles

interface ResultsDisplayProps {
  data: AnalysisData;
}

export const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ data }) => {
  // State to track which frame of the replay is currently selected by the slider
  const [currentFrameIndex, setCurrentFrameIndex] = useState(data.replay_history.length - 1);

  // Get the currently selected frame object from the history array
  const selectedFrame: ReplayFrame = data.replay_history[currentFrameIndex];

  return (
    <div className="results-container">
      {/* The Replay Slider */}
      <div className="slider-container">
        <label htmlFor="replay-slider">Replay Scrubber</label>
        <input
          id="replay-slider"
          type="range"
          min="0"
          max={data.replay_history.length - 1}
          value={currentFrameIndex}
          onChange={(e) => setCurrentFrameIndex(Number(e.target.value))}
          className="slider"
        />
      </div>

      {/* The Frame Display */}
      <div className="frame-display">
        <div className="frame-description">
          <p><strong>Step {currentFrameIndex + 1} / {data.replay_history.length}:</strong> {selectedFrame.description}</p>
        </div>
        <div className="frame-scores">
          <p>Clarity: {selectedFrame.scores.clarity.toFixed(2)}</p>
          <p>Specificity: {selectedFrame.scores.specificity.toFixed(2)}</p>
          <p className="risk-score">Hallucination Risk: {(selectedFrame.scores.hallucination_risk * 100).toFixed(2)}%</p>
        </div>
      </div>
    </div>
  );
};