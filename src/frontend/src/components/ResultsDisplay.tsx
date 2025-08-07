import React, { useState, useMemo } from 'react';
import type { AnalysisData } from '../types';
import { ScoreGauge } from './ScoreGauge';
import './ResultsDisplay.css';

interface ResultsDisplayProps {
  data: AnalysisData;
}

interface TimelineFrame {
  token: string;
  scores: { clarity: number; specificity: number; hallucination_risk: number; };
  description: string;
}

export const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ data }) => {
  const timeline: TimelineFrame[] = useMemo(() => {
    // ... (This logic is unchanged and correct)
    const newTimeline: TimelineFrame[] = [];
    let currentScores = data.replay_history[0].scores;
    let replayFrameIndex = 1;
    data.token_trace.forEach(traceItem => {
      let description = `Token '${traceItem.token}' processed. No change in scores.`;
      if (traceItem.explanation) {
        const replayFrame = data.replay_history[replayFrameIndex];
        if (replayFrame) {
          description = replayFrame.description;
          currentScores = replayFrame.scores;
          replayFrameIndex++;
        }
      }
      newTimeline.push({ token: traceItem.token, scores: currentScores, description: description });
    });
    return newTimeline;
  }, [data]);

  const [currentFrameIndex, setCurrentFrameIndex] = useState(timeline.length - 1);
  const selectedFrame = timeline[currentFrameIndex];

  return (
    <div className="results-card">
      <div className="slider-container">
        <label htmlFor="replay-slider">
          Token {currentFrameIndex + 1} / {timeline.length} ('{selectedFrame.token}')
        </label>
        <input
          id="replay-slider"
          type="range"
          min="0"
          max={timeline.length - 1}
          value={currentFrameIndex}
          onChange={(e) => setCurrentFrameIndex(Number(e.target.value))}
          className="slider"
        />
      </div>

      {/* --- THIS IS THE UPDATED LAYOUT --- */}
      <div className="frame-content">
        <ScoreGauge value={selectedFrame.scores.clarity * 100} label="Clarity" />
        <ScoreGauge value={selectedFrame.scores.specificity * 100} label="Specificity" />
        <ScoreGauge value={selectedFrame.scores.hallucination_risk * 100} label="Hallucination Risk" />
      </div>
      
      <div className="frame-description">{selectedFrame.description}</div>
    </div>
  );
};