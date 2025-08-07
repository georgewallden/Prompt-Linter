import React from 'react';
import './ScoreGauge.css';

interface ScoreGaugeProps {
  value: number; // A value between 0 and 100
  label: string;
}

export const ScoreGauge: React.FC<ScoreGaugeProps> = ({ value, label }) => {
  const radius = 50;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value / 100) * circumference;

  const getColor = () => {
    if (value > 66) return '#cf6679'; // High risk
    if (value > 33) return '#f8d263'; // Medium risk
    return '#03dac6'; // Low risk
  };

  return (
    <div className="gauge-container">
      <svg className="gauge-svg" viewBox="0 0 120 120">
        <circle
          className="gauge-background"
          cx="60"
          cy="60"
          r={radius}
        />
        <circle
          className="gauge-progress"
          cx="60"
          cy="60"
          r={radius}
          stroke={getColor()}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
        />
        <text x="50%" y="50%" className="gauge-value">{value.toFixed(0)}%</text>
      </svg>
      <div className="gauge-label">{label}</div>
    </div>
  );
};