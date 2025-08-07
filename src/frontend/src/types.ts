export interface Rule {
    token: string;
    type: 'RISK_INCREASE' | 'RISK_DECREASE';
    explanation: string;
    strength: number;
}

export interface TokenTraceItem {
    token: string;
    position: number;
    explanation?: Rule;
}

export interface Scores {
    intent: {
        label: string;
        confidence: number;
    };
    clarity: number;
    specificity: number;
    hallucination_risk: number;
}

export interface ReplayFrame {
    event_type: 'BASE_MODEL_SCORE' | 'RULE_TRIGGER';
    description: string;
    scores: Scores;
    triggered_at_token?: string;
}

export interface AnalysisData {
    prompt: string;
    scores: Scores;
    token_trace: TokenTraceItem[];
    replay_history: ReplayFrame[];
}