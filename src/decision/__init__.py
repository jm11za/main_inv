"""
Layer 5: Decision

LLM 기반 최종 투자 판정

Dual Persona 구조:
- Skeptic: 냉철한 애널리스트 (재료 분석 → S/A/B/C 등급)
- Sentiment Reader: 심리 분석가 (대중 심리 → 공포/의심/확신/환희)

Decision Matrix로 최종 판정:
- STRONG_BUY: 적극 매수
- BUY: 매수
- WATCH: 관망
- AVOID: 회피
"""
from src.decision.personas.skeptic import Skeptic, SkepticAnalysis
from src.decision.personas.sentiment import SentimentReader, SentimentAnalysis
from src.decision.decision_engine import DecisionEngine, DecisionResult
from src.decision.llm_analyzer import LLMAnalyzer

__all__ = [
    # Personas
    "Skeptic",
    "SkepticAnalysis",
    "SentimentReader",
    "SentimentAnalysis",
    # Engine
    "DecisionEngine",
    "DecisionResult",
    # Main Analyzer
    "LLMAnalyzer",
]
