"""
섹터/테마 분석 모듈 (v3.0)

=============================================================================
Stage 1: 섹터 분류 (StockThemeAnalyzer)
=============================================================================
목적: 개별 종목의 섹터(테마)와 부가정보를 구체화

INPUT (Stage 0에서 수집):
  - themes: 네이버 테마 목록 (ThemeInfo[])
  - theme_stocks: 테마-종목 참조 (ThemeStockRef[])
  - stock_codes: 분석 대상 종목코드
  - stock_names: {종목코드: 종목명}
  - dart_data: {종목코드: DART 사업개요 텍스트}
  - news_data: {종목코드: [뉴스 기사]}

OUTPUT:
  - theme_stocks_map: {테마명: [종목코드,...]}
  - stock_themes_map: {종목코드: [테마명,...]}  ← N:M 관계
  - stocks_data: StockAnalysisData[]
      - stock_code, stock_name
      - theme_tags: [테마명,...]  ← 섹터 (N개 가능)
      - business_summary: DART LLM 요약
      - news_summary: 뉴스 LLM 요약

핵심: 1종목이 여러 섹터(테마)에 속할 수 있음 (N:M)

=============================================================================
Stage 2: 섹터 타입 분류 (SectorTypeAnalyzer)
=============================================================================
Stage 3: 섹터 우선순위 (SectorPrioritizer)
=============================================================================
"""
from src.sector.classifier import (
    StockThemeAnalyzer,
    ThemeData,
    StockAnalysisData,
    ThemeStockRef,
    # 하위 호환성 (deprecated)
    SectorClassifier,
)

# type_analyzer (v2.0 업데이트 완료)
from src.sector.type_analyzer import SectorTypeAnalyzer, SectorTypeResult

# prioritizer (v2.0 업데이트 완료)
from src.sector.prioritizer import (
    SectorPrioritizer,
    SectorPriorityResult,
    ThemeMetrics,
    SectorMetrics,  # 하위 호환성 별칭
)

__all__ = [
    # Stage 1: 섹터 분류
    "StockThemeAnalyzer",
    "ThemeData",
    "StockAnalysisData",
    "ThemeStockRef",
    # Stage 2: Type A/B 분류
    "SectorTypeAnalyzer",
    "SectorTypeResult",
    # Stage 3: 우선순위 결정
    "SectorPrioritizer",
    "SectorPriorityResult",
    "ThemeMetrics",
    # 하위 호환성 (deprecated)
    "SectorClassifier",
    "SectorMetrics",
]
