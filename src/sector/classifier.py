"""
=============================================================================
Stage 1: 섹터 분류 (StockThemeAnalyzer)
=============================================================================

목적: 개별 종목의 섹터(테마)와 부가정보를 구체화

┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 1 범위                                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [어디서 받나] Stage 0 (데이터 수집)에서 받음                            │
│    - themes: 네이버 테마 목록 (ThemeInfo[])                              │
│    - theme_stocks: 테마-종목 참조 (ThemeStockRef[])                      │
│    - stock_codes: 분석 대상 종목코드                                     │
│    - stock_names: {종목코드: 종목명}                                     │
│    - dart_data: {종목코드: DART 사업개요 텍스트}                         │
│    - news_data: {종목코드: [뉴스 기사]}                                  │
│                                                                         │
│  [무엇을 하나] 개별 종목별 섹터 할당 + 부가정보 생성                      │
│    1. 테마-종목 N:M 매핑 구축                                            │
│       - theme_stocks_map: {테마명: [종목코드,...]}                       │
│       - stock_themes_map: {종목코드: [테마명,...]}                       │
│    2. 종목별 StockAnalysisData 생성                                      │
│       - theme_tags: 해당 종목이 속한 섹터(테마) 목록                     │
│       - business_summary: DART 사업개요 LLM 요약                         │
│       - news_summary: 뉴스 LLM 요약                                      │
│                                                                         │
│  [어디로 보내나] Stage 2 (섹터 타입 분류)로 전달                          │
│    - stocks_data: StockAnalysisData[]                                   │
│    - theme_stocks_map: 테마별 종목 그룹핑                                │
│    - stock_themes_map: 종목별 섹터 참조                                  │
│                                                                         │
│  핵심: 1종목이 여러 섹터(테마)에 속할 수 있음 (N:M)                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.core.logger import get_logger
from src.core.config import get_config


@dataclass
class ThemeStockRef:
    """테마-종목 참조 (Stage 1 INPUT용)"""
    theme_id: str
    stock_code: str


@dataclass
class ThemeData:
    """
    테마 데이터 (네이버 금융 기준)

    네이버에서 크롤링한 테마 정보를 그대로 사용.
    테마명을 카테고리로 변환하지 않음.
    """
    theme_id: str
    theme_name: str                 # 네이버 테마명 그대로 (예: "2차전지(소재)")
    change_rate: float = 0.0        # 등락률 (%)
    stock_count: int = 0            # 소속 종목 수
    stock_codes: list[str] = field(default_factory=list)  # 소속 종목 코드

    def to_dict(self) -> dict:
        return {
            "theme_id": self.theme_id,
            "theme_name": self.theme_name,
            "change_rate": round(self.change_rate, 2),
            "stock_count": self.stock_count,
            "stock_codes": self.stock_codes,
        }


@dataclass
class StockAnalysisData:
    """
    종목별 분석 데이터 (Stage 1 OUTPUT)

    - 섹터: theme_tags (N개 가능, 1종목 N섹터 관계)
    - 부가정보: business_summary, news_summary (LLM 요약)
    """
    stock_code: str
    stock_name: str

    # 섹터 (N개 가능) - 네이버 테마명 그대로
    theme_tags: list[str] = field(default_factory=list)

    # 부가정보 (LLM 요약)
    business_summary: str = ""      # DART 사업개요 요약 (3-5문장)
    news_summary: str = ""          # 뉴스 동향 요약 (3-5문장)

    # 메타데이터
    data_sources: list[str] = field(default_factory=list)  # ["theme", "dart", "news"]
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "stock_code": self.stock_code,
            "stock_name": self.stock_name,
            "theme_tags": self.theme_tags,
            "business_summary": self.business_summary,
            "news_summary": self.news_summary,
            "data_sources": self.data_sources,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class StockThemeAnalyzer:
    """
    ① 종목-테마 분석기 (v2.0)

    네이버 테마를 그대로 사용하여 종목별 데이터셋 구축:
    - 테마명을 카테고리로 변환하지 않음
    - 한 종목이 여러 테마에 속할 수 있음
    - DART 사업개요와 뉴스를 LLM으로 요약

    사용법:
        analyzer = StockThemeAnalyzer(llm_client)

        # 테마 데이터 구축
        themes_data = analyzer.build_themes_data(themes, theme_stocks)

        # 종목 분석 데이터 구축
        stocks_data = analyzer.build_stocks_data(
            stock_codes=stock_codes,
            stock_names=stock_names,
            stock_themes_map=stock_themes_map,
            dart_data=dart_data,
            news_data=news_data,
        )
    """

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: LLM 클라이언트 (Ollama 또는 Claude)
        """
        self.logger = get_logger(self.__class__.__name__)
        self.config = get_config()
        self._llm_client = llm_client

    def set_llm_client(self, client):
        """LLM 클라이언트 설정"""
        self._llm_client = client

    # =========================================================================
    # 테마 데이터 구축
    # =========================================================================

    def build_themes_data(
        self,
        themes: list,
        theme_stocks: list,
    ) -> list[ThemeData]:
        """
        테마 데이터 구축

        Args:
            themes: ThemeInfo 리스트 (크롤링 결과)
            theme_stocks: ThemeStock 리스트 (크롤링 결과)

        Returns:
            ThemeData 리스트
        """
        self.logger.info(f"테마 데이터 구축 시작: {len(themes)}개 테마")

        # 테마별 종목 그룹핑
        theme_stocks_map: dict[str, list[str]] = {}
        for ts in theme_stocks:
            theme_id = ts.theme_id
            if theme_id not in theme_stocks_map:
                theme_stocks_map[theme_id] = []
            theme_stocks_map[theme_id].append(ts.stock_code)

        # ThemeData 생성
        themes_data = []
        for theme in themes:
            stock_codes = theme_stocks_map.get(theme.theme_id, [])
            themes_data.append(ThemeData(
                theme_id=theme.theme_id,
                theme_name=theme.name,
                change_rate=theme.change_rate,
                stock_count=len(stock_codes),
                stock_codes=stock_codes,
            ))

        self.logger.info(f"테마 데이터 구축 완료: {len(themes_data)}개")
        return themes_data

    def build_theme_maps(
        self,
        theme_stocks: list,
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """
        테마-종목 매핑 구축

        Args:
            theme_stocks: ThemeStock 리스트

        Returns:
            (theme_stocks_map, stock_themes_map)
            - theme_stocks_map: {테마명: [종목코드, ...]}
            - stock_themes_map: {종목코드: [테마명, ...]}
        """
        theme_stocks_map: dict[str, list[str]] = {}  # 테마 → 종목들
        stock_themes_map: dict[str, list[str]] = {}  # 종목 → 테마들
        theme_id_to_name: dict[str, str] = {}        # 테마ID → 테마명

        for ts in theme_stocks:
            theme_id = ts.theme_id
            stock_code = ts.stock_code

            # 테마명 캐싱 (첫 번째 발견 시)
            if theme_id not in theme_id_to_name:
                # theme_stocks에는 theme_name이 없으므로 별도로 받아야 함
                # 여기서는 theme_id를 키로 사용
                pass

            # 테마 → 종목
            if theme_id not in theme_stocks_map:
                theme_stocks_map[theme_id] = []
            if stock_code not in theme_stocks_map[theme_id]:
                theme_stocks_map[theme_id].append(stock_code)

            # 종목 → 테마
            if stock_code not in stock_themes_map:
                stock_themes_map[stock_code] = []
            if theme_id not in stock_themes_map[stock_code]:
                stock_themes_map[stock_code].append(theme_id)

        return theme_stocks_map, stock_themes_map

    def build_theme_maps_with_names(
        self,
        themes: list,
        theme_stocks: list,
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """
        테마-종목 매핑 구축 (테마명 기준)

        Args:
            themes: ThemeInfo 리스트
            theme_stocks: ThemeStock 리스트

        Returns:
            (theme_stocks_map, stock_themes_map)
            - theme_stocks_map: {테마명: [종목코드, ...]}
            - stock_themes_map: {종목코드: [테마명, ...]}
        """
        # 테마 ID → 테마명 매핑
        theme_id_to_name = {t.theme_id: t.name for t in themes}

        theme_stocks_map: dict[str, list[str]] = {}  # 테마명 → 종목들
        stock_themes_map: dict[str, list[str]] = {}  # 종목 → 테마명들

        for ts in theme_stocks:
            theme_name = theme_id_to_name.get(ts.theme_id, ts.theme_id)
            stock_code = ts.stock_code

            # 테마명 → 종목
            if theme_name not in theme_stocks_map:
                theme_stocks_map[theme_name] = []
            if stock_code not in theme_stocks_map[theme_name]:
                theme_stocks_map[theme_name].append(stock_code)

            # 종목 → 테마명
            if stock_code not in stock_themes_map:
                stock_themes_map[stock_code] = []
            if theme_name not in stock_themes_map[stock_code]:
                stock_themes_map[stock_code].append(theme_name)

        return theme_stocks_map, stock_themes_map

    # =========================================================================
    # 종목 분석 데이터 구축
    # =========================================================================

    def build_stocks_data(
        self,
        stock_codes: list[str],
        stock_names: dict[str, str],
        stock_themes_map: dict[str, list[str]],
        dart_data: dict[str, str] | None = None,
        news_data: dict[str, list] | None = None,
        progress_callback=None,
    ) -> list[StockAnalysisData]:
        """
        종목별 분석 데이터 구축

        Args:
            stock_codes: 종목 코드 리스트
            stock_names: {종목코드: 종목명}
            stock_themes_map: {종목코드: [테마명, ...]}
            dart_data: {종목코드: DART 사업개요 텍스트}
            news_data: {종목코드: [뉴스 기사 리스트]}
            progress_callback: 진행 콜백 (current, total)

        Returns:
            StockAnalysisData 리스트
        """
        self.logger.info(f"종목 분석 데이터 구축 시작: {len(stock_codes)}개 종목")

        dart_data = dart_data or {}
        news_data = news_data or {}

        results = []
        for i, code in enumerate(stock_codes):
            stock_data = self.analyze_stock(
                stock_code=code,
                stock_name=stock_names.get(code, ""),
                theme_tags=stock_themes_map.get(code, []),
                dart_text=dart_data.get(code),
                news_articles=news_data.get(code),
            )
            results.append(stock_data)

            if progress_callback:
                progress_callback(i + 1, len(stock_codes))

        self.logger.info(f"종목 분석 데이터 구축 완료: {len(results)}개")
        return results

    def analyze_stock(
        self,
        stock_code: str,
        stock_name: str,
        theme_tags: list[str] | None = None,
        dart_text: str | None = None,
        news_articles: list | None = None,
    ) -> StockAnalysisData:
        """
        단일 종목 분석

        Args:
            stock_code: 종목코드
            stock_name: 종목명
            theme_tags: 소속 테마명 리스트
            dart_text: DART 사업개요 텍스트
            news_articles: 뉴스 기사 리스트

        Returns:
            StockAnalysisData
        """
        data_sources = []
        business_summary = ""
        news_summary = ""

        # 테마 데이터
        if theme_tags:
            data_sources.append("theme")

        # DART 사업개요 LLM 요약
        if dart_text:
            data_sources.append("dart")
            if self._llm_client:
                business_summary = self._summarize_business(stock_name, dart_text)

        # 뉴스 LLM 요약
        if news_articles:
            data_sources.append("news")
            if self._llm_client:
                news_summary = self._summarize_news(stock_name, news_articles)

        return StockAnalysisData(
            stock_code=stock_code,
            stock_name=stock_name,
            theme_tags=theme_tags or [],
            business_summary=business_summary,
            news_summary=news_summary,
            data_sources=data_sources,
            created_at=datetime.now(),
        )

    # =========================================================================
    # LLM 요약 기능
    # =========================================================================

    def _summarize_business(self, stock_name: str, dart_text: str) -> str:
        """
        DART 사업개요 LLM 요약

        Args:
            stock_name: 종목명
            dart_text: 사업개요 전문

        Returns:
            요약 (3-5문장)
        """
        if not self._llm_client or not dart_text:
            return ""

        # 텍스트 길이 제한
        text = dart_text[:3000] if len(dart_text) > 3000 else dart_text

        prompt = f"""다음은 '{stock_name}'의 DART 사업보고서 중 '사업의 개요' 부분이야.
이 회사의 주요 사업 내용을 3-5문장으로 간결하게 요약해줘.
핵심 제품/서비스, 주요 시장, 사업 구조를 포함해.

[사업개요]
{text}

요약:"""

        try:
            result = self._llm_client.generate(prompt, max_tokens=500)
            return result.strip()
        except Exception as e:
            self.logger.warning(f"사업개요 요약 실패 [{stock_name}]: {e}")
            return ""

    def _summarize_news(self, stock_name: str, news_articles: list) -> str:
        """
        뉴스 LLM 요약

        Args:
            stock_name: 종목명
            news_articles: 뉴스 기사 리스트 (NewsArticle 또는 dict)

        Returns:
            요약 (3-5문장)
        """
        if not self._llm_client or not news_articles:
            return ""

        # 뉴스 내용 구성
        news_text = ""
        for i, article in enumerate(news_articles[:10], 1):
            if hasattr(article, 'title'):
                title = article.title
                summary = getattr(article, 'summary', '')
                date = getattr(article, 'published_at', '')
            else:
                title = article.get('title', '')
                summary = article.get('summary', '')
                date = article.get('published_at', '')

            news_text += f"{i}. [{date}] {title}\n"
            if summary:
                news_text += f"   {summary[:200]}\n"

        prompt = f"""다음은 '{stock_name}'에 대한 최근 뉴스 헤드라인과 요약이야.

[중요] 먼저 각 기사가 '{stock_name}'과 직접적으로 관련이 있는지 판단해.
- 종목명이 단순히 나열된 기사(예: "A,B,C,D 상승")는 관련성이 낮으므로 무시해.
- 해당 종목에 대한 구체적인 내용(실적, 계약, 사업, 기술 등)이 있는 기사만 분석해.

관련성 높은 뉴스를 바탕으로 이 종목의 최근 동향을 3-5문장으로 요약해줘.
주요 이슈, 시장 반응, 향후 전망을 포함해.
관련 뉴스가 없으면 "관련 뉴스 없음"이라고 답해.

[최근 뉴스]
{news_text}

뉴스 동향 요약:"""

        try:
            result = self._llm_client.generate(prompt, max_tokens=500)
            return result.strip()
        except Exception as e:
            self.logger.warning(f"뉴스 요약 실패 [{stock_name}]: {e}")
            return ""

    # =========================================================================
    # 유틸리티
    # =========================================================================

    def get_themes_by_stock(
        self,
        stock_code: str,
        stock_themes_map: dict[str, list[str]]
    ) -> list[str]:
        """종목이 속한 테마 목록 반환"""
        return stock_themes_map.get(stock_code, [])

    def get_stocks_by_theme(
        self,
        theme_name: str,
        theme_stocks_map: dict[str, list[str]]
    ) -> list[str]:
        """테마에 속한 종목 목록 반환"""
        return theme_stocks_map.get(theme_name, [])

    def summarize_themes(self, themes_data: list[ThemeData]) -> dict:
        """테마 데이터 요약"""
        total_stocks = sum(t.stock_count for t in themes_data)
        avg_stocks = total_stocks / len(themes_data) if themes_data else 0

        # 등락률 기준 상위/하위 테마
        sorted_by_change = sorted(themes_data, key=lambda x: x.change_rate, reverse=True)
        top_themes = [t.theme_name for t in sorted_by_change[:5]]
        bottom_themes = [t.theme_name for t in sorted_by_change[-5:]]

        return {
            "total_themes": len(themes_data),
            "total_stocks": total_stocks,
            "avg_stocks_per_theme": round(avg_stocks, 1),
            "top_themes_by_change": top_themes,
            "bottom_themes_by_change": bottom_themes,
        }

    def summarize_stocks(self, stocks_data: list[StockAnalysisData]) -> dict:
        """종목 분석 데이터 요약"""
        with_themes = sum(1 for s in stocks_data if "theme" in s.data_sources)
        with_dart = sum(1 for s in stocks_data if "dart" in s.data_sources)
        with_news = sum(1 for s in stocks_data if "news" in s.data_sources)
        with_summary = sum(1 for s in stocks_data if s.business_summary or s.news_summary)

        # 테마 분포
        theme_counts: dict[str, int] = {}
        total_theme_tags = 0
        for s in stocks_data:
            total_theme_tags += len(s.theme_tags)
            for theme in s.theme_tags:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1

        top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        avg_themes = total_theme_tags / len(stocks_data) if stocks_data else 0

        return {
            "total_stocks": len(stocks_data),
            "with_themes": with_themes,
            "with_dart_data": with_dart,
            "with_news_data": with_news,
            "with_llm_summary": with_summary,
            "avg_themes_per_stock": round(avg_themes, 1),
            "top_themes": [{"theme": t, "count": c} for t, c in top_themes],
        }


# =============================================================================
# 하위 호환성을 위한 별칭 (deprecated)
# =============================================================================

# 기존 코드와의 호환성을 위해 임시로 유지
# 향후 제거 예정
SectorClassifier = StockThemeAnalyzer


def _deprecated_warning():
    """Deprecated 경고 (한 번만 출력)"""
    import warnings
    warnings.warn(
        "SectorClassifier is deprecated. Use StockThemeAnalyzer instead.",
        DeprecationWarning,
        stacklevel=3
    )
