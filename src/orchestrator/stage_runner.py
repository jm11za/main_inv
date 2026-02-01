"""
Stage Runner: 개별 단계 실행기

각 레이어를 독립적으로 실행하고 결과를 반환
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
from enum import Enum
import sys

from src.core.logger import get_logger
from src.core.config import get_config


def print_progress(current: int, total: int, prefix: str = "", suffix: str = "", width: int = 30):
    """진행률 바 출력"""
    percent = current / total if total > 0 else 0
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    print(f"\r  {prefix} [{bar}] {current}/{total} {suffix}", end="", flush=True)
    if current >= total:
        print()  # 완료 시 줄바꿈


class StageStatus(Enum):
    """단계 상태"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """단계 실행 결과"""
    stage_name: str
    status: StageStatus
    data: Any = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metrics: dict = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """실행 시간 (초)"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0

    def to_dict(self) -> dict:
        return {
            "stage": self.stage_name,
            "status": self.status.value,
            "duration": f"{self.duration_seconds:.1f}s",
            "error": self.error,
            "metrics": self.metrics,
        }


class StageRunner:
    """
    개별 단계 실행기

    각 레이어의 로직을 캡슐화하고 에러 핸들링 제공

    사용법:
        runner = StageRunner()

        # Layer 1 실행
        result = runner.run_ingest(theme_codes, stock_codes)

        # Layer 3 실행
        result = runner.run_analysis(price_data, supply_data)
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.config = get_config()

    def _run_stage(
        self,
        stage_name: str,
        func: callable,
        *args,
        **kwargs
    ) -> StageResult:
        """공통 단계 실행 래퍼"""
        result = StageResult(
            stage_name=stage_name,
            status=StageStatus.RUNNING,
            started_at=datetime.now()
        )

        self.logger.info(f"[{stage_name}] 시작")

        try:
            data = func(*args, **kwargs)
            result.data = data
            result.status = StageStatus.SUCCESS
            self.logger.info(f"[{stage_name}] 완료")

        except Exception as e:
            result.status = StageStatus.FAILED
            result.error = str(e)
            self.logger.error(f"[{stage_name}] 실패: {e}")

        result.completed_at = datetime.now()
        return result

    # ========== Layer 1: Ingest ==========

    def run_ingest_themes(
        self,
        max_pages: int = 1,
        verbose: bool = True
    ) -> StageResult:
        """테마 데이터 수집"""
        from src.ingest import NaverThemeCrawler, ThemeService

        def execute():
            crawler = NaverThemeCrawler(min_delay=1.0, max_delay=2.0)
            themes = []
            all_stocks = {}

            # 테마 목록 수집
            if verbose:
                print(f"\n  [1-1] 테마 목록 크롤링...")

            for i, theme in enumerate(crawler.fetch_theme_list()):
                themes.append(theme)
                if verbose:
                    print(f"    [{i+1:2}] {theme.name}")
                if i >= (max_pages * 30) - 1:
                    break

            if verbose:
                print(f"  ✓ 테마 {len(themes)}개 수집 완료\n")

            # 테마별 종목 수집
            if verbose:
                print(f"  [1-1b] 테마별 종목 수집...")

            for i, theme in enumerate(themes):
                stocks = list(crawler.fetch_theme_stocks(theme.theme_id, theme.name))
                theme.stocks = stocks
                for s in stocks:
                    all_stocks[s.stock_code] = s.stock_name

                if verbose:
                    print_progress(i + 1, len(themes), prefix="", suffix=f"{theme.name[:15]}: {len(stocks)}개")

            if verbose:
                print(f"  ✓ 종목 {len(all_stocks)}개 수집 완료 (중복 제거)\n")

            return {
                "theme_count": len(themes),
                "themes": themes,
                "stock_codes": list(all_stocks.keys()),
                "stock_names": all_stocks,
            }

        return self._run_stage("Ingest:Themes", execute)

    def run_ingest_prices(
        self,
        stock_codes: list[str],
        stock_names: dict[str, str] = None,
        days: int = 120,
        verbose: bool = True
    ) -> StageResult:
        """주가 데이터 수집"""
        from src.ingest import PriceDataFetcher

        def execute():
            fetcher = PriceDataFetcher(lookback_days=days)
            results = {}
            failed = []

            if verbose:
                print(f"\n  [1-2] 주가 데이터 수집 ({len(stock_codes)}개 종목)...")

            for i, code in enumerate(stock_codes):
                try:
                    ohlcv = fetcher.fetch_stock_ohlcv(code)
                    if not ohlcv.empty:
                        results[code] = ohlcv
                    else:
                        failed.append(code)
                except Exception:
                    failed.append(code)

                if verbose:
                    name = stock_names.get(code, code)[:10] if stock_names else code
                    print_progress(i + 1, len(stock_codes), suffix=f"{name}")

            if verbose:
                print(f"  ✓ 주가 {len(results)}개 수집 완료 (실패: {len(failed)}개)\n")

            return {
                "stock_count": len(results),
                "price_data": results,
                "failed": failed,
            }

        return self._run_stage("Ingest:Prices", execute)

    def run_ingest_financials(
        self,
        stock_codes: list[str],
        year: int,
        stock_names: dict[str, str] = None,
        verbose: bool = True
    ) -> StageResult:
        """재무 데이터 수집"""
        from src.ingest import DartApiClient

        def execute():
            client = DartApiClient()
            results = {}
            failed = []  # (code, name, reason) 튜플 리스트
            excluded = []  # 재무제표 없는 종목 (SPAC 등) - 파이프라인에서 제외

            if verbose:
                print(f"\n  [1-3a] 재무 데이터 수집 ({len(stock_codes)}개 종목)...")

            import time
            DART_DELAY = 2.0  # DART API 딜레이 (2초)

            for i, code in enumerate(stock_codes):
                name = stock_names.get(code, code) if stock_names else code

                # DART API 딜레이 (Rate Limit 방지)
                if i > 0:
                    time.sleep(DART_DELAY)

                try:
                    data = client.get_comprehensive_financials(code, year)
                    if data.get("has_data"):
                        results[code] = data
                    else:
                        # 데이터 없음 - SPAC/신규상장 등으로 간주하여 제외
                        reason = f"재무제표 없음 ({year}~{year-3}년)"
                        excluded.append(code)
                        self.logger.warning(f"[DART] {name}({code}): {reason} → 분석 제외")
                except Exception as e:
                    # API 오류 - 로깅 후 실패 목록에 추가
                    reason = str(e)
                    failed.append((code, name, reason))
                    self.logger.error(f"[DART] {name}({code}): API 오류 - {reason}")

                if verbose:
                    display_name = name[:10] if len(name) > 10 else name
                    print_progress(i + 1, len(stock_codes), suffix=f"{display_name}")

            if verbose:
                print(f"  ✓ 재무 {len(results)}개 수집 완료 (제외: {len(excluded)}개, 오류: {len(failed)}개)\n")

            # 오류가 있으면 상세 로그 출력
            if failed:
                self.logger.error(f"[DART] 재무 조회 실패 종목 ({len(failed)}개):")
                for code, name, reason in failed:
                    self.logger.error(f"  - {name}({code}): {reason}")

            if excluded:
                self.logger.info(f"[DART] 재무제표 없어 제외된 종목 ({len(excluded)}개): {excluded}")

            return {
                "stock_count": len(results),
                "financial_data": results,
                "failed": [f[0] for f in failed],  # 오류 종목 코드만
                "excluded": excluded,  # 제외 종목
            }

        return self._run_stage("Ingest:Financials", execute)

    def run_ingest_supply_demand(
        self,
        stock_codes: list[str],
        stock_names: dict[str, str] = None,
        days: int = 20,
        verbose: bool = True
    ) -> StageResult:
        """수급/펀더멘탈 데이터 수집 (pykrx 기반)"""
        from src.ingest import PriceDataFetcher

        def execute():
            fetcher = PriceDataFetcher(lookback_days=days)
            fundamental_data = {}
            supply_data = {}
            failed = []

            if verbose:
                print(f"\n  [1-3b] 수급/펀더멘탈 데이터 수집 ({len(stock_codes)}개 종목)...")

            for i, code in enumerate(stock_codes):
                try:
                    # 펀더멘탈 (PBR, PER)
                    fund = fetcher.fetch_stock_fundamental(code)
                    # 거래대금
                    trading = fetcher.fetch_stock_trading_value(code, days=days)
                    # 수급 (외인/기관 순매수, 시총)
                    supply = fetcher.fetch_stock_supply_demand(code, days=days)

                    fundamental_data[code] = {
                        "pbr": fund.get("pbr", 0),
                        "per": fund.get("per", 0),
                        "trading_value": trading,
                    }
                    supply_data[code] = supply

                except Exception:
                    failed.append(code)

                if verbose:
                    name = stock_names.get(code, code)[:10] if stock_names else code
                    print_progress(i + 1, len(stock_codes), suffix=f"{name}")

            if verbose:
                print(f"  ✓ 펀더멘탈 {len(fundamental_data)}개, 수급 {len(supply_data)}개 수집 완료\n")

            return {
                "stock_count": len(fundamental_data),
                "fundamental_data": fundamental_data,
                "supply_data": supply_data,
                "failed": failed,
            }

        return self._run_stage("Ingest:SupplyDemand", execute)

    def run_ingest_news(
        self,
        stock_codes: list[str],
        stock_names: dict[str, str] = None,
        max_articles: int = 10,
        period: str = "1m",
        verbose: bool = True
    ) -> StageResult:
        """
        뉴스 데이터 수집 (NaverNewsSearchCrawler 사용)

        Args:
            stock_codes: 종목코드 리스트
            stock_names: {종목코드: 종목명} (필수 - 검색에 사용)
            max_articles: 종목당 최대 기사 수
            period: 기간 필터 ("1d", "1w", "1m", "6m", "1y")
            verbose: 진행 상황 출력 여부
        """
        from src.ingest.naver_news_search import NaverNewsSearchCrawler
        import time

        def execute():
            NEWS_DELAY = 4.0  # 뉴스 크롤링 딜레이 (4초 - 차단 방지)
            crawler = NaverNewsSearchCrawler(delay_seconds=NEWS_DELAY)
            results = {}

            if verbose:
                print(f"\n  [1-4] 뉴스 수집 ({len(stock_codes)}개 종목, 간격: {NEWS_DELAY}초)...")

            stock_names_map = stock_names or {}

            for i, code in enumerate(stock_codes):
                # 뉴스 크롤링 딜레이
                if i > 0:
                    time.sleep(NEWS_DELAY)

                # 종목명으로 검색 (코드가 아님)
                name = stock_names_map.get(code, "")
                if not name:
                    self.logger.debug(f"종목명 없음, 스킵: {code}")
                    continue

                articles = crawler.search(
                    query=name,
                    period=period,
                    sort="recent",
                    max_results=max_articles
                )

                if articles:
                    results[code] = articles

                if verbose:
                    display_name = name[:8] if name else code
                    print_progress(i + 1, len(stock_codes), suffix=f"{display_name} ({len(articles)}건)")

            crawler.close()

            if verbose:
                total_articles = sum(len(v) for v in results.values())
                print(f"  ✓ 뉴스 {len(results)}개 종목, 총 {total_articles}건 수집 완료\n")

            return {
                "stock_count": len(results),
                "total_articles": sum(len(v) for v in results.values()),
                "news_data": results,
            }

        return self._run_stage("Ingest:News", execute)

    def run_ingest_business_overview(
        self,
        stock_codes: list[str],
        stock_names: dict[str, str] = None,
        verbose: bool = True
    ) -> StageResult:
        """DART 사업개요 수집"""
        from src.ingest import DartApiClient
        import time

        def execute():
            DART_DELAY = 2.0  # DART API 딜레이 (2초)
            client = DartApiClient()
            results = {}
            failed = []

            if verbose:
                print(f"\n  [1-5] DART 사업개요 수집 ({len(stock_codes)}개 종목)...")

            for i, code in enumerate(stock_codes):
                # DART API 딜레이 (Rate Limit 방지)
                if i > 0:
                    time.sleep(DART_DELAY)

                try:
                    overview = client.fetch_business_overview(code)
                    if overview:
                        results[code] = overview
                    else:
                        failed.append(code)
                except Exception as e:
                    failed.append(code)
                    self.logger.debug(f"사업개요 조회 실패 [{code}]: {e}")

                if verbose:
                    name = stock_names.get(code, code)[:10] if stock_names else code
                    print_progress(i + 1, len(stock_codes), suffix=f"{name}")

            if verbose:
                print(f"  ✓ 사업개요 {len(results)}개 수집 완료 (실패: {len(failed)}개)\n")

            return {
                "stock_count": len(results),
                "business_overview": results,
                "failed": failed,
            }

        return self._run_stage("Ingest:BusinessOverview", execute)

    def run_ingest_community(
        self,
        stock_codes: list[str],
        stock_names: dict[str, str],
        max_posts: int = 30
    ) -> StageResult:
        """커뮤니티 데이터 수집"""
        from src.ingest import CommunityCrawler

        def execute():
            crawler = CommunityCrawler(max_posts=max_posts, max_pages=2)
            results = {}

            for code in stock_codes:
                name = stock_names.get(code, "")
                data = crawler.get_sentiment_data(code, name)
                if data.get("community_posts"):
                    results[code] = data

            return {
                "stock_count": len(results),
                "community_data": results,
            }

        return self._run_stage("Ingest:Community", execute)

    def run_ingest_discussion(
        self,
        stock_codes: list[str],
        stock_names: dict[str, str] = None,
        max_posts: int = 30,
        verbose: bool = True
    ) -> StageResult:
        """
        네이버 금융 토론방 수집 (Sentiment Reader B 접근법)

        Args:
            stock_codes: 종목 코드 리스트
            stock_names: 종목명 매핑 (출력용)
            max_posts: 종목당 최대 수집 글 수
            verbose: 진행 상황 출력 여부

        Returns:
            StageResult with discussion_data
        """
        from src.ingest import DiscussionCrawler

        def execute():
            crawler = DiscussionCrawler(max_posts=max_posts)
            results = {}

            if verbose:
                print(f"\n  [1-5b] 토론방 수집 ({len(stock_codes)}개 종목)...")

            for i, code in enumerate(stock_codes):
                try:
                    # Sentiment Reader용 데이터 수집
                    data = crawler.fetch_for_sentiment_analysis(
                        stock_code=code,
                        stock_name=stock_names.get(code, "") if stock_names else ""
                    )
                    results[code] = data

                except Exception as e:
                    self.logger.debug(f"[{code}] 토론방 수집 실패: {e}")
                    results[code] = {
                        "community_posts": [],
                        "mention_count": 0,
                        "sentiment_data": {},
                    }

                if verbose:
                    name = stock_names.get(code, code)[:10] if stock_names else code
                    print_progress(i + 1, len(stock_codes), suffix=f"{name}")

            crawler.close()

            if verbose:
                valid_count = sum(1 for v in results.values() if v.get("community_posts"))
                print(f"  ✓ 토론방 {valid_count}/{len(stock_codes)}개 수집 완료\n")

            return {
                "stock_count": len(results),
                "discussion_data": results,
            }

        return self._run_stage("Ingest:Discussion", execute)

    # ========== Layer 2: Processing ==========

    def run_processing(
        self,
        news_data: dict,
        community_data: dict
    ) -> StageResult:
        """데이터 처리 및 키워드 추출"""
        from src.processing import Preprocessor, LLMExtractor

        def execute():
            preprocessor = Preprocessor()
            extractor = LLMExtractor()

            processed = {}

            # 뉴스 처리
            for code, articles in news_data.items():
                headlines = [a.title for a in articles] if articles else []
                cleaned = preprocessor.clean_headlines(headlines)
                keywords = extractor.extract_keywords_fallback("\n".join(cleaned))

                processed[code] = {
                    "headlines": cleaned,
                    "keywords": keywords,
                    "community_posts": community_data.get(code, {}).get("community_posts", []),
                }

            return {
                "stock_count": len(processed),
                "processed_data": processed,
            }

        return self._run_stage("Processing", execute)

    def run_sector_labeling(
        self,
        stock_codes: list[str],
        stock_names: dict[str, str],
        theme_data: dict[str, list[str]],
        dart_data: dict[str, str] | None = None,
        news_data: dict[str, list[str]] | None = None,
        save_to_db: bool = True,
    ) -> StageResult:
        """
        종합 섹터 라벨링

        Args:
            stock_codes: 종목 코드 리스트
            stock_names: 종목코드 → 종목명 매핑
            theme_data: 종목코드 → 소속 테마명 리스트
            dart_data: 종목코드 → DART 사업보고서 텍스트 (선택)
            news_data: 종목코드 → 뉴스 헤드라인 리스트 (선택)
            save_to_db: DB 저장 여부 (기본 True)
        """
        from src.processing import SectorLabeler

        def execute():
            labeler = SectorLabeler(use_llm=True)
            results = {}
            sector_types = {}

            for code in stock_codes:
                name = stock_names.get(code, "")
                themes = theme_data.get(code, [])
                dart_text = dart_data.get(code, "") if dart_data else None
                headlines = news_data.get(code, []) if news_data else None

                label = labeler.label_stock(
                    stock_code=code,
                    stock_name=name,
                    theme_names=themes if themes else None,
                    dart_business_text=dart_text if dart_text else None,
                    news_headlines=headlines if headlines else None,
                )

                results[code] = label
                # Track 타입 결정 (성장 섹터 = B, 그 외 = A)
                sector_types[code] = "B" if label.is_growth_sector else "A"

            # DB 저장
            labels_list = list(results.values())
            if save_to_db and labels_list:
                saved = labeler.save_to_db(labels_list)
                self.logger.info(f"섹터 라벨 {saved}개 DB 저장")

            summary = labeler.summarize_labels(labels_list)

            return {
                "stock_count": len(results),
                "sector_labels": results,
                "sector_types": sector_types,
                "summary": summary,
            }

        return self._run_stage("SectorLabeling", execute)

    # ========== Layer 3: Analysis ==========

    def run_analysis(
        self,
        price_data: dict,
        supply_data: dict,
        theme_stocks: dict[str, list[str]]
    ) -> StageResult:
        """섹터/종목 분석"""
        from src.analysis import FlowCalculator, BreadthCalculator, TrendCalculator, TierClassifier

        def execute():
            flow_calc = FlowCalculator()
            breadth_calc = BreadthCalculator()
            trend_calc = TrendCalculator()
            tier_classifier = TierClassifier()

            sector_metrics = {}

            for theme_name, stock_codes in theme_stocks.items():
                # 섹터 수급 계산
                sector_supply = []
                for code in stock_codes:
                    if code in supply_data:
                        sector_supply.append(supply_data[code])

                s_flow = flow_calc.calculate_sector(sector_supply) if sector_supply else 0

                # 종목별 추세 계산
                stock_trends = []
                for code in stock_codes:
                    if code in price_data:
                        trend = trend_calc.calculate_from_df(price_data[code])
                        stock_trends.append(trend)

                s_trend = sum(stock_trends) / len(stock_trends) if stock_trends else 0

                # Breadth 계산
                s_breadth = breadth_calc.calculate_sector(
                    [{"above_ma": t > 0} for t in stock_trends]
                ) if stock_trends else 0

                sector_metrics[theme_name] = {
                    "s_flow": s_flow,
                    "s_breadth": s_breadth,
                    "s_trend": s_trend,
                    "stock_count": len(stock_codes),
                }

            # Tier 분류
            tier_results = tier_classifier.rank_sectors(sector_metrics)

            return {
                "sector_count": len(sector_metrics),
                "sector_metrics": sector_metrics,
                "tier_results": tier_results,
            }

        return self._run_stage("Analysis", execute)

    # ========== Layer 3.5: Filtering ==========

    def run_filtering(
        self,
        stocks: list[dict],
        sector_types: dict[str, str]
    ) -> StageResult:
        """이원화 필터링"""
        from src.filtering import FilterRouter

        def execute():
            router = FilterRouter()
            results = {}

            for stock in stocks:
                code = stock.get("stock_code")
                sector = stock.get("sector", "")
                sector_type = sector_types.get(sector, "A")

                result = router.filter_stock(stock, sector_type)
                results[code] = {
                    "passed": result.passed,
                    "reason": result.reason,
                    "metrics": result.metrics,
                    "track": f"Track{sector_type}",
                }

            passed = [c for c, r in results.items() if r["passed"]]

            return {
                "total": len(stocks),
                "passed": len(passed),
                "filter_results": results,
                "passed_codes": passed,
            }

        return self._run_stage("Filtering", execute)

    # ========== Layer 4: Scoring ==========

    def run_scoring(
        self,
        passed_stocks: list[str],
        financial_data: dict,
        technical_data: dict,
        track_types: dict[str, str]
    ) -> StageResult:
        """종목 점수화"""
        from src.scoring import StockScorer
        from src.core.interfaces import TrackType

        def execute():
            scorer = StockScorer()
            results = {}

            for code in passed_stocks:
                fin = financial_data.get(code, {})
                tech = technical_data.get(code, {})
                track = TrackType.TRACK_A if track_types.get(code) == "A" else TrackType.TRACK_B

                score_result = scorer.score_from_dict(
                    stock_code=code,
                    financial_dict=fin,
                    technical_dict=tech,
                    track=track,
                )

                results[code] = {
                    "total_score": score_result.total_score,
                    "financial_score": score_result.financial_score,
                    "technical_score": score_result.technical_score,
                    "breakdown": score_result.breakdown,
                }

            # 순위 정렬
            ranked = sorted(
                results.items(),
                key=lambda x: x[1]["total_score"],
                reverse=True
            )

            return {
                "stock_count": len(results),
                "score_results": results,
                "ranking": [code for code, _ in ranked],
            }

        return self._run_stage("Scoring", execute)

    # ========== Layer 5: Decision ==========

    def run_decision(
        self,
        stock_codes: list[str],
        tier_data: dict,
        news_data: dict,
        community_data: dict,
        discussion_data: dict = None,
        price_indicators: dict = None,
        stock_names: dict = None,
    ) -> StageResult:
        """
        최종 판정 (Skeptic + Sentiment A+B+C)

        Args:
            stock_codes: 분석 대상 종목 코드
            tier_data: 섹터 Tier 정보
            news_data: 뉴스 데이터 (headlines 포함)
            community_data: 커뮤니티 데이터 (legacy)
            discussion_data: 토론방 데이터 (B 접근법)
            price_indicators: 가격 지표 (RSI, 수익률, 거래량)
            stock_names: 종목명 매핑

        Returns:
            StageResult with recommendations
        """
        from src.decision import LLMAnalyzer
        from src.core.interfaces import Tier, TrackType

        def execute():
            analyzer = LLMAnalyzer()
            results = {}

            discussion_data_safe = discussion_data or {}
            price_indicators_safe = price_indicators or {}
            stock_names_safe = stock_names or {}

            for code in stock_codes:
                # Tier 정보
                tier_info = tier_data.get(code, {})
                tier_value = tier_info.get("tier", 2)
                tier = Tier.TIER_1 if tier_value == 1 else Tier.TIER_2 if tier_value == 2 else Tier.TIER_3

                # Track 타입
                track = tier_info.get("track_type", TrackType.TRACK_A)

                # Skeptic 입력: 뉴스 헤드라인
                news_info = news_data.get(code, {})
                headlines = news_info.get("headlines", [])

                # Sentiment B: 토론방 데이터
                disc_info = discussion_data_safe.get(code, {})
                community_posts = disc_info.get("community_posts", [])
                sentiment_data = disc_info.get("sentiment_data", {})
                discussion_sentiment_ratio = sentiment_data.get("sentiment_ratio", 0.0)
                discussion_likes = sentiment_data.get("likes", 0)
                discussion_dislikes = sentiment_data.get("dislikes", 0)

                # Legacy 커뮤니티 데이터 폴백
                if not community_posts:
                    legacy_comm = community_data.get(code, {})
                    community_posts = legacy_comm.get("community_posts", [])

                # Sentiment C: 가격 지표
                price_info = price_indicators_safe.get(code, {})
                rsi = price_info.get("rsi", 50.0)
                return_1w = price_info.get("return_1w", 0.0)
                return_1m = price_info.get("return_1m", 0.0)
                volume_ratio = price_info.get("volume_ratio", 1.0)

                # 분석 실행
                analysis = analyzer.analyze(
                    stock_code=code,
                    stock_name=stock_names_safe.get(code, ""),
                    tier=tier,
                    track_type=track,
                    total_score=tier_info.get("total_score", 0.0),
                    # Skeptic 입력
                    news_headlines=headlines,
                    # Sentiment A+B+C 입력
                    community_posts=community_posts,
                    discussion_sentiment_ratio=discussion_sentiment_ratio,
                    discussion_likes=discussion_likes,
                    discussion_dislikes=discussion_dislikes,
                    rsi=rsi,
                    return_1w=return_1w,
                    return_1m=return_1m,
                    volume_ratio=volume_ratio,
                )

                results[code] = analysis

            # 추천 종목 추출
            recommendations = analyzer.get_top_picks(list(results.values()))

            return {
                "stock_count": len(results),
                "decision_results": results,
                "recommendations": recommendations,
            }

        return self._run_stage("Decision", execute)
