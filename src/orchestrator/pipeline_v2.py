"""
Pipeline v2.0: Top-Down 테마 중심 파이프라인

아키텍처 워크플로우 v2.0:
① 테마 데이터셋 구축 - 종목-테마 매핑, LLM 요약 (StockThemeAnalyzer)
② 테마 Type 분류 - 실적형(A) / 성장형(B) (SectorTypeAnalyzer)
③ 테마 우선순위 - 상위 테마 선정, 하위 테마 배제 (SectorPrioritizer)
④ 종목 선정 - 상위 테마 내 투자 후보 선정
⑤ 종목 재검증 - Skeptic + Sentiment 분석 → 최종 판정

변경사항 (v2.0):
- SectorCategory Enum 삭제 → 네이버 테마명 그대로 사용
- 섹터 → 테마 용어 통일 (1종목 N테마 관계 지원)
- 보조 섹터 삭제 → LLM 요약으로 대체
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import sys

from src.core.logger import get_logger, LoggerService
from src.core.config import get_config
from src.core.interfaces import SectorType, TrackType
from src.core.preflight import PreflightChecker, PreflightResult
from src.orchestrator.stage_runner import StageResult, StageStatus, print_progress


# 로거 초기화
LoggerService.configure(level="INFO", file_enabled=True)


@dataclass
class PipelineV2Result:
    """파이프라인 v2.0 실행 결과"""
    success: bool
    started_at: datetime
    completed_at: datetime | None = None
    stages: dict[str, StageResult] = field(default_factory=dict)
    final_decisions: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def duration_seconds(self) -> float:
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0

    def to_summary(self) -> dict:
        return {
            "success": self.success,
            "duration": f"{self.duration_seconds:.1f}s",
            "stages": {name: s.to_dict() for name, s in self.stages.items()},
            "decisions_count": len(self.final_decisions),
            "summary": self.summary,
            "error": self.error,
        }


class PipelineV2:
    """
    Top-Down 섹터 중심 파이프라인 v2.0

    사용법:
        pipeline = PipelineV2()

        # 전체 실행
        result = pipeline.run(
            max_themes=30,
            top_sectors=5,
            top_per_sector=3,
        )

        # 단계별 실행
        result = pipeline.run_stage("03_sector_priority")
    """

    # 단계 정의
    STAGES = [
        "00_data_collect",      # 데이터 수집 (테마, 주가, 재무, 뉴스)
        "01_sector_classify",   # 섹터 분류
        "02_sector_type",       # 섹터 Type 분류
        "03_sector_priority",   # 섹터 우선순위
        "04_stock_selection",   # 종목 선정
        "05_stock_verify",      # 재검증 및 최종 판정
    ]

    def __init__(self, model: str | None = None):
        self.logger = get_logger(self.__class__.__name__)
        self.config = get_config()

        # 모델 자동 선택 (deepseek 우선, GLM 제외)
        if model is None:
            from src.core.preflight import select_model
            self.model = select_model()
        else:
            self.model = model

        # 캐시 데이터
        self._cache: dict[str, Any] = {}

        # 컴포넌트 (lazy init)
        self._stage_saver = None
        self._ollama_client = None
        self._claude_client = None

    @property
    def stage_saver(self):
        """StageSaver 지연 로드"""
        if self._stage_saver is None:
            from src.output import StageSaver
            self._stage_saver = StageSaver()
        return self._stage_saver

    @property
    def ollama_client(self):
        """Ollama 클라이언트 지연 로드"""
        if self._ollama_client is None:
            try:
                from src.llm import OllamaClient
                self._ollama_client = OllamaClient(model=self.model)
            except Exception as e:
                self.logger.warning(f"Ollama 클라이언트 초기화 실패: {e}")
        return self._ollama_client

    @property
    def claude_client(self):
        """Claude CLI 클라이언트 지연 로드"""
        if self._claude_client is None:
            try:
                from src.llm import ClaudeCliClient
                self._claude_client = ClaudeCliClient(timeout=120)
            except Exception as e:
                self.logger.warning(f"Claude 클라이언트 초기화 실패: {e}")
        return self._claude_client

    def preflight_check(self, warmup_timeout: float = 300.0) -> PreflightResult:
        """사전 검사 (모든 항목 필수)"""
        checker = PreflightChecker()
        return checker.run(
            model=self.model,
            warmup_timeout=warmup_timeout,
        )

    def run(
        self,
        testmode: bool | None = None,
        max_themes: int | None = None,
        max_stocks: int | None = None,
        top_sectors: int | None = None,
        top_per_sector: int | None = None,
        max_candidates: int | None = None,
        year: int | None = None,
        save_stages: bool | None = None,
        skip_preflight: bool = False,
        verbose: bool | None = None,
    ) -> PipelineV2Result:
        """
        전체 파이프라인 실행

        Args:
            testmode: 테스트 모드 (1페이지 30개 테마만 조회)
            max_themes: 최대 테마 수 (testmode=True이면 30으로 고정)
            max_stocks: 최대 종목 수
            top_sectors: 선정할 상위 섹터 수
            top_per_sector: 섹터당 선정 종목 수
            max_candidates: 최대 후보 종목 수
            year: 재무 데이터 기준 연도
            save_stages: 단계별 결과 저장 여부
            skip_preflight: Preflight 검사 건너뛰기
            verbose: 상세 출력

        Returns:
            PipelineV2Result
        """
        # config에서 기본값 로드
        pipeline_config = self.config.get_section("pipeline_v2") or {}

        # 파라미터 결정 (인자 > config > 기본값)
        testmode = testmode if testmode is not None else pipeline_config.get("testmode", False)
        # max_themes: 사용자 지정 > testmode 기본값(30) > config > 300
        if max_themes is not None:
            pass  # 사용자 지정값 사용
        elif testmode:
            max_themes = 30
        else:
            max_themes = pipeline_config.get("max_themes", 300)
        # max_stocks도 testmode일 때는 줄임
        if max_stocks is not None:
            pass
        elif testmode:
            max_stocks = 50  # testmode 기본값
        else:
            max_stocks = pipeline_config.get("max_stocks", 500)
        top_sectors = top_sectors or pipeline_config.get("top_sectors", 5)
        top_per_sector = top_per_sector or pipeline_config.get("top_per_sector", 3)
        max_candidates = max_candidates or pipeline_config.get("max_candidates", 15)
        save_stages = save_stages if save_stages is not None else pipeline_config.get("save_stages", True)
        verbose = verbose if verbose is not None else pipeline_config.get("verbose", True)

        if year is None:
            year = datetime.now().year

        if testmode:
            self.logger.info(f"*** 테스트 모드: {max_themes}개 테마, {max_stocks}개 종목 제한 ***")

        result = PipelineV2Result(
            success=False,
            started_at=datetime.now()
        )

        # Preflight
        if not skip_preflight:
            preflight = self.preflight_check()
            if not preflight.passed:
                self.logger.error("Preflight Check 실패")
                result.error = f"Preflight 실패: {preflight.get_failures()}"
                result.completed_at = datetime.now()
                return result

        self.logger.info("=" * 60)
        self.logger.info("파이프라인 v2.0 시작 (Top-Down 섹터 분석)")
        self.logger.info("=" * 60)

        try:
            # ===== Stage 0: 데이터 수집 =====
            stage_result = self._run_data_collect(
                max_themes=max_themes,
                max_stocks=max_stocks,
                year=year,
                verbose=verbose,
            )
            result.stages["00_data_collect"] = stage_result
            if stage_result.status == StageStatus.FAILED:
                raise Exception(f"데이터 수집 실패: {stage_result.error}")
            if save_stages:
                self.stage_saver.save_stage("00_data_collect", stage_result.data, {"max_themes": max_themes})

            # ===== Stage 1: 섹터 분류 =====
            stage_result = self._run_sector_classify(verbose=verbose)
            result.stages["01_sector_classify"] = stage_result
            if stage_result.status == StageStatus.FAILED:
                raise Exception(f"섹터 분류 실패: {stage_result.error}")
            if save_stages:
                self.stage_saver.save_stage("01_sector_classify", stage_result.data.get("results", []))

            # ===== Stage 2: 섹터 Type 분류 =====
            stage_result = self._run_sector_type(verbose=verbose)
            result.stages["02_sector_type"] = stage_result
            if stage_result.status == StageStatus.FAILED:
                raise Exception(f"섹터 Type 분류 실패: {stage_result.error}")
            if save_stages:
                self.stage_saver.save_stage("02_sector_type", stage_result.data.get("results", []))

            # ===== Stage 3: 섹터 우선순위 =====
            stage_result = self._run_sector_priority(
                top_n=top_sectors,
                verbose=verbose,
            )
            result.stages["03_sector_priority"] = stage_result
            if stage_result.status == StageStatus.FAILED:
                raise Exception(f"섹터 우선순위 실패: {stage_result.error}")
            if save_stages:
                self.stage_saver.save_stage("03_sector_priority", stage_result.data.get("results", []))

            # 선정된 섹터 확인
            selected_sectors = stage_result.data.get("selected_sectors", [])
            if not selected_sectors:
                self.logger.warning("선정된 섹터 없음 - 파이프라인 종료")
                result.success = True
                result.completed_at = datetime.now()
                return result

            # ===== Stage 4: 종목 선정 =====
            stage_result = self._run_stock_selection(
                selected_sectors=selected_sectors,
                top_per_sector=top_per_sector,
                max_total=max_candidates,
                verbose=verbose,
            )
            result.stages["04_stock_selection"] = stage_result
            if stage_result.status == StageStatus.FAILED:
                raise Exception(f"종목 선정 실패: {stage_result.error}")
            if save_stages:
                self.stage_saver.save_stage("04_stock_selection", stage_result.data.get("results", []))

            # 선정된 종목 확인
            selected_stocks = stage_result.data.get("selected_stocks", [])
            if not selected_stocks:
                self.logger.warning("선정된 종목 없음 - 파이프라인 종료")
                result.success = True
                result.completed_at = datetime.now()
                return result

            # ===== Stage 5: 재검증 및 최종 판정 =====
            stage_result = self._run_stock_verify(
                selected_stocks=selected_stocks,
                verbose=verbose,
            )
            result.stages["05_stock_verify"] = stage_result
            if save_stages:
                self.stage_saver.save_stage("05_stock_verify", stage_result.data.get("results", []))

            # 최종 결과
            result.final_decisions = stage_result.data.get("results", [])
            result.success = True
            result.completed_at = datetime.now()

            # ===== 결과 집계 및 저장 =====
            if save_stages:
                aggregated = self.stage_saver.aggregate_all_stages()
                self.stage_saver.save_daily_report(aggregated)

                # 텔레그램 메시지 생성 및 저장
                from src.output import ReportGenerator
                generator = ReportGenerator()
                telegram_msg = generator.format_telegram_from_stages(aggregated)
                self.stage_saver.save_telegram_message(telegram_msg)

            # 요약 생성
            result.summary = self._generate_summary(result)

            self.logger.info("=" * 60)
            self.logger.info(f"파이프라인 완료: {result.duration_seconds:.1f}초")
            self.logger.info(f"최종 판정: {len(result.final_decisions)}개")
            self.logger.info("=" * 60)

        except Exception as e:
            result.success = False
            result.error = str(e)
            result.completed_at = datetime.now()
            self.logger.error(f"파이프라인 실패: {e}")

        return result

    # =========================================================================
    # 개별 단계 실행
    # =========================================================================

    def _run_data_collect(
        self,
        max_themes: int,
        max_stocks: int,
        year: int,
        verbose: bool,
    ) -> StageResult:
        """Stage 0: 데이터 수집"""
        from src.orchestrator.stage_runner import StageRunner

        runner = StageRunner()
        result = StageResult(
            stage_name="00_data_collect",
            status=StageStatus.RUNNING,
            started_at=datetime.now()
        )

        try:
            self.logger.info("[Stage 0] 데이터 수집 시작")

            # 0-1. 테마 수집
            theme_result = runner.run_ingest_themes(max_pages=max_themes // 30 + 1, verbose=verbose)
            if theme_result.status == StageStatus.FAILED:
                raise Exception(theme_result.error)

            all_themes = theme_result.data.get("themes", [])
            themes = all_themes[:max_themes]  # max_themes 제한 적용
            stock_names = theme_result.data.get("stock_names", {})
            stock_codes = list(stock_names.keys())[:max_stocks]

            self._cache["themes"] = themes
            self._cache["stock_codes"] = stock_codes
            self._cache["stock_names"] = stock_names

            # 0-2. 주가 수집
            price_result = runner.run_ingest_prices(stock_codes, stock_names, verbose=verbose)
            self._cache["price_data"] = price_result.data.get("price_data", {})

            # 0-3. 재무 수집
            financial_result = runner.run_ingest_financials(stock_codes, year, stock_names, verbose=verbose)
            self._cache["financial_data"] = financial_result.data.get("financial_data", {})

            # 0-4. 수급/펀더멘탈 수집
            supply_result = runner.run_ingest_supply_demand(stock_codes, stock_names, verbose=verbose)
            self._cache["fundamental_data"] = supply_result.data.get("fundamental_data", {})
            self._cache["supply_data"] = supply_result.data.get("supply_data", {})

            # 0-5. 뉴스 수집
            news_result = runner.run_ingest_news(stock_codes)
            self._cache["news_data"] = news_result.data.get("news_data", {})

            # 0-6. DART 사업개요 수집 (보조 섹터 분류용)
            overview_result = runner.run_ingest_business_overview(stock_codes, stock_names, verbose=verbose)
            self._cache["business_overview"] = overview_result.data.get("business_overview", {})

            result.status = StageStatus.SUCCESS
            result.data = {
                "theme_count": len(themes),
                "stock_count": len(stock_codes),
                "price_count": len(self._cache["price_data"]),
                "financial_count": len(self._cache["financial_data"]),
            }

        except Exception as e:
            result.status = StageStatus.FAILED
            result.error = str(e)
            self.logger.error(f"[Stage 0] 실패: {e}")

        result.completed_at = datetime.now()
        return result

    def _run_sector_classify(self, verbose: bool) -> StageResult:
        """Stage 1: 테마 데이터셋 구축 (v2.0)"""
        from src.sector import StockThemeAnalyzer

        result = StageResult(
            stage_name="01_sector_classify",
            status=StageStatus.RUNNING,
            started_at=datetime.now()
        )

        try:
            self.logger.info("[Stage 1] 테마 데이터셋 구축 시작")

            analyzer = StockThemeAnalyzer(llm_client=self.ollama_client)
            themes = self._cache.get("themes", [])
            stock_codes = self._cache.get("stock_codes", [])
            stock_names = self._cache.get("stock_names", {})
            news_data = self._cache.get("news_data", {})
            business_overview = self._cache.get("business_overview", {})

            if verbose:
                print(f"\n  [1] 테마 데이터셋 구축 ({len(stock_codes)}개 종목)...")

            # 테마-종목 매핑 구축 (테마명 기준)
            # themes는 ThemeInfo 리스트, 각 theme에 stocks 속성 있음
            from src.sector.classifier import ThemeStockRef
            theme_stocks = []
            for theme in themes:
                if theme.stocks:
                    for stock in theme.stocks:
                        theme_stocks.append(ThemeStockRef(
                            theme_id=theme.theme_id,
                            stock_code=stock.stock_code
                        ))

            # 테마명 기준 매핑 생성
            theme_stocks_map, stock_themes_map = analyzer.build_theme_maps_with_names(
                themes, theme_stocks
            )

            # 종목별 분석 데이터 구축 (LLM 요약 포함)
            stocks_data = analyzer.build_stocks_data(
                stock_codes=stock_codes,
                stock_names=stock_names,
                stock_themes_map=stock_themes_map,
                dart_data=business_overview,
                news_data=news_data,
                progress_callback=lambda c, t: print_progress(c, t) if verbose else None,
            )

            if verbose:
                print_progress(len(stock_codes), len(stock_codes))

            # 결과 저장
            self._cache["stocks_data"] = stocks_data
            self._cache["stock_themes_map"] = stock_themes_map
            self._cache["theme_stocks_map"] = theme_stocks_map

            # 테마별 종목 그룹핑 (테마명 문자열 기준)
            self._cache["sector_stocks"] = theme_stocks_map  # {테마명: [종목코드, ...]}

            # 요약
            summary = analyzer.summarize_stocks(stocks_data)

            result.status = StageStatus.SUCCESS
            result.data = {
                "stock_count": len(stocks_data),
                "theme_count": len(theme_stocks_map),
                "results": [s.to_dict() for s in stocks_data],
                "summary": summary,
            }

            self.logger.info(f"[Stage 1] 완료: {len(theme_stocks_map)}개 테마, {len(stocks_data)}개 종목")

        except Exception as e:
            result.status = StageStatus.FAILED
            result.error = str(e)
            self.logger.error(f"[Stage 1] 실패: {e}")

        result.completed_at = datetime.now()
        return result

    def _run_sector_type(self, verbose: bool) -> StageResult:
        """Stage 2: 테마 Type A/B 분류 (v2.0)"""
        from src.sector import SectorTypeAnalyzer

        result = StageResult(
            stage_name="02_sector_type",
            status=StageStatus.RUNNING,
            started_at=datetime.now()
        )

        try:
            self.logger.info("[Stage 2] 테마 Type 분류 시작")

            analyzer = SectorTypeAnalyzer(llm_client=self.ollama_client)
            theme_stocks_map = self._cache.get("sector_stocks", {})  # {테마명: [종목코드, ...]}

            # 테마명 리스트 (문자열)
            theme_names = list(theme_stocks_map.keys())

            if verbose:
                print(f"\n  [2] 테마 Type 분류 ({len(theme_names)}개 테마)...")

            # 배치 분류 (테마명 문자열 사용)
            type_results = analyzer.analyze_batch(theme_names)

            # 결과 저장 (테마명 기준)
            self._cache["theme_types"] = {r.theme_name: r for r in type_results}
            theme_type_map = {r.theme_name: r.sector_type for r in type_results}
            self._cache["theme_type_map"] = theme_type_map
            # 하위 호환성 (deprecated)
            self._cache["sector_type_map"] = theme_type_map

            result.status = StageStatus.SUCCESS
            result.data = {
                "theme_count": len(type_results),
                "results": [r.to_dict() for r in type_results],
                "summary": analyzer.summarize(type_results),
            }

            # 요약 출력
            type_a = sum(1 for r in type_results if r.sector_type == SectorType.TYPE_A)
            type_b = len(type_results) - type_a
            self.logger.info(f"[Stage 2] 완료: Type A {type_a}개, Type B {type_b}개")

        except Exception as e:
            result.status = StageStatus.FAILED
            result.error = str(e)
            self.logger.error(f"[Stage 2] 실패: {e}")

        result.completed_at = datetime.now()
        return result

    def _run_sector_priority(self, top_n: int, verbose: bool) -> StageResult:
        """Stage 3: 테마 우선순위 결정 (v2.0)"""
        from src.sector import SectorPrioritizer
        from src.sector.prioritizer import ThemeMetrics

        result = StageResult(
            stage_name="03_sector_priority",
            status=StageStatus.RUNNING,
            started_at=datetime.now()
        )

        try:
            self.logger.info("[Stage 3] 테마 우선순위 결정 시작")

            prioritizer = SectorPrioritizer(llm_client=self.ollama_client)
            theme_stocks_map = self._cache.get("sector_stocks", {})  # {테마명: [종목코드, ...]}
            theme_type_map = self._cache.get("theme_type_map", {})   # {테마명: SectorType}

            if verbose:
                print(f"\n  [3] 테마 우선순위 결정 ({len(theme_stocks_map)}개 테마)...")

            # 테마별 메트릭 계산
            theme_metrics = []
            for theme_name, stock_codes in theme_stocks_map.items():
                sector_type = theme_type_map.get(theme_name)
                metrics = self._calculate_theme_metrics(theme_name, stock_codes, sector_type)
                theme_metrics.append(metrics)

            # 우선순위 결정
            priority_results = prioritizer.prioritize(
                theme_metrics=theme_metrics,
                top_n=top_n,
            )

            # 결과 저장 (테마명 문자열 사용)
            self._cache["priority_results"] = priority_results
            selected_themes = [r.theme_name for r in priority_results if r.is_selected]
            self._cache["selected_themes"] = selected_themes
            # 하위 호환성 (deprecated)
            self._cache["selected_sectors"] = selected_themes

            result.status = StageStatus.SUCCESS
            result.data = {
                "total_themes": len(priority_results),
                "selected_count": len(selected_themes),
                "selected_sectors": selected_themes,  # 테마명 문자열 리스트
                "results": [r.to_dict() for r in priority_results],
                "summary": prioritizer.summarize(priority_results),
            }

            self.logger.info(f"[Stage 3] 완료: {len(selected_themes)}개 테마 선정")

        except Exception as e:
            result.status = StageStatus.FAILED
            result.error = str(e)
            self.logger.error(f"[Stage 3] 실패: {e}")

        result.completed_at = datetime.now()
        return result

    def _run_stock_selection(
        self,
        selected_sectors: list,  # 테마명 문자열 리스트
        top_per_sector: int,
        max_total: int,
        verbose: bool,
    ) -> StageResult:
        """Stage 4: 종목 선정 (v2.0)"""
        from src.stock import CandidateSelector

        result = StageResult(
            stage_name="04_stock_selection",
            status=StageStatus.RUNNING,
            started_at=datetime.now()
        )

        try:
            self.logger.info("[Stage 4] 종목 선정 시작")

            selector = CandidateSelector()
            theme_stocks_map = self._cache.get("sector_stocks", {})  # {테마명: [종목코드, ...]}
            theme_type_map = self._cache.get("theme_type_map", {})   # {테마명: SectorType}
            stock_names = self._cache.get("stock_names", {})

            # 선정된 테마의 종목만 추출
            stocks_data = []
            processed_codes = set()  # 중복 방지

            for theme_name in selected_sectors:  # 테마명 문자열
                codes = theme_stocks_map.get(theme_name, [])
                for code in codes:
                    if code not in processed_codes:
                        stock_data = self._prepare_stock_data(code, theme_name)
                        stocks_data.append(stock_data)
                        processed_codes.add(code)

            if verbose:
                print(f"\n  [4] 종목 선정 ({len(stocks_data)}개 종목)...")

            # 선정 실행
            selection_results = selector.select(
                stocks_data=stocks_data,
                sector_type_map=theme_type_map,  # 테마명 기준 type map
                top_per_sector=top_per_sector,
                max_total=max_total,
            )

            # 선정된 종목 추출
            selected = selector.get_selected(selection_results)
            selected_stocks = [
                {
                    "stock_code": r.stock_code,
                    "stock_name": stock_names.get(r.stock_code, ""),
                    "sector": r.sector,  # 테마명 문자열
                    "sector_rank": r.sector_rank,
                    "track_type": r.track_type,
                    "total_score": r.total_score,
                    "filter_passed": r.filter_passed,
                }
                for r in selected
            ]

            self._cache["selected_stocks"] = selected_stocks
            self._cache["selection_results"] = selection_results

            result.status = StageStatus.SUCCESS
            result.data = {
                "total_processed": len(stocks_data),
                "selected_count": len(selected_stocks),
                "selected_stocks": selected_stocks,
                "results": [r.to_dict() for r in selection_results],
            }

            self.logger.info(f"[Stage 4] 완료: {len(selected_stocks)}개 종목 선정")

        except Exception as e:
            result.status = StageStatus.FAILED
            result.error = str(e)
            self.logger.error(f"[Stage 4] 실패: {e}")

        result.completed_at = datetime.now()
        return result

    def _run_stock_verify(
        self,
        selected_stocks: list[dict],
        verbose: bool,
    ) -> StageResult:
        """Stage 5: 재검증 및 최종 판정"""
        from src.verify import MaterialAnalyzer, SentimentAnalyzer, DecisionEngine
        from src.orchestrator.stage_runner import StageRunner

        result = StageResult(
            stage_name="05_stock_verify",
            status=StageStatus.RUNNING,
            started_at=datetime.now()
        )

        try:
            self.logger.info("[Stage 5] 재검증 시작")

            # 분석기 초기화
            material_analyzer = MaterialAnalyzer(llm_client=self.claude_client)
            sentiment_analyzer = SentimentAnalyzer(llm_client=self.claude_client)
            decision_engine = DecisionEngine()

            stock_names = self._cache.get("stock_names", {})
            news_data = self._cache.get("news_data", {})

            # 커뮤니티 데이터 수집 (선정된 종목만)
            if verbose:
                print(f"\n  [5-1] 커뮤니티 수집 ({len(selected_stocks)}개 종목)...")

            runner = StageRunner()
            stock_codes = [s["stock_code"] for s in selected_stocks]
            discussion_result = runner.run_ingest_discussion(stock_codes, stock_names, verbose=verbose)
            discussion_data = discussion_result.data.get("discussion_data", {})

            # 재료 분석 데이터 준비
            material_data = []
            for stock in selected_stocks:
                code = stock["stock_code"]
                news = news_data.get(code, [])
                headlines = [n.title for n in news] if news else []

                material_data.append({
                    "stock_code": code,
                    "stock_name": stock_names.get(code, ""),
                    "news_headlines": headlines,
                })

            # 심리 분석 데이터 준비
            sentiment_data = []
            for stock in selected_stocks:
                code = stock["stock_code"]
                disc = discussion_data.get(code, {})

                sentiment_data.append({
                    "stock_code": code,
                    "stock_name": stock_names.get(code, ""),
                    "community_posts": disc.get("community_posts", []),
                    "likes": disc.get("sentiment_data", {}).get("likes", 0),
                    "dislikes": disc.get("sentiment_data", {}).get("dislikes", 0),
                })

            if verbose:
                print(f"\n  [5-2] 재료 분석...")

            # 재료 분석
            material_results = material_analyzer.analyze_batch(material_data)

            if verbose:
                print(f"  [5-3] 심리 분석...")

            # 심리 분석
            sentiment_results = sentiment_analyzer.analyze_batch(sentiment_data)

            if verbose:
                print(f"  [5-4] 최종 판정...")

            # 최종 판정
            decisions = decision_engine.decide_batch(
                candidates_data=selected_stocks,
                material_results=material_results,
                sentiment_results=sentiment_results,
            )

            result.status = StageStatus.SUCCESS
            result.data = {
                "stock_count": len(decisions),
                "results": [d.to_dict() for d in decisions],
                "summary": decision_engine.summarize(decisions),
            }

            # 판정 분포 로깅
            summary = decision_engine.summarize(decisions)
            rec_dist = summary.get("recommendation_distribution", {})
            self.logger.info(
                f"[Stage 5] 완료: STRONG_BUY {rec_dist.get('STRONG_BUY', 0)}, "
                f"BUY {rec_dist.get('BUY', 0)}, WATCH {rec_dist.get('WATCH', 0)}, "
                f"AVOID {rec_dist.get('AVOID', 0)}"
            )

        except Exception as e:
            result.status = StageStatus.FAILED
            result.error = str(e)
            self.logger.error(f"[Stage 5] 실패: {e}")

        result.completed_at = datetime.now()
        return result

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _calculate_theme_metrics(self, theme_name: str, stock_codes: list[str], sector_type):
        """테마별 메트릭 계산 (v2.0)"""
        from src.sector.prioritizer import ThemeMetrics
        from src.core.interfaces import SectorType
        from src.analysis import FlowCalculator, TrendCalculator

        price_data = self._cache.get("price_data", {})
        supply_data = self._cache.get("supply_data", {})
        news_data = self._cache.get("news_data", {})

        flow_calc = FlowCalculator()
        trend_calc = TrendCalculator()

        # S_Flow 계산
        theme_supply = []
        for code in stock_codes:
            if code in supply_data:
                data = supply_data[code].copy()
                data["stock_code"] = code
                theme_supply.append(data)

        if theme_supply:
            flow_result = flow_calc.calculate_sector(theme_name, theme_supply)
            s_flow = flow_result.s_flow
        else:
            s_flow = 0

        # S_Trend 계산
        stock_trends = []
        for code in stock_codes:
            if code in price_data:
                trend_result = trend_calc.calculate_stock(code, price_data[code])
                stock_trends.append(trend_result.s_trend)
        s_trend = sum(stock_trends) / len(stock_trends) if stock_trends else 0

        # S_Breadth 계산
        if stock_trends:
            above_ma_count = sum(1 for t in stock_trends if t > 0)
            s_breadth = (above_ma_count / len(stock_trends)) * 100
        else:
            s_breadth = 0

        # 뉴스 수 계산 (Type B용)
        news_count = 0
        for code in stock_codes:
            if code in news_data:
                news_count += len(news_data[code])

        # sector_type이 None인 경우 기본값 설정
        if sector_type is None:
            sector_type = SectorType.TYPE_A

        return ThemeMetrics(
            theme_name=theme_name,
            sector_type=sector_type,
            stock_count=len(stock_codes),
            s_flow=s_flow,
            s_breadth=s_breadth,
            s_trend=s_trend,
            news_count=news_count,
        )

    # 하위 호환성 (deprecated)
    def _calculate_sector_metrics(self, sector, stock_codes: list[str], sector_type):
        """섹터별 메트릭 계산 (deprecated: _calculate_theme_metrics 사용)"""
        theme_name = sector.value if hasattr(sector, "value") else str(sector)
        return self._calculate_theme_metrics(theme_name, stock_codes, sector_type)

    def _prepare_stock_data(self, code: str, theme_name: str) -> dict:
        """종목 데이터 준비 (v2.0 - 테마명 사용)"""
        financial = self._cache.get("financial_data", {}).get(code, {})
        fundamental = self._cache.get("fundamental_data", {}).get(code, {})
        price_data = self._cache.get("price_data", {}).get(code)

        # 기술 지표 계산
        supply_demand = 0.5
        ma20_gap = 0
        volume_ratio = 1.0
        high_52w_proximity = 0.5

        if price_data is not None and not price_data.empty:
            try:
                close_col = "종가" if "종가" in price_data.columns else "close"
                volume_col = "거래량" if "거래량" in price_data.columns else "volume"

                closes = price_data[close_col].values
                volumes = price_data[volume_col].values

                if len(closes) >= 20:
                    ma20 = closes[-20:].mean()
                    ma20_gap = ((closes[-1] - ma20) / ma20) * 100

                if len(volumes) >= 10:
                    avg_vol = volumes[-10:].mean()
                    if avg_vol > 0:
                        volume_ratio = volumes[-1] / avg_vol

                if len(closes) >= 60:
                    high_52w = closes[-60:].max()
                    if high_52w > 0:
                        high_52w_proximity = closes[-1] / high_52w
            except Exception:
                pass

        return {
            "stock_code": code,
            "stock_name": self._cache.get("stock_names", {}).get(code, ""),
            "sector": theme_name,  # 테마명 문자열
            # 필터 조건
            "operating_profit_4q": financial.get("operating_profit_4q", 0),
            "debt_ratio": financial.get("debt_ratio", 0),
            "pbr": fundamental.get("pbr", 1.0),
            "avg_trading_value": fundamental.get("trading_value", 0),
            "capital_impairment": financial.get("capital_impairment", 0),
            "current_ratio": financial.get("current_ratio", 100),
            # 점수화 조건
            "roe": financial.get("roe", 0),
            "operating_margin": financial.get("operating_margin", 0),
            "revenue_growth": financial.get("revenue_growth", 0),
            "supply_demand": supply_demand,
            "ma20_gap": ma20_gap,
            "volume_ratio": volume_ratio,
            "high_52w_proximity": high_52w_proximity,
        }

    def _generate_summary(self, result: PipelineV2Result) -> dict:
        """결과 요약 생성"""
        summary = {
            "duration_seconds": result.duration_seconds,
            "stages_completed": len([s for s in result.stages.values() if s.status == StageStatus.SUCCESS]),
            "stages_failed": len([s for s in result.stages.values() if s.status == StageStatus.FAILED]),
        }

        # 각 단계 요약
        if "01_sector_classify" in result.stages:
            summary["sectors_analyzed"] = result.stages["01_sector_classify"].data.get("sector_count", 0)

        if "03_sector_priority" in result.stages:
            summary["sectors_selected"] = result.stages["03_sector_priority"].data.get("selected_count", 0)

        if "04_stock_selection" in result.stages:
            summary["stocks_selected"] = result.stages["04_stock_selection"].data.get("selected_count", 0)

        if "05_stock_verify" in result.stages:
            verify_summary = result.stages["05_stock_verify"].data.get("summary", {})
            summary["recommendations"] = verify_summary.get("recommendation_distribution", {})

        return summary

    def get_cached_data(self) -> dict:
        """캐시된 데이터 반환"""
        return self._cache.copy()

    def clear_cache(self):
        """캐시 초기화"""
        self._cache.clear()


if __name__ == "__main__":
    """
    직접 실행 테스트:
        python -m src.orchestrator.pipeline_v2 [--themes N] [--stocks N] [--skip-preflight]

    예시:
        python -m src.orchestrator.pipeline_v2 --themes 6 --stocks 30
    """
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline v2.0 테스트 실행")
    parser.add_argument("--themes", type=int, default=6, help="최대 테마 수 (기본: 6)")
    parser.add_argument("--stocks", type=int, default=30, help="최대 종목 수 (기본: 30)")
    parser.add_argument("--skip-preflight", action="store_true", help="Preflight 건너뛰기")
    parser.add_argument("--stage", type=str, default="1", help="실행할 최대 스테이지 (0, 1, all)")
    parser.add_argument("--debug", action="store_true", help="상세 데이터 흐름 출력 (입력→LLM→출력)")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Pipeline v2.0 테스트 (테마: {args.themes}개, 종목: {args.stocks}개)")
    print("=" * 70)

    pipeline = PipelineV2()

    if args.stage == "all":
        # 전체 파이프라인 실행
        result = pipeline.run(
            max_themes=args.themes,
            max_stocks=args.stocks,
            skip_preflight=args.skip_preflight,
            verbose=True,
        )
        print(f"\n결과: {'SUCCESS' if result.success else 'FAILED'}")
        if result.error:
            print(f"오류: {result.error}")
    else:
        # 특정 스테이지까지만 실행
        max_stage = int(args.stage)

        # Preflight
        if not args.skip_preflight:
            print("\n[Preflight]")
            preflight = pipeline.preflight_check(warmup_timeout=180.0)
            print(preflight.summary())
            if not preflight.passed:
                print("✗ Preflight 실패 - 중단")
                print(f"  실패 항목: {preflight.get_failures()}")
                exit(1)

        # Stage 0
        if max_stage >= 0:
            print("\n[Stage 0] 데이터 수집")
            stage0 = pipeline._run_data_collect(
                max_themes=args.themes,
                max_stocks=args.stocks,
                year=datetime.now().year,
                verbose=True,
            )
            print(f"  결과: {stage0.status.value}")
            if stage0.data:
                print(f"  테마: {stage0.data.get('theme_count', 0)}개")
                print(f"  종목: {stage0.data.get('stock_count', 0)}개")

        # Stage 1
        if max_stage >= 1:
            print("\n[Stage 1] 섹터 분류")
            stage1 = pipeline._run_sector_classify(verbose=True)
            print(f"  결과: {stage1.status.value}")
            if stage1.data:
                print(f"  테마: {stage1.data.get('theme_count', 0)}개")
                print(f"  종목: {stage1.data.get('stock_count', 0)}개")
                summary = stage1.data.get("summary", {})
                print(f"  테마 보유: {summary.get('with_themes', 0)}개")
                print(f"  DART 보유: {summary.get('with_dart_data', 0)}개")
                print(f"  뉴스 보유: {summary.get('with_news_data', 0)}개")

                # 종목별 분석 결과 출력
                results = stage1.data.get("results", [])
                if results:
                    print("\n  [종목별 분석 결과]")
                    print("  " + "-" * 70)
                    for r in results:
                        code = r.get("stock_code", "")
                        name = r.get("stock_name", "")
                        themes = r.get("theme_tags", [])
                        biz = r.get("business_summary", "")[:50] if r.get("business_summary") else "-"
                        news = r.get("news_summary", "")[:50] if r.get("news_summary") else "-"
                        print(f"  [{code}] {name}")
                        print(f"    테마: {', '.join(themes) if themes else '-'}")
                        print(f"    사업: {biz}{'...' if len(r.get('business_summary', '') or '') > 50 else ''}")
                        print(f"    뉴스: {news}{'...' if len(r.get('news_summary', '') or '') > 50 else ''}")
                        print()

                # --debug: 상세 데이터 흐름 출력
                if args.debug and results:
                    print("\n" + "=" * 70)
                    print("📊 [DEBUG] 데이터 흐름 상세 비교 (입력 → LLM → 출력)")
                    print("=" * 70)

                    # 캐시에서 원본 데이터 가져오기
                    news_data = pipeline._cache.get("news_data", {})
                    business_overview = pipeline._cache.get("business_overview", {})

                    for r in results[:3]:  # 상위 3개 종목만 상세 출력
                        code = r.get("stock_code", "")
                        name = r.get("stock_name", "")

                        print(f"\n{'─' * 70}")
                        print(f"🏢 [{code}] {name}")
                        print(f"{'─' * 70}")

                        # 1. 테마 정보
                        themes = r.get("theme_tags", [])
                        print(f"\n📌 소속 테마: {', '.join(themes) if themes else '없음'}")

                        # 2. 뉴스 데이터 흐름
                        print(f"\n📰 뉴스 데이터 흐름")
                        print("-" * 50)
                        raw_news = news_data.get(code, [])
                        if raw_news:
                            print(f"  [INPUT] 수집된 뉴스 ({len(raw_news)}건):")
                            for i, n in enumerate(raw_news[:3], 1):
                                # NewsArticle은 dataclass이므로 속성으로 접근
                                title = getattr(n, 'title', str(n))[:60]
                                print(f"    {i}. {title}...")
                            if len(raw_news) > 3:
                                print(f"    ... 외 {len(raw_news) - 3}건")
                        else:
                            print("  [INPUT] 수집된 뉴스: 없음")

                        news_summary = r.get("news_summary", "")
                        print(f"\n  [OUTPUT] LLM 요약:")
                        if news_summary:
                            # 요약 전체 출력 (줄바꿈 유지)
                            for line in news_summary.split('\n')[:10]:
                                print(f"    {line}")
                            if len(news_summary.split('\n')) > 10:
                                print("    ...")
                        else:
                            print("    (요약 없음)")

                        # 3. 사업개요 데이터 흐름
                        print(f"\n📋 사업개요 데이터 흐름")
                        print("-" * 50)
                        raw_biz = business_overview.get(code, "")
                        if raw_biz:
                            print(f"  [INPUT] DART 사업개요 ({len(raw_biz)}자):")
                            preview = raw_biz[:200].replace('\n', ' ')
                            print(f"    {preview}...")
                        else:
                            print("  [INPUT] DART 사업개요: 없음")

                        biz_summary = r.get("business_summary", "")
                        print(f"\n  [OUTPUT] LLM 요약:")
                        if biz_summary:
                            for line in biz_summary.split('\n')[:10]:
                                print(f"    {line}")
                            if len(biz_summary.split('\n')) > 10:
                                print("    ...")
                        else:
                            print("    (요약 없음)")

                    print(f"\n{'=' * 70}")
                    print(f"💡 상위 3개 종목만 표시됨 (전체: {len(results)}개)")
                    print(f"{'=' * 70}")
