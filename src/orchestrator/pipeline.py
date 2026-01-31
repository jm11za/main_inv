"""
Pipeline: 전체 분석 파이프라인

Layer 1~6을 순차적으로 실행하고 결과를 집계
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.core.logger import get_logger
from src.core.config import get_config
from src.orchestrator.stage_runner import StageRunner, StageResult, StageStatus


@dataclass
class PipelineResult:
    """파이프라인 실행 결과"""
    success: bool
    started_at: datetime
    completed_at: datetime | None = None
    stage_results: list[StageResult] = field(default_factory=list)
    final_recommendations: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def duration_seconds(self) -> float:
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0

    def get_stage(self, stage_name: str) -> StageResult | None:
        """특정 단계 결과 조회"""
        for stage in self.stage_results:
            if stage.stage_name == stage_name:
                return stage
        return None

    def to_summary(self) -> dict:
        """요약 정보 반환"""
        return {
            "success": self.success,
            "duration": f"{self.duration_seconds:.1f}s",
            "stages": [s.to_dict() for s in self.stage_results],
            "recommendations": self.final_recommendations,
            "error": self.error,
        }


class Pipeline:
    """
    전체 분석 파이프라인

    사용법:
        pipeline = Pipeline()

        # 전체 실행
        result = pipeline.run_full(
            theme_codes=["바이오", "2차전지"],
            year=2025
        )

        # 단계별 실행
        result = pipeline.run_from_stage(
            start_stage="analysis",
            cached_data={...}
        )
    """

    # 단계 순서 정의
    STAGES = [
        "ingest",
        "processing",
        "sector_labeling",
        "analysis",
        "filtering",
        "scoring",
        "decision",
        "output",
    ]

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.config = get_config()
        self.runner = StageRunner()

        # 단계별 데이터 캐시
        self._cache: dict[str, Any] = {}

        # DB 저장 서비스 (지연 로드)
        self._analysis_service = None

    @property
    def analysis_service(self):
        """AnalysisService 지연 로드"""
        if self._analysis_service is None:
            from src.orchestrator.analysis_service import AnalysisService
            self._analysis_service = AnalysisService()
        return self._analysis_service

    def run_full(
        self,
        theme_codes: list[str] | None = None,
        stock_codes: list[str] | None = None,
        year: int | None = None,
        skip_themes: bool = False,
        max_stocks: int = 100,
        save_to_db: bool = True,
    ) -> PipelineResult:
        """
        전체 파이프라인 실행

        Args:
            theme_codes: 분석할 테마 코드 (없으면 전체)
            stock_codes: 분석할 종목 코드 (없으면 테마에서 추출)
            year: 재무 데이터 기준 연도
            skip_themes: 테마 수집 건너뛰기
            max_stocks: 최대 분석 종목 수
            save_to_db: DB에 결과 저장 여부 (기본값: True)

        Returns:
            PipelineResult
        """
        if year is None:
            year = datetime.now().year

        result = PipelineResult(
            success=False,
            started_at=datetime.now()
        )

        self.logger.info("=" * 50)
        self.logger.info("파이프라인 시작")
        self.logger.info("=" * 50)

        try:
            # ===== Layer 1: Ingest =====
            self.logger.info("[Layer 1] 데이터 수집 시작")

            # 1-1. 테마 수집 (선택적)
            if not skip_themes:
                theme_result = self.runner.run_ingest_themes()
                result.stage_results.append(theme_result)

                if theme_result.status == StageStatus.FAILED:
                    raise Exception(f"테마 수집 실패: {theme_result.error}")

                self._cache["themes"] = theme_result.data.get("themes", [])

            # 종목 코드 결정
            if stock_codes is None:
                stock_codes = self._get_stock_codes_from_themes(
                    theme_codes, max_stocks
                )

            if not stock_codes:
                raise Exception("분석할 종목이 없습니다")

            self.logger.info(f"분석 대상: {len(stock_codes)}개 종목")
            self._cache["stock_codes"] = stock_codes

            # 1-2. 주가 수집
            price_result = self.runner.run_ingest_prices(stock_codes)
            result.stage_results.append(price_result)
            self._cache["price_data"] = price_result.data.get("price_data", {})

            # 1-3. 재무 수집
            financial_result = self.runner.run_ingest_financials(stock_codes, year)
            result.stage_results.append(financial_result)
            self._cache["financial_data"] = financial_result.data.get("financial_data", {})

            # 1-4. 뉴스 수집
            news_result = self.runner.run_ingest_news(stock_codes)
            result.stage_results.append(news_result)
            self._cache["news_data"] = news_result.data.get("news_data", {})

            # 1-5. 커뮤니티 수집
            stock_names = self._get_stock_names()
            community_result = self.runner.run_ingest_community(
                stock_codes, stock_names
            )
            result.stage_results.append(community_result)
            self._cache["community_data"] = community_result.data.get("community_data", {})

            # 1-6. 토론방 수집 (Sentiment B 접근법)
            discussion_result = self.runner.run_ingest_discussion(
                stock_codes, stock_names
            )
            result.stage_results.append(discussion_result)
            self._cache["discussion_data"] = discussion_result.data.get("discussion_data", {})

            # ===== Layer 2: Processing =====
            self.logger.info("[Layer 2] 데이터 처리 시작")

            processing_result = self.runner.run_processing(
                self._cache["news_data"],
                self._cache["community_data"]
            )
            result.stage_results.append(processing_result)
            self._cache["processed_data"] = processing_result.data.get("processed_data", {})

            # 2-2. 섹터 라벨링
            self.logger.info("[Layer 2.5] 섹터 라벨링 시작")

            # 뉴스 헤드라인 추출
            news_headlines = {}
            for code, data in self._cache["processed_data"].items():
                news_headlines[code] = data.get("headlines", [])

            # DART 사업보고서 텍스트 추출
            dart_texts = {}
            for code, data in self._cache.get("financial_data", {}).items():
                if isinstance(data, dict) and data.get("business_report"):
                    dart_texts[code] = data.get("business_report", "")

            # 종목-테마 매핑
            theme_mapping = self._get_stock_theme_mapping()

            sector_result = self.runner.run_sector_labeling(
                stock_codes=stock_codes,
                stock_names=stock_names,
                theme_data=theme_mapping,
                dart_data=dart_texts if dart_texts else None,
                news_data=news_headlines if news_headlines else None,
            )
            result.stage_results.append(sector_result)
            self._cache["sector_labels"] = sector_result.data.get("sector_labels", {})
            self._cache["sector_types"] = sector_result.data.get("sector_types", {})

            # ===== Layer 3: Analysis =====
            self.logger.info("[Layer 3] 분석 시작")

            # 수급 데이터 준비
            supply_data = self._prepare_supply_data()
            theme_stocks = self._get_theme_stock_mapping()

            analysis_result = self.runner.run_analysis(
                self._cache["price_data"],
                supply_data,
                theme_stocks
            )
            result.stage_results.append(analysis_result)
            self._cache["sector_metrics"] = analysis_result.data.get("sector_metrics", {})
            self._cache["tier_results"] = analysis_result.data.get("tier_results", {})

            # ===== Layer 3.5: Filtering =====
            self.logger.info("[Layer 3.5] 필터링 시작")

            filter_stocks = self._prepare_filter_input()
            sector_types = self._get_sector_types()

            filtering_result = self.runner.run_filtering(filter_stocks, sector_types)
            result.stage_results.append(filtering_result)
            self._cache["filter_results"] = filtering_result.data.get("filter_results", {})
            passed_codes = filtering_result.data.get("passed_codes", [])

            if not passed_codes:
                self.logger.warning("필터 통과 종목 없음")
                result.success = True
                result.completed_at = datetime.now()
                return result

            # ===== Layer 4: Scoring =====
            self.logger.info("[Layer 4] 점수화 시작")

            technical_data = self._prepare_technical_data()
            track_types = self._get_track_types()

            scoring_result = self.runner.run_scoring(
                passed_codes,
                self._cache["financial_data"],
                technical_data,
                track_types
            )
            result.stage_results.append(scoring_result)
            self._cache["score_results"] = scoring_result.data.get("score_results", {})
            ranking = scoring_result.data.get("ranking", [])

            # ===== Layer 5: Decision =====
            self.logger.info("[Layer 5] 판정 시작")

            # 상위 종목만 판정
            top_stocks = ranking[:20]
            tier_data = self._get_stock_tiers(top_stocks)

            # 가격 지표 계산 (Sentiment C 접근법)
            price_indicators = self._calculate_price_indicators(top_stocks)

            decision_result = self.runner.run_decision(
                stock_codes=top_stocks,
                tier_data=tier_data,
                news_data=self._cache["processed_data"],
                community_data=self._cache["community_data"],
                discussion_data=self._cache.get("discussion_data", {}),
                price_indicators=price_indicators,
                stock_names=stock_names,
            )
            result.stage_results.append(decision_result)

            result.final_recommendations = decision_result.data.get("recommendations", [])

            # ===== 완료 =====
            result.success = True
            result.completed_at = datetime.now()

            # ===== DB 저장 =====
            if save_to_db:
                self._save_to_database(result)

            self.logger.info("=" * 50)
            self.logger.info(f"파이프라인 완료: {result.duration_seconds:.1f}초")
            self.logger.info(f"추천 종목: {len(result.final_recommendations)}개")
            self.logger.info("=" * 50)

        except Exception as e:
            result.success = False
            result.error = str(e)
            result.completed_at = datetime.now()
            self.logger.error(f"파이프라인 실패: {e}")

        return result

    def run_quick(
        self,
        stock_codes: list[str],
        year: int | None = None,
    ) -> PipelineResult:
        """
        빠른 분석 (테마/뉴스/커뮤니티 수집 생략)

        Args:
            stock_codes: 분석할 종목 코드
            year: 재무 데이터 기준 연도

        Returns:
            PipelineResult
        """
        return self.run_full(
            stock_codes=stock_codes,
            year=year,
            skip_themes=True,
        )

    # ========== Helper Methods ==========

    def _get_stock_codes_from_themes(
        self,
        theme_codes: list[str] | None,
        max_stocks: int
    ) -> list[str]:
        """테마에서 종목 코드 추출"""
        themes = self._cache.get("themes", [])
        codes = set()

        for theme in themes:
            if theme_codes is None or theme.name in theme_codes:
                for stock in theme.stocks:
                    codes.add(stock.stock_code)

                    if len(codes) >= max_stocks:
                        return list(codes)

        return list(codes)

    def _get_stock_names(self) -> dict[str, str]:
        """종목 코드 → 이름 매핑"""
        themes = self._cache.get("themes", [])
        names = {}

        for theme in themes:
            for stock in theme.stocks:
                names[stock.stock_code] = stock.stock_name

        return names

    def _prepare_supply_data(self) -> dict:
        """수급 데이터 준비"""
        from src.processing import DataTransformer

        transformer = DataTransformer()
        supply_data = {}

        for code in self._cache.get("stock_codes", []):
            supply = transformer.get_supply_demand(code)
            supply_data[code] = supply.get_s_flow_inputs()

        return supply_data

    def _get_theme_stock_mapping(self) -> dict[str, list[str]]:
        """테마 → 종목 매핑"""
        themes = self._cache.get("themes", [])
        mapping = {}

        for theme in themes:
            mapping[theme.name] = [s.stock_code for s in theme.stocks]

        return mapping

    def _get_stock_theme_mapping(self) -> dict[str, list[str]]:
        """종목 → 테마 매핑 (SectorLabeler용)"""
        themes = self._cache.get("themes", [])
        mapping: dict[str, list[str]] = {}

        for theme in themes:
            for stock in theme.stocks:
                if stock.stock_code not in mapping:
                    mapping[stock.stock_code] = []
                mapping[stock.stock_code].append(theme.name)

        return mapping

    def _prepare_filter_input(self) -> list[dict]:
        """필터 입력 데이터 준비 (캐시된 데이터 활용 - API 중복 호출 방지)"""
        stocks = []

        # 이미 수집된 캐시 데이터 사용
        fundamental_data = self._cache.get("fundamental_data", {})
        supply_data = self._cache.get("supply_data", {})

        for code in self._cache.get("stock_codes", []):
            fin = self._cache.get("financial_data", {}).get(code, {})
            fund = fundamental_data.get(code, {})

            # 캐시된 데이터에서 가져오기 (API 재호출 없음)
            pbr = fund.get("pbr", 0.0) or 1.0
            avg_trading_value = fund.get("trading_value", 0)

            stocks.append({
                "stock_code": code,
                "operating_profit_4q": fin.get("operating_profit_4q", 0),
                "debt_ratio": fin.get("debt_ratio", 0),
                "pbr": pbr,
                "avg_trading_value": avg_trading_value,
                "capital_impairment": fin.get("capital_impairment", 0),
                "current_ratio": fin.get("current_ratio", 0),
                "rd_ratio": fin.get("rd_ratio", 0),
            })

        return stocks

    def _get_sector_types(self) -> dict[str, str]:
        """섹터별 타입 (A/B) 반환"""
        # SectorLabeler 결과 사용
        cached_types = self._cache.get("sector_types", {})
        if cached_types:
            return cached_types

        # 폴백: 기존 SectorClassifier 사용
        from src.filtering import SectorClassifier
        from src.core.interfaces import SectorType

        classifier = SectorClassifier()
        types = {}

        themes = self._cache.get("themes", [])
        for theme in themes:
            sector_type = classifier.classify(theme.name)
            types[theme.name] = "A" if sector_type == SectorType.TYPE_A else "B"

        return types

    def _prepare_technical_data(self) -> dict:
        """기술적 분석 데이터 준비 (개별 종목 S_Flow 포함)"""
        from src.analysis import TrendCalculator, FlowCalculator

        trend_calc = TrendCalculator()
        flow_calc = FlowCalculator()
        tech_data = {}

        for code, df in self._cache.get("price_data", {}).items():
            s_trend = trend_calc.calculate_from_df(df)

            # 개별 종목 S_Flow 계산
            flow_result = flow_calc.calculate_stock_auto(code)
            s_flow = flow_result.s_flow

            tech_data[code] = {
                "s_flow": s_flow,
                "s_breadth": 0,  # 개별 종목에는 해당 없음 (섹터 지표)
                "s_trend": s_trend,
                "ma_gap": 0,
            }

        return tech_data

    def _get_track_types(self) -> dict[str, str]:
        """종목별 Track 타입"""
        # SectorLabeler 결과 사용 (성장 섹터 = B, 실적 섹터 = A)
        sector_types = self._cache.get("sector_types", {})
        if sector_types:
            return sector_types

        # 폴백: 모든 종목 Track A
        return {code: "A" for code in self._cache.get("stock_codes", [])}

    def _get_stock_tiers(self, stock_codes: list[str]) -> dict:
        """종목별 Tier 정보"""
        tier_results = self._cache.get("tier_results", {})

        stock_tiers = {}
        for code in stock_codes:
            # 테마에서 Tier 찾기
            for theme_name, tier_info in tier_results.items():
                if isinstance(tier_info, dict) and tier_info.get("tier"):
                    stock_tiers[code] = tier_info
                    break

        return stock_tiers

    def _calculate_price_indicators(self, stock_codes: list[str]) -> dict:
        """
        가격 지표 계산 (Sentiment C 접근법)

        RSI, 수익률, 거래량 비율 계산

        Args:
            stock_codes: 종목 코드 리스트

        Returns:
            {종목코드: {"rsi": float, "return_1w": float, "return_1m": float, "volume_ratio": float}}
        """
        import pandas as pd
        import numpy as np

        price_data = self._cache.get("price_data", {})
        indicators = {}

        for code in stock_codes:
            df = price_data.get(code)
            if df is None or df.empty:
                indicators[code] = {
                    "rsi": 50.0,
                    "return_1w": 0.0,
                    "return_1m": 0.0,
                    "volume_ratio": 1.0,
                }
                continue

            try:
                # 최신 데이터가 위에 있다고 가정하고 정렬
                if "date" in df.columns:
                    df = df.sort_values("date", ascending=True)

                close_col = "종가" if "종가" in df.columns else "close"
                volume_col = "거래량" if "거래량" in df.columns else "volume"

                closes = df[close_col].values if close_col in df.columns else []
                volumes = df[volume_col].values if volume_col in df.columns else []

                # RSI 계산 (14일)
                rsi = 50.0
                if len(closes) >= 15:
                    deltas = np.diff(closes[-15:])
                    gains = np.where(deltas > 0, deltas, 0)
                    losses = np.where(deltas < 0, -deltas, 0)
                    avg_gain = np.mean(gains)
                    avg_loss = np.mean(losses)
                    if avg_loss > 0:
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                    else:
                        rsi = 100 if avg_gain > 0 else 50

                # 수익률 계산
                return_1w = 0.0
                return_1m = 0.0

                if len(closes) >= 5:
                    return_1w = ((closes[-1] - closes[-5]) / closes[-5]) * 100

                if len(closes) >= 20:
                    return_1m = ((closes[-1] - closes[-20]) / closes[-20]) * 100

                # 거래량 비율 (최근 5일 평균 / 전체 평균)
                volume_ratio = 1.0
                if len(volumes) >= 20:
                    recent_avg = np.mean(volumes[-5:])
                    total_avg = np.mean(volumes)
                    if total_avg > 0:
                        volume_ratio = recent_avg / total_avg

                indicators[code] = {
                    "rsi": float(rsi),
                    "return_1w": float(return_1w),
                    "return_1m": float(return_1m),
                    "volume_ratio": float(volume_ratio),
                }

            except Exception as e:
                self.logger.debug(f"[{code}] 가격 지표 계산 실패: {e}")
                indicators[code] = {
                    "rsi": 50.0,
                    "return_1w": 0.0,
                    "return_1m": 0.0,
                    "volume_ratio": 1.0,
                }

        return indicators

    def get_cached_data(self) -> dict:
        """캐시된 데이터 반환"""
        return self._cache.copy()

    def clear_cache(self):
        """캐시 초기화"""
        self._cache.clear()

    def _save_to_database(self, result: PipelineResult):
        """
        파이프라인 결과를 DB에 저장

        Args:
            result: PipelineResult 객체
        """
        try:
            self.logger.info("[DB] 분석 결과 저장 시작")

            # 각 캐시 데이터 추출
            tier_results = self._cache.get("tier_results", {})
            score_results = self._cache.get("score_results", {})
            sector_labels = self._cache.get("sector_labels", {})

            # Decision 결과 추출
            decision_results = {}
            for rec in result.final_recommendations:
                code = rec.get("stock_code")
                if code:
                    decision_results[code] = {
                        "material_grade": rec.get("material_grade", ""),
                        "sentiment_stage": rec.get("sentiment_stage", ""),
                        "recommendation": rec.get("recommendation", ""),
                    }

            # AnalysisService로 저장
            history_id = self.analysis_service.save_analysis_result(
                pipeline_result=result,
                tier_results=tier_results,
                score_results=score_results,
                decision_results=decision_results,
                sector_labels=sector_labels,
            )

            self.logger.info(f"[DB] 분석 결과 저장 완료: History ID={history_id}")

        except Exception as e:
            self.logger.error(f"[DB] 분석 결과 저장 실패: {e}")
            # DB 저장 실패는 파이프라인 실패로 처리하지 않음 (결과는 이미 있음)
