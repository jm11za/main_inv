"""
분석 결과 DB 저장 서비스

파이프라인 실행 결과를 DB에 저장하고 조회
"""
import json
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.database import get_database
from src.core.models import (
    StockModel,
    ThemeModel,
    AnalysisHistoryModel,
    StockScoreHistoryModel,
)
from src.core.logger import get_logger


class AnalysisService:
    """
    분석 결과 DB 저장 서비스

    사용법:
        service = AnalysisService()

        # 분석 결과 저장
        service.save_analysis_result(pipeline_result)

        # 과거 분석 조회
        history = service.get_analysis_history(days=7)

        # 종목 점수 히스토리 조회
        scores = service.get_stock_score_history("005930", days=30)
    """

    def __init__(self):
        self.db = get_database()
        self.logger = get_logger(self.__class__.__name__)

    def save_analysis_result(
        self,
        pipeline_result: Any,
        tier_results: dict | None = None,
        score_results: dict | None = None,
        decision_results: dict | None = None,
        sector_labels: dict | None = None,
    ) -> int:
        """
        파이프라인 분석 결과 저장

        Args:
            pipeline_result: PipelineResult 객체
            tier_results: 섹터 Tier 분류 결과
            score_results: 종목 점수 결과
            decision_results: 최종 판정 결과
            sector_labels: 섹터 라벨 결과

        Returns:
            저장된 AnalysisHistory ID
        """
        self.logger.info("분석 결과 DB 저장 시작")

        try:
            with self.db.session() as session:
                # 1. 분석 히스토리 저장
                history = self._save_analysis_history(
                    session, pipeline_result, tier_results
                )

                # 2. 종목 점수 히스토리 저장
                if score_results:
                    self._save_score_history(
                        session, history.run_date, score_results,
                        decision_results, sector_labels
                    )

                # 3. 종목 테이블 업데이트 (최신 상태)
                if score_results:
                    self._update_stock_latest(
                        session, score_results, decision_results, sector_labels
                    )

                # 4. 테마 테이블 업데이트 (Tier 정보)
                if tier_results:
                    self._update_theme_tiers(session, tier_results)

                self.logger.info(f"분석 결과 저장 완료: ID={history.id}")
                return history.id

        except Exception as e:
            self.logger.error(f"분석 결과 저장 실패: {e}")
            raise

    def _save_analysis_history(
        self,
        session: Session,
        pipeline_result: Any,
        tier_results: dict | None
    ) -> AnalysisHistoryModel:
        """분석 히스토리 저장"""
        # Tier 1/2 섹터 추출
        tier1_sectors = []
        tier2_sectors = []

        if tier_results:
            for sector_name, info in tier_results.items():
                if isinstance(info, dict):
                    tier = info.get("tier", 3)
                    if tier == 1:
                        tier1_sectors.append(sector_name)
                    elif tier == 2:
                        tier2_sectors.append(sector_name)

        # 추천 종목 추출
        recommendations = getattr(pipeline_result, "final_recommendations", [])
        top_buy = [r for r in recommendations if r.get("recommendation") == "BUY"][:10]
        top_watch = [r for r in recommendations if r.get("recommendation") == "WATCH"][:10]

        history = AnalysisHistoryModel(
            run_date=datetime.now(),
            total_stocks=len(getattr(pipeline_result, "stage_results", [])),
            passed_filter=len(recommendations),
            tier1_sectors=",".join(tier1_sectors),
            tier2_sectors=",".join(tier2_sectors),
            top_buy=json.dumps(top_buy, ensure_ascii=False),
            top_watch=json.dumps(top_watch, ensure_ascii=False),
            duration_seconds=getattr(pipeline_result, "duration_seconds", 0),
            status="success" if getattr(pipeline_result, "success", False) else "failed",
            error_message=getattr(pipeline_result, "error", "") or "",
        )

        session.add(history)
        session.flush()

        return history

    def _save_score_history(
        self,
        session: Session,
        analysis_date: datetime,
        score_results: dict,
        decision_results: dict | None,
        sector_labels: dict | None
    ):
        """종목별 점수 히스토리 저장"""
        for stock_code, scores in score_results.items():
            decision = {}
            if decision_results and stock_code in decision_results:
                decision = decision_results[stock_code]

            sector_info = {}
            if sector_labels and stock_code in sector_labels:
                label = sector_labels[stock_code]
                if hasattr(label, "primary_sector"):
                    sector_info["primary_sector"] = label.primary_sector.value
                    sector_info["track_type"] = "TRACK_B" if label.is_growth_sector else "TRACK_A"

            history = StockScoreHistoryModel(
                stock_code=stock_code,
                analysis_date=analysis_date,
                total_score=scores.get("total_score", 0),
                financial_score=scores.get("financial_score", 0),
                technical_score=scores.get("technical_score", 0),
                s_flow=scores.get("breakdown", {}).get("s_flow", 0),
                s_trend=scores.get("breakdown", {}).get("s_trend", 0),
                material_grade=decision.get("material_grade", ""),
                sentiment_stage=decision.get("sentiment_stage", ""),
                recommendation=decision.get("recommendation", ""),
                primary_sector=sector_info.get("primary_sector", ""),
                track_type=sector_info.get("track_type", ""),
            )

            session.add(history)

    def _update_stock_latest(
        self,
        session: Session,
        score_results: dict,
        decision_results: dict | None,
        sector_labels: dict | None
    ):
        """종목 테이블 최신 상태 업데이트"""
        for stock_code, scores in score_results.items():
            stock = session.get(StockModel, stock_code)
            if not stock:
                continue

            # 점수 업데이트
            stock.total_score = scores.get("total_score", 0)
            stock.financial_score = scores.get("financial_score", 0)
            stock.technical_score = scores.get("technical_score", 0)
            stock.s_flow = scores.get("breakdown", {}).get("s_flow", 0)

            # 판정 업데이트
            if decision_results and stock_code in decision_results:
                decision = decision_results[stock_code]
                stock.material_grade = decision.get("material_grade", "")
                stock.sentiment_stage = decision.get("sentiment_stage", "")
                stock.recommendation = decision.get("recommendation", "")

            stock.updated_at = datetime.now()

    def _update_theme_tiers(self, session: Session, tier_results: dict):
        """테마 Tier 업데이트"""
        for sector_name, info in tier_results.items():
            if not isinstance(info, dict):
                continue

            # 테마 찾기 (이름으로)
            result = session.execute(
                select(ThemeModel).where(ThemeModel.name == sector_name)
            )
            theme = result.scalar_one_or_none()

            if theme:
                theme.tier = info.get("tier", 3)
                theme.s_flow = info.get("s_flow", 0)
                theme.s_breadth = info.get("s_breadth", 0)
                theme.s_trend = info.get("s_trend", 0)
                theme.updated_at = datetime.now()

    def get_analysis_history(self, days: int = 30) -> list[AnalysisHistoryModel]:
        """
        최근 분석 히스토리 조회

        Args:
            days: 조회 기간 (일)

        Returns:
            AnalysisHistoryModel 리스트
        """
        from datetime import timedelta

        with self.db.session() as session:
            cutoff = datetime.now() - timedelta(days=days)

            result = session.execute(
                select(AnalysisHistoryModel)
                .where(AnalysisHistoryModel.run_date >= cutoff)
                .order_by(AnalysisHistoryModel.run_date.desc())
            )

            return list(result.scalars().all())

    def get_stock_score_history(
        self,
        stock_code: str,
        days: int = 30
    ) -> list[StockScoreHistoryModel]:
        """
        종목 점수 히스토리 조회

        Args:
            stock_code: 종목코드
            days: 조회 기간 (일)

        Returns:
            StockScoreHistoryModel 리스트
        """
        from datetime import timedelta

        with self.db.session() as session:
            cutoff = datetime.now() - timedelta(days=days)

            result = session.execute(
                select(StockScoreHistoryModel)
                .where(StockScoreHistoryModel.stock_code == stock_code)
                .where(StockScoreHistoryModel.analysis_date >= cutoff)
                .order_by(StockScoreHistoryModel.analysis_date.desc())
            )

            return list(result.scalars().all())

    def get_latest_recommendations(
        self,
        recommendation_type: str = "BUY",
        limit: int = 20
    ) -> list[StockModel]:
        """
        최신 추천 종목 조회

        Args:
            recommendation_type: BUY, WATCH, HOLD, AVOID
            limit: 최대 개수

        Returns:
            StockModel 리스트
        """
        with self.db.session() as session:
            result = session.execute(
                select(StockModel)
                .where(StockModel.recommendation == recommendation_type)
                .order_by(StockModel.total_score.desc())
                .limit(limit)
            )

            return list(result.scalars().all())

    def get_sector_tier_summary(self) -> dict:
        """
        섹터 Tier 요약 조회

        Returns:
            {
                "tier1": [테마명, ...],
                "tier2": [테마명, ...],
                "tier3": [테마명, ...],
            }
        """
        with self.db.session() as session:
            result = session.execute(select(ThemeModel))
            themes = result.scalars().all()

            summary = {"tier1": [], "tier2": [], "tier3": []}

            for theme in themes:
                tier_key = f"tier{theme.tier}"
                if tier_key in summary:
                    summary[tier_key].append({
                        "name": theme.name,
                        "s_flow": theme.s_flow,
                        "s_breadth": theme.s_breadth,
                        "s_trend": theme.s_trend,
                    })

            return summary
