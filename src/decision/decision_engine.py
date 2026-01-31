"""
Decision Engine - 최종 투자 판정 엔진

Skeptic(재료 분석)과 Sentiment Reader(심리 분석) 결과를 종합하여
최종 투자 등급(STRONG_BUY / BUY / WATCH / AVOID)을 결정합니다.

Decision Matrix:
| Tier | 재료 등급 | 심리 단계   | 최종 판정    |
|------|----------|------------|-------------|
| 1    | S/A      | 공포/의심   | STRONG_BUY  |
| 1    | S/A      | 확신       | BUY         |
| 2    | A/B      | 의심/확신초기| BUY         |
| 2    | B/C      | 확신후기    | WATCH       |
| Any  | C        | Any        | WATCH       |
| Any  | Any      | 환희       | AVOID       |
"""
from dataclasses import dataclass, field
from typing import Any

from src.core.interfaces import (
    MaterialGrade,
    SentimentStage,
    Recommendation,
    Tier,
    TrackType,
)
from src.core.config import get_config
from src.core.logger import get_logger
from src.decision.personas.skeptic import SkepticAnalysis
from src.decision.personas.sentiment import SentimentAnalysis


@dataclass
class DecisionResult:
    """최종 투자 판정 결과"""
    stock_code: str
    stock_name: str

    # 입력 데이터
    tier: Tier
    track_type: TrackType
    total_score: float

    # 분석 결과
    material_grade: MaterialGrade
    sentiment_stage: SentimentStage

    # 최종 판정
    recommendation: Recommendation
    confidence: float  # 0.0 ~ 1.0

    # 상세 내역
    skeptic_analysis: SkepticAnalysis | None = None
    sentiment_analysis: SentimentAnalysis | None = None
    decision_reasoning: str = ""

    # 추가 정보
    key_factors: list[str] = field(default_factory=list)
    risk_warnings: list[str] = field(default_factory=list)


class DecisionEngine:
    """
    최종 투자 판정 엔진

    Skeptic과 Sentiment Reader의 분석 결과를 종합하여
    Decision Matrix에 따라 최종 투자 등급을 결정합니다.

    핵심 로직:
    1. 환희(EUPHORIA) 상태면 무조건 AVOID (고점 경고)
    2. 재료 C급이면 WATCH (재료 불명확)
    3. Tier + 재료 + 심리 조합으로 BUY/STRONG_BUY 결정
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.config = get_config()

    def decide(
        self,
        stock_code: str,
        stock_name: str,
        tier: Tier,
        track_type: TrackType,
        total_score: float,
        skeptic_analysis: SkepticAnalysis,
        sentiment_analysis: SentimentAnalysis,
    ) -> DecisionResult:
        """
        최종 투자 판정

        Args:
            stock_code: 종목코드
            stock_name: 종목명
            tier: 섹터 Tier
            track_type: Track 타입
            total_score: 스코어링 점수
            skeptic_analysis: Skeptic 분석 결과
            sentiment_analysis: Sentiment 분석 결과

        Returns:
            DecisionResult
        """
        material_grade = skeptic_analysis.material_grade
        sentiment_stage = sentiment_analysis.sentiment_stage

        # Decision Matrix 적용
        recommendation, confidence, reasoning = self._apply_decision_matrix(
            tier, material_grade, sentiment_stage, total_score
        )

        # Key Factors 추출
        key_factors = self._extract_key_factors(
            skeptic_analysis, sentiment_analysis, tier
        )

        # Risk Warnings 추출
        risk_warnings = self._extract_risk_warnings(
            skeptic_analysis, sentiment_analysis, recommendation
        )

        return DecisionResult(
            stock_code=stock_code,
            stock_name=stock_name,
            tier=tier,
            track_type=track_type,
            total_score=total_score,
            material_grade=material_grade,
            sentiment_stage=sentiment_stage,
            recommendation=recommendation,
            confidence=confidence,
            skeptic_analysis=skeptic_analysis,
            sentiment_analysis=sentiment_analysis,
            decision_reasoning=reasoning,
            key_factors=key_factors,
            risk_warnings=risk_warnings,
        )

    def _apply_decision_matrix(
        self,
        tier: Tier,
        material_grade: MaterialGrade,
        sentiment_stage: SentimentStage,
        total_score: float,
    ) -> tuple[Recommendation, float, str]:
        """
        Decision Matrix 적용

        Returns:
            (Recommendation, confidence, reasoning)
        """
        # 1. 환희 상태면 무조건 AVOID (고점 경고)
        if sentiment_stage == SentimentStage.EUPHORIA:
            return (
                Recommendation.AVOID,
                0.85,
                "심리 과열 상태입니다. 고점 리스크가 있어 진입을 피해야 합니다."
            )

        # 2. 재료 C급이면 WATCH (재료 불명확)
        if material_grade == MaterialGrade.C:
            return (
                Recommendation.WATCH,
                0.7,
                "뚜렷한 재료가 없습니다. 추가 재료 확인 후 진입을 고려하세요."
            )

        # 3. Tier 1 + 좋은 재료 + 초기 심리 = STRONG_BUY
        if tier == Tier.TIER_1:
            if material_grade in [MaterialGrade.S, MaterialGrade.A]:
                if sentiment_stage in [SentimentStage.FEAR, SentimentStage.DOUBT]:
                    return (
                        Recommendation.STRONG_BUY,
                        0.9,
                        "수급 빈집에 좋은 재료, 대중은 아직 모르는 상태입니다. 적극 매수 추천."
                    )
                elif sentiment_stage == SentimentStage.CONVICTION:
                    return (
                        Recommendation.BUY,
                        0.8,
                        "수급 빈집에 좋은 재료, 관심이 늘고 있습니다. 매수 추천."
                    )

            # Tier 1 + B급 재료
            if material_grade == MaterialGrade.B:
                if sentiment_stage in [SentimentStage.FEAR, SentimentStage.DOUBT]:
                    return (
                        Recommendation.BUY,
                        0.75,
                        "수급이 들어오고 있으나 재료는 보통 수준입니다."
                    )

        # 4. Tier 2 + 재료 + 초기/중기 심리 = BUY
        if tier == Tier.TIER_2:
            if material_grade in [MaterialGrade.A, MaterialGrade.B]:
                if sentiment_stage in [SentimentStage.DOUBT, SentimentStage.CONVICTION]:
                    # 확신 초기인지 후기인지 관심도로 판단
                    if sentiment_stage == SentimentStage.CONVICTION:
                        interest = getattr(
                            getattr(self, '_current_sentiment', None),
                            'interest_level', 0.5
                        )
                        # 관심도가 너무 높으면 후기로 판단
                        if interest > 0.7:
                            return (
                                Recommendation.WATCH,
                                0.65,
                                "주도 섹터이나 관심이 이미 높습니다. 눌림목 대기 권장."
                            )

                    return (
                        Recommendation.BUY,
                        0.75,
                        "주도 섹터로 재료와 심리가 양호합니다. 매수 추천."
                    )

            # Tier 2 + S급 재료
            if material_grade == MaterialGrade.S:
                if sentiment_stage in [SentimentStage.FEAR, SentimentStage.DOUBT]:
                    return (
                        Recommendation.STRONG_BUY,
                        0.85,
                        "주도 섹터에 대형 호재, 초기 진입 기회입니다."
                    )
                else:
                    return (
                        Recommendation.BUY,
                        0.8,
                        "주도 섹터에 대형 호재가 있습니다."
                    )

        # 5. Tier 3 = 가짜 상승, 진입 금지
        if tier == Tier.TIER_3:
            return (
                Recommendation.AVOID,
                0.8,
                "대장주만 상승하는 가짜 상승입니다. 진입을 피하세요."
            )

        # 6. 그 외 조합은 WATCH
        return (
            Recommendation.WATCH,
            0.6,
            "조건이 명확하지 않습니다. 추가 모니터링이 필요합니다."
        )

    def _extract_key_factors(
        self,
        skeptic: SkepticAnalysis,
        sentiment: SentimentAnalysis,
        tier: Tier
    ) -> list[str]:
        """핵심 팩터 추출"""
        factors = []

        # Tier 관련
        tier_desc = {
            Tier.TIER_1: "수급 빈집 (선취매 기회)",
            Tier.TIER_2: "주도 섹터",
            Tier.TIER_3: "가짜 상승 (주의)",
        }
        factors.append(tier_desc.get(tier, ""))

        # 재료 관련
        if skeptic.key_materials:
            factors.extend(skeptic.key_materials[:2])

        # 심리 관련
        stage_desc = {
            SentimentStage.FEAR: "대중 무관심 (바닥권)",
            SentimentStage.DOUBT: "초기 관심",
            SentimentStage.CONVICTION: "상승 확신 형성 중",
            SentimentStage.EUPHORIA: "과열 주의",
        }
        factors.append(stage_desc.get(sentiment.sentiment_stage, ""))

        return [f for f in factors if f]

    def _extract_risk_warnings(
        self,
        skeptic: SkepticAnalysis,
        sentiment: SentimentAnalysis,
        recommendation: Recommendation
    ) -> list[str]:
        """리스크 경고 추출"""
        warnings = []

        # Skeptic이 발굴한 리스크
        if skeptic.risks:
            warnings.extend(skeptic.risks[:3])

        # 심리 과열 경고
        if sentiment.sentiment_stage == SentimentStage.EUPHORIA:
            warnings.append("심리 과열 - 고점 주의")

        # 높은 관심도 + BUY 이상이면 타이밍 경고
        if recommendation in [Recommendation.STRONG_BUY, Recommendation.BUY]:
            if sentiment.interest_level > 0.7:
                warnings.append("관심도가 높음 - 분할 매수 권장")

        # 낮은 신뢰도 경고
        combined_confidence = (skeptic.confidence + sentiment.confidence) / 2
        if combined_confidence < 0.5:
            warnings.append("분석 신뢰도 낮음 - 추가 확인 필요")

        return warnings

    def decide_batch(
        self,
        stocks_data: list[dict[str, Any]],
        skeptic_results: list[SkepticAnalysis],
        sentiment_results: list[SentimentAnalysis],
    ) -> list[DecisionResult]:
        """
        복수 종목 일괄 판정

        Args:
            stocks_data: 종목 데이터 리스트
            skeptic_results: Skeptic 분석 결과 리스트
            sentiment_results: Sentiment 분석 결과 리스트

        Returns:
            DecisionResult 리스트
        """
        # stock_code로 인덱싱
        skeptic_map = {s.stock_code: s for s in skeptic_results}
        sentiment_map = {s.stock_code: s for s in sentiment_results}

        results = []
        for data in stocks_data:
            stock_code = data["stock_code"]

            skeptic = skeptic_map.get(stock_code)
            sentiment = sentiment_map.get(stock_code)

            if not skeptic or not sentiment:
                self.logger.warning(f"분석 결과 없음: {stock_code}")
                continue

            # 심리 분석 결과 임시 저장 (관심도 참조용)
            self._current_sentiment = sentiment

            result = self.decide(
                stock_code=stock_code,
                stock_name=data.get("stock_name", ""),
                tier=data.get("tier", Tier.TIER_3),
                track_type=data.get("track_type", TrackType.TRACK_A),
                total_score=data.get("total_score", 0.0),
                skeptic_analysis=skeptic,
                sentiment_analysis=sentiment,
            )
            results.append(result)

        return results

    def get_top_recommendations(
        self,
        results: list[DecisionResult],
        recommendation: Recommendation | None = None,
        n: int = 10
    ) -> list[DecisionResult]:
        """
        상위 추천 종목 필터링

        Args:
            results: DecisionResult 리스트
            recommendation: 특정 등급만 필터 (None이면 BUY 이상)
            n: 반환 개수

        Returns:
            필터링된 리스트
        """
        if recommendation:
            filtered = [r for r in results if r.recommendation == recommendation]
        else:
            # STRONG_BUY, BUY만 필터
            filtered = [
                r for r in results
                if r.recommendation in [Recommendation.STRONG_BUY, Recommendation.BUY]
            ]

        # 점수 + 신뢰도 기준 정렬
        sorted_results = sorted(
            filtered,
            key=lambda x: (
                x.recommendation == Recommendation.STRONG_BUY,  # STRONG_BUY 우선
                x.total_score,
                x.confidence
            ),
            reverse=True
        )

        return sorted_results[:n]

    def summarize_decisions(self, results: list[DecisionResult]) -> dict:
        """
        판정 결과 요약

        Returns:
            {
                "total": 전체 개수,
                "strong_buy": STRONG_BUY 개수,
                "buy": BUY 개수,
                "watch": WATCH 개수,
                "avoid": AVOID 개수,
                "top_picks": 상위 3개 종목
            }
        """
        summary = {
            "total": len(results),
            "strong_buy": 0,
            "buy": 0,
            "watch": 0,
            "avoid": 0,
            "top_picks": [],
        }

        for result in results:
            if result.recommendation == Recommendation.STRONG_BUY:
                summary["strong_buy"] += 1
            elif result.recommendation == Recommendation.BUY:
                summary["buy"] += 1
            elif result.recommendation == Recommendation.WATCH:
                summary["watch"] += 1
            elif result.recommendation == Recommendation.AVOID:
                summary["avoid"] += 1

        # Top picks
        top = self.get_top_recommendations(results, n=3)
        summary["top_picks"] = [
            {
                "stock_code": r.stock_code,
                "stock_name": r.stock_name,
                "recommendation": r.recommendation.value,
                "material_grade": r.material_grade.value,
                "sentiment_stage": r.sentiment_stage.value,
            }
            for r in top
        ]

        return summary
