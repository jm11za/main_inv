"""
Stage 5: DecisionEngine 모듈 테스트
- STRONG_BUY / BUY / WATCH / AVOID 판정 로직
- 배치 판정 / 유틸리티
"""
import pytest

from src.core.interfaces import (
    Recommendation, MaterialGrade, SentimentStage, TrackType,
)
from src.verify.decision_engine import DecisionEngine, FinalDecision
from src.verify.material_analyzer import MaterialResult
from src.verify.sentiment_analyzer import SentimentResult


def _make_material(
    grade=MaterialGrade.A,
    confidence=0.8,
    key_materials=None,
    negative_factors=None,
) -> MaterialResult:
    return MaterialResult(
        stock_code="005930",
        stock_name="삼성전자",
        grade=grade,
        confidence=confidence,
        key_materials=key_materials or ["핵심재료"],
        positive_factors=["긍정"],
        negative_factors=negative_factors or [],
        llm_analysis="분석 내용",
    )


def _make_sentiment(
    stage=SentimentStage.DOUBT,
    confidence=0.8,
) -> SentimentResult:
    return SentimentResult(
        stock_code="005930",
        stock_name="삼성전자",
        stage=stage,
        confidence=confidence,
        bullish_ratio=0.4,
        activity_level="중간",
        key_sentiments=["관망"],
        llm_analysis="심리 분석",
    )


class TestDecisionEngineAvoid:
    """AVOID 판정 테스트"""

    def test_avoid_filter_fail(self):
        """필터 미통과 → AVOID"""
        engine = DecisionEngine()
        decision = engine.decide(
            stock_code="005930", stock_name="삼성전자",
            sector="반도체", track_type=TrackType.TRACK_A,
            sector_rank=1, total_score=80,
            filter_passed=False,
            material=_make_material(),
            sentiment=_make_sentiment(),
        )
        assert decision.recommendation == Recommendation.AVOID
        assert "필터 미통과" in decision.decision_factors

    def test_avoid_euphoria(self):
        """심리 환희 → AVOID"""
        engine = DecisionEngine()
        decision = engine.decide(
            stock_code="005930", stock_name="삼성전자",
            sector="반도체", track_type=TrackType.TRACK_A,
            sector_rank=1, total_score=80,
            filter_passed=True,
            material=_make_material(grade=MaterialGrade.S),
            sentiment=_make_sentiment(stage=SentimentStage.EUPHORIA),
        )
        assert decision.recommendation == Recommendation.AVOID
        assert "환희" in " ".join(decision.decision_factors)


class TestDecisionEngineStrongBuy:
    """STRONG_BUY 판정 테스트"""

    def test_strong_buy_all_conditions(self):
        """섹터 1~2위 + 재료 S/A + 심리 공포/의심 → STRONG_BUY"""
        engine = DecisionEngine()
        decision = engine.decide(
            stock_code="005930", stock_name="삼성전자",
            sector="반도체", track_type=TrackType.TRACK_A,
            sector_rank=1, total_score=80,
            filter_passed=True,
            material=_make_material(grade=MaterialGrade.S),
            sentiment=_make_sentiment(stage=SentimentStage.FEAR),
        )
        assert decision.recommendation == Recommendation.STRONG_BUY

    def test_strong_buy_rank_2(self):
        """섹터 2위도 STRONG_BUY 가능"""
        engine = DecisionEngine()
        decision = engine.decide(
            stock_code="005930", stock_name="삼성전자",
            sector="반도체", track_type=TrackType.TRACK_A,
            sector_rank=2, total_score=75,
            filter_passed=True,
            material=_make_material(grade=MaterialGrade.A),
            sentiment=_make_sentiment(stage=SentimentStage.DOUBT),
        )
        assert decision.recommendation == Recommendation.STRONG_BUY

    def test_not_strong_buy_rank_3(self):
        """섹터 3위면 STRONG_BUY 불가"""
        engine = DecisionEngine()
        decision = engine.decide(
            stock_code="005930", stock_name="삼성전자",
            sector="반도체", track_type=TrackType.TRACK_A,
            sector_rank=3, total_score=80,
            filter_passed=True,
            material=_make_material(grade=MaterialGrade.S),
            sentiment=_make_sentiment(stage=SentimentStage.FEAR),
        )
        assert decision.recommendation != Recommendation.STRONG_BUY

    def test_not_strong_buy_material_b(self):
        """재료 B등급이면 STRONG_BUY 불가"""
        engine = DecisionEngine()
        decision = engine.decide(
            stock_code="005930", stock_name="삼성전자",
            sector="반도체", track_type=TrackType.TRACK_A,
            sector_rank=1, total_score=80,
            filter_passed=True,
            material=_make_material(grade=MaterialGrade.B),
            sentiment=_make_sentiment(stage=SentimentStage.FEAR),
        )
        assert decision.recommendation != Recommendation.STRONG_BUY


class TestDecisionEngineBuy:
    """BUY 판정 테스트"""

    def test_buy_conditions(self):
        """점수>=60 + 재료 A이상 + 환희 아님 → BUY"""
        engine = DecisionEngine()
        decision = engine.decide(
            stock_code="005930", stock_name="삼성전자",
            sector="반도체", track_type=TrackType.TRACK_A,
            sector_rank=5, total_score=70,
            filter_passed=True,
            material=_make_material(grade=MaterialGrade.A),
            sentiment=_make_sentiment(stage=SentimentStage.CONVICTION),
        )
        assert decision.recommendation == Recommendation.BUY

    def test_buy_with_material_b(self):
        """재료 B등급도 BUY 가능"""
        engine = DecisionEngine()
        decision = engine.decide(
            stock_code="005930", stock_name="삼성전자",
            sector="반도체", track_type=TrackType.TRACK_A,
            sector_rank=5, total_score=65,
            filter_passed=True,
            material=_make_material(grade=MaterialGrade.B),
            sentiment=_make_sentiment(stage=SentimentStage.DOUBT),
        )
        assert decision.recommendation == Recommendation.BUY

    def test_not_buy_low_score(self):
        """점수 < 60이면 BUY 불가"""
        engine = DecisionEngine()
        decision = engine.decide(
            stock_code="005930", stock_name="삼성전자",
            sector="반도체", track_type=TrackType.TRACK_A,
            sector_rank=5, total_score=55,
            filter_passed=True,
            material=_make_material(grade=MaterialGrade.A),
            sentiment=_make_sentiment(stage=SentimentStage.DOUBT),
        )
        assert decision.recommendation != Recommendation.BUY

    def test_not_buy_material_c(self):
        """재료 C등급이면 BUY 불가"""
        engine = DecisionEngine()
        decision = engine.decide(
            stock_code="005930", stock_name="삼성전자",
            sector="반도체", track_type=TrackType.TRACK_A,
            sector_rank=5, total_score=70,
            filter_passed=True,
            material=_make_material(grade=MaterialGrade.C),
            sentiment=_make_sentiment(stage=SentimentStage.DOUBT),
        )
        assert decision.recommendation != Recommendation.BUY


class TestDecisionEngineWatch:
    """WATCH 판정 테스트"""

    def test_watch_score_50(self):
        """점수 >= 50 (BUY 미충족) → WATCH"""
        engine = DecisionEngine()
        decision = engine.decide(
            stock_code="005930", stock_name="삼성전자",
            sector="반도체", track_type=TrackType.TRACK_A,
            sector_rank=5, total_score=55,
            filter_passed=True,
            material=_make_material(grade=MaterialGrade.C),
            sentiment=_make_sentiment(stage=SentimentStage.DOUBT),
        )
        assert decision.recommendation == Recommendation.WATCH

    def test_not_watch_below_50(self):
        """점수 < 50 → AVOID"""
        engine = DecisionEngine()
        decision = engine.decide(
            stock_code="005930", stock_name="삼성전자",
            sector="반도체", track_type=TrackType.TRACK_A,
            sector_rank=5, total_score=40,
            filter_passed=True,
            material=_make_material(grade=MaterialGrade.C),
            sentiment=_make_sentiment(stage=SentimentStage.DOUBT),
        )
        assert decision.recommendation == Recommendation.AVOID


class TestFinalDecision:
    """FinalDecision 데이터 클래스 테스트"""

    def test_to_dict(self):
        engine = DecisionEngine()
        decision = engine.decide(
            stock_code="005930", stock_name="삼성전자",
            sector="반도체", track_type=TrackType.TRACK_A,
            sector_rank=1, total_score=80,
            filter_passed=True,
            material=_make_material(grade=MaterialGrade.S),
            sentiment=_make_sentiment(stage=SentimentStage.FEAR),
        )
        d = decision.to_dict()
        assert d["stock_code"] == "005930"
        assert d["recommendation"] == "STRONG_BUY"
        assert d["sector"] == "반도체"
        assert "confidence" in d

    def test_confidence_boost_strong_buy(self):
        """STRONG_BUY는 confidence +0.1 보너스"""
        engine = DecisionEngine()
        decision = engine.decide(
            stock_code="005930", stock_name="삼성전자",
            sector="반도체", track_type=TrackType.TRACK_A,
            sector_rank=1, total_score=80,
            filter_passed=True,
            material=_make_material(grade=MaterialGrade.S, confidence=0.8),
            sentiment=_make_sentiment(stage=SentimentStage.FEAR, confidence=0.8),
        )
        # (0.8 + 0.8) / 2 + 0.1 = 0.9
        assert abs(decision.confidence - 0.9) < 0.01

    def test_confidence_penalty_avoid(self):
        """AVOID는 confidence -0.1 패널티"""
        engine = DecisionEngine()
        decision = engine.decide(
            stock_code="005930", stock_name="삼성전자",
            sector="반도체", track_type=TrackType.TRACK_A,
            sector_rank=5, total_score=30,
            filter_passed=True,
            material=_make_material(grade=MaterialGrade.C, confidence=0.6),
            sentiment=_make_sentiment(stage=SentimentStage.DOUBT, confidence=0.6),
        )
        # (0.6 + 0.6) / 2 - 0.1 = 0.5
        assert abs(decision.confidence - 0.5) < 0.01


class TestDecisionEngineBatch:
    """배치 판정 테스트"""

    def test_decide_batch(self):
        """배치 판정"""
        engine = DecisionEngine()
        candidates = [
            {
                "stock_code": "005930", "stock_name": "삼성전자",
                "sector": "반도체", "track_type": TrackType.TRACK_A,
                "sector_rank": 1, "total_score": 80, "filter_passed": True,
            },
            {
                "stock_code": "068270", "stock_name": "셀트리온",
                "sector": "바이오", "track_type": TrackType.TRACK_B,
                "sector_rank": 3, "total_score": 45, "filter_passed": True,
            },
        ]
        materials = [
            _make_material(grade=MaterialGrade.S),
            MaterialResult("068270", "셀트리온", MaterialGrade.C, 0.5, [], [], [], "", 0, 0),
        ]
        sentiments = [
            _make_sentiment(stage=SentimentStage.FEAR),
            SentimentResult("068270", "셀트리온", SentimentStage.DOUBT, 0.5, 0.3, "낮음", [], "", 0, 0, 0),
        ]
        decisions = engine.decide_batch(candidates, materials, sentiments)
        assert len(decisions) == 2

    def test_decide_batch_missing_analysis(self):
        """분석 결과 없는 종목은 건너뜀"""
        engine = DecisionEngine()
        candidates = [
            {
                "stock_code": "005930", "stock_name": "삼성전자",
                "sector": "반도체", "track_type": TrackType.TRACK_A,
                "sector_rank": 1, "total_score": 80, "filter_passed": True,
            },
        ]
        decisions = engine.decide_batch(candidates, [], [])
        assert len(decisions) == 0

    def test_get_by_recommendation(self):
        """판정별 필터링"""
        engine = DecisionEngine()
        d1 = engine.decide(
            "A", "종목A", "반도체", TrackType.TRACK_A,
            1, 80, True,
            _make_material(grade=MaterialGrade.S),
            _make_sentiment(stage=SentimentStage.FEAR),
        )
        d2 = engine.decide(
            "B", "종목B", "바이오", TrackType.TRACK_B,
            5, 40, True,
            _make_material(grade=MaterialGrade.C),
            _make_sentiment(stage=SentimentStage.DOUBT),
        )
        decisions = [d1, d2]
        strong = engine.get_by_recommendation(decisions, Recommendation.STRONG_BUY)
        avoid = engine.get_by_recommendation(decisions, Recommendation.AVOID)
        assert len(strong) == 1
        assert strong[0].stock_code == "A"
        assert len(avoid) == 1


class TestDecisionEngineRiskWarnings:
    """리스크 경고 테스트"""

    def test_conviction_warning_on_buy(self):
        """BUY + 확신 단계 → 추격 매수 주의"""
        engine = DecisionEngine()
        decision = engine.decide(
            "005930", "삼성전자", "반도체", TrackType.TRACK_A,
            5, 70, True,
            _make_material(grade=MaterialGrade.A),
            _make_sentiment(stage=SentimentStage.CONVICTION),
        )
        assert decision.recommendation == Recommendation.BUY
        assert any("추격" in w for w in decision.risk_warnings)

    def test_negative_factors_in_warnings(self):
        """부정 요소 → 리스크 경고에 포함"""
        engine = DecisionEngine()
        decision = engine.decide(
            "005930", "삼성전자", "반도체", TrackType.TRACK_A,
            1, 80, True,
            _make_material(
                grade=MaterialGrade.S,
                negative_factors=["경쟁 심화"],
            ),
            _make_sentiment(stage=SentimentStage.FEAR),
        )
        assert decision.recommendation == Recommendation.STRONG_BUY
        assert any("부정" in w or "경쟁" in w for w in decision.risk_warnings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
