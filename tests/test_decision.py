"""
Layer 5: Decision 모듈 테스트
"""
import pytest

from src.core.interfaces import (
    MaterialGrade,
    SentimentStage,
    Recommendation,
    Tier,
    TrackType,
)
from src.decision.personas.skeptic import Skeptic, SkepticAnalysis
from src.decision.personas.sentiment import SentimentReader, SentimentAnalysis
from src.decision.decision_engine import DecisionEngine, DecisionResult
from src.decision.llm_analyzer import LLMAnalyzer


class TestSkeptic:
    """Skeptic 페르소나 테스트"""

    def test_analyze_s_grade_keywords(self):
        """S급 재료 키워드 감지"""
        skeptic = Skeptic(llm_client=None)  # 규칙 기반만

        result = skeptic.analyze(
            stock_code="005930",
            stock_name="삼성전자",
            news_headlines=[
                "삼성전자, 대규모 수주 100조원 계약 체결",
                "FDA 승인으로 글로벌 시장 진출",
            ]
        )

        assert result.material_grade == MaterialGrade.S
        assert result.confidence >= 0.7
        assert len(result.key_materials) > 0

    def test_analyze_a_grade_keywords(self):
        """A급 재료 키워드 감지"""
        skeptic = Skeptic(llm_client=None)

        result = skeptic.analyze(
            stock_code="000660",
            stock_name="SK하이닉스",
            news_headlines=[
                "SK하이닉스 실적 개선 전망",
                "신규 계약 체결로 매출 증가",
            ]
        )

        assert result.material_grade == MaterialGrade.A
        assert len(result.key_materials) > 0

    def test_analyze_with_risks(self):
        """리스크 키워드 감지"""
        skeptic = Skeptic(llm_client=None)

        result = skeptic.analyze(
            stock_code="123456",
            stock_name="리스크기업",
            announcements=[
                "유상증자 결정 공시",
                "전환사채(CB) 발행 결정",
            ]
        )

        assert len(result.risks) > 0
        # 리스크가 많으면 등급 하향
        assert result.material_grade in [MaterialGrade.B, MaterialGrade.C]

    def test_analyze_no_material(self):
        """재료 없음"""
        skeptic = Skeptic(llm_client=None)

        result = skeptic.analyze(
            stock_code="999999",
            stock_name="무재료기업",
            news_headlines=[]
        )

        assert result.material_grade == MaterialGrade.C

    def test_analyze_batch(self):
        """일괄 분석"""
        skeptic = Skeptic(llm_client=None)

        stocks_data = [
            {"stock_code": "001", "stock_name": "A기업", "news_headlines": ["대규모 수주"]},
            {"stock_code": "002", "stock_name": "B기업", "news_headlines": ["실적 개선"]},
        ]

        results = skeptic.analyze_batch(stocks_data)

        assert len(results) == 2
        assert results[0].stock_code == "001"


class TestSentimentReader:
    """Sentiment Reader 페르소나 테스트"""

    def test_analyze_fear_stage(self):
        """공포 단계 감지"""
        reader = SentimentReader(llm_client=None)

        result = reader.analyze(
            stock_code="005930",
            stock_name="삼성전자",
            community_posts=[
                "이 종목 포기했어요",
                "손절하고 나왔습니다",
                "반토막 났네요",
            ],
            mention_count=5
        )

        assert result.sentiment_stage == SentimentStage.FEAR
        assert result.interest_level < 0.3

    def test_analyze_euphoria_stage(self):
        """환희 단계 감지"""
        reader = SentimentReader(llm_client=None)

        result = reader.analyze(
            stock_code="000660",
            stock_name="SK하이닉스",
            community_posts=[
                "떡상 갑니다!",
                "10배 간다 가즈아!",
                "전재산 풀매수",
                "영끌해서 샀어요",
                "인생역전 기회",
            ] * 10,  # 많은 글
            mention_count=200
        )

        assert result.sentiment_stage == SentimentStage.EUPHORIA
        assert result.interest_level > 0.7

    def test_analyze_conviction_stage(self):
        """확신 단계 감지"""
        reader = SentimentReader(llm_client=None)

        result = reader.analyze(
            stock_code="035720",
            stock_name="카카오",
            community_posts=[
                "상승 추세 좋아요",
                "매수 추천합니다",
                "신고가 돌파할 듯",
            ],
            mention_count=50
        )

        assert result.sentiment_stage in [SentimentStage.CONVICTION, SentimentStage.DOUBT]

    def test_analyze_low_interest(self):
        """낮은 관심도"""
        reader = SentimentReader(llm_client=None)

        result = reader.analyze(
            stock_code="999999",
            stock_name="관심없는기업",
            community_posts=[],
            mention_count=0
        )

        assert result.interest_level < 0.2
        assert result.sentiment_stage == SentimentStage.FEAR  # 관심 없음 = 바닥권

    def test_tone_score_calculation(self):
        """어조 점수 계산"""
        reader = SentimentReader(llm_client=None)

        # 긍정적 의견 다수
        result_positive = reader.analyze(
            stock_code="001",
            stock_name="긍정기업",
            community_posts=["좋아요", "추천합니다", "대박 기대"]
        )

        # 부정적 의견 다수
        result_negative = reader.analyze(
            stock_code="002",
            stock_name="부정기업",
            community_posts=["사기", "폭락", "손해봤어요"]
        )

        assert result_positive.tone_score > result_negative.tone_score


class TestDecisionEngine:
    """Decision Engine 테스트"""

    def test_euphoria_always_avoid(self):
        """환희 상태면 무조건 AVOID"""
        engine = DecisionEngine()

        skeptic = SkepticAnalysis(
            stock_code="005930",
            stock_name="삼성전자",
            material_grade=MaterialGrade.S,  # 최고급 재료
            confidence=0.9,
        )

        sentiment = SentimentAnalysis(
            stock_code="005930",
            stock_name="삼성전자",
            sentiment_stage=SentimentStage.EUPHORIA,  # 과열
            confidence=0.9,
            interest_level=0.9,
            tone_score=0.9,
        )

        result = engine.decide(
            stock_code="005930",
            stock_name="삼성전자",
            tier=Tier.TIER_1,
            track_type=TrackType.TRACK_A,
            total_score=95.0,
            skeptic_analysis=skeptic,
            sentiment_analysis=sentiment,
        )

        assert result.recommendation == Recommendation.AVOID

    def test_no_material_watch(self):
        """재료 C급이면 WATCH"""
        engine = DecisionEngine()

        skeptic = SkepticAnalysis(
            stock_code="005930",
            stock_name="삼성전자",
            material_grade=MaterialGrade.C,
            confidence=0.7,
        )

        sentiment = SentimentAnalysis(
            stock_code="005930",
            stock_name="삼성전자",
            sentiment_stage=SentimentStage.FEAR,
            confidence=0.7,
            interest_level=0.1,
            tone_score=0.0,
        )

        result = engine.decide(
            stock_code="005930",
            stock_name="삼성전자",
            tier=Tier.TIER_1,
            track_type=TrackType.TRACK_A,
            total_score=80.0,
            skeptic_analysis=skeptic,
            sentiment_analysis=sentiment,
        )

        assert result.recommendation == Recommendation.WATCH

    def test_tier1_strong_buy(self):
        """Tier 1 + 좋은 재료 + 초기 심리 = STRONG_BUY"""
        engine = DecisionEngine()

        skeptic = SkepticAnalysis(
            stock_code="005930",
            stock_name="삼성전자",
            material_grade=MaterialGrade.S,
            confidence=0.9,
            key_materials=["대규모 수주"],
        )

        sentiment = SentimentAnalysis(
            stock_code="005930",
            stock_name="삼성전자",
            sentiment_stage=SentimentStage.DOUBT,  # 초기
            confidence=0.8,
            interest_level=0.3,
            tone_score=0.2,
        )

        result = engine.decide(
            stock_code="005930",
            stock_name="삼성전자",
            tier=Tier.TIER_1,
            track_type=TrackType.TRACK_A,
            total_score=85.0,
            skeptic_analysis=skeptic,
            sentiment_analysis=sentiment,
        )

        assert result.recommendation == Recommendation.STRONG_BUY
        assert result.confidence >= 0.8

    def test_tier2_buy(self):
        """Tier 2 + 재료 + 중기 심리 = BUY"""
        engine = DecisionEngine()

        skeptic = SkepticAnalysis(
            stock_code="000660",
            stock_name="SK하이닉스",
            material_grade=MaterialGrade.A,
            confidence=0.8,
        )

        sentiment = SentimentAnalysis(
            stock_code="000660",
            stock_name="SK하이닉스",
            sentiment_stage=SentimentStage.CONVICTION,
            confidence=0.7,
            interest_level=0.5,  # 중간 관심도
            tone_score=0.4,
        )

        result = engine.decide(
            stock_code="000660",
            stock_name="SK하이닉스",
            tier=Tier.TIER_2,
            track_type=TrackType.TRACK_A,
            total_score=78.0,
            skeptic_analysis=skeptic,
            sentiment_analysis=sentiment,
        )

        assert result.recommendation == Recommendation.BUY

    def test_tier3_avoid(self):
        """Tier 3 = 가짜 상승 = AVOID"""
        engine = DecisionEngine()

        skeptic = SkepticAnalysis(
            stock_code="123456",
            stock_name="가짜상승기업",
            material_grade=MaterialGrade.A,
            confidence=0.8,
        )

        sentiment = SentimentAnalysis(
            stock_code="123456",
            stock_name="가짜상승기업",
            sentiment_stage=SentimentStage.CONVICTION,
            confidence=0.7,
            interest_level=0.5,
            tone_score=0.4,
        )

        result = engine.decide(
            stock_code="123456",
            stock_name="가짜상승기업",
            tier=Tier.TIER_3,
            track_type=TrackType.TRACK_A,
            total_score=70.0,
            skeptic_analysis=skeptic,
            sentiment_analysis=sentiment,
        )

        assert result.recommendation == Recommendation.AVOID

    def test_key_factors_extraction(self):
        """핵심 팩터 추출"""
        engine = DecisionEngine()

        skeptic = SkepticAnalysis(
            stock_code="005930",
            stock_name="삼성전자",
            material_grade=MaterialGrade.S,
            confidence=0.9,
            key_materials=["대규모 수주", "FDA 승인"],
        )

        sentiment = SentimentAnalysis(
            stock_code="005930",
            stock_name="삼성전자",
            sentiment_stage=SentimentStage.DOUBT,
            confidence=0.8,
            interest_level=0.3,
            tone_score=0.2,
        )

        result = engine.decide(
            stock_code="005930",
            stock_name="삼성전자",
            tier=Tier.TIER_1,
            track_type=TrackType.TRACK_A,
            total_score=85.0,
            skeptic_analysis=skeptic,
            sentiment_analysis=sentiment,
        )

        assert len(result.key_factors) > 0

    def test_risk_warnings_extraction(self):
        """리스크 경고 추출"""
        engine = DecisionEngine()

        skeptic = SkepticAnalysis(
            stock_code="005930",
            stock_name="삼성전자",
            material_grade=MaterialGrade.A,
            confidence=0.7,
            risks=["유상증자", "CB 발행"],
        )

        sentiment = SentimentAnalysis(
            stock_code="005930",
            stock_name="삼성전자",
            sentiment_stage=SentimentStage.CONVICTION,
            confidence=0.6,
            interest_level=0.5,
            tone_score=0.3,
        )

        result = engine.decide(
            stock_code="005930",
            stock_name="삼성전자",
            tier=Tier.TIER_2,
            track_type=TrackType.TRACK_A,
            total_score=75.0,
            skeptic_analysis=skeptic,
            sentiment_analysis=sentiment,
        )

        assert len(result.risk_warnings) > 0

    def test_summarize_decisions(self):
        """판정 요약"""
        engine = DecisionEngine()

        results = [
            DecisionResult(
                stock_code="001", stock_name="A기업",
                tier=Tier.TIER_1, track_type=TrackType.TRACK_A, total_score=90,
                material_grade=MaterialGrade.S, sentiment_stage=SentimentStage.DOUBT,
                recommendation=Recommendation.STRONG_BUY, confidence=0.9
            ),
            DecisionResult(
                stock_code="002", stock_name="B기업",
                tier=Tier.TIER_2, track_type=TrackType.TRACK_A, total_score=80,
                material_grade=MaterialGrade.A, sentiment_stage=SentimentStage.CONVICTION,
                recommendation=Recommendation.BUY, confidence=0.8
            ),
            DecisionResult(
                stock_code="003", stock_name="C기업",
                tier=Tier.TIER_3, track_type=TrackType.TRACK_B, total_score=60,
                material_grade=MaterialGrade.B, sentiment_stage=SentimentStage.EUPHORIA,
                recommendation=Recommendation.AVOID, confidence=0.85
            ),
        ]

        summary = engine.summarize_decisions(results)

        assert summary["total"] == 3
        assert summary["strong_buy"] == 1
        assert summary["buy"] == 1
        assert summary["avoid"] == 1
        assert len(summary["top_picks"]) <= 3


class TestLLMAnalyzer:
    """LLM Analyzer 통합 테스트"""

    def test_analyze_single_stock(self):
        """단일 종목 분석"""
        analyzer = LLMAnalyzer(use_llm=False)  # 규칙 기반만

        result = analyzer.analyze(
            stock_code="005930",
            stock_name="삼성전자",
            tier=Tier.TIER_1,
            track_type=TrackType.TRACK_A,
            total_score=85.0,
            news_headlines=["삼성전자 대규모 수주 계약 체결"],
            community_posts=["바닥권 같아요", "아직 모르는 사람 많네"],
        )

        assert result.stock_code == "005930"
        assert result.recommendation in [
            Recommendation.STRONG_BUY,
            Recommendation.BUY,
            Recommendation.WATCH,
            Recommendation.AVOID,
        ]

    def test_analyze_batch(self):
        """일괄 분석"""
        analyzer = LLMAnalyzer(use_llm=False)

        stocks_data = [
            {
                "stock_code": "005930",
                "stock_name": "삼성전자",
                "tier": Tier.TIER_1,
                "track_type": TrackType.TRACK_A,
                "total_score": 90.0,
                "news_headlines": ["대규모 수주"],
            },
            {
                "stock_code": "000660",
                "stock_name": "SK하이닉스",
                "tier": Tier.TIER_2,
                "track_type": TrackType.TRACK_A,
                "total_score": 80.0,
                "news_headlines": ["실적 개선"],
            },
        ]

        results = analyzer.analyze_batch(stocks_data, top_n=2)

        assert len(results) == 2

    def test_get_top_picks(self):
        """상위 추천 종목 필터링"""
        analyzer = LLMAnalyzer(use_llm=False)

        results = [
            DecisionResult(
                stock_code="001", stock_name="A기업",
                tier=Tier.TIER_1, track_type=TrackType.TRACK_A, total_score=90,
                material_grade=MaterialGrade.S, sentiment_stage=SentimentStage.DOUBT,
                recommendation=Recommendation.STRONG_BUY, confidence=0.9
            ),
            DecisionResult(
                stock_code="002", stock_name="B기업",
                tier=Tier.TIER_2, track_type=TrackType.TRACK_A, total_score=80,
                material_grade=MaterialGrade.A, sentiment_stage=SentimentStage.CONVICTION,
                recommendation=Recommendation.BUY, confidence=0.8
            ),
            DecisionResult(
                stock_code="003", stock_name="C기업",
                tier=Tier.TIER_3, track_type=TrackType.TRACK_B, total_score=60,
                material_grade=MaterialGrade.C, sentiment_stage=SentimentStage.FEAR,
                recommendation=Recommendation.WATCH, confidence=0.7
            ),
        ]

        top_picks = analyzer.get_top_picks(results, n=2)

        assert len(top_picks) == 2
        assert top_picks[0].recommendation == Recommendation.STRONG_BUY

    def test_format_report(self):
        """리포트 포맷팅"""
        analyzer = LLMAnalyzer(use_llm=False)

        results = [
            DecisionResult(
                stock_code="005930", stock_name="삼성전자",
                tier=Tier.TIER_1, track_type=TrackType.TRACK_A, total_score=90,
                material_grade=MaterialGrade.S, sentiment_stage=SentimentStage.DOUBT,
                recommendation=Recommendation.STRONG_BUY, confidence=0.9,
                key_factors=["수급 빈집", "대규모 수주"],
            ),
        ]

        report = analyzer.format_report(results)

        assert "삼성전자" in report
        assert "STRONG_BUY" in report


class TestDecisionMatrix:
    """Decision Matrix 로직 상세 테스트"""

    def test_matrix_tier1_s_fear(self):
        """Tier1 + S급 + 공포 = STRONG_BUY"""
        engine = DecisionEngine()

        skeptic = SkepticAnalysis(
            stock_code="TEST", stock_name="테스트",
            material_grade=MaterialGrade.S, confidence=0.9
        )
        sentiment = SentimentAnalysis(
            stock_code="TEST", stock_name="테스트",
            sentiment_stage=SentimentStage.FEAR,
            confidence=0.8, interest_level=0.1, tone_score=-0.3
        )

        result = engine.decide(
            "TEST", "테스트", Tier.TIER_1, TrackType.TRACK_A, 85.0,
            skeptic, sentiment
        )

        assert result.recommendation == Recommendation.STRONG_BUY

    def test_matrix_tier1_a_doubt(self):
        """Tier1 + A급 + 의심 = STRONG_BUY"""
        engine = DecisionEngine()

        skeptic = SkepticAnalysis(
            stock_code="TEST", stock_name="테스트",
            material_grade=MaterialGrade.A, confidence=0.8
        )
        sentiment = SentimentAnalysis(
            stock_code="TEST", stock_name="테스트",
            sentiment_stage=SentimentStage.DOUBT,
            confidence=0.7, interest_level=0.3, tone_score=0.1
        )

        result = engine.decide(
            "TEST", "테스트", Tier.TIER_1, TrackType.TRACK_A, 80.0,
            skeptic, sentiment
        )

        assert result.recommendation == Recommendation.STRONG_BUY

    def test_matrix_tier1_a_conviction(self):
        """Tier1 + A급 + 확신 = BUY"""
        engine = DecisionEngine()

        skeptic = SkepticAnalysis(
            stock_code="TEST", stock_name="테스트",
            material_grade=MaterialGrade.A, confidence=0.8
        )
        sentiment = SentimentAnalysis(
            stock_code="TEST", stock_name="테스트",
            sentiment_stage=SentimentStage.CONVICTION,
            confidence=0.7, interest_level=0.5, tone_score=0.4
        )

        result = engine.decide(
            "TEST", "테스트", Tier.TIER_1, TrackType.TRACK_A, 80.0,
            skeptic, sentiment
        )

        assert result.recommendation == Recommendation.BUY

    def test_matrix_tier2_b_doubt(self):
        """Tier2 + B급 + 의심 = BUY"""
        engine = DecisionEngine()

        skeptic = SkepticAnalysis(
            stock_code="TEST", stock_name="테스트",
            material_grade=MaterialGrade.B, confidence=0.7
        )
        sentiment = SentimentAnalysis(
            stock_code="TEST", stock_name="테스트",
            sentiment_stage=SentimentStage.DOUBT,
            confidence=0.7, interest_level=0.4, tone_score=0.2
        )

        result = engine.decide(
            "TEST", "테스트", Tier.TIER_2, TrackType.TRACK_A, 75.0,
            skeptic, sentiment
        )

        assert result.recommendation == Recommendation.BUY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
