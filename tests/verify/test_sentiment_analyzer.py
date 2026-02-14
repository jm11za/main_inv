"""
Stage 5: SentimentAnalyzer 모듈 테스트
- 규칙 기반 심리 분석
- LLM 기반 심리 분석 (mock)
- 배치 분석 / 요약
"""
import pytest
from unittest.mock import MagicMock

from src.core.interfaces import SentimentStage
from src.verify.sentiment_analyzer import SentimentAnalyzer, SentimentResult


class TestSentimentResult:
    """SentimentResult 데이터 클래스 테스트"""

    def test_to_dict(self):
        r = SentimentResult(
            stock_code="005930",
            stock_name="삼성전자",
            stage=SentimentStage.DOUBT,
            confidence=0.8,
            bullish_ratio=0.4,
            activity_level="중간",
            key_sentiments=["관망"],
            llm_analysis="테스트",
            post_count=10,
            total_likes=50,
            total_dislikes=10,
        )
        d = r.to_dict()
        assert d["stage"] == "의심"
        assert d["bullish_ratio"] == 0.4
        assert d["post_count"] == 10


class TestSentimentAnalyzerRules:
    """규칙 기반 분석 테스트"""

    def test_fear_keywords(self):
        """공포 키워드 → FEAR"""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(
            "005930", "삼성전자",
            community_posts=["손절해야 하나", "바닥이 어딘지 모르겠다", "망함", "최악이다"],
        )
        assert result.stage == SentimentStage.FEAR

    def test_euphoria_keywords(self):
        """환희 키워드 → EUPHORIA"""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(
            "005930", "삼성전자",
            community_posts=["대박이다", "부자된다", "올인했다", "영끌 가즈아", "무조건 간다"],
        )
        assert result.stage == SentimentStage.EUPHORIA

    def test_conviction_keywords(self):
        """확신 키워드 → CONVICTION"""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(
            "005930", "삼성전자",
            community_posts=["오른다", "가즈아", "목표가 달성", "홀딩 중", "수익 중"],
        )
        assert result.stage == SentimentStage.CONVICTION

    def test_doubt_default_no_posts(self):
        """게시물 없으면 기본 DOUBT"""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("005930", "삼성전자")
        assert result.stage == SentimentStage.DOUBT
        assert result.post_count == 0

    def test_likes_boost_conviction(self):
        """공감 비율 높으면 확신 쪽으로 보정"""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(
            "005930", "삼성전자",
            community_posts=["좋은 종목"],
            likes=100,
            dislikes=10,
        )
        assert result.stage in [SentimentStage.CONVICTION, SentimentStage.EUPHORIA]

    def test_dislikes_boost_fear(self):
        """비공감 비율 높으면 공포 쪽으로 보정"""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(
            "005930", "삼성전자",
            community_posts=["이 종목 어떤가요"],
            likes=5,
            dislikes=100,
        )
        assert result.stage == SentimentStage.FEAR

    def test_activity_level_high(self):
        """게시물 50개 초과 = 높음"""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(
            "005930", "삼성전자",
            community_posts=[f"글{i}" for i in range(60)],
        )
        assert result.activity_level == "높음"

    def test_activity_level_medium(self):
        """게시물 21~50개 = 중간"""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(
            "005930", "삼성전자",
            community_posts=[f"글{i}" for i in range(30)],
        )
        assert result.activity_level == "중간"

    def test_activity_level_low(self):
        """게시물 20개 이하 = 낮음"""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(
            "005930", "삼성전자",
            community_posts=[f"글{i}" for i in range(10)],
        )
        assert result.activity_level == "낮음"

    def test_confidence_rule_based(self):
        """규칙 기반은 confidence=0.6"""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("000000", "테스트종목")
        assert result.confidence == 0.6

    def test_bullish_ratio_range(self):
        """낙관 비율 0~1 범위"""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(
            "005930", "삼성전자",
            community_posts=["가즈아", "오른다", "손절"],
        )
        assert 0.0 <= result.bullish_ratio <= 1.0


class TestSentimentAnalyzerLLM:
    """LLM 기반 분석 테스트"""

    def test_llm_analysis(self):
        """LLM 응답 파싱"""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            "심리단계: 의심\n"
            "낙관비율: 35%\n"
            "활동수준: 중간\n"
            "핵심심리: 관망, 반신반의\n"
            "분석: 초기 상승에 대한 의심이 많은 상태"
        )

        analyzer = SentimentAnalyzer(llm_client=mock_llm)
        result = analyzer.analyze(
            "005930", "삼성전자",
            community_posts=["진짜 오를까?"],
        )
        assert result.stage == SentimentStage.DOUBT
        assert abs(result.bullish_ratio - 0.35) < 0.01
        assert result.confidence == 0.85

    def test_llm_fallback_on_error(self):
        """LLM 실패 시 규칙 기반 폴백"""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("연결 실패")

        analyzer = SentimentAnalyzer(llm_client=mock_llm)
        result = analyzer.analyze(
            "000000", "테스트종목",
            community_posts=["손절해야 하나", "바닥이다"],
        )
        assert result.confidence == 0.6  # 규칙 기반 폴백


class TestSentimentAnalyzerBatch:
    """배치 & 유틸리티 테스트"""

    def test_analyze_batch(self):
        """배치 분석"""
        analyzer = SentimentAnalyzer()
        data = [
            {"stock_code": "A", "stock_name": "종목A", "community_posts": ["손절", "망함"]},
            {"stock_code": "B", "stock_name": "종목B", "community_posts": ["대박", "올인"]},
        ]
        results = analyzer.analyze_batch(data)
        assert len(results) == 2

    def test_analyze_batch_progress_callback(self):
        """진행 콜백"""
        analyzer = SentimentAnalyzer()
        data = [{"stock_code": f"S{i}", "stock_name": f"종목{i}"} for i in range(10)]
        calls = []
        analyzer.analyze_batch(data, progress_callback=lambda c, t: calls.append((c, t)))
        assert (5, 10) in calls
        assert (10, 10) in calls

    def test_summarize(self):
        """요약"""
        analyzer = SentimentAnalyzer()
        results = [
            SentimentResult("A", "종목A", SentimentStage.FEAR, 0.6, 0.1, "낮음", [], "", 5, 0, 0),
            SentimentResult("B", "종목B", SentimentStage.EUPHORIA, 0.6, 0.9, "높음", [], "", 60, 0, 0),
            SentimentResult("C", "종목C", SentimentStage.DOUBT, 0.6, 0.3, "중간", [], "", 20, 0, 0),
        ]
        summary = analyzer.summarize(results)
        assert summary["total"] == 3
        assert "종목A" in summary["fear_stocks"]
        assert "종목B" in summary["euphoria_stocks"]
        assert summary["stage_distribution"]["공포"] == 1

    def test_get_stage_description(self):
        """단계 설명 조회"""
        analyzer = SentimentAnalyzer()
        desc = analyzer.get_stage_description(SentimentStage.FEAR)
        assert desc["name"] == "공포"
        assert "keywords" in desc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
