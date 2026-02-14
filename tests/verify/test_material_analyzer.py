"""
Stage 5: MaterialAnalyzer 모듈 테스트
- 규칙 기반 재료 분석
- LLM 기반 재료 분석 (mock)
- 배치 분석 / 요약
"""
import pytest
from unittest.mock import MagicMock

from src.core.interfaces import MaterialGrade
from src.verify.material_analyzer import MaterialAnalyzer, MaterialResult


class TestMaterialResult:
    """MaterialResult 데이터 클래스 테스트"""

    def test_to_dict(self):
        r = MaterialResult(
            stock_code="005930",
            stock_name="삼성전자",
            grade=MaterialGrade.A,
            confidence=0.85,
            key_materials=["대규모 수주"],
            positive_factors=["수주"],
            negative_factors=[],
            llm_analysis="테스트 분석",
            news_count=3,
            disclosure_count=1,
        )
        d = r.to_dict()
        assert d["grade"] == "A"
        assert d["confidence"] == 0.85
        assert d["news_count"] == 3


class TestMaterialAnalyzerRules:
    """규칙 기반 분석 테스트 (LLM 없음)"""

    def test_grade_s_keywords(self):
        """S등급 키워드 매칭"""
        analyzer = MaterialAnalyzer()
        result = analyzer.analyze(
            "005930", "삼성전자",
            news_headlines=["삼성전자 실적 서프라이즈, 사상 최대 영업이익"],
        )
        assert result.grade == MaterialGrade.S
        assert len(result.key_materials) > 0

    def test_grade_a_keywords(self):
        """A등급 키워드 매칭"""
        analyzer = MaterialAnalyzer()
        result = analyzer.analyze(
            "000000", "테스트종목",
            news_headlines=["신사업 진출 발표", "글로벌 파트너십 체결"],
        )
        assert result.grade in [MaterialGrade.S, MaterialGrade.A]

    def test_grade_c_negative(self):
        """C등급 (악재 다수)"""
        analyzer = MaterialAnalyzer()
        result = analyzer.analyze(
            "000000", "테스트종목",
            news_headlines=["적자 전환 확인", "매출 하락 지속", "소송 제기"],
        )
        assert result.grade == MaterialGrade.C
        assert len(result.negative_factors) > 0

    def test_no_news_grade_c(self):
        """뉴스 없으면 C등급"""
        analyzer = MaterialAnalyzer()
        result = analyzer.analyze("000000", "테스트종목")
        assert result.grade == MaterialGrade.C
        assert result.news_count == 0

    def test_disclosures_included(self):
        """공시도 분석에 반영"""
        analyzer = MaterialAnalyzer()
        result = analyzer.analyze(
            "000000", "테스트종목",
            disclosures=["대규모 수주 공시"],
        )
        assert result.disclosure_count == 1
        assert result.grade in [MaterialGrade.S, MaterialGrade.A]

    def test_confidence_rule_based(self):
        """규칙 기반은 confidence=0.6"""
        analyzer = MaterialAnalyzer()
        result = analyzer.analyze("000000", "테스트종목")
        assert result.confidence == 0.6


class TestMaterialAnalyzerLLM:
    """LLM 기반 분석 테스트 (mock)"""

    def test_llm_analysis(self):
        """LLM 응답 파싱"""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            "등급: A\n"
            "핵심재료: 신사업 진출, MOU 체결\n"
            "긍정요소: 성장 모멘텀\n"
            "부정요소: 경쟁 심화\n"
            "분석: 신사업 진출이 긍정적이나 경쟁 심화 우려"
        )

        analyzer = MaterialAnalyzer(llm_client=mock_llm)
        result = analyzer.analyze(
            "005930", "삼성전자",
            news_headlines=["신사업 진출 발표"],
        )
        assert result.grade == MaterialGrade.A
        assert "신사업 진출" in result.key_materials
        assert result.confidence == 0.85

    def test_llm_fallback_on_error(self):
        """LLM 실패 시 규칙 기반 폴백"""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("LLM 연결 실패")

        analyzer = MaterialAnalyzer(llm_client=mock_llm)
        result = analyzer.analyze(
            "000000", "테스트종목",
            news_headlines=["실적 서프라이즈"],
        )
        assert result.confidence == 0.6  # 규칙 기반 폴백
        assert result.grade == MaterialGrade.S

    def test_set_llm_client(self):
        """LLM 클라이언트 후설정"""
        analyzer = MaterialAnalyzer()
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "등급: B\n분석: 소형 호재"

        analyzer.set_llm_client(mock_llm)
        result = analyzer.analyze("000000", "테스트종목", news_headlines=["개발 완료"])
        assert mock_llm.generate.called


class TestMaterialAnalyzerBatch:
    """배치 분석 테스트"""

    def test_analyze_batch(self):
        """배치 분석"""
        analyzer = MaterialAnalyzer()
        stocks_data = [
            {"stock_code": "A", "stock_name": "종목A", "news_headlines": ["실적 서프라이즈"]},
            {"stock_code": "B", "stock_name": "종목B", "news_headlines": ["적자 전환"]},
        ]
        results = analyzer.analyze_batch(stocks_data)
        assert len(results) == 2
        assert results[0].stock_code == "A"
        assert results[1].stock_code == "B"

    def test_analyze_batch_progress_callback(self):
        """진행 콜백 호출"""
        analyzer = MaterialAnalyzer()
        stocks_data = [
            {"stock_code": f"S{i}", "stock_name": f"종목{i}"}
            for i in range(10)
        ]
        callback_calls = []
        results = analyzer.analyze_batch(
            stocks_data,
            progress_callback=lambda cur, total: callback_calls.append((cur, total)),
        )
        assert len(results) == 10
        assert (5, 10) in callback_calls
        assert (10, 10) in callback_calls

    def test_summarize(self):
        """요약"""
        analyzer = MaterialAnalyzer()
        results = [
            MaterialResult("A", "종목A", MaterialGrade.S, 0.8, [], [], [], "", 1, 0),
            MaterialResult("B", "종목B", MaterialGrade.A, 0.7, [], [], [], "", 2, 0),
            MaterialResult("C", "종목C", MaterialGrade.C, 0.6, [], [], [], "", 0, 0),
        ]
        summary = analyzer.summarize(results)
        assert summary["total"] == 3
        assert "종목A" in summary["s_grade_stocks"]
        assert "종목B" in summary["a_grade_stocks"]
        assert summary["grade_distribution"]["S"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
