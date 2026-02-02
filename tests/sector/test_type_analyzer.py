"""
SectorTypeAnalyzer 테스트 (v2.0)

테마명 기반 Type A/B 분류 테스트
"""
import pytest
from unittest.mock import patch, MagicMock

from src.sector.type_analyzer import SectorTypeAnalyzer, SectorTypeResult
from src.core.interfaces import SectorType


# =============================================================================
# SectorTypeResult 테스트
# =============================================================================

class TestSectorTypeResult:
    """SectorTypeResult 데이터클래스 테스트"""

    def test_create_type_a_result(self):
        """Type A 결과 생성"""
        result = SectorTypeResult(
            theme_name="은행",
            sector_type=SectorType.TYPE_A,
            reasoning="실적형 키워드 매칭",
            confidence=0.85,
            matched_keywords=["은행"],
        )

        assert result.theme_name == "은행"
        assert result.sector_type == SectorType.TYPE_A
        assert result.confidence == 0.85
        assert "은행" in result.matched_keywords

    def test_create_type_b_result(self):
        """Type B 결과 생성"""
        result = SectorTypeResult(
            theme_name="2차전지",
            sector_type=SectorType.TYPE_B,
            reasoning="성장형 키워드 매칭",
            confidence=0.9,
            matched_keywords=["2차전지", "배터리"],
        )

        assert result.theme_name == "2차전지"
        assert result.sector_type == SectorType.TYPE_B
        assert result.confidence == 0.9

    def test_to_dict(self):
        """to_dict 변환"""
        result = SectorTypeResult(
            theme_name="반도체",
            sector_type=SectorType.TYPE_B,
            reasoning="테스트",
            confidence=0.8,
            matched_keywords=["반도체"],
        )

        d = result.to_dict()

        assert d["theme_name"] == "반도체"
        assert d["sector_type"] == SectorType.TYPE_B.value  # "growth_driven"
        assert d["confidence"] == 0.8


# =============================================================================
# SectorTypeAnalyzer 초기화 테스트
# =============================================================================

class TestSectorTypeAnalyzerInit:
    """SectorTypeAnalyzer 초기화 테스트"""

    @pytest.fixture
    def analyzer(self):
        with patch("src.sector.type_analyzer.get_config"), \
             patch("src.sector.type_analyzer.get_logger"):
            return SectorTypeAnalyzer(llm_client=None, use_llm=False)

    def test_init(self, analyzer):
        """초기화"""
        assert analyzer._llm_client is None
        assert analyzer.use_llm is False

    def test_set_llm_client(self, analyzer):
        """LLM 클라이언트 설정"""
        mock_llm = MagicMock()
        analyzer.set_llm_client(mock_llm)
        assert analyzer._llm_client == mock_llm

    def test_type_b_keywords_defined(self, analyzer):
        """Type B 키워드 정의 확인"""
        assert "반도체" in analyzer.TYPE_B_KEYWORDS
        assert "2차전지" in analyzer.TYPE_B_KEYWORDS
        assert "바이오" in analyzer.TYPE_B_KEYWORDS
        assert "AI" in analyzer.TYPE_B_KEYWORDS

    def test_type_a_keywords_defined(self, analyzer):
        """Type A 키워드 정의 확인"""
        assert "은행" in analyzer.TYPE_A_KEYWORDS
        assert "건설" in analyzer.TYPE_A_KEYWORDS
        assert "자동차" in analyzer.TYPE_A_KEYWORDS
        assert "철강" in analyzer.TYPE_A_KEYWORDS


# =============================================================================
# 키워드 기반 분류 테스트
# =============================================================================

class TestKeywordBasedClassification:
    """키워드 기반 분류 테스트"""

    @pytest.fixture
    def analyzer(self):
        with patch("src.sector.type_analyzer.get_config"), \
             patch("src.sector.type_analyzer.get_logger"):
            return SectorTypeAnalyzer(llm_client=None, use_llm=False)

    def test_type_b_theme_semiconductor(self, analyzer):
        """반도체 테마 → Type B"""
        result = analyzer.analyze_theme("반도체")
        assert result.sector_type == SectorType.TYPE_B
        assert result.confidence >= 0.8
        assert "반도체" in result.matched_keywords

    def test_type_b_theme_battery(self, analyzer):
        """2차전지 테마 → Type B"""
        result = analyzer.analyze_theme("2차전지(소재)")
        assert result.sector_type == SectorType.TYPE_B
        assert "2차전지" in result.matched_keywords

    def test_type_b_theme_bio(self, analyzer):
        """바이오 테마 → Type B"""
        result = analyzer.analyze_theme("바이오시밀러")
        assert result.sector_type == SectorType.TYPE_B
        assert "바이오" in result.matched_keywords

    def test_type_b_theme_ai(self, analyzer):
        """AI 테마 → Type B"""
        result = analyzer.analyze_theme("AI/인공지능")
        assert result.sector_type == SectorType.TYPE_B

    def test_type_a_theme_bank(self, analyzer):
        """은행 테마 → Type A"""
        result = analyzer.analyze_theme("은행")
        assert result.sector_type == SectorType.TYPE_A
        assert result.confidence >= 0.8
        assert "은행" in result.matched_keywords

    def test_type_a_theme_construction(self, analyzer):
        """건설 테마 → Type A"""
        result = analyzer.analyze_theme("건설/인테리어")
        assert result.sector_type == SectorType.TYPE_A

    def test_type_a_theme_auto(self, analyzer):
        """자동차 테마 → Type A"""
        result = analyzer.analyze_theme("자동차부품")
        assert result.sector_type == SectorType.TYPE_A

    def test_type_a_theme_steel(self, analyzer):
        """철강 테마 → Type A"""
        result = analyzer.analyze_theme("철강/금속")
        assert result.sector_type == SectorType.TYPE_A

    def test_unknown_theme_default_type_a(self, analyzer):
        """알 수 없는 테마 → 기본값 Type A"""
        result = analyzer.analyze_theme("알수없는테마XYZ")
        assert result.sector_type == SectorType.TYPE_A
        assert result.confidence < 0.8  # 낮은 신뢰도
        assert result.matched_keywords is None

    def test_mixed_keywords_type_b_dominant(self, analyzer):
        """혼합 키워드 (Type B 우세)"""
        # "전기차" = Type B, 둘 다 있으면 개수 비교
        result = analyzer.analyze_theme("전기차/자동차")
        # 전기차(B), 자동차(A) → 개수 같으면 B 우선
        assert result.sector_type in [SectorType.TYPE_A, SectorType.TYPE_B]
        assert result.confidence < 0.85  # 혼합이므로 낮은 신뢰도


# =============================================================================
# 배치 분류 테스트
# =============================================================================

class TestBatchClassification:
    """배치 분류 테스트"""

    @pytest.fixture
    def analyzer(self):
        with patch("src.sector.type_analyzer.get_config"), \
             patch("src.sector.type_analyzer.get_logger"):
            return SectorTypeAnalyzer(llm_client=None, use_llm=False)

    def test_batch_classification(self, analyzer):
        """배치 분류"""
        themes = ["반도체", "은행", "2차전지", "건설", "바이오"]

        results = analyzer.analyze_batch(themes)

        assert len(results) == 5

        # 반도체, 2차전지, 바이오 → Type B
        semiconductor = next(r for r in results if r.theme_name == "반도체")
        assert semiconductor.sector_type == SectorType.TYPE_B

        battery = next(r for r in results if r.theme_name == "2차전지")
        assert battery.sector_type == SectorType.TYPE_B

        bio = next(r for r in results if r.theme_name == "바이오")
        assert bio.sector_type == SectorType.TYPE_B

        # 은행, 건설 → Type A
        bank = next(r for r in results if r.theme_name == "은행")
        assert bank.sector_type == SectorType.TYPE_A

        construction = next(r for r in results if r.theme_name == "건설")
        assert construction.sector_type == SectorType.TYPE_A

    def test_batch_empty(self, analyzer):
        """빈 리스트"""
        results = analyzer.analyze_batch([])
        assert results == []


# =============================================================================
# LLM 통합 테스트
# =============================================================================

class TestLLMIntegration:
    """LLM 통합 테스트"""

    @pytest.fixture
    def analyzer_with_llm(self):
        with patch("src.sector.type_analyzer.get_config"), \
             patch("src.sector.type_analyzer.get_logger"):
            mock_llm = MagicMock()
            mock_llm.generate.return_value = "Type: B\n이유: 기술 혁신이 주가를 이끔"
            return SectorTypeAnalyzer(llm_client=mock_llm, use_llm=True)

    def test_llm_classification_for_uncertain(self, analyzer_with_llm):
        """불확실한 테마 LLM 분류"""
        # 키워드 매칭 없는 테마
        result = analyzer_with_llm.analyze_theme("알수없는테마")

        # LLM이 Type B로 분류
        assert result.sector_type == SectorType.TYPE_B
        assert result.confidence == 0.9

    def test_llm_not_called_for_certain(self, analyzer_with_llm):
        """확실한 테마는 LLM 호출 안함"""
        result = analyzer_with_llm.analyze_theme("반도체")

        # 키워드로 확실히 분류되므로 LLM 호출 안함
        assert result.sector_type == SectorType.TYPE_B
        assert result.confidence == 0.85
        analyzer_with_llm._llm_client.generate.assert_not_called()

    def test_llm_failure_fallback(self):
        """LLM 실패 시 키워드 결과 반환"""
        with patch("src.sector.type_analyzer.get_config"), \
             patch("src.sector.type_analyzer.get_logger"):
            mock_llm = MagicMock()
            mock_llm.generate.side_effect = Exception("LLM Error")
            analyzer = SectorTypeAnalyzer(llm_client=mock_llm, use_llm=True)

            result = analyzer.analyze_theme("알수없는테마")

            # 키워드 기반 결과 (기본값 Type A)
            assert result.sector_type == SectorType.TYPE_A
            assert result.confidence == 0.5


# =============================================================================
# 유틸리티 메서드 테스트
# =============================================================================

class TestUtilityMethods:
    """유틸리티 메서드 테스트"""

    @pytest.fixture
    def analyzer(self):
        with patch("src.sector.type_analyzer.get_config"), \
             patch("src.sector.type_analyzer.get_logger"):
            return SectorTypeAnalyzer(llm_client=None, use_llm=False)

    def test_get_type_description(self, analyzer):
        """Type 설명 조회"""
        desc_a = analyzer.get_type_description(SectorType.TYPE_A)
        assert desc_a["name"] == "실적 기반"
        assert "영업이익" in desc_a["key_factors"]

        desc_b = analyzer.get_type_description(SectorType.TYPE_B)
        assert desc_b["name"] == "성장 기반"
        assert "기술력" in desc_b["key_factors"]

    def test_summarize(self, analyzer):
        """분류 결과 요약"""
        results = [
            SectorTypeResult(
                theme_name="반도체", sector_type=SectorType.TYPE_B,
                reasoning="", confidence=0.9, matched_keywords=[]
            ),
            SectorTypeResult(
                theme_name="2차전지", sector_type=SectorType.TYPE_B,
                reasoning="", confidence=0.85, matched_keywords=[]
            ),
            SectorTypeResult(
                theme_name="은행", sector_type=SectorType.TYPE_A,
                reasoning="", confidence=0.9, matched_keywords=[]
            ),
        ]

        summary = analyzer.summarize(results)

        assert summary["total_themes"] == 3
        assert summary["type_a_count"] == 1
        assert summary["type_b_count"] == 2
        assert "반도체" in summary["type_b_themes"]
        assert "은행" in summary["type_a_themes"]

    def test_get_filter_criteria_type_a(self, analyzer):
        """Type A 필터 기준"""
        criteria = analyzer.get_filter_criteria(SectorType.TYPE_A)

        assert "min_operating_profit_4q" in criteria
        assert "max_debt_ratio" in criteria
        assert criteria["max_debt_ratio"] == 200

    def test_get_filter_criteria_type_b(self, analyzer):
        """Type B 필터 기준"""
        criteria = analyzer.get_filter_criteria(SectorType.TYPE_B)

        assert "max_capital_impairment" in criteria
        assert "min_rd_ratio" in criteria
        assert criteria["min_rd_ratio"] == 5


# =============================================================================
# 통합 테스트
# =============================================================================

class TestIntegration:
    """통합 테스트"""

    @pytest.fixture
    def analyzer(self):
        with patch("src.sector.type_analyzer.get_config"), \
             patch("src.sector.type_analyzer.get_logger"):
            return SectorTypeAnalyzer(llm_client=None, use_llm=False)

    def test_full_workflow(self, analyzer):
        """전체 워크플로우"""
        # 1. 테마 목록 (실제 네이버 테마명)
        themes = [
            "2차전지(소재/부품)",
            "반도체 장비",
            "AI/인공지능",
            "은행주",
            "건설/인테리어",
            "바이오시밀러",
            "철강/비철금속",
            "전기차 충전",
        ]

        # 2. 배치 분류
        results = analyzer.analyze_batch(themes)
        assert len(results) == len(themes)

        # 3. 요약
        summary = analyzer.summarize(results)
        assert summary["total_themes"] == len(themes)

        # Type B 테마들 (2차전지, 반도체, AI, 바이오, 전기차)
        assert summary["type_b_count"] >= 4

        # Type A 테마들 (은행, 건설, 철강)
        assert summary["type_a_count"] >= 2

        # 4. 각 테마에 대한 필터 기준 확인
        for result in results:
            criteria = analyzer.get_filter_criteria(result.sector_type)
            assert "min_trading_value" in criteria
