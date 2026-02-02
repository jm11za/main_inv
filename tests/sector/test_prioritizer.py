"""
SectorPrioritizer 테스트 (v2.0)

테마명 기반 우선순위 결정 테스트
"""
import pytest
from unittest.mock import patch, MagicMock

from src.sector.prioritizer import (
    SectorPrioritizer,
    SectorPriorityResult,
    ThemeMetrics,
    SectorMetrics,  # 하위 호환성 별칭
)
from src.core.interfaces import SectorType


# =============================================================================
# ThemeMetrics 테스트
# =============================================================================

class TestThemeMetrics:
    """ThemeMetrics 데이터클래스 테스트"""

    def test_create_type_a_metrics(self):
        """Type A 테마 지표 생성"""
        metrics = ThemeMetrics(
            theme_name="은행",
            sector_type=SectorType.TYPE_A,
            stock_count=10,
            s_flow=0.5,
            s_breadth=70.0,
            avg_operating_profit_yoy=15.0,
            positive_profit_ratio=0.8,
        )

        assert metrics.theme_name == "은행"
        assert metrics.sector_type == SectorType.TYPE_A
        assert metrics.stock_count == 10
        assert metrics.positive_profit_ratio == 0.8

    def test_create_type_b_metrics(self):
        """Type B 테마 지표 생성"""
        metrics = ThemeMetrics(
            theme_name="2차전지",
            sector_type=SectorType.TYPE_B,
            stock_count=20,
            s_trend=65.0,
            news_count=50,
            hot_keywords=["배터리", "전기차", "LFP"],
        )

        assert metrics.theme_name == "2차전지"
        assert metrics.sector_type == SectorType.TYPE_B
        assert metrics.s_trend == 65.0
        assert len(metrics.hot_keywords) == 3

    def test_to_dict(self):
        """to_dict 변환"""
        metrics = ThemeMetrics(
            theme_name="반도체",
            sector_type=SectorType.TYPE_B,
            stock_count=15,
            change_rate=2.5,
        )

        d = metrics.to_dict()

        assert d["theme_name"] == "반도체"
        assert d["sector_type"] == SectorType.TYPE_B.value
        assert d["stock_count"] == 15
        assert d["change_rate"] == 2.5

    def test_sector_metrics_alias(self):
        """SectorMetrics 별칭 확인"""
        assert SectorMetrics is ThemeMetrics


# =============================================================================
# SectorPriorityResult 테스트
# =============================================================================

class TestSectorPriorityResult:
    """SectorPriorityResult 데이터클래스 테스트"""

    def test_create_result(self):
        """결과 생성"""
        result = SectorPriorityResult(
            theme_name="반도체",
            sector_type=SectorType.TYPE_B,
            rank=1,
            score=85.5,
            is_selected=True,
            is_excluded=False,
        )

        assert result.theme_name == "반도체"
        assert result.rank == 1
        assert result.score == 85.5
        assert result.is_selected is True

    def test_to_dict(self):
        """to_dict 변환"""
        result = SectorPriorityResult(
            theme_name="2차전지",
            sector_type=SectorType.TYPE_B,
            rank=2,
            score=75.0,
            is_selected=True,
            is_excluded=False,
            flow_score=60.0,
            trend_score=80.0,
            llm_outlook="긍정",
        )

        d = result.to_dict()

        assert d["theme_name"] == "2차전지"
        assert d["rank"] == 2
        assert d["score"] == 75.0
        assert d["flow_score"] == 60.0
        assert d["llm_outlook"] == "긍정"


# =============================================================================
# SectorPrioritizer 초기화 테스트
# =============================================================================

class TestSectorPrioritizerInit:
    """SectorPrioritizer 초기화 테스트"""

    @pytest.fixture
    def prioritizer(self):
        with patch("src.sector.prioritizer.get_config"), \
             patch("src.sector.prioritizer.get_logger"):
            return SectorPrioritizer(llm_client=None, use_llm=False)

    def test_init(self, prioritizer):
        """초기화"""
        assert prioritizer._llm_client is None
        assert prioritizer.use_llm is False

    def test_set_llm_client(self, prioritizer):
        """LLM 클라이언트 설정"""
        mock_llm = MagicMock()
        prioritizer.set_llm_client(mock_llm)
        assert prioritizer._llm_client == mock_llm

    def test_type_weights_defined(self, prioritizer):
        """Type별 가중치 정의 확인"""
        assert prioritizer.TYPE_A_WEIGHTS["s_flow"] == 0.40
        assert prioritizer.TYPE_A_WEIGHTS["fundamental"] == 0.40
        assert prioritizer.TYPE_B_WEIGHTS["s_trend"] == 0.40
        assert prioritizer.TYPE_B_WEIGHTS["momentum"] == 0.40


# =============================================================================
# 점수 계산 테스트
# =============================================================================

class TestScoreCalculation:
    """점수 계산 테스트"""

    @pytest.fixture
    def prioritizer(self):
        with patch("src.sector.prioritizer.get_config"), \
             patch("src.sector.prioritizer.get_logger"):
            return SectorPrioritizer(llm_client=None, use_llm=False)

    def test_type_a_score(self, prioritizer):
        """Type A 점수 계산"""
        metrics = ThemeMetrics(
            theme_name="은행",
            sector_type=SectorType.TYPE_A,
            stock_count=10,
            s_flow=2.0,           # 수급 양호
            s_breadth=80.0,       # 결속력 높음
            avg_operating_profit_yoy=30.0,   # 실적 성장
            positive_profit_ratio=0.9,       # 흑자 비율 높음
        )

        result = prioritizer._calculate_score(metrics)

        assert result.theme_name == "은행"
        assert result.sector_type == SectorType.TYPE_A
        assert result.score > 50  # 높은 점수
        assert result.flow_score > 0
        assert result.fundamental_score > 0
        assert result.breadth_score == 80.0

    def test_type_b_score(self, prioritizer):
        """Type B 점수 계산"""
        metrics = ThemeMetrics(
            theme_name="2차전지",
            sector_type=SectorType.TYPE_B,
            stock_count=20,
            s_flow=1.0,
            s_trend=70.0,        # 추세 강함
            news_count=80,       # 뉴스 많음
            hot_keywords=["배터리", "전기차", "LFP", "ESS"],
        )

        result = prioritizer._calculate_score(metrics)

        assert result.theme_name == "2차전지"
        assert result.sector_type == SectorType.TYPE_B
        assert result.score > 50  # 높은 점수
        assert result.trend_score > 0
        assert result.momentum_score > 0

    def test_normalize_score(self, prioritizer):
        """정규화 테스트"""
        assert prioritizer._normalize_score(0, 0, 100) == 0.0
        assert prioritizer._normalize_score(50, 0, 100) == 0.5
        assert prioritizer._normalize_score(100, 0, 100) == 1.0
        assert prioritizer._normalize_score(-10, 0, 100) == 0.0  # 최소값 클램프
        assert prioritizer._normalize_score(110, 0, 100) == 1.0  # 최대값 클램프


# =============================================================================
# 우선순위 결정 테스트
# =============================================================================

class TestPrioritize:
    """prioritize 메서드 테스트"""

    @pytest.fixture
    def prioritizer(self):
        with patch("src.sector.prioritizer.get_config"), \
             patch("src.sector.prioritizer.get_logger"):
            return SectorPrioritizer(llm_client=None, use_llm=False)

    def test_prioritize_basic(self, prioritizer):
        """기본 우선순위 결정"""
        metrics_list = [
            ThemeMetrics(
                theme_name="2차전지",
                sector_type=SectorType.TYPE_B,
                stock_count=20,
                s_trend=80.0,
                news_count=100,
                hot_keywords=["배터리", "전기차"],
            ),
            ThemeMetrics(
                theme_name="은행",
                sector_type=SectorType.TYPE_A,
                stock_count=15,
                s_flow=1.5,
                s_breadth=60.0,
                avg_operating_profit_yoy=10.0,
                positive_profit_ratio=0.7,
            ),
            ThemeMetrics(
                theme_name="반도체",
                sector_type=SectorType.TYPE_B,
                stock_count=25,
                s_trend=70.0,
                news_count=80,
                hot_keywords=["HBM", "AI"],
            ),
        ]

        results = prioritizer.prioritize(metrics_list, top_n=2)

        assert len(results) == 3
        # 순위 확인
        assert results[0].rank == 1
        assert results[1].rank == 2
        assert results[2].rank == 3
        # 선정 확인 (top 2)
        assert results[0].is_selected is True
        assert results[1].is_selected is True
        assert results[2].is_selected is False

    def test_exclude_low_score(self, prioritizer):
        """낮은 점수 배제"""
        metrics_list = [
            ThemeMetrics(
                theme_name="저조한테마",
                sector_type=SectorType.TYPE_A,
                stock_count=10,
                s_flow=-0.5,
                s_breadth=20.0,
                avg_operating_profit_yoy=-10.0,
                positive_profit_ratio=0.2,
            ),
        ]

        results = prioritizer.prioritize(metrics_list, top_n=5, min_score=50.0)

        assert len(results) == 1
        assert results[0].is_excluded is True
        assert "최소 점수 미달" in results[0].exclude_reason

    def test_exclude_few_stocks(self, prioritizer):
        """종목 수 부족 배제"""
        metrics_list = [
            ThemeMetrics(
                theme_name="소규모테마",
                sector_type=SectorType.TYPE_B,
                stock_count=2,  # 3개 미만
                s_trend=80.0,
                news_count=100,
            ),
        ]

        results = prioritizer.prioritize(metrics_list, top_n=5, min_score=0)

        assert len(results) == 1
        assert results[0].is_excluded is True
        assert "종목 수 부족" in results[0].exclude_reason

    def test_empty_list(self, prioritizer):
        """빈 리스트"""
        results = prioritizer.prioritize([], top_n=5)
        assert results == []


# =============================================================================
# LLM 통합 테스트
# =============================================================================

class TestLLMIntegration:
    """LLM 통합 테스트"""

    @pytest.fixture
    def prioritizer_with_llm(self):
        with patch("src.sector.prioritizer.get_config"), \
             patch("src.sector.prioritizer.get_logger"):
            mock_llm = MagicMock()
            mock_llm.generate.return_value = "긍정: 실적 개선 추세가 뚜렷함"
            return SectorPrioritizer(llm_client=mock_llm, use_llm=True)

    def test_llm_positive_outlook(self, prioritizer_with_llm):
        """LLM 긍정 전망"""
        metrics_list = [
            ThemeMetrics(
                theme_name="은행",
                sector_type=SectorType.TYPE_A,
                stock_count=10,
                s_flow=1.0,
                s_breadth=60.0,
                avg_operating_profit_yoy=15.0,
                positive_profit_ratio=0.8,
            ),
        ]

        results = prioritizer_with_llm.prioritize(metrics_list, top_n=5)

        assert len(results) == 1
        assert results[0].llm_outlook == "긍정"
        # 긍정이면 점수 가산 (+5)
        assert results[0].score > 0

    def test_llm_negative_excludes(self):
        """LLM 부정 전망 배제"""
        with patch("src.sector.prioritizer.get_config"), \
             patch("src.sector.prioritizer.get_logger"):
            mock_llm = MagicMock()
            mock_llm.generate.return_value = "부정: 시장 상황이 위험함"
            prioritizer = SectorPrioritizer(llm_client=mock_llm, use_llm=True)

            metrics_list = [
                ThemeMetrics(
                    theme_name="위험테마",
                    sector_type=SectorType.TYPE_B,
                    stock_count=10,
                    s_trend=80.0,        # 높은 추세 점수
                    s_flow=2.0,          # 높은 수급
                    news_count=100,      # 많은 뉴스
                    hot_keywords=["키워드1", "키워드2", "키워드3"],
                ),
            ]

            # min_score를 낮게 설정하여 점수 미달 배제 방지
            results = prioritizer.prioritize(metrics_list, top_n=5, min_score=0)

            assert results[0].llm_outlook == "부정"
            assert results[0].is_excluded is True
            assert "LLM 부정 전망" in results[0].exclude_reason


# =============================================================================
# 유틸리티 메서드 테스트
# =============================================================================

class TestUtilityMethods:
    """유틸리티 메서드 테스트"""

    @pytest.fixture
    def prioritizer(self):
        with patch("src.sector.prioritizer.get_config"), \
             patch("src.sector.prioritizer.get_logger"):
            return SectorPrioritizer(llm_client=None, use_llm=False)

    def test_get_selected_themes(self, prioritizer):
        """선정 테마 조회"""
        results = [
            SectorPriorityResult(
                theme_name="A", sector_type=SectorType.TYPE_A,
                rank=1, score=80, is_selected=True, is_excluded=False
            ),
            SectorPriorityResult(
                theme_name="B", sector_type=SectorType.TYPE_B,
                rank=2, score=70, is_selected=True, is_excluded=False
            ),
            SectorPriorityResult(
                theme_name="C", sector_type=SectorType.TYPE_A,
                rank=3, score=60, is_selected=False, is_excluded=False
            ),
        ]

        selected = prioritizer.get_selected_themes(results)

        assert len(selected) == 2
        assert selected[0].theme_name == "A"
        assert selected[1].theme_name == "B"

    def test_get_selected_sectors_alias(self, prioritizer):
        """get_selected_sectors 별칭 확인"""
        results = [
            SectorPriorityResult(
                theme_name="A", sector_type=SectorType.TYPE_A,
                rank=1, score=80, is_selected=True, is_excluded=False
            ),
        ]

        # 별칭 호출
        selected = prioritizer.get_selected_sectors(results)
        assert len(selected) == 1

    def test_get_excluded_themes(self, prioritizer):
        """배제 테마 조회"""
        results = [
            SectorPriorityResult(
                theme_name="A", sector_type=SectorType.TYPE_A,
                rank=1, score=80, is_selected=True, is_excluded=False
            ),
            SectorPriorityResult(
                theme_name="B", sector_type=SectorType.TYPE_B,
                rank=2, score=20, is_selected=False, is_excluded=True,
                exclude_reason="점수 미달"
            ),
        ]

        excluded = prioritizer.get_excluded_themes(results)

        assert len(excluded) == 1
        assert excluded[0].theme_name == "B"

    def test_summarize(self, prioritizer):
        """요약"""
        results = [
            SectorPriorityResult(
                theme_name="2차전지", sector_type=SectorType.TYPE_B,
                rank=1, score=85, is_selected=True, is_excluded=False,
                llm_outlook="긍정"
            ),
            SectorPriorityResult(
                theme_name="반도체", sector_type=SectorType.TYPE_B,
                rank=2, score=75, is_selected=True, is_excluded=False,
                llm_outlook="중립"
            ),
            SectorPriorityResult(
                theme_name="저조함", sector_type=SectorType.TYPE_A,
                rank=3, score=25, is_selected=False, is_excluded=True,
                exclude_reason="점수 미달"
            ),
        ]

        summary = prioritizer.summarize(results)

        assert summary["total_themes"] == 3
        assert summary["selected_count"] == 2
        assert summary["excluded_count"] == 1
        assert len(summary["selected_themes"]) == 2
        assert summary["selected_themes"][0]["theme_name"] == "2차전지"


# =============================================================================
# 통합 테스트
# =============================================================================

class TestIntegration:
    """통합 테스트"""

    @pytest.fixture
    def prioritizer(self):
        with patch("src.sector.prioritizer.get_config"), \
             patch("src.sector.prioritizer.get_logger"):
            return SectorPrioritizer(llm_client=None, use_llm=False)

    def test_full_workflow(self, prioritizer):
        """전체 워크플로우"""
        # 1. 테마 지표 생성 (실제 네이버 테마명)
        metrics_list = [
            ThemeMetrics(
                theme_name="2차전지(소재/부품)",
                sector_type=SectorType.TYPE_B,
                stock_count=25,
                change_rate=3.5,
                s_flow=1.5,
                s_trend=75.0,
                news_count=120,
                hot_keywords=["배터리", "전기차", "LFP", "전고체"],
            ),
            ThemeMetrics(
                theme_name="반도체 장비",
                sector_type=SectorType.TYPE_B,
                stock_count=18,
                change_rate=2.0,
                s_flow=2.0,
                s_trend=65.0,
                news_count=90,
                hot_keywords=["HBM", "AI", "메모리"],
            ),
            ThemeMetrics(
                theme_name="은행주",
                sector_type=SectorType.TYPE_A,
                stock_count=12,
                change_rate=0.5,
                s_flow=0.8,
                s_breadth=55.0,
                avg_operating_profit_yoy=12.0,
                positive_profit_ratio=0.9,
            ),
            ThemeMetrics(
                theme_name="건설/인테리어",
                sector_type=SectorType.TYPE_A,
                stock_count=15,
                change_rate=-1.0,
                s_flow=-0.3,
                s_breadth=35.0,
                avg_operating_profit_yoy=-5.0,
                positive_profit_ratio=0.5,
            ),
            ThemeMetrics(
                theme_name="소규모테마",
                sector_type=SectorType.TYPE_B,
                stock_count=2,  # 배제 대상
                s_trend=80.0,
                news_count=50,
            ),
        ]

        # 2. 우선순위 결정 (top 3)
        results = prioritizer.prioritize(metrics_list, top_n=3, min_score=30.0)

        assert len(results) == 5

        # 3. 선정/배제 확인
        selected = prioritizer.get_selected_themes(results)
        excluded = prioritizer.get_excluded_themes(results)

        # top 3 선정 (소규모테마 제외됨)
        assert len(selected) == 3
        # 종목 수 부족으로 배제
        assert any(r.theme_name == "소규모테마" for r in excluded)

        # 4. 요약
        summary = prioritizer.summarize(results)
        assert summary["total_themes"] == 5
        assert summary["selected_count"] == 3
        assert summary["excluded_count"] >= 1

        # 5. 순위 1위 확인
        top_theme = results[0]
        assert top_theme.rank == 1
        assert top_theme.is_selected is True
