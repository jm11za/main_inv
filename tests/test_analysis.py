"""
Layer 3: Analysis 모듈 테스트
"""
import pytest
import numpy as np
import pandas as pd

from src.analysis.metrics.flow import FlowCalculator, FlowResult, SectorFlowResult
from src.analysis.metrics.breadth import BreadthCalculator, BreadthResult, SectorBreadthResult
from src.analysis.metrics.trend import TrendCalculator, TrendResult, SectorTrendResult
from src.analysis.tier_classifier import TierClassifier, SectorTier, SectorAnalysisResult


class TestFlowCalculator:
    """S_Flow 계산기 테스트"""

    def test_calculate_stock_positive(self):
        """양수 수급"""
        calc = FlowCalculator()

        result = calc.calculate_stock(
            stock_code="005930",
            foreign_net=100_000_000_000,   # 외인 1000억 순매수
            institution_net=50_000_000_000, # 기관 500억 순매수
            market_cap=400_000_000_000_000  # 시총 400조
        )

        assert result.s_flow > 0
        assert result.foreign_net == 100_000_000_000
        assert result.institution_net == 50_000_000_000

    def test_calculate_stock_negative(self):
        """음수 수급"""
        calc = FlowCalculator()

        result = calc.calculate_stock(
            stock_code="005930",
            foreign_net=-50_000_000_000,
            institution_net=-30_000_000_000,
            market_cap=100_000_000_000_000
        )

        assert result.s_flow < 0

    def test_calculate_stock_zero_market_cap(self):
        """시총 0 처리"""
        calc = FlowCalculator()

        result = calc.calculate_stock(
            stock_code="000000",
            foreign_net=100,
            institution_net=100,
            market_cap=0
        )

        assert result.s_flow == 0.0

    def test_calculate_sector(self):
        """섹터 수급 계산"""
        calc = FlowCalculator()

        stocks_data = [
            {"stock_code": "005930", "foreign_net": 100e9, "institution_net": 50e9, "market_cap": 400e12},
            {"stock_code": "000660", "foreign_net": 30e9, "institution_net": 20e9, "market_cap": 100e12},
        ]

        result = calc.calculate_sector("반도체", stocks_data)

        assert result.sector_name == "반도체"
        assert len(result.stock_flows) == 2
        assert result.leader_stock == "005930"  # 시총 1위

    def test_leader_distortion_detection(self):
        """대장주 착시 감지"""
        calc = FlowCalculator()

        # 대장주만 강한 순매수, 나머지는 순매도
        stocks_data = [
            {"stock_code": "005930", "foreign_net": 200e9, "institution_net": 100e9, "market_cap": 400e12},
            {"stock_code": "000660", "foreign_net": -50e9, "institution_net": -30e9, "market_cap": 100e12},
            {"stock_code": "000270", "foreign_net": -40e9, "institution_net": -20e9, "market_cap": 50e12},
        ]

        result = calc.calculate_sector("반도체", stocks_data)

        # 전체 S_Flow > 0 이지만 대장주 제외 시 < 0 이면 착시
        assert result.s_flow > 0
        assert result.s_flow_ex_leader < 0
        assert result.is_leader_distorted is True

    def test_classify_flow_level(self):
        """수급 수준 분류"""
        calc = FlowCalculator()

        assert calc.classify_flow_level(1.0) == "HIGH"
        assert calc.classify_flow_level(0.0) == "MEDIUM"
        assert calc.classify_flow_level(-1.0) == "LOW"


class TestBreadthCalculator:
    """S_Breadth 계산기 테스트"""

    def test_calculate_stock_above_ma(self):
        """MA20 위"""
        calc = BreadthCalculator()

        result = calc.calculate_stock(
            stock_code="005930",
            close_price=80000,
            ma20=75000
        )

        assert result.above_ma20 is True
        assert result.ma_gap_pct > 0

    def test_calculate_stock_below_ma(self):
        """MA20 아래"""
        calc = BreadthCalculator()

        result = calc.calculate_stock(
            stock_code="005930",
            close_price=70000,
            ma20=75000
        )

        assert result.above_ma20 is False
        assert result.ma_gap_pct < 0

    def test_calculate_sector(self):
        """섹터 결속력 계산"""
        calc = BreadthCalculator()

        stocks_data = [
            {"stock_code": "001", "close_price": 10000, "ma20": 9000},   # 위
            {"stock_code": "002", "close_price": 10000, "ma20": 9500},   # 위
            {"stock_code": "003", "close_price": 10000, "ma20": 11000},  # 아래
        ]

        result = calc.calculate_sector("테스트섹터", stocks_data)

        assert result.above_count == 2
        assert result.total_count == 3
        assert result.s_breadth == pytest.approx(66.67, rel=0.01)

    def test_breadth_level_strong(self):
        """결속력 STRONG"""
        calc = BreadthCalculator()

        # 모두 MA20 위
        stocks_data = [
            {"stock_code": f"00{i}", "close_price": 10000, "ma20": 9000}
            for i in range(10)
        ]

        result = calc.calculate_sector("강한섹터", stocks_data)

        assert result.s_breadth == 100.0
        assert result.breadth_level == "STRONG"

    def test_breadth_level_weak(self):
        """결속력 WEAK"""
        calc = BreadthCalculator()

        # 모두 MA20 아래
        stocks_data = [
            {"stock_code": f"00{i}", "close_price": 8000, "ma20": 9000}
            for i in range(10)
        ]

        result = calc.calculate_sector("약한섹터", stocks_data)

        assert result.s_breadth == 0.0
        assert result.breadth_level == "WEAK"


class TestTrendCalculator:
    """S_Trend 계산기 테스트"""

    def test_calculate_stock_aligned(self):
        """정배열 종목"""
        calc = TrendCalculator()

        data = {
            "ma5": 100,
            "ma20": 95,
            "ma60": 90,
            "ma120": 85,
            "momentum_20d": 10,
            "rsi": 60
        }

        result = calc.calculate_stock_from_dict("005930", data)

        assert result.is_aligned is True
        assert result.alignment_score == 50  # 완전 정배열
        assert result.s_trend > 50

    def test_calculate_stock_not_aligned(self):
        """역배열 종목"""
        calc = TrendCalculator()

        data = {
            "ma5": 80,
            "ma20": 90,
            "ma60": 95,
            "ma120": 100,
            "momentum_20d": -5,
            "rsi": 40
        }

        result = calc.calculate_stock_from_dict("005930", data)

        assert result.is_aligned is False
        assert result.alignment_score < 50

    def test_overheat_penalty(self):
        """과열 패널티"""
        calc = TrendCalculator()

        # RSI 80 (과열)
        data = {
            "ma5": 100,
            "ma20": 95,
            "ma60": 90,
            "ma120": 85,
            "momentum_20d": 15,
            "rsi": 80  # 과열
        }

        result = calc.calculate_stock_from_dict("005930", data)

        assert result.overheat_penalty > 0
        assert result.rsi == 80

    def test_calculate_from_df(self):
        """DataFrame에서 계산"""
        calc = TrendCalculator()

        # 상승 추세 데이터 생성
        dates = pd.date_range(end="2026-01-31", periods=150)
        prices = np.linspace(50000, 80000, 150)  # 상승
        df = pd.DataFrame({"date": dates, "close": prices})

        result = calc.calculate_stock("005930", df)

        assert result.s_trend >= 0
        assert result.ma5 > result.ma120  # 상승 추세

    def test_calculate_sector(self):
        """섹터 추세 계산"""
        calc = TrendCalculator()

        stocks_data = [
            {"stock_code": "001", "ma5": 100, "ma20": 95, "ma60": 90, "ma120": 85, "momentum_20d": 10, "rsi": 60},
            {"stock_code": "002", "ma5": 100, "ma20": 98, "ma60": 95, "ma120": 92, "momentum_20d": 5, "rsi": 55},
        ]

        result = calc.calculate_sector("테스트", stocks_data)

        assert result.sector_name == "테스트"
        assert result.aligned_ratio > 0
        assert len(result.stock_results) == 2


class TestTierClassifier:
    """Tier 분류기 테스트"""

    def test_tier_1_classification(self):
        """Tier 1: 수급 빈집"""
        classifier = TierClassifier()

        # 수급 높음, 추세 낮음
        result = classifier.classify_simple(
            sector_name="HBM",
            s_flow=1.0,       # HIGH
            s_breadth=50,
            s_trend=25,       # WEAK
            is_leader_distorted=False
        )

        assert result.tier == SectorTier.TIER_1
        assert "선취매" in result.comment

    def test_tier_2_classification(self):
        """Tier 2: 주도 섹터"""
        classifier = TierClassifier()

        # 수급 높음, 추세 높음
        result = classifier.classify_simple(
            sector_name="AI",
            s_flow=1.0,       # HIGH
            s_breadth=80,
            s_trend=70,       # STRONG
            is_leader_distorted=False
        )

        assert result.tier == SectorTier.TIER_2
        assert "주도" in result.comment or "눌림목" in result.comment

    def test_tier_3_classification(self):
        """Tier 3: 가짜 상승"""
        classifier = TierClassifier()

        # 수급 낮음, 추세 높음
        result = classifier.classify_simple(
            sector_name="테마",
            s_flow=-0.5,      # LOW
            s_breadth=60,
            s_trend=70,       # STRONG
            is_leader_distorted=False
        )

        assert result.tier == SectorTier.TIER_3
        assert "주의" in result.comment or "금지" in result.comment

    def test_skip_classification(self):
        """SKIP: 무관심"""
        classifier = TierClassifier()

        # 수급 낮음, 추세 낮음
        result = classifier.classify_simple(
            sector_name="한산한섹터",
            s_flow=-0.2,
            s_breadth=40,
            s_trend=20,
            is_leader_distorted=False
        )

        assert result.tier == SectorTier.SKIP
        assert "관망" in result.comment

    def test_leader_distortion_demotion(self):
        """대장주 착시 시 Tier 3 강등"""
        classifier = TierClassifier()

        # 원래 Tier 1이 될 조건이지만 착시로 강등
        result = classifier.classify_simple(
            sector_name="착시섹터",
            s_flow=1.0,
            s_breadth=60,
            s_trend=25,
            is_leader_distorted=True  # 착시!
        )

        assert result.tier == SectorTier.TIER_3
        assert "착시" in str(result.warnings)

    def test_rank_sectors(self):
        """섹터 순위 정렬"""
        classifier = TierClassifier()

        results = [
            classifier.classify_simple("섹터A", s_flow=0.8, s_breadth=70, s_trend=65, is_leader_distorted=False),  # Tier 2
            classifier.classify_simple("섹터B", s_flow=1.2, s_breadth=50, s_trend=25, is_leader_distorted=False),  # Tier 1
            classifier.classify_simple("섹터C", s_flow=0.5, s_breadth=60, s_trend=55, is_leader_distorted=False),  # Tier 2
        ]

        ranked = classifier.rank_sectors(results)

        # Tier 1이 먼저, 그 다음 Tier 2
        assert ranked[0].sector_name == "섹터B"  # Tier 1
        assert ranked[0].tier == SectorTier.TIER_1

    def test_get_actionable_sectors(self):
        """투자 가능 섹터 분류"""
        classifier = TierClassifier()

        results = [
            classifier.classify_simple("섹터1", s_flow=1.0, s_breadth=50, s_trend=25, is_leader_distorted=False),  # Tier 1
            classifier.classify_simple("섹터2", s_flow=0.8, s_breadth=70, s_trend=65, is_leader_distorted=False),  # Tier 2
            classifier.classify_simple("섹터3", s_flow=-0.5, s_breadth=60, s_trend=70, is_leader_distorted=False), # Tier 3
            classifier.classify_simple("섹터4", s_flow=-0.2, s_breadth=40, s_trend=20, is_leader_distorted=False), # Skip
        ]

        actionable = classifier.get_actionable_sectors(results)

        assert len(actionable["선취매"]) == 1
        assert len(actionable["눌림목대기"]) == 1
        assert len(actionable["주의"]) == 1
        assert len(actionable["관망"]) == 1


class TestIntegration:
    """통합 테스트"""

    def test_full_analysis_pipeline(self):
        """전체 분석 파이프라인"""
        classifier = TierClassifier()

        # 데이터 준비
        flow_data = [
            {"stock_code": "005930", "foreign_net": 100e9, "institution_net": 50e9, "market_cap": 400e12},
            {"stock_code": "000660", "foreign_net": 30e9, "institution_net": 20e9, "market_cap": 100e12},
        ]

        breadth_data = [
            {"stock_code": "005930", "close_price": 80000, "ma20": 75000},
            {"stock_code": "000660", "close_price": 150000, "ma20": 145000},
        ]

        trend_data = [
            {"stock_code": "005930", "ma5": 80000, "ma20": 78000, "ma60": 75000, "ma120": 72000, "momentum_20d": 8, "rsi": 55},
            {"stock_code": "000660", "ma5": 150000, "ma20": 148000, "ma60": 145000, "ma120": 140000, "momentum_20d": 10, "rsi": 60},
        ]

        result = classifier.classify(
            sector_name="반도체",
            flow_data=flow_data,
            breadth_data=breadth_data,
            trend_data=trend_data
        )

        assert result.sector_name == "반도체"
        assert result.tier in [SectorTier.TIER_1, SectorTier.TIER_2, SectorTier.TIER_3, SectorTier.SKIP]
        assert result.s_flow > 0
        assert result.s_breadth == 100.0  # 모두 MA20 위


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
