"""
Layer 3.5: Filtering 모듈 테스트 (Track A/B 필터)
"""
import pytest

from src.core.interfaces import TrackType
from src.filtering.track_filters import (
    TrackAFilter,
    TrackBFilter,
    FilterConditions,
    get_filter_for_track,
)


class TestTrackAFilter:
    """Track A 필터 (Hard Filter) 테스트"""

    def test_pass_all_conditions(self):
        """모든 조건 충족 시 통과"""
        filter_obj = TrackAFilter()

        stock = {
            "stock_code": "005930",
            "name": "삼성전자",
            "operating_profit_4q": 50_000_000_000,  # 500억 흑자
            "debt_ratio": 50,  # 부채비율 50%
            "pbr": 1.5,  # PBR 1.5
            "avg_trading_value": 5_000_000_000,  # 거래대금 50억
        }

        result = filter_obj.apply(stock)
        assert result.passed is True
        assert "모든 조건 충족" in result.reason

    def test_fail_operating_profit(self):
        """영업이익 적자 시 실패"""
        filter_obj = TrackAFilter()

        stock = {
            "stock_code": "000000",
            "operating_profit_4q": -10_000_000_000,  # 적자
            "debt_ratio": 50,
            "pbr": 1.5,
            "avg_trading_value": 5_000_000_000,
        }

        result = filter_obj.apply(stock)
        assert result.passed is False
        assert "영업이익 적자" in result.reason

    def test_fail_debt_ratio(self):
        """부채비율 초과 시 실패"""
        filter_obj = TrackAFilter()

        stock = {
            "stock_code": "000000",
            "operating_profit_4q": 10_000_000_000,
            "debt_ratio": 250,  # 200% 초과
            "pbr": 1.5,
            "avg_trading_value": 5_000_000_000,
        }

        result = filter_obj.apply(stock)
        assert result.passed is False
        assert "부채비율 초과" in result.reason

    def test_fail_pbr(self):
        """PBR 과열 시 실패"""
        filter_obj = TrackAFilter()

        stock = {
            "stock_code": "000000",
            "operating_profit_4q": 10_000_000_000,
            "debt_ratio": 100,
            "pbr": 5.0,  # 3.0 초과
            "avg_trading_value": 5_000_000_000,
        }

        result = filter_obj.apply(stock)
        assert result.passed is False
        assert "PBR 과열" in result.reason

    def test_fail_trading_value(self):
        """거래대금 부족 시 실패"""
        filter_obj = TrackAFilter()

        stock = {
            "stock_code": "000000",
            "operating_profit_4q": 10_000_000_000,
            "debt_ratio": 100,
            "pbr": 1.5,
            "avg_trading_value": 100_000_000,  # 1억 (10억 미만)
        }

        result = filter_obj.apply(stock)
        assert result.passed is False
        assert "거래대금 부족" in result.reason

    def test_get_filter_name(self):
        """필터 이름 반환"""
        filter_obj = TrackAFilter()
        assert "Hard" in filter_obj.get_filter_name()

    def test_get_conditions(self):
        """필터 조건 반환"""
        filter_obj = TrackAFilter()
        conditions = filter_obj.get_conditions()

        assert "영업이익" in str(conditions)
        assert "부채비율" in str(conditions)


class TestTrackBFilter:
    """Track B 필터 (Soft Filter) 테스트"""

    def test_pass_all_conditions(self):
        """모든 조건 충족 시 통과"""
        filter_obj = TrackBFilter()

        stock = {
            "stock_code": "000000",
            "name": "바이오종목",
            "capital_impairment": 0,  # 자본잠식 없음
            "current_ratio": 150,  # 유동비율 150%
            "rd_ratio": 10,  # R&D 10%
            "avg_trading_value": 1_000_000_000,  # 10억
        }

        result = filter_obj.apply(stock)
        assert result.passed is True
        assert "생존력 검증 통과" in result.reason

    def test_pass_with_rd_bonus(self):
        """R&D 가산점 표시"""
        filter_obj = TrackBFilter()

        stock = {
            "stock_code": "000000",
            "capital_impairment": 0,
            "current_ratio": 150,
            "rd_ratio": 15,  # R&D 15% (5% 이상)
            "avg_trading_value": 1_000_000_000,
        }

        result = filter_obj.apply(stock)
        assert result.passed is True
        assert "가산점" in result.reason

    def test_fail_capital_impairment(self):
        """자본잠식 위험 시 실패"""
        filter_obj = TrackBFilter()

        stock = {
            "stock_code": "000000",
            "capital_impairment": 60,  # 50% 초과
            "current_ratio": 150,
            "rd_ratio": 10,
            "avg_trading_value": 1_000_000_000,
        }

        result = filter_obj.apply(stock)
        assert result.passed is False
        assert "자본잠식 위험" in result.reason

    def test_fail_current_ratio(self):
        """유동성 부족 시 실패"""
        filter_obj = TrackBFilter()

        stock = {
            "stock_code": "000000",
            "capital_impairment": 0,
            "current_ratio": 80,  # 100% 미만
            "rd_ratio": 10,
            "avg_trading_value": 1_000_000_000,
        }

        result = filter_obj.apply(stock)
        assert result.passed is False
        assert "유동성 부족" in result.reason

    def test_fail_trading_value(self):
        """거래대금 부족 시 실패"""
        filter_obj = TrackBFilter()

        stock = {
            "stock_code": "000000",
            "capital_impairment": 0,
            "current_ratio": 150,
            "rd_ratio": 10,
            "avg_trading_value": 100_000_000,  # 1억 (5억 미만)
        }

        result = filter_obj.apply(stock)
        assert result.passed is False
        assert "거래대금 부족" in result.reason

    def test_get_filter_name(self):
        """필터 이름 반환"""
        filter_obj = TrackBFilter()
        assert "Soft" in filter_obj.get_filter_name()


class TestGetFilterForTrack:
    """트랙별 필터 팩토리 테스트"""

    def test_track_a(self):
        """Track A -> TrackAFilter"""
        filter_obj = get_filter_for_track(TrackType.TRACK_A)
        assert isinstance(filter_obj, TrackAFilter)

    def test_track_b(self):
        """Track B -> TrackBFilter"""
        filter_obj = get_filter_for_track(TrackType.TRACK_B)
        assert isinstance(filter_obj, TrackBFilter)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
