"""
Stage 4: StockFilter 모듈 테스트
- Track A (Hard Filter): 영업이익, 부채비율, PBR, 거래대금
- Track B (Soft Filter): 자본잠식률, 유동비율, 거래대금(완화)
"""
import pytest

from src.core.interfaces import TrackType, SectorType
from src.stock.filter import StockFilter, StockFilterResult


def _make_stock_a(**overrides) -> dict:
    """Track A 기본 통과 종목 데이터"""
    base = {
        "stock_code": "005930",
        "stock_name": "삼성전자",
        "operating_profit_4q": 50_000_000_000,
        "debt_ratio": 50,
        "pbr": 1.5,
        "avg_trading_value": 5_000_000_000,
    }
    base.update(overrides)
    return base


def _make_stock_b(**overrides) -> dict:
    """Track B 기본 통과 종목 데이터"""
    base = {
        "stock_code": "068270",
        "stock_name": "셀트리온",
        "capital_impairment": 0,
        "current_ratio": 150,
        "avg_trading_value": 1_000_000_000,
    }
    base.update(overrides)
    return base


class TestStockFilterTrackA:
    """Track A (Hard Filter) 테스트"""

    def test_pass_all_conditions(self):
        """모든 조건 충족 시 통과"""
        f = StockFilter()
        result = f.apply(_make_stock_a(), TrackType.TRACK_A)
        assert result.passed is True
        assert result.track_type == TrackType.TRACK_A
        assert "모든 조건 충족" in result.reason

    def test_fail_operating_profit(self):
        """영업이익 적자 시 실패"""
        f = StockFilter()
        result = f.apply(
            _make_stock_a(operating_profit_4q=-10_000_000_000),
            TrackType.TRACK_A,
        )
        assert result.passed is False
        assert "적자" in result.reason

    def test_fail_debt_ratio(self):
        """부채비율 >= 200% 시 실패"""
        f = StockFilter()
        result = f.apply(
            _make_stock_a(debt_ratio=250),
            TrackType.TRACK_A,
        )
        assert result.passed is False
        assert "부채비율" in result.reason

    def test_fail_pbr(self):
        """PBR >= 3.0 시 실패"""
        f = StockFilter()
        result = f.apply(
            _make_stock_a(pbr=5.0),
            TrackType.TRACK_A,
        )
        assert result.passed is False
        assert "PBR" in result.reason

    def test_fail_trading_value(self):
        """거래대금 < 10억 시 실패"""
        f = StockFilter()
        result = f.apply(
            _make_stock_a(avg_trading_value=100_000_000),
            TrackType.TRACK_A,
        )
        assert result.passed is False
        assert "거래대금" in result.reason

    def test_result_has_metrics(self):
        """결과에 metrics 포함"""
        f = StockFilter()
        result = f.apply(_make_stock_a(), TrackType.TRACK_A)
        assert "operating_profit_4q" in result.metrics
        assert "debt_ratio" in result.metrics
        assert "pbr" in result.metrics
        assert "avg_trading_value" in result.metrics

    def test_result_has_conditions(self):
        """결과에 개별 조건 결과 포함"""
        f = StockFilter()
        result = f.apply(_make_stock_a(), TrackType.TRACK_A)
        assert len(result.conditions) == 4

    def test_to_dict(self):
        """to_dict 직렬화"""
        f = StockFilter()
        result = f.apply(_make_stock_a(), TrackType.TRACK_A)
        d = result.to_dict()
        assert d["stock_code"] == "005930"
        assert d["passed"] is True
        assert "track_type" in d

    def test_missing_field_treated_as_zero(self):
        """누락 필드는 0으로 처리"""
        f = StockFilter()
        stock = {"stock_code": "000000", "stock_name": "테스트"}
        result = f.apply(stock, TrackType.TRACK_A)
        assert result.passed is False


class TestStockFilterTrackB:
    """Track B (Soft Filter) 테스트"""

    def test_pass_all_conditions(self):
        """모든 조건 충족 시 통과"""
        f = StockFilter()
        result = f.apply(_make_stock_b(), TrackType.TRACK_B)
        assert result.passed is True
        assert result.track_type == TrackType.TRACK_B

    def test_fail_capital_impairment(self):
        """자본잠식률 >= 50% 시 실패"""
        f = StockFilter()
        result = f.apply(
            _make_stock_b(capital_impairment=60),
            TrackType.TRACK_B,
        )
        assert result.passed is False
        assert "자본잠식" in result.reason

    def test_fail_current_ratio(self):
        """유동비율 < 100% 시 실패"""
        f = StockFilter()
        result = f.apply(
            _make_stock_b(current_ratio=80),
            TrackType.TRACK_B,
        )
        assert result.passed is False
        assert "유동비율" in result.reason

    def test_fail_trading_value_soft(self):
        """거래대금 < 5억 시 실패 (완화 기준)"""
        f = StockFilter()
        result = f.apply(
            _make_stock_b(avg_trading_value=100_000_000),
            TrackType.TRACK_B,
        )
        assert result.passed is False
        assert "거래대금" in result.reason

    def test_b_conditions_count(self):
        """Track B는 3개 조건"""
        f = StockFilter()
        result = f.apply(_make_stock_b(), TrackType.TRACK_B)
        assert len(result.conditions) == 3


class TestStockFilterBatch:
    """배치 필터 테스트"""

    def test_apply_batch_with_track_type(self):
        """track_type 지정 배치"""
        f = StockFilter()
        stocks = [_make_stock_a(), _make_stock_a(operating_profit_4q=-1)]
        results = f.apply_batch(stocks, track_type=TrackType.TRACK_A)
        assert len(results) == 2
        assert results[0].passed is True
        assert results[1].passed is False

    def test_apply_batch_with_sector_type_b(self):
        """SectorType.TYPE_B -> TrackType.TRACK_B 자동 매핑"""
        f = StockFilter()
        stocks = [_make_stock_b()]
        results = f.apply_batch(stocks, sector_type=SectorType.TYPE_B)
        assert results[0].track_type == TrackType.TRACK_B

    def test_apply_batch_default_track_a(self):
        """기본 트랙은 TRACK_A"""
        f = StockFilter()
        stocks = [_make_stock_a()]
        results = f.apply_batch(stocks)
        assert results[0].track_type == TrackType.TRACK_A

    def test_get_passed_and_failed(self):
        """통과/탈락 분리"""
        f = StockFilter()
        stocks = [
            _make_stock_a(),
            _make_stock_a(operating_profit_4q=-1),
            _make_stock_a(debt_ratio=300),
        ]
        results = f.apply_batch(stocks, track_type=TrackType.TRACK_A)
        assert len(f.get_passed(results)) == 1
        assert len(f.get_failed(results)) == 2

    def test_summarize(self):
        """요약 통계"""
        f = StockFilter()
        stocks = [_make_stock_a(), _make_stock_a(pbr=10)]
        results = f.apply_batch(stocks, track_type=TrackType.TRACK_A)
        summary = f.summarize(results)
        assert summary["total"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert "pass_rate" in summary

    def test_get_conditions(self):
        """트랙별 조건 조회"""
        f = StockFilter()
        conds_a = f.get_conditions(TrackType.TRACK_A)
        conds_b = f.get_conditions(TrackType.TRACK_B)
        assert "operating_profit_4q" in conds_a
        assert "capital_impairment" in conds_b
        assert "operating_profit_4q" not in conds_b


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
