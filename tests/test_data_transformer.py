"""
데이터 변환 모듈 테스트
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime

from src.processing.data_transformer import (
    DataTransformer,
    StockFinancials,
    StockSupplyDemand,
)
from src.ingest.dart_client import FinancialData


class TestStockFinancials:
    """StockFinancials 데이터클래스 테스트"""

    def test_create_financials(self):
        """재무 데이터 생성"""
        fin = StockFinancials(
            stock_code="005930",
            stock_name="삼성전자",
            operating_profit_4q=50_000_000_000_000,
            debt_ratio=50.0,
            pbr=1.5,
            avg_trading_value=500_000_000_000,
        )

        assert fin.stock_code == "005930"
        assert fin.operating_profit_4q == 50_000_000_000_000
        assert fin.pbr == 1.5

    def test_to_filter_dict(self):
        """필터용 dict 변환"""
        fin = StockFinancials(
            stock_code="005930",
            stock_name="삼성전자",
            operating_profit_4q=100,
            debt_ratio=150.0,
            pbr=2.0,
            avg_trading_value=10_000_000_000,
            capital_impairment=0.0,
            current_ratio=200.0,
            rd_ratio=8.5,
        )

        d = fin.to_filter_dict()

        assert d["stock_code"] == "005930"
        assert d["operating_profit_4q"] == 100
        assert d["debt_ratio"] == 150.0
        assert d["rd_ratio"] == 8.5


class TestStockSupplyDemand:
    """StockSupplyDemand 데이터클래스 테스트"""

    def test_create_supply_demand(self):
        """수급 데이터 생성"""
        supply = StockSupplyDemand(
            stock_code="005930",
            foreign_net=1_000_000,
            institution_net=500_000,
            individual_net=-1_500_000,
            market_cap=300_000_000_000_000,
        )

        assert supply.stock_code == "005930"
        assert supply.foreign_net == 1_000_000

    def test_get_s_flow_inputs(self):
        """S_Flow 입력값 반환"""
        supply = StockSupplyDemand(
            stock_code="005930",
            foreign_net_amount=50_000_000_000,
            institution_net_amount=25_000_000_000,
            market_cap=300_000_000_000_000,
        )

        inputs = supply.get_s_flow_inputs()

        assert inputs["foreign_net"] == 50_000_000_000
        assert inputs["institution_net"] == 25_000_000_000
        assert inputs["market_cap"] == 300_000_000_000_000


class TestFinancialDataEnhancements:
    """FinancialData 확장 필드 테스트"""

    def test_capital_impairment_calculation(self):
        """자본잠식률 계산"""
        # 정상 (자본 > 자본금)
        data = FinancialData(
            stock_code="000001",
            stock_name="테스트",
            report_type="11011",
            year=2025,
            quarter=4,
            total_equity=100_000_000,
            capital_stock=50_000_000,
        )
        assert data.capital_impairment == 0.0

        # 부분 잠식 (총자본 < 자본금)
        data2 = FinancialData(
            stock_code="000002",
            stock_name="테스트2",
            report_type="11011",
            year=2025,
            quarter=4,
            total_equity=30_000_000,
            capital_stock=50_000_000,
        )
        assert data2.capital_impairment == 40.0  # (50-30)/50 * 100

    def test_rd_ratio_calculation(self):
        """R&D 비중 계산"""
        data = FinancialData(
            stock_code="000001",
            stock_name="테스트",
            report_type="11011",
            year=2025,
            quarter=4,
            revenue=100_000_000,
            rd_expense=10_000_000,
        )

        assert data.rd_ratio == 10.0  # 10%

    def test_rd_ratio_zero_revenue(self):
        """매출 0일 때 R&D 비중"""
        data = FinancialData(
            stock_code="000001",
            stock_name="테스트",
            report_type="11011",
            year=2025,
            quarter=4,
            revenue=0,
            rd_expense=10_000_000,
        )

        assert data.rd_ratio == 0.0


class TestDataTransformer:
    """DataTransformer 테스트"""

    @pytest.fixture
    def transformer(self):
        """Transformer with mocked dependencies"""
        with patch("src.processing.data_transformer.PriceDataFetcher") as mock_pf, \
             patch("src.processing.data_transformer.DartApiClient") as mock_dart:
            return DataTransformer()

    def test_init(self, transformer):
        """초기화"""
        assert transformer is not None

    @patch.object(DataTransformer, "_refresh_cache_if_needed")
    @patch.object(DataTransformer, "get_stock_financials")
    def test_prepare_filter_data(self, mock_get_fin, mock_refresh, transformer):
        """필터 데이터 일괄 준비"""
        mock_fin = StockFinancials(
            stock_code="005930",
            operating_profit_4q=100,
            debt_ratio=50.0,
        )
        mock_get_fin.return_value = mock_fin

        result = transformer.prepare_filter_data(["005930", "000660"], 2025)

        assert len(result) == 2
        assert mock_get_fin.call_count == 2

    def test_transform_filter_result_to_scorer_input(self, transformer):
        """FilterResult → Scorer 입력 변환"""
        filter_result = {
            "operating_profit_4q": 100_000_000,
            "debt_ratio": 80.0,
            "current_ratio": 150.0,
            "capital_impairment": 0.0,
            "rd_ratio": 5.0,
            "pbr": 1.2,
        }

        scorer_input = transformer.transform_filter_result_to_scorer_input(filter_result)

        assert scorer_input["financial"]["operating_profit"] == 100_000_000
        assert scorer_input["financial"]["debt_ratio"] == 80.0
        assert scorer_input["financial"]["rd_ratio"] == 5.0
        assert "technical" in scorer_input


class TestDataTransformerIntegration:
    """통합 테스트 (실제 네트워크 호출)"""

    @pytest.mark.skip(reason="실제 네트워크 호출 - 수동 실행")
    def test_real_get_stock_financials(self):
        """실제 재무 데이터 조회"""
        transformer = DataTransformer()
        financials = transformer.get_stock_financials("005930", 2024)

        print(f"종목: {financials.stock_name}")
        print(f"PBR: {financials.pbr}")
        print(f"부채비율: {financials.debt_ratio}")
        print(f"DART 데이터: {financials.has_dart_data}")

    @pytest.mark.skip(reason="실제 네트워크 호출 - 수동 실행")
    def test_real_get_supply_demand(self):
        """실제 수급 데이터 조회"""
        transformer = DataTransformer()
        supply = transformer.get_supply_demand("005930", days=20)

        print(f"외국인 순매수: {supply.foreign_net:,.0f}")
        print(f"기관 순매수: {supply.institution_net:,.0f}")
        print(f"시가총액: {supply.market_cap:,.0f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
