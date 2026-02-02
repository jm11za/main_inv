"""
Layer 3.5: Filtering 모듈 테스트
"""
import pytest
from unittest.mock import Mock, patch

from src.core.interfaces import SectorType, TrackType, FilterResult
from src.filtering.sector_classifier import SectorClassifier
from src.filtering.track_filters import (
    TrackAFilter,
    TrackBFilter,
    FilterConditions,
    get_filter_for_track,
)
from src.filtering.filter_router import FilterRouter, FilteredStock


class TestSectorClassifier:
    """섹터 분류기 테스트"""

    def test_classify_type_b_keyword(self):
        """Type B 키워드 분류"""
        classifier = SectorClassifier(use_llm=False)

        # 바이오 -> Type B
        result = classifier.classify("바이오")
        assert result == SectorType.TYPE_B

        # AI -> Type B
        result = classifier.classify("AI반도체")
        assert result == SectorType.TYPE_B

    def test_classify_type_a_keyword(self):
        """Type A 키워드 분류"""
        classifier = SectorClassifier(use_llm=False)

        # 자동차 -> Type A
        result = classifier.classify("자동차")
        assert result == SectorType.TYPE_A

        # 은행 -> Type A
        result = classifier.classify("은행")
        assert result == SectorType.TYPE_A

    def test_classify_fallback(self):
        """분류 불가 시 폴백"""
        classifier = SectorClassifier(use_llm=False)

        # 알 수 없는 섹터 -> 폴백 (기본 A)
        result = classifier.classify("알수없는섹터XYZ")
        assert result in [SectorType.TYPE_A, SectorType.TYPE_B]

    def test_classify_batch(self):
        """배치 분류"""
        classifier = SectorClassifier(use_llm=False)

        results = classifier.classify_batch(["바이오", "자동차", "AI"])

        assert results["바이오"] == SectorType.TYPE_B
        assert results["자동차"] == SectorType.TYPE_A
        assert results["AI"] == SectorType.TYPE_B

    def test_type_description(self):
        """타입 설명 반환"""
        classifier = SectorClassifier(use_llm=False)

        desc_a = classifier.get_type_description(SectorType.TYPE_A)
        assert desc_a["type"] == "A"
        assert desc_a["name"] == "실적 기반"

        desc_b = classifier.get_type_description(SectorType.TYPE_B)
        assert desc_b["type"] == "B"
        assert desc_b["name"] == "성장 기반"


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


class TestFilterRouter:
    """필터 라우터 테스트"""

    def test_filter_stock_type_a(self):
        """Type A 종목 필터링"""
        router = FilterRouter(use_llm=False)

        stock = {
            "stock_code": "005930",
            "name": "삼성전자",
            "operating_profit_4q": 50_000_000_000,
            "debt_ratio": 50,
            "pbr": 1.5,
            "avg_trading_value": 5_000_000_000,
        }

        result = router.filter_stock(stock, sector_name="자동차")

        assert isinstance(result, FilteredStock)
        assert result.sector_type == SectorType.TYPE_A
        assert result.track_type == TrackType.TRACK_A
        assert result.filter_passed is True

    def test_filter_stock_type_b(self):
        """Type B 종목 필터링"""
        router = FilterRouter(use_llm=False)

        stock = {
            "stock_code": "000000",
            "name": "바이오",
            "capital_impairment": 0,
            "current_ratio": 150,
            "rd_ratio": 10,
            "avg_trading_value": 1_000_000_000,
        }

        result = router.filter_stock(stock, sector_name="바이오")

        assert result.sector_type == SectorType.TYPE_B
        assert result.track_type == TrackType.TRACK_B
        assert result.filter_passed is True

    def test_filter_stocks_by_sector(self):
        """섹터별 일괄 필터링"""
        router = FilterRouter(use_llm=False)

        stocks = [
            {
                "stock_code": "001",
                "name": "종목1",
                "operating_profit_4q": 10_000_000_000,
                "debt_ratio": 100,
                "pbr": 1.5,
                "avg_trading_value": 2_000_000_000,
            },
            {
                "stock_code": "002",
                "name": "종목2",
                "operating_profit_4q": -5_000_000_000,  # 적자
                "debt_ratio": 100,
                "pbr": 1.5,
                "avg_trading_value": 2_000_000_000,
            },
        ]

        results = router.filter_stocks_by_sector(stocks, sector_name="자동차")

        assert len(results) == 2
        assert results[0].filter_passed is True
        assert results[1].filter_passed is False

    def test_get_passed_stocks(self):
        """통과 종목만 필터"""
        router = FilterRouter(use_llm=False)

        filtered = [
            FilteredStock("001", "종목1", "섹터", SectorType.TYPE_A, TrackType.TRACK_A, True, "OK", {}),
            FilteredStock("002", "종목2", "섹터", SectorType.TYPE_A, TrackType.TRACK_A, False, "FAIL", {}),
        ]

        passed = router.get_passed_stocks(filtered)
        assert len(passed) == 1
        assert passed[0].stock_code == "001"

    def test_get_summary(self):
        """요약 통계"""
        router = FilterRouter(use_llm=False)

        filtered = [
            FilteredStock("001", "종목1", "섹터", SectorType.TYPE_A, TrackType.TRACK_A, True, "OK", {}),
            FilteredStock("002", "종목2", "섹터", SectorType.TYPE_B, TrackType.TRACK_B, True, "OK", {}),
            FilteredStock("003", "종목3", "섹터", SectorType.TYPE_A, TrackType.TRACK_A, False, "FAIL", {}),
        ]

        summary = router.get_summary(filtered)

        assert summary["total"] == 3
        assert summary["passed"] == 2
        assert summary["failed"] == 1
        assert summary["track_a_count"] == 2
        assert summary["track_b_count"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
