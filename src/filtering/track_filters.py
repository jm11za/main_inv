"""
Track A/B 필터

Type A (실적 기반): Hard Filter - 엄격한 재무 조건
Type B (성장 기반): Soft Filter - 생존력 위주 검증
"""
from dataclasses import dataclass
from typing import Any

from src.core.interfaces import StockFilter, FilterResult, TrackType
from src.core.config import get_config
from src.core.logger import get_logger


@dataclass
class FilterConditions:
    """필터 조건 설정"""
    # Track A (Hard Filter)
    min_operating_profit_4q: float = 0  # 4분기 합산 영업이익 > 0
    max_debt_ratio: float = 200  # 부채비율 < 200%
    max_pbr: float = 3.0  # PBR < 3.0 (과열 제외)
    min_trading_value_a: float = 1_000_000_000  # 일평균 거래대금 10억

    # Track B (Soft Filter)
    max_capital_impairment: float = 50  # 자본잠식률 < 50%
    min_current_ratio: float = 100  # 유동비율 > 100%
    min_rd_ratio: float = 5  # R&D 비중 > 5% (가산점)
    min_trading_value_b: float = 500_000_000  # 일평균 거래대금 5억


class TrackAFilter(StockFilter):
    """
    Track A 필터 (Hard Filter)

    실적 기반 섹터용 엄격한 필터
    - 4분기 합산 영업이익 > 0 (필수)
    - 부채비율 < 200%
    - PBR < 3.0 (과열 제외)
    - 일평균 거래대금 > 10억
    """

    def __init__(self, conditions: FilterConditions | None = None):
        self.logger = get_logger(self.__class__.__name__)
        self.conditions = conditions or self._load_conditions()

    def _load_conditions(self) -> FilterConditions:
        """설정에서 조건 로드"""
        config = get_config()
        filter_config = config.get_section("pipeline.filter.track_a") or {}

        return FilterConditions(
            min_operating_profit_4q=filter_config.get("min_operating_profit_4q", 0),
            max_debt_ratio=filter_config.get("max_debt_ratio", 200),
            max_pbr=filter_config.get("max_pbr", 3.0),
            min_trading_value_a=filter_config.get("min_trading_value", 1_000_000_000),
        )

    def get_filter_name(self) -> str:
        return "TrackAFilter (Hard)"

    def get_conditions(self) -> dict:
        return {
            "영업이익 4Q 합산": f"> {self.conditions.min_operating_profit_4q}",
            "부채비율": f"< {self.conditions.max_debt_ratio}%",
            "PBR": f"< {self.conditions.max_pbr}",
            "일평균 거래대금": f"> {self.conditions.min_trading_value_a / 1e8:.0f}억",
        }

    def apply(self, stock: dict) -> FilterResult:
        """
        단일 종목 필터 적용

        Args:
            stock: 종목 데이터 (dict)
                - stock_code: 종목코드
                - operating_profit_4q: 4분기 합산 영업이익
                - debt_ratio: 부채비율 (%)
                - pbr: PBR
                - avg_trading_value: 일평균 거래대금

        Returns:
            FilterResult
        """
        stock_code = stock.get("stock_code", "")
        reasons = []
        metrics = {}

        # 1. 영업이익 체크 (필수)
        op_profit = stock.get("operating_profit_4q", 0)
        metrics["operating_profit_4q"] = op_profit
        if op_profit <= self.conditions.min_operating_profit_4q:
            reasons.append(f"영업이익 적자 ({op_profit:,.0f})")

        # 2. 부채비율 체크
        debt_ratio = stock.get("debt_ratio", 0)
        metrics["debt_ratio"] = debt_ratio
        if debt_ratio > self.conditions.max_debt_ratio:
            reasons.append(f"부채비율 초과 ({debt_ratio:.1f}% > {self.conditions.max_debt_ratio}%)")

        # 3. PBR 체크
        pbr = stock.get("pbr", 0)
        metrics["pbr"] = pbr
        if pbr > self.conditions.max_pbr:
            reasons.append(f"PBR 과열 ({pbr:.2f} > {self.conditions.max_pbr})")

        # 4. 거래대금 체크
        trading_value = stock.get("avg_trading_value", 0)
        metrics["avg_trading_value"] = trading_value
        if trading_value < self.conditions.min_trading_value_a:
            reasons.append(f"거래대금 부족 ({trading_value/1e8:.1f}억 < {self.conditions.min_trading_value_a/1e8:.0f}억)")

        passed = len(reasons) == 0
        reason = "모든 조건 충족" if passed else " / ".join(reasons)

        return FilterResult(
            passed=passed,
            stock_code=stock_code,
            reason=reason,
            metrics=metrics,
        )

    def apply_batch(self, stocks: list[dict]) -> list[FilterResult]:
        """복수 종목 일괄 필터"""
        return [self.apply(stock) for stock in stocks]


class TrackBFilter(StockFilter):
    """
    Track B 필터 (Soft Filter)

    성장 기반 섹터용 완화된 필터
    - 적자 허용
    - 자본잠식률 < 50% (필수)
    - 유동비율 > 100% (필수)
    - R&D 비중 > 5% (가산점)
    - 일평균 거래대금 > 5억
    """

    def __init__(self, conditions: FilterConditions | None = None):
        self.logger = get_logger(self.__class__.__name__)
        self.conditions = conditions or self._load_conditions()

    def _load_conditions(self) -> FilterConditions:
        """설정에서 조건 로드"""
        config = get_config()
        filter_config = config.get_section("pipeline.filter.track_b") or {}

        return FilterConditions(
            max_capital_impairment=filter_config.get("max_capital_impairment", 50),
            min_current_ratio=filter_config.get("min_current_ratio", 100),
            min_rd_ratio=filter_config.get("min_rd_ratio", 5),
            min_trading_value_b=filter_config.get("min_trading_value", 500_000_000),
        )

    def get_filter_name(self) -> str:
        return "TrackBFilter (Soft)"

    def get_conditions(self) -> dict:
        return {
            "자본잠식률": f"< {self.conditions.max_capital_impairment}%",
            "유동비율": f"> {self.conditions.min_current_ratio}%",
            "R&D 비중": f"> {self.conditions.min_rd_ratio}% (가산점)",
            "일평균 거래대금": f"> {self.conditions.min_trading_value_b / 1e8:.0f}억",
        }

    def apply(self, stock: dict) -> FilterResult:
        """
        단일 종목 필터 적용

        Args:
            stock: 종목 데이터 (dict)
                - stock_code: 종목코드
                - capital_impairment: 자본잠식률 (%)
                - current_ratio: 유동비율 (%)
                - rd_ratio: R&D 비중 (%)
                - avg_trading_value: 일평균 거래대금

        Returns:
            FilterResult
        """
        stock_code = stock.get("stock_code", "")
        reasons = []
        metrics = {}

        # 1. 자본잠식률 체크 (필수)
        capital_impairment = stock.get("capital_impairment", 0)
        metrics["capital_impairment"] = capital_impairment
        if capital_impairment > self.conditions.max_capital_impairment:
            reasons.append(f"자본잠식 위험 ({capital_impairment:.1f}% > {self.conditions.max_capital_impairment}%)")

        # 2. 유동비율 체크 (필수)
        current_ratio = stock.get("current_ratio", 0)
        metrics["current_ratio"] = current_ratio
        if current_ratio < self.conditions.min_current_ratio:
            reasons.append(f"유동성 부족 ({current_ratio:.1f}% < {self.conditions.min_current_ratio}%)")

        # 3. R&D 비중 (가산점, 탈락 조건 아님)
        rd_ratio = stock.get("rd_ratio", 0)
        metrics["rd_ratio"] = rd_ratio
        metrics["rd_bonus"] = rd_ratio >= self.conditions.min_rd_ratio

        # 4. 거래대금 체크
        trading_value = stock.get("avg_trading_value", 0)
        metrics["avg_trading_value"] = trading_value
        if trading_value < self.conditions.min_trading_value_b:
            reasons.append(f"거래대금 부족 ({trading_value/1e8:.1f}억 < {self.conditions.min_trading_value_b/1e8:.0f}억)")

        passed = len(reasons) == 0
        reason = "생존력 검증 통과" if passed else " / ".join(reasons)

        # R&D 가산점 표시
        if passed and metrics["rd_bonus"]:
            reason += f" (R&D {rd_ratio:.1f}% 가산점)"

        return FilterResult(
            passed=passed,
            stock_code=stock_code,
            reason=reason,
            metrics=metrics,
        )

    def apply_batch(self, stocks: list[dict]) -> list[FilterResult]:
        """복수 종목 일괄 필터"""
        return [self.apply(stock) for stock in stocks]


def get_filter_for_track(track_type: TrackType) -> StockFilter:
    """Track 타입에 맞는 필터 반환"""
    if track_type == TrackType.TRACK_A:
        return TrackAFilter()
    else:
        return TrackBFilter()
