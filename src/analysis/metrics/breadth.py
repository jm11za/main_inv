"""
S_Breadth (내부 결속력) 계산

"섹터 내 종목들이 함께 가고 있는가?"

공식: S_Breadth = (종가 > MA20 종목 수) / 전체 종목 수 × 100
- 대장주만 가고 나머지는 빠지면 → 결속력 낮음 (가짜 상승)
- 70% 이상이 MA20 위면 → 진짜 섹터 상승
"""
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.core.logger import get_logger
from src.core.config import get_config


@dataclass
class BreadthResult:
    """종목 결속력 결과"""
    stock_code: str
    close_price: float
    ma20: float
    above_ma20: bool
    ma_gap_pct: float  # (종가 - MA20) / MA20 * 100


@dataclass
class SectorBreadthResult:
    """섹터 결속력 결과"""
    sector_name: str
    s_breadth: float           # 0-100%
    above_count: int           # MA20 위 종목 수
    total_count: int           # 전체 종목 수
    stock_results: list[BreadthResult]
    breadth_level: str         # "STRONG", "MODERATE", "WEAK"


class BreadthCalculator:
    """
    S_Breadth 계산기

    섹터 내 MA20 위 종목 비율로 내부 결속력 측정

    사용법:
        calculator = BreadthCalculator()

        # 단일 종목
        result = calculator.calculate_stock(stock_code, close_price, ma20)

        # 섹터 전체
        sector_result = calculator.calculate_sector(sector_name, stocks_data)
    """

    def __init__(self, ma_period: int = 20):
        """
        Args:
            ma_period: 이동평균 기간 (기본 20일)
        """
        self.logger = get_logger(self.__class__.__name__)

        config = get_config()
        self.ma_period = config.get("analysis.breadth.ma_period", ma_period)

        # 결속력 수준 임계값
        self.strong_threshold = config.get("analysis.breadth.strong_threshold", 70)
        self.weak_threshold = config.get("analysis.breadth.weak_threshold", 30)

    def calculate_stock(
        self,
        stock_code: str,
        close_price: float,
        ma20: float
    ) -> BreadthResult:
        """
        단일 종목 MA20 대비 위치 계산

        Args:
            stock_code: 종목코드
            close_price: 현재가 (종가)
            ma20: 20일 이동평균

        Returns:
            BreadthResult
        """
        above_ma20 = close_price > ma20 if ma20 > 0 else False

        if ma20 > 0:
            ma_gap_pct = (close_price - ma20) / ma20 * 100
        else:
            ma_gap_pct = 0.0

        return BreadthResult(
            stock_code=stock_code,
            close_price=close_price,
            ma20=ma20,
            above_ma20=above_ma20,
            ma_gap_pct=ma_gap_pct
        )

    def calculate_stock_from_df(
        self,
        stock_code: str,
        price_df: pd.DataFrame
    ) -> BreadthResult:
        """
        DataFrame에서 종목 Breadth 계산

        Args:
            stock_code: 종목코드
            price_df: 가격 데이터 (columns: date, close)

        Returns:
            BreadthResult
        """
        if price_df.empty or len(price_df) < self.ma_period:
            return BreadthResult(
                stock_code=stock_code,
                close_price=0,
                ma20=0,
                above_ma20=False,
                ma_gap_pct=0
            )

        # MA20 계산
        close_col = "close" if "close" in price_df.columns else "종가"
        ma20 = price_df[close_col].tail(self.ma_period).mean()
        close_price = price_df[close_col].iloc[-1]

        return self.calculate_stock(stock_code, close_price, ma20)

    def calculate_sector(
        self,
        sector_name: str,
        stocks_data: list[dict]
    ) -> SectorBreadthResult:
        """
        섹터 전체 S_Breadth 계산

        Args:
            sector_name: 섹터명
            stocks_data: 종목 데이터 리스트
                [{
                    "stock_code": str,
                    "close_price": float,
                    "ma20": float
                }, ...]

        Returns:
            SectorBreadthResult
        """
        if not stocks_data:
            return SectorBreadthResult(
                sector_name=sector_name,
                s_breadth=0.0,
                above_count=0,
                total_count=0,
                stock_results=[],
                breadth_level="WEAK"
            )

        # 각 종목 계산
        stock_results = []
        for data in stocks_data:
            result = self.calculate_stock(
                stock_code=data["stock_code"],
                close_price=data.get("close_price", 0),
                ma20=data.get("ma20", 0)
            )
            stock_results.append(result)

        # S_Breadth 계산
        total_count = len(stock_results)
        above_count = sum(1 for r in stock_results if r.above_ma20)

        s_breadth = (above_count / total_count * 100) if total_count > 0 else 0

        # 결속력 수준 분류
        breadth_level = self.classify_breadth_level(s_breadth)

        return SectorBreadthResult(
            sector_name=sector_name,
            s_breadth=s_breadth,
            above_count=above_count,
            total_count=total_count,
            stock_results=stock_results,
            breadth_level=breadth_level
        )

    def classify_breadth_level(self, s_breadth: float) -> str:
        """
        결속력 수준 분류

        Args:
            s_breadth: S_Breadth 값 (0-100)

        Returns:
            "STRONG", "MODERATE", "WEAK"
        """
        if s_breadth >= self.strong_threshold:
            return "STRONG"
        elif s_breadth <= self.weak_threshold:
            return "WEAK"
        else:
            return "MODERATE"

    def get_laggards(
        self,
        sector_result: SectorBreadthResult,
        gap_threshold: float = -10.0
    ) -> list[BreadthResult]:
        """
        뒤처진 종목 (MA20 대비 크게 하락) 반환

        Args:
            sector_result: 섹터 결과
            gap_threshold: MA20 대비 이격률 임계값 (%)

        Returns:
            뒤처진 종목 리스트
        """
        return [
            r for r in sector_result.stock_results
            if r.ma_gap_pct < gap_threshold
        ]

    def get_leaders(
        self,
        sector_result: SectorBreadthResult,
        gap_threshold: float = 10.0
    ) -> list[BreadthResult]:
        """
        선도 종목 (MA20 대비 크게 상승) 반환

        Args:
            sector_result: 섹터 결과
            gap_threshold: MA20 대비 이격률 임계값 (%)

        Returns:
            선도 종목 리스트
        """
        return [
            r for r in sector_result.stock_results
            if r.ma_gap_pct > gap_threshold
        ]
