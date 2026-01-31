"""
S_Trend (섹터 추세) 계산

"추세가 살아있는가?"

공식: S_Trend = 정배열 점수(50) + 모멘텀 점수(30) - 과열 패널티(20)
- 정배열: MA5 > MA20 > MA60 > MA120
- 모멘텀: 최근 20일 수익률
- 과열: RSI > 70이면 패널티
"""
from dataclasses import dataclass
from typing import Any

import pandas as pd
import numpy as np

from src.core.logger import get_logger
from src.core.config import get_config


@dataclass
class TrendResult:
    """종목 추세 결과"""
    stock_code: str
    s_trend: float              # 총점 (0-100)
    alignment_score: float      # 정배열 점수 (0-50)
    momentum_score: float       # 모멘텀 점수 (0-30)
    overheat_penalty: float     # 과열 패널티 (0-20)

    # 상세 지표
    ma5: float
    ma20: float
    ma60: float
    ma120: float
    is_aligned: bool            # 정배열 여부
    momentum_20d: float         # 20일 수익률 (%)
    rsi: float                  # RSI 값


@dataclass
class SectorTrendResult:
    """섹터 추세 결과"""
    sector_name: str
    s_trend: float              # 평균 추세 점수
    trend_level: str            # "STRONG", "MODERATE", "WEAK"
    aligned_ratio: float        # 정배열 종목 비율 (%)
    avg_momentum: float         # 평균 모멘텀
    overheated_count: int       # 과열 종목 수
    stock_results: list[TrendResult]


class TrendCalculator:
    """
    S_Trend 계산기

    정배열 + 모멘텀 - 과열로 추세 강도 측정

    사용법:
        calculator = TrendCalculator()

        # 단일 종목 (DataFrame 필요)
        result = calculator.calculate_stock(stock_code, price_df)

        # 섹터 전체
        sector_result = calculator.calculate_sector(sector_name, stocks_price_data)
    """

    # 점수 가중치
    ALIGNMENT_MAX = 50
    MOMENTUM_MAX = 30
    OVERHEAT_MAX = 20

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

        config = get_config()
        self.rsi_period = config.get("analysis.trend.rsi_period", 14)
        self.overheat_rsi = config.get("analysis.trend.overheat_rsi", 70)
        self.strong_threshold = config.get("analysis.trend.strong_threshold", 60)
        self.weak_threshold = config.get("analysis.trend.weak_threshold", 30)

    def calculate_stock(
        self,
        stock_code: str,
        price_df: pd.DataFrame
    ) -> TrendResult:
        """
        단일 종목 S_Trend 계산

        Args:
            stock_code: 종목코드
            price_df: 가격 데이터 (columns: date, close 또는 종가)

        Returns:
            TrendResult
        """
        if price_df.empty or len(price_df) < 120:
            return self._empty_result(stock_code)

        # 종가 컬럼 찾기
        close_col = "close" if "close" in price_df.columns else "종가"
        closes = price_df[close_col].values

        # 이동평균 계산
        ma5 = np.mean(closes[-5:])
        ma20 = np.mean(closes[-20:])
        ma60 = np.mean(closes[-60:])
        ma120 = np.mean(closes[-120:])

        # 1. 정배열 점수 (0-50)
        is_aligned = ma5 > ma20 > ma60 > ma120
        alignment_score = self._calculate_alignment_score(ma5, ma20, ma60, ma120)

        # 2. 모멘텀 점수 (0-30)
        momentum_20d = (closes[-1] / closes[-20] - 1) * 100 if closes[-20] != 0 else 0
        momentum_score = self._calculate_momentum_score(momentum_20d)

        # 3. RSI 계산 및 과열 패널티 (0-20)
        rsi = self._calculate_rsi(closes)
        overheat_penalty = self._calculate_overheat_penalty(rsi)

        # 총점
        s_trend = alignment_score + momentum_score - overheat_penalty

        return TrendResult(
            stock_code=stock_code,
            s_trend=s_trend,
            alignment_score=alignment_score,
            momentum_score=momentum_score,
            overheat_penalty=overheat_penalty,
            ma5=ma5,
            ma20=ma20,
            ma60=ma60,
            ma120=ma120,
            is_aligned=is_aligned,
            momentum_20d=momentum_20d,
            rsi=rsi
        )

    def calculate_stock_from_dict(
        self,
        stock_code: str,
        data: dict
    ) -> TrendResult:
        """
        딕셔너리 데이터로 계산 (이미 계산된 지표 사용)

        Args:
            stock_code: 종목코드
            data: {ma5, ma20, ma60, ma120, momentum_20d, rsi}

        Returns:
            TrendResult
        """
        ma5 = data.get("ma5", 0)
        ma20 = data.get("ma20", 0)
        ma60 = data.get("ma60", 0)
        ma120 = data.get("ma120", 0)
        momentum_20d = data.get("momentum_20d", 0)
        rsi = data.get("rsi", 50)

        is_aligned = ma5 > ma20 > ma60 > ma120 if all([ma5, ma20, ma60, ma120]) else False
        alignment_score = self._calculate_alignment_score(ma5, ma20, ma60, ma120)
        momentum_score = self._calculate_momentum_score(momentum_20d)
        overheat_penalty = self._calculate_overheat_penalty(rsi)

        s_trend = alignment_score + momentum_score - overheat_penalty

        return TrendResult(
            stock_code=stock_code,
            s_trend=s_trend,
            alignment_score=alignment_score,
            momentum_score=momentum_score,
            overheat_penalty=overheat_penalty,
            ma5=ma5,
            ma20=ma20,
            ma60=ma60,
            ma120=ma120,
            is_aligned=is_aligned,
            momentum_20d=momentum_20d,
            rsi=rsi
        )

    def calculate_sector(
        self,
        sector_name: str,
        stocks_data: list[dict]
    ) -> SectorTrendResult:
        """
        섹터 전체 S_Trend 계산

        Args:
            sector_name: 섹터명
            stocks_data: 종목별 지표 데이터 리스트
                [{
                    "stock_code": str,
                    "ma5": float, "ma20": float, "ma60": float, "ma120": float,
                    "momentum_20d": float, "rsi": float
                }, ...]

        Returns:
            SectorTrendResult
        """
        if not stocks_data:
            return SectorTrendResult(
                sector_name=sector_name,
                s_trend=0.0,
                trend_level="WEAK",
                aligned_ratio=0.0,
                avg_momentum=0.0,
                overheated_count=0,
                stock_results=[]
            )

        # 각 종목 계산
        stock_results = []
        for data in stocks_data:
            result = self.calculate_stock_from_dict(
                stock_code=data["stock_code"],
                data=data
            )
            stock_results.append(result)

        # 섹터 통계
        s_trend = np.mean([r.s_trend for r in stock_results])
        aligned_count = sum(1 for r in stock_results if r.is_aligned)
        aligned_ratio = (aligned_count / len(stock_results) * 100)
        avg_momentum = np.mean([r.momentum_20d for r in stock_results])
        overheated_count = sum(1 for r in stock_results if r.rsi > self.overheat_rsi)

        trend_level = self.classify_trend_level(s_trend)

        return SectorTrendResult(
            sector_name=sector_name,
            s_trend=s_trend,
            trend_level=trend_level,
            aligned_ratio=aligned_ratio,
            avg_momentum=avg_momentum,
            overheated_count=overheated_count,
            stock_results=stock_results
        )

    def _calculate_alignment_score(
        self,
        ma5: float,
        ma20: float,
        ma60: float,
        ma120: float
    ) -> float:
        """정배열 점수 계산 (0-50)"""
        if not all([ma5, ma20, ma60, ma120]):
            return 0.0

        score = 0.0

        # 완전 정배열: 50점
        if ma5 > ma20 > ma60 > ma120:
            score = 50.0
        else:
            # 부분 정배열 점수
            if ma5 > ma20:
                score += 15
            if ma20 > ma60:
                score += 15
            if ma60 > ma120:
                score += 10

            # 역배열 페널티
            if ma5 < ma120:
                score -= 10

        return max(0, min(score, self.ALIGNMENT_MAX))

    def _calculate_momentum_score(self, momentum_20d: float) -> float:
        """모멘텀 점수 계산 (0-30)"""
        # 20일 수익률 기반
        # 10% 이상: 30점, 5% 이상: 20점, 0% 이상: 10점, 음수: 0점
        if momentum_20d >= 10:
            return 30.0
        elif momentum_20d >= 5:
            return 20.0 + (momentum_20d - 5) * 2  # 5~10%: 20~30점
        elif momentum_20d >= 0:
            return 10.0 + momentum_20d * 2  # 0~5%: 10~20점
        elif momentum_20d >= -5:
            return max(0, 10.0 + momentum_20d * 2)  # -5~0%: 0~10점
        else:
            return 0.0

    def _calculate_overheat_penalty(self, rsi: float) -> float:
        """과열 패널티 계산 (0-20)"""
        if rsi > self.overheat_rsi:
            # RSI 70 이상부터 패널티, 90이면 최대 20점
            excess = rsi - self.overheat_rsi
            return min(excess, self.OVERHEAT_MAX)
        return 0.0

    def _calculate_rsi(self, closes: np.ndarray, period: int = None) -> float:
        """RSI 계산"""
        period = period or self.rsi_period

        if len(closes) < period + 1:
            return 50.0  # 기본값

        deltas = np.diff(closes[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def classify_trend_level(self, s_trend: float) -> str:
        """추세 수준 분류"""
        if s_trend >= self.strong_threshold:
            return "STRONG"
        elif s_trend <= self.weak_threshold:
            return "WEAK"
        else:
            return "MODERATE"

    def _empty_result(self, stock_code: str) -> TrendResult:
        """빈 결과 반환"""
        return TrendResult(
            stock_code=stock_code,
            s_trend=0.0,
            alignment_score=0.0,
            momentum_score=0.0,
            overheat_penalty=0.0,
            ma5=0.0,
            ma20=0.0,
            ma60=0.0,
            ma120=0.0,
            is_aligned=False,
            momentum_20d=0.0,
            rsi=50.0
        )
