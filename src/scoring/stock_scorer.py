"""
종목 스코어러

Track A/B별 가중치를 적용하여 종목 점수 산출

Track A (실적형): 재무(50%) + 차트/수급(50%)
Track B (성장형): 재무(20%) + 차트/수급(80%)
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.core.interfaces import TrackType
from src.core.config import get_config
from src.core.logger import get_logger


@dataclass
class FinancialMetrics:
    """재무 지표"""
    operating_profit_yoy: float = 0.0      # 영업이익 YoY 성장률 (%)
    revenue_yoy: float = 0.0               # 매출액 YoY 성장률 (%)
    roe: float = 0.0                       # ROE (%)
    rd_ratio: float = 0.0                  # R&D 비중 (%)
    debt_ratio: float = 0.0                # 부채비율 (%)
    current_ratio: float = 0.0             # 유동비율 (%)


@dataclass
class TechnicalMetrics:
    """차트/수급 지표"""
    foreign_net_ratio: float = 0.0         # 외인 순매수 / 시총 (%)
    institution_net_ratio: float = 0.0     # 기관 순매수 / 시총 (%)
    ma20_gap: float = 0.0                  # MA20 이격도 (%)
    volume_ratio: float = 0.0              # 거래량 증가율 (%)
    high_52w_proximity: float = 0.0        # 52주 신고가 근접도 (%)


@dataclass
class ScoreResult:
    """점수 결과"""
    stock_code: str
    stock_name: str
    track_type: TrackType

    # 세부 점수 (0-100)
    financial_score: float
    technical_score: float
    total_score: float

    # 가중치
    financial_weight: float
    technical_weight: float

    # 순위 (외부에서 설정)
    rank: int = 0

    # 상세 지표
    financial_metrics: FinancialMetrics = field(default_factory=FinancialMetrics)
    technical_metrics: TechnicalMetrics = field(default_factory=TechnicalMetrics)

    # 점수 상세 내역
    score_breakdown: dict = field(default_factory=dict)


class StockScorer:
    """
    종목 스코어러

    Track 타입에 따라 다른 가중치로 종목 점수 산출

    사용법:
        scorer = StockScorer()

        # 단일 종목 점수
        result = scorer.score(
            stock_code="005930",
            stock_name="삼성전자",
            track_type=TrackType.TRACK_A,
            financial=FinancialMetrics(...),
            technical=TechnicalMetrics(...)
        )

        # 일괄 점수 + 순위
        results = scorer.score_and_rank(stocks_data)
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

        config = get_config()

        # Track별 가중치 로드
        self.weights = {
            TrackType.TRACK_A: {
                "financial": config.get("scoring.track_a.financial_weight", 0.5),
                "technical": config.get("scoring.track_a.technical_weight", 0.5),
            },
            TrackType.TRACK_B: {
                "financial": config.get("scoring.track_b.financial_weight", 0.2),
                "technical": config.get("scoring.track_b.technical_weight", 0.8),
            },
        }

        # 세부 점수 가중치
        self.financial_weights = {
            "operating_profit_yoy": config.get("scoring.financial.operating_profit_yoy_weight", 0.3),
            "revenue_yoy": config.get("scoring.financial.revenue_yoy_weight", 0.2),
            "roe": config.get("scoring.financial.roe_weight", 0.3),
            "rd_ratio": config.get("scoring.financial.rd_ratio_weight", 0.1),
            "debt_penalty": config.get("scoring.financial.debt_penalty_weight", 0.1),
        }

        self.technical_weights = {
            "supply_demand": config.get("scoring.technical.supply_demand_weight", 0.35),
            "ma_gap": config.get("scoring.technical.ma_gap_weight", 0.2),
            "volume": config.get("scoring.technical.volume_weight", 0.2),
            "high_proximity": config.get("scoring.technical.high_proximity_weight", 0.25),
        }

    def score(
        self,
        stock_code: str,
        stock_name: str,
        track_type: TrackType,
        financial: FinancialMetrics,
        technical: TechnicalMetrics
    ) -> ScoreResult:
        """
        단일 종목 점수 계산

        Args:
            stock_code: 종목코드
            stock_name: 종목명
            track_type: Track 타입 (A/B)
            financial: 재무 지표
            technical: 차트/수급 지표

        Returns:
            ScoreResult
        """
        # 가중치 가져오기
        weights = self.weights[track_type]
        fin_weight = weights["financial"]
        tech_weight = weights["technical"]

        # 재무 점수 계산
        financial_score, fin_breakdown = self._calculate_financial_score(
            financial, track_type
        )

        # 기술적 점수 계산
        technical_score, tech_breakdown = self._calculate_technical_score(technical)

        # 총점 계산
        total_score = (financial_score * fin_weight) + (technical_score * tech_weight)

        return ScoreResult(
            stock_code=stock_code,
            stock_name=stock_name,
            track_type=track_type,
            financial_score=financial_score,
            technical_score=technical_score,
            total_score=total_score,
            financial_weight=fin_weight,
            technical_weight=tech_weight,
            financial_metrics=financial,
            technical_metrics=technical,
            score_breakdown={
                "financial": fin_breakdown,
                "technical": tech_breakdown,
            }
        )

    def score_from_dict(
        self,
        stock_code: str,
        stock_name: str,
        track_type: TrackType,
        data: dict
    ) -> ScoreResult:
        """
        딕셔너리 데이터로 점수 계산

        Args:
            stock_code: 종목코드
            stock_name: 종목명
            track_type: Track 타입
            data: 지표 데이터 딕셔너리

        Returns:
            ScoreResult
        """
        financial = FinancialMetrics(
            operating_profit_yoy=data.get("operating_profit_yoy", 0),
            revenue_yoy=data.get("revenue_yoy", 0),
            roe=data.get("roe", 0),
            rd_ratio=data.get("rd_ratio", 0),
            debt_ratio=data.get("debt_ratio", 0),
            current_ratio=data.get("current_ratio", 0),
        )

        technical = TechnicalMetrics(
            foreign_net_ratio=data.get("foreign_net_ratio", 0),
            institution_net_ratio=data.get("institution_net_ratio", 0),
            ma20_gap=data.get("ma20_gap", 0),
            volume_ratio=data.get("volume_ratio", 0),
            high_52w_proximity=data.get("high_52w_proximity", 0),
        )

        return self.score(stock_code, stock_name, track_type, financial, technical)

    def score_batch(
        self,
        stocks_data: list[dict]
    ) -> list[ScoreResult]:
        """
        일괄 점수 계산

        Args:
            stocks_data: 종목 데이터 리스트
                [{stock_code, stock_name, track_type, ...지표들...}]

        Returns:
            ScoreResult 리스트
        """
        results = []

        for data in stocks_data:
            track_type = data.get("track_type", TrackType.TRACK_A)
            if isinstance(track_type, str):
                track_type = TrackType.TRACK_A if track_type == "A" else TrackType.TRACK_B

            result = self.score_from_dict(
                stock_code=data["stock_code"],
                stock_name=data.get("stock_name", ""),
                track_type=track_type,
                data=data
            )
            results.append(result)

        return results

    def score_and_rank(
        self,
        stocks_data: list[dict],
        top_n: int | None = None
    ) -> list[ScoreResult]:
        """
        점수 계산 후 순위 부여

        Args:
            stocks_data: 종목 데이터 리스트
            top_n: 상위 N개만 반환 (None이면 전체)

        Returns:
            순위가 부여된 ScoreResult 리스트 (점수 내림차순)
        """
        results = self.score_batch(stocks_data)

        # 점수 내림차순 정렬
        sorted_results = sorted(results, key=lambda x: x.total_score, reverse=True)

        # 순위 부여
        for i, result in enumerate(sorted_results):
            result.rank = i + 1

        if top_n:
            return sorted_results[:top_n]

        return sorted_results

    def _calculate_financial_score(
        self,
        metrics: FinancialMetrics,
        track_type: TrackType
    ) -> tuple[float, dict]:
        """
        재무 점수 계산

        Track A: 영업이익, ROE 중시
        Track B: 매출 성장, R&D 중시
        """
        breakdown = {}
        scores = []

        if track_type == TrackType.TRACK_A:
            # Track A: 영업이익 성장률
            op_score = self._normalize_score(metrics.operating_profit_yoy, -20, 50)
            breakdown["영업이익YoY"] = op_score
            scores.append((op_score, self.financial_weights["operating_profit_yoy"]))

            # ROE
            roe_score = self._normalize_score(metrics.roe, 0, 20)
            breakdown["ROE"] = roe_score
            scores.append((roe_score, self.financial_weights["roe"]))

            # 부채비율 패널티
            debt_penalty = self._calculate_debt_penalty(metrics.debt_ratio)
            breakdown["부채패널티"] = -debt_penalty
            scores.append((-debt_penalty, self.financial_weights["debt_penalty"]))

        else:
            # Track B: 매출 성장률
            rev_score = self._normalize_score(metrics.revenue_yoy, -10, 100)
            breakdown["매출YoY"] = rev_score
            scores.append((rev_score, self.financial_weights["revenue_yoy"]))

            # R&D 비중 (가산점)
            rd_score = self._normalize_score(metrics.rd_ratio, 0, 20)
            breakdown["R&D비중"] = rd_score
            scores.append((rd_score, self.financial_weights["rd_ratio"]))

        # 가중 평균
        total_weight = sum(w for _, w in scores)
        if total_weight > 0:
            final_score = sum(s * w for s, w in scores) / total_weight
        else:
            final_score = 0

        return max(0, min(100, final_score)), breakdown

    def _calculate_technical_score(
        self,
        metrics: TechnicalMetrics
    ) -> tuple[float, dict]:
        """차트/수급 점수 계산"""
        breakdown = {}
        scores = []

        # 1. 수급 점수 (외인 + 기관)
        supply_score = self._calculate_supply_score(
            metrics.foreign_net_ratio,
            metrics.institution_net_ratio
        )
        breakdown["수급"] = supply_score
        scores.append((supply_score, self.technical_weights["supply_demand"]))

        # 2. MA20 이격도
        ma_score = self._calculate_ma_gap_score(metrics.ma20_gap)
        breakdown["이격도"] = ma_score
        scores.append((ma_score, self.technical_weights["ma_gap"]))

        # 3. 거래량 증가율
        vol_score = self._normalize_score(metrics.volume_ratio, 0, 200)
        breakdown["거래량"] = vol_score
        scores.append((vol_score, self.technical_weights["volume"]))

        # 4. 52주 신고가 근접도
        high_score = self._normalize_score(metrics.high_52w_proximity, 50, 100)
        breakdown["신고가근접"] = high_score
        scores.append((high_score, self.technical_weights["high_proximity"]))

        # 가중 평균
        total_weight = sum(w for _, w in scores)
        if total_weight > 0:
            final_score = sum(s * w for s, w in scores) / total_weight
        else:
            final_score = 0

        return max(0, min(100, final_score)), breakdown

    def _normalize_score(
        self,
        value: float,
        min_val: float,
        max_val: float
    ) -> float:
        """값을 0-100 점수로 정규화"""
        if max_val == min_val:
            return 50.0

        normalized = (value - min_val) / (max_val - min_val) * 100
        return max(0, min(100, normalized))

    def _calculate_debt_penalty(self, debt_ratio: float) -> float:
        """부채비율 패널티 계산 (0-30점)"""
        if debt_ratio <= 100:
            return 0
        elif debt_ratio <= 200:
            return (debt_ratio - 100) / 100 * 15
        else:
            return 15 + min((debt_ratio - 200) / 100 * 15, 15)

    def _calculate_supply_score(
        self,
        foreign_ratio: float,
        institution_ratio: float
    ) -> float:
        """수급 점수 계산 (기관 1.2배 가중)"""
        weighted = foreign_ratio + (institution_ratio * 1.2)

        # -1% ~ +1% 범위를 0-100으로 매핑
        return self._normalize_score(weighted, -1, 1)

    def _calculate_ma_gap_score(self, ma_gap: float) -> float:
        """
        MA20 이격도 점수

        -10% ~ 0%: 눌림목 (고점수)
        0% ~ 10%: 적정 (중간 점수)
        10% 이상: 과열 (저점수)
        """
        if ma_gap < -10:
            return 30  # 너무 많이 빠짐
        elif ma_gap < 0:
            # 눌림목 구간 (높은 점수)
            return 70 + (ma_gap + 10) * 3  # -10% -> 70점, 0% -> 100점
        elif ma_gap < 5:
            return 100 - ma_gap * 4  # 0% -> 100점, 5% -> 80점
        elif ma_gap < 15:
            return 80 - (ma_gap - 5) * 4  # 5% -> 80점, 15% -> 40점
        else:
            return max(0, 40 - (ma_gap - 15) * 2)

    def get_weight_config(self, track_type: TrackType) -> dict:
        """Track별 가중치 반환"""
        return self.weights[track_type]

    def get_top_stocks(
        self,
        results: list[ScoreResult],
        n: int = 10,
        track_type: TrackType | None = None
    ) -> list[ScoreResult]:
        """
        상위 N개 종목 반환

        Args:
            results: 점수 결과 리스트
            n: 반환할 개수
            track_type: 특정 Track만 필터 (None이면 전체)

        Returns:
            상위 종목 리스트
        """
        if track_type:
            filtered = [r for r in results if r.track_type == track_type]
        else:
            filtered = results

        sorted_results = sorted(filtered, key=lambda x: x.total_score, reverse=True)
        return sorted_results[:n]
