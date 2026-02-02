"""
④ 종목 점수화 모듈

Track A/B별 가중치로 종목 점수 계산
- Track A: 재무 50% + 기술 50%
- Track B: 재무 20% + 기술 80%
"""
from dataclasses import dataclass, field
from typing import Any

from src.core.logger import get_logger
from src.core.config import get_config
from src.core.interfaces import TrackType


@dataclass
class StockScoreResult:
    """종목 점수 결과"""
    stock_code: str
    stock_name: str
    track_type: TrackType

    # 점수
    financial_score: float  # 재무 점수 (0~100)
    technical_score: float  # 기술 점수 (0~100)
    total_score: float  # 종합 점수 (0~100)

    # 순위 (나중에 설정)
    rank: int = 0

    # 세부 점수
    financial_breakdown: dict = field(default_factory=dict)
    technical_breakdown: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "stock_code": self.stock_code,
            "stock_name": self.stock_name,
            "track_type": self.track_type.value,
            "financial_score": round(self.financial_score, 1),
            "technical_score": round(self.technical_score, 1),
            "total_score": round(self.total_score, 1),
            "rank": self.rank,
            "financial_breakdown": self.financial_breakdown,
            "technical_breakdown": self.technical_breakdown,
        }


class StockScorer:
    """
    ④ 종목 점수화

    Track별 가중치:
    - Track A (실적형): 재무 50% + 기술 50%
    - Track B (성장형): 재무 20% + 기술 80%

    재무 점수 (Track A):
    - 영업이익 YoY (30%)
    - ROE (30%)
    - 부채비율 패널티 (40%)

    재무 점수 (Track B):
    - 매출액 YoY (70%)
    - R&D 비중 (30%)

    기술 점수 (공통):
    - 수급 강도 (35%)
    - MA20 이격도 (20%)
    - 거래량 증가율 (20%)
    - 52주 신고가 근접도 (25%)

    사용법:
        scorer = StockScorer()

        # 단일 종목 점수화
        result = scorer.score(stock_data, track_type=TrackType.TRACK_A)

        # 배치 점수화
        results = scorer.score_batch(stocks_data, track_type)
    """

    # Track별 가중치
    TRACK_WEIGHTS = {
        TrackType.TRACK_A: {"financial": 0.5, "technical": 0.5},
        TrackType.TRACK_B: {"financial": 0.2, "technical": 0.8},
    }

    # 재무 점수 가중치
    FINANCIAL_WEIGHTS_A = {
        "operating_profit_yoy": 0.30,
        "roe": 0.30,
        "debt_penalty": 0.40,
    }

    FINANCIAL_WEIGHTS_B = {
        "revenue_yoy": 0.70,
        "rd_ratio": 0.30,
    }

    # 기술 점수 가중치 (공통)
    TECHNICAL_WEIGHTS = {
        "supply_demand": 0.35,  # 수급 강도
        "ma20_gap": 0.20,  # MA20 이격도
        "volume_ratio": 0.20,  # 거래량 비율
        "high_52w_proximity": 0.25,  # 52주 신고가 근접도
    }

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.config = get_config()

    def score(
        self,
        stock_data: dict,
        track_type: TrackType,
    ) -> StockScoreResult:
        """
        단일 종목 점수화

        Args:
            stock_data: 종목 데이터
                [재무 - Track A]
                - operating_profit_yoy: 영업이익 YoY (%)
                - roe: ROE (%)
                - debt_ratio: 부채비율 (%)

                [재무 - Track B]
                - revenue_yoy: 매출액 YoY (%)
                - rd_ratio: R&D 비중 (%)

                [기술 - 공통]
                - foreign_net_ratio: 외인순매수비율 (%)
                - institution_net_ratio: 기관순매수비율 (%)
                - ma20_gap: MA20 이격도 (%)
                - volume_ratio: 거래량 비율
                - high_52w_proximity: 52주 신고가 근접도 (0~1)

            track_type: 트랙 타입

        Returns:
            StockScoreResult
        """
        stock_code = stock_data.get("stock_code", "")
        stock_name = stock_data.get("stock_name", "")

        # 재무 점수 계산
        if track_type == TrackType.TRACK_A:
            financial_score, financial_breakdown = self._calc_financial_score_a(stock_data)
        else:
            financial_score, financial_breakdown = self._calc_financial_score_b(stock_data)

        # 기술 점수 계산
        technical_score, technical_breakdown = self._calc_technical_score(stock_data)

        # 종합 점수 계산
        weights = self.TRACK_WEIGHTS[track_type]
        total_score = (
            financial_score * weights["financial"] +
            technical_score * weights["technical"]
        )

        return StockScoreResult(
            stock_code=stock_code,
            stock_name=stock_name,
            track_type=track_type,
            financial_score=financial_score,
            technical_score=technical_score,
            total_score=total_score,
            financial_breakdown=financial_breakdown,
            technical_breakdown=technical_breakdown,
        )

    def _calc_financial_score_a(self, data: dict) -> tuple[float, dict]:
        """Track A 재무 점수"""
        breakdown = {}

        # 영업이익 YoY (-20% ~ +50% → 0~100)
        op_yoy = data.get("operating_profit_yoy", 0) or 0
        op_score = self._normalize(op_yoy, -20, 50) * 100
        breakdown["operating_profit_yoy"] = {
            "value": op_yoy,
            "score": op_score,
            "weight": self.FINANCIAL_WEIGHTS_A["operating_profit_yoy"],
        }

        # ROE (0% ~ 20% → 0~100)
        roe = data.get("roe", 0) or 0
        roe_score = self._normalize(roe, 0, 20) * 100
        breakdown["roe"] = {
            "value": roe,
            "score": roe_score,
            "weight": self.FINANCIAL_WEIGHTS_A["roe"],
        }

        # 부채비율 패널티 (0% ~ 200% → 100~0)
        debt = data.get("debt_ratio", 0) or 0
        debt_score = (1 - self._normalize(debt, 0, 200)) * 100
        breakdown["debt_penalty"] = {
            "value": debt,
            "score": debt_score,
            "weight": self.FINANCIAL_WEIGHTS_A["debt_penalty"],
        }

        # 가중 합산
        total = (
            op_score * self.FINANCIAL_WEIGHTS_A["operating_profit_yoy"] +
            roe_score * self.FINANCIAL_WEIGHTS_A["roe"] +
            debt_score * self.FINANCIAL_WEIGHTS_A["debt_penalty"]
        )

        return total, breakdown

    def _calc_financial_score_b(self, data: dict) -> tuple[float, dict]:
        """Track B 재무 점수"""
        breakdown = {}

        # 매출액 YoY (-10% ~ +100% → 0~100)
        rev_yoy = data.get("revenue_yoy", 0) or 0
        rev_score = self._normalize(rev_yoy, -10, 100) * 100
        breakdown["revenue_yoy"] = {
            "value": rev_yoy,
            "score": rev_score,
            "weight": self.FINANCIAL_WEIGHTS_B["revenue_yoy"],
        }

        # R&D 비중 (0% ~ 20% → 0~100)
        rd = data.get("rd_ratio", 0) or 0
        rd_score = self._normalize(rd, 0, 20) * 100
        breakdown["rd_ratio"] = {
            "value": rd,
            "score": rd_score,
            "weight": self.FINANCIAL_WEIGHTS_B["rd_ratio"],
        }

        # 가중 합산
        total = (
            rev_score * self.FINANCIAL_WEIGHTS_B["revenue_yoy"] +
            rd_score * self.FINANCIAL_WEIGHTS_B["rd_ratio"]
        )

        return total, breakdown

    def _calc_technical_score(self, data: dict) -> tuple[float, dict]:
        """기술 점수 (공통)"""
        breakdown = {}

        # 수급 강도 (외인 + 기관*1.2)
        foreign = data.get("foreign_net_ratio", 0) or 0
        inst = data.get("institution_net_ratio", 0) or 0
        supply = foreign + inst * 1.2
        supply_score = self._normalize(supply, -5, 10) * 100
        breakdown["supply_demand"] = {
            "value": supply,
            "score": supply_score,
            "weight": self.TECHNICAL_WEIGHTS["supply_demand"],
        }

        # MA20 이격도 (-10% ~ +10%, 0에 가까울수록 좋음 = 눌림목)
        ma20_gap = data.get("ma20_gap", 0) or 0
        # 이격도가 0~5% 사이면 최고점, 너무 멀면 감점
        ma20_score = 100 - abs(ma20_gap - 2.5) * 10
        ma20_score = max(0, min(100, ma20_score))
        breakdown["ma20_gap"] = {
            "value": ma20_gap,
            "score": ma20_score,
            "weight": self.TECHNICAL_WEIGHTS["ma20_gap"],
        }

        # 거래량 비율 (0.5 ~ 3.0)
        vol_ratio = data.get("volume_ratio", 1.0) or 1.0
        vol_score = self._normalize(vol_ratio, 0.5, 3.0) * 100
        breakdown["volume_ratio"] = {
            "value": vol_ratio,
            "score": vol_score,
            "weight": self.TECHNICAL_WEIGHTS["volume_ratio"],
        }

        # 52주 신고가 근접도 (0 ~ 1)
        high_prox = data.get("high_52w_proximity", 0.5) or 0.5
        high_score = high_prox * 100
        breakdown["high_52w_proximity"] = {
            "value": high_prox,
            "score": high_score,
            "weight": self.TECHNICAL_WEIGHTS["high_52w_proximity"],
        }

        # 가중 합산
        total = (
            supply_score * self.TECHNICAL_WEIGHTS["supply_demand"] +
            ma20_score * self.TECHNICAL_WEIGHTS["ma20_gap"] +
            vol_score * self.TECHNICAL_WEIGHTS["volume_ratio"] +
            high_score * self.TECHNICAL_WEIGHTS["high_52w_proximity"]
        )

        return total, breakdown

    def _normalize(self, value: float, min_val: float, max_val: float) -> float:
        """0~1로 정규화"""
        if max_val == min_val:
            return 0.5
        return max(0, min(1, (value - min_val) / (max_val - min_val)))

    def score_batch(
        self,
        stocks_data: list[dict],
        track_type: TrackType,
        progress_callback=None,
    ) -> list[StockScoreResult]:
        """
        배치 점수화

        Args:
            stocks_data: 종목 데이터 리스트
            track_type: 트랙 타입
            progress_callback: 진행 콜백

        Returns:
            StockScoreResult 리스트 (점수순 정렬, 순위 포함)
        """
        results = []

        for i, stock_data in enumerate(stocks_data):
            result = self.score(stock_data, track_type)
            results.append(result)

            if progress_callback and (i + 1) % 50 == 0:
                progress_callback(i + 1, len(stocks_data))

        # 점수순 정렬 및 순위 부여
        results = sorted(results, key=lambda x: x.total_score, reverse=True)
        for i, result in enumerate(results):
            result.rank = i + 1

        return results

    def get_top_n(self, results: list[StockScoreResult], n: int) -> list[StockScoreResult]:
        """상위 N개 종목 반환"""
        sorted_results = sorted(results, key=lambda x: x.total_score, reverse=True)
        return sorted_results[:n]

    def summarize(self, results: list[StockScoreResult]) -> dict:
        """점수 결과 요약"""
        if not results:
            return {"total": 0}

        scores = [r.total_score for r in results]

        return {
            "total": len(results),
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "score_distribution": {
                "80+": sum(1 for s in scores if s >= 80),
                "60-80": sum(1 for s in scores if 60 <= s < 80),
                "40-60": sum(1 for s in scores if 40 <= s < 60),
                "40-": sum(1 for s in scores if s < 40),
            },
        }

    def get_weight_config(self, track_type: TrackType) -> dict:
        """트랙별 가중치 설정 반환"""
        return {
            "track_weights": self.TRACK_WEIGHTS[track_type],
            "financial_weights": (
                self.FINANCIAL_WEIGHTS_A if track_type == TrackType.TRACK_A
                else self.FINANCIAL_WEIGHTS_B
            ),
            "technical_weights": self.TECHNICAL_WEIGHTS,
        }
