"""
S_Flow (수급 강도) 계산

"돈이 들어오고 있는가?"

공식: S_Flow = Σ(외인순매수 + 기관순매수 × 1.2) / 시가총액
- 기관에 1.2 가중치: 국내 기관의 정보력 우위 가정
- 시총으로 정규화: 대형주/소형주 비교 가능
"""
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.core.logger import get_logger
from src.core.config import get_config


@dataclass
class FlowResult:
    """수급 강도 계산 결과"""
    stock_code: str
    s_flow: float
    foreign_net: float      # 외인 순매수 (원)
    institution_net: float  # 기관 순매수 (원)
    market_cap: float       # 시가총액 (원)
    period_days: int        # 계산 기간


@dataclass
class SectorFlowResult:
    """섹터 수급 강도 결과"""
    sector_name: str
    s_flow: float
    s_flow_ex_leader: float  # 대장주 제외
    stock_flows: list[FlowResult]
    leader_stock: str | None
    is_leader_distorted: bool  # 대장주 착시 여부


class FlowCalculator:
    """
    S_Flow 계산기

    외인/기관 순매수 합계를 시가총액으로 정규화하여 수급 강도 측정

    사용법:
        calculator = FlowCalculator()

        # 단일 종목
        result = calculator.calculate_stock(stock_code, trading_data, market_cap)

        # 섹터 전체
        sector_result = calculator.calculate_sector(sector_name, stocks_data)
    """

    def __init__(
        self,
        institution_weight: float = 1.2,
        period_days: int = 20
    ):
        """
        Args:
            institution_weight: 기관 순매수 가중치 (기본 1.2)
            period_days: 수급 계산 기간 (기본 20일)
        """
        self.logger = get_logger(self.__class__.__name__)

        config = get_config()
        self.institution_weight = config.get(
            "analysis.flow.institution_weight",
            institution_weight
        )
        self.period_days = config.get(
            "analysis.flow.period_days",
            period_days
        )

    def calculate_stock(
        self,
        stock_code: str,
        foreign_net: float,
        institution_net: float,
        market_cap: float
    ) -> FlowResult:
        """
        단일 종목 S_Flow 계산

        Args:
            stock_code: 종목코드
            foreign_net: 외인 순매수 금액 (원)
            institution_net: 기관 순매수 금액 (원)
            market_cap: 시가총액 (원)

        Returns:
            FlowResult
        """
        if market_cap <= 0:
            self.logger.warning(f"[{stock_code}] 시가총액이 0 이하")
            s_flow = 0.0
        else:
            # S_Flow = (외인 + 기관 × 1.2) / 시총
            weighted_net = foreign_net + (institution_net * self.institution_weight)
            s_flow = weighted_net / market_cap * 100  # 백분율

        return FlowResult(
            stock_code=stock_code,
            s_flow=s_flow,
            foreign_net=foreign_net,
            institution_net=institution_net,
            market_cap=market_cap,
            period_days=self.period_days
        )

    def calculate_stock_from_df(
        self,
        stock_code: str,
        trading_df: pd.DataFrame,
        market_cap: float
    ) -> FlowResult:
        """
        DataFrame에서 종목 S_Flow 계산

        Args:
            stock_code: 종목코드
            trading_df: 거래 데이터 (columns: date, foreign_net, institution_net)
            market_cap: 시가총액

        Returns:
            FlowResult
        """
        if trading_df.empty:
            return FlowResult(
                stock_code=stock_code,
                s_flow=0.0,
                foreign_net=0,
                institution_net=0,
                market_cap=market_cap,
                period_days=0
            )

        # 최근 N일 데이터만 사용
        recent_df = trading_df.tail(self.period_days)

        foreign_net = recent_df.get("foreign_net", pd.Series([0])).sum()
        institution_net = recent_df.get("institution_net", pd.Series([0])).sum()

        return self.calculate_stock(
            stock_code=stock_code,
            foreign_net=foreign_net,
            institution_net=institution_net,
            market_cap=market_cap
        )

    def calculate_sector(
        self,
        sector_name: str,
        stocks_data: list[dict]
    ) -> SectorFlowResult:
        """
        섹터 전체 S_Flow 계산

        Args:
            sector_name: 섹터명
            stocks_data: 종목 데이터 리스트
                [{
                    "stock_code": str,
                    "foreign_net": float,
                    "institution_net": float,
                    "market_cap": float
                }, ...]

        Returns:
            SectorFlowResult
        """
        if not stocks_data:
            return SectorFlowResult(
                sector_name=sector_name,
                s_flow=0.0,
                s_flow_ex_leader=0.0,
                stock_flows=[],
                leader_stock=None,
                is_leader_distorted=False
            )

        # 각 종목 S_Flow 계산
        stock_flows = []
        for data in stocks_data:
            result = self.calculate_stock(
                stock_code=data["stock_code"],
                foreign_net=data.get("foreign_net", 0),
                institution_net=data.get("institution_net", 0),
                market_cap=data.get("market_cap", 1)
            )
            stock_flows.append(result)

        # 섹터 전체 S_Flow (시총 가중 평균)
        total_market_cap = sum(r.market_cap for r in stock_flows)
        if total_market_cap > 0:
            weighted_flow = sum(
                r.s_flow * r.market_cap for r in stock_flows
            ) / total_market_cap
        else:
            weighted_flow = 0.0

        # 대장주 찾기 (시총 1위)
        sorted_by_cap = sorted(stock_flows, key=lambda x: x.market_cap, reverse=True)
        leader = sorted_by_cap[0] if sorted_by_cap else None
        leader_code = leader.stock_code if leader else None

        # 대장주 제외 S_Flow
        ex_leader_flows = [r for r in stock_flows if r.stock_code != leader_code]
        if ex_leader_flows:
            ex_leader_cap = sum(r.market_cap for r in ex_leader_flows)
            if ex_leader_cap > 0:
                s_flow_ex_leader = sum(
                    r.s_flow * r.market_cap for r in ex_leader_flows
                ) / ex_leader_cap
            else:
                s_flow_ex_leader = 0.0
        else:
            s_flow_ex_leader = 0.0

        # 대장주 착시 판단
        # 전체 S_Flow > 0 이지만, 대장주 제외 시 <= 0 이면 착시
        is_distorted = weighted_flow > 0 and s_flow_ex_leader <= 0

        if is_distorted:
            self.logger.warning(
                f"[{sector_name}] 대장주 착시 감지: "
                f"전체 S_Flow={weighted_flow:.2f}, 제외 시={s_flow_ex_leader:.2f}"
            )

        return SectorFlowResult(
            sector_name=sector_name,
            s_flow=weighted_flow,
            s_flow_ex_leader=s_flow_ex_leader,
            stock_flows=stock_flows,
            leader_stock=leader_code,
            is_leader_distorted=is_distorted
        )

    def classify_flow_level(self, s_flow: float) -> str:
        """
        S_Flow 수준 분류

        Args:
            s_flow: S_Flow 값

        Returns:
            "HIGH", "MEDIUM", "LOW"
        """
        config = get_config()
        high_threshold = config.get("analysis.flow.high_threshold", 0.5)
        low_threshold = config.get("analysis.flow.low_threshold", -0.5)

        if s_flow >= high_threshold:
            return "HIGH"
        elif s_flow <= low_threshold:
            return "LOW"
        else:
            return "MEDIUM"

    def calculate_stock_auto(
        self,
        stock_code: str,
        days: int | None = None
    ) -> FlowResult:
        """
        개별 종목 S_Flow 자동 계산 (데이터 조회 포함)

        Args:
            stock_code: 종목코드
            days: 조회 기간 (기본값: self.period_days)

        Returns:
            FlowResult
        """
        from src.ingest import PriceDataFetcher

        if days is None:
            days = self.period_days

        fetcher = PriceDataFetcher(lookback_days=days)
        supply_data = fetcher.fetch_stock_supply_demand(stock_code, days)

        return self.calculate_stock(
            stock_code=stock_code,
            foreign_net=supply_data.get("foreign_net", 0),
            institution_net=supply_data.get("institution_net", 0),
            market_cap=supply_data.get("market_cap", 1)
        )
