"""
섹터 Tier 분류기

S_Flow, S_Breadth, S_Trend를 종합하여 섹터 등급 결정

Tier 분류 매트릭스:
                 S_Trend (추세)
              Low/Flat     High
         ┌───────────┬───────────┐
S_Flow   │  TIER 1   │  TIER 2   │
 High    │  수급빈집  │  주도섹터  │
         ├───────────┼───────────┤
 Low     │  무관심    │  TIER 3   │
         │  (스킵)   │  가짜상승   │
         └───────────┴───────────┘

대장주 왜곡 검증: 대장주 1개 제외 후에도 S_Flow > 0 이어야 Tier 1,2 유지
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.core.logger import get_logger
from src.core.config import get_config
from src.analysis.metrics.flow import FlowCalculator, SectorFlowResult
from src.analysis.metrics.breadth import BreadthCalculator, SectorBreadthResult
from src.analysis.metrics.trend import TrendCalculator, SectorTrendResult


class SectorTier(Enum):
    """섹터 등급"""
    TIER_1 = 1  # 수급 빈집 - 선취매 기회
    TIER_2 = 2  # 주도 섹터 - 눌림목 대기
    TIER_3 = 3  # 가짜 상승 - 진입 금지
    SKIP = 0    # 무관심 - 스킵


@dataclass
class SectorAnalysisResult:
    """섹터 분석 종합 결과"""
    sector_name: str
    tier: SectorTier

    # 지표별 결과
    flow_result: SectorFlowResult
    breadth_result: SectorBreadthResult
    trend_result: SectorTrendResult

    # 핵심 지표
    s_flow: float
    s_breadth: float
    s_trend: float

    # 수준 분류
    flow_level: str      # "HIGH", "MEDIUM", "LOW"
    breadth_level: str   # "STRONG", "MODERATE", "WEAK"
    trend_level: str     # "STRONG", "MODERATE", "WEAK"

    # 경고
    is_leader_distorted: bool  # 대장주 착시
    warnings: list[str]

    # 투자 코멘트
    comment: str


class TierClassifier:
    """
    섹터 Tier 분류기

    사용법:
        classifier = TierClassifier()

        # 섹터 분석
        result = classifier.classify(
            sector_name="HBM",
            stocks_data=[...],
            price_data=[...]
        )

        # 다중 섹터 분석
        results = classifier.classify_batch(sectors_data)
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

        config = get_config()

        # 임계값 설정
        self.flow_high = config.get("analysis.tier.flow_high_threshold", 0.3)
        self.trend_high = config.get("analysis.tier.trend_high_threshold", 50)

        # 계산기 초기화
        self.flow_calc = FlowCalculator()
        self.breadth_calc = BreadthCalculator()
        self.trend_calc = TrendCalculator()

    def classify(
        self,
        sector_name: str,
        flow_data: list[dict],
        breadth_data: list[dict],
        trend_data: list[dict]
    ) -> SectorAnalysisResult:
        """
        섹터 Tier 분류

        Args:
            sector_name: 섹터명
            flow_data: 수급 데이터 [{stock_code, foreign_net, institution_net, market_cap}]
            breadth_data: 결속력 데이터 [{stock_code, close_price, ma20}]
            trend_data: 추세 데이터 [{stock_code, ma5, ma20, ma60, ma120, momentum_20d, rsi}]

        Returns:
            SectorAnalysisResult
        """
        # 1. 각 지표 계산
        flow_result = self.flow_calc.calculate_sector(sector_name, flow_data)
        breadth_result = self.breadth_calc.calculate_sector(sector_name, breadth_data)
        trend_result = self.trend_calc.calculate_sector(sector_name, trend_data)

        # 2. 수준 분류
        flow_level = self.flow_calc.classify_flow_level(flow_result.s_flow)
        breadth_level = breadth_result.breadth_level
        trend_level = trend_result.trend_level

        # 3. Tier 결정
        tier, warnings = self._determine_tier(
            flow_result=flow_result,
            trend_result=trend_result,
            flow_level=flow_level,
            trend_level=trend_level
        )

        # 4. 결속력 경고 추가
        if breadth_level == "WEAK":
            warnings.append(f"결속력 약함 ({breadth_result.s_breadth:.1f}%)")

        # 5. 투자 코멘트 생성
        comment = self._generate_comment(tier, flow_level, trend_level, warnings)

        return SectorAnalysisResult(
            sector_name=sector_name,
            tier=tier,
            flow_result=flow_result,
            breadth_result=breadth_result,
            trend_result=trend_result,
            s_flow=flow_result.s_flow,
            s_breadth=breadth_result.s_breadth,
            s_trend=trend_result.s_trend,
            flow_level=flow_level,
            breadth_level=breadth_level,
            trend_level=trend_level,
            is_leader_distorted=flow_result.is_leader_distorted,
            warnings=warnings,
            comment=comment
        )

    def classify_simple(
        self,
        sector_name: str,
        s_flow: float,
        s_breadth: float,
        s_trend: float,
        is_leader_distorted: bool = False
    ) -> SectorAnalysisResult:
        """
        이미 계산된 지표로 간단히 분류

        Args:
            sector_name: 섹터명
            s_flow: 수급 강도
            s_breadth: 결속력
            s_trend: 추세 점수
            is_leader_distorted: 대장주 착시 여부

        Returns:
            SectorAnalysisResult
        """
        flow_level = self.flow_calc.classify_flow_level(s_flow)
        breadth_level = self.breadth_calc.classify_breadth_level(s_breadth)
        trend_level = self.trend_calc.classify_trend_level(s_trend)

        # Tier 결정 (간소화)
        tier, warnings = self._determine_tier_simple(
            s_flow=s_flow,
            s_trend=s_trend,
            flow_level=flow_level,
            trend_level=trend_level,
            is_leader_distorted=is_leader_distorted
        )

        if breadth_level == "WEAK":
            warnings.append(f"결속력 약함 ({s_breadth:.1f}%)")

        comment = self._generate_comment(tier, flow_level, trend_level, warnings)

        # 빈 결과 객체 생성 (간소화 버전)
        empty_flow = SectorFlowResult(
            sector_name=sector_name,
            s_flow=s_flow,
            s_flow_ex_leader=s_flow,
            stock_flows=[],
            leader_stock=None,
            is_leader_distorted=is_leader_distorted
        )
        empty_breadth = SectorBreadthResult(
            sector_name=sector_name,
            s_breadth=s_breadth,
            above_count=0,
            total_count=0,
            stock_results=[],
            breadth_level=breadth_level
        )
        empty_trend = SectorTrendResult(
            sector_name=sector_name,
            s_trend=s_trend,
            trend_level=trend_level,
            aligned_ratio=0,
            avg_momentum=0,
            overheated_count=0,
            stock_results=[]
        )

        return SectorAnalysisResult(
            sector_name=sector_name,
            tier=tier,
            flow_result=empty_flow,
            breadth_result=empty_breadth,
            trend_result=empty_trend,
            s_flow=s_flow,
            s_breadth=s_breadth,
            s_trend=s_trend,
            flow_level=flow_level,
            breadth_level=breadth_level,
            trend_level=trend_level,
            is_leader_distorted=is_leader_distorted,
            warnings=warnings,
            comment=comment
        )

    def _determine_tier(
        self,
        flow_result: SectorFlowResult,
        trend_result: SectorTrendResult,
        flow_level: str,
        trend_level: str
    ) -> tuple[SectorTier, list[str]]:
        """Tier 결정 로직"""
        warnings = []

        # 대장주 착시 검증
        if flow_result.is_leader_distorted:
            warnings.append("대장주 착시 감지 - Tier 3 강등")
            return SectorTier.TIER_3, warnings

        # 매트릭스 기반 분류
        is_flow_high = flow_level == "HIGH"
        is_trend_high = trend_level in ["STRONG", "MODERATE"]

        if is_flow_high and not is_trend_high:
            # 수급 빈집 - 선취매 기회
            return SectorTier.TIER_1, warnings

        elif is_flow_high and is_trend_high:
            # 주도 섹터 - 눌림목 대기
            return SectorTier.TIER_2, warnings

        elif not is_flow_high and is_trend_high:
            # 가짜 상승 - 진입 금지
            warnings.append("수급 없이 차트만 상승 - 주의")
            return SectorTier.TIER_3, warnings

        else:
            # 무관심 - 스킵
            return SectorTier.SKIP, warnings

    def _determine_tier_simple(
        self,
        s_flow: float,
        s_trend: float,
        flow_level: str,
        trend_level: str,
        is_leader_distorted: bool
    ) -> tuple[SectorTier, list[str]]:
        """간단한 Tier 결정"""
        warnings = []

        if is_leader_distorted:
            warnings.append("대장주 착시 감지 - Tier 3 강등")
            return SectorTier.TIER_3, warnings

        is_flow_high = flow_level == "HIGH"
        is_trend_high = trend_level in ["STRONG", "MODERATE"]

        if is_flow_high and not is_trend_high:
            return SectorTier.TIER_1, warnings
        elif is_flow_high and is_trend_high:
            return SectorTier.TIER_2, warnings
        elif not is_flow_high and is_trend_high:
            warnings.append("수급 없이 차트만 상승 - 주의")
            return SectorTier.TIER_3, warnings
        else:
            return SectorTier.SKIP, warnings

    def _generate_comment(
        self,
        tier: SectorTier,
        flow_level: str,
        trend_level: str,
        warnings: list[str]
    ) -> str:
        """투자 코멘트 생성"""
        comments = {
            SectorTier.TIER_1: "수급 빈집 - 돈은 들어오는데 차트는 아직. 선취매 기회",
            SectorTier.TIER_2: "주도 섹터 - 돈도 들어오고 차트도 좋음. 눌림목 대기",
            SectorTier.TIER_3: "가짜 상승 - 대장주만 상승 또는 수급 없는 상승. 진입 금지",
            SectorTier.SKIP: "무관심 - 수급도 없고 추세도 없음. 관망",
        }

        base = comments.get(tier, "분류 불가")

        if warnings:
            base += f" ({'; '.join(warnings)})"

        return base

    def rank_sectors(
        self,
        results: list[SectorAnalysisResult]
    ) -> list[SectorAnalysisResult]:
        """
        섹터 순위 정렬

        Tier 1 > Tier 2 순서, 같은 Tier 내에서는 S_Flow 높은 순

        Args:
            results: 분석 결과 리스트

        Returns:
            정렬된 리스트
        """
        # Tier 1, 2만 필터링
        tier_1_2 = [r for r in results if r.tier in [SectorTier.TIER_1, SectorTier.TIER_2]]

        # Tier 우선, 그 다음 S_Flow 높은 순
        sorted_results = sorted(
            tier_1_2,
            key=lambda x: (x.tier.value, -x.s_flow)
        )

        return sorted_results

    def get_actionable_sectors(
        self,
        results: list[SectorAnalysisResult]
    ) -> dict[str, list[SectorAnalysisResult]]:
        """
        투자 가능 섹터 분류

        Args:
            results: 분석 결과 리스트

        Returns:
            {
                "선취매": [Tier 1 섹터들],
                "눌림목대기": [Tier 2 섹터들],
                "주의": [Tier 3 섹터들],
                "관망": [SKIP 섹터들]
            }
        """
        return {
            "선취매": [r for r in results if r.tier == SectorTier.TIER_1],
            "눌림목대기": [r for r in results if r.tier == SectorTier.TIER_2],
            "주의": [r for r in results if r.tier == SectorTier.TIER_3],
            "관망": [r for r in results if r.tier == SectorTier.SKIP],
        }
