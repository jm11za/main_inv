"""
③ 테마 우선순위 결정 모듈 (v2.0)

테마별 집계 데이터 + LLM 분석으로 우선순위 결정
- Type A: 실적 전망 평가
- Type B: 모멘텀/이슈 강도 평가
- 하위 테마 배제

변경사항 (v2.0):
- SectorCategory Enum 삭제 → 테마명 문자열 그대로 사용
- 섹터 → 테마 용어 통일
"""
from dataclasses import dataclass, field
from typing import Any

from src.core.logger import get_logger
from src.core.config import get_config
from src.core.interfaces import SectorType


@dataclass
class ThemeMetrics:
    """테마 집계 지표"""
    theme_name: str             # 네이버 테마명 그대로
    sector_type: SectorType     # TYPE_A or TYPE_B

    # 종목 수
    stock_count: int = 0

    # 등락률 (네이버 테마 기준)
    change_rate: float = 0.0

    # 수급 지표
    s_flow: float = 0.0         # (외인순매수 + 기관순매수*1.2) / 시총 평균
    s_flow_rank: int = 0

    # 결속력 지표
    s_breadth: float = 0.0      # 종가 > MA20 비율
    s_breadth_rank: int = 0

    # 추세 지표
    s_trend: float = 0.0        # 정배열 + 모멘텀 - 과열
    s_trend_rank: int = 0

    # 재무 집계 (Type A용)
    avg_operating_profit_yoy: float = 0.0   # 평균 영업이익 YoY
    positive_profit_ratio: float = 0.0      # 흑자 종목 비율

    # 이슈 강도 (Type B용)
    news_count: int = 0                     # 뉴스 노출 횟수
    hot_keywords: list[str] = field(default_factory=list)   # 핫 키워드

    def to_dict(self) -> dict:
        return {
            "theme_name": self.theme_name,
            "sector_type": self.sector_type.value,
            "stock_count": self.stock_count,
            "change_rate": round(self.change_rate, 2),
            "s_flow": round(self.s_flow, 4),
            "s_breadth": round(self.s_breadth, 2),
            "s_trend": round(self.s_trend, 2),
            "avg_operating_profit_yoy": round(self.avg_operating_profit_yoy, 2),
            "positive_profit_ratio": round(self.positive_profit_ratio, 2),
            "news_count": self.news_count,
            "hot_keywords": self.hot_keywords,
        }


# 하위 호환성을 위한 별칭
SectorMetrics = ThemeMetrics


@dataclass
class SectorPriorityResult:
    """테마 우선순위 결과"""
    theme_name: str             # 네이버 테마명 그대로
    sector_type: SectorType
    rank: int                   # 순위 (1부터)
    score: float                # 종합 점수 (0~100)
    is_selected: bool           # 상위 테마로 선정 여부
    is_excluded: bool           # 배제 여부

    # 세부 점수
    flow_score: float = 0.0
    breadth_score: float = 0.0
    trend_score: float = 0.0
    fundamental_score: float = 0.0  # Type A용
    momentum_score: float = 0.0     # Type B용

    # LLM 분석 결과
    llm_outlook: str = ""           # 전망 (긍정/중립/부정)
    llm_comment: str = ""           # LLM 코멘트
    llm_confidence: float = 0.0

    # 배제 사유
    exclude_reason: str = ""

    # 원본 지표
    metrics: ThemeMetrics | None = None

    def to_dict(self) -> dict:
        return {
            "theme_name": self.theme_name,
            "sector_type": self.sector_type.value,
            "rank": self.rank,
            "score": round(self.score, 2),
            "is_selected": self.is_selected,
            "is_excluded": self.is_excluded,
            "flow_score": round(self.flow_score, 2),
            "breadth_score": round(self.breadth_score, 2),
            "trend_score": round(self.trend_score, 2),
            "fundamental_score": round(self.fundamental_score, 2),
            "momentum_score": round(self.momentum_score, 2),
            "llm_outlook": self.llm_outlook,
            "llm_comment": self.llm_comment,
            "exclude_reason": self.exclude_reason,
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }


class SectorPrioritizer:
    """
    ③ 테마 우선순위 결정기 (v2.0)

    테마별 집계 데이터 + LLM 분석으로 우선순위 결정:
    - Type A 가중치: S_Flow(40%) + 실적(40%) + S_Breadth(20%)
    - Type B 가중치: S_Trend(40%) + 이슈강도(40%) + S_Flow(20%)

    사용법:
        prioritizer = SectorPrioritizer(llm_client)

        # 우선순위 결정
        results = prioritizer.prioritize(
            theme_metrics_list,
            top_n=5
        )

        # 상위 테마만 추출
        selected = prioritizer.get_selected_themes(results)
    """

    # Type별 가중치
    TYPE_A_WEIGHTS = {
        "s_flow": 0.40,
        "fundamental": 0.40,
        "s_breadth": 0.20,
    }

    TYPE_B_WEIGHTS = {
        "s_trend": 0.40,
        "momentum": 0.40,
        "s_flow": 0.20,
    }

    def __init__(self, llm_client=None, use_llm: bool = True):
        """
        Args:
            llm_client: LLM 클라이언트
            use_llm: LLM 사용 여부
        """
        self.logger = get_logger(self.__class__.__name__)
        self.config = get_config()
        self._llm_client = llm_client
        self.use_llm = use_llm

    def set_llm_client(self, client):
        """LLM 클라이언트 설정"""
        self._llm_client = client

    def prioritize(
        self,
        theme_metrics: list[ThemeMetrics],
        top_n: int = 5,
        min_score: float = 30.0,
    ) -> list[SectorPriorityResult]:
        """
        테마 우선순위 결정

        Args:
            theme_metrics: 테마별 집계 지표
            top_n: 선정할 상위 테마 수
            min_score: 최소 점수 (이하는 배제)

        Returns:
            SectorPriorityResult 리스트 (순위순)
        """
        self.logger.info(f"{len(theme_metrics)}개 테마 우선순위 결정 시작")

        # 1. 기본 점수 계산
        results = []
        for metrics in theme_metrics:
            result = self._calculate_score(metrics)
            results.append(result)

        # 2. LLM 분석 (선택적)
        if self.use_llm and self._llm_client:
            self._apply_llm_analysis(results)

        # 3. 최종 순위 결정
        results = sorted(results, key=lambda x: x.score, reverse=True)

        # 4. 순위 및 선정/배제 결정
        for i, result in enumerate(results):
            result.rank = i + 1

            # 배제 조건
            if result.score < min_score:
                result.is_excluded = True
                result.exclude_reason = f"최소 점수 미달 ({result.score:.1f} < {min_score})"
            elif result.metrics and result.metrics.stock_count < 3:
                result.is_excluded = True
                result.exclude_reason = f"종목 수 부족 ({result.metrics.stock_count}개)"
            elif result.llm_outlook == "부정" and result.llm_confidence > 0.7:
                result.is_excluded = True
                result.exclude_reason = f"LLM 부정 전망: {result.llm_comment[:50]}"
            else:
                result.is_excluded = False

            # 상위 N개 선정
            result.is_selected = (result.rank <= top_n and not result.is_excluded)

        self.logger.info(
            f"우선순위 결정 완료: 선정 {sum(1 for r in results if r.is_selected)}개, "
            f"배제 {sum(1 for r in results if r.is_excluded)}개"
        )

        return results

    def _calculate_score(self, metrics: ThemeMetrics) -> SectorPriorityResult:
        """기본 점수 계산"""
        sector_type = metrics.sector_type

        if sector_type == SectorType.TYPE_A:
            # Type A: S_Flow(40%) + 실적(40%) + S_Breadth(20%)
            flow_score = self._normalize_score(metrics.s_flow, -1, 5) * 100
            fundamental_score = self._calculate_fundamental_score(metrics)
            breadth_score = metrics.s_breadth  # 이미 0~100

            score = (
                flow_score * self.TYPE_A_WEIGHTS["s_flow"] +
                fundamental_score * self.TYPE_A_WEIGHTS["fundamental"] +
                breadth_score * self.TYPE_A_WEIGHTS["s_breadth"]
            )

            return SectorPriorityResult(
                theme_name=metrics.theme_name,
                sector_type=sector_type,
                rank=0,
                score=score,
                is_selected=False,
                is_excluded=False,
                flow_score=flow_score,
                breadth_score=breadth_score,
                fundamental_score=fundamental_score,
                metrics=metrics,
            )

        else:
            # Type B: S_Trend(40%) + 이슈강도(40%) + S_Flow(20%)
            trend_score = self._normalize_score(metrics.s_trend, 0, 100) * 100
            momentum_score = self._calculate_momentum_score(metrics)
            flow_score = self._normalize_score(metrics.s_flow, -1, 5) * 100

            score = (
                trend_score * self.TYPE_B_WEIGHTS["s_trend"] +
                momentum_score * self.TYPE_B_WEIGHTS["momentum"] +
                flow_score * self.TYPE_B_WEIGHTS["s_flow"]
            )

            return SectorPriorityResult(
                theme_name=metrics.theme_name,
                sector_type=sector_type,
                rank=0,
                score=score,
                is_selected=False,
                is_excluded=False,
                flow_score=flow_score,
                trend_score=trend_score,
                momentum_score=momentum_score,
                metrics=metrics,
            )

    def _normalize_score(self, value: float, min_val: float, max_val: float) -> float:
        """값을 0~1로 정규화"""
        if max_val == min_val:
            return 0.5
        return max(0, min(1, (value - min_val) / (max_val - min_val)))

    def _calculate_fundamental_score(self, metrics: ThemeMetrics) -> float:
        """Type A용 펀더멘탈 점수"""
        # 영업이익 YoY 점수 (50%)
        profit_yoy_score = self._normalize_score(
            metrics.avg_operating_profit_yoy, -20, 50
        ) * 100

        # 흑자 비율 점수 (50%)
        profit_ratio_score = metrics.positive_profit_ratio * 100

        return profit_yoy_score * 0.5 + profit_ratio_score * 0.5

    def _calculate_momentum_score(self, metrics: ThemeMetrics) -> float:
        """Type B용 모멘텀 점수"""
        # 뉴스 노출 점수 (60%)
        news_score = self._normalize_score(metrics.news_count, 0, 100) * 100

        # 핫 키워드 점수 (40%)
        keyword_score = min(len(metrics.hot_keywords) * 20, 100)

        return news_score * 0.6 + keyword_score * 0.4

    def _apply_llm_analysis(self, results: list[SectorPriorityResult]):
        """LLM으로 테마별 전망 분석"""
        if not self._llm_client:
            return

        for result in results:
            try:
                self._analyze_theme_outlook(result)
            except Exception as e:
                self.logger.debug(f"LLM 분석 실패 ({result.theme_name}): {e}")

    def _analyze_theme_outlook(self, result: SectorPriorityResult):
        """단일 테마 LLM 전망 분석"""
        metrics = result.metrics
        if not metrics:
            return

        # Type에 따라 다른 프롬프트
        if result.sector_type == SectorType.TYPE_A:
            prompt = self._build_type_a_prompt(result)
        else:
            prompt = self._build_type_b_prompt(result)

        try:
            llm_result = self._llm_client.generate(prompt).strip()

            # 결과 파싱
            outlook = "중립"
            comment = llm_result

            llm_lower = llm_result.lower()
            if "긍정" in llm_lower or "positive" in llm_lower or "좋" in llm_lower:
                outlook = "긍정"
                # 긍정이면 점수 가산
                result.score += 5
            elif "부정" in llm_lower or "negative" in llm_lower or "나쁨" in llm_lower or "위험" in llm_lower:
                outlook = "부정"
                # 부정이면 점수 감산
                result.score -= 10

            result.llm_outlook = outlook
            result.llm_comment = comment[:200]
            result.llm_confidence = 0.8

        except Exception as e:
            self.logger.debug(f"LLM 전망 분석 실패: {e}")

    def _build_type_a_prompt(self, result: SectorPriorityResult) -> str:
        """Type A 테마용 LLM 프롬프트"""
        metrics = result.metrics
        return f"""한국 주식시장에서 "{result.theme_name}" 테마의 실적 전망을 분석해.

이 테마는 Type A (실적 기반)로, "숫자가 찍혀야 주가가 간다".

[현재 지표]
- 종목 수: {metrics.stock_count}개
- 수급 (S_Flow): {metrics.s_flow:.2%}
- 평균 영업이익 YoY: {metrics.avg_operating_profit_yoy:.1f}%
- 흑자 종목 비율: {metrics.positive_profit_ratio:.1%}

이 테마의 향후 실적 전망은?
긍정/중립/부정 중 하나로 답하고, 한 문장으로 이유를 설명해."""

    def _build_type_b_prompt(self, result: SectorPriorityResult) -> str:
        """Type B 테마용 LLM 프롬프트"""
        metrics = result.metrics
        keywords = ", ".join(metrics.hot_keywords[:5]) if metrics.hot_keywords else "없음"

        return f"""한국 주식시장에서 "{result.theme_name}" 테마의 모멘텀을 분석해.

이 테마는 Type B (성장 기반)로, "꿈을 먹고 주가가 간다".

[현재 지표]
- 종목 수: {metrics.stock_count}개
- 추세 (S_Trend): {metrics.s_trend:.1f}
- 뉴스 노출: {metrics.news_count}건
- 핫 키워드: {keywords}

이 테마의 모멘텀/이슈 강도는?
긍정/중립/부정 중 하나로 답하고, 한 문장으로 이유를 설명해."""

    def get_selected_themes(
        self,
        results: list[SectorPriorityResult]
    ) -> list[SectorPriorityResult]:
        """선정된 상위 테마만 반환"""
        return [r for r in results if r.is_selected]

    # 하위 호환성 별칭
    def get_selected_sectors(
        self,
        results: list[SectorPriorityResult]
    ) -> list[SectorPriorityResult]:
        """선정된 상위 테마만 반환 (deprecated: get_selected_themes 사용)"""
        return self.get_selected_themes(results)

    def get_excluded_themes(
        self,
        results: list[SectorPriorityResult]
    ) -> list[SectorPriorityResult]:
        """배제된 테마만 반환"""
        return [r for r in results if r.is_excluded]

    # 하위 호환성 별칭
    def get_excluded_sectors(
        self,
        results: list[SectorPriorityResult]
    ) -> list[SectorPriorityResult]:
        """배제된 테마만 반환 (deprecated: get_excluded_themes 사용)"""
        return self.get_excluded_themes(results)

    def summarize(self, results: list[SectorPriorityResult]) -> dict:
        """우선순위 결과 요약"""
        selected = self.get_selected_themes(results)
        excluded = self.get_excluded_themes(results)

        return {
            "total_themes": len(results),
            "selected_count": len(selected),
            "excluded_count": len(excluded),
            "selected_themes": [
                {
                    "rank": r.rank,
                    "theme_name": r.theme_name,
                    "type": r.sector_type.value,
                    "score": round(r.score, 1),
                    "outlook": r.llm_outlook,
                }
                for r in selected
            ],
            "excluded_themes": [
                {
                    "theme_name": r.theme_name,
                    "reason": r.exclude_reason,
                }
                for r in excluded
            ],
            "avg_score": sum(r.score for r in results) / len(results) if results else 0,
        }
