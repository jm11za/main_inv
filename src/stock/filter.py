"""
④ 종목 필터링 모듈

Track A/B별 필터 조건 적용
- Track A (Hard Filter): 실적형 - 영업이익, 부채비율, PBR, 거래대금
- Track B (Soft Filter): 성장형 - 자본잠식, 유동비율, 거래대금 (완화)
"""
from dataclasses import dataclass, field
from typing import Any

from src.core.logger import get_logger
from src.core.config import get_config
from src.core.interfaces import SectorType, TrackType


@dataclass
class StockFilterResult:
    """종목 필터 결과"""
    stock_code: str
    stock_name: str
    track_type: TrackType
    passed: bool
    reason: str  # 통과/탈락 사유

    # 검사된 지표
    metrics: dict = field(default_factory=dict)

    # 세부 조건 결과
    conditions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "stock_code": self.stock_code,
            "stock_name": self.stock_name,
            "track_type": self.track_type.value,
            "passed": self.passed,
            "reason": self.reason,
            "metrics": self.metrics,
            "conditions": self.conditions,
        }


class StockFilter:
    """
    ④ 종목 필터

    Track A/B별 필터 조건 적용:

    Track A (Hard Filter) - 실적형:
    - 영업이익 4Q > 0 (필수)
    - 부채비율 < 200%
    - PBR < 3.0
    - 거래대금 > 10억

    Track B (Soft Filter) - 성장형:
    - 자본잠식률 < 50%
    - 유동비율 > 100%
    - 거래대금 > 5억 (완화)

    사용법:
        filter = StockFilter()

        # 단일 종목 필터
        result = filter.apply(stock_data, track_type=TrackType.TRACK_A)

        # 배치 필터
        results = filter.apply_batch(stocks_data, sector_type=SectorType.TYPE_A)
    """

    # Track A 조건 (Hard Filter)
    TRACK_A_CONDITIONS = {
        "operating_profit_4q": {
            "name": "영업이익 4Q",
            "check": lambda v: v > 0,
            "threshold": 0,
            "unit": "원",
            "message_pass": "흑자 유지",
            "message_fail": "적자 (영업이익 4Q <= 0)",
            "required": True,
        },
        "debt_ratio": {
            "name": "부채비율",
            "check": lambda v: v < 200,
            "threshold": 200,
            "unit": "%",
            "message_pass": "부채비율 양호",
            "message_fail": "부채비율 과다 ({value:.0f}% >= 200%)",
            "required": False,
        },
        "pbr": {
            "name": "PBR",
            "check": lambda v: v < 3.0,
            "threshold": 3.0,
            "unit": "배",
            "message_pass": "밸류에이션 양호",
            "message_fail": "PBR 과열 ({value:.2f} >= 3.0)",
            "required": False,
        },
        "avg_trading_value": {
            "name": "거래대금",
            "check": lambda v: v >= 1_000_000_000,  # 10억
            "threshold": 1_000_000_000,
            "unit": "원",
            "message_pass": "유동성 충분",
            "message_fail": "거래대금 부족 ({value_bil:.1f}억 < 10억)",
            "required": False,
        },
    }

    # Track B 조건 (Soft Filter)
    TRACK_B_CONDITIONS = {
        "capital_impairment": {
            "name": "자본잠식률",
            "check": lambda v: v < 50,
            "threshold": 50,
            "unit": "%",
            "message_pass": "자본 건전",
            "message_fail": "자본잠식 위험 ({value:.0f}% >= 50%)",
            "required": True,
        },
        "current_ratio": {
            "name": "유동비율",
            "check": lambda v: v >= 100,
            "threshold": 100,
            "unit": "%",
            "message_pass": "유동성 양호",
            "message_fail": "유동비율 부족 ({value:.0f}% < 100%)",
            "required": True,
        },
        "avg_trading_value": {
            "name": "거래대금",
            "check": lambda v: v >= 500_000_000,  # 5억 (완화)
            "threshold": 500_000_000,
            "unit": "원",
            "message_pass": "유동성 충분",
            "message_fail": "거래대금 부족 ({value_bil:.1f}억 < 5억)",
            "required": False,
        },
    }

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.config = get_config()

    def apply(
        self,
        stock_data: dict,
        track_type: TrackType,
    ) -> StockFilterResult:
        """
        단일 종목 필터 적용

        Args:
            stock_data: 종목 데이터
                - stock_code: 종목코드
                - stock_name: 종목명
                - operating_profit_4q: 영업이익 4Q 합산
                - debt_ratio: 부채비율 (%)
                - pbr: PBR
                - avg_trading_value: 평균 거래대금 (원)
                - capital_impairment: 자본잠식률 (%)
                - current_ratio: 유동비율 (%)
            track_type: 트랙 타입

        Returns:
            StockFilterResult
        """
        stock_code = stock_data.get("stock_code", "")
        stock_name = stock_data.get("stock_name", "")

        # 조건 선택
        if track_type == TrackType.TRACK_A:
            conditions = self.TRACK_A_CONDITIONS
        else:
            conditions = self.TRACK_B_CONDITIONS

        # 조건 검사
        passed = True
        fail_reasons = []
        condition_results = []

        for key, cond in conditions.items():
            value = stock_data.get(key, 0) or 0

            # 검사
            check_passed = cond["check"](value)

            # 결과 기록
            condition_results.append({
                "name": cond["name"],
                "value": value,
                "threshold": cond["threshold"],
                "passed": check_passed,
                "required": cond["required"],
            })

            # 실패 시
            if not check_passed:
                # 필수 조건이면 전체 실패
                if cond["required"]:
                    passed = False

                # 실패 메시지 생성
                msg = cond["message_fail"].format(
                    value=value,
                    value_bil=value / 1e8 if "trading_value" in key else value,
                )
                fail_reasons.append(msg)

                # 필수가 아니어도 실패 표시 (경고)
                if not cond["required"]:
                    passed = False  # 이 조건에서도 실패 처리

        # 결과 생성
        if passed:
            reason = "모든 조건 충족"
        else:
            reason = " / ".join(fail_reasons[:2])  # 최대 2개 사유

        return StockFilterResult(
            stock_code=stock_code,
            stock_name=stock_name,
            track_type=track_type,
            passed=passed,
            reason=reason,
            metrics={
                key: stock_data.get(key, 0)
                for key in conditions.keys()
            },
            conditions=condition_results,
        )

    def apply_batch(
        self,
        stocks_data: list[dict],
        sector_type: SectorType | None = None,
        track_type: TrackType | None = None,
        progress_callback=None,
    ) -> list[StockFilterResult]:
        """
        배치 필터 적용

        Args:
            stocks_data: 종목 데이터 리스트
            sector_type: 섹터 타입 (지정 시 자동 Track 결정)
            track_type: 트랙 타입 (직접 지정)
            progress_callback: 진행 콜백

        Returns:
            StockFilterResult 리스트
        """
        results = []

        for i, stock_data in enumerate(stocks_data):
            # Track 결정
            if track_type:
                track = track_type
            elif sector_type:
                track = TrackType.TRACK_B if sector_type == SectorType.TYPE_B else TrackType.TRACK_A
            else:
                track = TrackType.TRACK_A

            result = self.apply(stock_data, track)
            results.append(result)

            if progress_callback and (i + 1) % 50 == 0:
                progress_callback(i + 1, len(stocks_data))

        return results

    def get_passed(self, results: list[StockFilterResult]) -> list[StockFilterResult]:
        """통과 종목만 반환"""
        return [r for r in results if r.passed]

    def get_failed(self, results: list[StockFilterResult]) -> list[StockFilterResult]:
        """탈락 종목만 반환"""
        return [r for r in results if not r.passed]

    def summarize(self, results: list[StockFilterResult]) -> dict:
        """필터 결과 요약"""
        passed = self.get_passed(results)
        failed = self.get_failed(results)

        # 탈락 사유 집계
        fail_reasons = {}
        for r in failed:
            reason_key = r.reason.split(" /")[0].split("(")[0].strip()
            fail_reasons[reason_key] = fail_reasons.get(reason_key, 0) + 1

        return {
            "total": len(results),
            "passed": len(passed),
            "failed": len(failed),
            "pass_rate": len(passed) / len(results) * 100 if results else 0,
            "fail_reasons": fail_reasons,
        }

    def get_conditions(self, track_type: TrackType) -> dict:
        """트랙별 조건 반환"""
        if track_type == TrackType.TRACK_A:
            return {
                k: {
                    "name": v["name"],
                    "threshold": v["threshold"],
                    "unit": v["unit"],
                    "required": v["required"],
                }
                for k, v in self.TRACK_A_CONDITIONS.items()
            }
        else:
            return {
                k: {
                    "name": v["name"],
                    "threshold": v["threshold"],
                    "unit": v["unit"],
                    "required": v["required"],
                }
                for k, v in self.TRACK_B_CONDITIONS.items()
            }
