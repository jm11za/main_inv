"""
필터 라우터

섹터 타입에 따라 적절한 필터로 라우팅
"""
from dataclasses import dataclass
from typing import Any

from src.core.interfaces import SectorType, TrackType, FilterResult
from src.core.logger import get_logger
from src.filtering.sector_classifier import SectorClassifier
from src.filtering.track_filters import TrackAFilter, TrackBFilter, get_filter_for_track


@dataclass
class FilteredStock:
    """필터링된 종목 정보"""
    stock_code: str
    stock_name: str
    sector_name: str
    sector_type: SectorType
    track_type: TrackType
    filter_passed: bool
    filter_reason: str
    metrics: dict


class FilterRouter:
    """
    필터 라우터

    섹터를 분류하고 적절한 필터를 적용하는 통합 클래스

    사용법:
        router = FilterRouter()

        # 단일 종목 필터링
        result = router.filter_stock(stock_data, sector_name="바이오")

        # 섹터별 종목 일괄 필터링
        results = router.filter_stocks_by_sector(stocks, sector_name="AI반도체")
    """

    def __init__(self, use_llm: bool = True):
        """
        Args:
            use_llm: LLM 섹터 분류 사용 여부
        """
        self.logger = get_logger(self.__class__.__name__)
        self.classifier = SectorClassifier(use_llm=use_llm)
        self.track_a_filter = TrackAFilter()
        self.track_b_filter = TrackBFilter()

    def filter_stock(
        self,
        stock: dict,
        sector_name: str,
        sector_type: SectorType | None = None
    ) -> FilteredStock:
        """
        단일 종목 필터링

        Args:
            stock: 종목 데이터
            sector_name: 섹터명
            sector_type: 섹터 타입 (없으면 자동 분류)

        Returns:
            FilteredStock
        """
        stock_code = stock.get("stock_code", "")
        stock_name = stock.get("name", stock.get("stock_name", ""))

        # 1. 섹터 타입 결정
        if sector_type is None:
            sector_type = self.classifier.classify(sector_name)

        # 2. 트랙 타입 결정
        track_type = (
            TrackType.TRACK_A if sector_type == SectorType.TYPE_A
            else TrackType.TRACK_B
        )

        # 3. 필터 적용
        filter_instance = get_filter_for_track(track_type)
        result = filter_instance.apply(stock)

        self.logger.debug(
            f"[{stock_code}] {stock_name} - {sector_name}({sector_type.name}) "
            f"-> {track_type.name} -> {'PASS' if result.passed else 'FAIL'}"
        )

        return FilteredStock(
            stock_code=stock_code,
            stock_name=stock_name,
            sector_name=sector_name,
            sector_type=sector_type,
            track_type=track_type,
            filter_passed=result.passed,
            filter_reason=result.reason,
            metrics=result.metrics,
        )

    def filter_stocks_by_sector(
        self,
        stocks: list[dict],
        sector_name: str
    ) -> list[FilteredStock]:
        """
        동일 섹터의 여러 종목 필터링

        Args:
            stocks: 종목 데이터 리스트
            sector_name: 섹터명

        Returns:
            FilteredStock 리스트
        """
        # 섹터 타입 한 번만 분류
        sector_type = self.classifier.classify(sector_name)

        results = []
        for stock in stocks:
            result = self.filter_stock(stock, sector_name, sector_type)
            results.append(result)

        passed_count = sum(1 for r in results if r.filter_passed)
        self.logger.info(
            f"[{sector_name}] 필터링 완료: {passed_count}/{len(results)}개 통과"
        )

        return results

    def filter_all(
        self,
        sector_stocks: dict[str, list[dict]]
    ) -> dict[str, list[FilteredStock]]:
        """
        전체 섹터-종목 필터링

        Args:
            sector_stocks: {섹터명: [종목 데이터, ...], ...}

        Returns:
            {섹터명: [FilteredStock, ...], ...}
        """
        results = {}

        for sector_name, stocks in sector_stocks.items():
            results[sector_name] = self.filter_stocks_by_sector(stocks, sector_name)

        # 전체 통계
        total_stocks = sum(len(stocks) for stocks in results.values())
        total_passed = sum(
            sum(1 for r in stocks if r.filter_passed)
            for stocks in results.values()
        )

        self.logger.info(
            f"전체 필터링 완료: {total_passed}/{total_stocks}개 통과 "
            f"({total_passed/total_stocks*100:.1f}%)" if total_stocks > 0 else "종목 없음"
        )

        return results

    def get_passed_stocks(
        self,
        filtered_results: list[FilteredStock]
    ) -> list[FilteredStock]:
        """통과한 종목만 반환"""
        return [r for r in filtered_results if r.filter_passed]

    def get_failed_stocks(
        self,
        filtered_results: list[FilteredStock]
    ) -> list[FilteredStock]:
        """탈락한 종목만 반환"""
        return [r for r in filtered_results if not r.filter_passed]

    def get_summary(self, filtered_results: list[FilteredStock]) -> dict:
        """필터링 결과 요약"""
        if not filtered_results:
            return {"total": 0, "passed": 0, "failed": 0}

        passed = [r for r in filtered_results if r.filter_passed]
        failed = [r for r in filtered_results if not r.filter_passed]

        # 트랙별 통계
        track_a_count = sum(1 for r in filtered_results if r.track_type == TrackType.TRACK_A)
        track_b_count = sum(1 for r in filtered_results if r.track_type == TrackType.TRACK_B)

        return {
            "total": len(filtered_results),
            "passed": len(passed),
            "failed": len(failed),
            "pass_rate": len(passed) / len(filtered_results) * 100,
            "track_a_count": track_a_count,
            "track_b_count": track_b_count,
        }
