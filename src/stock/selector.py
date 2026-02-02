"""
④ 후보 종목 선정 모듈 (v2.0)

필터링 + 점수화를 결합하여 테마별 상위 종목 선정

변경사항 (v2.0):
- SectorCategory Enum 삭제 → 테마명 문자열 사용
- 섹터 → 테마 용어 통일
"""
from dataclasses import dataclass, field
from typing import Any

from src.core.logger import get_logger
from src.core.config import get_config
from src.core.interfaces import SectorType, TrackType
from src.stock.filter import StockFilter, StockFilterResult
from src.stock.scorer import StockScorer, StockScoreResult


@dataclass
class CandidateResult:
    """후보 종목 결과"""
    stock_code: str
    stock_name: str
    sector: str                 # 테마명 문자열 (v2.0)
    sector_rank: int            # 해당 테마의 순위
    track_type: TrackType

    # 필터 결과
    filter_passed: bool
    filter_reason: str

    # 점수 결과
    financial_score: float
    technical_score: float
    total_score: float
    overall_rank: int           # 전체 순위

    # 선정 여부
    is_selected: bool
    selection_reason: str

    def to_dict(self) -> dict:
        return {
            "stock_code": self.stock_code,
            "stock_name": self.stock_name,
            "sector": self.sector,  # 문자열 그대로
            "sector_rank": self.sector_rank,
            "track_type": self.track_type.value,
            "filter_passed": self.filter_passed,
            "filter_reason": self.filter_reason,
            "financial_score": round(self.financial_score, 1),
            "technical_score": round(self.technical_score, 1),
            "total_score": round(self.total_score, 1),
            "overall_rank": self.overall_rank,
            "is_selected": self.is_selected,
            "selection_reason": self.selection_reason,
        }


class CandidateSelector:
    """
    ④ 후보 종목 선정기 (v2.0)

    상위 테마 내에서 투자 후보 종목 선정:
    1. Track A/B 필터링
    2. 종목 점수화
    3. 테마별 상위 N개 선정

    사용법:
        selector = CandidateSelector()

        # 후보 선정
        results = selector.select(
            stocks_data=stocks_data,  # 상위 테마 종목 데이터
            sector_type_map=theme_type_map,  # 테마별 Type (테마명 문자열 기준)
            top_per_sector=3,  # 테마당 선정 수
            max_total=15,  # 최대 총 선정 수
        )

        # 선정된 종목만
        selected = selector.get_selected(results)
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.config = get_config()
        self.filter = StockFilter()
        self.scorer = StockScorer()

    def select(
        self,
        stocks_data: list[dict],
        sector_type_map: dict[str, SectorType],  # {테마명: SectorType}
        top_per_sector: int = 3,
        max_total: int = 15,
        progress_callback=None,
    ) -> list[CandidateResult]:
        """
        후보 종목 선정

        Args:
            stocks_data: 종목 데이터 리스트
                - stock_code: 종목코드
                - stock_name: 종목명
                - sector: 테마명 (문자열)
                - (필터/점수화에 필요한 재무/기술 데이터)
            sector_type_map: 테마별 Type 매핑 ({테마명: SectorType})
            top_per_sector: 테마당 선정 수
            max_total: 최대 총 선정 수
            progress_callback: 진행 콜백

        Returns:
            CandidateResult 리스트
        """
        self.logger.info(f"{len(stocks_data)}개 종목에서 후보 선정 시작")

        # 테마별로 그룹핑
        theme_groups: dict[str, list[dict]] = {}
        for stock in stocks_data:
            theme_name = stock.get("sector", "기타")  # 테마명 문자열
            if theme_name not in theme_groups:
                theme_groups[theme_name] = []
            theme_groups[theme_name].append(stock)

        all_results = []

        # 테마별 처리
        for theme_name, stocks in theme_groups.items():
            sector_type = sector_type_map.get(theme_name, SectorType.TYPE_A)
            track_type = (
                TrackType.TRACK_B if sector_type == SectorType.TYPE_B
                else TrackType.TRACK_A
            )

            # 필터링
            filter_results = self.filter.apply_batch(stocks, track_type=track_type)

            # 통과 종목만 점수화
            passed_stocks = []
            filter_map = {}
            for i, fr in enumerate(filter_results):
                filter_map[fr.stock_code] = fr
                if fr.passed:
                    passed_stocks.append(stocks[i])

            # 점수화
            score_results = []
            if passed_stocks:
                score_results = self.scorer.score_batch(passed_stocks, track_type)

            score_map = {sr.stock_code: sr for sr in score_results}

            # 테마 내 결과 생성
            theme_candidates = []
            for stock in stocks:
                code = stock.get("stock_code", "")
                name = stock.get("stock_name", "")

                fr = filter_map.get(code)
                sr = score_map.get(code)

                if sr:
                    candidate = CandidateResult(
                        stock_code=code,
                        stock_name=name,
                        sector=theme_name,  # 테마명 문자열
                        sector_rank=sr.rank,
                        track_type=track_type,
                        filter_passed=True,
                        filter_reason=fr.reason if fr else "",
                        financial_score=sr.financial_score,
                        technical_score=sr.technical_score,
                        total_score=sr.total_score,
                        overall_rank=0,
                        is_selected=False,
                        selection_reason="",
                    )
                else:
                    candidate = CandidateResult(
                        stock_code=code,
                        stock_name=name,
                        sector=theme_name,  # 테마명 문자열
                        sector_rank=0,
                        track_type=track_type,
                        filter_passed=False,
                        filter_reason=fr.reason if fr else "필터 미통과",
                        financial_score=0,
                        technical_score=0,
                        total_score=0,
                        overall_rank=0,
                        is_selected=False,
                        selection_reason="필터 탈락",
                    )

                theme_candidates.append(candidate)

            # 테마 내 상위 N개 선정
            passed_candidates = [c for c in theme_candidates if c.filter_passed]
            passed_candidates = sorted(
                passed_candidates,
                key=lambda x: x.total_score,
                reverse=True
            )

            for i, candidate in enumerate(passed_candidates[:top_per_sector]):
                candidate.is_selected = True
                candidate.selection_reason = f"테마 내 {i+1}위"

            all_results.extend(theme_candidates)

        # 전체 순위 부여
        selected = [r for r in all_results if r.is_selected]
        selected = sorted(selected, key=lambda x: x.total_score, reverse=True)

        # 최대 개수 제한
        for i, candidate in enumerate(selected):
            if i < max_total:
                candidate.overall_rank = i + 1
            else:
                candidate.is_selected = False
                candidate.selection_reason = f"총 선정 수 초과 (>{max_total})"

        self.logger.info(
            f"후보 선정 완료: 선정 {sum(1 for r in all_results if r.is_selected)}개 / "
            f"전체 {len(all_results)}개"
        )

        return all_results

    def get_selected(self, results: list[CandidateResult]) -> list[CandidateResult]:
        """선정된 종목만 반환 (순위순)"""
        selected = [r for r in results if r.is_selected]
        return sorted(selected, key=lambda x: x.overall_rank)

    def get_by_theme(
        self,
        results: list[CandidateResult],
        theme_name: str
    ) -> list[CandidateResult]:
        """특정 테마 종목만 반환"""
        return [r for r in results if r.sector == theme_name]

    # 하위 호환성 (deprecated)
    def get_by_sector(
        self,
        results: list[CandidateResult],
        sector: str
    ) -> list[CandidateResult]:
        """특정 섹터 종목만 반환 (deprecated: get_by_theme 사용)"""
        return self.get_by_theme(results, sector)

    def summarize(self, results: list[CandidateResult]) -> dict:
        """선정 결과 요약"""
        selected = self.get_selected(results)

        # 테마별 선정 수
        theme_counts = {}
        for r in selected:
            theme_name = r.sector  # 이미 문자열
            theme_counts[theme_name] = theme_counts.get(theme_name, 0) + 1

        # 트랙별 선정 수
        track_a = [r for r in selected if r.track_type == TrackType.TRACK_A]
        track_b = [r for r in selected if r.track_type == TrackType.TRACK_B]

        return {
            "total_processed": len(results),
            "filter_passed": sum(1 for r in results if r.filter_passed),
            "filter_failed": sum(1 for r in results if not r.filter_passed),
            "selected_count": len(selected),
            "theme_distribution": theme_counts,
            # 하위 호환성
            "sector_distribution": theme_counts,
            "track_a_count": len(track_a),
            "track_b_count": len(track_b),
            "avg_score": sum(r.total_score for r in selected) / len(selected) if selected else 0,
            "selected_stocks": [
                {
                    "rank": r.overall_rank,
                    "code": r.stock_code,
                    "name": r.stock_name,
                    "theme": r.sector,  # 테마명
                    "sector": r.sector,  # 하위 호환성
                    "score": round(r.total_score, 1),
                }
                for r in selected
            ],
        }
