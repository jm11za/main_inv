"""
⑤ 최종 판정 모듈

재료 분석 + 심리 분석을 종합하여 최종 투자 판정
- STRONG_BUY: Tier1/2 + 재료 S/A + 심리 공포/의심
- BUY: 필터 PASS + 점수 >= 60 + 재료 A 이상
- WATCH: 필터 PASS + 점수 >= 50
- AVOID: 필터 FAIL 또는 심리 환희
"""
from dataclasses import dataclass, field
from typing import Any

from src.core.logger import get_logger
from src.core.config import get_config
from src.core.interfaces import Recommendation, MaterialGrade, SentimentStage, TrackType
from src.verify.material_analyzer import MaterialResult
from src.verify.sentiment_analyzer import SentimentResult


@dataclass
class FinalDecision:
    """최종 투자 판정"""
    stock_code: str
    stock_name: str
    recommendation: Recommendation  # STRONG_BUY, BUY, WATCH, AVOID

    # 입력 데이터
    sector: str  # 테마명 문자열 (v3.0)
    track_type: TrackType
    sector_rank: int
    total_score: float
    filter_passed: bool

    # 분석 결과
    material_grade: MaterialGrade
    sentiment_stage: SentimentStage
    material_confidence: float
    sentiment_confidence: float

    # 판정 근거
    decision_factors: list[str]
    risk_warnings: list[str]
    investment_thesis: str

    # 신뢰도
    confidence: float

    def to_dict(self) -> dict:
        return {
            "stock_code": self.stock_code,
            "stock_name": self.stock_name,
            "recommendation": self.recommendation.value,
            "sector": self.sector,  # 이미 문자열 (v3.0)
            "track_type": self.track_type.value,
            "sector_rank": self.sector_rank,
            "total_score": round(self.total_score, 1),
            "filter_passed": self.filter_passed,
            "material_grade": self.material_grade.value,
            "sentiment_stage": self.sentiment_stage.value,
            "decision_factors": self.decision_factors,
            "risk_warnings": self.risk_warnings,
            "investment_thesis": self.investment_thesis,
            "confidence": round(self.confidence, 2),
        }


class DecisionEngine:
    """
    ⑤ 최종 판정 엔진

    Decision Matrix:
    | 조건                              | 판정        |
    |----------------------------------|------------|
    | 섹터1~2위 + 재료S/A + 심리공포/의심  | STRONG_BUY |
    | 필터PASS + 점수>=60 + 재료A이상    | BUY        |
    | 필터PASS + 점수>=50              | WATCH      |
    | 필터FAIL 또는 심리환희            | AVOID      |

    사용법:
        engine = DecisionEngine()

        # 단일 종목 판정
        decision = engine.decide(
            stock_code="005930",
            stock_name="삼성전자",
            sector="반도체",  # 테마명 문자열 (v3.0)
            track_type=TrackType.TRACK_B,
            sector_rank=1,
            total_score=75.5,
            filter_passed=True,
            material=material_result,
            sentiment=sentiment_result,
        )

        # 배치 판정
        decisions = engine.decide_batch(...)
    """

    # STRONG_BUY 조건
    STRONG_BUY_CRITERIA = {
        "max_sector_rank": 2,
        "min_score": 40,
        "min_material_grade": [MaterialGrade.S, MaterialGrade.A],
        "allowed_sentiment": [SentimentStage.FEAR, SentimentStage.DOUBT],
    }

    # BUY 조건
    BUY_CRITERIA = {
        "min_score": 60,
        "min_material_grade": [MaterialGrade.S, MaterialGrade.A, MaterialGrade.B],
        "forbidden_sentiment": [SentimentStage.EUPHORIA],
    }

    # WATCH 조건
    WATCH_CRITERIA = {
        "min_score": 50,
    }

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.config = get_config()

    def decide(
        self,
        stock_code: str,
        stock_name: str,
        sector: str,  # 테마명 문자열 (v3.0)
        track_type: TrackType,
        sector_rank: int,
        total_score: float,
        filter_passed: bool,
        material: MaterialResult,
        sentiment: SentimentResult,
    ) -> FinalDecision:
        """
        단일 종목 최종 판정

        Args:
            stock_code: 종목코드
            stock_name: 종목명
            sector: 테마명 (문자열)
            track_type: 트랙 타입
            sector_rank: 섹터 내 순위
            total_score: 총 점수
            filter_passed: 필터 통과 여부
            material: 재료 분석 결과
            sentiment: 심리 분석 결과

        Returns:
            FinalDecision
        """
        decision_factors = []
        risk_warnings = []

        # 1. AVOID 조건 먼저 체크
        if not filter_passed:
            return self._create_decision(
                stock_code, stock_name, sector, track_type, sector_rank,
                total_score, filter_passed, material, sentiment,
                Recommendation.AVOID,
                ["필터 미통과"],
                ["기본 재무 조건 불충족"],
                "재무 안정성 미달로 투자 부적합"
            )

        if sentiment.stage == SentimentStage.EUPHORIA:
            return self._create_decision(
                stock_code, stock_name, sector, track_type, sector_rank,
                total_score, filter_passed, material, sentiment,
                Recommendation.AVOID,
                ["심리 환희 단계"],
                ["고점 리스크 높음", "대중 낙관 과열"],
                "시장 과열로 탈출 권장"
            )

        # 2. STRONG_BUY 조건 체크
        is_top_sector = sector_rank <= self.STRONG_BUY_CRITERIA["max_sector_rank"]
        is_good_material = material.grade in self.STRONG_BUY_CRITERIA["min_material_grade"]
        is_early_sentiment = sentiment.stage in self.STRONG_BUY_CRITERIA["allowed_sentiment"]
        is_min_score = total_score >= self.STRONG_BUY_CRITERIA["min_score"]

        if is_top_sector and is_good_material and is_early_sentiment and is_min_score:
            decision_factors = [
                f"상위 섹터 ({sector_rank}위)",
                f"재료 {material.grade.value}등급",
                f"심리 {sentiment.stage.value} 단계 (초기)",
                f"점수 {total_score:.1f}",
            ]

            if material.key_materials:
                decision_factors.append(f"핵심재료: {', '.join(material.key_materials[:2])}")

            risk_warnings = []
            if material.negative_factors:
                risk_warnings.append(f"부정요소: {', '.join(material.negative_factors[:2])}")

            key_mat = ', '.join(material.key_materials[:2]) if material.key_materials else "양호"
            thesis = (
                f"{stock_name}({sector} {sector_rank}위) "
                f"재료 {material.grade.value}급({key_mat}), "
                f"{sentiment.stage.value} 단계 → 적극 매수 검토"
            )

            return self._create_decision(
                stock_code, stock_name, sector, track_type, sector_rank,
                total_score, filter_passed, material, sentiment,
                Recommendation.STRONG_BUY,
                decision_factors,
                risk_warnings,
                thesis,
            )

        # 3. BUY 조건 체크
        is_high_score = total_score >= self.BUY_CRITERIA["min_score"]
        is_decent_material = material.grade in self.BUY_CRITERIA["min_material_grade"]
        not_euphoria = sentiment.stage not in self.BUY_CRITERIA["forbidden_sentiment"]

        if is_high_score and is_decent_material and not_euphoria:
            decision_factors = [
                f"높은 점수 ({total_score:.1f})",
                f"재료 {material.grade.value}등급",
                f"심리 {sentiment.stage.value}",
            ]

            risk_warnings = []
            if sentiment.stage == SentimentStage.CONVICTION:
                risk_warnings.append("심리 확신 단계 - 추격 매수 주의")
            if material.negative_factors:
                risk_warnings.append(f"부정요소: {', '.join(material.negative_factors[:2])}")

            key_mat = ', '.join(material.key_materials[:2]) if material.key_materials else "양호"
            thesis = (
                f"{stock_name}({sector}) 점수 {total_score:.0f}, "
                f"재료 {material.grade.value}급({key_mat}) → 매수 검토"
            )

            return self._create_decision(
                stock_code, stock_name, sector, track_type, sector_rank,
                total_score, filter_passed, material, sentiment,
                Recommendation.BUY,
                decision_factors,
                risk_warnings,
                thesis,
            )

        # 4. WATCH 조건 체크
        if total_score >= self.WATCH_CRITERIA["min_score"]:
            decision_factors = [
                f"점수 {total_score:.1f} (기준 이상)",
                f"심리 {sentiment.stage.value}",
            ]

            risk_warnings = []
            if material.grade == MaterialGrade.C:
                risk_warnings.append("재료 부족 - 추가 모니터링 필요")
            if total_score < 60:
                risk_warnings.append("점수 보통 - 추가 상승 동력 확인 필요")

            thesis = (
                f"{stock_name}({sector}) 점수 {total_score:.0f} - "
                f"관찰 후 추가 조건 충족 시 진입 검토"
            )

            return self._create_decision(
                stock_code, stock_name, sector, track_type, sector_rank,
                total_score, filter_passed, material, sentiment,
                Recommendation.WATCH,
                decision_factors,
                risk_warnings,
                thesis,
            )

        # 5. 그 외는 AVOID
        decision_factors = [
            f"점수 미달 ({total_score:.1f} < 50)",
        ]
        if material.grade == MaterialGrade.C:
            decision_factors.append("재료 부족")

        return self._create_decision(
            stock_code, stock_name, sector, track_type, sector_rank,
            total_score, filter_passed, material, sentiment,
            Recommendation.AVOID,
            decision_factors,
            ["전반적 지표 부족"],
            "투자 매력도 낮음"
        )

    def _create_decision(
        self,
        stock_code: str,
        stock_name: str,
        sector: str,  # 테마명 문자열 (v3.0)
        track_type: TrackType,
        sector_rank: int,
        total_score: float,
        filter_passed: bool,
        material: MaterialResult,
        sentiment: SentimentResult,
        recommendation: Recommendation,
        decision_factors: list[str],
        risk_warnings: list[str],
        investment_thesis: str,
    ) -> FinalDecision:
        """FinalDecision 객체 생성"""
        # 신뢰도 계산
        confidence = (material.confidence + sentiment.confidence) / 2
        if recommendation == Recommendation.STRONG_BUY:
            confidence = min(confidence + 0.1, 1.0)
        elif recommendation == Recommendation.AVOID:
            confidence = max(confidence - 0.1, 0.0)

        return FinalDecision(
            stock_code=stock_code,
            stock_name=stock_name,
            recommendation=recommendation,
            sector=sector,
            track_type=track_type,
            sector_rank=sector_rank,
            total_score=total_score,
            filter_passed=filter_passed,
            material_grade=material.grade,
            sentiment_stage=sentiment.stage,
            material_confidence=material.confidence,
            sentiment_confidence=sentiment.confidence,
            decision_factors=decision_factors,
            risk_warnings=risk_warnings,
            investment_thesis=investment_thesis,
            confidence=confidence,
        )

    def decide_batch(
        self,
        candidates_data: list[dict],
        material_results: list[MaterialResult],
        sentiment_results: list[SentimentResult],
        progress_callback=None,
    ) -> list[FinalDecision]:
        """
        배치 최종 판정

        Args:
            candidates_data: 후보 종목 데이터 리스트
            material_results: 재료 분석 결과 리스트
            sentiment_results: 심리 분석 결과 리스트
            progress_callback: 진행 콜백

        Returns:
            FinalDecision 리스트
        """
        self.logger.info(f"{len(candidates_data)}개 종목 최종 판정 시작")

        # 결과 맵 생성
        material_map = {r.stock_code: r for r in material_results}
        sentiment_map = {r.stock_code: r for r in sentiment_results}

        decisions = []
        for i, data in enumerate(candidates_data):
            code = data.get("stock_code", "")

            material = material_map.get(code)
            sentiment = sentiment_map.get(code)

            if not material or not sentiment:
                self.logger.warning(f"분석 결과 없음: {code}")
                continue

            decision = self.decide(
                stock_code=code,
                stock_name=data.get("stock_name", ""),
                sector=data.get("sector", "기타"),  # 테마명 문자열 (v3.0)
                track_type=data.get("track_type", TrackType.TRACK_A),
                sector_rank=data.get("sector_rank", 99),
                total_score=data.get("total_score", 0),
                filter_passed=data.get("filter_passed", False),
                material=material,
                sentiment=sentiment,
            )
            decisions.append(decision)

            if progress_callback and (i + 1) % 5 == 0:
                progress_callback(i + 1, len(candidates_data))

        self.logger.info(f"최종 판정 완료: {len(decisions)}개")
        return decisions

    def get_by_recommendation(
        self,
        decisions: list[FinalDecision],
        recommendation: Recommendation
    ) -> list[FinalDecision]:
        """특정 판정 종목만 반환"""
        return [d for d in decisions if d.recommendation == recommendation]

    def summarize(self, decisions: list[FinalDecision]) -> dict:
        """판정 결과 요약"""
        rec_counts = {r: 0 for r in Recommendation}
        for d in decisions:
            rec_counts[d.recommendation] += 1

        strong_buys = self.get_by_recommendation(decisions, Recommendation.STRONG_BUY)
        buys = self.get_by_recommendation(decisions, Recommendation.BUY)

        return {
            "total": len(decisions),
            "recommendation_distribution": {r.value: c for r, c in rec_counts.items()},
            "strong_buy_stocks": [
                {
                    "code": d.stock_code,
                    "name": d.stock_name,
                    "sector": d.sector,
                    "thesis": d.investment_thesis,
                }
                for d in strong_buys
            ],
            "buy_stocks": [
                {
                    "code": d.stock_code,
                    "name": d.stock_name,
                    "sector": d.sector,
                    "thesis": d.investment_thesis,
                }
                for d in buys
            ],
            "avoid_count": rec_counts[Recommendation.AVOID],
            "avg_confidence": sum(d.confidence for d in decisions) / len(decisions) if decisions else 0,
        }
