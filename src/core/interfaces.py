"""
핵심 인터페이스 정의

모든 레이어에서 사용하는 표준 인터페이스
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# ============================================
# Enums
# ============================================
class SectorType(Enum):
    """섹터 유형 (LLM 분류 결과)"""
    TYPE_A = "earnings_driven"  # 실적 기반
    TYPE_B = "growth_driven"    # 성장 기반


class TrackType(Enum):
    """필터/스코어링 트랙"""
    TRACK_A = "earnings_driven"  # 실적 기반 (Hard Filter, 50:50)
    TRACK_B = "growth_driven"    # 성장 기반 (Soft Filter, 20:80)


class Tier(Enum):
    """섹터 Tier"""
    TIER_1 = 1  # 수급 빈집 (선취매 기회)
    TIER_2 = 2  # 주도 섹터 (눌림목 대기)
    TIER_3 = 3  # 가짜 상승 (진입 금지)


class Recommendation(Enum):
    """최종 투자 판정"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WATCH = "WATCH"
    AVOID = "AVOID"


class MaterialGrade(Enum):
    """재료 등급 (Skeptic 분석)"""
    S = "S"  # 대형 호재
    A = "A"  # 중형 호재
    B = "B"  # 소형 호재
    C = "C"  # 재료 없음


class SentimentStage(Enum):
    """심리 단계 (Sentiment Reader 분석)"""
    FEAR = "공포"       # 바닥권
    DOUBT = "의심"      # 초기
    CONVICTION = "확신"  # 중기
    EUPHORIA = "환희"   # 고점 (위험)


# ============================================
# Data Classes
# ============================================
@dataclass
class Theme:
    """테마/섹터 데이터"""
    theme_id: str
    name: str
    theme_type: str = ""        # "실체형" or "기대형"
    sector_type: SectorType = SectorType.TYPE_A
    keywords: list[str] = field(default_factory=list)
    leader_stock_code: str = ""
    related_stocks: list[str] = field(default_factory=list)
    tier: Tier = Tier.TIER_3

    # Metrics
    s_flow: float = 0.0
    s_breadth: float = 0.0
    s_trend: float = 0.0

    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class Stock:
    """종목 데이터"""
    stock_code: str
    name: str

    # 소속 정보
    themes: list[str] = field(default_factory=list)
    track_type: TrackType = TrackType.TRACK_A

    # 재무 데이터
    operating_profit_4q: float = 0.0
    debt_ratio: float = 0.0
    pbr: float = 0.0
    current_ratio: float = 0.0
    capital_impairment: float = 0.0
    rd_ratio: float = 0.0

    # 필터 결과
    filter_passed: bool = False
    filter_reason: str = ""

    # 점수
    financial_score: float = 0.0
    technical_score: float = 0.0
    total_score: float = 0.0
    rank: int = 0

    # 최종 판정
    material_grade: MaterialGrade | None = None
    sentiment_stage: SentimentStage | None = None
    recommendation: Recommendation | None = None


@dataclass
class FilterResult:
    """필터 적용 결과"""
    passed: bool
    stock_code: str
    reason: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreResult:
    """스코어링 결과"""
    stock_code: str
    track_type: TrackType
    financial_score: float
    technical_score: float
    total_score: float
    rank: int = 0


# ============================================
# Abstract Interfaces
# ============================================
class DataFetcher(ABC):
    """데이터 수집기 인터페이스 (Layer 1)"""

    @abstractmethod
    def fetch(self) -> Any:
        """데이터 수집"""
        pass

    @abstractmethod
    def get_source_name(self) -> str:
        """데이터 소스 이름"""
        pass


class StockFilter(ABC):
    """종목 필터 인터페이스 (Layer 3.5)"""

    @abstractmethod
    def apply(self, stock: dict) -> FilterResult:
        """단일 종목 필터 적용"""
        pass

    @abstractmethod
    def apply_batch(self, stocks: list[dict]) -> list[FilterResult]:
        """복수 종목 일괄 필터 적용"""
        pass

    @abstractmethod
    def get_filter_name(self) -> str:
        """필터 이름 반환"""
        pass

    @abstractmethod
    def get_conditions(self) -> dict:
        """필터 조건 반환"""
        pass


class SectorClassifier(ABC):
    """섹터 분류기 인터페이스 (Layer 3.5)"""

    @abstractmethod
    def classify(self, sector_name: str) -> SectorType:
        """단일 섹터 분류"""
        pass

    @abstractmethod
    def classify_batch(self, sectors: list[str]) -> dict[str, SectorType]:
        """복수 섹터 일괄 분류"""
        pass


class Scorer(ABC):
    """스코어러 인터페이스 (Layer 4)"""

    @abstractmethod
    def score(self, stock: dict, track_type: TrackType) -> ScoreResult:
        """종목 점수 계산"""
        pass

    @abstractmethod
    def get_weight_config(self, track_type: TrackType) -> dict:
        """Track별 가중치 반환"""
        pass


class LLMClient(ABC):
    """LLM 클라이언트 인터페이스"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """텍스트 생성"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """사용 가능 여부"""
        pass
