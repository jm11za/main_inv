"""
섹터 분류기

LLM 기반 섹터 Type A/B 분류 + 규칙 기반 폴백
"""
from enum import Enum

from src.core.interfaces import SectorType
from src.core.config import get_config
from src.core.logger import get_logger
from src.core.cache import get_cache


class SectorClassifier:
    """
    섹터 성격 분류기

    LLM을 사용하여 섹터가 실적 기반(Type A)인지 성장 기반(Type B)인지 분류
    LLM 실패 시 규칙 기반으로 폴백

    사용법:
        classifier = SectorClassifier()

        # 단일 섹터 분류
        sector_type = classifier.classify("바이오")  # SectorType.TYPE_B

        # 여러 섹터 분류
        results = classifier.classify_batch(["바이오", "자동차", "AI"])
    """

    # 규칙 기반 분류 키워드
    TYPE_B_KEYWORDS = [
        "바이오", "제약", "신약", "임상", "헬스케어",
        "AI", "인공지능", "로봇", "자율주행",
        "2차전지", "배터리", "전기차", "수소",
        "반도체", "HBM", "파운드리",
        "우주", "항공", "방산",
        "메타버스", "VR", "AR", "게임",
        "양자", "클라우드", "SaaS",
        "신재생", "태양광", "풍력",
        "플랫폼", "핀테크", "블록체인",
    ]

    TYPE_A_KEYWORDS = [
        "자동차", "완성차",
        "은행", "금융", "보험", "증권",
        "건설", "시멘트", "레미콘",
        "유통", "백화점", "마트",
        "철강", "비철금속",
        "화학", "정유", "석유",
        "음식료", "식품", "음료", "주류",
        "섬유", "의류", "패션",
        "제지", "포장", "인쇄",
        "운송", "물류", "해운", "항공사",
        "통신", "미디어", "광고",
        "호텔", "여행", "레저",
    ]

    def __init__(self, use_llm: bool = True, cache_ttl: int = 86400):
        """
        Args:
            use_llm: LLM 사용 여부 (False면 규칙 기반만 사용)
            cache_ttl: 분류 결과 캐시 TTL (기본 24시간)
        """
        self.logger = get_logger(self.__class__.__name__)
        self.use_llm = use_llm
        self.cache_ttl = cache_ttl
        self.cache = get_cache()

        self._llm_client = None

        config = get_config()
        self.fallback_type = config.get("filtering.fallback_type", "A")

    def _get_llm_client(self):
        """LLM 클라이언트 (Lazy init)"""
        if self._llm_client is None:
            try:
                from src.llm import ClaudeCliClient
                self._llm_client = ClaudeCliClient(timeout=30)
            except Exception as e:
                self.logger.warning(f"LLM 클라이언트 초기화 실패: {e}")
                self._llm_client = None
        return self._llm_client

    def classify(self, sector_name: str) -> SectorType:
        """
        단일 섹터 분류

        Args:
            sector_name: 섹터명

        Returns:
            SectorType.TYPE_A 또는 SectorType.TYPE_B
        """
        # 1. 캐시 확인
        cache_key = f"sector_type:{sector_name}"
        cached = self.cache.get(cache_key)
        if cached:
            return SectorType.TYPE_A if cached == "A" else SectorType.TYPE_B

        # 2. 규칙 기반 분류 먼저 시도
        rule_result = self._classify_by_rules(sector_name)
        if rule_result:
            self.cache.set(cache_key, rule_result, self.cache_ttl)
            return SectorType.TYPE_A if rule_result == "A" else SectorType.TYPE_B

        # 3. LLM 분류
        if self.use_llm:
            llm_result = self._classify_by_llm(sector_name)
            if llm_result:
                self.cache.set(cache_key, llm_result, self.cache_ttl)
                return SectorType.TYPE_A if llm_result == "A" else SectorType.TYPE_B

        # 4. 폴백
        self.logger.warning(f"섹터 분류 폴백 적용: {sector_name} -> Type {self.fallback_type}")
        return SectorType.TYPE_A if self.fallback_type == "A" else SectorType.TYPE_B

    def classify_batch(self, sector_names: list[str]) -> dict[str, SectorType]:
        """
        여러 섹터 일괄 분류

        Args:
            sector_names: 섹터명 리스트

        Returns:
            {섹터명: SectorType, ...}
        """
        results = {}
        for name in sector_names:
            results[name] = self.classify(name)
        return results

    def _classify_by_rules(self, sector_name: str) -> str | None:
        """규칙 기반 분류"""
        sector_lower = sector_name.lower()

        # Type B 키워드 체크
        for keyword in self.TYPE_B_KEYWORDS:
            if keyword.lower() in sector_lower or sector_lower in keyword.lower():
                self.logger.debug(f"규칙 분류: {sector_name} -> B (키워드: {keyword})")
                return "B"

        # Type A 키워드 체크
        for keyword in self.TYPE_A_KEYWORDS:
            if keyword.lower() in sector_lower or sector_lower in keyword.lower():
                self.logger.debug(f"규칙 분류: {sector_name} -> A (키워드: {keyword})")
                return "A"

        return None

    def _classify_by_llm(self, sector_name: str) -> str | None:
        """LLM 기반 분류"""
        client = self._get_llm_client()
        if not client:
            return None

        try:
            result = client.classify_sector(sector_name)
            self.logger.debug(f"LLM 분류: {sector_name} -> {result}")
            return result
        except Exception as e:
            self.logger.warning(f"LLM 분류 실패: {sector_name} - {e}")
            return None

    def get_type_description(self, sector_type: SectorType) -> dict:
        """섹터 타입 설명 반환"""
        if sector_type == SectorType.TYPE_A:
            return {
                "type": "A",
                "name": "실적 기반",
                "logic": "숫자가 찍혀야 주가가 간다",
                "key_metrics": ["영업이익", "PER", "ROE", "실적 턴어라운드"],
                "filter": "Hard Filter",
            }
        else:
            return {
                "type": "B",
                "name": "성장 기반",
                "logic": "꿈을 먹고 주가가 간다",
                "key_metrics": ["기술력", "파이프라인", "시장 성장성", "R&D"],
                "filter": "Soft Filter",
            }
