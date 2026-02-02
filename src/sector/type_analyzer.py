"""
② 테마 Type A/B 분류 모듈 (v2.0)

각 테마가 실적형(A)인지 성장형(B)인지 판단
- Type A (실적형): 영업이익, PER, 실적 턴어라운드가 중요
- Type B (성장형): 기술력, 파이프라인, 미래 성장성이 중요

변경사항 (v2.0):
- SectorCategory Enum 삭제 → 테마명 문자열 그대로 사용
- 키워드 기반 + LLM 분류 방식
"""
from dataclasses import dataclass
from typing import Any

from src.core.logger import get_logger
from src.core.config import get_config
from src.core.interfaces import SectorType


@dataclass
class SectorTypeResult:
    """테마 Type 분류 결과"""
    theme_name: str             # 네이버 테마명 그대로
    sector_type: SectorType     # TYPE_A or TYPE_B
    reasoning: str              # 분류 근거
    confidence: float           # 신뢰도 (0~1)
    matched_keywords: list[str] | None = None  # 매칭된 키워드

    def to_dict(self) -> dict:
        return {
            "theme_name": self.theme_name,
            "sector_type": self.sector_type.value,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "matched_keywords": self.matched_keywords,
        }


class SectorTypeAnalyzer:
    """
    ② 테마 Type A/B 분류기 (v2.0)

    각 테마의 투자 성격을 키워드 + LLM으로 판단:
    - Type A (실적형): "숫자가 찍혀야 주가가 간다"
    - Type B (성장형): "꿈을 먹고 주가가 간다"

    사용법:
        analyzer = SectorTypeAnalyzer(llm_client)

        # 단일 테마 분류
        result = analyzer.analyze_theme("2차전지")

        # 배치 분류
        results = analyzer.analyze_batch(["2차전지", "반도체", "은행"])
    """

    # Type B (성장형) 키워드 - 이 키워드가 포함되면 Type B
    TYPE_B_KEYWORDS = [
        # 기술/성장 섹터
        "반도체", "HBM", "AI", "인공지능", "로봇", "자율주행",
        "2차전지", "배터리", "전기차", "수소", "연료전지",
        "바이오", "제약", "신약", "임상", "헬스케어",
        "우주", "항공", "방산", "드론", "UAM",
        "메타버스", "NFT", "블록체인", "클라우드", "SaaS",
        "핀테크", "디지털", "플랫폼", "소프트웨어", "IT",
        "태양광", "풍력", "신재생", "에너지저장", "ESS",
        "5G", "6G", "통신장비", "네트워크",
        "반도체 장비", "반도체 소재", "반도체 부품",
        "게임", "엔터", "콘텐츠", "OTT", "미디어",
        "전자", "디스플레이", "OLED", "LED",
        "스마트팩토리", "자동화", "IoT",
    ]

    # Type A (실적형) 키워드 - 이 키워드가 포함되면 Type A
    TYPE_A_KEYWORDS = [
        # 전통/실적 섹터
        "은행", "금융", "보험", "증권", "카드",
        "건설", "시멘트", "레미콘", "인테리어",
        "철강", "비철금속", "알루미늄", "동",
        "화학", "정유", "석유", "가스",
        "자동차", "부품", "타이어",
        "조선", "해운", "물류", "운송", "항공사",
        "유통", "백화점", "마트", "홈쇼핑", "편의점",
        "식품", "음료", "주류", "담배", "농업",
        "섬유", "의류", "패션", "화장품",
        "제지", "포장", "인쇄",
        "기계", "중공업", "플랜트",
        "통신사", "케이블", "방송",
        "전력", "가스공급", "수도",
        "리츠", "부동산", "임대",
        "지주사", "지주", "홀딩스",
    ]

    # Type별 특성 설명
    TYPE_DESCRIPTIONS = {
        SectorType.TYPE_A: {
            "name": "실적 기반",
            "philosophy": "숫자가 찍혀야 주가가 간다",
            "key_factors": ["영업이익", "PER", "실적 턴어라운드", "배당"],
            "filter_focus": ["부채비율", "영업이익률", "ROE"],
        },
        SectorType.TYPE_B: {
            "name": "성장 기반",
            "philosophy": "꿈을 먹고 주가가 간다",
            "key_factors": ["기술력", "파이프라인", "미래 성장성", "시장 점유율"],
            "filter_focus": ["매출성장률", "R&D비율", "시장지배력"],
        },
    }

    def __init__(self, llm_client=None, use_llm: bool = True):
        """
        Args:
            llm_client: LLM 클라이언트
            use_llm: LLM 사용 여부 (False면 키워드 기반만)
        """
        self.logger = get_logger(self.__class__.__name__)
        self.config = get_config()
        self._llm_client = llm_client
        self.use_llm = use_llm

    def set_llm_client(self, client):
        """LLM 클라이언트 설정"""
        self._llm_client = client

    def analyze_theme(
        self,
        theme_name: str,
        theme_context: dict | None = None,
    ) -> SectorTypeResult:
        """
        단일 테마 Type 분류

        Args:
            theme_name: 테마명 (네이버 테마명 그대로)
            theme_context: 테마 맥락 정보 (선택)
                - stock_count: 종목 수
                - change_rate: 등락률
                - top_stocks: 상위 종목명

        Returns:
            SectorTypeResult
        """
        # 1차: 키워드 기반 분류
        keyword_result = self._analyze_by_keywords(theme_name)

        # 키워드로 확실히 분류되면 반환
        if keyword_result.confidence >= 0.8:
            return keyword_result

        # 2차: LLM으로 분류 (키워드 불확실 + LLM 활성화)
        if self.use_llm and self._llm_client:
            return self._analyze_with_llm(theme_name, theme_context, keyword_result)

        return keyword_result

    def analyze_batch(
        self,
        theme_names: list[str],
        progress_callback=None,
    ) -> list[SectorTypeResult]:
        """
        배치 분류

        Args:
            theme_names: 테마명 리스트
            progress_callback: 진행 콜백 함수

        Returns:
            SectorTypeResult 리스트
        """
        self.logger.info(f"{len(theme_names)}개 테마 Type 분류 시작")

        results = []
        uncertain_themes = []

        # 1차: 키워드 기반 분류
        for i, theme_name in enumerate(theme_names):
            result = self._analyze_by_keywords(theme_name)
            results.append(result)

            if result.confidence < 0.8:
                uncertain_themes.append((i, theme_name))

            if progress_callback and (i + 1) % 50 == 0:
                progress_callback(i + 1, len(theme_names))

        # 2차: 불확실한 테마들 LLM으로 재분류
        if uncertain_themes and self.use_llm and self._llm_client:
            self.logger.info(f"{len(uncertain_themes)}개 불확실 테마 LLM 분류")
            llm_results = self._analyze_batch_with_llm(
                [t[1] for t in uncertain_themes]
            )
            for (idx, _), llm_result in zip(uncertain_themes, llm_results):
                results[idx] = llm_result

        self.logger.info(f"테마 Type 분류 완료: {len(results)}개")
        return results

    def _match_keywords(self, theme_name: str, keywords: list[str]) -> list[str]:
        """
        키워드 매칭 (단어 단위, 부분 문자열 오매칭 방지)

        규칙:
        1. 짧은 키워드(2자 이하): 토큰 완전 일치만 허용
        2. 긴 키워드(3자 이상): 부분 문자열 매칭 허용

        예시:
        - "동" → "부동산"에 매칭 안됨 (2자 이하, 토큰 불일치)
        - "부동산" → "부동산 보유 자산주"에 매칭됨 (3자 이상, 부분 문자열)
        - "반도체" → "반도체 장비"에 매칭됨 (3자 이상, 부분 문자열)
        """
        import re

        # 테마명을 토큰으로 분리 (공백, 괄호, 슬래시, 하이픈 등)
        theme_tokens = set(re.split(r'[\s/\(\)\[\]\-]+', theme_name.lower()))
        # 테마명 전체 (소문자)
        theme_lower = theme_name.lower()

        matches = []
        for kw in keywords:
            kw_lower = kw.lower()

            # 1. 토큰 완전 일치 (모든 키워드)
            if kw_lower in theme_tokens:
                matches.append(kw)
            # 2. 긴 키워드(3자 이상)는 부분 문자열 매칭 허용
            elif len(kw) >= 3 and kw_lower in theme_lower:
                matches.append(kw)

        return matches

    def _analyze_by_keywords(self, theme_name: str) -> SectorTypeResult:
        """키워드 기반 분류"""
        # Type B 키워드 매칭
        type_b_matches = self._match_keywords(theme_name, self.TYPE_B_KEYWORDS)

        # Type A 키워드 매칭
        type_a_matches = self._match_keywords(theme_name, self.TYPE_A_KEYWORDS)

        # 판정
        if type_b_matches and not type_a_matches:
            return SectorTypeResult(
                theme_name=theme_name,
                sector_type=SectorType.TYPE_B,
                reasoning=f"성장형 키워드 매칭: {', '.join(type_b_matches[:3])}",
                confidence=0.85,
                matched_keywords=type_b_matches,
            )
        elif type_a_matches and not type_b_matches:
            return SectorTypeResult(
                theme_name=theme_name,
                sector_type=SectorType.TYPE_A,
                reasoning=f"실적형 키워드 매칭: {', '.join(type_a_matches[:3])}",
                confidence=0.85,
                matched_keywords=type_a_matches,
            )
        elif type_b_matches and type_a_matches:
            # 둘 다 매칭되면 더 많이 매칭된 쪽
            if len(type_b_matches) >= len(type_a_matches):
                return SectorTypeResult(
                    theme_name=theme_name,
                    sector_type=SectorType.TYPE_B,
                    reasoning=f"복합 테마 (성장형 우세): B={len(type_b_matches)}, A={len(type_a_matches)}",
                    confidence=0.6,
                    matched_keywords=type_b_matches,
                )
            else:
                return SectorTypeResult(
                    theme_name=theme_name,
                    sector_type=SectorType.TYPE_A,
                    reasoning=f"복합 테마 (실적형 우세): A={len(type_a_matches)}, B={len(type_b_matches)}",
                    confidence=0.6,
                    matched_keywords=type_a_matches,
                )
        else:
            # 매칭 없음 - 기본값 Type A (보수적)
            return SectorTypeResult(
                theme_name=theme_name,
                sector_type=SectorType.TYPE_A,
                reasoning="키워드 매칭 없음 - 기본값(실적형)",
                confidence=0.5,
                matched_keywords=None,
            )

    def _analyze_with_llm(
        self,
        theme_name: str,
        theme_context: dict | None,
        keyword_result: SectorTypeResult,
    ) -> SectorTypeResult:
        """LLM으로 단일 테마 분석"""
        context_str = ""
        if theme_context:
            context_str = f"""
[테마 현황]
- 종목 수: {theme_context.get('stock_count', 'N/A')}
- 등락률: {theme_context.get('change_rate', 'N/A')}%
- 상위 종목: {', '.join(theme_context.get('top_stocks', [])[:5])}
"""

        prompt = f"""한국 주식시장에서 "{theme_name}" 테마의 주가를 움직이는 핵심 동력을 분석해.

Type A (실적 기반): 영업이익, PER, 실적 턴어라운드가 중요. "숫자가 찍혀야 주가가 간다"
예시 테마: 은행, 건설, 자동차, 유통, 철강, 화학

Type B (기대감 기반): 기술력, 파이프라인, 미래 성장성이 중요. "꿈을 먹고 주가가 간다"
예시 테마: 바이오, AI, 로봇, 우주항공, 2차전지
{context_str}
"{theme_name}" 테마는 Type A인가 Type B인가?

다음 형식으로 답해:
Type: [A 또는 B]
이유: [한 문장으로]"""

        try:
            result = self._llm_client.generate(prompt).strip()

            # 결과 파싱
            sector_type = keyword_result.sector_type  # 기본값: 키워드 결과
            reasoning = ""

            for line in result.split("\n"):
                line_lower = line.lower()
                if "type:" in line_lower or "타입:" in line_lower:
                    if "b" in line_lower:
                        sector_type = SectorType.TYPE_B
                    else:
                        sector_type = SectorType.TYPE_A
                elif "이유:" in line or "reason:" in line_lower:
                    reasoning = line.split(":", 1)[-1].strip()

            if not reasoning:
                reasoning = f"LLM 분석: {result[:100]}"

            return SectorTypeResult(
                theme_name=theme_name,
                sector_type=sector_type,
                reasoning=reasoning,
                confidence=0.9,
                matched_keywords=keyword_result.matched_keywords,
            )

        except Exception as e:
            self.logger.warning(f"LLM 분석 실패 ({theme_name}): {e}")
            return keyword_result

    def _analyze_batch_with_llm(
        self,
        theme_names: list[str]
    ) -> list[SectorTypeResult]:
        """LLM으로 배치 분석"""
        themes_text = ", ".join(theme_names[:30])  # 최대 30개

        prompt = f"""다음 테마들을 Type A 또는 Type B로 분류해.

Type A (실적 기반): 영업이익, 실적이 주가 핵심. 예: 은행, 건설, 자동차
Type B (기대감 기반): 기술력, 미래 성장성이 주가 핵심. 예: 바이오, AI, 로봇

테마: {themes_text}

각 테마에 대해 "테마명:A" 또는 "테마명:B" 형식으로 쉼표로 구분해서 답해.
예: 은행:A, 바이오:B"""

        try:
            result = self._llm_client.generate(prompt).strip()

            # 파싱
            type_map = {}
            for part in result.replace("\n", ",").split(","):
                part = part.strip()
                if ":" in part:
                    name, type_str = part.rsplit(":", 1)
                    name = name.strip()
                    type_str = type_str.strip().upper()

                    for theme_name in theme_names:
                        if theme_name in name or name in theme_name:
                            sector_type = SectorType.TYPE_B if "B" in type_str else SectorType.TYPE_A
                            type_map[theme_name] = sector_type
                            break

            # 결과 생성
            results = []
            for theme_name in theme_names:
                if theme_name in type_map:
                    results.append(SectorTypeResult(
                        theme_name=theme_name,
                        sector_type=type_map[theme_name],
                        reasoning="LLM 배치 분석",
                        confidence=0.85,
                        matched_keywords=None,
                    ))
                else:
                    # LLM 매핑 실패 시 키워드 기반
                    results.append(self._analyze_by_keywords(theme_name))

            return results

        except Exception as e:
            self.logger.warning(f"LLM 배치 분석 실패: {e}")
            return [self._analyze_by_keywords(t) for t in theme_names]

    def get_type_description(self, sector_type: SectorType) -> dict:
        """Type 설명 반환"""
        return self.TYPE_DESCRIPTIONS.get(sector_type, {})

    def summarize(self, results: list[SectorTypeResult]) -> dict:
        """분류 결과 요약"""
        type_a = [r for r in results if r.sector_type == SectorType.TYPE_A]
        type_b = [r for r in results if r.sector_type == SectorType.TYPE_B]

        return {
            "total_themes": len(results),
            "type_a_count": len(type_a),
            "type_b_count": len(type_b),
            "type_a_themes": [r.theme_name for r in type_a],
            "type_b_themes": [r.theme_name for r in type_b],
            "avg_confidence": sum(r.confidence for r in results) / len(results) if results else 0,
        }

    def get_filter_criteria(self, sector_type: SectorType) -> dict:
        """Type별 필터 기준 반환"""
        if sector_type == SectorType.TYPE_A:
            return {
                "min_operating_profit_4q": 0,
                "max_debt_ratio": 200,
                "max_pbr": 3.0,
                "min_trading_value": 1_000_000_000,
            }
        else:  # TYPE_B
            return {
                "max_capital_impairment": 50,
                "min_current_ratio": 100,
                "min_rd_ratio": 5,
                "min_trading_value": 500_000_000,
            }
