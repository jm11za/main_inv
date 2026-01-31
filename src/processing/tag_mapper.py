"""
태그 매퍼

동의어 해결 및 Theme-Stock 매핑
"""
from dataclasses import dataclass, field
from typing import Any

from src.core.logger import get_logger
from src.core.cache import get_cache


@dataclass
class ThemeMapping:
    """테마-종목 매핑 결과"""
    theme_id: str
    theme_name: str
    theme_type: str  # "실체형", "기대형"
    keywords: list[str]
    stock_codes: list[str]
    leader_stock: str | None = None


@dataclass
class StockThemeInfo:
    """종목의 테마 정보"""
    stock_code: str
    stock_name: str
    themes: list[str]  # theme_ids
    keywords: list[str]
    primary_theme: str | None = None


class SynonymResolver:
    """
    동의어 해결기

    다양한 표현을 표준 키워드로 통합

    사용법:
        resolver = SynonymResolver()
        standard = resolver.resolve("이차전지")  # "2차전지"
    """

    # 동의어 사전: {표준어: [동의어들]}
    SYNONYM_DICT = {
        # 2차전지/배터리
        "2차전지": ["이차전지", "리튬이온", "리튬배터리", "배터리셀", "전지"],
        "배터리": ["밧데리", "battery"],
        "전기차": ["EV", "전기자동차", "BEV", "전기승용차"],

        # 반도체
        "반도체": ["semiconductor", "칩", "chip"],
        "HBM": ["고대역폭메모리", "High Bandwidth Memory"],
        "파운드리": ["foundry", "위탁생산"],
        "시스템반도체": ["비메모리", "non-memory"],

        # AI/로봇
        "AI": ["인공지능", "artificial intelligence", "머신러닝", "딥러닝"],
        "로봇": ["robot", "로보틱스", "robotics"],
        "자율주행": ["autonomous", "무인차", "자율차"],

        # 바이오
        "바이오": ["bio", "생명공학", "생명과학"],
        "신약": ["신약개발", "drug development", "파이프라인"],
        "임상": ["clinical", "임상시험", "FDA"],
        "바이오시밀러": ["biosimilar", "바이오복제약"],

        # 에너지
        "태양광": ["solar", "솔라", "태양전지"],
        "풍력": ["wind", "풍력발전"],
        "수소": ["hydrogen", "수소연료전지", "수전해"],
        "신재생에너지": ["재생에너지", "renewable", "그린에너지"],

        # 기타 테마
        "메타버스": ["metaverse", "가상현실", "가상세계"],
        "VR": ["virtual reality", "가상현실"],
        "AR": ["augmented reality", "증강현실"],
        "우주항공": ["aerospace", "우주산업", "항공우주"],
        "방산": ["방위산업", "defense", "국방"],
        "조선": ["shipbuilding", "선박"],
        "핀테크": ["fintech", "금융기술"],
        "블록체인": ["blockchain", "암호화폐", "가상자산"],
        "클라우드": ["cloud", "클라우드컴퓨팅"],
        "SaaS": ["서비스형소프트웨어", "saas"],
    }

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

        # 역방향 매핑 구축 (동의어 -> 표준어)
        self._reverse_map = {}
        for standard, synonyms in self.SYNONYM_DICT.items():
            self._reverse_map[standard.lower()] = standard
            for syn in synonyms:
                self._reverse_map[syn.lower()] = standard

    def resolve(self, keyword: str) -> str:
        """
        동의어를 표준 키워드로 변환

        Args:
            keyword: 입력 키워드

        Returns:
            표준 키워드 (없으면 원본 반환)
        """
        return self._reverse_map.get(keyword.lower(), keyword)

    def resolve_list(self, keywords: list[str]) -> list[str]:
        """
        키워드 리스트 표준화 (중복 제거)

        Args:
            keywords: 키워드 리스트

        Returns:
            표준화된 키워드 리스트
        """
        resolved = []
        seen = set()

        for kw in keywords:
            standard = self.resolve(kw)
            if standard.lower() not in seen:
                resolved.append(standard)
                seen.add(standard.lower())

        return resolved

    def add_synonym(self, standard: str, synonym: str):
        """동의어 추가"""
        if standard not in self.SYNONYM_DICT:
            self.SYNONYM_DICT[standard] = []
        self.SYNONYM_DICT[standard].append(synonym)
        self._reverse_map[synonym.lower()] = standard

    def get_all_forms(self, keyword: str) -> list[str]:
        """키워드의 모든 형태(표준어 + 동의어) 반환"""
        standard = self.resolve(keyword)
        forms = [standard]
        if standard in self.SYNONYM_DICT:
            forms.extend(self.SYNONYM_DICT[standard])
        return forms


class TagMapper:
    """
    테마-종목 매퍼

    추출된 키워드와 네이버 테마를 결합하여 최종 매핑 생성

    사용법:
        mapper = TagMapper()

        # 종목의 테마 매핑
        themes = mapper.map_stock_to_themes(
            stock_code="005930",
            stock_name="삼성전자",
            extracted_keywords=["반도체", "HBM", "파운드리"],
            naver_themes=["AI반도체", "HBM"]
        )

        # 테마의 종목 매핑
        stocks = mapper.map_theme_to_stocks(
            theme_name="HBM",
            naver_stock_codes=["005930", "000660"],
            keyword_matches={"005930": ["HBM", "메모리"]}
        )
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.synonym_resolver = SynonymResolver()
        self.cache = get_cache()

    def map_stock_to_themes(
        self,
        stock_code: str,
        stock_name: str,
        extracted_keywords: list[str],
        naver_themes: list[str]
    ) -> StockThemeInfo:
        """
        종목 -> 테마 매핑

        Args:
            stock_code: 종목코드
            stock_name: 종목명
            extracted_keywords: LLM 추출 키워드
            naver_themes: 네이버 테마 리스트

        Returns:
            StockThemeInfo
        """
        # 키워드 표준화
        normalized_keywords = self.synonym_resolver.resolve_list(extracted_keywords)

        # 테마 리스트 (네이버 테마 + 키워드 기반)
        themes = list(set(naver_themes))

        # 키워드 기반 추가 테마 매핑
        keyword_themes = self._keywords_to_themes(normalized_keywords)
        for theme in keyword_themes:
            if theme not in themes:
                themes.append(theme)

        # 주요 테마 결정 (첫 번째)
        primary_theme = themes[0] if themes else None

        return StockThemeInfo(
            stock_code=stock_code,
            stock_name=stock_name,
            themes=themes,
            keywords=normalized_keywords,
            primary_theme=primary_theme
        )

    def map_theme_to_stocks(
        self,
        theme_name: str,
        theme_type: str,
        naver_stock_codes: list[str],
        keyword_matches: dict[str, list[str]] | None = None
    ) -> ThemeMapping:
        """
        테마 -> 종목 매핑

        Args:
            theme_name: 테마명
            theme_type: 테마 유형 ("실체형", "기대형")
            naver_stock_codes: 네이버 테마 소속 종목코드
            keyword_matches: {종목코드: [매칭 키워드]} (키워드 기반 추가)

        Returns:
            ThemeMapping
        """
        # 종목 코드 통합 (중복 제거)
        all_codes = list(set(naver_stock_codes))

        # 키워드 매칭 종목 추가
        if keyword_matches:
            for code in keyword_matches.keys():
                if code not in all_codes:
                    all_codes.append(code)

        # 키워드 추출
        all_keywords = self._extract_theme_keywords(theme_name)
        if keyword_matches:
            for keywords in keyword_matches.values():
                for kw in keywords:
                    if kw not in all_keywords:
                        all_keywords.append(kw)

        # 대장주 결정 (첫 번째 = 네이버 테마 첫 번째)
        leader = naver_stock_codes[0] if naver_stock_codes else None

        return ThemeMapping(
            theme_id=self._generate_theme_id(theme_name),
            theme_name=theme_name,
            theme_type=theme_type,
            keywords=all_keywords[:10],
            stock_codes=all_codes,
            leader_stock=leader
        )

    def merge_mappings(
        self,
        naver_themes: dict[str, list[str]],
        llm_keywords: dict[str, list[str]],
        stock_names: dict[str, str]
    ) -> tuple[list[ThemeMapping], list[StockThemeInfo]]:
        """
        네이버 테마와 LLM 키워드 통합

        Args:
            naver_themes: {테마명: [종목코드]} - 네이버 테마
            llm_keywords: {종목코드: [키워드]} - LLM 추출
            stock_names: {종목코드: 종목명}

        Returns:
            (테마 매핑 리스트, 종목 테마 정보 리스트)
        """
        theme_mappings = []
        stock_infos = {}

        # 1. 네이버 테마 기반 매핑
        for theme_name, codes in naver_themes.items():
            # 키워드 매치 정보 구축
            keyword_matches = {}
            for code in codes:
                if code in llm_keywords:
                    keyword_matches[code] = llm_keywords[code]

            # 테마 타입 결정
            theme_type = self._determine_theme_type(theme_name, keyword_matches)

            mapping = self.map_theme_to_stocks(
                theme_name=theme_name,
                theme_type=theme_type,
                naver_stock_codes=codes,
                keyword_matches=keyword_matches
            )
            theme_mappings.append(mapping)

            # 종목별 테마 정보 누적
            for code in codes:
                if code not in stock_infos:
                    stock_infos[code] = {
                        "themes": [],
                        "keywords": llm_keywords.get(code, [])
                    }
                stock_infos[code]["themes"].append(theme_name)

        # 2. 종목 정보 생성
        stock_info_list = []
        for code, info in stock_infos.items():
            stock_info = StockThemeInfo(
                stock_code=code,
                stock_name=stock_names.get(code, ""),
                themes=info["themes"],
                keywords=self.synonym_resolver.resolve_list(info["keywords"]),
                primary_theme=info["themes"][0] if info["themes"] else None
            )
            stock_info_list.append(stock_info)

        self.logger.info(
            f"매핑 완료: {len(theme_mappings)}개 테마, "
            f"{len(stock_info_list)}개 종목"
        )

        return theme_mappings, stock_info_list

    def _keywords_to_themes(self, keywords: list[str]) -> list[str]:
        """키워드 -> 테마명 변환"""
        # 키워드-테마 매핑 규칙
        KEYWORD_THEME_MAP = {
            "반도체": "반도체",
            "HBM": "HBM",
            "파운드리": "파운드리",
            "AI": "AI반도체",
            "2차전지": "2차전지",
            "배터리": "2차전지",
            "전기차": "전기차",
            "바이오": "바이오",
            "신약": "바이오",
            "로봇": "로봇",
            "자율주행": "자율주행",
            "우주항공": "우주항공",
            "방산": "방산",
            "조선": "조선",
            "태양광": "신재생에너지",
            "수소": "수소경제",
        }

        themes = []
        for kw in keywords:
            if kw in KEYWORD_THEME_MAP:
                theme = KEYWORD_THEME_MAP[kw]
                if theme not in themes:
                    themes.append(theme)

        return themes

    def _extract_theme_keywords(self, theme_name: str) -> list[str]:
        """테마명에서 핵심 키워드 추출"""
        # 테마명 자체를 키워드로
        keywords = [theme_name]

        # 테마명에 포함된 알려진 키워드
        for standard in self.synonym_resolver.SYNONYM_DICT.keys():
            if standard.lower() in theme_name.lower():
                keywords.append(standard)

        return list(set(keywords))

    def _determine_theme_type(
        self,
        theme_name: str,
        keyword_matches: dict[str, list[str]]
    ) -> str:
        """테마 타입 결정"""
        # 기대형 테마 키워드
        EXPECTATION_THEMES = [
            "바이오", "신약", "AI", "로봇", "자율주행",
            "우주", "메타버스", "양자", "신사업"
        ]

        theme_lower = theme_name.lower()
        for exp in EXPECTATION_THEMES:
            if exp.lower() in theme_lower:
                return "기대형"

        # 키워드 기반 판단
        all_keywords = []
        for keywords in keyword_matches.values():
            all_keywords.extend(keywords)

        for exp in EXPECTATION_THEMES:
            for kw in all_keywords:
                if exp.lower() in kw.lower():
                    return "기대형"

        return "실체형"

    def _generate_theme_id(self, theme_name: str) -> str:
        """테마 ID 생성"""
        # 간단한 해시 기반 ID
        import hashlib
        hash_obj = hashlib.md5(theme_name.encode())
        return f"theme_{hash_obj.hexdigest()[:8]}"
