"""
종합 섹터 라벨링

네이버 테마 + DART 사업보고서 + 뉴스를 종합하여 정확한 섹터 분류
"""
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

from src.core.logger import get_logger
from src.core.config import get_config
from src.core.cache import get_cache


class SectorCategory(Enum):
    """섹터 대분류"""
    SEMICONDUCTOR = "반도체"
    BATTERY = "2차전지"
    BIO_PHARMA = "바이오/제약"
    IT_SOFTWARE = "IT/소프트웨어"
    AUTOMOTIVE = "자동차"
    FINANCE = "금융"
    CHEMICALS = "화학"
    STEEL_METAL = "철강/비철"
    CONSTRUCTION = "건설"
    RETAIL = "유통/소비재"
    FOOD_BEVERAGE = "음식료"
    ENTERTAINMENT = "엔터/미디어"
    TELECOM = "통신"
    LOGISTICS = "물류/운송"
    ENERGY = "에너지"
    DEFENSE = "방산"
    MACHINERY = "기계/장비"
    HEALTHCARE = "헬스케어"
    OTHER = "기타"


@dataclass
class SectorLabel:
    """종목 섹터 라벨"""
    stock_code: str
    stock_name: str

    # 주요 섹터 (1개)
    primary_sector: SectorCategory

    # 부 섹터 (최대 2개)
    secondary_sectors: list[SectorCategory] = field(default_factory=list)

    # 네이버 테마 태그
    theme_tags: list[str] = field(default_factory=list)

    # 사업 키워드 (DART에서 추출)
    business_keywords: list[str] = field(default_factory=list)

    # 현재 이슈 (뉴스에서 추출)
    current_issues: list[str] = field(default_factory=list)

    # 섹터 타입 (실적/성장 기반)
    is_growth_sector: bool = False

    # 신뢰도 (0~1)
    confidence: float = 0.0

    # 라벨링 소스
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "stock_code": self.stock_code,
            "stock_name": self.stock_name,
            "primary_sector": self.primary_sector.value,
            "secondary_sectors": [s.value for s in self.secondary_sectors],
            "theme_tags": self.theme_tags,
            "business_keywords": self.business_keywords,
            "current_issues": self.current_issues,
            "is_growth_sector": self.is_growth_sector,
            "confidence": self.confidence,
        }


class SectorLabeler:
    """
    종합 섹터 라벨링

    세 가지 소스를 종합하여 종목의 섹터를 정확하게 분류:
    1. 네이버 테마 → 투자 테마/트렌드 태그
    2. DART 사업보고서 → 실제 사업 내용
    3. 최근 뉴스 → 현재 주력 사업/이슈

    사용법:
        labeler = SectorLabeler()

        # 단일 종목 라벨링
        label = labeler.label_stock(
            stock_code="005930",
            stock_name="삼성전자",
            theme_names=["반도체", "AI"],
            dart_business_text="...",
            news_headlines=["삼성 HBM3E 양산...", ...]
        )

        # 배치 라벨링
        labels = labeler.label_batch(stocks_data)
    """

    # 섹터별 키워드 매핑
    SECTOR_KEYWORDS = {
        SectorCategory.SEMICONDUCTOR: [
            "반도체", "파운드리", "HBM", "메모리", "DRAM", "낸드", "NAND",
            "팹리스", "패키징", "EUV", "웨이퍼", "칩", "AP", "GPU", "NPU",
            "시스템반도체", "DDR", "SSD", "플래시"
        ],
        SectorCategory.BATTERY: [
            "2차전지", "배터리", "전기차", "EV", "리튬", "양극재", "음극재",
            "분리막", "전해질", "셀", "팩", "ESS", "전고체", "LFP", "NCM",
            "에너지저장", "충전", "BMS"
        ],
        SectorCategory.BIO_PHARMA: [
            "바이오", "제약", "신약", "임상", "FDA", "허가", "파이프라인",
            "항체", "유전자", "세포치료", "CDMO", "CMO", "위탁생산",
            "백신", "치료제", "진단키트", "바이오시밀러", "혁신신약"
        ],
        SectorCategory.IT_SOFTWARE: [
            "AI", "인공지능", "소프트웨어", "플랫폼", "클라우드", "SaaS",
            "핀테크", "보안", "게임", "메타버스", "블록체인", "빅데이터",
            "자율주행", "로봇", "IoT", "디지털전환"
        ],
        SectorCategory.AUTOMOTIVE: [
            "자동차", "완성차", "자동차부품", "타이어", "모빌리티",
            "전기차", "수소차", "FCEV", "BEV", "차량용"
        ],
        SectorCategory.FINANCE: [
            "은행", "증권", "보험", "금융", "저축은행", "캐피탈",
            "카드", "자산운용", "투자", "대출", "핀테크금융"
        ],
        SectorCategory.CHEMICALS: [
            "화학", "석유화학", "정밀화학", "화장품", "소재",
            "고분자", "페인트", "접착제", "플라스틱"
        ],
        SectorCategory.STEEL_METAL: [
            "철강", "비철금속", "알루미늄", "구리", "스테인리스",
            "특수강", "주물", "압연"
        ],
        SectorCategory.CONSTRUCTION: [
            "건설", "건축", "토목", "플랜트", "주택", "분양",
            "시멘트", "레미콘", "인테리어"
        ],
        SectorCategory.RETAIL: [
            "유통", "백화점", "마트", "편의점", "이커머스", "온라인쇼핑",
            "패션", "의류", "화장품유통"
        ],
        SectorCategory.FOOD_BEVERAGE: [
            "음식료", "식품", "음료", "주류", "제과", "유제품",
            "육가공", "수산", "농산물"
        ],
        SectorCategory.ENTERTAINMENT: [
            "엔터", "미디어", "방송", "콘텐츠", "영화", "음악",
            "게임", "레저", "여행", "호텔", "카지노"
        ],
        SectorCategory.TELECOM: [
            "통신", "5G", "6G", "네트워크", "인터넷", "케이블",
            "통신장비", "기지국"
        ],
        SectorCategory.LOGISTICS: [
            "물류", "운송", "해운", "항공", "택배", "창고",
            "포워딩", "컨테이너", "항만"
        ],
        SectorCategory.ENERGY: [
            "에너지", "태양광", "풍력", "신재생", "수소", "원자력",
            "정유", "가스", "전력", "발전"
        ],
        SectorCategory.DEFENSE: [
            "방산", "방위", "국방", "무기", "항공우주", "위성",
            "드론", "UAM", "로켓"
        ],
        SectorCategory.MACHINERY: [
            "기계", "장비", "공작기계", "산업용로봇", "자동화",
            "중공업", "조선", "엔진"
        ],
        SectorCategory.HEALTHCARE: [
            "헬스케어", "의료기기", "진단", "디지털헬스", "원격의료",
            "요양", "병원", "의료서비스"
        ],
    }

    # 성장 섹터 (Track B 대상)
    GROWTH_SECTORS = {
        SectorCategory.SEMICONDUCTOR,
        SectorCategory.BATTERY,
        SectorCategory.BIO_PHARMA,
        SectorCategory.IT_SOFTWARE,
        SectorCategory.HEALTHCARE,
        SectorCategory.DEFENSE,
        SectorCategory.ENERGY,
    }

    def __init__(self, use_llm: bool = True):
        """
        Args:
            use_llm: LLM 보조 분석 사용 여부
        """
        self.logger = get_logger(self.__class__.__name__)
        self.config = get_config()
        self.cache = get_cache()
        self.use_llm = use_llm
        self._llm_client = None

    def _get_llm_client(self):
        """Claude CLI 클라이언트"""
        if self._llm_client is None and self.use_llm:
            try:
                from src.llm import ClaudeCliClient
                self._llm_client = ClaudeCliClient(timeout=60)
            except Exception as e:
                self.logger.warning(f"LLM 클라이언트 초기화 실패: {e}")
        return self._llm_client

    def label_stock(
        self,
        stock_code: str,
        stock_name: str,
        theme_names: list[str] | None = None,
        dart_business_text: str | None = None,
        news_headlines: list[str] | None = None,
    ) -> SectorLabel:
        """
        단일 종목 섹터 라벨링

        Args:
            stock_code: 종목코드
            stock_name: 종목명
            theme_names: 소속 테마명 리스트 (primary_sector 결정에 사용)
            dart_business_text: DART 사업보고서 텍스트 (secondary_sectors 결정에 사용)
            news_headlines: 최근 뉴스 헤드라인 (secondary_sectors 결정에 사용)

        Returns:
            SectorLabel

        Note:
            - primary_sector: 네이버 테마에서 직접 결정 (투자자 관심 기준)
            - secondary_sectors: DART + 뉴스를 LLM으로 분석하여 결정 (실제 사업 기준)
        """
        # 캐시 확인
        cache_key = f"sector_label:{stock_code}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        sources = []

        # ============================================================
        # 1. Primary Sector: 네이버 테마에서 직접 결정
        # ============================================================
        primary_sector = SectorCategory.OTHER
        theme_sector_scores: dict[SectorCategory, float] = {s: 0.0 for s in SectorCategory}

        if theme_names:
            sources.append("theme")

            for sector, keywords in self.SECTOR_KEYWORDS.items():
                for theme in theme_names:
                    theme_lower = theme.lower()
                    for kw in keywords:
                        if kw.lower() in theme_lower:
                            theme_sector_scores[sector] += 1.0

            # 종목명에서 힌트 추출 (보조)
            for sector, keywords in self.SECTOR_KEYWORDS.items():
                for kw in keywords:
                    if kw.lower() in stock_name.lower():
                        theme_sector_scores[sector] += 0.3

            # 테마 기반 primary_sector 결정
            sorted_theme_sectors = sorted(
                theme_sector_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            if sorted_theme_sectors[0][1] > 0:
                primary_sector = sorted_theme_sectors[0][0]

        # ============================================================
        # 2. Secondary Sectors: DART + 뉴스 → LLM 분석
        # ============================================================
        business_keywords = []
        current_issues = []
        secondary_sectors = []

        # DART 사업보고서에서 키워드 추출
        if dart_business_text:
            sources.append("dart")
            business_keywords = self._extract_keywords_from_business(dart_business_text)

        # 뉴스에서 이슈 키워드 추출
        if news_headlines:
            sources.append("news")
            current_issues = self._extract_issues_from_news(news_headlines)

        # LLM으로 보조 섹터 분석 (DART + 뉴스 데이터가 있을 때)
        if self.use_llm and (dart_business_text or news_headlines):
            secondary_sectors = self._analyze_secondary_sectors_with_llm(
                stock_name=stock_name,
                primary_sector=primary_sector,
                dart_text=dart_business_text,
                news_headlines=news_headlines,
                business_keywords=business_keywords,
                current_issues=current_issues,
            )
        elif dart_business_text or news_headlines:
            # LLM 없을 때 규칙 기반 폴백
            secondary_sectors = self._determine_secondary_sectors_by_rules(
                primary_sector=primary_sector,
                dart_text=dart_business_text,
                news_headlines=news_headlines,
            )

        # ============================================================
        # 3. 결과 조합 및 신뢰도 계산
        # ============================================================
        confidence = 0.5
        if "theme" in sources:
            confidence += 0.2
        if "dart" in sources:
            confidence += 0.2
        if "news" in sources:
            confidence += 0.1
        confidence = min(confidence, 1.0)

        label = SectorLabel(
            stock_code=stock_code,
            stock_name=stock_name,
            primary_sector=primary_sector,
            secondary_sectors=secondary_sectors[:2],  # 최대 2개
            theme_tags=theme_names or [],
            business_keywords=business_keywords[:10],
            current_issues=current_issues[:5],
            is_growth_sector=primary_sector in self.GROWTH_SECTORS,
            confidence=round(confidence, 2),
            sources=sources,
        )

        # 캐시 저장 (24시간)
        self.cache.set(cache_key, label, ttl=86400)

        return label

    def _analyze_secondary_sectors_with_llm(
        self,
        stock_name: str,
        primary_sector: SectorCategory,
        dart_text: str | None,
        news_headlines: list[str] | None,
        business_keywords: list[str],
        current_issues: list[str],
    ) -> list[SectorCategory]:
        """
        DART + 뉴스 데이터를 LLM으로 분석하여 보조 섹터 결정

        Args:
            stock_name: 종목명
            primary_sector: 이미 결정된 메인 섹터 (네이버 테마 기반)
            dart_text: DART 사업보고서 텍스트
            news_headlines: 뉴스 헤드라인 리스트
            business_keywords: 추출된 사업 키워드
            current_issues: 추출된 이슈 키워드

        Returns:
            보조 섹터 리스트 (최대 2개)
        """
        client = self._get_llm_client()
        if not client:
            return self._determine_secondary_sectors_by_rules(
                primary_sector, dart_text, news_headlines
            )

        # 입력 데이터 구성
        dart_summary = ""
        if dart_text:
            # 너무 긴 경우 앞부분만 사용
            dart_summary = dart_text[:1500] if len(dart_text) > 1500 else dart_text

        news_summary = ""
        if news_headlines:
            news_summary = "\n".join(f"- {h}" for h in news_headlines[:10])

        # 선택 가능한 섹터 (primary 제외)
        available_sectors = [
            s.value for s in SectorCategory
            if s != primary_sector and s != SectorCategory.OTHER
        ]

        prompt = f"""다음 종목의 보조 사업 섹터를 분석해줘.
메인 섹터({primary_sector.value})는 이미 결정됨. 그 외 관련 사업 영역을 찾아줘.

종목명: {stock_name}
메인 섹터: {primary_sector.value}

[DART 사업보고서 내용]
{dart_summary if dart_summary else "없음"}

[최근 뉴스]
{news_summary if news_summary else "없음"}

[추출된 키워드]
사업: {", ".join(business_keywords) if business_keywords else "없음"}
이슈: {", ".join(current_issues) if current_issues else "없음"}

선택 가능한 보조 섹터: {", ".join(available_sectors)}

다음 형식으로 답해 (다른 설명 없이):
보조1: [섹터명]
보조2: [섹터명]

관련 보조 섹터가 없으면 "보조1: 없음" 이라고 답해."""

        try:
            result = client.generate(prompt).strip()

            secondary_sectors = []
            lines = result.split("\n")

            for line in lines:
                if "보조" in line and ":" in line:
                    sector_text = line.split(":", 1)[-1].strip()
                    if "없음" not in sector_text:
                        for sector in SectorCategory:
                            if sector.value in sector_text and sector != primary_sector:
                                if sector not in secondary_sectors:
                                    secondary_sectors.append(sector)
                                break

            return secondary_sectors[:2]

        except Exception as e:
            self.logger.debug(f"LLM 보조 섹터 분석 실패: {e}")
            return self._determine_secondary_sectors_by_rules(
                primary_sector, dart_text, news_headlines
            )

    def _determine_secondary_sectors_by_rules(
        self,
        primary_sector: SectorCategory,
        dart_text: str | None,
        news_headlines: list[str] | None,
    ) -> list[SectorCategory]:
        """
        규칙 기반으로 보조 섹터 결정 (LLM 폴백용)

        Args:
            primary_sector: 메인 섹터
            dart_text: DART 사업보고서 텍스트
            news_headlines: 뉴스 헤드라인 리스트

        Returns:
            보조 섹터 리스트
        """
        sector_scores: dict[SectorCategory, float] = {s: 0.0 for s in SectorCategory}

        # DART에서 키워드 매칭
        if dart_text:
            dart_lower = dart_text.lower()
            for sector, keywords in self.SECTOR_KEYWORDS.items():
                if sector == primary_sector:
                    continue
                for kw in keywords:
                    if kw.lower() in dart_lower:
                        sector_scores[sector] += 1.0

        # 뉴스에서 키워드 매칭
        if news_headlines:
            news_text = " ".join(news_headlines).lower()
            for sector, keywords in self.SECTOR_KEYWORDS.items():
                if sector == primary_sector:
                    continue
                for kw in keywords:
                    if kw.lower() in news_text:
                        sector_scores[sector] += 0.5

        # 점수 순으로 정렬하여 상위 2개 반환
        sorted_sectors = sorted(
            sector_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        secondary = []
        for sector, score in sorted_sectors[:2]:
            if score > 0:
                secondary.append(sector)

        return secondary

    def label_batch(
        self,
        stocks_data: list[dict]
    ) -> list[SectorLabel]:
        """
        배치 라벨링

        Args:
            stocks_data: [
                {
                    "stock_code": "005930",
                    "stock_name": "삼성전자",
                    "theme_names": ["반도체", "AI"],
                    "dart_business_text": "...",
                    "news_headlines": [...]
                },
                ...
            ]

        Returns:
            SectorLabel 리스트
        """
        self.logger.info(f"{len(stocks_data)}개 종목 섹터 라벨링 시작...")

        results = []
        for i, data in enumerate(stocks_data):
            label = self.label_stock(
                stock_code=data.get("stock_code", ""),
                stock_name=data.get("stock_name", ""),
                theme_names=data.get("theme_names"),
                dart_business_text=data.get("dart_business_text"),
                news_headlines=data.get("news_headlines"),
            )
            results.append(label)

            if (i + 1) % 50 == 0:
                self.logger.debug(f"진행: {i + 1}/{len(stocks_data)}")

        self.logger.info(f"섹터 라벨링 완료: {len(results)}개")
        return results

    def _extract_keywords_from_themes(self, theme_names: list[str]) -> list[str]:
        """테마명에서 키워드 추출"""
        keywords = []
        for theme in theme_names:
            # 테마명 자체가 키워드
            keywords.append(theme)

            # 복합 테마 분해 (예: "2차전지(소재)" → "2차전지", "소재")
            for sep in ["(", "/", ",", "·"]:
                if sep in theme:
                    parts = theme.replace(")", "").split(sep)
                    keywords.extend([p.strip() for p in parts if p.strip()])

        return list(set(keywords))

    def _extract_keywords_from_business(self, text: str) -> list[str]:
        """사업보고서에서 사업 키워드 추출"""
        keywords = []

        # 모든 섹터 키워드 매칭
        text_lower = text.lower()
        for sector_keywords in self.SECTOR_KEYWORDS.values():
            for kw in sector_keywords:
                if kw.lower() in text_lower:
                    keywords.append(kw)

        return list(set(keywords))[:15]

    def _extract_issues_from_news(self, headlines: list[str]) -> list[str]:
        """뉴스에서 현재 이슈 추출"""
        issues = []

        combined = " ".join(headlines).lower()

        # 주요 이슈 키워드 매칭
        issue_keywords = [
            "수주", "계약", "투자", "인수", "합병", "M&A",
            "승인", "허가", "FDA", "임상", "출시",
            "흑자", "적자", "실적", "매출", "영업이익",
            "신사업", "진출", "확장", "증설",
            "협력", "MOU", "파트너십",
        ]

        for kw in issue_keywords:
            if kw.lower() in combined:
                issues.append(kw)

        return list(set(issues))

    def _analyze_with_llm(
        self,
        stock_name: str,
        themes: list[str] | None,
        business_kw: list[str],
        issues: list[str]
    ) -> tuple[SectorCategory | None, list[SectorCategory]]:
        """
        LLM으로 섹터 분석 (메인 + 보조 섹터)

        Returns:
            (메인 섹터, 보조 섹터 리스트)
        """
        client = self._get_llm_client()
        if not client:
            return None, []

        themes_str = ", ".join(themes) if themes else "없음"
        business_str = ", ".join(business_kw) if business_kw else "없음"
        issues_str = ", ".join(issues) if issues else "없음"

        sector_list = ", ".join([s.value for s in SectorCategory if s != SectorCategory.OTHER])

        prompt = f"""다음 종목의 섹터를 분류해줘.

종목명: {stock_name}
소속 테마: {themes_str}
사업 키워드: {business_str}
현재 이슈: {issues_str}

선택 가능한 섹터: {sector_list}

다음 형식으로 답해 (다른 설명 없이):
메인: [섹터명]
보조: [섹터명1], [섹터명2]

보조 섹터가 없으면 "보조: 없음" 이라고 답해."""

        try:
            result = client.generate(prompt).strip()

            main_sector = None
            secondary_sectors = []

            lines = result.split("\n")
            for line in lines:
                if "메인:" in line or "메인 :" in line:
                    main_text = line.split(":", 1)[-1].strip()
                    for sector in SectorCategory:
                        if sector.value in main_text:
                            main_sector = sector
                            break

                elif "보조:" in line or "보조 :" in line:
                    sub_text = line.split(":", 1)[-1].strip()
                    if "없음" not in sub_text:
                        for sector in SectorCategory:
                            if sector.value in sub_text and sector != main_sector:
                                secondary_sectors.append(sector)

            return main_sector, secondary_sectors[:2]  # 최대 2개

        except Exception as e:
            self.logger.debug(f"LLM 섹터 분석 실패: {e}")

        return None, []

    def get_sector_type(self, sector: SectorCategory) -> str:
        """섹터 타입 반환 (A: 실적, B: 성장)"""
        if sector in self.GROWTH_SECTORS:
            return "B"
        return "A"

    def summarize_labels(self, labels: list[SectorLabel]) -> dict:
        """라벨링 결과 요약"""
        sector_counts = {}
        growth_count = 0

        for label in labels:
            sector = label.primary_sector.value
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            if label.is_growth_sector:
                growth_count += 1

        return {
            "total": len(labels),
            "sector_distribution": sector_counts,
            "growth_sector_count": growth_count,
            "earnings_sector_count": len(labels) - growth_count,
            "avg_confidence": sum(l.confidence for l in labels) / len(labels) if labels else 0,
        }

    def save_to_db(self, labels: list[SectorLabel]) -> int:
        """
        섹터 라벨을 DB에 저장

        Args:
            labels: SectorLabel 리스트

        Returns:
            저장된 레코드 수
        """
        from src.core.database import get_database
        from src.core.models import StockModel

        saved_count = 0

        try:
            with get_database().session() as session:
                for label in labels:
                    # 기존 종목 조회 또는 생성
                    stock = session.query(StockModel).filter_by(
                        stock_code=label.stock_code
                    ).first()

                    if not stock:
                        stock = StockModel(
                            stock_code=label.stock_code,
                            name=label.stock_name,
                        )
                        session.add(stock)

                    # 섹터 정보 업데이트
                    stock.primary_sector = label.primary_sector.value
                    stock.secondary_sectors = ",".join(
                        [s.value for s in label.secondary_sectors]
                    )
                    stock.sector_confidence = label.confidence

                    # Track 타입 결정
                    stock.track_type = "TRACK_B" if label.is_growth_sector else "TRACK_A"

                    saved_count += 1

                session.commit()
                self.logger.info(f"섹터 라벨 {saved_count}개 DB 저장 완료")

        except Exception as e:
            self.logger.error(f"섹터 라벨 DB 저장 실패: {e}")

        return saved_count

    def load_from_db(self, stock_codes: list[str] | None = None) -> list[SectorLabel]:
        """
        DB에서 섹터 라벨 로드

        Args:
            stock_codes: 조회할 종목 코드 (None이면 전체)

        Returns:
            SectorLabel 리스트
        """
        from src.core.database import get_database
        from src.core.models import StockModel

        labels = []

        try:
            with get_database().session() as session:
                query = session.query(StockModel)

                if stock_codes:
                    query = query.filter(StockModel.stock_code.in_(stock_codes))

                stocks = query.all()

                for stock in stocks:
                    if not stock.primary_sector:
                        continue

                    # 메인 섹터 파싱
                    primary = SectorCategory.OTHER
                    for sector in SectorCategory:
                        if sector.value == stock.primary_sector:
                            primary = sector
                            break

                    # 보조 섹터 파싱
                    secondary = []
                    if stock.secondary_sectors:
                        for sec_name in stock.secondary_sectors.split(","):
                            for sector in SectorCategory:
                                if sector.value == sec_name.strip():
                                    secondary.append(sector)
                                    break

                    labels.append(SectorLabel(
                        stock_code=stock.stock_code,
                        stock_name=stock.name,
                        primary_sector=primary,
                        secondary_sectors=secondary,
                        is_growth_sector=stock.track_type == "TRACK_B",
                        confidence=stock.sector_confidence or 0.0,
                    ))

        except Exception as e:
            self.logger.error(f"섹터 라벨 DB 로드 실패: {e}")

        return labels
