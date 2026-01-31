"""
LLM 키워드 추출기

뉴스, DART 사업보고서에서 투자 관련 키워드 추출
"""
from dataclasses import dataclass, field
from typing import Any

from src.core.logger import get_logger
from src.core.cache import get_cache
from src.processing.preprocessor import Preprocessor


@dataclass
class ExtractionResult:
    """키워드 추출 결과"""
    stock_code: str
    stock_name: str
    source_type: str  # "news", "dart"
    keywords: list[str]
    theme_type: str  # "실체형", "기대형"
    summary: str
    raw_text_length: int


@dataclass
class StockKeywords:
    """종목별 통합 키워드"""
    stock_code: str
    stock_name: str
    news_keywords: list[str] = field(default_factory=list)
    dart_keywords: list[str] = field(default_factory=list)
    combined_keywords: list[str] = field(default_factory=list)
    theme_type: str = "실체형"


class LLMExtractor:
    """
    LLM 기반 키워드 추출기

    Ollama를 사용하여 뉴스/DART에서 투자 관련 키워드 추출

    사용법:
        extractor = LLMExtractor()

        # 뉴스에서 키워드 추출
        result = extractor.extract_from_news(stock_code, stock_name, headlines)

        # DART에서 키워드 추출
        result = extractor.extract_from_dart(stock_code, stock_name, content)

        # 통합 추출
        keywords = extractor.extract_all(stock_code, stock_name, news, dart)
    """

    def __init__(self, use_cache: bool = True, cache_ttl: int = 86400):
        """
        Args:
            use_cache: 캐시 사용 여부
            cache_ttl: 캐시 TTL (기본 24시간)
        """
        self.logger = get_logger(self.__class__.__name__)
        self.preprocessor = Preprocessor()
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl
        self.cache = get_cache() if use_cache else None

        self._llm_client = None

    def _get_llm_client(self):
        """LLM 클라이언트 (Lazy init)"""
        if self._llm_client is None:
            try:
                from src.llm import OllamaClient
                self._llm_client = OllamaClient()

                if not self._llm_client.is_available():
                    self.logger.warning("Ollama 사용 불가, Claude CLI로 폴백")
                    from src.llm import ClaudeCliClient
                    self._llm_client = ClaudeCliClient(timeout=60)

            except Exception as e:
                self.logger.error(f"LLM 클라이언트 초기화 실패: {e}")
                self._llm_client = None

        return self._llm_client

    def extract_from_news(
        self,
        stock_code: str,
        stock_name: str,
        headlines: list[str],
        max_keywords: int = 10
    ) -> ExtractionResult:
        """
        뉴스 헤드라인에서 키워드 추출

        Args:
            stock_code: 종목코드
            stock_name: 종목명
            headlines: 뉴스 헤드라인 리스트
            max_keywords: 최대 키워드 수

        Returns:
            ExtractionResult
        """
        # 캐시 확인
        cache_key = f"news_keywords:{stock_code}"
        if self.cache and self.use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                self.logger.debug(f"캐시 히트: {stock_code} 뉴스 키워드")
                return ExtractionResult(**cached)

        # 헤드라인 정제
        cleaned_headlines = self.preprocessor.clean_headlines(headlines)
        if not cleaned_headlines:
            return ExtractionResult(
                stock_code=stock_code,
                stock_name=stock_name,
                source_type="news",
                keywords=[],
                theme_type="실체형",
                summary="",
                raw_text_length=0
            )

        # 텍스트 합치기
        combined_text = f"{stock_name} 관련 뉴스:\n" + "\n".join(
            f"- {h}" for h in cleaned_headlines[:20]
        )

        # LLM으로 키워드 추출
        keywords = self._extract_keywords(combined_text, max_keywords)

        result = ExtractionResult(
            stock_code=stock_code,
            stock_name=stock_name,
            source_type="news",
            keywords=keywords,
            theme_type=self._classify_theme_type(stock_name, keywords),
            summary="",
            raw_text_length=len(combined_text)
        )

        # 캐시 저장
        if self.cache and self.use_cache:
            self.cache.set(cache_key, result.__dict__, self.cache_ttl)

        return result

    def extract_from_dart(
        self,
        stock_code: str,
        stock_name: str,
        content: str,
        max_keywords: int = 10
    ) -> ExtractionResult:
        """
        DART 사업보고서에서 키워드 추출

        Args:
            stock_code: 종목코드
            stock_name: 종목명
            content: 사업보고서 내용
            max_keywords: 최대 키워드 수

        Returns:
            ExtractionResult
        """
        # 캐시 확인
        cache_key = f"dart_keywords:{stock_code}"
        if self.cache and self.use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                self.logger.debug(f"캐시 히트: {stock_code} DART 키워드")
                return ExtractionResult(**cached)

        # 내용 정제
        cleaned_content = self.preprocessor.clean_dart_content(content)
        if not cleaned_content:
            return ExtractionResult(
                stock_code=stock_code,
                stock_name=stock_name,
                source_type="dart",
                keywords=[],
                theme_type="실체형",
                summary="",
                raw_text_length=0
            )

        # 길이 제한
        truncated = self.preprocessor.truncate(cleaned_content, max_length=3000)

        # LLM으로 키워드 추출
        keywords = self._extract_keywords(truncated, max_keywords)

        result = ExtractionResult(
            stock_code=stock_code,
            stock_name=stock_name,
            source_type="dart",
            keywords=keywords,
            theme_type=self._classify_theme_type(stock_name, keywords),
            summary="",
            raw_text_length=len(cleaned_content)
        )

        # 캐시 저장
        if self.cache and self.use_cache:
            self.cache.set(cache_key, result.__dict__, self.cache_ttl)

        return result

    def extract_all(
        self,
        stock_code: str,
        stock_name: str,
        news_headlines: list[str] | None = None,
        dart_content: str | None = None
    ) -> StockKeywords:
        """
        뉴스 + DART 통합 키워드 추출

        Args:
            stock_code: 종목코드
            stock_name: 종목명
            news_headlines: 뉴스 헤드라인 리스트
            dart_content: DART 사업보고서 내용

        Returns:
            StockKeywords
        """
        result = StockKeywords(
            stock_code=stock_code,
            stock_name=stock_name
        )

        # 뉴스 키워드
        if news_headlines:
            news_result = self.extract_from_news(
                stock_code, stock_name, news_headlines
            )
            result.news_keywords = news_result.keywords

        # DART 키워드
        if dart_content:
            dart_result = self.extract_from_dart(
                stock_code, stock_name, dart_content
            )
            result.dart_keywords = dart_result.keywords

        # 키워드 통합 (중복 제거, 우선순위: DART > 뉴스)
        combined = []
        seen = set()

        # DART 키워드 먼저 (실체 기반)
        for kw in result.dart_keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                combined.append(kw)
                seen.add(kw_lower)

        # 뉴스 키워드 추가
        for kw in result.news_keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                combined.append(kw)
                seen.add(kw_lower)

        result.combined_keywords = combined[:15]

        # 테마 타입 결정 (DART 기반 우선)
        if result.dart_keywords:
            result.theme_type = self._classify_theme_type(stock_name, result.dart_keywords)
        elif result.news_keywords:
            result.theme_type = self._classify_theme_type(stock_name, result.news_keywords)

        self.logger.debug(
            f"[{stock_code}] 키워드 추출 완료: "
            f"뉴스 {len(result.news_keywords)}개, "
            f"DART {len(result.dart_keywords)}개, "
            f"통합 {len(result.combined_keywords)}개"
        )

        return result

    def _extract_keywords(self, text: str, max_keywords: int = 10) -> list[str]:
        """LLM으로 키워드 추출"""
        client = self._get_llm_client()
        if not client:
            return self._fallback_keyword_extraction(text)

        try:
            # OllamaClient의 extract_keywords 메서드 사용
            if hasattr(client, 'extract_keywords'):
                return client.extract_keywords(text, max_keywords)
            else:
                # ClaudeCliClient 등 다른 클라이언트용 프롬프트
                return self._extract_keywords_with_prompt(client, text, max_keywords)

        except Exception as e:
            self.logger.warning(f"LLM 키워드 추출 실패, 폴백 사용: {e}")
            return self._fallback_keyword_extraction(text)

    def _extract_keywords_with_prompt(
        self,
        client,
        text: str,
        max_keywords: int
    ) -> list[str]:
        """프롬프트 기반 키워드 추출"""
        prompt = f'''다음 텍스트에서 주식 투자 관련 핵심 키워드를 추출해.

텍스트:
{text[:2000]}

규칙:
1. 산업/섹터 키워드 (예: 반도체, 바이오, 2차전지)
2. 사업 내용 키워드 (예: 신약개발, 배터리소재)
3. 최대 {max_keywords}개
4. 일반 단어 제외

키워드만 쉼표로 구분해서 출력:'''

        response = client.generate(prompt)

        keywords = []
        for part in response.replace("\n", ",").split(","):
            keyword = part.strip().strip("\"'.-")
            if keyword and 2 <= len(keyword) <= 20:
                keywords.append(keyword)

        return keywords[:max_keywords]

    def _fallback_keyword_extraction(self, text: str) -> list[str]:
        """LLM 실패 시 규칙 기반 추출"""
        # 주요 산업/테마 키워드 사전
        KEYWORD_DICT = [
            "반도체", "HBM", "파운드리", "AI", "인공지능",
            "바이오", "신약", "임상", "제약", "헬스케어",
            "2차전지", "배터리", "전기차", "수소",
            "로봇", "자율주행", "드론", "우주", "항공",
            "메타버스", "VR", "AR", "게임",
            "태양광", "풍력", "신재생", "에너지",
            "플랫폼", "핀테크", "블록체인", "클라우드",
            "자동차", "조선", "철강", "화학", "건설",
            "은행", "금융", "보험", "증권",
            "음식료", "유통", "물류", "통신",
        ]

        text_lower = text.lower()
        found = []

        for keyword in KEYWORD_DICT:
            if keyword.lower() in text_lower:
                found.append(keyword)

        return found[:10]

    def _classify_theme_type(self, stock_name: str, keywords: list[str]) -> str:
        """테마 타입 분류 (실체형 vs 기대형)"""
        client = self._get_llm_client()

        if client and hasattr(client, 'classify_theme_type'):
            try:
                return client.classify_theme_type(stock_name, keywords)
            except Exception:
                pass

        # 폴백: 규칙 기반
        EXPECTATION_KEYWORDS = [
            "바이오", "신약", "임상", "AI", "인공지능", "로봇",
            "자율주행", "우주", "메타버스", "양자", "신사업"
        ]

        for kw in keywords:
            for exp_kw in EXPECTATION_KEYWORDS:
                if exp_kw in kw:
                    return "기대형"

        return "실체형"
