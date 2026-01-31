"""
텍스트 전처리기

HTML 태그 제거, 정규화, 중복 제거
"""
import re
from dataclasses import dataclass
from typing import Any

from src.core.logger import get_logger


@dataclass
class CleanedText:
    """정제된 텍스트 결과"""
    original: str
    cleaned: str
    word_count: int
    source_type: str  # "news", "dart", "community"


class Preprocessor:
    """
    텍스트 전처리기

    뉴스, DART 사업보고서 등의 원시 텍스트를 정제

    사용법:
        preprocessor = Preprocessor()

        # 단일 텍스트 정제
        cleaned = preprocessor.clean(raw_text, source_type="news")

        # 뉴스 헤드라인 정제
        headlines = preprocessor.clean_headlines(raw_headlines)

        # DART 사업보고서 정제
        content = preprocessor.clean_dart_content(raw_content)
    """

    # 제거할 HTML 태그 패턴
    HTML_PATTERN = re.compile(r'<[^>]+>')

    # 제거할 특수문자 (일부만)
    SPECIAL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

    # 연속 공백 정규화
    MULTI_SPACE = re.compile(r'\s+')

    # 뉴스 불필요 패턴
    NEWS_NOISE_PATTERNS = [
        re.compile(r'\[.*?기자\]'),
        re.compile(r'\(.*?기자\)'),
        re.compile(r'【.*?】'),
        re.compile(r'기자\s*='),
        re.compile(r'사진\s*='),
        re.compile(r'자료\s*='),
        re.compile(r'연합뉴스'),
        re.compile(r'뉴스1'),
        re.compile(r'한국경제'),
        re.compile(r'매일경제'),
        re.compile(r'©.*$'),
        re.compile(r'ⓒ.*$'),
    ]

    # DART 불필요 패턴
    DART_NOISE_PATTERNS = [
        re.compile(r'단위\s*:\s*원'),
        re.compile(r'단위\s*:\s*백만원'),
        re.compile(r'주\s*\d+\)'),
        re.compile(r'\(\s*주\s*\d+\s*\)'),
        re.compile(r'전자공시시스템'),
    ]

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    def clean(self, text: str, source_type: str = "general") -> CleanedText:
        """
        텍스트 정제

        Args:
            text: 원시 텍스트
            source_type: 소스 유형 ("news", "dart", "community", "general")

        Returns:
            CleanedText
        """
        if not text:
            return CleanedText(
                original="",
                cleaned="",
                word_count=0,
                source_type=source_type
            )

        original = text
        cleaned = text

        # 1. HTML 태그 제거
        cleaned = self.HTML_PATTERN.sub(' ', cleaned)

        # 2. 특수 제어 문자 제거
        cleaned = self.SPECIAL_CHARS.sub('', cleaned)

        # 3. 소스별 노이즈 제거
        if source_type == "news":
            cleaned = self._remove_news_noise(cleaned)
        elif source_type == "dart":
            cleaned = self._remove_dart_noise(cleaned)

        # 4. 연속 공백 정규화
        cleaned = self.MULTI_SPACE.sub(' ', cleaned)

        # 5. 앞뒤 공백 제거
        cleaned = cleaned.strip()

        return CleanedText(
            original=original,
            cleaned=cleaned,
            word_count=len(cleaned.split()),
            source_type=source_type
        )

    def clean_headlines(self, headlines: list[str]) -> list[str]:
        """
        뉴스 헤드라인 리스트 정제

        Args:
            headlines: 원시 헤드라인 리스트

        Returns:
            정제된 헤드라인 리스트 (중복 제거됨)
        """
        cleaned = []
        seen = set()

        for headline in headlines:
            result = self.clean(headline, source_type="news")
            if result.cleaned and result.cleaned not in seen:
                cleaned.append(result.cleaned)
                seen.add(result.cleaned)

        return cleaned

    def clean_dart_content(self, content: str) -> str:
        """
        DART 사업보고서 내용 정제

        Args:
            content: 원시 사업보고서 텍스트

        Returns:
            정제된 텍스트
        """
        result = self.clean(content, source_type="dart")
        return result.cleaned

    def _remove_news_noise(self, text: str) -> str:
        """뉴스 노이즈 제거"""
        for pattern in self.NEWS_NOISE_PATTERNS:
            text = pattern.sub('', text)
        return text

    def _remove_dart_noise(self, text: str) -> str:
        """DART 노이즈 제거"""
        for pattern in self.DART_NOISE_PATTERNS:
            text = pattern.sub('', text)
        return text

    def extract_sentences(self, text: str, min_length: int = 10) -> list[str]:
        """
        문장 단위 분리

        Args:
            text: 텍스트
            min_length: 최소 문장 길이

        Returns:
            문장 리스트
        """
        # 한국어 문장 종결 패턴
        sentences = re.split(r'[.!?]\s+', text)

        result = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) >= min_length:
                result.append(sent)

        return result

    def truncate(self, text: str, max_length: int = 2000) -> str:
        """
        텍스트 길이 제한 (LLM 입력용)

        Args:
            text: 텍스트
            max_length: 최대 길이

        Returns:
            잘린 텍스트
        """
        if len(text) <= max_length:
            return text

        # 단어 단위로 자르기
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.8:
            truncated = truncated[:last_space]

        return truncated + "..."

    def remove_duplicates(self, texts: list[str], threshold: float = 0.8) -> list[str]:
        """
        중복/유사 텍스트 제거

        Args:
            texts: 텍스트 리스트
            threshold: 유사도 임계값

        Returns:
            중복 제거된 리스트
        """
        if not texts:
            return []

        result = [texts[0]]

        for text in texts[1:]:
            is_duplicate = False
            for existing in result:
                similarity = self._simple_similarity(text, existing)
                if similarity >= threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                result.append(text)

        return result

    def _simple_similarity(self, text1: str, text2: str) -> float:
        """간단한 유사도 계산 (Jaccard)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0
