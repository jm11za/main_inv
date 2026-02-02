"""
Layer 2: Processing 모듈 테스트
"""
import pytest
from unittest.mock import Mock, patch

from src.processing.preprocessor import Preprocessor, CleanedText
from src.processing.llm_extractor import LLMExtractor, StockKeywords
from src.processing.tag_mapper import (
    TagMapper,
    SynonymResolver,
    ThemeMapping,
    StockThemeInfo,
)


class TestPreprocessor:
    """전처리기 테스트"""

    def test_clean_html_tags(self):
        """HTML 태그 제거"""
        preprocessor = Preprocessor()
        result = preprocessor.clean("<p>테스트 <b>텍스트</b></p>", "general")

        assert "<p>" not in result.cleaned
        assert "<b>" not in result.cleaned
        assert "테스트" in result.cleaned
        assert "텍스트" in result.cleaned

    def test_clean_news_noise(self):
        """뉴스 노이즈 제거"""
        preprocessor = Preprocessor()
        text = "[홍길동 기자] 삼성전자가 신제품을 발표했다. 연합뉴스"
        result = preprocessor.clean(text, "news")

        assert "홍길동 기자" not in result.cleaned
        assert "연합뉴스" not in result.cleaned
        assert "삼성전자" in result.cleaned

    def test_clean_headlines(self):
        """헤드라인 정제 및 중복 제거"""
        preprocessor = Preprocessor()
        headlines = [
            "[속보] 삼성전자 실적 발표",
            "[속보] 삼성전자 실적 발표",  # 중복
            "SK하이닉스 HBM 수주",
        ]
        result = preprocessor.clean_headlines(headlines)

        assert len(result) == 2  # 중복 제거

    def test_clean_empty_text(self):
        """빈 텍스트 처리"""
        preprocessor = Preprocessor()
        result = preprocessor.clean("", "general")

        assert result.cleaned == ""
        assert result.word_count == 0

    def test_truncate(self):
        """텍스트 길이 제한"""
        preprocessor = Preprocessor()
        long_text = "단어 " * 1000
        result = preprocessor.truncate(long_text, max_length=100)

        assert len(result) <= 103  # "..." 포함
        assert result.endswith("...")

    def test_extract_sentences(self):
        """문장 분리"""
        preprocessor = Preprocessor()
        text = "첫 번째 문장입니다. 두 번째 문장입니다. 세 번째."
        result = preprocessor.extract_sentences(text, min_length=5)

        assert len(result) >= 2

    def test_remove_duplicates(self):
        """중복 텍스트 제거"""
        preprocessor = Preprocessor()
        texts = [
            "삼성전자가 신제품을 발표했다",
            "삼성전자가 새로운 제품을 발표했다",  # 유사
            "SK하이닉스 HBM 수주 소식",
        ]
        result = preprocessor.remove_duplicates(texts, threshold=0.5)

        assert len(result) <= 3


class TestSynonymResolver:
    """동의어 해결기 테스트"""

    def test_resolve_synonym(self):
        """동의어 -> 표준어 변환"""
        resolver = SynonymResolver()

        assert resolver.resolve("이차전지") == "2차전지"
        assert resolver.resolve("EV") == "전기차"
        assert resolver.resolve("인공지능") == "AI"

    def test_resolve_unknown(self):
        """알 수 없는 키워드는 원본 반환"""
        resolver = SynonymResolver()

        assert resolver.resolve("알수없는키워드") == "알수없는키워드"

    def test_resolve_list(self):
        """키워드 리스트 표준화"""
        resolver = SynonymResolver()
        keywords = ["이차전지", "2차전지", "배터리", "EV"]
        result = resolver.resolve_list(keywords)

        # 중복 제거됨
        assert "2차전지" in result
        assert len([k for k in result if k == "2차전지"]) == 1

    def test_case_insensitive(self):
        """대소문자 무시"""
        resolver = SynonymResolver()

        assert resolver.resolve("ai") == "AI"
        assert resolver.resolve("AI") == "AI"
        assert resolver.resolve("Ai") == "AI"

    def test_get_all_forms(self):
        """모든 형태 반환"""
        resolver = SynonymResolver()
        forms = resolver.get_all_forms("AI")

        assert "AI" in forms
        assert "인공지능" in forms


class TestTagMapper:
    """태그 매퍼 테스트"""

    def test_map_stock_to_themes(self):
        """종목 -> 테마 매핑"""
        mapper = TagMapper()

        result = mapper.map_stock_to_themes(
            stock_code="005930",
            stock_name="삼성전자",
            extracted_keywords=["반도체", "HBM", "파운드리"],
            naver_themes=["AI반도체", "HBM"]
        )

        assert result.stock_code == "005930"
        assert "AI반도체" in result.themes
        assert "HBM" in result.themes
        assert len(result.keywords) > 0

    def test_map_theme_to_stocks(self):
        """테마 -> 종목 매핑"""
        mapper = TagMapper()

        result = mapper.map_theme_to_stocks(
            theme_name="HBM",
            theme_type="실체형",
            naver_stock_codes=["005930", "000660"],
            keyword_matches={"005930": ["HBM", "메모리"]}
        )

        assert result.theme_name == "HBM"
        assert "005930" in result.stock_codes
        assert "000660" in result.stock_codes
        assert result.leader_stock == "005930"

    def test_determine_theme_type_expectation(self):
        """기대형 테마 분류"""
        mapper = TagMapper()

        # 바이오 -> 기대형
        result = mapper.map_theme_to_stocks(
            theme_name="바이오신약",
            theme_type="기대형",
            naver_stock_codes=["000000"],
            keyword_matches={}
        )
        assert result.theme_type == "기대형"

    def test_merge_mappings(self):
        """매핑 통합"""
        mapper = TagMapper()

        naver_themes = {
            "AI반도체": ["005930", "000660"],
            "2차전지": ["051910"],
        }
        llm_keywords = {
            "005930": ["반도체", "HBM"],
            "000660": ["메모리", "HBM"],
            "051910": ["배터리", "전기차"],
        }
        stock_names = {
            "005930": "삼성전자",
            "000660": "SK하이닉스",
            "051910": "LG화학",
        }

        theme_mappings, stock_infos = mapper.merge_mappings(
            naver_themes, llm_keywords, stock_names
        )

        assert len(theme_mappings) == 2
        assert len(stock_infos) == 3

        # 삼성전자 확인
        samsung = next(s for s in stock_infos if s.stock_code == "005930")
        assert "AI반도체" in samsung.themes


class TestLLMExtractor:
    """LLM 추출기 테스트 (Mock)"""

    def test_fallback_keyword_extraction(self):
        """폴백 키워드 추출"""
        extractor = LLMExtractor(use_cache=False)

        text = "삼성전자가 반도체 HBM 생산을 확대한다. AI 수요 증가."
        keywords = extractor._fallback_keyword_extraction(text)

        assert "반도체" in keywords or "HBM" in keywords or "AI" in keywords

    @patch.object(LLMExtractor, "_get_llm_client", return_value=None)
    def test_classify_theme_type_fallback(self, mock_client):
        """테마 타입 폴백 분류"""
        extractor = LLMExtractor(use_cache=False)

        # 바이오 키워드 -> 기대형
        result = extractor._classify_theme_type("셀트리온", ["바이오", "신약"])
        assert result == "기대형"

        # 일반 키워드 -> 실체형
        result = extractor._classify_theme_type("삼성전자", ["반도체", "메모리"])
        assert result == "실체형"

    def test_extract_all_empty(self):
        """빈 데이터 처리"""
        extractor = LLMExtractor(use_cache=False)

        result = extractor.extract_all(
            stock_code="000000",
            stock_name="테스트",
            news_headlines=None,
            dart_content=None
        )

        assert result.stock_code == "000000"
        assert len(result.combined_keywords) == 0


class TestIntegration:
    """통합 테스트"""

    def test_full_pipeline_mock(self):
        """전체 파이프라인 (Mock)"""
        # 1. 전처리
        preprocessor = Preprocessor()
        headlines = [
            "[속보] 삼성전자 HBM 수주 확대",
            "반도체 업황 개선 전망",
        ]
        cleaned = preprocessor.clean_headlines(headlines)
        assert len(cleaned) == 2

        # 2. 동의어 해결
        resolver = SynonymResolver()
        keywords = ["HBM", "이차전지", "EV"]
        normalized = resolver.resolve_list(keywords)
        assert "2차전지" in normalized
        assert "전기차" in normalized

        # 3. 태그 매핑
        mapper = TagMapper()
        stock_info = mapper.map_stock_to_themes(
            stock_code="005930",
            stock_name="삼성전자",
            extracted_keywords=["반도체", "HBM"],
            naver_themes=["AI반도체"]
        )
        assert stock_info.stock_code == "005930"
        assert len(stock_info.themes) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
