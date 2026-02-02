"""
StockThemeAnalyzer 테스트 (v2.0)

새로운 테마 기반 종목 분류 시스템 테스트
"""
import pytest
from dataclasses import dataclass
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.sector.classifier import (
    ThemeData,
    StockAnalysisData,
    StockThemeAnalyzer,
    SectorClassifier,  # deprecated alias
)


# =============================================================================
# Mock 데이터 클래스 (크롤링 결과 시뮬레이션)
# =============================================================================

@dataclass
class MockThemeInfo:
    """테마 정보 Mock (naver_theme.py의 ThemeInfo 대체)"""
    theme_id: str
    name: str
    change_rate: float = 0.0


@dataclass
class MockThemeStock:
    """테마-종목 매핑 Mock (naver_theme.py의 ThemeStock 대체)"""
    theme_id: str
    stock_code: str
    stock_name: str = ""


# =============================================================================
# ThemeData 테스트
# =============================================================================

class TestThemeData:
    """ThemeData 데이터클래스 테스트"""

    def test_create_basic_theme(self):
        """기본 테마 생성"""
        theme = ThemeData(
            theme_id="12345",
            theme_name="2차전지",
        )

        assert theme.theme_id == "12345"
        assert theme.theme_name == "2차전지"
        assert theme.change_rate == 0.0
        assert theme.stock_count == 0
        assert theme.stock_codes == []

    def test_create_full_theme(self):
        """모든 필드 포함 테마 생성"""
        theme = ThemeData(
            theme_id="12345",
            theme_name="2차전지(소재)",
            change_rate=2.5,
            stock_count=15,
            stock_codes=["005930", "373220", "003670"],
        )

        assert theme.theme_name == "2차전지(소재)"
        assert theme.change_rate == 2.5
        assert theme.stock_count == 15
        assert len(theme.stock_codes) == 3

    def test_to_dict(self):
        """to_dict 변환 테스트"""
        theme = ThemeData(
            theme_id="12345",
            theme_name="반도체",
            change_rate=1.234567,
            stock_count=10,
            stock_codes=["005930"],
        )

        result = theme.to_dict()

        assert result["theme_id"] == "12345"
        assert result["theme_name"] == "반도체"
        assert result["change_rate"] == 1.23  # 소수점 2자리 반올림
        assert result["stock_count"] == 10
        assert result["stock_codes"] == ["005930"]


# =============================================================================
# StockAnalysisData 테스트
# =============================================================================

class TestStockAnalysisData:
    """StockAnalysisData 데이터클래스 테스트"""

    def test_create_basic_stock(self):
        """기본 종목 생성"""
        stock = StockAnalysisData(
            stock_code="005930",
            stock_name="삼성전자",
        )

        assert stock.stock_code == "005930"
        assert stock.stock_name == "삼성전자"
        assert stock.theme_tags == []
        assert stock.business_summary == ""
        assert stock.news_summary == ""
        assert stock.data_sources == []

    def test_create_full_stock(self):
        """모든 필드 포함 종목 생성"""
        stock = StockAnalysisData(
            stock_code="005930",
            stock_name="삼성전자",
            theme_tags=["반도체", "HBM", "AI"],
            business_summary="반도체 메모리 및 시스템 LSI 제조업체.",
            news_summary="HBM 수요 급증으로 실적 개선 전망.",
            data_sources=["theme", "dart", "news"],
        )

        assert len(stock.theme_tags) == 3
        assert "HBM" in stock.theme_tags
        assert "반도체" in stock.business_summary
        assert len(stock.data_sources) == 3

    def test_to_dict(self):
        """to_dict 변환 테스트"""
        stock = StockAnalysisData(
            stock_code="005930",
            stock_name="삼성전자",
            theme_tags=["반도체"],
            business_summary="테스트",
            data_sources=["theme"],
        )

        result = stock.to_dict()

        assert result["stock_code"] == "005930"
        assert result["stock_name"] == "삼성전자"
        assert result["theme_tags"] == ["반도체"]
        assert result["business_summary"] == "테스트"
        assert "created_at" in result

    def test_multiple_themes(self):
        """한 종목이 여러 테마에 속함 (N:M 관계)"""
        stock = StockAnalysisData(
            stock_code="005930",
            stock_name="삼성전자",
            theme_tags=["반도체", "HBM", "AI칩", "데이터센터", "스마트폰"],
        )

        # 5개 테마 허용
        assert len(stock.theme_tags) == 5
        assert "반도체" in stock.theme_tags
        assert "HBM" in stock.theme_tags


# =============================================================================
# StockThemeAnalyzer 테스트
# =============================================================================

class TestStockThemeAnalyzer:
    """StockThemeAnalyzer 클래스 테스트"""

    @pytest.fixture
    def analyzer(self):
        """LLM 없는 분석기"""
        with patch("src.sector.classifier.get_config") as mock_config, \
             patch("src.sector.classifier.get_logger") as mock_logger:
            return StockThemeAnalyzer(llm_client=None)

    def test_init(self, analyzer):
        """초기화 테스트"""
        assert analyzer._llm_client is None

    def test_set_llm_client(self, analyzer):
        """LLM 클라이언트 설정"""
        mock_client = MagicMock()
        analyzer.set_llm_client(mock_client)

        assert analyzer._llm_client == mock_client


class TestBuildThemesData:
    """build_themes_data 메서드 테스트"""

    @pytest.fixture
    def analyzer(self):
        with patch("src.sector.classifier.get_config"), \
             patch("src.sector.classifier.get_logger"):
            return StockThemeAnalyzer(llm_client=None)

    def test_build_themes_basic(self, analyzer):
        """기본 테마 데이터 구축"""
        themes = [
            MockThemeInfo(theme_id="1", name="2차전지", change_rate=2.5),
            MockThemeInfo(theme_id="2", name="반도체", change_rate=-1.0),
        ]
        theme_stocks = [
            MockThemeStock(theme_id="1", stock_code="005930"),
            MockThemeStock(theme_id="1", stock_code="373220"),
            MockThemeStock(theme_id="2", stock_code="005930"),
        ]

        result = analyzer.build_themes_data(themes, theme_stocks)

        assert len(result) == 2

        battery_theme = next(t for t in result if t.theme_name == "2차전지")
        assert battery_theme.stock_count == 2
        assert "005930" in battery_theme.stock_codes
        assert "373220" in battery_theme.stock_codes

        semi_theme = next(t for t in result if t.theme_name == "반도체")
        assert semi_theme.stock_count == 1
        assert semi_theme.change_rate == -1.0

    def test_build_themes_empty(self, analyzer):
        """빈 테마 처리"""
        result = analyzer.build_themes_data([], [])
        assert result == []


class TestBuildThemeMapsWithNames:
    """build_theme_maps_with_names 메서드 테스트"""

    @pytest.fixture
    def analyzer(self):
        with patch("src.sector.classifier.get_config"), \
             patch("src.sector.classifier.get_logger"):
            return StockThemeAnalyzer(llm_client=None)

    def test_build_maps(self, analyzer):
        """테마-종목 매핑 구축"""
        themes = [
            MockThemeInfo(theme_id="1", name="2차전지"),
            MockThemeInfo(theme_id="2", name="반도체"),
        ]
        theme_stocks = [
            MockThemeStock(theme_id="1", stock_code="005930"),
            MockThemeStock(theme_id="1", stock_code="373220"),
            MockThemeStock(theme_id="2", stock_code="005930"),  # 삼성전자는 두 테마에 속함
        ]

        theme_stocks_map, stock_themes_map = analyzer.build_theme_maps_with_names(themes, theme_stocks)

        # 테마 → 종목
        assert "2차전지" in theme_stocks_map
        assert len(theme_stocks_map["2차전지"]) == 2
        assert "반도체" in theme_stocks_map
        assert len(theme_stocks_map["반도체"]) == 1

        # 종목 → 테마 (N:M 관계 확인)
        assert "005930" in stock_themes_map
        assert len(stock_themes_map["005930"]) == 2  # 삼성전자는 2개 테마
        assert "2차전지" in stock_themes_map["005930"]
        assert "반도체" in stock_themes_map["005930"]

        assert "373220" in stock_themes_map
        assert len(stock_themes_map["373220"]) == 1  # LG에너지솔루션은 1개 테마

    def test_no_duplicate_in_maps(self, analyzer):
        """중복 제거 확인"""
        themes = [MockThemeInfo(theme_id="1", name="2차전지")]
        theme_stocks = [
            MockThemeStock(theme_id="1", stock_code="005930"),
            MockThemeStock(theme_id="1", stock_code="005930"),  # 중복
        ]

        theme_stocks_map, stock_themes_map = analyzer.build_theme_maps_with_names(themes, theme_stocks)

        # 중복 제거됨
        assert len(theme_stocks_map["2차전지"]) == 1
        assert len(stock_themes_map["005930"]) == 1


class TestBuildStocksData:
    """build_stocks_data 메서드 테스트"""

    @pytest.fixture
    def analyzer(self):
        with patch("src.sector.classifier.get_config"), \
             patch("src.sector.classifier.get_logger"):
            return StockThemeAnalyzer(llm_client=None)

    def test_build_stocks_basic(self, analyzer):
        """기본 종목 데이터 구축"""
        stock_codes = ["005930", "373220"]
        stock_names = {"005930": "삼성전자", "373220": "LG에너지솔루션"}
        stock_themes_map = {
            "005930": ["반도체", "AI"],
            "373220": ["2차전지"],
        }

        result = analyzer.build_stocks_data(
            stock_codes=stock_codes,
            stock_names=stock_names,
            stock_themes_map=stock_themes_map,
        )

        assert len(result) == 2

        samsung = next(s for s in result if s.stock_code == "005930")
        assert samsung.stock_name == "삼성전자"
        assert len(samsung.theme_tags) == 2
        assert "반도체" in samsung.theme_tags
        assert "AI" in samsung.theme_tags
        assert "theme" in samsung.data_sources

        lg = next(s for s in result if s.stock_code == "373220")
        assert len(lg.theme_tags) == 1
        assert "2차전지" in lg.theme_tags

    def test_build_stocks_with_dart(self, analyzer):
        """DART 데이터 포함"""
        stock_codes = ["005930"]
        stock_names = {"005930": "삼성전자"}
        stock_themes_map = {"005930": ["반도체"]}
        dart_data = {"005930": "당사는 반도체 메모리 제조업체입니다."}

        result = analyzer.build_stocks_data(
            stock_codes=stock_codes,
            stock_names=stock_names,
            stock_themes_map=stock_themes_map,
            dart_data=dart_data,
        )

        assert len(result) == 1
        assert "dart" in result[0].data_sources
        # LLM이 없으므로 요약은 비어있음
        assert result[0].business_summary == ""

    def test_build_stocks_with_news(self, analyzer):
        """뉴스 데이터 포함"""
        stock_codes = ["005930"]
        stock_names = {"005930": "삼성전자"}
        stock_themes_map = {"005930": ["반도체"]}
        news_data = {"005930": [{"title": "삼성전자 수주", "summary": "대규모 수주"}]}

        result = analyzer.build_stocks_data(
            stock_codes=stock_codes,
            stock_names=stock_names,
            stock_themes_map=stock_themes_map,
            news_data=news_data,
        )

        assert len(result) == 1
        assert "news" in result[0].data_sources

    def test_build_stocks_empty(self, analyzer):
        """빈 종목 리스트 처리"""
        result = analyzer.build_stocks_data(
            stock_codes=[],
            stock_names={},
            stock_themes_map={},
        )
        assert result == []


class TestAnalyzeStock:
    """analyze_stock 메서드 테스트"""

    @pytest.fixture
    def analyzer(self):
        with patch("src.sector.classifier.get_config"), \
             patch("src.sector.classifier.get_logger"):
            return StockThemeAnalyzer(llm_client=None)

    def test_analyze_with_themes_only(self, analyzer):
        """테마만 있는 경우"""
        result = analyzer.analyze_stock(
            stock_code="005930",
            stock_name="삼성전자",
            theme_tags=["반도체", "HBM"],
        )

        assert result.stock_code == "005930"
        assert result.theme_tags == ["반도체", "HBM"]
        assert "theme" in result.data_sources
        assert result.business_summary == ""
        assert result.news_summary == ""

    def test_analyze_with_all_data(self, analyzer):
        """모든 데이터 소스"""
        result = analyzer.analyze_stock(
            stock_code="005930",
            stock_name="삼성전자",
            theme_tags=["반도체"],
            dart_text="사업개요 텍스트",
            news_articles=[{"title": "뉴스"}],
        )

        assert "theme" in result.data_sources
        assert "dart" in result.data_sources
        assert "news" in result.data_sources

    def test_analyze_no_data(self, analyzer):
        """데이터 없는 경우"""
        result = analyzer.analyze_stock(
            stock_code="000000",
            stock_name="테스트종목",
        )

        assert result.theme_tags == []
        assert result.data_sources == []


class TestLLMSummarization:
    """LLM 요약 기능 테스트"""

    @pytest.fixture
    def analyzer_with_llm(self):
        with patch("src.sector.classifier.get_config"), \
             patch("src.sector.classifier.get_logger"):
            mock_llm = MagicMock()
            mock_llm.generate.return_value = "요약 결과입니다."
            return StockThemeAnalyzer(llm_client=mock_llm)

    def test_summarize_business(self, analyzer_with_llm):
        """사업개요 요약"""
        result = analyzer_with_llm._summarize_business(
            "삼성전자",
            "당사는 반도체 메모리를 제조하는 회사입니다."
        )

        assert result == "요약 결과입니다."
        analyzer_with_llm._llm_client.generate.assert_called_once()

    def test_summarize_news(self, analyzer_with_llm):
        """뉴스 요약"""
        news = [
            {"title": "삼성전자 수주", "summary": "대규모 수주 체결"},
        ]

        result = analyzer_with_llm._summarize_news("삼성전자", news)

        assert result == "요약 결과입니다."

    def test_summarize_with_no_llm(self):
        """LLM 없을 때 빈 문자열"""
        with patch("src.sector.classifier.get_config"), \
             patch("src.sector.classifier.get_logger"):
            analyzer = StockThemeAnalyzer(llm_client=None)

            result = analyzer._summarize_business("테스트", "텍스트")
            assert result == ""

    def test_summarize_llm_error(self, analyzer_with_llm):
        """LLM 오류 시 빈 문자열"""
        analyzer_with_llm._llm_client.generate.side_effect = Exception("LLM Error")

        result = analyzer_with_llm._summarize_business("테스트", "텍스트")
        assert result == ""


class TestUtilityMethods:
    """유틸리티 메서드 테스트"""

    @pytest.fixture
    def analyzer(self):
        with patch("src.sector.classifier.get_config"), \
             patch("src.sector.classifier.get_logger"):
            return StockThemeAnalyzer(llm_client=None)

    def test_get_themes_by_stock(self, analyzer):
        """종목별 테마 조회"""
        stock_themes_map = {
            "005930": ["반도체", "AI"],
            "373220": ["2차전지"],
        }

        result = analyzer.get_themes_by_stock("005930", stock_themes_map)
        assert result == ["반도체", "AI"]

        result = analyzer.get_themes_by_stock("000000", stock_themes_map)
        assert result == []

    def test_get_stocks_by_theme(self, analyzer):
        """테마별 종목 조회"""
        theme_stocks_map = {
            "반도체": ["005930", "000660"],
            "2차전지": ["373220"],
        }

        result = analyzer.get_stocks_by_theme("반도체", theme_stocks_map)
        assert result == ["005930", "000660"]

        result = analyzer.get_stocks_by_theme("없는테마", theme_stocks_map)
        assert result == []

    def test_summarize_themes(self, analyzer):
        """테마 데이터 요약"""
        themes_data = [
            ThemeData(theme_id="1", theme_name="2차전지", change_rate=3.5, stock_count=20),
            ThemeData(theme_id="2", theme_name="반도체", change_rate=2.0, stock_count=15),
            ThemeData(theme_id="3", theme_name="바이오", change_rate=-1.0, stock_count=10),
        ]

        result = analyzer.summarize_themes(themes_data)

        assert result["total_themes"] == 3
        assert result["total_stocks"] == 45
        assert result["avg_stocks_per_theme"] == 15.0
        assert "2차전지" in result["top_themes_by_change"]

    def test_summarize_stocks(self, analyzer):
        """종목 데이터 요약"""
        stocks_data = [
            StockAnalysisData(
                stock_code="005930",
                stock_name="삼성전자",
                theme_tags=["반도체", "AI"],
                business_summary="요약",
                data_sources=["theme", "dart"],
            ),
            StockAnalysisData(
                stock_code="373220",
                stock_name="LG에너지솔루션",
                theme_tags=["2차전지"],
                data_sources=["theme", "news"],
            ),
        ]

        result = analyzer.summarize_stocks(stocks_data)

        assert result["total_stocks"] == 2
        assert result["with_dart_data"] == 1
        assert result["with_news_data"] == 1
        assert result["with_llm_summary"] == 1


# =============================================================================
# 하위 호환성 테스트
# =============================================================================

class TestBackwardCompatibility:
    """하위 호환성 테스트"""

    def test_sector_classifier_alias(self):
        """SectorClassifier 별칭 확인"""
        assert SectorClassifier is StockThemeAnalyzer

    def test_create_via_alias(self):
        """별칭으로 인스턴스 생성"""
        with patch("src.sector.classifier.get_config"), \
             patch("src.sector.classifier.get_logger"):
            analyzer = SectorClassifier(llm_client=None)
            assert isinstance(analyzer, StockThemeAnalyzer)


# =============================================================================
# 통합 테스트
# =============================================================================

class TestIntegration:
    """통합 테스트"""

    @pytest.fixture
    def analyzer(self):
        with patch("src.sector.classifier.get_config"), \
             patch("src.sector.classifier.get_logger"):
            return StockThemeAnalyzer(llm_client=None)

    def test_full_workflow(self, analyzer):
        """전체 워크플로우 테스트"""
        # 1. 테마 데이터
        themes = [
            MockThemeInfo(theme_id="1", name="2차전지", change_rate=2.5),
            MockThemeInfo(theme_id="2", name="반도체", change_rate=1.0),
            MockThemeInfo(theme_id="3", name="AI", change_rate=3.0),
        ]
        theme_stocks = [
            MockThemeStock(theme_id="1", stock_code="373220"),
            MockThemeStock(theme_id="2", stock_code="005930"),
            MockThemeStock(theme_id="3", stock_code="005930"),  # 삼성전자: 반도체 + AI
        ]

        # 2. 테마 데이터 구축
        themes_data = analyzer.build_themes_data(themes, theme_stocks)
        assert len(themes_data) == 3

        # 3. 매핑 구축
        theme_stocks_map, stock_themes_map = analyzer.build_theme_maps_with_names(themes, theme_stocks)

        # N:M 관계 확인
        assert len(stock_themes_map["005930"]) == 2  # 삼성전자는 2개 테마

        # 4. 종목 분석 데이터 구축
        stock_codes = ["005930", "373220"]
        stock_names = {"005930": "삼성전자", "373220": "LG에너지솔루션"}

        stocks_data = analyzer.build_stocks_data(
            stock_codes=stock_codes,
            stock_names=stock_names,
            stock_themes_map=stock_themes_map,
        )

        assert len(stocks_data) == 2

        # 삼성전자 확인
        samsung = next(s for s in stocks_data if s.stock_code == "005930")
        assert len(samsung.theme_tags) == 2
        assert "반도체" in samsung.theme_tags
        assert "AI" in samsung.theme_tags

        # LG에너지솔루션 확인
        lg = next(s for s in stocks_data if s.stock_code == "373220")
        assert len(lg.theme_tags) == 1
        assert "2차전지" in lg.theme_tags

        # 5. 요약
        summary = analyzer.summarize_stocks(stocks_data)
        assert summary["total_stocks"] == 2
