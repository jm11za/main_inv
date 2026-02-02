"""
Data Ingest Layer 단위 테스트
"""
import pytest
import pandas as pd
from datetime import datetime


class TestPriceDataFetcher:
    """PriceDataFetcher 테스트"""

    @pytest.fixture
    def fetcher(self):
        """테스트용 fetcher 생성"""
        from src.core.config import Config
        from src.core.logger import LoggerService

        Config.reset()
        LoggerService.reset()
        LoggerService.configure(level="DEBUG", file_enabled=False)

        from src.ingest.price_fetcher import PriceDataFetcher
        return PriceDataFetcher(lookback_days=30)

    def test_get_source_name(self, fetcher):
        """소스 이름 확인"""
        assert fetcher.get_source_name() == "KRX/pykrx"

    def test_get_date_range(self, fetcher):
        """날짜 범위 계산"""
        start, end = fetcher.get_date_range(30)

        assert len(start) == 8  # YYYYMMDD
        assert len(end) == 8
        assert start < end

    @pytest.mark.integration
    def test_fetch_all_ohlcv(self, fetcher):
        """전체 종목 OHLCV 수집 (실제 API 호출)"""
        df = fetcher.fetch()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "종목코드" in df.columns

    @pytest.mark.integration
    def test_fetch_stock_ohlcv(self, fetcher):
        """특정 종목 OHLCV 수집 (삼성전자)"""
        df = fetcher.fetch_stock_ohlcv("005930")

        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "시가" in df.columns or "open" in df.columns.str.lower()
            assert "종가" in df.columns or "close" in df.columns.str.lower()

    @pytest.mark.integration
    def test_fetch_market_cap(self, fetcher):
        """시가총액 데이터 수집"""
        df = fetcher.fetch_market_cap()

        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "종목코드" in df.columns
            assert "시가총액" in df.columns

    @pytest.mark.integration
    def test_fetch_fundamental(self, fetcher):
        """기본 재무지표 수집"""
        df = fetcher.fetch_fundamental()

        assert isinstance(df, pd.DataFrame)
        # PER, PBR 등이 있어야 함
        if not df.empty:
            assert "종목코드" in df.columns


class TestNaverThemeCrawler:
    """NaverThemeCrawler 테스트"""

    @pytest.fixture
    def crawler(self):
        """테스트용 crawler 생성"""
        from src.core.config import Config
        from src.core.logger import LoggerService

        Config.reset()
        LoggerService.reset()
        LoggerService.configure(level="DEBUG", file_enabled=False)

        from src.ingest.naver_theme import NaverThemeCrawler
        return NaverThemeCrawler(min_delay=0.1, max_delay=0.3)

    def test_get_source_name(self, crawler):
        """소스 이름 확인"""
        assert crawler.get_source_name() == "NaverTheme"

    @pytest.mark.integration
    def test_fetch_theme_list(self, crawler):
        """테마 목록 수집 (실제 크롤링, 첫 페이지만)"""
        themes = list(crawler.fetch_theme_list())

        # 최소 1개 이상 테마가 있어야 함
        assert len(themes) > 0

        # ThemeInfo 구조 확인
        first_theme = themes[0]
        assert first_theme.theme_id
        assert first_theme.name

        crawler.close()

    @pytest.mark.integration
    def test_fetch_theme_stocks(self, crawler):
        """테마 소속 종목 수집 (실제 크롤링)"""
        # 먼저 테마 하나 가져오기
        themes = list(crawler.fetch_theme_list())
        if themes:
            first_theme = themes[0]
            stocks = list(crawler.fetch_theme_stocks(first_theme.theme_id, first_theme.name))

            # 종목이 있을 수도 없을 수도 있음
            if stocks:
                first_stock = stocks[0]
                assert first_stock.stock_code
                assert first_stock.stock_name

        crawler.close()


class TestBaseDataFetcher:
    """BaseDataFetcher 유틸리티 테스트"""

    def test_validate_dataframe_success(self):
        """DataFrame 검증 성공"""
        from src.ingest.base import BaseDataFetcher

        df = pd.DataFrame({
            "종목코드": ["005930"],
            "종가": [70000]
        })

        # 예외 없이 통과해야 함
        BaseDataFetcher.validate_dataframe(
            df, ["종목코드", "종가"], "테스트"
        )

    def test_validate_dataframe_missing_column(self):
        """DataFrame 필수 컬럼 누락"""
        from src.ingest.base import BaseDataFetcher
        from src.core.exceptions import IngestError

        df = pd.DataFrame({
            "종목코드": ["005930"]
        })

        with pytest.raises(IngestError) as exc_info:
            BaseDataFetcher.validate_dataframe(
                df, ["종목코드", "종가"], "테스트"
            )

        assert "필수 컬럼 누락" in str(exc_info.value)

    def test_validate_dataframe_empty(self):
        """빈 DataFrame 검증"""
        from src.ingest.base import BaseDataFetcher
        from src.core.exceptions import IngestError

        df = pd.DataFrame()

        with pytest.raises(IngestError) as exc_info:
            BaseDataFetcher.validate_dataframe(df, ["종목코드"], "테스트")

        assert "빈 DataFrame" in str(exc_info.value)


class TestNewsCrawler:
    """NewsCrawler 테스트"""

    @pytest.fixture
    def crawler(self):
        """테스트용 crawler 생성"""
        from src.core.config import Config
        from src.core.logger import LoggerService

        Config.reset()
        LoggerService.reset()
        LoggerService.configure(level="DEBUG", file_enabled=False)

        from src.ingest.news_crawler import NewsCrawler
        return NewsCrawler(max_articles=10, delay_seconds=0.5)

    def test_get_source_name(self, crawler):
        """소스 이름 확인"""
        assert crawler.get_source_name() == "NaverNews"

    def test_news_article_dataclass(self):
        """NewsArticle 데이터클래스 확인"""
        from src.ingest.news_crawler import NewsArticle

        article = NewsArticle(
            stock_code="005930",
            title="삼성전자 실적 발표",
            link="https://example.com/news/1",
            source="한국경제",
            published_at="2025.01.31",
        )

        assert article.stock_code == "005930"
        assert "삼성전자" in article.title

    @pytest.mark.integration
    def test_fetch_stock_news(self, crawler):
        """종목 뉴스 수집 (실제 크롤링)"""
        articles = crawler.fetch_stock_news("005930", max_pages=1)

        assert isinstance(articles, list)
        if articles:
            first = articles[0]
            assert first.stock_code == "005930"
            assert first.title
            assert first.link

        crawler.close()


class TestDartApiClient:
    """DartApiClient 테스트"""

    @pytest.fixture
    def client(self):
        """테스트용 client 생성"""
        from src.core.config import Config
        from src.core.logger import LoggerService

        Config.reset()
        LoggerService.reset()
        LoggerService.configure(level="DEBUG", file_enabled=False)

        from src.ingest.dart_client import DartApiClient
        return DartApiClient()

    def test_get_source_name(self, client):
        """소스 이름 확인"""
        assert client.get_source_name() == "DART"

    def test_report_type_enum(self):
        """ReportType Enum 확인"""
        from src.ingest.dart_client import ReportType

        assert ReportType.ANNUAL.value == "11011"
        assert ReportType.Q1.value == "11013"

    def test_financial_data_calculation(self):
        """FinancialData 자동 계산 확인"""
        from src.ingest.dart_client import FinancialData

        data = FinancialData(
            stock_code="005930",
            stock_name="삼성전자",
            report_type="11011",
            year=2024,
            quarter=4,
            total_liabilities=100000,
            total_equity=200000,
            current_assets=50000,
            current_liabilities=25000,
        )

        # 부채비율 = 100000/200000 * 100 = 50%
        assert data.debt_ratio == 50.0

        # 유동비율 = 50000/25000 * 100 = 200%
        assert data.current_ratio == 200.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not integration"])
