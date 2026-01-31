"""
Data Ingest Layer (Layer 1)

외부 데이터 소스에서 원시 데이터 수집
"""
from src.ingest.base import BaseDataFetcher
from src.ingest.price_fetcher import PriceDataFetcher
from src.ingest.naver_theme import NaverThemeCrawler, ThemeInfo, ThemeStock
from src.ingest.theme_service import ThemeService
from src.ingest.dart_client import DartApiClient, ReportType, FinancialStatementType, FinancialData
from src.ingest.news_crawler import NewsCrawler, NewsArticle
from src.ingest.community_crawler import CommunityCrawler, CommunityPost, CommunityData
from src.ingest.discussion_crawler import DiscussionCrawler, DiscussionPost

__all__ = [
    "BaseDataFetcher",
    "PriceDataFetcher",
    "NaverThemeCrawler",
    "ThemeInfo",
    "ThemeStock",
    "ThemeService",
    "DartApiClient",
    "ReportType",
    "FinancialStatementType",
    "FinancialData",
    "NewsCrawler",
    "NewsArticle",
    "CommunityCrawler",
    "CommunityPost",
    "CommunityData",
    "DiscussionCrawler",
    "DiscussionPost",
]
