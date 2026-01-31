"""
네이버 뉴스 크롤러

종목별 최근 뉴스 헤드라인 수집
"""
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Generator

import requests
from bs4 import BeautifulSoup

from src.ingest.base import BaseDataFetcher
from src.core.config import get_config
from src.core.exceptions import CrawlerError


@dataclass
class NewsArticle:
    """뉴스 기사 정보"""
    stock_code: str
    title: str
    link: str
    source: str  # 언론사
    published_at: str  # 발행일
    summary: str = ""  # 요약 (있는 경우)

    def __repr__(self):
        return f"<News [{self.source}] {self.title[:30]}...>"


class NewsCrawler(BaseDataFetcher):
    """
    네이버 뉴스 크롤러

    종목별 관련 뉴스 헤드라인 수집 (네이버 금융 뉴스 탭)

    사용법:
        crawler = NewsCrawler()

        # 특정 종목 뉴스 수집
        articles = crawler.fetch_stock_news("005930")

        # 여러 종목 뉴스 수집
        all_articles = crawler.fetch_multiple_stocks(["005930", "000660"])
    """

    # 네이버 금융 종목 뉴스 URL
    NEWS_URL = "https://finance.naver.com/item/news_news.naver"

    def __init__(
        self,
        max_articles: int | None = None,
        delay_seconds: float | None = None
    ):
        super().__init__()

        config = get_config()
        self.max_articles = max_articles or config.get(
            "ingest.news.max_articles_per_stock", 30
        )
        self.delay = delay_seconds or config.get(
            "ingest.naver_theme.delay_seconds", 1.0
        )

        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "ko-KR,ko;q=0.9",
            "Referer": "https://finance.naver.com/",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def get_source_name(self) -> str:
        return "NaverNews"

    def fetch(self) -> list[NewsArticle]:
        """기본 fetch는 미구현 (종목 지정 필요)"""
        raise NotImplementedError("fetch_stock_news() 사용")

    def fetch_stock_news(
        self,
        stock_code: str,
        max_pages: int = 5
    ) -> list[NewsArticle]:
        """
        특정 종목의 뉴스 수집

        Args:
            stock_code: 종목코드 (예: "005930")
            max_pages: 최대 페이지 수

        Returns:
            NewsArticle 리스트
        """
        self.logger.debug(f"뉴스 수집 시작: {stock_code}")

        articles = []
        page = 1

        while page <= max_pages and len(articles) < self.max_articles:
            try:
                page_articles = list(self._fetch_news_page(stock_code, page))

                if not page_articles:
                    break

                articles.extend(page_articles)
                page += 1

                time.sleep(self.delay * 0.5)

            except Exception as e:
                self.logger.warning(f"뉴스 페이지 {page} 수집 실패: {e}")
                break

        # max_articles 제한 적용
        articles = articles[:self.max_articles]

        self.logger.debug(f"[{stock_code}] 뉴스 {len(articles)}건 수집 완료")
        return articles

    def _fetch_news_page(
        self,
        stock_code: str,
        page: int
    ) -> Generator[NewsArticle, None, None]:
        """뉴스 페이지 파싱"""
        params = {
            "code": stock_code,
            "page": page,
            "sm": "title_entity_id.basic",
            "clusterId": "",
        }

        try:
            response = self.session.get(
                self.NEWS_URL,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            response.encoding = "euc-kr"

            soup = BeautifulSoup(response.text, "lxml")

            # 뉴스 테이블 찾기
            news_table = soup.select_one("table.type5")
            if not news_table:
                return

            rows = news_table.select("tr")

            for row in rows:
                # 제목 셀
                title_cell = row.select_one("td.title")
                if not title_cell:
                    continue

                link_tag = title_cell.select_one("a")
                if not link_tag:
                    continue

                title = link_tag.get_text(strip=True)
                href = link_tag.get("href", "")

                # 링크 정규화
                if href.startswith("/"):
                    link = f"https://finance.naver.com{href}"
                else:
                    link = href

                # 언론사
                source_cell = row.select_one("td.info")
                source = source_cell.get_text(strip=True) if source_cell else ""

                # 날짜
                date_cell = row.select_one("td.date")
                published_at = date_cell.get_text(strip=True) if date_cell else ""

                yield NewsArticle(
                    stock_code=stock_code,
                    title=title,
                    link=link,
                    source=source,
                    published_at=published_at,
                )

        except requests.RequestException as e:
            self.logger.error(f"뉴스 페이지 요청 실패: {e}")
            raise CrawlerError(f"뉴스 크롤링 실패: {e}")

    def fetch_multiple_stocks(
        self,
        stock_codes: list[str],
        max_articles_per_stock: int | None = None
    ) -> dict[str, list[NewsArticle]]:
        """
        여러 종목의 뉴스 일괄 수집

        Args:
            stock_codes: 종목코드 리스트
            max_articles_per_stock: 종목당 최대 기사 수

        Returns:
            {종목코드: [NewsArticle, ...], ...}
        """
        self._log_fetch_start()

        if max_articles_per_stock:
            original_max = self.max_articles
            self.max_articles = max_articles_per_stock

        result = {}
        total_count = 0

        for i, code in enumerate(stock_codes):
            try:
                articles = self.fetch_stock_news(code)
                result[code] = articles
                total_count += len(articles)

                self.logger.debug(
                    f"[{i+1}/{len(stock_codes)}] {code}: {len(articles)}건"
                )

            except Exception as e:
                self.logger.warning(f"[{code}] 뉴스 수집 실패: {e}")
                result[code] = []

            time.sleep(self.delay)

        if max_articles_per_stock:
            self.max_articles = original_max

        self._log_fetch_complete(count=total_count)
        return result

    def fetch_news_for_analysis(
        self,
        stock_code: str,
        days: int = 30
    ) -> str:
        """
        LLM 분석용 뉴스 텍스트 생성

        Args:
            stock_code: 종목코드
            days: 수집 기간 (일)

        Returns:
            뉴스 헤드라인 텍스트 (LLM 분석용)
        """
        articles = self.fetch_stock_news(stock_code)

        if not articles:
            return f"[{stock_code}] 최근 뉴스 없음"

        # 헤드라인 텍스트 생성
        lines = [f"[{stock_code}] 최근 뉴스 헤드라인 ({len(articles)}건):\n"]

        for i, article in enumerate(articles, 1):
            lines.append(
                f"{i}. [{article.published_at}] {article.title} ({article.source})"
            )

        return "\n".join(lines)

    def close(self):
        """세션 종료"""
        self.session.close()
