"""
네이버 뉴스 검색 크롤러

네이버 통합검색의 뉴스 탭에서 종목 관련 뉴스를 수집
- 헤드라인 + 요약(description) 수집
- 기간 필터링 지원 (1일, 1주, 1개월, 6개월 등)
"""
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Generator

import requests
from bs4 import BeautifulSoup

from src.core.logger import get_logger


@dataclass
class NaverNewsArticle:
    """네이버 뉴스 검색 결과 기사"""
    query: str              # 검색어 (종목명)
    title: str              # 헤드라인
    link: str               # 기사 URL
    summary: str            # 요약/미리보기
    press: str              # 언론사
    published_at: str       # 발행일 (예: "3시간 전", "2026.01.30")

    def __repr__(self):
        return f"<NaverNews [{self.press}] {self.title[:30]}...>"

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "title": self.title,
            "link": self.link,
            "summary": self.summary,
            "press": self.press,
            "published_at": self.published_at,
        }


class NaverNewsSearchCrawler:
    """
    네이버 뉴스 검색 크롤러

    사용법:
        crawler = NaverNewsSearchCrawler()

        # 종목명으로 뉴스 검색 (최근 1개월)
        articles = crawler.search("성우전자", period="1m")

        for article in articles:
            print(f"{article.title}")
            print(f"  요약: {article.summary}")
            print(f"  언론사: {article.press} | {article.published_at}")
    """

    SEARCH_URL = "https://search.naver.com/search.naver"

    # 기간 필터 옵션 (nso 파라미터용)
    PERIOD_OPTIONS = {
        "1d": "1d",   # 1일
        "1w": "1w",   # 1주
        "1m": "1m",   # 1개월
        "6m": "6m",   # 6개월
        "1y": "1y",   # 1년
        "all": "all", # 전체
    }

    def __init__(self, delay_seconds: float = 2.0):
        self.logger = get_logger(self.__class__.__name__)
        self.delay = delay_seconds

        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "ko-KR,ko;q=0.9",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def search(
        self,
        query: str,
        period: str = "1m",
        sort: str = "recent",
        max_results: int = 20,
    ) -> list[NaverNewsArticle]:
        """
        네이버 뉴스 검색

        Args:
            query: 검색어 (종목명)
            period: 기간 필터 ("1d", "1w", "1m", "6m", "1y")
            sort: 정렬 ("recent"=최신순, "relevant"=관련도순)
            max_results: 최대 결과 수

        Returns:
            NaverNewsArticle 리스트
        """
        self.logger.info(f"뉴스 검색 시작: {query} (기간: {period})")

        # nso 파라미터: so=정렬(r:관련도,d:최신), p=기간
        period_code = self.PERIOD_OPTIONS.get(period, "1m")
        sort_code = "d" if sort == "recent" else "r"
        nso_param = f"so:{sort_code},p:{period_code}"

        params = {
            "ssc": "tab.news.all",
            "query": query,
            "sm": "tab_opt",
            "sort": "1" if sort == "recent" else "0",
            "photo": "0",
            "field": "0",
            "pd": "0",
            "mynews": "0",
            "office_type": "0",
            "office_section_code": "0",
            "nso": nso_param,
            "is_sug_officeid": "0",
            "office_category": "0",
            "service_area": "0",
        }

        try:
            response = self.session.get(
                self.SEARCH_URL,
                params=params,
                timeout=15
            )
            response.raise_for_status()

            articles = list(self._parse_search_results(response.text, query))
            articles = articles[:max_results]

            self.logger.info(f"[{query}] 뉴스 {len(articles)}건 검색 완료")
            return articles

        except requests.RequestException as e:
            self.logger.error(f"뉴스 검색 실패 [{query}]: {e}")
            return []

    def _parse_search_results(
        self,
        html: str,
        query: str
    ) -> Generator[NaverNewsArticle, None, None]:
        """검색 결과 HTML 파싱"""
        soup = BeautifulSoup(html, "lxml")

        # 여러 뉴스 링크 패턴 지원
        # 1. 외부 언론사 링크 (articleView)
        # 2. 네이버 뉴스 링크 (n.news.naver)
        all_links = soup.find_all("a", href=True)

        seen_links = set()  # 중복 제거용

        for link in all_links:
            href = link.get("href", "")

            # 뉴스 링크 패턴 확인
            is_news_link = (
                "article" in href or
                "n.news.naver" in href
            )
            if not is_news_link:
                continue

            # 중복 체크
            if href in seen_links:
                continue
            seen_links.add(href)

            # 제목 추출
            title = link.get_text(strip=True)
            if len(title) < 15:  # 너무 짧은 건 제목이 아님
                continue

            # 홍보성/시스템 링크 제외
            if "언론사가 선정한" in title or "네이버 메인에서" in title:
                continue

            # 부모 컨테이너 찾기 (뉴스 아이템 단위)
            container = self._find_news_container(link)

            # 요약 추출
            summary = self._extract_summary(container, title)

            # 언론사/날짜 추출 (sds-comps-profile 구조)
            press, published_at = self._extract_press_and_date(link, container)

            yield NaverNewsArticle(
                query=query,
                title=title,
                link=href,
                summary=summary,
                press=press,
                published_at=published_at,
            )

    def _find_news_container(self, link_element):
        """뉴스 아이템 컨테이너 찾기"""
        # 부모를 타고 올라가면서 적절한 컨테이너 찾기
        parent = link_element.parent
        for _ in range(8):
            if parent is None:
                return None

            classes = parent.get("class", [])
            class_str = " ".join(classes) if classes else ""

            # 뉴스 아이템 컨테이너로 추정되는 패턴
            if any(keyword in class_str for keyword in [
                "news-item", "fds-news-item", "sds-comps-vertical-layout"
            ]):
                return parent

            # sds-comps-profile을 포함하는 컨테이너 (언론사/날짜 정보 있음)
            if parent.select_one('div.sds-comps-profile, [class*="sds-comps-profile"]'):
                return parent

            # div 태그이고 충분한 내용이 있으면 컨테이너일 가능성
            if parent.name == "div":
                text_len = len(parent.get_text(strip=True))
                if text_len > 100:
                    return parent

            parent = parent.parent

        return link_element.parent  # 못 찾으면 바로 위 부모

    def _extract_summary(self, container, title: str) -> str:
        """요약/미리보기 추출"""
        if container is None:
            return ""

        # 여러 패턴 시도
        selectors = [
            'div[class*="dsc"]',
            'a[class*="dsc"]',
            'span[class*="summary"]',
            'p',
        ]

        for selector in selectors:
            elem = container.select_one(selector)
            if elem:
                text = elem.get_text(strip=True)
                # 제목과 다르고 충분히 긴 텍스트
                if text != title and len(text) > 20:
                    return text[:300]  # 최대 300자

        # 컨테이너 내 모든 텍스트에서 제목 제외하고 가장 긴 것
        all_texts = []
        for elem in container.find_all(string=True):
            text = elem.strip()
            if text and text != title and len(text) > 30:
                all_texts.append(text)

        if all_texts:
            return max(all_texts, key=len)[:300]

        return ""

    def _extract_press_and_date(self, link_element, container) -> tuple[str, str]:
        """
        언론사와 발행일 동시 추출 (sds-comps-profile 구조 활용)

        Args:
            link_element: 뉴스 링크 요소 (a 태그)
            container: 뉴스 아이템 컨테이너

        Returns:
            (press, published_at) 튜플
        """
        press = ""
        published_at = ""

        # 방법 1: 링크 요소의 부모를 탐색하며 sds-comps-profile 찾기
        parent = link_element
        for _ in range(6):
            parent = parent.parent
            if parent is None:
                break

            profile = parent.select_one('[class*="sds-comps-profile"]')
            if profile:
                # title 클래스에서 언론사
                title_elem = profile.select_one('[class*="title"]')
                if title_elem:
                    text = title_elem.get_text(strip=True)
                    if text and len(text) < 30:
                        press = text

                # sub 클래스에서 날짜
                sub_elem = profile.select_one('[class*="sub"]')
                if sub_elem:
                    text = sub_elem.get_text(strip=True)
                    if self._looks_like_date(text):
                        published_at = self._clean_date(text)
                break

        # 방법 2: 컨테이너에서 전통적인 클래스 패턴 (fallback)
        if container and not press:
            for selector in ['a.info.press', 'span[class*="press"]', 'a[class*="press"]']:
                elem = container.select_one(selector)
                if elem:
                    text = elem.get_text(strip=True)
                    if text and len(text) < 30:
                        press = text
                        break

        if container and not published_at:
            for selector in ['span[class*="date"]', 'span[class*="time"]', 'span.info']:
                elems = container.select(selector)
                for elem in elems:
                    text = elem.get_text(strip=True)
                    if self._looks_like_date(text):
                        published_at = self._clean_date(text)
                        break
                if published_at:
                    break

        # 방법 3: 컨테이너 텍스트에서 패턴 매칭
        if container and (not press or not published_at):
            all_text = container.get_text(separator="|", strip=True)
            parts = [p.strip() for p in all_text.split("|") if p.strip()]

            for part in parts:
                # 날짜 패턴
                if not published_at and self._looks_like_date(part):
                    published_at = self._clean_date(part)
                # 언론사 패턴 (짧고, 뉴스/일보/신문/경제 등 포함)
                elif not press and 3 < len(part) < 20:
                    if any(kw in part for kw in ["뉴스", "일보", "신문", "경제", "타임", "투데이", "미디어"]):
                        press = part

        return press, published_at

    def _looks_like_date(self, text: str) -> bool:
        """날짜처럼 보이는지 확인"""
        if not text:
            return False

        # "X시간 전", "X일 전", "X분 전" 패턴
        if re.search(r'\d+\s*(시간|일|분|주)\s*전', text):
            return True

        # "2026.01.30" 또는 "2026-01-30" 패턴
        if re.search(r'\d{4}[.\-]\d{2}[.\-]\d{2}', text):
            return True

        return False

    def _clean_date(self, text: str) -> str:
        """날짜 텍스트에서 불필요한 부분 제거"""
        if not text:
            return ""

        # "X시간 전", "X분 전", "X일 전" 패턴만 추출
        match = re.search(r'(\d+\s*(시간|일|분|주)\s*전)', text)
        if match:
            return match.group(1)

        # "2026.01.30" 패턴만 추출
        match = re.search(r'(\d{4}[.\-]\d{2}[.\-]\d{2})', text)
        if match:
            return match.group(1)

        return text

    def search_multiple(
        self,
        queries: list[str],
        period: str = "1m",
        max_per_query: int = 10,
    ) -> dict[str, list[NaverNewsArticle]]:
        """
        여러 종목 뉴스 일괄 검색

        Args:
            queries: 검색어 리스트 (종목명들)
            period: 기간 필터
            max_per_query: 종목당 최대 결과 수

        Returns:
            {종목명: [NaverNewsArticle, ...], ...}
        """
        results = {}

        for i, query in enumerate(queries):
            self.logger.debug(f"[{i+1}/{len(queries)}] {query} 검색 중...")

            articles = self.search(
                query=query,
                period=period,
                max_results=max_per_query
            )
            results[query] = articles

            # 요청 간 딜레이
            if i < len(queries) - 1:
                time.sleep(self.delay)

        return results

    def format_for_llm(
        self,
        articles: list[NaverNewsArticle],
        stock_name: str = "",
    ) -> str:
        """
        LLM 분석용 텍스트 포맷 생성

        Args:
            articles: 뉴스 기사 리스트
            stock_name: 종목명

        Returns:
            LLM 프롬프트용 텍스트
        """
        if not articles:
            return f"[{stock_name}] 최근 뉴스 없음"

        lines = [f"[{stock_name}] 최근 뉴스 ({len(articles)}건):\n"]

        for i, article in enumerate(articles, 1):
            lines.append(f"{i}. [{article.published_at}] {article.title}")
            lines.append(f"   ({article.press})")
            if article.summary:
                # 요약을 2줄로 나눠서 표시
                summary = article.summary[:150]
                lines.append(f"   → {summary}...")
            lines.append("")

        lines.append("※ 위 뉴스를 바탕으로 다음을 분석해주세요:")
        lines.append("  1. 최근 주요 이슈와 트렌드")
        lines.append("  2. 시간 흐름에 따른 뉴스 톤 변화")
        lines.append("  3. 투자자 관점에서 주목할 점")

        return "\n".join(lines)

    def close(self):
        """세션 종료"""
        self.session.close()


# 테스트용
if __name__ == "__main__":
    import sys

    crawler = NaverNewsSearchCrawler(delay_seconds=2.0)

    # 테스트 종목 (인자로 받거나 기본값)
    test_stock = sys.argv[1] if len(sys.argv) > 1 else "삼성전자"

    print("=" * 70)
    print(f"네이버 뉴스 검색 테스트: {test_stock}")
    print("=" * 70)

    articles = crawler.search(test_stock, period="1m", max_results=10)

    print(f"\n수집된 기사: {len(articles)}건\n")

    for i, article in enumerate(articles, 1):
        print(f"{i}. {article.title}")
        print(f"   언론사: {article.press} | {article.published_at}")
        if article.summary:
            print(f"   요약: {article.summary[:100]}...")
        else:
            print("   요약: (없음)")
        print(f"   링크: {article.link[:70]}...")
        print()

    print("\n" + "=" * 70)
    print("LLM 전달 형식:")
    print("=" * 70)
    print(crawler.format_for_llm(articles, test_stock))

    crawler.close()
