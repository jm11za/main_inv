"""
네이버 금융 토론방 크롤러

종목별 투자자 토론 글 수집 (Sentiment Reader B 접근법)
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
class DiscussionPost:
    """토론방 글 정보"""
    stock_code: str
    title: str
    link: str
    author: str
    created_at: str  # 작성일
    views: int  # 조회수
    likes: int  # 공감수
    dislikes: int  # 비공감수
    content: str = ""  # 본문 (선택적 수집)

    def __repr__(self):
        return f"<Discussion [{self.author}] {self.title[:20]}... 조회:{self.views}>"

    @property
    def sentiment_score(self) -> float:
        """공감 비율 기반 감성 점수 (-1 ~ 1)"""
        total = self.likes + self.dislikes
        if total == 0:
            return 0.0
        return (self.likes - self.dislikes) / total


class DiscussionCrawler(BaseDataFetcher):
    """
    네이버 금융 토론방 크롤러

    종목별 투자자 토론 글 수집 (공감/비공감 포함)

    사용법:
        crawler = DiscussionCrawler()

        # 특정 종목 토론 글 수집
        posts = crawler.fetch_stock_discussions("005930")

        # 심리 분석용 요약
        summary = crawler.get_sentiment_summary("005930")
    """

    # 네이버 금융 토론방 URL
    BOARD_URL = "https://finance.naver.com/item/board.naver"
    BOARD_READ_URL = "https://finance.naver.com/item/board_read.naver"

    def __init__(
        self,
        max_posts: int | None = None,
        delay_seconds: float | None = None,
        fetch_content: bool = False,
    ):
        """
        Args:
            max_posts: 종목당 최대 수집 글 수
            delay_seconds: 요청 간 딜레이
            fetch_content: 본문 수집 여부 (느림, 더 정확한 분석 가능)
        """
        super().__init__()

        config = get_config()
        self.max_posts = max_posts or config.get(
            "ingest.discussion.max_posts_per_stock", 50
        )
        self.delay = delay_seconds or config.get(
            "ingest.naver_theme.delay_seconds", 0.5
        )
        self.fetch_content = fetch_content

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
        return "NaverDiscussion"

    def fetch(self) -> list[DiscussionPost]:
        """기본 fetch는 미구현 (종목 지정 필요)"""
        raise NotImplementedError("fetch_stock_discussions() 사용")

    def fetch_stock_discussions(
        self,
        stock_code: str,
        max_pages: int = 5
    ) -> list[DiscussionPost]:
        """
        특정 종목의 토론 글 수집

        Args:
            stock_code: 종목코드 (예: "005930")
            max_pages: 최대 페이지 수

        Returns:
            DiscussionPost 리스트
        """
        self.logger.debug(f"토론방 수집 시작: {stock_code}")

        posts = []
        page = 1

        while page <= max_pages and len(posts) < self.max_posts:
            try:
                page_posts = list(self._fetch_board_page(stock_code, page))

                if not page_posts:
                    break

                posts.extend(page_posts)
                page += 1

                time.sleep(self.delay * 0.3)

            except Exception as e:
                self.logger.warning(f"토론방 페이지 {page} 수집 실패: {e}")
                break

        # max_posts 제한 적용
        posts = posts[:self.max_posts]

        self.logger.debug(f"[{stock_code}] 토론 글 {len(posts)}건 수집 완료")
        return posts

    def _fetch_board_page(
        self,
        stock_code: str,
        page: int
    ) -> Generator[DiscussionPost, None, None]:
        """토론방 페이지 파싱"""
        params = {
            "code": stock_code,
            "page": page,
        }

        try:
            response = self.session.get(
                self.BOARD_URL,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            response.encoding = "euc-kr"

            soup = BeautifulSoup(response.text, "lxml")

            # 게시글 테이블 찾기
            board_table = soup.select_one("table.type2")
            if not board_table:
                return

            rows = board_table.select("tr")

            for row in rows:
                # 공지사항 제외
                if row.get("class") and "notice" in " ".join(row.get("class", [])):
                    continue

                cells = row.select("td")
                if len(cells) < 6:
                    continue

                try:
                    # 제목
                    title_cell = cells[1]
                    link_tag = title_cell.select_one("a.tit")
                    if not link_tag:
                        continue

                    title = link_tag.get_text(strip=True)
                    href = link_tag.get("href", "")

                    # 링크 정규화
                    if href.startswith("/"):
                        link = f"https://finance.naver.com{href}"
                    else:
                        link = href

                    # 작성자
                    author = cells[2].get_text(strip=True)

                    # 작성일
                    created_at = cells[3].get_text(strip=True)

                    # 조회수
                    views_text = cells[4].get_text(strip=True)
                    views = int(views_text.replace(",", "")) if views_text.isdigit() else 0

                    # 공감/비공감
                    likes_text = cells[5].get_text(strip=True)
                    likes = int(likes_text.replace(",", "")) if likes_text else 0

                    dislikes_text = cells[6].get_text(strip=True) if len(cells) > 6 else "0"
                    dislikes = int(dislikes_text.replace(",", "")) if dislikes_text else 0

                    post = DiscussionPost(
                        stock_code=stock_code,
                        title=title,
                        link=link,
                        author=author,
                        created_at=created_at,
                        views=views,
                        likes=likes,
                        dislikes=dislikes,
                    )

                    # 본문 수집 (옵션)
                    if self.fetch_content:
                        post.content = self._fetch_post_content(link)

                    yield post

                except (ValueError, IndexError) as e:
                    self.logger.debug(f"게시글 파싱 오류: {e}")
                    continue

        except requests.RequestException as e:
            self.logger.error(f"토론방 페이지 요청 실패: {e}")
            raise CrawlerError(f"토론방 크롤링 실패: {e}")

    def _fetch_post_content(self, url: str) -> str:
        """게시글 본문 수집"""
        try:
            time.sleep(self.delay * 0.5)

            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            response.encoding = "euc-kr"

            soup = BeautifulSoup(response.text, "lxml")

            # 본문 영역
            content_div = soup.select_one("div.view_se")
            if content_div:
                return content_div.get_text(strip=True)[:500]

            return ""

        except Exception as e:
            self.logger.debug(f"본문 수집 실패: {e}")
            return ""

    def get_sentiment_summary(
        self,
        stock_code: str,
        max_posts: int = 30
    ) -> dict:
        """
        심리 분석용 토론방 요약

        Args:
            stock_code: 종목코드
            max_posts: 분석할 최대 글 수

        Returns:
            {
                "post_count": int,
                "avg_views": float,
                "total_likes": int,
                "total_dislikes": int,
                "sentiment_ratio": float,  # -1 ~ 1
                "hot_posts": list[str],  # 인기 글 제목
                "recent_titles": list[str],  # 최근 글 제목
            }
        """
        original_max = self.max_posts
        self.max_posts = max_posts

        posts = self.fetch_stock_discussions(stock_code, max_pages=3)

        self.max_posts = original_max

        if not posts:
            return {
                "post_count": 0,
                "avg_views": 0,
                "total_likes": 0,
                "total_dislikes": 0,
                "sentiment_ratio": 0.0,
                "hot_posts": [],
                "recent_titles": [],
            }

        total_likes = sum(p.likes for p in posts)
        total_dislikes = sum(p.dislikes for p in posts)
        total_reactions = total_likes + total_dislikes

        # 공감 비율 기반 감성 점수
        if total_reactions > 0:
            sentiment_ratio = (total_likes - total_dislikes) / total_reactions
        else:
            sentiment_ratio = 0.0

        # 인기 글 (조회수 + 공감 기준)
        hot_posts = sorted(
            posts,
            key=lambda p: p.views + p.likes * 10,
            reverse=True
        )[:5]

        return {
            "post_count": len(posts),
            "avg_views": sum(p.views for p in posts) / len(posts),
            "total_likes": total_likes,
            "total_dislikes": total_dislikes,
            "sentiment_ratio": round(sentiment_ratio, 3),
            "hot_posts": [p.title for p in hot_posts],
            "recent_titles": [p.title for p in posts[:10]],
        }

    def fetch_for_sentiment_analysis(
        self,
        stock_code: str,
        stock_name: str = ""
    ) -> dict:
        """
        Sentiment Reader 입력용 데이터 생성

        Args:
            stock_code: 종목코드
            stock_name: 종목명

        Returns:
            {
                "community_posts": list[str],  # 글 제목 + 본문
                "mention_count": int,
                "sentiment_data": dict,  # 공감/비공감 통계
            }
        """
        summary = self.get_sentiment_summary(stock_code)

        # 글 제목들을 community_posts로 변환
        community_posts = summary.get("recent_titles", [])

        # 인기 글도 추가 (중복 제거)
        for title in summary.get("hot_posts", []):
            if title not in community_posts:
                community_posts.append(title)

        return {
            "community_posts": community_posts,
            "mention_count": summary.get("post_count", 0),
            "sentiment_data": {
                "likes": summary.get("total_likes", 0),
                "dislikes": summary.get("total_dislikes", 0),
                "sentiment_ratio": summary.get("sentiment_ratio", 0.0),
                "avg_views": summary.get("avg_views", 0),
            }
        }

    def fetch_multiple_stocks(
        self,
        stock_codes: list[str],
        max_posts_per_stock: int | None = None
    ) -> dict[str, list[DiscussionPost]]:
        """
        여러 종목의 토론 글 일괄 수집

        Args:
            stock_codes: 종목코드 리스트
            max_posts_per_stock: 종목당 최대 글 수

        Returns:
            {종목코드: [DiscussionPost, ...], ...}
        """
        self._log_fetch_start()

        if max_posts_per_stock:
            original_max = self.max_posts
            self.max_posts = max_posts_per_stock

        result = {}
        total_count = 0

        for i, code in enumerate(stock_codes):
            try:
                posts = self.fetch_stock_discussions(code)
                result[code] = posts
                total_count += len(posts)

                self.logger.debug(
                    f"[{i+1}/{len(stock_codes)}] {code}: {len(posts)}건"
                )

            except Exception as e:
                self.logger.warning(f"[{code}] 토론방 수집 실패: {e}")
                result[code] = []

            time.sleep(self.delay)

        if max_posts_per_stock:
            self.max_posts = original_max

        self._log_fetch_complete(count=total_count)
        return result

    def close(self):
        """세션 종료"""
        self.session.close()
