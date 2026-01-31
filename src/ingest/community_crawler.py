"""
커뮤니티 크롤러

종목별 커뮤니티 글/댓글 수집 (Sentiment 분석용)
- 네이버 종목토론실
- 추후 확장: 팍스넷, 디시 주갤 등

차단 회피 기능:
- User-Agent 로테이션
- 랜덤 딜레이
- 재시도 with exponential backoff
- 요청 실패 시 자동 대기 시간 증가
"""
import time
import re
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Generator

import requests
from bs4 import BeautifulSoup

from src.ingest.base import BaseDataFetcher
from src.core.config import get_config
from src.core.exceptions import CrawlerError


@dataclass
class CommunityPost:
    """커뮤니티 글 정보"""
    stock_code: str
    title: str
    content: str = ""  # 본문 (수집 시)
    author: str = ""
    date: str = ""
    views: int = 0
    likes: int = 0
    dislikes: int = 0
    comments_count: int = 0
    link: str = ""
    source: str = "naver"  # naver, paxnet, dcinside 등

    def __repr__(self):
        return f"<Post [{self.source}] {self.title[:30]}...>"


@dataclass
class CommunityData:
    """종목별 커뮤니티 데이터 집계"""
    stock_code: str
    stock_name: str = ""
    posts: list[CommunityPost] = field(default_factory=list)
    total_posts: int = 0
    mention_count: int = 0  # 언급량
    avg_views: float = 0.0
    avg_likes: float = 0.0
    collected_at: datetime = field(default_factory=datetime.now)

    @property
    def titles(self) -> list[str]:
        """제목 목록"""
        return [p.title for p in self.posts]

    @property
    def contents(self) -> list[str]:
        """본문 목록 (수집된 경우)"""
        return [p.content for p in self.posts if p.content]


class CommunityCrawler(BaseDataFetcher):
    """
    커뮤니티 크롤러

    네이버 종목토론실에서 최근 글 수집

    차단 회피 기능:
    - User-Agent 로테이션
    - 랜덤 딜레이 (min_delay ~ max_delay)
    - 재시도 with exponential backoff
    - 연속 실패 시 자동 쿨다운

    사용법:
        crawler = CommunityCrawler()

        # 특정 종목 커뮤니티 글 수집
        data = crawler.fetch_stock_community("005930")
        print(data.titles)  # 제목 목록
        print(data.mention_count)  # 언급량

        # 여러 종목 수집
        results = crawler.fetch_multiple_stocks(["005930", "000660"])
    """

    # 네이버 종목토론실 URL
    BOARD_URL = "https://finance.naver.com/item/board.naver"
    BOARD_READ_URL = "https://finance.naver.com/item/board_read.naver"

    # User-Agent 목록 (로테이션용)
    USER_AGENTS = [
        # Chrome Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
        # Chrome Mac
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        # Firefox Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        # Firefox Mac
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
        # Edge
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
        # Safari
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    ]

    def __init__(
        self,
        max_posts: int = 50,
        max_pages: int = 3,
        fetch_content: bool = False,
        min_delay: float = 1.0,
        max_delay: float = 3.0,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ):
        """
        Args:
            max_posts: 종목당 최대 수집 글 수
            max_pages: 최대 페이지 수
            fetch_content: 본문도 수집할지 여부 (느려짐)
            min_delay: 최소 대기 시간 (초)
            max_delay: 최대 대기 시간 (초)
            max_retries: 최대 재시도 횟수
            backoff_factor: 재시도 시 대기 시간 증가 배수
        """
        super().__init__()

        config = get_config()
        self.max_posts = max_posts
        self.max_pages = max_pages
        self.fetch_content = fetch_content
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        # 연속 실패 카운터 (쿨다운용)
        self._consecutive_failures = 0
        self._cooldown_until = 0

        # 세션 초기화
        self.session = requests.Session()
        self._rotate_user_agent()

    def _rotate_user_agent(self):
        """User-Agent 로테이션"""
        user_agent = random.choice(self.USER_AGENTS)
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Referer": "https://finance.naver.com/",
            "DNT": "1",
        })

    def _random_delay(self):
        """랜덤 딜레이"""
        delay = random.uniform(self.min_delay, self.max_delay)
        time.sleep(delay)

    def _check_cooldown(self):
        """쿨다운 체크 및 대기"""
        if time.time() < self._cooldown_until:
            wait_time = self._cooldown_until - time.time()
            self.logger.warning(f"쿨다운 중... {wait_time:.1f}초 대기")
            time.sleep(wait_time)

    def _handle_failure(self):
        """실패 처리 - 연속 실패 시 쿨다운"""
        self._consecutive_failures += 1

        if self._consecutive_failures >= 3:
            # 3회 연속 실패 시 30초 쿨다운
            cooldown = 30 * (self._consecutive_failures - 2)
            self._cooldown_until = time.time() + cooldown
            self.logger.warning(f"연속 {self._consecutive_failures}회 실패. {cooldown}초 쿨다운 적용")

            # User-Agent 변경
            self._rotate_user_agent()

    def _handle_success(self):
        """성공 처리 - 실패 카운터 리셋"""
        self._consecutive_failures = 0

    def _request_with_retry(
        self,
        url: str,
        params: dict | None = None,
        timeout: int = 10
    ) -> requests.Response:
        """
        재시도 로직이 포함된 요청

        Args:
            url: 요청 URL
            params: 쿼리 파라미터
            timeout: 타임아웃 (초)

        Returns:
            Response 객체

        Raises:
            CrawlerError: 모든 재시도 실패 시
        """
        self._check_cooldown()

        last_error = None

        for attempt in range(self.max_retries):
            try:
                # User-Agent 로테이션 (매 요청마다)
                if attempt > 0:
                    self._rotate_user_agent()

                response = self.session.get(
                    url,
                    params=params,
                    timeout=timeout
                )

                # 차단 감지 (403, 429, 503 등)
                if response.status_code == 403:
                    self.logger.warning("403 Forbidden - 차단 감지")
                    self._handle_failure()
                    raise requests.RequestException("Access Forbidden")

                if response.status_code == 429:
                    self.logger.warning("429 Too Many Requests - 속도 제한")
                    self._handle_failure()
                    # 429는 더 긴 대기
                    time.sleep(30)
                    raise requests.RequestException("Rate Limited")

                if response.status_code == 503:
                    self.logger.warning("503 Service Unavailable")
                    self._handle_failure()
                    raise requests.RequestException("Service Unavailable")

                response.raise_for_status()

                # 성공
                self._handle_success()
                return response

            except requests.RequestException as e:
                last_error = e
                self.logger.warning(f"요청 실패 (시도 {attempt + 1}/{self.max_retries}): {e}")

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = (self.backoff_factor ** attempt) * self.min_delay
                    wait_time = min(wait_time, 30)  # 최대 30초
                    self.logger.debug(f"{wait_time:.1f}초 후 재시도...")
                    time.sleep(wait_time)

        raise CrawlerError(f"요청 실패 (최대 재시도 초과): {last_error}")

    def get_source_name(self) -> str:
        return "NaverCommunity"

    def fetch(self) -> list[CommunityData]:
        """기본 fetch - 구현 필요시 사용"""
        raise NotImplementedError("fetch_stock_community()를 사용하세요")

    def fetch_stock_community(
        self,
        stock_code: str,
        stock_name: str = ""
    ) -> CommunityData:
        """
        특정 종목의 커뮤니티 글 수집

        Args:
            stock_code: 종목코드 (예: "005930")
            stock_name: 종목명 (선택)

        Returns:
            CommunityData
        """
        self.logger.info(f"커뮤니티 수집 시작: {stock_code}")

        posts = []
        page = 1

        while page <= self.max_pages and len(posts) < self.max_posts:
            try:
                page_posts = self._fetch_board_page(stock_code, page)

                if not page_posts:
                    self.logger.debug(f"[{stock_code}] 페이지 {page}: 글 없음, 종료")
                    break

                posts.extend(page_posts)
                self.logger.debug(f"[{stock_code}] 페이지 {page}: {len(page_posts)}개 수집")

                page += 1

                # 다음 페이지 전 랜덤 딜레이
                if page <= self.max_pages and len(posts) < self.max_posts:
                    self._random_delay()

            except CrawlerError as e:
                self.logger.warning(f"[{stock_code}] 페이지 {page} 수집 실패: {e}")
                break
            except Exception as e:
                self.logger.error(f"[{stock_code}] 예상치 못한 오류: {e}")
                break

        # 최대 개수 제한
        posts = posts[:self.max_posts]

        # 본문 수집 (옵션)
        if self.fetch_content and posts:
            posts = self._fetch_contents(posts)

        # 통계 계산
        total_views = sum(p.views for p in posts)
        total_likes = sum(p.likes for p in posts)
        avg_views = total_views / len(posts) if posts else 0
        avg_likes = total_likes / len(posts) if posts else 0

        self.logger.info(f"[{stock_code}] 커뮤니티 수집 완료: {len(posts)}개")

        return CommunityData(
            stock_code=stock_code,
            stock_name=stock_name,
            posts=posts,
            total_posts=len(posts),
            mention_count=len(posts),  # 글 수 = 언급량
            avg_views=avg_views,
            avg_likes=avg_likes,
        )

    def _fetch_board_page(
        self,
        stock_code: str,
        page: int = 1
    ) -> list[CommunityPost]:
        """게시판 페이지 수집"""
        params = {
            "code": stock_code,
            "page": page,
        }

        response = self._request_with_retry(self.BOARD_URL, params=params)
        return self._parse_board_page(response.text, stock_code)

    def _parse_board_page(
        self,
        html: str,
        stock_code: str
    ) -> list[CommunityPost]:
        """게시판 HTML 파싱"""
        soup = BeautifulSoup(html, "html.parser")
        posts = []

        # 게시글 테이블 찾기
        table = soup.find("table", class_="type2")
        if not table:
            return []

        rows = table.find_all("tr")

        for row in rows:
            try:
                # 제목 셀 찾기
                title_cell = row.find("td", class_="title")
                if not title_cell:
                    continue

                title_link = title_cell.find("a")
                if not title_link:
                    continue

                title = title_link.get_text(strip=True)
                href = title_link.get("href", "")

                # 링크에서 글 번호 추출
                link = f"https://finance.naver.com{href}" if href else ""

                # 다른 정보 추출
                cells = row.find_all("td")
                if len(cells) < 6:
                    continue

                # Cell 순서: 날짜(0), 제목(1), 글쓴이(2), 조회수(3), 공감(4), 비공감(5)
                date = cells[0].get_text(strip=True) if len(cells) > 0 else ""
                author = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                views = self._parse_int(cells[3].get_text(strip=True)) if len(cells) > 3 else 0
                likes = self._parse_int(cells[4].get_text(strip=True)) if len(cells) > 4 else 0
                dislikes = self._parse_int(cells[5].get_text(strip=True)) if len(cells) > 5 else 0

                # 광고/공지 제외
                if self._is_ad_or_notice(title, author):
                    continue

                posts.append(CommunityPost(
                    stock_code=stock_code,
                    title=title,
                    author=author,
                    date=date,
                    views=views,
                    likes=likes,
                    dislikes=dislikes,
                    link=link,
                    source="naver",
                ))

            except Exception as e:
                self.logger.debug(f"행 파싱 실패: {e}")
                continue

        return posts

    def _fetch_contents(
        self,
        posts: list[CommunityPost]
    ) -> list[CommunityPost]:
        """글 본문 수집"""
        for post in posts:
            if not post.link:
                continue

            try:
                response = self._request_with_retry(post.link)

                soup = BeautifulSoup(response.text, "html.parser")

                # 본문 영역 찾기
                content_div = soup.find("div", id="body")
                if content_div:
                    post.content = content_div.get_text(strip=True)[:500]  # 500자 제한

                self._random_delay()

            except Exception as e:
                self.logger.debug(f"본문 수집 실패: {post.link} - {e}")

        return posts

    def _parse_int(self, text: str) -> int:
        """문자열에서 정수 추출"""
        try:
            # 쉼표 제거 후 숫자만 추출
            cleaned = re.sub(r"[^\d]", "", text)
            return int(cleaned) if cleaned else 0
        except ValueError:
            return 0

    def _is_ad_or_notice(self, title: str, author: str) -> bool:
        """광고/공지 여부 판단"""
        ad_keywords = ["광고", "공지", "이벤트", "[AD]", "협찬"]
        admin_authors = ["운영자", "관리자", "네이버"]

        title_lower = title.lower()
        for keyword in ad_keywords:
            if keyword in title_lower:
                return True

        for admin in admin_authors:
            if admin in author:
                return True

        return False

    def fetch_multiple_stocks(
        self,
        stock_codes: list[str],
        stock_names: dict[str, str] | None = None
    ) -> list[CommunityData]:
        """
        여러 종목 커뮤니티 수집

        Args:
            stock_codes: 종목코드 리스트
            stock_names: {종목코드: 종목명} 매핑 (선택)

        Returns:
            CommunityData 리스트
        """
        stock_names = stock_names or {}
        results = []

        self.logger.info(f"커뮤니티 일괄 수집: {len(stock_codes)}개 종목")

        for i, code in enumerate(stock_codes):
            try:
                name = stock_names.get(code, "")
                data = self.fetch_stock_community(code, name)
                results.append(data)

                # 종목 간 딜레이 (더 길게)
                if i < len(stock_codes) - 1:
                    delay = random.uniform(self.max_delay, self.max_delay * 2)
                    time.sleep(delay)

            except CrawlerError as e:
                self.logger.error(f"[{code}] 수집 실패: {e}")
            except Exception as e:
                self.logger.error(f"[{code}] 예상치 못한 오류: {e}")

        self.logger.info(f"커뮤니티 수집 완료: {len(results)}/{len(stock_codes)}")
        return results

    def get_sentiment_data(
        self,
        stock_code: str,
        stock_name: str = ""
    ) -> dict:
        """
        Sentiment 분석용 데이터 반환

        Layer 5 Decision의 SentimentReader에서 사용할 형식

        Returns:
            {
                "stock_code": str,
                "stock_name": str,
                "community_posts": list[str],  # 제목 목록
                "mention_count": int,
            }
        """
        data = self.fetch_stock_community(stock_code, stock_name)

        return {
            "stock_code": stock_code,
            "stock_name": stock_name or data.stock_name,
            "community_posts": data.titles,
            "mention_count": data.mention_count,
        }
