"""
네이버 금융 테마 크롤러

테마 목록 및 테마별 소속 종목 수집
"""
import time
import random
from dataclasses import dataclass
from typing import Generator

import requests
from bs4 import BeautifulSoup

from src.ingest.base import BaseDataFetcher
from src.core.config import get_config
from src.core.exceptions import CrawlerError


@dataclass
class ThemeInfo:
    """테마 정보"""
    theme_id: str
    name: str
    change_rate: float = 0.0  # 등락률
    stock_count: int = 0


@dataclass
class ThemeStock:
    """테마 소속 종목"""
    theme_id: str
    stock_code: str
    stock_name: str
    price: int = 0
    change_rate: float = 0.0


class NaverThemeCrawler(BaseDataFetcher):
    """
    네이버 금융 테마 크롤러

    사용법:
        crawler = NaverThemeCrawler()

        # 전체 테마 목록
        themes = crawler.fetch_theme_list()

        # 특정 테마의 소속 종목
        stocks = crawler.fetch_theme_stocks("123")

        # 전체 수집 (테마 + 소속종목)
        all_data = crawler.fetch()
    """

    BASE_URL = "https://finance.naver.com/sise/theme.naver"
    THEME_DETAIL_URL = "https://finance.naver.com/sise/sise_group_detail.naver"

    # User-Agent 풀 (브라우저별 다양화)
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    ]

    def __init__(
        self,
        min_delay: float = 1.5,
        max_delay: float = 4.0,
        max_retries: int = 3,
        backoff_factor: float = 2.0
    ):
        """
        Args:
            min_delay: 최소 딜레이 (초)
            max_delay: 최대 딜레이 (초)
            max_retries: 최대 재시도 횟수
            backoff_factor: 재시도 시 딜레이 배수
        """
        super().__init__()

        config = get_config()
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        # 연속 실패 카운터
        self._consecutive_failures = 0
        self._request_count = 0

        self.session = requests.Session()
        self._rotate_user_agent()

    def _rotate_user_agent(self):
        """User-Agent 회전"""
        ua = random.choice(self.USER_AGENTS)
        self.session.headers.update({
            "User-Agent": ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Referer": "https://finance.naver.com/",
        })

    def _random_delay(self, multiplier: float = 1.0):
        """불규칙 딜레이"""
        delay = random.uniform(self.min_delay, self.max_delay) * multiplier
        time.sleep(delay)

    def _request_with_retry(self, url: str, timeout: int = 15) -> requests.Response:
        """재시도 로직이 포함된 요청"""
        for attempt in range(self.max_retries):
            try:
                # 일정 요청마다 User-Agent 회전
                self._request_count += 1
                if self._request_count % 10 == 0:
                    self._rotate_user_agent()

                response = self.session.get(url, timeout=timeout)

                # 차단 감지
                if response.status_code == 403:
                    self.logger.warning(f"403 Forbidden - 쿨다운 적용")
                    self._consecutive_failures += 1
                    time.sleep(30 * self._consecutive_failures)  # 점진적 쿨다운
                    self._rotate_user_agent()
                    continue

                if response.status_code == 429:
                    self.logger.warning(f"429 Too Many Requests - 대기")
                    time.sleep(60)
                    continue

                response.raise_for_status()
                self._consecutive_failures = 0  # 성공 시 리셋
                return response

            except requests.RequestException as e:
                self.logger.warning(f"요청 실패 (시도 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    wait = self.backoff_factor ** attempt * 5
                    time.sleep(wait)
                else:
                    raise

        raise CrawlerError(f"최대 재시도 횟수 초과: {url}")

    def get_source_name(self) -> str:
        return "NaverTheme"

    def fetch(self) -> dict:
        """
        전체 테마 및 소속 종목 수집

        Returns:
            {
                "themes": [ThemeInfo, ...],
                "theme_stocks": [ThemeStock, ...],
            }
        """
        self._log_fetch_start()

        themes = list(self.fetch_theme_list())
        self.logger.info(f"테마 {len(themes)}개 수집 완료")

        all_stocks = []
        for i, theme in enumerate(themes):
            try:
                stocks = list(self.fetch_theme_stocks(theme.theme_id, theme.name))
                all_stocks.extend(stocks)

                if (i + 1) % 10 == 0:
                    self.logger.info(f"진행: {i+1}/{len(themes)} 테마, {len(all_stocks)} 종목")
            except Exception as e:
                self.logger.warning(f"테마 '{theme.name}' 종목 수집 실패: {e}")

            # 불규칙 딜레이
            self._random_delay()

        self._log_fetch_complete(count=len(all_stocks))

        return {
            "themes": themes,
            "theme_stocks": all_stocks,
        }

    def fetch_theme_list(self) -> Generator[ThemeInfo, None, None]:
        """
        테마 목록 수집 (전체 페이지)

        Yields:
            ThemeInfo 객체
        """
        page = 1
        total_themes = 0

        self.logger.info("테마 목록 수집 시작...")

        while True:
            url = f"{self.BASE_URL}?page={page}"

            try:
                response = self._request_with_retry(url)
                response.encoding = "euc-kr"

                soup = BeautifulSoup(response.text, "lxml")
                table = soup.select_one("table.type_1")

                if not table:
                    self.logger.debug(f"페이지 {page}: 테이블 없음, 종료")
                    break

                rows = table.select("tr")
                found_themes = False
                page_themes = 0

                for row in rows:
                    cols = row.select("td")
                    if len(cols) < 4:
                        continue

                    # 테마명 링크에서 ID 추출
                    link = cols[0].select_one("a")
                    if not link:
                        continue

                    href = link.get("href", "")
                    if "no=" not in href:
                        continue

                    theme_id = href.split("no=")[-1].split("&")[0]
                    name = link.get_text(strip=True)

                    # 등락률 파싱
                    change_text = cols[2].get_text(strip=True).replace("%", "").replace(",", "")
                    try:
                        change_rate = float(change_text) if change_text else 0.0
                    except ValueError:
                        change_rate = 0.0

                    found_themes = True
                    page_themes += 1
                    total_themes += 1

                    yield ThemeInfo(
                        theme_id=theme_id,
                        name=name,
                        change_rate=change_rate,
                    )

                if not found_themes:
                    self.logger.debug(f"페이지 {page}: 테마 없음, 종료")
                    break

                self.logger.debug(f"페이지 {page}: {page_themes}개 테마 (누적 {total_themes}개)")

                page += 1
                self._random_delay()

            except CrawlerError:
                self.logger.error(f"테마 목록 페이지 {page} 수집 실패, 종료")
                break
            except Exception as e:
                self.logger.error(f"테마 목록 페이지 {page} 예외: {e}")
                break

        self.logger.info(f"테마 목록 수집 완료: 총 {total_themes}개")

    def fetch_theme_stocks(
        self,
        theme_id: str,
        theme_name: str = ""
    ) -> Generator[ThemeStock, None, None]:
        """
        특정 테마의 소속 종목 수집 (전체 페이지)

        Args:
            theme_id: 테마 ID
            theme_name: 테마명 (로깅용)

        Yields:
            ThemeStock 객체
        """
        page = 1
        total_stocks = 0

        while True:
            url = f"{self.THEME_DETAIL_URL}?type=theme&no={theme_id}&page={page}"

            try:
                response = self._request_with_retry(url)
                response.encoding = "euc-kr"

                soup = BeautifulSoup(response.text, "lxml")
                table = soup.select_one("table.type_5")

                if not table:
                    break

                rows = table.select("tr")
                found_stocks = False

                for row in rows:
                    cols = row.select("td")
                    if len(cols) < 6:
                        continue

                    # 종목명 링크에서 코드 추출
                    link = cols[0].select_one("a")
                    if not link:
                        continue

                    href = link.get("href", "")
                    if "code=" not in href:
                        continue

                    stock_code = href.split("code=")[-1].split("&")[0]
                    stock_name = link.get_text(strip=True)

                    # 종목 코드 유효성 검사 (6자리 숫자만 허용)
                    if not stock_code.isdigit() or len(stock_code) != 6:
                        continue

                    # 현재가
                    price_text = cols[1].get_text(strip=True).replace(",", "")
                    try:
                        price = int(price_text) if price_text else 0
                    except ValueError:
                        price = 0

                    # 등락률
                    change_text = cols[3].get_text(strip=True).replace("%", "").replace(",", "")
                    try:
                        change_rate = float(change_text) if change_text else 0.0
                    except ValueError:
                        change_rate = 0.0

                    found_stocks = True
                    total_stocks += 1

                    yield ThemeStock(
                        theme_id=theme_id,
                        stock_code=stock_code,
                        stock_name=stock_name,
                        price=price,
                        change_rate=change_rate,
                    )

                if not found_stocks:
                    break

                # 다음 페이지 확인
                paging = soup.select_one("td.pgRR")
                if not paging:
                    break

                page += 1
                self._random_delay(multiplier=0.5)  # 상세 페이지는 더 짧은 딜레이

            except CrawlerError:
                self.logger.warning(f"테마 '{theme_name}' 종목 수집 실패, 현재까지 {total_stocks}개")
                break
            except Exception as e:
                self.logger.warning(f"테마 '{theme_name}' 종목 페이지 {page} 예외: {e}")
                break

    def close(self):
        """세션 종료"""
        self.session.close()
