"""
테마 데이터 서비스

크롤링 데이터를 DB에 저장/조회하는 서비스 레이어
"""
from datetime import datetime
from typing import Any

from sqlalchemy import select, delete
from sqlalchemy.orm import Session

from src.core.database import DatabaseManager, get_database
from src.core.models import ThemeModel, StockModel, theme_stock_association, CrawlHistoryModel
from src.core.logger import get_logger
from src.ingest.naver_theme import NaverThemeCrawler, ThemeInfo, ThemeStock


class ThemeService:
    """
    테마 데이터 관리 서비스

    사용법:
        service = ThemeService()

        # 전체 테마 수집 및 저장 (초기화 또는 주간 업데이트)
        service.sync_all_themes()

        # 특정 테마 조회
        theme = service.get_theme("123")

        # 테마별 종목 조회
        stocks = service.get_stocks_by_theme("123")
    """

    def __init__(self, db: DatabaseManager | None = None):
        self.db = db or get_database()
        self.logger = get_logger(self.__class__.__name__)

    def sync_all_themes(self) -> dict[str, int]:
        """
        전체 테마 및 소속 종목 동기화

        1. 네이버에서 크롤링
        2. DB에 저장 (upsert)
        3. 크롤링 히스토리 기록

        Returns:
            {"themes": 테마 수, "stocks": 종목 수, "mappings": 매핑 수}
        """
        self.logger.info("테마 동기화 시작")

        crawler = NaverThemeCrawler()
        history = CrawlHistoryModel(source="naver_theme", started_at=datetime.now())

        try:
            # 크롤링
            data = crawler.fetch()
            themes = data["themes"]
            theme_stocks = data["theme_stocks"]

            # DB 저장
            with self.db.session() as session:
                # 테마 저장
                theme_count = self._save_themes(session, themes)

                # 종목 저장 (종목 테이블에 없는 종목 추가)
                stock_count = self._save_stocks(session, theme_stocks)

                # 테마-종목 매핑 저장
                mapping_count = self._save_theme_stock_mappings(session, theme_stocks)

                # 히스토리 저장
                history.status = "success"
                history.record_count = mapping_count
                history.completed_at = datetime.now()
                session.add(history)

            result = {
                "themes": theme_count,
                "stocks": stock_count,
                "mappings": mapping_count,
            }

            self.logger.info(f"테마 동기화 완료: {result}")
            return result

        except Exception as e:
            self.logger.error(f"테마 동기화 실패: {e}")

            # 실패 히스토리 저장
            with self.db.session() as session:
                history.status = "failed"
                history.error_message = str(e)
                history.completed_at = datetime.now()
                session.add(history)

            raise

        finally:
            crawler.close()

    def _save_themes(self, session: Session, themes: list[ThemeInfo]) -> int:
        """테마 저장 (upsert)"""
        count = 0
        for theme in themes:
            existing = session.get(ThemeModel, theme.theme_id)

            if existing:
                existing.name = theme.name
                existing.updated_at = datetime.now()
            else:
                new_theme = ThemeModel(
                    theme_id=theme.theme_id,
                    name=theme.name,
                )
                session.add(new_theme)
                count += 1

        session.flush()
        return count

    def _save_stocks(self, session: Session, theme_stocks: list[ThemeStock]) -> int:
        """종목 저장 (없는 종목만 추가)"""
        # 유니크한 종목 추출
        unique_stocks = {}
        for ts in theme_stocks:
            if ts.stock_code not in unique_stocks:
                unique_stocks[ts.stock_code] = ts.stock_name

        count = 0
        for code, name in unique_stocks.items():
            existing = session.get(StockModel, code)
            if not existing:
                new_stock = StockModel(
                    stock_code=code,
                    name=name,
                )
                session.add(new_stock)
                count += 1

        session.flush()
        return count

    def _save_theme_stock_mappings(
        self,
        session: Session,
        theme_stocks: list[ThemeStock]
    ) -> int:
        """테마-종목 매핑 저장"""
        # 기존 매핑 삭제
        session.execute(delete(theme_stock_association))

        # 새 매핑 추가
        mappings = []
        for ts in theme_stocks:
            mappings.append({
                "theme_id": ts.theme_id,
                "stock_code": ts.stock_code,
            })

        if mappings:
            session.execute(theme_stock_association.insert(), mappings)

        # 테마별 종목 수 업데이트
        theme_stock_counts = {}
        for ts in theme_stocks:
            theme_stock_counts[ts.theme_id] = theme_stock_counts.get(ts.theme_id, 0) + 1

        for theme_id, count in theme_stock_counts.items():
            theme = session.get(ThemeModel, theme_id)
            if theme:
                theme.stock_count = count

        return len(mappings)

    def get_theme(self, theme_id: str) -> ThemeModel | None:
        """테마 조회"""
        with self.db.session() as session:
            return session.get(ThemeModel, theme_id)

    def get_all_themes(self) -> list[ThemeModel]:
        """전체 테마 목록 조회"""
        with self.db.session() as session:
            result = session.execute(select(ThemeModel))
            return list(result.scalars().all())

    def get_stocks_by_theme(self, theme_id: str) -> list[StockModel]:
        """테마별 소속 종목 조회"""
        with self.db.session() as session:
            theme = session.get(ThemeModel, theme_id)
            if theme:
                return list(theme.stocks)
            return []

    def get_themes_by_stock(self, stock_code: str) -> list[ThemeModel]:
        """종목이 속한 테마 목록 조회"""
        with self.db.session() as session:
            stock = session.get(StockModel, stock_code)
            if stock:
                return list(stock.themes)
            return []

    def get_last_sync_time(self) -> datetime | None:
        """마지막 동기화 시간 조회"""
        with self.db.session() as session:
            result = session.execute(
                select(CrawlHistoryModel)
                .where(CrawlHistoryModel.source == "naver_theme")
                .where(CrawlHistoryModel.status == "success")
                .order_by(CrawlHistoryModel.completed_at.desc())
                .limit(1)
            )
            history = result.scalar_one_or_none()
            return history.completed_at if history else None

    def needs_sync(self, days: int = 7) -> bool:
        """동기화 필요 여부 (기본 7일)"""
        last_sync = self.get_last_sync_time()
        if not last_sync:
            return True

        from datetime import timedelta
        return datetime.now() - last_sync > timedelta(days=days)
