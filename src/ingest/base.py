"""
Data Ingest Layer - 기본 클래스

모든 데이터 수집기의 공통 인터페이스 및 유틸리티
"""
from abc import abstractmethod
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from src.core.interfaces import DataFetcher
from src.core.logger import get_logger
from src.core.exceptions import IngestError


class BaseDataFetcher(DataFetcher):
    """
    데이터 수집기 기본 클래스

    모든 Fetcher/Crawler가 상속해야 하는 공통 기능 제공
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self._last_fetch_time: datetime | None = None

    @abstractmethod
    def fetch(self) -> Any:
        """데이터 수집 (하위 클래스에서 구현)"""
        pass

    @abstractmethod
    def get_source_name(self) -> str:
        """데이터 소스 이름 반환"""
        pass

    def _log_fetch_start(self, **kwargs) -> None:
        """수집 시작 로깅"""
        self.logger.info(f"[{self.get_source_name()}] 데이터 수집 시작", **kwargs)
        self._last_fetch_time = datetime.now()

    def _log_fetch_complete(self, count: int = 0, **kwargs) -> None:
        """수집 완료 로깅"""
        elapsed = None
        if self._last_fetch_time:
            elapsed = (datetime.now() - self._last_fetch_time).total_seconds()

        self.logger.info(
            f"[{self.get_source_name()}] 데이터 수집 완료: {count}건, "
            f"소요시간: {elapsed:.2f}초" if elapsed else f"[{self.get_source_name()}] 데이터 수집 완료: {count}건",
            **kwargs
        )

    def _log_fetch_error(self, error: Exception, **kwargs) -> None:
        """수집 오류 로깅"""
        self.logger.error(
            f"[{self.get_source_name()}] 데이터 수집 실패: {error}",
            **kwargs
        )

    @staticmethod
    def get_date_range(
        lookback_days: int,
        end_date: datetime | None = None
    ) -> tuple[str, str]:
        """
        조회 기간 계산

        Args:
            lookback_days: 조회 기간 (일)
            end_date: 종료일 (기본: 오늘)

        Returns:
            (시작일, 종료일) - 'YYYYMMDD' 형식
        """
        if end_date is None:
            end_date = datetime.now()

        start_date = end_date - timedelta(days=lookback_days)

        return (
            start_date.strftime("%Y%m%d"),
            end_date.strftime("%Y%m%d")
        )

    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        required_columns: list[str],
        source_name: str
    ) -> None:
        """
        DataFrame 유효성 검증

        Args:
            df: 검증할 DataFrame
            required_columns: 필수 컬럼 목록
            source_name: 데이터 소스 이름 (에러 메시지용)

        Raises:
            IngestError: 필수 컬럼이 없는 경우
        """
        if df.empty:
            raise IngestError(f"[{source_name}] 빈 DataFrame 반환됨")

        missing = set(required_columns) - set(df.columns)
        if missing:
            raise IngestError(
                f"[{source_name}] 필수 컬럼 누락: {missing}",
                {"missing_columns": list(missing)}
            )
