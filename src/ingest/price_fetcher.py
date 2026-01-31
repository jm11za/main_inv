"""
주가 데이터 수집기 (pykrx 기반)

KRX에서 OHLCV, 수급, 시가총액 데이터 수집
"""
import time
import random
from datetime import datetime
from typing import Any

import pandas as pd
from pykrx import stock

from src.ingest.base import BaseDataFetcher
from src.core.config import get_config
from src.core.exceptions import IngestError


class PriceDataFetcher(BaseDataFetcher):
    """
    주가 데이터 수집기

    pykrx를 사용하여 KRX에서 데이터 수집

    사용법:
        fetcher = PriceDataFetcher(lookback_days=120)

        # 전체 종목 OHLCV
        ohlcv_df = fetcher.fetch()

        # 특정 종목 OHLCV
        samsung_df = fetcher.fetch_stock_ohlcv("005930")

        # 수급 데이터 (외국인/기관 순매수)
        supply_df = fetcher.fetch_investor_trading("005930")
    """

    def __init__(
        self,
        lookback_days: int | None = None,
        min_delay: float = 0.1,
        max_delay: float = 0.2,
    ):
        super().__init__()

        config = get_config()
        self.lookback_days = lookback_days or config.get(
            "ingest.price.lookback_days", 120
        )

        # API 호출 딜레이 설정
        self.min_delay = min_delay
        self.max_delay = max_delay
        self._request_count = 0

    def _api_delay(self):
        """API 호출 간 딜레이 (rate limit 방지)"""
        self._request_count += 1
        # 10회마다 조금 더 긴 딜레이
        if self._request_count % 10 == 0:
            time.sleep(random.uniform(0.3, 0.5))
        else:
            time.sleep(random.uniform(self.min_delay, self.max_delay))

    def get_source_name(self) -> str:
        return "KRX/pykrx"

    def fetch(self) -> pd.DataFrame:
        """
        전체 종목 일별 OHLCV 데이터 수집 (최근 거래일 기준)

        Returns:
            DataFrame with columns:
                - 종목코드, 종목명, 시가, 고가, 저가, 종가, 거래량, 거래대금, 등락률
        """
        self._log_fetch_start()

        try:
            # 최근 거래일 조회
            today = datetime.now().strftime("%Y%m%d")
            df = stock.get_market_ohlcv_by_ticker(today, market="ALL")

            if df.empty:
                # 오늘이 휴장일이면 이전 거래일 조회
                yesterday = stock.get_previous_business_days(year=datetime.now().year)
                if len(yesterday) > 0:
                    last_trading_day = yesterday[-1].strftime("%Y%m%d")
                    df = stock.get_market_ohlcv_by_ticker(last_trading_day, market="ALL")

            if df.empty:
                raise IngestError("OHLCV 데이터를 가져올 수 없습니다")

            # 인덱스(종목코드)를 컬럼으로 변환
            df = df.reset_index()
            df = df.rename(columns={"티커": "종목코드"})

            self._log_fetch_complete(count=len(df))
            return df

        except Exception as e:
            self._log_fetch_error(e)
            raise IngestError(f"OHLCV 데이터 수집 실패: {e}")

    def fetch_stock_ohlcv(
        self,
        stock_code: str,
        start_date: str | None = None,
        end_date: str | None = None
    ) -> pd.DataFrame:
        """
        특정 종목의 기간별 OHLCV 데이터 수집

        Args:
            stock_code: 종목코드 (예: "005930")
            start_date: 시작일 (YYYYMMDD), 기본값: lookback_days 전
            end_date: 종료일 (YYYYMMDD), 기본값: 오늘

        Returns:
            DataFrame with columns:
                - 날짜, 시가, 고가, 저가, 종가, 거래량, 거래대금, 등락률
        """
        if start_date is None or end_date is None:
            start_date, end_date = self.get_date_range(self.lookback_days)

        try:
            self._api_delay()
            df = stock.get_market_ohlcv_by_date(
                start_date, end_date, stock_code
            )

            if df.empty:
                self.logger.warning(f"[{stock_code}] OHLCV 데이터 없음")
                return pd.DataFrame()

            df = df.reset_index()
            df = df.rename(columns={"날짜": "date"})

            return df

        except Exception as e:
            self.logger.error(f"[{stock_code}] OHLCV 조회 실패: {e}")
            raise IngestError(f"종목 OHLCV 수집 실패: {stock_code}", {"error": str(e)})

    def fetch_investor_trading(
        self,
        stock_code: str,
        start_date: str | None = None,
        end_date: str | None = None
    ) -> pd.DataFrame:
        """
        투자자별 순매수 데이터 수집 (수급 분석용)

        Args:
            stock_code: 종목코드
            start_date: 시작일 (YYYYMMDD)
            end_date: 종료일 (YYYYMMDD)

        Returns:
            DataFrame with columns:
                - 날짜, 기관합계, 외국인합계, 개인, ...
        """
        self.logger.debug(f"투자자별 순매수 조회: {stock_code}")

        if start_date is None or end_date is None:
            start_date, end_date = self.get_date_range(self.lookback_days)

        try:
            df = stock.get_market_net_purchases_of_equities_by_ticker(
                start_date, end_date, stock_code
            )

            if df.empty:
                self.logger.warning(f"[{stock_code}] 수급 데이터 없음")
                return pd.DataFrame()

            df = df.reset_index()

            return df

        except Exception as e:
            self.logger.error(f"[{stock_code}] 수급 데이터 조회 실패: {e}")
            return pd.DataFrame()

    def fetch_market_cap(
        self,
        date: str | None = None,
        market: str = "ALL"
    ) -> pd.DataFrame:
        """
        전체 종목 시가총액 데이터 수집

        Args:
            date: 조회일 (YYYYMMDD), 기본값: 오늘
            market: 시장 구분 (ALL, KOSPI, KOSDAQ)

        Returns:
            DataFrame with columns:
                - 종목코드, 종목명, 시가총액, 상장주식수
        """
        self.logger.debug(f"시가총액 조회: {date or '오늘'}")

        if date is None:
            date = datetime.now().strftime("%Y%m%d")

        try:
            df = stock.get_market_cap_by_ticker(date, market=market)

            if df.empty:
                self.logger.warning("시가총액 데이터 없음")
                return pd.DataFrame()

            df = df.reset_index()
            df = df.rename(columns={"티커": "종목코드"})

            return df

        except Exception as e:
            self.logger.error(f"시가총액 조회 실패: {e}")
            raise IngestError(f"시가총액 수집 실패: {e}")

    def fetch_all_stock_codes(self, market: str = "ALL") -> pd.DataFrame:
        """
        전체 종목 코드 및 종목명 조회

        Args:
            market: 시장 구분 (ALL, KOSPI, KOSDAQ)

        Returns:
            DataFrame with columns: 종목코드, 종목명
        """
        try:
            date = datetime.now().strftime("%Y%m%d")

            # KOSPI + KOSDAQ 조회
            if market == "ALL":
                kospi_tickers = stock.get_market_ticker_list(date, market="KOSPI")
                kosdaq_tickers = stock.get_market_ticker_list(date, market="KOSDAQ")
                tickers = kospi_tickers + kosdaq_tickers
            else:
                tickers = stock.get_market_ticker_list(date, market=market)

            # 종목명 조회
            data = []
            for ticker in tickers:
                name = stock.get_market_ticker_name(ticker)
                data.append({"종목코드": ticker, "종목명": name})

            return pd.DataFrame(data)

        except Exception as e:
            self.logger.error(f"종목 목록 조회 실패: {e}")
            raise IngestError(f"종목 목록 수집 실패: {e}")

    def fetch_fundamental(
        self,
        date: str | None = None,
        market: str = "ALL"
    ) -> pd.DataFrame:
        """
        전체 종목 기본 재무지표 (PER, PBR, EPS, BPS, DIV, DPS)

        Args:
            date: 조회일 (YYYYMMDD)
            market: 시장 구분

        Returns:
            DataFrame with fundamental metrics
        """
        if date is None:
            date = datetime.now().strftime("%Y%m%d")

        try:
            df = stock.get_market_fundamental_by_ticker(date, market=market)

            if df.empty:
                return pd.DataFrame()

            df = df.reset_index()
            df = df.rename(columns={"티커": "종목코드"})

            return df

        except Exception as e:
            self.logger.error(f"재무지표 조회 실패: {e}")
            return pd.DataFrame()

    def fetch_stock_fundamental(
        self,
        stock_code: str,
        date: str | None = None
    ) -> dict:
        """
        개별 종목 기본 재무지표 (PER, PBR)

        Args:
            stock_code: 종목코드
            date: 조회일 (YYYYMMDD)

        Returns:
            dict with PER, PBR values
        """
        from datetime import timedelta

        # 오늘 데이터는 없을 수 있으므로 최근 5일 중 데이터 있는 날짜 사용
        if date is None:
            for days_back in range(5):
                check_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
                try:
                    self._api_delay()
                    df = stock.get_market_fundamental(check_date, check_date, stock_code)
                    if not df.empty:
                        date = check_date
                        break
                except Exception:
                    continue

        if date is None:
            return {"per": 0.0, "pbr": 0.0}

        try:
            df = stock.get_market_fundamental(date, date, stock_code)

            if df.empty:
                return {"per": 0.0, "pbr": 0.0}

            # 단일 종목 조회 시 첫 행 사용
            row = df.iloc[0]
            return {
                "per": float(row.get("PER", 0) or 0),
                "pbr": float(row.get("PBR", 0) or 0),
            }

        except Exception as e:
            self.logger.error(f"[{stock_code}] 재무지표 조회 실패: {e}")
            return {"per": 0.0, "pbr": 0.0}

    def fetch_stock_trading_value(
        self,
        stock_code: str,
        days: int = 20
    ) -> float:
        """
        개별 종목 평균 거래대금 (N일 평균)

        Args:
            stock_code: 종목코드
            days: 평균 계산 일수 (기본 20일)

        Returns:
            평균 거래대금 (원)
        """
        start_date, end_date = self.get_date_range(days)

        try:
            self._api_delay()
            # get_market_cap 사용 (거래대금 포함)
            df = stock.get_market_cap(start_date, end_date, stock_code)

            if df.empty:
                return 0.0

            if "거래대금" in df.columns:
                return float(df["거래대금"].mean())
            else:
                self.logger.warning(f"[{stock_code}] 거래대금 컬럼 없음: {df.columns.tolist()}")
                return 0.0

        except Exception as e:
            self.logger.error(f"[{stock_code}] 거래대금 조회 실패: {e}")
            return 0.0

    def fetch_stock_supply_demand(
        self,
        stock_code: str,
        days: int = 20
    ) -> dict:
        """
        개별 종목 수급 데이터 (S_Flow 계산용)

        Args:
            stock_code: 종목코드
            days: 조회 기간

        Returns:
            dict with foreign_net, institution_net, market_cap
        """
        start_date, end_date = self.get_date_range(days)

        result = {
            "foreign_net": 0.0,
            "institution_net": 0.0,
            "market_cap": 0.0,
        }

        try:
            self._api_delay()
            # 투자자별 순매수 데이터 (금액 기준)
            supply_df = stock.get_market_trading_value_by_date(
                start_date, end_date, stock_code
            )

            if not supply_df.empty:
                # 기간 합산 (금액 단위)
                if "외국인합계" in supply_df.columns:
                    result["foreign_net"] = float(supply_df["외국인합계"].sum())
                if "기관합계" in supply_df.columns:
                    result["institution_net"] = float(supply_df["기관합계"].sum())

            # 시가총액 (최근 거래일 찾기)
            from datetime import timedelta
            for days_back in range(5):
                check_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=days_back)).strftime("%Y%m%d")
                try:
                    self._api_delay()
                    cap_df = stock.get_market_cap(check_date, check_date, stock_code)
                    if not cap_df.empty and "시가총액" in cap_df.columns:
                        result["market_cap"] = float(cap_df["시가총액"].iloc[0])
                        break
                except Exception:
                    continue

        except Exception as e:
            self.logger.error(f"[{stock_code}] 수급 데이터 조회 실패: {e}")

        return result
