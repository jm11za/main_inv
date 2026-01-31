"""
데이터 변환 모듈

Layer 1 (Ingest) → Layer 3/3.5/4 간 데이터 변환
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from src.core.logger import get_logger
from src.core.config import get_config
from src.ingest.price_fetcher import PriceDataFetcher
from src.ingest.dart_client import DartApiClient


@dataclass
class StockFinancials:
    """필터링/스코어링용 종목 재무 데이터"""
    stock_code: str
    stock_name: str = ""

    # Track A 필터용
    operating_profit_4q: float = 0.0  # 4분기 합산 영업이익
    debt_ratio: float = 0.0           # 부채비율 (%)
    pbr: float = 0.0                  # PBR
    avg_trading_value: float = 0.0    # 일평균 거래대금

    # Track B 필터용
    capital_impairment: float = 0.0   # 자본잠식률 (%)
    current_ratio: float = 0.0        # 유동비율 (%)
    rd_ratio: float = 0.0             # R&D 비중 (%)

    # 분석용 추가 데이터
    market_cap: float = 0.0           # 시가총액
    per: float = 0.0                  # PER

    # 메타
    data_date: str = ""
    has_dart_data: bool = False
    has_price_data: bool = False

    def to_filter_dict(self) -> dict:
        """필터 적용용 dict 변환"""
        return {
            "stock_code": self.stock_code,
            "stock_name": self.stock_name,
            "operating_profit_4q": self.operating_profit_4q,
            "debt_ratio": self.debt_ratio,
            "pbr": self.pbr,
            "avg_trading_value": self.avg_trading_value,
            "capital_impairment": self.capital_impairment,
            "current_ratio": self.current_ratio,
            "rd_ratio": self.rd_ratio,
        }


@dataclass
class StockSupplyDemand:
    """수급 분석용 데이터"""
    stock_code: str

    # 기간 누적 순매수 (단위: 주)
    foreign_net: float = 0.0      # 외국인 순매수
    institution_net: float = 0.0  # 기관 순매수
    individual_net: float = 0.0   # 개인 순매수

    # 금액 기준 (단위: 원)
    foreign_net_amount: float = 0.0
    institution_net_amount: float = 0.0

    # 최근 거래량 평균
    avg_volume: float = 0.0
    avg_trading_value: float = 0.0

    # 시가총액 (S_Flow 계산용)
    market_cap: float = 0.0

    # 기간
    start_date: str = ""
    end_date: str = ""

    def get_s_flow_inputs(self) -> dict:
        """S_Flow 계산용 입력값"""
        return {
            "foreign_net": self.foreign_net_amount,
            "institution_net": self.institution_net_amount,
            "market_cap": self.market_cap,
        }


class DataTransformer:
    """
    Layer 1 데이터를 분석/필터링 레이어용으로 변환

    사용법:
        transformer = DataTransformer()

        # 종목별 재무 데이터 (필터용)
        financials = transformer.get_stock_financials("005930", 2025)

        # 수급 데이터 (S_Flow용)
        supply = transformer.get_supply_demand("005930", days=20)

        # 일괄 조회
        all_data = transformer.prepare_filter_data(stock_codes, 2025)
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.config = get_config()

        self.price_fetcher = PriceDataFetcher()
        self.dart_client = DartApiClient()

        # 캐시
        self._fundamental_cache: pd.DataFrame | None = None
        self._market_cap_cache: pd.DataFrame | None = None
        self._cache_date: str = ""

    def _refresh_cache_if_needed(self, date: str | None = None):
        """일별 캐시 갱신"""
        if date is None:
            date = datetime.now().strftime("%Y%m%d")

        if self._cache_date != date:
            self.logger.debug(f"캐시 갱신: {date}")
            self._fundamental_cache = self.price_fetcher.fetch_fundamental(date)
            self._market_cap_cache = self.price_fetcher.fetch_market_cap(date)
            self._cache_date = date

    def get_stock_financials(
        self,
        stock_code: str,
        year: int,
        date: str | None = None
    ) -> StockFinancials:
        """
        종목 재무 데이터 통합 조회

        Args:
            stock_code: 종목코드
            year: 사업연도 (DART용)
            date: 기준일 (pykrx용)

        Returns:
            StockFinancials 객체
        """
        if date is None:
            date = datetime.now().strftime("%Y%m%d")

        self._refresh_cache_if_needed(date)

        result = StockFinancials(stock_code=stock_code, data_date=date)

        # 1. DART 재무 데이터
        try:
            dart_data = self.dart_client.get_comprehensive_financials(stock_code, year)

            if dart_data.get("has_data"):
                result.has_dart_data = True
                result.stock_name = dart_data.get("stock_name", "")
                result.operating_profit_4q = dart_data.get("operating_profit_4q", 0)
                result.debt_ratio = dart_data.get("debt_ratio", 0)
                result.capital_impairment = dart_data.get("capital_impairment", 0)
                result.current_ratio = dart_data.get("current_ratio", 0)
                result.rd_ratio = dart_data.get("rd_ratio", 0)

        except Exception as e:
            self.logger.warning(f"[{stock_code}] DART 조회 실패: {e}")

        # 2. pykrx 시장 데이터 (PBR, PER)
        if self._fundamental_cache is not None and not self._fundamental_cache.empty:
            fund_row = self._fundamental_cache[
                self._fundamental_cache["종목코드"] == stock_code
            ]
            if not fund_row.empty:
                result.has_price_data = True
                result.pbr = float(fund_row.iloc[0].get("PBR", 0))
                result.per = float(fund_row.iloc[0].get("PER", 0))

        # 3. 시가총액
        if self._market_cap_cache is not None and not self._market_cap_cache.empty:
            cap_row = self._market_cap_cache[
                self._market_cap_cache["종목코드"] == stock_code
            ]
            if not cap_row.empty:
                result.market_cap = float(cap_row.iloc[0].get("시가총액", 0))

        # 4. 거래대금 (최근 20일 평균)
        try:
            ohlcv = self.price_fetcher.fetch_stock_ohlcv(stock_code)
            if not ohlcv.empty and "거래대금" in ohlcv.columns:
                result.avg_trading_value = ohlcv["거래대금"].tail(20).mean()
        except Exception as e:
            self.logger.debug(f"[{stock_code}] 거래대금 조회 실패: {e}")

        return result

    def get_supply_demand(
        self,
        stock_code: str,
        days: int = 20
    ) -> StockSupplyDemand:
        """
        수급 데이터 조회 (S_Flow 계산용)

        Args:
            stock_code: 종목코드
            days: 조회 기간 (일)

        Returns:
            StockSupplyDemand 객체
        """
        result = StockSupplyDemand(stock_code=stock_code)

        # 날짜 범위 계산
        start_date, end_date = self.price_fetcher.get_date_range(days)
        result.start_date = start_date
        result.end_date = end_date

        try:
            # 투자자별 순매수 조회
            supply_df = self.price_fetcher.fetch_investor_trading(
                stock_code, start_date, end_date
            )

            if not supply_df.empty:
                # 기간 합계
                if "외국인합계" in supply_df.columns:
                    result.foreign_net = supply_df["외국인합계"].sum()
                if "기관합계" in supply_df.columns:
                    result.institution_net = supply_df["기관합계"].sum()
                if "개인" in supply_df.columns:
                    result.individual_net = supply_df["개인"].sum()

            # OHLCV에서 거래량/거래대금 평균
            ohlcv = self.price_fetcher.fetch_stock_ohlcv(
                stock_code, start_date, end_date
            )

            if not ohlcv.empty:
                if "거래량" in ohlcv.columns:
                    result.avg_volume = ohlcv["거래량"].mean()
                if "거래대금" in ohlcv.columns:
                    result.avg_trading_value = ohlcv["거래대금"].mean()

                # 순매수 금액 추정 (주수 * 평균가)
                if "종가" in ohlcv.columns:
                    avg_price = ohlcv["종가"].mean()
                    result.foreign_net_amount = result.foreign_net * avg_price
                    result.institution_net_amount = result.institution_net * avg_price

            # 시가총액 조회
            self._refresh_cache_if_needed()
            if self._market_cap_cache is not None:
                cap_row = self._market_cap_cache[
                    self._market_cap_cache["종목코드"] == stock_code
                ]
                if not cap_row.empty:
                    result.market_cap = float(cap_row.iloc[0].get("시가총액", 0))

        except Exception as e:
            self.logger.warning(f"[{stock_code}] 수급 데이터 조회 실패: {e}")

        return result

    def prepare_filter_data(
        self,
        stock_codes: list[str],
        year: int,
        date: str | None = None
    ) -> list[dict]:
        """
        복수 종목 필터용 데이터 일괄 준비

        Args:
            stock_codes: 종목코드 리스트
            year: 사업연도
            date: 기준일

        Returns:
            필터 입력용 dict 리스트
        """
        self.logger.info(f"{len(stock_codes)}개 종목 필터 데이터 준비 중...")

        if date is None:
            date = datetime.now().strftime("%Y%m%d")

        self._refresh_cache_if_needed(date)

        results = []
        for i, code in enumerate(stock_codes):
            if (i + 1) % 50 == 0:
                self.logger.debug(f"진행: {i + 1}/{len(stock_codes)}")

            financials = self.get_stock_financials(code, year, date)
            results.append(financials.to_filter_dict())

        self.logger.info(f"필터 데이터 준비 완료: {len(results)}개")
        return results

    def prepare_flow_data(
        self,
        stock_codes: list[str],
        days: int = 20
    ) -> list[dict]:
        """
        복수 종목 S_Flow 계산용 데이터 일괄 준비

        Args:
            stock_codes: 종목코드 리스트
            days: 조회 기간

        Returns:
            S_Flow 입력용 dict 리스트
        """
        self.logger.info(f"{len(stock_codes)}개 종목 수급 데이터 준비 중...")

        results = []
        for i, code in enumerate(stock_codes):
            if (i + 1) % 50 == 0:
                self.logger.debug(f"진행: {i + 1}/{len(stock_codes)}")

            supply = self.get_supply_demand(code, days)
            results.append(supply.get_s_flow_inputs())

        self.logger.info(f"수급 데이터 준비 완료: {len(results)}개")
        return results

    def transform_filter_result_to_scorer_input(
        self,
        filter_result: dict,
        supply_data: dict | None = None
    ) -> dict:
        """
        FilterResult → StockScorer 입력 변환

        Args:
            filter_result: 필터 결과 (FilterResult.metrics)
            supply_data: 수급 데이터 (optional)

        Returns:
            StockScorer 입력용 dict
        """
        scorer_input = {
            # 재무 지표
            "financial": {
                "operating_profit": filter_result.get("operating_profit_4q", 0),
                "debt_ratio": filter_result.get("debt_ratio", 0),
                "current_ratio": filter_result.get("current_ratio", 0),
                "capital_impairment": filter_result.get("capital_impairment", 0),
                "rd_ratio": filter_result.get("rd_ratio", 0),
                "pbr": filter_result.get("pbr", 0),
            },
            # 기술적 지표 (나중에 분석 레이어에서 추가)
            "technical": {
                "s_flow": 0,
                "s_breadth": 0,
                "s_trend": 0,
            },
        }

        # 수급 데이터가 있으면 추가
        if supply_data:
            scorer_input["supply"] = supply_data

        return scorer_input
