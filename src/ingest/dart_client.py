"""
DART OpenAPI 클라이언트 (OpenDartReader 기반)

재무제표 및 사업보고서 데이터 수집
- OpenDartReader 라이브러리 사용
- 종목코드/회사명으로 자동 조회
- 파일 기반 캐시 (24시간 유효)
"""
import os
import json
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from enum import Enum

import pandas as pd
import OpenDartReader

from src.ingest.base import BaseDataFetcher
from src.core.config import get_config
from src.core.exceptions import APIError


class ReportType(Enum):
    """보고서 유형"""
    ANNUAL = "11011"      # 사업보고서
    SEMI_ANNUAL = "11012"  # 반기보고서
    Q1 = "11013"          # 1분기보고서
    Q3 = "11014"          # 3분기보고서


class FinancialStatementType(Enum):
    """재무제표 유형"""
    BS = "BS"   # 재무상태표 (Balance Sheet)
    IS = "IS"   # 손익계산서 (Income Statement)
    CIS = "CIS"  # 포괄손익계산서
    CF = "CF"   # 현금흐름표
    SCE = "SCE"  # 자본변동표


@dataclass
class FinancialData:
    """재무제표 데이터"""
    stock_code: str
    stock_name: str
    report_type: str
    year: int
    quarter: int

    # 손익계산서 항목
    revenue: float = 0.0           # 매출액
    operating_profit: float = 0.0  # 영업이익
    net_income: float = 0.0        # 당기순이익
    rd_expense: float = 0.0        # 연구개발비

    # 재무상태표 항목
    total_assets: float = 0.0      # 총자산
    total_liabilities: float = 0.0  # 총부채
    total_equity: float = 0.0      # 총자본
    current_assets: float = 0.0    # 유동자산
    current_liabilities: float = 0.0  # 유동부채
    capital_stock: float = 0.0     # 자본금
    retained_earnings: float = 0.0  # 이익잉여금 (결손금은 음수)

    # 계산 지표
    debt_ratio: float = 0.0        # 부채비율
    current_ratio: float = 0.0     # 유동비율
    capital_impairment: float = 0.0  # 자본잠식률
    rd_ratio: float = 0.0          # R&D 비중

    fetched_at: datetime = None

    def __post_init__(self):
        if self.fetched_at is None:
            self.fetched_at = datetime.now()

        # 부채비율 계산 (부채/자본 * 100)
        if self.total_equity > 0:
            self.debt_ratio = (self.total_liabilities / self.total_equity) * 100

        # 유동비율 계산 (유동자산/유동부채 * 100)
        if self.current_liabilities > 0:
            self.current_ratio = (self.current_assets / self.current_liabilities) * 100

        # 자본잠식률 계산: (자본금 - 총자본) / 자본금 * 100
        if self.capital_stock > 0:
            if self.total_equity < self.capital_stock:
                self.capital_impairment = ((self.capital_stock - self.total_equity) / self.capital_stock) * 100
            else:
                self.capital_impairment = 0.0

        # R&D 비중 계산: 연구개발비 / 매출액 * 100
        if self.revenue > 0 and self.rd_expense > 0:
            self.rd_ratio = (self.rd_expense / self.revenue) * 100


class DartApiClient(BaseDataFetcher):
    """
    DART OpenAPI 클라이언트 (OpenDartReader 기반)

    사용법:
        client = DartApiClient()

        # 재무제표 조회 (종목코드 또는 회사명)
        financial = client.fetch_financial_summary("005930", 2024)
        financial = client.fetch_financial_summary("삼성전자", 2024)

        # 회사 정보 조회
        company = client.get_company_info("005930")
    """

    def __init__(self, api_key: str | None = None):
        super().__init__()

        config = get_config()

        # API 키: 인자 > 환경변수 > 설정파일
        self.api_key = (
            api_key
            or os.getenv("DART_API_KEY_2")  # 백업 키 우선
            or os.getenv("DART_API_KEY")
            or config.get("ingest.dart.api_key", "")
        )

        if not self.api_key or self.api_key.startswith("${"):
            self.logger.warning("DART API 키가 설정되지 않았습니다")
            self._dart = None
        else:
            # OpenDartReader 초기화
            self._dart = OpenDartReader(self.api_key)
            self.logger.info("OpenDartReader 초기화 완료")

        # 파일 기반 캐시 설정
        self.cache_enabled = config.get("ingest.dart.cache_enabled", True)
        self.cache_ttl_hours = config.get("ingest.dart.cache_ttl_hours", 24)
        self.cache_dir = Path(config.get("ingest.dart.cache_dir", "./cache/dart"))
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_source_name(self) -> str:
        return "DART"

    def fetch(self) -> Any:
        """기본 fetch는 미구현 (특정 종목 지정 필요)"""
        raise NotImplementedError("fetch_financial_summary() 사용")

    def _get_cache_key(self, method: str, *args) -> str:
        """캐시 키 생성"""
        key_str = f"{method}:{':'.join(str(a) for a in args)}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> dict | None:
        """캐시에서 데이터 조회"""
        if not self.cache_enabled:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)

            # TTL 확인
            cached_at = datetime.fromisoformat(cached.get("_cached_at", "2000-01-01"))
            if datetime.now() - cached_at > timedelta(hours=self.cache_ttl_hours):
                cache_file.unlink()  # 만료된 캐시 삭제
                return None

            self.logger.debug(f"캐시 히트: {cache_key[:8]}...")
            return cached.get("data")
        except Exception:
            return None

    def _save_to_cache(self, cache_key: str, data: dict) -> None:
        """캐시에 데이터 저장"""
        if not self.cache_enabled:
            return

        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({
                    "_cached_at": datetime.now().isoformat(),
                    "data": data
                }, f, ensure_ascii=False)
        except Exception as e:
            self.logger.debug(f"캐시 저장 실패: {e}")

    def get_company_info(self, corp: str) -> dict | None:
        """
        회사 정보 조회

        Args:
            corp: 종목코드 (005930) 또는 회사명 (삼성전자)

        Returns:
            회사 정보 dict 또는 None
        """
        if not self._dart:
            return None

        cache_key = self._get_cache_key("company", corp)
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        try:
            result = self._dart.company(corp)
            if result and result.get("status") == "000":
                self._save_to_cache(cache_key, result)
                return result
        except Exception as e:
            self.logger.warning(f"회사 정보 조회 실패 [{corp}]: {e}")

        return None

    def fetch_financial_statements(
        self,
        corp: str,
        year: int,
        report_type: ReportType = ReportType.ANNUAL
    ) -> pd.DataFrame:
        """
        재무제표 조회

        Args:
            corp: 종목코드 또는 회사명
            year: 사업연도
            report_type: 보고서 유형

        Returns:
            DataFrame with 재무제표 항목
        """
        if not self._dart:
            return pd.DataFrame()

        self.logger.debug(f"재무제표 조회: {corp}, {year}, {report_type.name}")

        try:
            df = self._dart.finstate(corp, year, reprt_code=report_type.value)
            if df is not None and len(df) > 0:
                return df
        except ValueError as e:
            # 회사를 찾을 수 없는 경우
            self.logger.debug(f"재무제표 조회 실패 [{corp}]: {e}")
        except Exception as e:
            self.logger.warning(f"재무제표 조회 실패 [{corp}]: {e}")

        return pd.DataFrame()

    def fetch_financial_summary(
        self,
        corp: str,
        year: int,
        report_type: ReportType = ReportType.ANNUAL
    ) -> FinancialData | None:
        """
        주요 재무지표 요약 조회

        Args:
            corp: 종목코드 또는 회사명
            year: 사업연도
            report_type: 보고서 유형

        Returns:
            FinancialData 객체 또는 None
        """
        # 캐시 확인
        cache_key = self._get_cache_key("financial_summary", corp, year, report_type.value)
        cached = self._get_from_cache(cache_key)
        if cached:
            return FinancialData(**cached)

        df = self.fetch_financial_statements(corp, year, report_type)

        if df.empty:
            return None

        try:
            # 주요 계정과목 추출
            def get_amount(account_nm: str, default: float = 0.0) -> float:
                """계정과목별 금액 추출"""
                row = df[df["account_nm"].str.contains(account_nm, na=False)]
                if not row.empty:
                    amt = row.iloc[0].get("thstrm_amount", "0")
                    if amt and amt != "-":
                        # 숫자 문자열 처리 (쉼표 제거)
                        return float(str(amt).replace(",", ""))
                return default

            # 회사 정보 가져오기
            company_info = self.get_company_info(corp)
            stock_code = company_info.get("stock_code", corp) if company_info else corp
            stock_name = company_info.get("corp_name", "") if company_info else ""

            # 분기 결정
            quarter_map = {
                ReportType.Q1: 1,
                ReportType.SEMI_ANNUAL: 2,
                ReportType.Q3: 3,
                ReportType.ANNUAL: 4,
            }

            result = FinancialData(
                stock_code=stock_code,
                stock_name=stock_name,
                report_type=report_type.value,
                year=year,
                quarter=quarter_map.get(report_type, 4),
                revenue=get_amount("매출액") or get_amount("수익"),
                operating_profit=get_amount("영업이익"),
                net_income=get_amount("당기순이익"),
                rd_expense=get_amount("연구개발비") or get_amount("경상연구개발비"),
                total_assets=get_amount("자산총계"),
                total_liabilities=get_amount("부채총계"),
                total_equity=get_amount("자본총계"),
                current_assets=get_amount("유동자산"),
                current_liabilities=get_amount("유동부채"),
                capital_stock=get_amount("자본금"),
                retained_earnings=get_amount("이익잉여금") or get_amount("결손금", 0.0),
            )

            # 캐시 저장
            self._save_to_cache(cache_key, {
                "stock_code": result.stock_code,
                "stock_name": result.stock_name,
                "report_type": result.report_type,
                "year": result.year,
                "quarter": result.quarter,
                "revenue": result.revenue,
                "operating_profit": result.operating_profit,
                "net_income": result.net_income,
                "rd_expense": result.rd_expense,
                "total_assets": result.total_assets,
                "total_liabilities": result.total_liabilities,
                "total_equity": result.total_equity,
                "current_assets": result.current_assets,
                "current_liabilities": result.current_liabilities,
                "capital_stock": result.capital_stock,
                "retained_earnings": result.retained_earnings,
            })

            return result

        except Exception as e:
            self.logger.error(f"재무지표 파싱 실패 [{corp}]: {e}")
            return None

    def get_comprehensive_financials(
        self,
        corp: str,
        year: int,
        max_fallback_years: int = 3
    ) -> dict:
        """
        종목의 종합 재무 정보 조회 (필터용)

        Args:
            corp: 종목코드 또는 회사명
            year: 기준 사업연도
            max_fallback_years: 데이터 없을 시 이전 연도 조회 횟수 (기본 3년)

        Returns:
            dict with all financial metrics needed for filtering
        """
        latest = None

        # 최근 연도부터 순차적으로 조회 (최대 max_fallback_years년 전까지)
        for try_year in range(year, year - max_fallback_years - 1, -1):
            # 사업보고서 먼저 시도
            latest = self.fetch_financial_summary(corp, try_year, ReportType.ANNUAL)
            if latest:
                break

            # 사업보고서 없으면 반기보고서 시도
            latest = self.fetch_financial_summary(corp, try_year, ReportType.SEMI_ANNUAL)
            if latest:
                break

        if not latest:
            self.logger.debug(f"재무 데이터 없음: {corp} ({year}~{year - max_fallback_years}년)")
            return {
                "stock_code": corp,
                "has_data": False,
            }

        return {
            "stock_code": latest.stock_code,
            "stock_name": latest.stock_name,
            "has_data": True,
            # Track A 필터용
            "operating_profit_4q": latest.operating_profit,  # 연간 보고서는 이미 4분기 합산
            "debt_ratio": latest.debt_ratio,
            # Track B 필터용
            "capital_impairment": latest.capital_impairment,
            "current_ratio": latest.current_ratio,
            "rd_ratio": latest.rd_ratio,
            # 원본 데이터
            "revenue": latest.revenue,
            "operating_profit": latest.operating_profit,
            "net_income": latest.net_income,
            "total_assets": latest.total_assets,
            "total_equity": latest.total_equity,
            "capital_stock": latest.capital_stock,
            "rd_expense": latest.rd_expense,
            # 메타
            "year": latest.year,
            "quarter": latest.quarter,
            "fetched_at": latest.fetched_at.isoformat() if latest.fetched_at else None,
        }

    def fetch_business_overview(
        self,
        corp: str,
        year: int | None = None,
        max_length: int = 5000
    ) -> str | None:
        """
        사업보고서에서 '사업의 개요' 섹션 조회

        Args:
            corp: 종목코드 또는 회사명
            year: 사업연도 (None이면 최신)
            max_length: 최대 텍스트 길이

        Returns:
            사업개요 텍스트 또는 None
        """
        if not self._dart:
            return None

        # 캐시 확인
        cache_key = self._get_cache_key("business_overview", corp, year or "latest")
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached.get("text")

        try:
            import requests
            from bs4 import BeautifulSoup

            # 사업보고서 목록 조회
            if year:
                start_date = f"{year}-01-01"
                end_date = f"{year}-12-31"
            else:
                from datetime import datetime
                current_year = datetime.now().year
                start_date = f"{current_year - 2}-01-01"
                end_date = f"{current_year}-12-31"

            reports = self._dart.list(corp, start=start_date, end=end_date, kind='A')

            if reports is None or len(reports) == 0:
                self.logger.debug(f"사업보고서 없음: {corp}")
                return None

            # 사업보고서 필터링
            annual = reports[reports['report_nm'].str.contains('사업보고서', na=False)]
            if len(annual) == 0:
                self.logger.debug(f"사업보고서 없음: {corp}")
                return None

            rcept_no = annual.iloc[0]['rcept_no']

            # 문서 목차 조회
            subdocs = self._dart.sub_docs(rcept_no)
            if subdocs is None or len(subdocs) == 0:
                return None

            # '사업의 개요' 섹션 찾기
            biz_overview = subdocs[subdocs['title'].str.contains('사업의 개요', na=False)]
            if len(biz_overview) == 0:
                # 대체: 'II. 사업의 내용' 섹션
                biz_overview = subdocs[subdocs['title'].str.contains('사업의 내용', na=False)]

            if len(biz_overview) == 0:
                self.logger.debug(f"사업개요 섹션 없음: {corp}")
                return None

            url = biz_overview.iloc[0]['url']

            # HTML 내용 가져오기
            resp = requests.get(url, timeout=10)
            resp.encoding = 'utf-8'
            soup = BeautifulSoup(resp.text, 'html.parser')

            # 텍스트 추출 (HTML 태그 제거)
            text = soup.get_text(separator=' ', strip=True)

            # 길이 제한
            if len(text) > max_length:
                text = text[:max_length] + "..."

            # 캐시 저장
            self._save_to_cache(cache_key, {"text": text})

            self.logger.debug(f"사업개요 조회 성공: {corp} ({len(text)}자)")
            return text

        except Exception as e:
            self.logger.warning(f"사업개요 조회 실패 [{corp}]: {e}")
            return None

    def close(self):
        """리소스 정리"""
        pass
