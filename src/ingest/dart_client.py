"""
DART OpenAPI 클라이언트

재무제표 및 사업보고서 데이터 수집
"""
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from enum import Enum

import requests
import pandas as pd

from src.ingest.base import BaseDataFetcher
from src.core.config import get_config
from src.core.exceptions import APIError, RateLimitError


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
        # 총자본 < 자본금 이면 자본잠식 상태
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
    DART OpenAPI 클라이언트

    사용법:
        client = DartApiClient()

        # 재무제표 조회
        financial = client.fetch_financial_statements("005930", 2024)

        # 공시 목록 조회
        disclosures = client.fetch_disclosure_list("005930")
    """

    BASE_URL = "https://opendart.fss.or.kr/api"

    def __init__(self, api_key: str | None = None):
        super().__init__()

        config = get_config()

        # API 키: 인자 > 환경변수 > 설정파일
        self.api_key = (
            api_key
            or os.getenv("DART_API_KEY")
            or config.get("ingest.dart.api_key", "")
        )

        if not self.api_key or self.api_key.startswith("${"):
            self.logger.warning("DART API 키가 설정되지 않았습니다")

        self.timeout = config.get("ingest.dart.timeout_seconds", 30)
        self.retry_count = config.get("ingest.dart.retry_count", 3)
        self.session = requests.Session()

        # 종목코드 → DART 고유번호 매핑 캐시
        self._corp_code_cache: dict[str, str] = {}

    def get_source_name(self) -> str:
        return "DART"

    def fetch(self) -> Any:
        """기본 fetch는 미구현 (특정 종목 지정 필요)"""
        raise NotImplementedError("fetch_financial_statements() 또는 fetch_disclosure_list() 사용")

    def _request(
        self,
        endpoint: str,
        params: dict | None = None,
        retry: int = 0
    ) -> dict:
        """API 요청 공통 처리"""
        url = f"{self.BASE_URL}/{endpoint}"

        if params is None:
            params = {}
        params["crtfc_key"] = self.api_key

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()

            # DART API 에러 코드 처리
            status = data.get("status", "000")
            if status == "013":
                raise RateLimitError("DART API 요청 한도 초과", retry_after=60)
            elif status == "020":
                # 조회된 데이터 없음 (정상 케이스)
                return {"status": "020", "message": "조회된 데이터 없음", "list": []}
            elif status != "000":
                raise APIError(f"DART API 오류: {data.get('message', status)}")

            return data

        except requests.RequestException as e:
            if retry < self.retry_count:
                self.logger.warning(f"DART API 재시도 ({retry + 1}/{self.retry_count}): {e}")
                time.sleep(2 ** retry)  # 지수 백오프
                return self._request(endpoint, params, retry + 1)
            raise APIError(f"DART API 요청 실패: {e}")

    def fetch_corp_code(self, stock_code: str) -> str | None:
        """
        종목코드로 DART 고유번호 조회

        Args:
            stock_code: 종목코드 (예: "005930")

        Returns:
            DART 고유번호 (8자리) 또는 None
        """
        # 캐시 확인
        if stock_code in self._corp_code_cache:
            return self._corp_code_cache[stock_code]

        # 고유번호 파일이 없으면 다운로드 필요
        # (실제로는 corpCode.xml 파일을 다운받아 파싱해야 함)
        # 여기서는 간단히 API로 회사 검색
        try:
            data = self._request("company.json", {"corp_code": stock_code})
            if data.get("status") == "000":
                corp_code = data.get("corp_code")
                self._corp_code_cache[stock_code] = corp_code
                return corp_code
        except Exception:
            pass

        return None

    def fetch_financial_statements(
        self,
        stock_code: str,
        year: int,
        report_type: ReportType = ReportType.ANNUAL,
        fs_div: str = "CFS"  # CFS: 연결, OFS: 개별
    ) -> pd.DataFrame:
        """
        재무제표 조회 (전체 계정과목)

        Args:
            stock_code: 종목코드
            year: 사업연도
            report_type: 보고서 유형
            fs_div: 재무제표 구분 (CFS: 연결, OFS: 개별)

        Returns:
            DataFrame with 재무제표 항목
        """
        self.logger.debug(f"재무제표 조회: {stock_code}, {year}, {report_type.name}")

        try:
            data = self._request("fnlttSinglAcntAll.json", {
                "corp_code": stock_code,
                "bsns_year": str(year),
                "reprt_code": report_type.value,
                "fs_div": fs_div,
            })

            if not data.get("list"):
                return pd.DataFrame()

            df = pd.DataFrame(data["list"])
            return df

        except Exception as e:
            self.logger.error(f"재무제표 조회 실패 [{stock_code}]: {e}")
            return pd.DataFrame()

    def fetch_financial_summary(
        self,
        stock_code: str,
        year: int,
        report_type: ReportType = ReportType.ANNUAL
    ) -> FinancialData | None:
        """
        주요 재무지표 요약 조회

        Args:
            stock_code: 종목코드
            year: 사업연도
            report_type: 보고서 유형

        Returns:
            FinancialData 객체 또는 None
        """
        df = self.fetch_financial_statements(stock_code, year, report_type)

        if df.empty:
            return None

        try:
            # 주요 계정과목 추출
            def get_amount(account_nm: str, default: float = 0.0) -> float:
                """계정과목별 금액 추출"""
                row = df[df["account_nm"].str.contains(account_nm, na=False)]
                if not row.empty:
                    # thstrm_amount: 당기금액
                    amt = row.iloc[0].get("thstrm_amount", "0")
                    if amt and amt != "-":
                        return float(str(amt).replace(",", ""))
                return default

            # 분기 결정
            quarter_map = {
                ReportType.Q1: 1,
                ReportType.SEMI_ANNUAL: 2,
                ReportType.Q3: 3,
                ReportType.ANNUAL: 4,
            }

            return FinancialData(
                stock_code=stock_code,
                stock_name=df.iloc[0].get("corp_name", "") if not df.empty else "",
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

        except Exception as e:
            self.logger.error(f"재무지표 파싱 실패 [{stock_code}]: {e}")
            return None

    def fetch_financial_4q(
        self,
        stock_code: str,
        year: int,
        fs_div: str = "CFS"
    ) -> list[FinancialData]:
        """
        최근 4분기 재무제표 조회

        Args:
            stock_code: 종목코드
            year: 기준 사업연도
            fs_div: 재무제표 구분 (CFS: 연결, OFS: 개별)

        Returns:
            4분기 FinancialData 리스트 (최신순)
        """
        results = []
        report_types = [
            (year, ReportType.ANNUAL),       # 4Q
            (year, ReportType.Q3),           # 3Q
            (year, ReportType.SEMI_ANNUAL),  # 2Q
            (year, ReportType.Q1),           # 1Q
        ]

        for y, rt in report_types:
            data = self.fetch_financial_summary(stock_code, y, rt)
            if data:
                results.append(data)

        return results

    def get_operating_profit_4q_sum(
        self,
        stock_code: str,
        year: int
    ) -> float:
        """
        최근 4분기 영업이익 합산

        Args:
            stock_code: 종목코드
            year: 기준 사업연도

        Returns:
            4분기 합산 영업이익 (데이터 없으면 0)
        """
        quarters = self.fetch_financial_4q(stock_code, year)

        if not quarters:
            return 0.0

        # 연간 보고서가 있으면 그걸 사용 (이미 4분기 합산됨)
        for q in quarters:
            if q.quarter == 4:
                return q.operating_profit

        # 분기별 합산
        return sum(q.operating_profit for q in quarters)

    def get_comprehensive_financials(
        self,
        stock_code: str,
        year: int
    ) -> dict:
        """
        종목의 종합 재무 정보 조회 (필터용)

        Args:
            stock_code: 종목코드
            year: 기준 사업연도

        Returns:
            dict with all financial metrics needed for filtering
        """
        # 최신 재무제표 조회
        latest = self.fetch_financial_summary(stock_code, year, ReportType.ANNUAL)

        if not latest:
            # 사업보고서 없으면 반기 시도
            latest = self.fetch_financial_summary(stock_code, year, ReportType.SEMI_ANNUAL)

        if not latest:
            self.logger.warning(f"재무 데이터 없음: {stock_code}")
            return {
                "stock_code": stock_code,
                "has_data": False,
            }

        # 4분기 합산 영업이익
        op_profit_4q = self.get_operating_profit_4q_sum(stock_code, year)

        # 사업보고서 텍스트 조회 (섹터 라벨링용)
        business_text = self.fetch_business_report(stock_code, year)

        return {
            "stock_code": stock_code,
            "stock_name": latest.stock_name,
            "has_data": True,
            # Track A 필터용
            "operating_profit_4q": op_profit_4q,
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
            # 사업보고서 (섹터 라벨링용)
            "business_report": business_text,
            # 메타
            "year": latest.year,
            "quarter": latest.quarter,
            "fetched_at": latest.fetched_at.isoformat() if latest.fetched_at else None,
        }

    def fetch_disclosure_list(
        self,
        stock_code: str,
        start_date: str | None = None,
        end_date: str | None = None,
        pblntf_ty: str = ""  # 공시유형: A=정기공시, B=주요사항, ...
    ) -> pd.DataFrame:
        """
        공시 목록 조회

        Args:
            stock_code: 종목코드
            start_date: 시작일 (YYYYMMDD)
            end_date: 종료일 (YYYYMMDD)
            pblntf_ty: 공시유형 필터

        Returns:
            DataFrame with 공시 목록
        """
        if not start_date or not end_date:
            start_date, end_date = self.get_date_range(365)  # 최근 1년

        try:
            data = self._request("list.json", {
                "corp_code": stock_code,
                "bgn_de": start_date,
                "end_de": end_date,
                "pblntf_ty": pblntf_ty,
                "page_count": "100",
            })

            if not data.get("list"):
                return pd.DataFrame()

            return pd.DataFrame(data["list"])

        except Exception as e:
            self.logger.error(f"공시 목록 조회 실패 [{stock_code}]: {e}")
            return pd.DataFrame()

    def fetch_business_report(self, stock_code: str, year: int) -> str:
        """
        사업보고서 '사업의 내용' 텍스트 조회

        Args:
            stock_code: 종목코드
            year: 사업연도

        Returns:
            사업의 내용 텍스트 (LLM 분석용)
        """
        # 사업보고서 공시 검색
        df = self.fetch_disclosure_list(
            stock_code,
            start_date=f"{year}0101",
            end_date=f"{year}1231",
            pblntf_ty="A"  # 정기공시
        )

        if df.empty:
            # 전년도 사업보고서도 시도
            df = self.fetch_disclosure_list(
                stock_code,
                start_date=f"{year-1}0101",
                end_date=f"{year-1}1231",
                pblntf_ty="A"
            )

        if df.empty:
            return ""

        # '사업보고서' 필터
        annual_reports = df[df["report_nm"].str.contains("사업보고서", na=False)]
        if annual_reports.empty:
            return ""

        # 최신 사업보고서의 접수번호
        rcept_no = annual_reports.iloc[0]["rcept_no"]
        corp_cls = annual_reports.iloc[0].get("corp_cls", "")

        try:
            # 문서 상세 조회 (사업의 내용 API)
            doc_data = self._request("document.json", {
                "rcept_no": rcept_no,
            })

            # DART API document.json은 문서 목록만 반환
            # 실제 텍스트 추출은 document.xml API 사용 필요
            # 여기서는 회사 개요 API로 대체

            company_data = self._fetch_company_overview(stock_code)
            if company_data:
                return company_data

        except Exception as e:
            self.logger.debug(f"문서 상세 조회 실패: {e}")

        # 폴백: 주요 사항 보고서 텍스트 조합
        return self._build_business_summary(stock_code, year)

    def _fetch_company_overview(self, stock_code: str) -> str:
        """
        회사 개요 조회 (사업의 내용 대체)
        """
        try:
            data = self._request("company.json", {
                "corp_code": stock_code,
            })

            if data.get("status") != "000":
                return ""

            # 회사 정보에서 주요 내용 추출
            parts = []

            if data.get("induty_code"):
                parts.append(f"업종코드: {data.get('induty_code')}")

            if data.get("est_dt"):
                parts.append(f"설립일: {data.get('est_dt')}")

            if data.get("phn_no"):
                parts.append(f"대표전화: {data.get('phn_no')}")

            # 회사 기본 정보는 섹터 분류에 도움이 됨
            return " / ".join(parts)

        except Exception as e:
            self.logger.debug(f"회사 개요 조회 실패: {e}")
            return ""

    def _build_business_summary(self, stock_code: str, year: int) -> str:
        """
        재무제표에서 사업 내용 요약 생성
        """
        try:
            fin_data = self.fetch_financial_summary(stock_code, year, ReportType.ANNUAL)

            if not fin_data:
                return ""

            parts = [f"종목명: {fin_data.stock_name}"]

            if fin_data.revenue > 0:
                parts.append(f"매출액: {fin_data.revenue / 100000000:.0f}억원")

            if fin_data.operating_profit != 0:
                parts.append(f"영업이익: {fin_data.operating_profit / 100000000:.0f}억원")

            if fin_data.rd_expense > 0:
                parts.append(f"R&D비용: {fin_data.rd_expense / 100000000:.0f}억원")

            return " / ".join(parts)

        except Exception as e:
            self.logger.debug(f"사업 요약 생성 실패: {e}")
            return ""

    def close(self):
        """세션 종료"""
        self.session.close()
