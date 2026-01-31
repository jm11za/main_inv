"""
데이터베이스 모델 정의

SQLAlchemy ORM 모델
"""
from datetime import datetime

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime,
    Text, ForeignKey, Table, Index
)
from sqlalchemy.orm import relationship

from src.core.database import Base


# ============================================
# 다대다 관계 테이블
# ============================================
theme_stock_association = Table(
    "theme_stock",
    Base.metadata,
    Column("theme_id", String(20), ForeignKey("themes.theme_id"), primary_key=True),
    Column("stock_code", String(10), ForeignKey("stocks.stock_code"), primary_key=True),
    Column("created_at", DateTime, default=datetime.now),
)


# ============================================
# 테마 모델
# ============================================
class ThemeModel(Base):
    """테마/섹터 테이블"""
    __tablename__ = "themes"

    theme_id = Column(String(20), primary_key=True)  # 네이버 테마 ID
    name = Column(String(100), nullable=False)
    theme_type = Column(String(20), default="")  # 실체형/기대형
    sector_type = Column(String(20), default="TYPE_A")  # TYPE_A/TYPE_B
    description = Column(Text, default="")

    # Metrics (Layer 3에서 계산)
    s_flow = Column(Float, default=0.0)
    s_breadth = Column(Float, default=0.0)
    s_trend = Column(Float, default=0.0)
    tier = Column(Integer, default=3)  # 1, 2, 3

    # 메타데이터
    stock_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # 관계
    stocks = relationship(
        "StockModel",
        secondary=theme_stock_association,
        back_populates="themes"
    )

    def __repr__(self):
        return f"<Theme {self.theme_id}: {self.name}>"


# ============================================
# 종목 모델
# ============================================
class StockModel(Base):
    """종목 테이블"""
    __tablename__ = "stocks"

    stock_code = Column(String(10), primary_key=True)
    name = Column(String(100), nullable=False)
    market = Column(String(10), default="")  # KOSPI/KOSDAQ

    # 섹터 분류 (Layer 2.5 SectorLabeler에서 결정)
    primary_sector = Column(String(50), default="")  # 메인 섹터 (반도체, 2차전지 등)
    secondary_sectors = Column(String(200), default="")  # 보조 섹터 (쉼표 구분)
    sector_confidence = Column(Float, default=0.0)  # 섹터 분류 신뢰도

    # 트랙 분류 (Layer 3.5에서 결정)
    track_type = Column(String(20), default="TRACK_A")

    # 재무 데이터
    operating_profit_4q = Column(Float, default=0.0)
    debt_ratio = Column(Float, default=0.0)
    pbr = Column(Float, default=0.0)
    per = Column(Float, default=0.0)
    current_ratio = Column(Float, default=0.0)
    capital_impairment = Column(Float, default=0.0)
    rd_ratio = Column(Float, default=0.0)
    market_cap = Column(Float, default=0.0)
    avg_trading_value = Column(Float, default=0.0)  # 평균 거래대금

    # 기술적 지표 (개별 종목)
    s_flow = Column(Float, default=0.0)  # 개별 종목 수급 강도

    # 필터 결과
    filter_passed = Column(Boolean, default=False)
    filter_reason = Column(String(200), default="")

    # 점수 (Layer 4에서 계산)
    financial_score = Column(Float, default=0.0)
    technical_score = Column(Float, default=0.0)
    total_score = Column(Float, default=0.0)
    rank = Column(Integer, default=0)

    # 최종 판정 (Layer 5에서 결정)
    material_grade = Column(String(5), default="")  # S/A/B/C
    sentiment_stage = Column(String(20), default="")  # 공포/의심/확신/환희
    recommendation = Column(String(20), default="")  # STRONG_BUY/BUY/WATCH/AVOID

    # 메타데이터
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # 관계
    themes = relationship(
        "ThemeModel",
        secondary=theme_stock_association,
        back_populates="stocks"
    )

    # 인덱스
    __table_args__ = (
        Index("idx_stock_recommendation", "recommendation"),
        Index("idx_stock_total_score", "total_score"),
    )

    def __repr__(self):
        return f"<Stock {self.stock_code}: {self.name}>"


# ============================================
# 일별 가격 데이터 (캐시용)
# ============================================
class DailyPriceModel(Base):
    """일별 가격 데이터"""
    __tablename__ = "daily_prices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_code = Column(String(10), ForeignKey("stocks.stock_code"), nullable=False)
    date = Column(DateTime, nullable=False)

    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    trading_value = Column(Float)  # 거래대금
    change_rate = Column(Float)  # 등락률

    # 수급 데이터
    foreign_net = Column(Float, default=0.0)  # 외국인 순매수
    institution_net = Column(Float, default=0.0)  # 기관 순매수

    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("idx_price_stock_date", "stock_code", "date", unique=True),
    )

    def __repr__(self):
        return f"<DailyPrice {self.stock_code} @ {self.date}>"


# ============================================
# 크롤링 히스토리
# ============================================
class CrawlHistoryModel(Base):
    """크롤링 실행 기록"""
    __tablename__ = "crawl_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(50), nullable=False)  # naver_theme, dart, news
    status = Column(String(20), default="success")  # success/failed
    record_count = Column(Integer, default=0)
    error_message = Column(Text, default="")
    started_at = Column(DateTime, default=datetime.now)
    completed_at = Column(DateTime)

    def __repr__(self):
        return f"<CrawlHistory {self.source} @ {self.started_at}>"


# ============================================
# 분석 히스토리
# ============================================
class AnalysisHistoryModel(Base):
    """파이프라인 분석 실행 히스토리"""
    __tablename__ = "analysis_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_date = Column(DateTime, nullable=False, default=datetime.now)

    # 분석 대상
    total_stocks = Column(Integer, default=0)  # 분석 대상 종목 수
    passed_filter = Column(Integer, default=0)  # 필터 통과 종목 수

    # 섹터 분석 결과
    tier1_sectors = Column(String(500), default="")  # Tier 1 섹터 (쉼표 구분)
    tier2_sectors = Column(String(500), default="")  # Tier 2 섹터 (쉼표 구분)

    # 추천 결과 (JSON)
    top_buy = Column(Text, default="")  # BUY 추천 종목 JSON
    top_watch = Column(Text, default="")  # WATCH 추천 종목 JSON

    # 실행 정보
    duration_seconds = Column(Float, default=0.0)
    status = Column(String(20), default="success")  # success/failed
    error_message = Column(Text, default="")

    # 메타
    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("idx_analysis_run_date", "run_date"),
    )

    def __repr__(self):
        return f"<AnalysisHistory {self.run_date} - {self.status}>"


class StockScoreHistoryModel(Base):
    """종목 점수 히스토리 (시계열)"""
    __tablename__ = "stock_score_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_code = Column(String(10), ForeignKey("stocks.stock_code"), nullable=False)
    analysis_date = Column(DateTime, nullable=False, default=datetime.now)

    # 점수
    total_score = Column(Float, default=0.0)
    financial_score = Column(Float, default=0.0)
    technical_score = Column(Float, default=0.0)

    # 기술적 지표
    s_flow = Column(Float, default=0.0)
    s_trend = Column(Float, default=0.0)

    # 판정 결과
    material_grade = Column(String(5), default="")  # S/A/B/C
    sentiment_stage = Column(String(20), default="")  # 공포/의심/확신/환희
    recommendation = Column(String(20), default="")  # BUY/WATCH/HOLD/AVOID

    # 섹터/트랙 정보
    primary_sector = Column(String(50), default="")
    track_type = Column(String(20), default="")

    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("idx_score_stock_date", "stock_code", "analysis_date"),
        Index("idx_score_date", "analysis_date"),
    )

    def __repr__(self):
        return f"<StockScoreHistory {self.stock_code} @ {self.analysis_date}>"
