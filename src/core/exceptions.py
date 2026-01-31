"""
커스텀 예외 클래스 정의

모든 레이어에서 사용하는 표준화된 예외 처리
"""
from typing import Any


class BaseError(Exception):
    """모든 커스텀 예외의 기본 클래스"""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# ============================================
# Configuration Errors
# ============================================
class ConfigError(BaseError):
    """설정 관련 오류"""
    pass


class ConfigNotFoundError(ConfigError):
    """설정 파일을 찾을 수 없음"""
    pass


class ConfigValidationError(ConfigError):
    """설정 값 유효성 검증 실패"""
    pass


# ============================================
# Data Ingest Errors (Layer 1)
# ============================================
class IngestError(BaseError):
    """데이터 수집 관련 오류"""
    pass


class CrawlerError(IngestError):
    """크롤링 실패"""
    pass


class APIError(IngestError):
    """외부 API 호출 실패"""
    pass


class RateLimitError(APIError):
    """API Rate Limit 초과"""

    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message, {"retry_after": retry_after})
        self.retry_after = retry_after


# ============================================
# Processing Errors (Layer 2)
# ============================================
class ProcessingError(BaseError):
    """데이터 처리 관련 오류"""
    pass


class DataValidationError(ProcessingError):
    """데이터 유효성 검증 실패"""
    pass


# ============================================
# Analysis Errors (Layer 3)
# ============================================
class AnalysisError(BaseError):
    """분석 관련 오류"""
    pass


class MetricCalculationError(AnalysisError):
    """지표 계산 실패"""
    pass


# ============================================
# Filtering Errors (Layer 3.5)
# ============================================
class FilteringError(BaseError):
    """필터링 관련 오류"""
    pass


class SectorClassificationError(FilteringError):
    """섹터 분류 실패"""
    pass


# ============================================
# LLM Errors
# ============================================
class LLMError(BaseError):
    """LLM 관련 오류"""
    pass


class LLMConnectionError(LLMError):
    """LLM 연결 실패"""
    pass


class LLMResponseError(LLMError):
    """LLM 응답 파싱 실패"""
    pass


# ============================================
# Database Errors
# ============================================
class DatabaseError(BaseError):
    """데이터베이스 관련 오류"""
    pass


class ConnectionError(DatabaseError):
    """DB 연결 실패"""
    pass


class QueryError(DatabaseError):
    """쿼리 실행 실패"""
    pass


# ============================================
# Cache Errors
# ============================================
class CacheError(BaseError):
    """캐시 관련 오류"""
    pass


class CacheMissError(CacheError):
    """캐시 미스"""
    pass
