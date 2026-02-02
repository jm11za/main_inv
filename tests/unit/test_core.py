"""
Core 모듈 단위 테스트
"""
import time
from pathlib import Path

import pytest


class TestConfig:
    """Config 모듈 테스트"""

    def setup_method(self):
        """각 테스트 전 Config 리셋"""
        from src.core.config import Config
        Config.reset()

    def teardown_method(self):
        """각 테스트 후 Config 리셋"""
        from src.core.config import Config
        Config.reset()

    def test_config_load_success(self):
        """설정 파일 로드 성공"""
        from src.core.config import Config

        config = Config()
        assert config.get("app.name") is not None

    def test_config_singleton(self):
        """싱글톤 패턴 확인"""
        from src.core.config import Config

        config1 = Config()
        config2 = Config()
        assert config1 is config2

    def test_config_get_nested(self):
        """중첩 설정 조회"""
        from src.core.config import Config

        config = Config()
        # database.connection_string 존재 확인
        db_conn = config.get("database.connection_string")
        assert db_conn is not None

    def test_config_get_default(self):
        """존재하지 않는 키 기본값 반환"""
        from src.core.config import Config

        config = Config()
        value = config.get("non.existent.key", default="default_value")
        assert value == "default_value"

    def test_config_get_section(self):
        """섹션 전체 조회"""
        from src.core.config import Config

        config = Config()
        db_section = config.get_section("database")
        assert isinstance(db_section, dict)
        assert "connection_string" in db_section


class TestLogger:
    """Logger 모듈 테스트"""

    def setup_method(self):
        """각 테스트 전 Logger 리셋"""
        from src.core.logger import LoggerService
        LoggerService.reset()

    def teardown_method(self):
        """각 테스트 후 Logger 리셋"""
        from src.core.logger import LoggerService
        LoggerService.reset()

    def test_logger_configure(self):
        """로거 설정 성공"""
        from src.core.logger import LoggerService

        LoggerService.configure(level="DEBUG", file_enabled=False)
        assert LoggerService._configured is True

    def test_get_logger(self):
        """모듈별 로거 획득"""
        from src.core.logger import get_logger, LoggerService

        LoggerService.configure(level="DEBUG", file_enabled=False)
        logger = get_logger(__name__)
        assert logger is not None

    def test_logger_log_message(self, capsys):
        """로그 메시지 출력"""
        from src.core.logger import get_logger, LoggerService

        LoggerService.configure(level="DEBUG", file_enabled=False)
        logger = get_logger("test")
        logger.info("테스트 메시지")
        # capsys로 직접 확인하기 어려움 (loguru는 stderr 사용)
        # 예외 없이 실행되면 성공


class TestCache:
    """Cache 모듈 테스트"""

    def setup_method(self):
        """각 테스트 전 Cache 리셋"""
        from src.core.cache import CacheService
        CacheService.reset()

    def teardown_method(self):
        """각 테스트 후 Cache 리셋"""
        from src.core.cache import CacheService
        CacheService.reset()

    def test_cache_set_get(self):
        """캐시 저장 및 조회"""
        from src.core.cache import CacheService

        cache = CacheService()
        cache.set("test_key", {"data": "value"})
        result = cache.get("test_key")
        assert result == {"data": "value"}

    def test_cache_ttl_expiration(self):
        """TTL 만료 테스트"""
        from src.core.cache import CacheService

        cache = CacheService(default_ttl=1)
        cache.set("expire_key", "value", ttl=1)

        # 즉시 조회 - 값 존재
        assert cache.get("expire_key") == "value"

        # 2초 대기 후 조회 - 만료
        time.sleep(2)
        assert cache.get("expire_key") is None

    def test_cache_delete(self):
        """캐시 삭제"""
        from src.core.cache import CacheService

        cache = CacheService()
        cache.set("delete_key", "value")
        assert cache.exists("delete_key") is True

        cache.delete("delete_key")
        assert cache.exists("delete_key") is False

    def test_cache_get_or_set(self):
        """get_or_set 테스트"""
        from src.core.cache import CacheService

        cache = CacheService()
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return "generated_value"

        # 첫 호출 - factory 실행
        result1 = cache.get_or_set("gen_key", factory)
        assert result1 == "generated_value"
        assert call_count == 1

        # 두 번째 호출 - 캐시에서 반환 (factory 미실행)
        result2 = cache.get_or_set("gen_key", factory)
        assert result2 == "generated_value"
        assert call_count == 1  # 여전히 1


class TestDatabase:
    """Database 모듈 테스트"""

    def setup_method(self):
        """각 테스트 전 Database 리셋"""
        from src.core.database import DatabaseManager
        DatabaseManager.reset()

    def teardown_method(self):
        """각 테스트 후 Database 리셋"""
        from src.core.database import DatabaseManager
        DatabaseManager.reset()

    def test_database_connection(self, tmp_path):
        """데이터베이스 연결 테스트"""
        from src.core.database import DatabaseManager

        db_path = tmp_path / "test.db"
        db = DatabaseManager(connection_string=f"sqlite:///{db_path}")

        assert db.health_check() is True

    def test_database_session_context(self, tmp_path):
        """세션 컨텍스트 매니저 테스트"""
        from sqlalchemy import text
        from src.core.database import DatabaseManager

        db_path = tmp_path / "test.db"
        db = DatabaseManager(connection_string=f"sqlite:///{db_path}")

        with db.session() as session:
            result = session.execute(text("SELECT 1 as num"))
            row = result.fetchone()
            assert row[0] == 1


class TestExceptions:
    """Exception 모듈 테스트"""

    def test_base_error(self):
        """BaseError 테스트"""
        from src.core.exceptions import BaseError

        error = BaseError("테스트 오류", {"key": "value"})
        assert error.message == "테스트 오류"
        assert error.details == {"key": "value"}
        assert "Details:" in str(error)

    def test_rate_limit_error(self):
        """RateLimitError 테스트"""
        from src.core.exceptions import RateLimitError

        error = RateLimitError("Rate limit exceeded", retry_after=60)
        assert error.retry_after == 60
        assert error.details["retry_after"] == 60


class TestInterfaces:
    """Interfaces 모듈 테스트"""

    def test_enums(self):
        """Enum 정의 확인"""
        from src.core.interfaces import (
            SectorType, TrackType, Tier, Recommendation
        )

        assert SectorType.TYPE_A.value == "earnings_driven"
        assert TrackType.TRACK_B.value == "growth_driven"
        assert Tier.TIER_1.value == 1
        assert Recommendation.STRONG_BUY.value == "STRONG_BUY"

    def test_stock_dataclass(self):
        """Stock 데이터클래스 테스트"""
        from src.core.interfaces import Stock, TrackType

        stock = Stock(
            stock_code="005930",
            name="삼성전자",
            track_type=TrackType.TRACK_A,
        )

        assert stock.stock_code == "005930"
        assert stock.name == "삼성전자"
        assert stock.filter_passed is False  # 기본값

    def test_filter_result_dataclass(self):
        """FilterResult 데이터클래스 테스트"""
        from src.core.interfaces import FilterResult

        result = FilterResult(
            passed=True,
            stock_code="005930",
            reason="모든 조건 충족",
            metrics={"debt_ratio": 50.0},
        )

        assert result.passed is True
        assert result.metrics["debt_ratio"] == 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
