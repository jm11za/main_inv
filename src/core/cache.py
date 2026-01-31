"""
캐시 서비스

메모리 기반 캐시 (Redis 확장 가능)
"""
import time
from abc import ABC, abstractmethod
from typing import Any


class CacheBackend(ABC):
    """캐시 백엔드 인터페이스"""

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """캐시 조회"""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """캐시 저장"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """캐시 삭제"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """전체 캐시 삭제"""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """키 존재 여부"""
        pass


class MemoryCache(CacheBackend):
    """인메모리 캐시 (TTL 지원)"""

    def __init__(self, default_ttl: int = 3600):
        self._cache: dict[str, tuple[Any, float | None]] = {}
        self._default_ttl = default_ttl

    def get(self, key: str) -> Any | None:
        """캐시 조회 (만료 확인)"""
        if key not in self._cache:
            return None

        value, expires_at = self._cache[key]

        # TTL 체크
        if expires_at is not None and time.time() > expires_at:
            del self._cache[key]
            return None

        return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """캐시 저장"""
        if ttl is None:
            ttl = self._default_ttl

        expires_at = time.time() + ttl if ttl > 0 else None
        self._cache[key] = (value, expires_at)

    def delete(self, key: str) -> bool:
        """캐시 삭제"""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """전체 캐시 삭제"""
        self._cache.clear()

    def exists(self, key: str) -> bool:
        """키 존재 여부 (만료 확인 포함)"""
        return self.get(key) is not None

    def cleanup_expired(self) -> int:
        """만료된 항목 정리"""
        now = time.time()
        expired_keys = [
            key
            for key, (_, expires_at) in self._cache.items()
            if expires_at is not None and now > expires_at
        ]
        for key in expired_keys:
            del self._cache[key]
        return len(expired_keys)


class CacheService:
    """
    캐시 서비스 (Facade)

    사용법:
        cache = CacheService()

        # 저장
        cache.set("stock:005930", {"name": "삼성전자"}, ttl=3600)

        # 조회
        data = cache.get("stock:005930")

        # 캐시 데코레이터
        @cache.cached(ttl=600)
        def get_stock_price(code: str):
            return fetch_price(code)
    """

    _instance: "CacheService | None" = None

    def __new__(cls, *args, **kwargs) -> "CacheService":
        """싱글톤 패턴"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        backend: str = "memory",
        default_ttl: int = 3600,
        **backend_options,
    ):
        if self._initialized:
            return

        self._default_ttl = default_ttl
        self._backend: CacheBackend

        if backend == "memory":
            self._backend = MemoryCache(default_ttl=default_ttl)
        elif backend == "redis":
            # Redis 확장 시 구현
            raise NotImplementedError("Redis 캐시는 아직 구현되지 않았습니다")
        else:
            raise ValueError(f"지원하지 않는 캐시 백엔드: {backend}")

        self._initialized = True

    def get(self, key: str, default: Any = None) -> Any:
        """캐시 조회"""
        value = self._backend.get(key)
        return value if value is not None else default

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """캐시 저장"""
        self._backend.set(key, value, ttl or self._default_ttl)

    def delete(self, key: str) -> bool:
        """캐시 삭제"""
        return self._backend.delete(key)

    def clear(self) -> None:
        """전체 캐시 삭제"""
        self._backend.clear()

    def exists(self, key: str) -> bool:
        """키 존재 여부"""
        return self._backend.exists(key)

    def get_or_set(
        self, key: str, factory: callable, ttl: int | None = None
    ) -> Any:
        """
        캐시 조회, 없으면 factory 호출하여 저장

        Args:
            key: 캐시 키
            factory: 값 생성 함수
            ttl: TTL (초)

        Returns:
            캐시된 값 또는 새로 생성된 값
        """
        value = self.get(key)
        if value is not None:
            return value

        value = factory()
        self.set(key, value, ttl)
        return value

    def cached(self, ttl: int | None = None, key_prefix: str = ""):
        """
        캐시 데코레이터

        사용법:
            @cache.cached(ttl=600, key_prefix="price")
            def get_price(stock_code: str):
                return fetch_price(stock_code)
        """

        def decorator(func):
            def wrapper(*args, **kwargs):
                # 캐시 키 생성
                cache_key = f"{key_prefix}:{func.__name__}:{args}:{kwargs}"
                return self.get_or_set(cache_key, lambda: func(*args, **kwargs), ttl)

            return wrapper

        return decorator

    @classmethod
    def reset(cls) -> None:
        """싱글톤 인스턴스 리셋 (테스트용)"""
        if cls._instance:
            cls._instance.clear()
        cls._instance = None


def get_cache() -> CacheService:
    """CacheService 인스턴스 반환"""
    return CacheService()


def init_cache_from_config() -> CacheService:
    """설정 파일 기반 캐시 초기화"""
    from src.core.config import get_config

    config = get_config()
    cache_config = config.get_section("cache")

    return CacheService(
        backend=cache_config.get("backend", "memory"),
        default_ttl=cache_config.get("ttl", 3600),
    )
