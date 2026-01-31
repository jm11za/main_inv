"""
설정 관리 모듈

YAML 기반 설정 파일 로드 및 환경별 설정 분리
"""
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.core.exceptions import ConfigError, ConfigNotFoundError, ConfigValidationError


class Config:
    """
    설정 관리자

    사용법:
        config = Config()  # 기본: development 환경
        config = Config(env="production")

        # 설정 값 접근
        db_url = config.get("database.url")
        api_key = config.get("api.dart_key", default="")
    """

    _instance: "Config | None" = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs) -> "Config":
        """싱글톤 패턴"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, env: str | None = None, config_dir: Path | None = None):
        if Config._initialized:
            return

        # .env 파일 로드
        load_dotenv()

        # 환경 결정: 인자 > 환경변수 > 기본값
        self.env = env or os.getenv("APP_ENV", "development")

        # 설정 디렉토리
        if config_dir:
            self.config_dir = config_dir
        else:
            # 프로젝트 루트/config 디렉토리
            self.config_dir = Path(__file__).parent.parent.parent / "config"

        self._config: dict[str, Any] = {}
        self._load_config()

        Config._initialized = True

    def _load_config(self) -> None:
        """설정 파일 로드 (기본 + 환경별)"""
        # 1. 기본 설정 로드
        base_config_path = self.config_dir / "settings.yaml"
        if base_config_path.exists():
            self._config = self._load_yaml(base_config_path)
        else:
            raise ConfigNotFoundError(
                f"기본 설정 파일을 찾을 수 없습니다: {base_config_path}"
            )

        # 2. 환경별 설정 오버라이드
        env_config_path = self.config_dir / f"settings.{self.env}.yaml"
        if env_config_path.exists():
            env_config = self._load_yaml(env_config_path)
            self._deep_merge(self._config, env_config)

        # 3. 환경 변수로 오버라이드 (선택적)
        self._apply_env_overrides()

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        """YAML 파일 로드"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"YAML 파싱 오류: {path}", {"error": str(e)})

    def _deep_merge(self, base: dict, override: dict) -> None:
        """딕셔너리 깊은 병합 (override가 base를 덮어씀)"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _apply_env_overrides(self) -> None:
        """환경 변수로 설정 오버라이드 (STOCK_ 접두사)"""
        for key, value in os.environ.items():
            if key.startswith("STOCK_"):
                # STOCK_DATABASE_URL -> database.url
                config_key = key[6:].lower().replace("_", ".")
                self._set_nested(config_key, value)

    def _set_nested(self, key: str, value: Any) -> None:
        """점(.) 표기법으로 중첩 설정 값 설정"""
        keys = key.split(".")
        current = self._config

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        설정 값 조회 (점 표기법 지원)

        Args:
            key: 설정 키 (예: "database.url", "api.dart_key")
            default: 기본값

        Returns:
            설정 값 또는 기본값
        """
        keys = key.split(".")
        current = self._config

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default

        return current

    def get_required(self, key: str) -> Any:
        """
        필수 설정 값 조회 (없으면 예외 발생)

        Args:
            key: 설정 키

        Returns:
            설정 값

        Raises:
            ConfigValidationError: 설정 값이 없는 경우
        """
        value = self.get(key)
        if value is None:
            raise ConfigValidationError(f"필수 설정 값이 없습니다: {key}")
        return value

    def get_section(self, section: str) -> dict[str, Any]:
        """섹션 전체 조회"""
        return self.get(section, {})

    @property
    def is_production(self) -> bool:
        """운영 환경 여부"""
        return self.env == "production"

    @property
    def is_development(self) -> bool:
        """개발 환경 여부"""
        return self.env == "development"

    @classmethod
    def reset(cls) -> None:
        """싱글톤 인스턴스 리셋 (테스트용)"""
        cls._instance = None
        cls._initialized = False


def get_config() -> Config:
    """Config 인스턴스 반환 (편의 함수)"""
    return Config()
