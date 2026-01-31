"""
로깅 서비스

loguru 기반 구조화된 로깅
"""
import sys
from pathlib import Path
from typing import Any

from loguru import logger


class LoggerService:
    """
    로깅 서비스

    사용법:
        from src.core.logger import get_logger

        logger = get_logger(__name__)
        logger.info("작업 시작")
        logger.error("오류 발생", extra={"stock_code": "005930"})
    """

    _configured: bool = False

    @classmethod
    def configure(
        cls,
        level: str = "INFO",
        log_dir: str = "./logs",
        log_format: str | None = None,
        file_enabled: bool = True,
        rotation: str = "10 MB",
        retention: str = "7 days",
    ) -> None:
        """
        로거 설정

        Args:
            level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: 로그 파일 디렉토리
            log_format: 로그 포맷 (None이면 기본값 사용)
            file_enabled: 파일 로깅 활성화 여부
            rotation: 로그 파일 로테이션 크기
            retention: 로그 파일 보관 기간
        """
        if cls._configured:
            return

        # 기존 핸들러 제거
        logger.remove()

        # 기본 포맷
        if log_format is None:
            log_format = (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            )

        # 콘솔 핸들러
        logger.add(
            sys.stderr,
            format=log_format,
            level=level,
            colorize=True,
        )

        # 파일 핸들러
        if file_enabled:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)

            # 일반 로그
            logger.add(
                log_path / "app.log",
                format=log_format,
                level=level,
                rotation=rotation,
                retention=retention,
                compression="zip",
                encoding="utf-8",
            )

            # 에러 전용 로그
            logger.add(
                log_path / "error.log",
                format=log_format,
                level="ERROR",
                rotation=rotation,
                retention=retention,
                compression="zip",
                encoding="utf-8",
            )

        cls._configured = True

    @classmethod
    def reset(cls) -> None:
        """설정 리셋 (테스트용)"""
        logger.remove()
        cls._configured = False


def get_logger(name: str) -> Any:
    """
    모듈별 로거 반환

    Args:
        name: 모듈 이름 (보통 __name__ 사용)

    Returns:
        loguru logger 인스턴스
    """
    return logger.bind(name=name)


def setup_logger_from_config() -> None:
    """설정 파일 기반 로거 초기화"""
    try:
        from src.core.config import get_config

        config = get_config()
        logging_config = config.get_section("logging")

        LoggerService.configure(
            level=logging_config.get("level", "INFO"),
            log_dir=logging_config.get("log_dir", "./logs"),
            log_format=logging_config.get("format"),
            file_enabled=logging_config.get("file", {}).get("enabled", True),
            rotation=logging_config.get("file", {}).get("rotation", "10 MB"),
            retention=logging_config.get("file", {}).get("retention", "7 days"),
        )
    except Exception:
        # 설정 로드 실패 시 기본 설정 사용
        LoggerService.configure()
