"""
데이터베이스 관리 모듈

SQLAlchemy 기반 커넥션 풀 및 세션 관리
"""
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker, declarative_base

from src.core.exceptions import DatabaseError

# ORM Base 클래스
Base = declarative_base()


class DatabaseManager:
    """
    데이터베이스 연결 관리자

    사용법:
        db = DatabaseManager("sqlite:///data/stock.db")

        # 세션 사용
        with db.session() as session:
            result = session.execute(text("SELECT 1"))

        # 또는 get_session 사용
        session = db.get_session()
        try:
            session.execute(text("SELECT 1"))
            session.commit()
        finally:
            session.close()
    """

    _instance: "DatabaseManager | None" = None

    def __new__(cls, *args, **kwargs) -> "DatabaseManager":
        """싱글톤 패턴"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        connection_string: str | None = None,
        pool_size: int = 5,
        pool_timeout: int = 30,
        echo: bool = False,
    ):
        if self._initialized:
            return

        self._connection_string = connection_string
        self._pool_size = pool_size
        self._pool_timeout = pool_timeout
        self._echo = echo

        self._engine: Engine | None = None
        self._session_factory: sessionmaker | None = None
        self._initialized = True

    def _ensure_engine(self) -> Engine:
        """엔진 생성 (Lazy initialization)"""
        if self._engine is None:
            if self._connection_string is None:
                raise DatabaseError("데이터베이스 연결 문자열이 설정되지 않았습니다")

            # SQLite인 경우 디렉토리 생성
            if self._connection_string.startswith("sqlite:///"):
                db_path = Path(self._connection_string.replace("sqlite:///", ""))
                db_path.parent.mkdir(parents=True, exist_ok=True)

            # 엔진 생성
            if self._connection_string.startswith("sqlite"):
                # SQLite는 pool_size 지원 안함
                self._engine = create_engine(
                    self._connection_string,
                    echo=self._echo,
                    connect_args={"check_same_thread": False},
                )
            else:
                self._engine = create_engine(
                    self._connection_string,
                    pool_size=self._pool_size,
                    pool_timeout=self._pool_timeout,
                    echo=self._echo,
                )

            self._session_factory = sessionmaker(bind=self._engine)

        return self._engine

    @property
    def engine(self) -> Engine:
        """SQLAlchemy 엔진 반환"""
        return self._ensure_engine()

    def get_session(self) -> Session:
        """새 세션 반환 (수동 관리)"""
        self._ensure_engine()
        if self._session_factory is None:
            raise DatabaseError("세션 팩토리가 초기화되지 않았습니다")
        return self._session_factory()

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """
        세션 컨텍스트 매니저

        자동 커밋/롤백 처리

        사용법:
            with db.session() as session:
                session.execute(text("INSERT INTO ..."))
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise DatabaseError(f"데이터베이스 작업 실패: {e}")
        finally:
            session.close()

    def create_all_tables(self) -> None:
        """모든 테이블 생성 (ORM 모델 기반)"""
        Base.metadata.create_all(self.engine)

    def drop_all_tables(self) -> None:
        """모든 테이블 삭제 (주의!)"""
        Base.metadata.drop_all(self.engine)

    def health_check(self) -> bool:
        """연결 상태 확인"""
        try:
            with self.session() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def close(self) -> None:
        """연결 종료"""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None

    @classmethod
    def reset(cls) -> None:
        """싱글톤 인스턴스 리셋 (테스트용)"""
        if cls._instance:
            cls._instance.close()
        cls._instance = None


def get_database() -> DatabaseManager:
    """DatabaseManager 인스턴스 반환"""
    return DatabaseManager()


def init_database_from_config() -> DatabaseManager:
    """설정 파일 기반 데이터베이스 초기화"""
    from src.core.config import get_config

    config = get_config()
    db_config = config.get_section("database")

    db = DatabaseManager(
        connection_string=db_config.get("connection_string"),
        pool_size=db_config.get("pool_size", 5),
        echo=db_config.get("echo", False),
    )

    return db
