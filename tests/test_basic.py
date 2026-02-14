"""
기초 테스트 - 프로젝트 초기화 검증
"""
import pytest


class TestProjectSetup:
    """프로젝트 초기 설정 검증 테스트"""

    def test_python_version(self):
        """Python 버전이 3.10 이상인지 확인"""
        import sys
        assert sys.version_info >= (3, 10), "Python 3.10 이상 필요"

    def test_pandas_import(self):
        """pandas 라이브러리 임포트 확인"""
        import pandas as pd
        assert pd is not None

    def test_numpy_import(self):
        """numpy 라이브러리 임포트 확인"""
        import numpy as np
        assert np is not None

    def test_requests_import(self):
        """requests 라이브러리 임포트 확인"""
        import requests
        assert requests is not None

    def test_src_package_import(self):
        """src 패키지 구조 확인"""
        import src
        import src.core
        import src.ingest
        import src.processing
        import src.analysis
        import src.filtering
        import src.sector
        import src.stock
        import src.verify
        import src.output
        import src.llm
        assert src is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
