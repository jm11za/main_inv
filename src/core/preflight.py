"""
Preflight Check 모듈

파이프라인/테스트 실행 전 필수 환경 확인
- Ollama LLM 연결 및 모델 로드
- 네이버 금융 접근
- pykrx 모듈
- 필수 모듈 임포트
"""
from dataclasses import dataclass, field
from typing import Any

from src.core.logger import get_logger


@dataclass
class PreflightResult:
    """Preflight 검사 결과"""
    passed: bool
    ollama: dict = field(default_factory=dict)
    claude_cli: dict = field(default_factory=dict)
    naver: dict = field(default_factory=dict)
    dart_api: dict = field(default_factory=dict)
    telegram: dict = field(default_factory=dict)
    pykrx: dict = field(default_factory=dict)
    modules: dict = field(default_factory=dict)

    # 편의 property
    @property
    def ollama_ok(self) -> bool:
        return self.ollama.get("available", False)

    @property
    def claude_ok(self) -> bool:
        return self.claude_cli.get("available", False)

    @property
    def naver_ok(self) -> bool:
        return self.naver.get("available", False)

    @property
    def dart_ok(self) -> bool:
        return self.dart_api.get("available", False)

    @property
    def telegram_ok(self) -> bool:
        return self.telegram.get("available", False)

    @property
    def pykrx_ok(self) -> bool:
        return self.pykrx.get("available", False)

    @property
    def modules_ok(self) -> bool:
        return self.modules.get("available", False)

    def get_failures(self) -> list[str]:
        """실패한 항목 목록 반환 (모든 항목 필수)"""
        failures = []
        if not self.ollama.get("available"):
            failures.append(f"Ollama: {self.ollama.get('error', 'Unknown')}")
        if not self.claude_cli.get("available"):
            failures.append(f"Claude CLI: {self.claude_cli.get('error', 'Unknown')}")
        if not self.naver.get("available"):
            failures.append(f"Naver: {self.naver.get('error', 'Unknown')}")
        if not self.dart_api.get("available"):
            failures.append(f"DART API: {self.dart_api.get('error', 'Unknown')}")
        if not self.telegram.get("available"):
            failures.append(f"Telegram: {self.telegram.get('error', 'Unknown')}")
        if not self.pykrx.get("available"):
            failures.append(f"pykrx: {self.pykrx.get('error', 'Unknown')}")
        if not self.modules.get("available"):
            failures.append(f"Modules: {self.modules.get('error', 'Unknown')}")
        return failures

    def summary(self) -> str:
        """결과 요약 문자열"""
        lines = []
        lines.append(f"Preflight: {'PASSED' if self.passed else 'FAILED'}")
        lines.append(f"  Ollama: {'OK' if self.ollama.get('available') else 'FAIL'}")
        if self.ollama.get("model"):
            lines.append(f"    Model: {self.ollama.get('model')}")
        lines.append(f"  Claude CLI: {'OK' if self.claude_cli.get('available') else 'FAIL'}")
        lines.append(f"  Naver: {'OK' if self.naver.get('available') else 'FAIL'}")
        lines.append(f"  DART API: {'OK' if self.dart_api.get('available') else 'FAIL'}")
        lines.append(f"  Telegram: {'OK' if self.telegram.get('available') else 'FAIL'}")
        lines.append(f"  pykrx: {'OK' if self.pykrx.get('available') else 'FAIL'}")
        lines.append(f"  Modules: {'OK' if self.modules.get('available') else 'FAIL'}")
        return "\n".join(lines)


class PreflightChecker:
    """
    Preflight 검사기

    파이프라인/테스트 실행 전 환경 검증

    사용법:
        checker = PreflightChecker()

        # 사용 가능한 모델 목록 확인
        models = checker.get_available_models()
        print(models)  # ['gemma3:4b', 'deepseek-r1:14b', ...]

        # 모델 선택하여 검사
        result = checker.run(model="gemma3:4b")

        if not result.passed:
            print("Preflight 실패:", result.get_failures())
            return
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self._ollama_client = None

    # 선호 모델 목록 (우선순위 순)
    PREFERRED_MODELS = [
        "qwen2.5:7b",       # 7B - 요약 품질 개선
        "llama3.1:8b",      # 8B
        "gemma3:4b",        # 4B - 가벼움
        "deepseek-r1:8b",   # 8B - 추론 모델 (느림)
        "gemma2:9b",        # 9B
        "deepseek-r1:14b",  # 14B - 가장 무거움
    ]

    # 제외할 모델 (절대 사용 안함)
    EXCLUDED_MODELS = ["glm", "chatglm"]

    def get_available_models(self) -> list[str]:
        """
        Ollama에서 사용 가능한 모델 목록 반환

        Returns:
            모델명 리스트 (빈 리스트면 Ollama 연결 실패)
        """
        try:
            import requests
            response = requests.get(
                "http://127.0.0.1:11434/api/tags",
                timeout=(3, 3)
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m.get("name", "") for m in models]
        except Exception:
            pass
        return []

    def select_best_model(self, available_models: list[str] | None = None) -> str:
        """
        최적 모델 자동 선택 (deepseek 우선, GLM 제외)

        Args:
            available_models: 사용 가능한 모델 목록 (None이면 자동 조회)

        Returns:
            선택된 모델명
        """
        if available_models is None:
            available_models = self.get_available_models()

        # 1. 선호 모델 중 사용 가능한 것 찾기
        for preferred in self.PREFERRED_MODELS:
            for m in available_models:
                if preferred in m:
                    self.logger.info(f"선호 모델 선택: {m}")
                    return m

        # 2. 선호 모델 없으면 GLM 제외한 첫 번째 모델
        for m in available_models:
            if not any(exc in m.lower() for exc in self.EXCLUDED_MODELS):
                self.logger.info(f"대체 모델 선택: {m}")
                return m

        # 3. 기본값 (PREFERRED_MODELS 첫 번째)
        default_model = self.PREFERRED_MODELS[0]
        self.logger.warning(f"사용 가능한 모델 없음, 기본값 사용: {default_model}")
        return default_model

    def run(
        self,
        model: str | None = None,
        warmup_timeout: float = 300.0,
    ) -> PreflightResult:
        """
        Preflight 검사 실행 (모든 항목 필수)

        Args:
            model: 사용할 Ollama 모델명 (None이면 자동 선택: deepseek 우선, GLM 제외)
            warmup_timeout: 모델 Warm-up 타임아웃 (초)

        Returns:
            PreflightResult (하나라도 실패 시 passed=False)
        """
        # 모델 자동 선택 (deepseek 우선, GLM 제외)
        if model is None:
            model = self.select_best_model()

        self.logger.info("=" * 50)
        self.logger.info("Preflight Check 시작 (모든 항목 필수)")
        self.logger.info(f"사용 모델: {model}")
        self.logger.info("=" * 50)

        result = PreflightResult(
            passed=True,
            ollama={"available": False, "error": None, "model": None},
            claude_cli={"available": False, "error": None, "version": None},
            naver={"available": False, "error": None},
            dart_api={"available": False, "error": None},
            telegram={"available": False, "error": None, "bot_name": None},
            pykrx={"available": False, "error": None},
            modules={"available": False, "error": None},
        )

        # 1. Ollama 검사 (필수)
        self._check_ollama(result, model, warmup_timeout)
        if not result.ollama["available"]:
            result.passed = False

        # 2. Claude CLI 검사 (필수)
        self._check_claude_cli(result)
        if not result.claude_cli["available"]:
            result.passed = False

        # 3. 네이버 금융 검사 (필수)
        self._check_naver(result)
        if not result.naver["available"]:
            result.passed = False

        # 4. DART API 검사 (필수)
        self._check_dart_api(result)
        if not result.dart_api["available"]:
            result.passed = False

        # 5. Telegram 검사 (필수)
        self._check_telegram(result)
        if not result.telegram["available"]:
            result.passed = False

        # 6. pykrx 검사 (필수)
        self._check_pykrx(result)
        if not result.pykrx["available"]:
            result.passed = False

        # 7. 필수 모듈 검사 (필수)
        self._check_modules(result)
        if not result.modules["available"]:
            result.passed = False

        # 결과 요약
        self.logger.info("=" * 50)
        if result.passed:
            self.logger.info("Preflight Check 완료 - 모든 검사 통과")
        else:
            self.logger.warning("Preflight Check 실패")
            for failure in result.get_failures():
                self.logger.warning(f"  - {failure}")
        self.logger.info("=" * 50)

        return result

    # 빠른 응답 테스트 타임아웃 (초)
    QUICK_TEST_TIMEOUT = 20.0

    def _check_ollama(
        self,
        result: PreflightResult,
        model: str | None,
        warmup_timeout: float
    ):
        """Ollama 연결 및 모델 Warm-up (응답 느리면 하위 모델로 자동 전환)"""
        self.logger.info("[1/7] Ollama 연결 및 모델 Warm-up...")

        try:
            from src.llm import OllamaClient
            import requests

            # 서버 연결 확인
            try:
                response = requests.get("http://127.0.0.1:11434/api/tags", timeout=(3, 3))
                if response.status_code != 200:
                    result.ollama["error"] = "서버 연결 실패"
                    self.logger.warning("  ✗ Ollama 서버 연결 실패")
                    return
            except Exception as e:
                result.ollama["error"] = f"서버 연결 실패: {e}"
                self.logger.warning(f"  ✗ Ollama 서버 연결 실패: {e}")
                return

            # 사용 가능한 모델 목록
            available_models = self.get_available_models()
            if not available_models:
                result.ollama["error"] = "사용 가능한 모델 없음"
                self.logger.warning("  ✗ 사용 가능한 모델 없음")
                return

            # 시도할 모델 목록 구성
            models_to_try = []
            if model:
                models_to_try.append(model)

            # 선호 모델 중 사용 가능한 것 추가
            for preferred in self.PREFERRED_MODELS:
                for m in available_models:
                    if preferred in m and m not in models_to_try:
                        models_to_try.append(m)

            # GLM 제외한 나머지 모델 추가
            for m in available_models:
                if m not in models_to_try:
                    if not any(exc in m.lower() for exc in self.EXCLUDED_MODELS):
                        models_to_try.append(m)

            self.logger.info(f"  - 시도할 모델: {models_to_try[:5]}...")

            # 모델별로 시도
            for i, try_model in enumerate(models_to_try):
                self.logger.info(f"  - [{i+1}] {try_model} 테스트 중 (최대 {self.QUICK_TEST_TIMEOUT}초)...")

                try:
                    client = OllamaClient(model=try_model)

                    # 빠른 응답 테스트 (20초 내 응답 필요)
                    if client.warmup(timeout=self.QUICK_TEST_TIMEOUT):
                        # 성공
                        self._ollama_client = client
                        result.ollama["available"] = True
                        result.ollama["model"] = try_model
                        self.logger.info(f"  ✓ Ollama 준비 완료: {try_model}")
                        return
                    else:
                        self.logger.warning(f"    → {try_model} 응답 느림, 다음 모델 시도...")

                except Exception as e:
                    self.logger.warning(f"    → {try_model} 오류: {e}")
                    continue

            # 모든 모델 실패
            result.ollama["error"] = "모든 모델 응답 실패"
            self.logger.warning("  ✗ 모든 모델 Warm-up 실패")

        except Exception as e:
            result.ollama["error"] = str(e)
            self.logger.warning(f"  ✗ Ollama 확인 실패: {e}")

    def _check_claude_cli(self, result: PreflightResult):
        """Claude CLI 사용 가능 여부 확인"""
        self.logger.info("[2/7] Claude CLI 확인...")

        try:
            from src.llm.cli_client import ClaudeCliClient

            client = ClaudeCliClient(timeout=10)

            if client.is_available():
                result.claude_cli["available"] = True
                self.logger.info("  ✓ Claude CLI 사용 가능")

                # 간단한 테스트 (선택적)
                try:
                    import subprocess
                    ver_result = subprocess.run(
                        ["claude", "--version"],
                        capture_output=True,
                        text=True,
                        shell=True,
                        timeout=5
                    )
                    if ver_result.returncode == 0:
                        version = ver_result.stdout.strip()
                        result.claude_cli["version"] = version
                        self.logger.info(f"    Version: {version[:50]}")
                except Exception:
                    pass
            else:
                result.claude_cli["error"] = "CLI 명령어 실행 실패"
                self.logger.warning("  ✗ Claude CLI 사용 불가")

        except ImportError as e:
            result.claude_cli["error"] = f"모듈 임포트 실패: {e}"
            self.logger.warning(f"  ✗ Claude CLI 모듈 로드 실패: {e}")
        except Exception as e:
            result.claude_cli["error"] = str(e)
            self.logger.warning(f"  ✗ Claude CLI 확인 실패: {e}")

    def _check_naver(self, result: PreflightResult):
        """네이버 금융 접근 확인"""
        self.logger.info("[3/7] 네이버 금융 접근 확인...")

        try:
            import requests
            response = requests.get(
                "https://finance.naver.com/sise/theme.naver",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10
            )
            if response.status_code == 200:
                result.naver["available"] = True
                self.logger.info("  ✓ 네이버 금융 접근 가능")
            else:
                result.naver["error"] = f"HTTP {response.status_code}"
                self.logger.warning(f"  ✗ 네이버 금융 응답 오류: {response.status_code}")
        except Exception as e:
            result.naver["error"] = str(e)
            self.logger.warning(f"  ✗ 네이버 금융 접근 실패: {e}")

    def _check_dart_api(self, result: PreflightResult):
        """DART API 연결 확인 (직접 연결 → 실패 시 프록시)"""
        self.logger.info("[4/7] DART API 확인...")

        try:
            import os
            import requests
            from src.core.config import get_config

            config = get_config()

            # API 키 확인 (백업 키 우선)
            api_key = os.getenv("DART_API_KEY_2") or os.getenv("DART_API_KEY") or config.get("api.dart_key", "")

            if not api_key:
                result.dart_api["error"] = "API 키가 설정되지 않음 (DART_API_KEY)"
                self.logger.warning("  ✗ DART API 키 없음")
                return

            test_url = "https://opendart.fss.or.kr/api/company.json"
            test_params = {"crtfc_key": api_key, "corp_code": "00126380"}

            # 1차: 직접 연결 시도
            self.logger.info("  - 직접 연결 시도...")
            try:
                response = requests.get(test_url, params=test_params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "000":
                        result.dart_api["available"] = True
                        result.dart_api["use_proxy"] = False
                        self.logger.info("  ✓ DART API 직접 연결 성공")
                        return
                    elif data.get("status") == "013":
                        self.logger.warning("  - 직접 연결 Rate Limit, 프록시 시도...")
                    else:
                        self.logger.warning(f"  - 직접 연결 실패: {data.get('message')}")
            except Exception as e:
                self.logger.warning(f"  - 직접 연결 실패: {e}")

            # 2차: 프록시로 재시도
            self.logger.info("  - 프록시 연결 시도...")
            try:
                from fp.fp import FreeProxy
                proxy = FreeProxy(country_id=['KR', 'JP', 'US'], timeout=1, rand=True).get()
                proxies = {"http": proxy, "https": proxy}
                self.logger.info(f"  - 프록시: {proxy}")

                response = requests.get(test_url, params=test_params, proxies=proxies, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "000":
                        result.dart_api["available"] = True
                        result.dart_api["use_proxy"] = True
                        result.dart_api["proxy"] = proxy
                        # 검증된 프록시 저장 (이후 DartApiClient에서 재사용)
                        from src.ingest.dart_client import set_verified_proxy
                        set_verified_proxy(proxy)
                        self.logger.info("  ✓ DART API 프록시 연결 성공")
                        return
                    else:
                        result.dart_api["error"] = f"API 오류: {data.get('message')}"
                        self.logger.warning(f"  ✗ DART API 오류: {data.get('message')}")
                else:
                    result.dart_api["error"] = f"HTTP {response.status_code}"
                    self.logger.warning(f"  ✗ DART API 프록시 응답 오류: {response.status_code}")
            except ImportError:
                result.dart_api["error"] = "프록시 모듈(free-proxy) 설치 필요"
                self.logger.warning("  ✗ free-proxy 모듈 없음 (pip install free-proxy)")
            except Exception as e:
                result.dart_api["error"] = f"프록시 연결 실패: {e}"
                self.logger.warning(f"  ✗ DART API 프록시 연결 실패: {e}")

        except ImportError as e:
            result.dart_api["error"] = f"requests 모듈 없음: {e}"
            self.logger.warning(f"  ✗ requests 모듈 없음: {e}")
        except Exception as e:
            result.dart_api["error"] = str(e)
            self.logger.warning(f"  ✗ DART API 확인 실패: {e}")

    def _check_telegram(self, result: PreflightResult):
        """Telegram Bot 연결 확인"""
        self.logger.info("[5/7] Telegram Bot 확인...")

        try:
            import os
            from src.core.config import get_config

            config = get_config()

            # Bot Token 확인
            bot_token = os.getenv("TELEGRAM_BOT_TOKEN") or config.get("output.telegram.bot_token", "")
            chat_id = os.getenv("TELEGRAM_CHAT_ID") or config.get("output.telegram.chat_id", "")

            if not bot_token:
                result.telegram["error"] = "Bot 토큰이 설정되지 않음 (TELEGRAM_BOT_TOKEN)"
                self.logger.warning("  ✗ Telegram Bot 토큰 없음 (선택 사항)")
                return

            if not chat_id:
                result.telegram["error"] = "Chat ID가 설정되지 않음 (TELEGRAM_CHAT_ID)"
                self.logger.warning("  ✗ Telegram Chat ID 없음 (선택 사항)")
                return

            # Bot 연결 테스트 (getMe API)
            import requests
            response = requests.get(
                f"https://api.telegram.org/bot{bot_token}/getMe",
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    bot_info = data.get("result", {})
                    bot_name = bot_info.get("username", "")
                    result.telegram["available"] = True
                    result.telegram["bot_name"] = bot_name
                    self.logger.info(f"  ✓ Telegram Bot 연결 성공: @{bot_name}")
                else:
                    result.telegram["error"] = data.get("description", "Unknown error")
                    self.logger.warning(f"  ✗ Telegram Bot 응답 오류: {data.get('description')}")
            else:
                result.telegram["error"] = f"HTTP {response.status_code}"
                self.logger.warning(f"  ✗ Telegram API 응답 오류: {response.status_code}")

        except Exception as e:
            result.telegram["error"] = str(e)
            self.logger.warning(f"  ✗ Telegram Bot 확인 실패: {e} (선택 사항)")

    def _check_pykrx(self, result: PreflightResult):
        """pykrx 모듈 확인"""
        self.logger.info("[6/7] pykrx 모듈 확인...")

        try:
            from pykrx import stock
            result.pykrx["available"] = True
            self.logger.info("  ✓ pykrx 모듈 로드 성공")
        except ImportError as e:
            result.pykrx["error"] = str(e)
            self.logger.warning(f"  ✗ pykrx 임포트 실패: {e}")

    def _check_modules(self, result: PreflightResult):
        """필수 모듈 확인"""
        self.logger.info("[7/7] 필수 모듈 확인...")

        try:
            # 데이터 수집 모듈
            from src.ingest import NaverThemeCrawler, PriceDataFetcher, NewsCrawler

            # 데이터 처리 모듈
            from src.processing import Preprocessor

            # 섹터/테마 분석 모듈 (v3.0)
            from src.sector import StockThemeAnalyzer, SectorTypeAnalyzer, SectorPrioritizer

            # 종목 분석 모듈 (v3.0)
            from src.stock import StockFilter, StockScorer, CandidateSelector

            # 검증 모듈
            from src.verify import MaterialAnalyzer, SentimentAnalyzer, DecisionEngine

            # 출력 모듈
            from src.output import StageSaver, ReportGenerator, TelegramNotifier

            # 분석 모듈
            from src.analysis import FlowCalculator, TierClassifier

            result.modules["available"] = True
            self.logger.info("  ✓ 모든 필수 모듈 로드 성공")
        except ImportError as e:
            result.modules["error"] = str(e)
            self.logger.warning(f"  ✗ 모듈 임포트 실패: {e}")

    def get_ollama_client(self):
        """Warm-up된 Ollama 클라이언트 반환"""
        return self._ollama_client


def run_preflight(
    model: str | None = None,
    warmup_timeout: float = 300.0,
) -> PreflightResult:
    """
    Preflight 검사 편의 함수 (모든 항목 필수)

    Args:
        model: 사용할 Ollama 모델명
        warmup_timeout: Warm-up 타임아웃 (초)

    Returns:
        PreflightResult (하나라도 실패 시 passed=False)
    """
    checker = PreflightChecker()
    return checker.run(
        model=model,
        warmup_timeout=warmup_timeout,
    )


def list_models() -> list[str]:
    """사용 가능한 Ollama 모델 목록"""
    checker = PreflightChecker()
    return checker.get_available_models()


def select_model() -> str:
    """최적 모델 자동 선택 (deepseek 우선, GLM 제외)"""
    checker = PreflightChecker()
    return checker.select_best_model()
