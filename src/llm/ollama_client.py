"""
Ollama 로컬 LLM 클라이언트

키워드 추출, 테마 분류 등 빈번한 호출에 사용
"""
import requests
from typing import Any

from src.core.interfaces import LLMClient
from src.core.config import get_config
from src.core.logger import get_logger
from src.core.exceptions import LLMError


class OllamaClient(LLMClient):
    """
    Ollama 로컬 LLM 클라이언트

    사용법:
        client = OllamaClient()

        # 텍스트 생성
        response = client.generate("키워드를 추출해줘: ...")

        # 키워드 추출
        keywords = client.extract_keywords(text)
    """

    # 추천 모델 (한국어 지원 및 성능 기준)
    RECOMMENDED_MODELS = [
        "deepseek-r1:14b",      # 추론 특화, 정확도 높음
        "llama3.1:8b",          # 범용, 빠름
        "glm-4.7-flash",        # 다국어 지원
        "gemma3:4b",            # 경량, 빠름
    ]

    def __init__(
        self,
        model: str | None = None,
        base_url: str = "http://127.0.0.1:11434",
        timeout: int = 300
    ):
        """
        Args:
            model: 사용할 모델명 (없으면 설정에서 로드)
            base_url: Ollama 서버 URL
            timeout: 요청 타임아웃 (초)
        """
        self.logger = get_logger(self.__class__.__name__)

        config = get_config()
        self.model = model or config.get("llm.local.model", "deepseek-r1:14b")
        self.base_url = config.get("llm.local.base_url", base_url)
        self.timeout = timeout or config.get("llm.local.timeout_seconds", 300)

        self.logger.info(f"Ollama 클라이언트 초기화: {self.model} @ {self.base_url}")

    def is_available(self) -> bool:
        """Ollama 서버 및 모델 사용 가능 여부"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            if response.status_code != 200:
                return False

            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]

            # 정확히 일치하거나 ":latest" 버전 체크
            for name in model_names:
                if self.model in name or name in self.model:
                    return True

            self.logger.warning(f"모델 {self.model}을 찾을 수 없음. 사용 가능: {model_names}")
            return False

        except Exception as e:
            self.logger.error(f"Ollama 연결 실패: {e}")
            return False

    def generate(self, prompt: str, **kwargs) -> str:
        """
        텍스트 생성

        Args:
            prompt: 프롬프트
            **kwargs: 추가 옵션 (temperature, top_p 등)

        Returns:
            생성된 텍스트
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                **kwargs
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code != 200:
                raise LLMError(f"Ollama API 오류: {response.status_code} - {response.text}")

            result = response.json()
            return result.get("response", "").strip()

        except requests.Timeout:
            raise LLMError(f"Ollama 타임아웃 ({self.timeout}초)")
        except requests.RequestException as e:
            raise LLMError(f"Ollama 요청 실패: {e}")

    def extract_keywords(self, text: str, max_keywords: int = 10) -> list[str]:
        """
        텍스트에서 투자 관련 키워드 추출

        Args:
            text: 분석할 텍스트 (뉴스, 사업보고서 등)
            max_keywords: 최대 키워드 수

        Returns:
            키워드 리스트
        """
        prompt = f'''다음 텍스트에서 주식 투자와 관련된 핵심 키워드를 추출해.

텍스트:
{text[:2000]}

규칙:
1. 산업/섹터 관련 키워드 (예: 반도체, 바이오, 2차전지, AI)
2. 사업 내용 키워드 (예: 신약개발, 배터리소재, 자율주행)
3. 최대 {max_keywords}개까지만 추출
4. 일반적인 단어(회사, 사업, 매출 등)는 제외

출력 형식: 키워드1, 키워드2, 키워드3, ...
키워드만 쉼표로 구분해서 출력해. 다른 설명 없이.'''

        try:
            response = self.generate(prompt)

            # 파싱: 쉼표로 구분된 키워드 추출
            keywords = []
            for part in response.replace("\n", ",").split(","):
                keyword = part.strip().strip("\"'.-")
                if keyword and len(keyword) >= 2 and len(keyword) <= 20:
                    keywords.append(keyword)

            return keywords[:max_keywords]

        except Exception as e:
            self.logger.error(f"키워드 추출 실패: {e}")
            return []

    def classify_theme_type(self, theme_name: str, keywords: list[str]) -> str:
        """
        테마 유형 분류 (실체형 vs 기대형)

        Args:
            theme_name: 테마명
            keywords: 관련 키워드

        Returns:
            "실체형" 또는 "기대형"
        """
        keywords_str = ", ".join(keywords) if keywords else "없음"

        prompt = f'''다음 테마의 유형을 분류해.

테마명: {theme_name}
관련 키워드: {keywords_str}

분류 기준:
- 실체형: 실제 매출/이익이 발생하는 사업 (예: 2차전지 소재, 반도체 장비, 자동차 부품)
- 기대형: 아직 실적보다 기대감으로 움직이는 사업 (예: AI 신사업, 로봇, 바이오 신약)

반드시 "실체형" 또는 "기대형" 중 하나만 답해.'''

        try:
            response = self.generate(prompt)
            result = response.strip()

            if "기대" in result:
                return "기대형"
            else:
                return "실체형"

        except Exception:
            return "실체형"  # 기본값

    def summarize_news(self, headlines: list[str], stock_name: str) -> str:
        """
        뉴스 헤드라인 요약

        Args:
            headlines: 뉴스 헤드라인 리스트
            stock_name: 종목명

        Returns:
            요약 텍스트
        """
        headlines_text = "\n".join(f"- {h}" for h in headlines[:20])

        prompt = f'''{stock_name} 관련 최근 뉴스를 요약해.

뉴스 헤드라인:
{headlines_text}

3줄 이내로 핵심 내용만 요약해.'''

        try:
            return self.generate(prompt)
        except Exception as e:
            self.logger.error(f"뉴스 요약 실패: {e}")
            return ""

    def get_model_info(self) -> dict:
        """현재 모델 정보 반환"""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "available": self.is_available(),
        }
