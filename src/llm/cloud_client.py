"""
Cloud LLM 클라이언트 (Anthropic Claude)

섹터 분류, 재료 분석, 심리 분석에 사용
"""
import os
from typing import Any

import anthropic

from src.core.interfaces import LLMClient
from src.core.config import get_config
from src.core.logger import get_logger
from src.core.exceptions import LLMError, LLMConnectionError, LLMResponseError


class ClaudeClient(LLMClient):
    """
    Anthropic Claude API 클라이언트

    사용법:
        client = ClaudeClient()

        # 텍스트 생성
        response = client.generate("삼성전자에 대해 분석해줘")

        # 섹터 분류
        sector_type = client.classify_sector("바이오")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        config = get_config()
        self.logger = get_logger(self.__class__.__name__)

        # API 키: 인자 > 환경변수 > 설정파일
        self.api_key = (
            api_key
            or os.getenv("ANTHROPIC_API_KEY")
            or config.get("llm.cloud.api_key", "")
        )

        if not self.api_key:
            self.logger.warning("Anthropic API 키가 설정되지 않았습니다")

        self.model = model or config.get("llm.cloud.model", "claude-sonnet-4-20250514")
        self.max_tokens = max_tokens
        self.temperature = temperature

        self._client: anthropic.Anthropic | None = None

    def _get_client(self) -> anthropic.Anthropic:
        """Anthropic 클라이언트 (Lazy init)"""
        if self._client is None:
            try:
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except Exception as e:
                raise LLMConnectionError(f"Anthropic 클라이언트 초기화 실패: {e}")
        return self._client

    def is_available(self) -> bool:
        """API 사용 가능 여부"""
        if not self.api_key:
            return False
        try:
            # 간단한 테스트 요청
            self._get_client()
            return True
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs
    ) -> str:
        """
        텍스트 생성

        Args:
            prompt: 사용자 프롬프트
            system: 시스템 프롬프트 (선택)
            max_tokens: 최대 토큰 수
            temperature: 온도 (0.0 = 결정적, 1.0 = 창의적)

        Returns:
            생성된 텍스트
        """
        client = self._get_client()

        try:
            messages = [{"role": "user", "content": prompt}]

            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature,
                system=system or "",
                messages=messages,
            )

            # 응답 텍스트 추출
            if response.content and len(response.content) > 0:
                return response.content[0].text
            else:
                raise LLMResponseError("빈 응답 반환됨")

        except anthropic.APIError as e:
            self.logger.error(f"Anthropic API 오류: {e}")
            raise LLMError(f"API 오류: {e}")
        except Exception as e:
            self.logger.error(f"LLM 생성 실패: {e}")
            raise LLMError(f"생성 실패: {e}")

    def classify_sector(self, sector_name: str) -> str:
        """
        섹터 Type A/B 분류

        Args:
            sector_name: 섹터명 (예: "바이오", "자동차")

        Returns:
            "A" 또는 "B"
        """
        system_prompt = """너는 한국 주식시장 섹터 분석 전문가다.
섹터의 주가를 움직이는 핵심 동력을 분석하여 Type A 또는 Type B로 분류해라.

Type A (실적 기반): 영업이익, PER, 실적 턴어라운드가 핵심. 숫자가 찍혀야 주가가 간다.
예시: 자동차, 음식료, 은행, 반도체부품, 건설, 유통, 철강, 화학

Type B (기대감 기반): 기술력, 파이프라인, 미래 성장성이 핵심. 꿈을 먹고 주가가 간다.
예시: 바이오, AI, 로봇, 우주항공, 2차전지, 메타버스, 양자컴퓨터

반드시 "A" 또는 "B" 한 글자만 답변해라."""

        prompt = f'섹터: "{sector_name}"\n\n이 섹터는 Type A인가, Type B인가?'

        try:
            response = self.generate(prompt, system=system_prompt, max_tokens=10)
            result = response.strip().upper()

            # A 또는 B만 추출
            if "A" in result:
                return "A"
            elif "B" in result:
                return "B"
            else:
                self.logger.warning(f"섹터 분류 불명확: {sector_name} -> {response}")
                return "A"  # 기본값은 보수적으로 A

        except Exception as e:
            self.logger.error(f"섹터 분류 실패: {sector_name} - {e}")
            return "A"  # 실패 시 보수적으로 A

    def classify_sectors_batch(self, sector_names: list[str]) -> dict[str, str]:
        """
        여러 섹터 일괄 분류 (비용 최적화)

        Args:
            sector_names: 섹터명 리스트

        Returns:
            {섹터명: "A" 또는 "B", ...}
        """
        if not sector_names:
            return {}

        system_prompt = """너는 한국 주식시장 섹터 분석 전문가다.
각 섹터의 주가를 움직이는 핵심 동력을 분석하여 Type A 또는 Type B로 분류해라.

Type A (실적 기반): 영업이익, PER, 실적이 핵심. 예: 자동차, 음식료, 은행, 건설
Type B (기대감 기반): 기술력, 미래 성장성이 핵심. 예: 바이오, AI, 로봇, 2차전지

각 섹터에 대해 "섹터명: A" 또는 "섹터명: B" 형식으로 한 줄씩 답변해라."""

        sectors_text = "\n".join([f"- {name}" for name in sector_names])
        prompt = f"다음 섹터들을 분류해라:\n\n{sectors_text}"

        try:
            response = self.generate(prompt, system=system_prompt, max_tokens=500)

            # 응답 파싱
            result = {}
            for line in response.strip().split("\n"):
                line = line.strip()
                if ":" in line:
                    parts = line.split(":")
                    sector = parts[0].strip().lstrip("- ")
                    type_str = parts[-1].strip().upper()

                    if "A" in type_str:
                        result[sector] = "A"
                    elif "B" in type_str:
                        result[sector] = "B"

            # 누락된 섹터는 A로 기본 설정
            for name in sector_names:
                if name not in result:
                    result[name] = "A"

            return result

        except Exception as e:
            self.logger.error(f"배치 섹터 분류 실패: {e}")
            # 실패 시 모두 A로
            return {name: "A" for name in sector_names}
