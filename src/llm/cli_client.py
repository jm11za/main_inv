"""
Claude CLI 클라이언트

subprocess로 Claude CLI를 호출하여 현재 인증을 그대로 사용
"""
import subprocess
from typing import Any

from src.core.interfaces import LLMClient
from src.core.logger import get_logger
from src.core.exceptions import LLMError


class ClaudeCliClient(LLMClient):
    """
    Claude CLI 클라이언트

    subprocess로 Claude CLI를 호출하여 OAuth 토큰을 그대로 사용

    사용법:
        client = ClaudeCliClient()

        # 텍스트 생성
        response = client.generate("삼성전자에 대해 분석해줘")

        # 섹터 분류
        sector_type = client.classify_sector("바이오")
    """

    def __init__(self, timeout: int = 60):
        self.logger = get_logger(self.__class__.__name__)
        self.timeout = timeout

    def is_available(self) -> bool:
        """CLI 사용 가능 여부"""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                shell=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Claude CLI로 텍스트 생성

        Args:
            prompt: 프롬프트

        Returns:
            생성된 텍스트
        """
        try:
            result = subprocess.run(
                ["claude", "-p", prompt],
                capture_output=True,
                text=True,
                encoding="utf-8",
                shell=True,
                timeout=self.timeout,
            )

            if result.returncode != 0:
                error_msg = result.stderr or "Unknown error"
                raise LLMError(f"CLI 오류: {error_msg}")

            return result.stdout.strip()

        except subprocess.TimeoutExpired:
            raise LLMError(f"CLI 타임아웃 ({self.timeout}초)")
        except Exception as e:
            self.logger.error(f"CLI 호출 실패: {e}")
            raise LLMError(f"CLI 호출 실패: {e}")

    def classify_sector(self, sector_name: str) -> str:
        """
        섹터 Type A/B 분류

        Args:
            sector_name: 섹터명

        Returns:
            "A" 또는 "B"
        """
        prompt = f'''한국 주식시장에서 "{sector_name}" 섹터의 주가를 움직이는 핵심 동력을 분석해.

Type A (실적 기반): 영업이익, PER, 실적 턴어라운드가 중요. "숫자가 찍혀야 주가가 간다"
예시: 자동차, 음식료, 은행, 건설, 유통, 철강, 화학

Type B (기대감 기반): 기술력, 파이프라인, 미래 성장성이 중요. "꿈을 먹고 주가가 간다"
예시: 바이오, AI, 로봇, 우주항공, 2차전지, 메타버스

"{sector_name}" 섹터는 Type A인가 Type B인가?
반드시 "A" 또는 "B" 한 글자만 답해.'''

        try:
            response = self.generate(prompt)
            result = response.strip().upper()

            # A 또는 B 추출
            if "B" in result:
                return "B"
            else:
                return "A"  # 기본값은 보수적으로 A

        except Exception as e:
            self.logger.error(f"섹터 분류 실패: {sector_name} - {e}")
            return "A"

    def classify_sectors_batch(self, sector_names: list[str]) -> dict[str, str]:
        """
        여러 섹터 일괄 분류

        Args:
            sector_names: 섹터명 리스트

        Returns:
            {섹터명: "A" 또는 "B", ...}
        """
        if not sector_names:
            return {}

        sectors_text = ", ".join(sector_names)
        prompt = f'''다음 섹터들을 Type A 또는 Type B로 분류해.

Type A (실적 기반): 영업이익, 실적이 주가 핵심. 예: 자동차, 은행, 건설
Type B (기대감 기반): 기술력, 미래 성장성이 주가 핵심. 예: 바이오, AI, 로봇

섹터: {sectors_text}

각 섹터에 대해 "섹터명:A" 또는 "섹터명:B" 형식으로 쉼표로 구분해서 답해.
예: 자동차:A, 바이오:B'''

        try:
            response = self.generate(prompt)

            result = {}
            # 파싱
            for part in response.replace("\n", ",").split(","):
                part = part.strip()
                if ":" in part:
                    name, type_str = part.rsplit(":", 1)
                    name = name.strip()
                    type_str = type_str.strip().upper()

                    if name in sector_names or any(name in s for s in sector_names):
                        matched_name = next((s for s in sector_names if name in s or s in name), name)
                        result[matched_name] = "B" if "B" in type_str else "A"

            # 누락된 섹터는 A로 기본 설정
            for name in sector_names:
                if name not in result:
                    result[name] = "A"

            return result

        except Exception as e:
            self.logger.error(f"배치 섹터 분류 실패: {e}")
            return {name: "A" for name in sector_names}

    def analyze_material(self, stock_name: str, news_headlines: str) -> str:
        """
        재료 분석 (Skeptic 페르소나)

        Args:
            stock_name: 종목명
            news_headlines: 최근 뉴스 헤드라인

        Returns:
            "S", "A", "B", "C" 등급
        """
        prompt = f'''너는 냉철한 주식 애널리스트다. "{stock_name}" 종목의 재료를 분석해.

최근 뉴스:
{news_headlines}

재료 등급을 매겨:
S: 대형 호재 (실적 서프라이즈, 대규모 수주, M&A)
A: 중형 호재 (신사업 진출, 파트너십)
B: 소형 호재 (일반 뉴스)
C: 재료 없음 또는 악재

반드시 S, A, B, C 중 한 글자만 답해.'''

        try:
            response = self.generate(prompt)
            result = response.strip().upper()

            for grade in ["S", "A", "B", "C"]:
                if grade in result:
                    return grade
            return "C"

        except Exception:
            return "C"

    def analyze_sentiment(self, stock_name: str, community_posts: str) -> str:
        """
        심리 분석 (Sentiment Reader 페르소나)

        Args:
            stock_name: 종목명
            community_posts: 커뮤니티 글/댓글

        Returns:
            "공포", "의심", "확신", "환희"
        """
        prompt = f'''너는 투자 심리 분석가다. "{stock_name}" 종목에 대한 시장 심리를 분석해.

커뮤니티 반응:
{community_posts}

현재 심리 단계는?
- 공포: 바닥권, 모두가 비관적
- 의심: 초기 상승, 반신반의
- 확신: 중기, 대중이 관심 갖기 시작
- 환희: 고점, 모두가 낙관적 (위험)

반드시 공포, 의심, 확신, 환희 중 하나만 답해.'''

        try:
            response = self.generate(prompt)
            result = response.strip()

            for stage in ["공포", "의심", "확신", "환희"]:
                if stage in result:
                    return stage
            return "의심"

        except Exception:
            return "의심"
