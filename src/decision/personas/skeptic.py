"""
Skeptic Persona - 냉철한 애널리스트

뉴스와 공시를 분석하여 재료 등급(S/A/B/C)을 부여
- S: 대형 호재 (정부 정책, 대규모 수주, 임상 승인 등)
- A: 중형 호재 (실적 개선, 신규 계약, 기술 개발 등)
- B: 소형 호재 (일반 뉴스, 경미한 호재)
- C: 재료 없음 또는 악재
"""
from dataclasses import dataclass, field
from typing import Any

from src.core.interfaces import MaterialGrade
from src.core.logger import get_logger
from src.llm.ollama_client import OllamaClient


@dataclass
class SkepticAnalysis:
    """Skeptic 분석 결과"""
    stock_code: str
    stock_name: str
    material_grade: MaterialGrade
    confidence: float  # 0.0 ~ 1.0
    key_materials: list[str] = field(default_factory=list)  # 핵심 재료 목록
    risks: list[str] = field(default_factory=list)  # 발굴된 리스크
    reasoning: str = ""  # 판단 근거


class Skeptic:
    """
    냉철한 애널리스트 페르소나

    뉴스, 공시(CB, 유증), IR 자료를 분석하여 재료의 실체를 검증하고
    리스크를 발굴하며 지속성을 평가합니다.
    """

    # 재료 등급별 키워드 (규칙 기반 폴백용)
    S_GRADE_KEYWORDS = [
        "대규모 수주", "정부 지원", "국책 과제", "임상 3상 승인",
        "FDA 승인", "특허 취득", "독점 계약", "합작 법인",
        "인수합병", "대형 계약", "양산 시작", "흑자 전환",
        "어닝 서프라이즈", "수출 계약",
    ]

    A_GRADE_KEYWORDS = [
        "실적 개선", "매출 증가", "수주 확보", "MOU 체결",
        "신규 계약", "기술 개발", "시장 진출", "제품 출시",
        "인증 획득", "투자 유치", "생산 증설", "R&D 성과",
    ]

    B_GRADE_KEYWORDS = [
        "관심 증가", "업황 개선", "기대감", "협력 논의",
        "검토 중", "계획 발표", "전망 밝아", "가능성",
    ]

    RISK_KEYWORDS = [
        "유상증자", "전환사채", "CB 발행", "자금 조달",
        "적자 지속", "매출 감소", "소송", "분쟁",
        "감사 의견", "상폐 위기", "관리 종목", "거래 정지",
        "대주주 매도", "최대주주 변경", "횡령", "배임",
    ]

    SYSTEM_PROMPT = """당신은 10년 경력의 냉철한 주식 애널리스트입니다.
감정에 휩쓸리지 않고 오직 사실과 숫자만을 봅니다.
투자자들이 놓치기 쉬운 리스크를 발굴하는 것이 특기입니다.

주어진 종목의 뉴스와 공시를 분석하여:
1. 재료의 실체가 있는지 검증하세요
2. 숨겨진 리스크를 발굴하세요
3. 재료의 지속성을 평가하세요

재료 등급 기준:
- S급: 정부 정책, 대규모 수주, FDA/임상 승인, 대형 인수합병 등 (주가 20%+ 상승 잠재력)
- A급: 실적 개선, 신규 계약, 기술 개발 등 (주가 10%+ 상승 잠재력)
- B급: 일반 호재, 업황 개선 기대 등 (주가 5% 내외)
- C급: 재료 없음, 악재 존재, 루머성 뉴스

반드시 다음 JSON 형식으로 응답하세요:
{
    "grade": "S" or "A" or "B" or "C",
    "confidence": 0.0 ~ 1.0,
    "key_materials": ["핵심 재료 1", "핵심 재료 2"],
    "risks": ["리스크 1", "리스크 2"],
    "reasoning": "판단 근거 (2-3문장)"
}"""

    def __init__(self, llm_client: OllamaClient | None = None):
        self.logger = get_logger(self.__class__.__name__)
        self.llm_client = llm_client

    def analyze(
        self,
        stock_code: str,
        stock_name: str,
        news_headlines: list[str] | None = None,
        announcements: list[str] | None = None,
        ir_content: str | None = None,
    ) -> SkepticAnalysis:
        """
        종목 재료 분석

        Args:
            stock_code: 종목코드
            stock_name: 종목명
            news_headlines: 최근 뉴스 헤드라인 목록
            announcements: 공시 목록 (CB, 유증 등)
            ir_content: IR 자료 내용

        Returns:
            SkepticAnalysis
        """
        news_headlines = news_headlines or []
        announcements = announcements or []

        # LLM 분석 시도
        if self.llm_client and self.llm_client.is_available():
            try:
                return self._analyze_with_llm(
                    stock_code, stock_name,
                    news_headlines, announcements, ir_content
                )
            except Exception as e:
                self.logger.warning(f"LLM 분석 실패, 규칙 기반 폴백: {e}")

        # 규칙 기반 폴백
        return self._analyze_with_rules(
            stock_code, stock_name,
            news_headlines, announcements
        )

    def _analyze_with_llm(
        self,
        stock_code: str,
        stock_name: str,
        news_headlines: list[str],
        announcements: list[str],
        ir_content: str | None
    ) -> SkepticAnalysis:
        """LLM 기반 분석"""
        # 프롬프트 구성
        content_parts = []

        if news_headlines:
            content_parts.append(f"[최근 뉴스 헤드라인]\n" + "\n".join(f"- {h}" for h in news_headlines[:20]))

        if announcements:
            content_parts.append(f"[공시 내용]\n" + "\n".join(f"- {a}" for a in announcements[:10]))

        if ir_content:
            content_parts.append(f"[IR 자료]\n{ir_content[:2000]}")

        if not content_parts:
            return SkepticAnalysis(
                stock_code=stock_code,
                stock_name=stock_name,
                material_grade=MaterialGrade.C,
                confidence=1.0,
                reasoning="분석할 재료가 없습니다."
            )

        prompt = f"""종목: {stock_name} ({stock_code})

{chr(10).join(content_parts)}

위 내용을 분석하여 재료 등급을 평가하세요."""

        response = self.llm_client.generate(
            prompt=prompt,
            system=self.SYSTEM_PROMPT,
            temperature=0.3
        )

        return self._parse_llm_response(stock_code, stock_name, response)

    def _parse_llm_response(
        self,
        stock_code: str,
        stock_name: str,
        response: str
    ) -> SkepticAnalysis:
        """LLM 응답 파싱"""
        import json
        import re

        # JSON 추출 시도
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())

                grade_map = {
                    "S": MaterialGrade.S,
                    "A": MaterialGrade.A,
                    "B": MaterialGrade.B,
                    "C": MaterialGrade.C,
                }

                return SkepticAnalysis(
                    stock_code=stock_code,
                    stock_name=stock_name,
                    material_grade=grade_map.get(data.get("grade", "C"), MaterialGrade.C),
                    confidence=float(data.get("confidence", 0.7)),
                    key_materials=data.get("key_materials", []),
                    risks=data.get("risks", []),
                    reasoning=data.get("reasoning", "")
                )
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.warning(f"JSON 파싱 실패: {e}")

        # 파싱 실패 시 텍스트에서 등급 추출
        grade = MaterialGrade.C
        if "S급" in response or "grade\": \"S" in response:
            grade = MaterialGrade.S
        elif "A급" in response or "grade\": \"A" in response:
            grade = MaterialGrade.A
        elif "B급" in response or "grade\": \"B" in response:
            grade = MaterialGrade.B

        return SkepticAnalysis(
            stock_code=stock_code,
            stock_name=stock_name,
            material_grade=grade,
            confidence=0.5,
            reasoning=response[:200]
        )

    def _analyze_with_rules(
        self,
        stock_code: str,
        stock_name: str,
        news_headlines: list[str],
        announcements: list[str]
    ) -> SkepticAnalysis:
        """규칙 기반 분석 (폴백)"""
        all_text = " ".join(news_headlines + announcements).lower()

        # 리스크 체크 (최우선)
        risks = []
        for keyword in self.RISK_KEYWORDS:
            if keyword in all_text:
                risks.append(keyword)

        # 등급 결정
        key_materials = []
        grade = MaterialGrade.C
        confidence = 0.6

        # S급 키워드 체크
        for keyword in self.S_GRADE_KEYWORDS:
            if keyword in all_text:
                key_materials.append(keyword)
                grade = MaterialGrade.S
                confidence = 0.8

        # A급 키워드 체크 (S급 아닌 경우)
        if grade == MaterialGrade.C:
            for keyword in self.A_GRADE_KEYWORDS:
                if keyword in all_text:
                    key_materials.append(keyword)
                    grade = MaterialGrade.A
                    confidence = 0.7

        # B급 키워드 체크
        if grade == MaterialGrade.C:
            for keyword in self.B_GRADE_KEYWORDS:
                if keyword in all_text:
                    key_materials.append(keyword)
                    grade = MaterialGrade.B
                    confidence = 0.6

        # 리스크가 많으면 등급 하향
        if len(risks) >= 2:
            if grade == MaterialGrade.S:
                grade = MaterialGrade.A
            elif grade == MaterialGrade.A:
                grade = MaterialGrade.B
            elif grade == MaterialGrade.B:
                grade = MaterialGrade.C

        reasoning = self._generate_reasoning(grade, key_materials, risks)

        return SkepticAnalysis(
            stock_code=stock_code,
            stock_name=stock_name,
            material_grade=grade,
            confidence=confidence,
            key_materials=key_materials[:5],
            risks=risks[:5],
            reasoning=reasoning
        )

    def _generate_reasoning(
        self,
        grade: MaterialGrade,
        materials: list[str],
        risks: list[str]
    ) -> str:
        """판단 근거 생성"""
        parts = []

        if materials:
            parts.append(f"발견된 호재: {', '.join(materials[:3])}")

        if risks:
            parts.append(f"주의 필요: {', '.join(risks[:3])}")

        if not materials and not risks:
            parts.append("특별한 재료가 발견되지 않았습니다.")

        grade_desc = {
            MaterialGrade.S: "대형 호재로 판단됩니다.",
            MaterialGrade.A: "중형 호재입니다.",
            MaterialGrade.B: "소형 호재 수준입니다.",
            MaterialGrade.C: "재료가 불명확합니다.",
        }
        parts.append(grade_desc[grade])

        return " ".join(parts)

    def analyze_batch(
        self,
        stocks_data: list[dict[str, Any]]
    ) -> list[SkepticAnalysis]:
        """
        복수 종목 일괄 분석

        Args:
            stocks_data: 종목 데이터 리스트
                [{stock_code, stock_name, news_headlines, announcements, ...}]

        Returns:
            SkepticAnalysis 리스트
        """
        results = []

        for data in stocks_data:
            result = self.analyze(
                stock_code=data["stock_code"],
                stock_name=data.get("stock_name", ""),
                news_headlines=data.get("news_headlines", []),
                announcements=data.get("announcements", []),
                ir_content=data.get("ir_content")
            )
            results.append(result)

        return results
