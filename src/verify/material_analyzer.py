"""
⑤ Skeptic - 재료 분석 모듈

후보 종목의 뉴스/공시를 분석하여 재료 등급 판정
- S: 대형 호재 (실적 서프라이즈, 대규모 수주, M&A)
- A: 중형 호재 (신사업 진출, 파트너십)
- B: 소형 호재 (일반 뉴스)
- C: 재료 없음 또는 악재
"""
from dataclasses import dataclass, field
from typing import Any

from src.core.logger import get_logger
from src.core.config import get_config
from src.core.interfaces import MaterialGrade


@dataclass
class MaterialResult:
    """재료 분석 결과"""
    stock_code: str
    stock_name: str
    grade: MaterialGrade  # S, A, B, C
    confidence: float  # 신뢰도 (0~1)

    # 분석 상세
    key_materials: list[str]  # 핵심 재료 목록
    positive_factors: list[str]  # 긍정 요소
    negative_factors: list[str]  # 부정 요소
    llm_analysis: str  # LLM 분석 내용

    # 입력 데이터
    news_count: int = 0
    disclosure_count: int = 0

    def to_dict(self) -> dict:
        return {
            "stock_code": self.stock_code,
            "stock_name": self.stock_name,
            "grade": self.grade.value,
            "confidence": round(self.confidence, 2),
            "key_materials": self.key_materials,
            "positive_factors": self.positive_factors,
            "negative_factors": self.negative_factors,
            "llm_analysis": self.llm_analysis,
            "news_count": self.news_count,
            "disclosure_count": self.disclosure_count,
        }


class MaterialAnalyzer:
    """
    ⑤ Skeptic - 재료 분석기

    냉철한 시각으로 종목의 재료(뉴스/공시)를 분석:
    - 과장된 뉴스와 실제 영향력 있는 재료 구분
    - 숨겨진 악재 탐지
    - 재료의 주가 영향력 평가

    사용법:
        analyzer = MaterialAnalyzer(llm_client)

        # 단일 종목 분석
        result = analyzer.analyze(
            stock_code="005930",
            stock_name="삼성전자",
            news_headlines=["삼성 HBM3E 양산...", ...],
            disclosures=["분기보고서 제출", ...]
        )

        # 배치 분석
        results = analyzer.analyze_batch(stocks_data)
    """

    # 등급별 기준
    GRADE_CRITERIA = {
        MaterialGrade.S: {
            "name": "대형 호재",
            "keywords": [
                "실적 서프라이즈", "어닝 서프라이즈", "사상 최대",
                "대규모 수주", "메가딜", "M&A", "인수합병",
                "FDA 승인", "임상 성공", "글로벌 계약",
                "상향 조정", "목표가 상향"
            ],
            "min_impact": 0.8,
        },
        MaterialGrade.A: {
            "name": "중형 호재",
            "keywords": [
                "신사업", "진출", "파트너십", "MOU", "협력",
                "증설", "투자", "확장", "흑자전환",
                "수주", "계약", "공급", "납품"
            ],
            "min_impact": 0.5,
        },
        MaterialGrade.B: {
            "name": "소형 호재",
            "keywords": [
                "참여", "출시", "개발", "연구", "특허",
                "전시회", "컨퍼런스", "인터뷰"
            ],
            "min_impact": 0.2,
        },
        MaterialGrade.C: {
            "name": "재료 없음/악재",
            "keywords": [
                "적자", "손실", "감소", "하락", "부진",
                "소송", "분쟁", "제재", "리콜", "사고",
                "하향", "매도", "경고"
            ],
            "min_impact": 0.0,
        },
    }

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: LLM 클라이언트 (Claude CLI 권장)
        """
        self.logger = get_logger(self.__class__.__name__)
        self.config = get_config()
        self._llm_client = llm_client

    def set_llm_client(self, client):
        """LLM 클라이언트 설정"""
        self._llm_client = client

    def analyze(
        self,
        stock_code: str,
        stock_name: str,
        news_headlines: list[str] | None = None,
        disclosures: list[str] | None = None,
    ) -> MaterialResult:
        """
        단일 종목 재료 분석

        Args:
            stock_code: 종목코드
            stock_name: 종목명
            news_headlines: 뉴스 헤드라인 리스트
            disclosures: 공시 제목 리스트

        Returns:
            MaterialResult
        """
        news = news_headlines or []
        disc = disclosures or []

        # LLM이 있으면 LLM 분석
        if self._llm_client:
            return self._analyze_with_llm(stock_code, stock_name, news, disc)

        # 없으면 규칙 기반
        return self._analyze_by_rules(stock_code, stock_name, news, disc)

    def _analyze_with_llm(
        self,
        stock_code: str,
        stock_name: str,
        news: list[str],
        disclosures: list[str],
    ) -> MaterialResult:
        """LLM으로 재료 분석"""
        news_text = "\n".join(f"- {h}" for h in news[:15]) if news else "없음"
        disc_text = "\n".join(f"- {d}" for d in disclosures[:5]) if disclosures else "없음"

        prompt = f"""너는 냉철한 주식 애널리스트다. "{stock_name}" 종목의 재료를 분석해.

[최근 뉴스]
{news_text}

[최근 공시]
{disc_text}

재료 등급을 매겨:
S: 대형 호재 (실적 서프라이즈, 대규모 수주, M&A, FDA 승인)
A: 중형 호재 (신사업 진출, 파트너십, 증설)
B: 소형 호재 (일반 뉴스, 전시회 참여)
C: 재료 없음 또는 악재

다음 형식으로 답해:
등급: [S/A/B/C]
핵심재료: [쉼표로 구분]
긍정요소: [쉼표로 구분]
부정요소: [쉼표로 구분]
분석: [2-3문장]"""

        try:
            result = self._llm_client.generate(prompt).strip()

            # 파싱
            grade = MaterialGrade.C
            key_materials = []
            positive = []
            negative = []
            analysis = result

            for line in result.split("\n"):
                line_lower = line.lower()
                if "등급:" in line or "grade:" in line_lower:
                    grade_text = line.split(":", 1)[-1].strip().upper()
                    for g in MaterialGrade:
                        if g.value in grade_text:
                            grade = g
                            break
                elif "핵심재료:" in line or "핵심 재료:" in line:
                    key_materials = [m.strip() for m in line.split(":", 1)[-1].split(",") if m.strip()]
                elif "긍정요소:" in line or "긍정 요소:" in line:
                    positive = [p.strip() for p in line.split(":", 1)[-1].split(",") if p.strip()]
                elif "부정요소:" in line or "부정 요소:" in line:
                    negative = [n.strip() for n in line.split(":", 1)[-1].split(",") if n.strip()]
                elif "분석:" in line:
                    analysis = line.split(":", 1)[-1].strip()

            return MaterialResult(
                stock_code=stock_code,
                stock_name=stock_name,
                grade=grade,
                confidence=0.85,
                key_materials=key_materials[:5],
                positive_factors=positive[:5],
                negative_factors=negative[:5],
                llm_analysis=analysis[:300],
                news_count=len(news),
                disclosure_count=len(disclosures),
            )

        except Exception as e:
            self.logger.warning(f"LLM 재료 분석 실패 ({stock_name}): {e}")
            return self._analyze_by_rules(stock_code, stock_name, news, disclosures)

    def _analyze_by_rules(
        self,
        stock_code: str,
        stock_name: str,
        news: list[str],
        disclosures: list[str],
    ) -> MaterialResult:
        """규칙 기반 재료 분석"""
        combined_text = " ".join(news + disclosures).lower()

        # 등급별 점수 계산
        grade_scores = {g: 0.0 for g in MaterialGrade}

        for grade, criteria in self.GRADE_CRITERIA.items():
            for kw in criteria["keywords"]:
                if kw.lower() in combined_text:
                    grade_scores[grade] += 1.0

        # 최고 점수 등급 선택
        best_grade = MaterialGrade.C
        best_score = 0

        for grade in [MaterialGrade.S, MaterialGrade.A, MaterialGrade.B]:
            if grade_scores[grade] > best_score:
                best_grade = grade
                best_score = grade_scores[grade]

        # 악재 키워드가 많으면 C로 하향
        if grade_scores[MaterialGrade.C] >= 2:
            best_grade = MaterialGrade.C

        # 키워드 추출
        key_materials = []
        positive = []
        negative = []

        for grade, criteria in self.GRADE_CRITERIA.items():
            for kw in criteria["keywords"]:
                if kw.lower() in combined_text:
                    if grade == MaterialGrade.C:
                        negative.append(kw)
                    else:
                        positive.append(kw)
                        if grade in [MaterialGrade.S, MaterialGrade.A]:
                            key_materials.append(kw)

        return MaterialResult(
            stock_code=stock_code,
            stock_name=stock_name,
            grade=best_grade,
            confidence=0.6,
            key_materials=list(set(key_materials))[:5],
            positive_factors=list(set(positive))[:5],
            negative_factors=list(set(negative))[:5],
            llm_analysis="규칙 기반 분석",
            news_count=len(news),
            disclosure_count=len(disclosures),
        )

    def analyze_batch(
        self,
        stocks_data: list[dict],
        progress_callback=None,
    ) -> list[MaterialResult]:
        """
        배치 재료 분석

        Args:
            stocks_data: [
                {
                    "stock_code": "005930",
                    "stock_name": "삼성전자",
                    "news_headlines": [...],
                    "disclosures": [...]
                },
                ...
            ]
            progress_callback: 진행 콜백

        Returns:
            MaterialResult 리스트
        """
        self.logger.info(f"{len(stocks_data)}개 종목 재료 분석 시작")

        results = []
        for i, data in enumerate(stocks_data):
            result = self.analyze(
                stock_code=data.get("stock_code", ""),
                stock_name=data.get("stock_name", ""),
                news_headlines=data.get("news_headlines"),
                disclosures=data.get("disclosures"),
            )
            results.append(result)

            if progress_callback and (i + 1) % 5 == 0:
                progress_callback(i + 1, len(stocks_data))

        self.logger.info(f"재료 분석 완료: {len(results)}개")
        return results

    def summarize(self, results: list[MaterialResult]) -> dict:
        """재료 분석 결과 요약"""
        grade_counts = {g: 0 for g in MaterialGrade}
        for r in results:
            grade_counts[r.grade] += 1

        return {
            "total": len(results),
            "grade_distribution": {g.value: c for g, c in grade_counts.items()},
            "s_grade_stocks": [r.stock_name for r in results if r.grade == MaterialGrade.S],
            "a_grade_stocks": [r.stock_name for r in results if r.grade == MaterialGrade.A],
            "avg_confidence": sum(r.confidence for r in results) / len(results) if results else 0,
        }
