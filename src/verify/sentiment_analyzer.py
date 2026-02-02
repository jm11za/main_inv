"""
⑤ Sentiment Reader - 심리 분석 모듈

후보 종목의 커뮤니티 반응을 분석하여 심리 단계 판정
- 공포: 바닥권, 모두가 비관적
- 의심: 초기 상승, 반신반의
- 확신: 중기, 대중이 관심 갖기 시작
- 환희: 고점, 모두가 낙관적 (위험)
"""
from dataclasses import dataclass, field
from typing import Any

from src.core.logger import get_logger
from src.core.config import get_config
from src.core.interfaces import SentimentStage


@dataclass
class SentimentResult:
    """심리 분석 결과"""
    stock_code: str
    stock_name: str
    stage: SentimentStage  # 공포, 의심, 확신, 환희
    confidence: float  # 신뢰도 (0~1)

    # 분석 상세
    bullish_ratio: float  # 낙관 비율 (0~1)
    activity_level: str  # 활동 수준 (높음/중간/낮음)
    key_sentiments: list[str]  # 핵심 심리 키워드
    llm_analysis: str  # LLM 분석 내용

    # 입력 데이터
    post_count: int = 0
    total_likes: int = 0
    total_dislikes: int = 0

    def to_dict(self) -> dict:
        return {
            "stock_code": self.stock_code,
            "stock_name": self.stock_name,
            "stage": self.stage.value,
            "confidence": round(self.confidence, 2),
            "bullish_ratio": round(self.bullish_ratio, 2),
            "activity_level": self.activity_level,
            "key_sentiments": self.key_sentiments,
            "llm_analysis": self.llm_analysis,
            "post_count": self.post_count,
            "total_likes": self.total_likes,
            "total_dislikes": self.total_dislikes,
        }


class SentimentAnalyzer:
    """
    ⑤ Sentiment Reader - 심리 분석기

    커뮤니티 반응을 분석하여 시장 심리 단계 판단:
    - 공포: 매수 기회 (역발상)
    - 의심: 초기 진입 타이밍
    - 확신: 추세 추종
    - 환희: 탈출 시그널 (위험)

    사용법:
        analyzer = SentimentAnalyzer(llm_client)

        # 단일 종목 분석
        result = analyzer.analyze(
            stock_code="005930",
            stock_name="삼성전자",
            community_posts=["주가가 오를까요?", "손절해야 하나...", ...]
        )

        # 배치 분석
        results = analyzer.analyze_batch(stocks_data)
    """

    # 심리 단계별 특성
    STAGE_CHARACTERISTICS = {
        SentimentStage.FEAR: {
            "name": "공포",
            "description": "바닥권, 모두가 비관적",
            "bullish_range": (0.0, 0.25),
            "keywords": [
                "손절", "물타기", "망함", "바닥", "포기",
                "언제 오르냐", "하락", "최악", "희망없다"
            ],
            "investment_signal": "매수 검토 (역발상)",
        },
        SentimentStage.DOUBT: {
            "name": "의심",
            "description": "초기 상승, 반신반의",
            "bullish_range": (0.25, 0.50),
            "keywords": [
                "반등", "진짜?", "속지마", "의심", "관망",
                "기다려", "확인 필요", "아직"
            ],
            "investment_signal": "진입 검토",
        },
        SentimentStage.CONVICTION: {
            "name": "확신",
            "description": "중기, 대중이 관심 갖기 시작",
            "bullish_range": (0.50, 0.75),
            "keywords": [
                "오른다", "가즈아", "상승", "좋아", "추매",
                "목표가", "수익", "홀딩", "믿음"
            ],
            "investment_signal": "보유/추세 추종",
        },
        SentimentStage.EUPHORIA: {
            "name": "환희",
            "description": "고점, 모두가 낙관적 (위험)",
            "bullish_range": (0.75, 1.0),
            "keywords": [
                "부자", "대박", "100만원", "무조건", "올인",
                "절대 안떨어짐", "영끌", "빚투"
            ],
            "investment_signal": "탈출 검토 (위험)",
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
        community_posts: list[str] | None = None,
        likes: int = 0,
        dislikes: int = 0,
    ) -> SentimentResult:
        """
        단일 종목 심리 분석

        Args:
            stock_code: 종목코드
            stock_name: 종목명
            community_posts: 커뮤니티 글/댓글 리스트
            likes: 총 공감 수
            dislikes: 총 비공감 수

        Returns:
            SentimentResult
        """
        posts = community_posts or []

        # LLM이 있으면 LLM 분석
        if self._llm_client:
            return self._analyze_with_llm(stock_code, stock_name, posts, likes, dislikes)

        # 없으면 규칙 기반
        return self._analyze_by_rules(stock_code, stock_name, posts, likes, dislikes)

    def _analyze_with_llm(
        self,
        stock_code: str,
        stock_name: str,
        posts: list[str],
        likes: int,
        dislikes: int,
    ) -> SentimentResult:
        """LLM으로 심리 분석"""
        posts_text = "\n".join(f"- {p[:100]}" for p in posts[:20]) if posts else "없음"

        prompt = f"""너는 투자 심리 분석가다. "{stock_name}" 종목에 대한 시장 심리를 분석해.

[커뮤니티 반응]
{posts_text}

[공감/비공감]
공감: {likes}개, 비공감: {dislikes}개

현재 심리 단계는?
- 공포: 바닥권, 모두가 비관적
- 의심: 초기 상승, 반신반의
- 확신: 중기, 대중이 관심 갖기 시작
- 환희: 고점, 모두가 낙관적 (위험)

다음 형식으로 답해:
심리단계: [공포/의심/확신/환희]
낙관비율: [0~100]%
활동수준: [높음/중간/낮음]
핵심심리: [쉼표로 구분]
분석: [2-3문장]"""

        try:
            result = self._llm_client.generate(prompt).strip()

            # 파싱
            stage = SentimentStage.DOUBT
            bullish_ratio = 0.5
            activity_level = "중간"
            key_sentiments = []
            analysis = result

            for line in result.split("\n"):
                if "심리단계:" in line or "심리 단계:" in line:
                    stage_text = line.split(":", 1)[-1].strip()
                    for s in SentimentStage:
                        if s.value in stage_text:
                            stage = s
                            break
                elif "낙관비율:" in line or "낙관 비율:" in line:
                    ratio_text = line.split(":", 1)[-1].strip().replace("%", "")
                    try:
                        bullish_ratio = float(ratio_text) / 100
                    except ValueError:
                        pass
                elif "활동수준:" in line or "활동 수준:" in line:
                    activity_level = line.split(":", 1)[-1].strip()
                elif "핵심심리:" in line or "핵심 심리:" in line:
                    key_sentiments = [k.strip() for k in line.split(":", 1)[-1].split(",") if k.strip()]
                elif "분석:" in line:
                    analysis = line.split(":", 1)[-1].strip()

            return SentimentResult(
                stock_code=stock_code,
                stock_name=stock_name,
                stage=stage,
                confidence=0.85,
                bullish_ratio=bullish_ratio,
                activity_level=activity_level,
                key_sentiments=key_sentiments[:5],
                llm_analysis=analysis[:300],
                post_count=len(posts),
                total_likes=likes,
                total_dislikes=dislikes,
            )

        except Exception as e:
            self.logger.warning(f"LLM 심리 분석 실패 ({stock_name}): {e}")
            return self._analyze_by_rules(stock_code, stock_name, posts, likes, dislikes)

    def _analyze_by_rules(
        self,
        stock_code: str,
        stock_name: str,
        posts: list[str],
        likes: int,
        dislikes: int,
    ) -> SentimentResult:
        """규칙 기반 심리 분석"""
        combined_text = " ".join(posts).lower()

        # 각 단계별 키워드 점수
        stage_scores = {s: 0.0 for s in SentimentStage}

        for stage, chars in self.STAGE_CHARACTERISTICS.items():
            for kw in chars["keywords"]:
                if kw.lower() in combined_text:
                    stage_scores[stage] += 1.0

        # 공감/비공감 비율로 보정
        total_reactions = likes + dislikes
        if total_reactions > 0:
            like_ratio = likes / total_reactions
            # 공감 비율이 높으면 긍정적 단계로 이동
            if like_ratio > 0.7:
                stage_scores[SentimentStage.CONVICTION] += 2.0
                stage_scores[SentimentStage.EUPHORIA] += 1.0
            elif like_ratio < 0.3:
                stage_scores[SentimentStage.FEAR] += 2.0

        # 최고 점수 단계 선택
        best_stage = SentimentStage.DOUBT
        best_score = 0

        for stage, score in stage_scores.items():
            if score > best_score:
                best_stage = stage
                best_score = score

        # 낙관 비율 계산
        bullish_count = stage_scores[SentimentStage.CONVICTION] + stage_scores[SentimentStage.EUPHORIA]
        bearish_count = stage_scores[SentimentStage.FEAR] + stage_scores[SentimentStage.DOUBT]
        total_count = bullish_count + bearish_count
        bullish_ratio = bullish_count / total_count if total_count > 0 else 0.5

        # 활동 수준
        if len(posts) > 50:
            activity_level = "높음"
        elif len(posts) > 20:
            activity_level = "중간"
        else:
            activity_level = "낮음"

        # 핵심 심리 키워드 추출
        key_sentiments = []
        for stage, chars in self.STAGE_CHARACTERISTICS.items():
            for kw in chars["keywords"]:
                if kw.lower() in combined_text:
                    key_sentiments.append(kw)

        return SentimentResult(
            stock_code=stock_code,
            stock_name=stock_name,
            stage=best_stage,
            confidence=0.6,
            bullish_ratio=bullish_ratio,
            activity_level=activity_level,
            key_sentiments=list(set(key_sentiments))[:5],
            llm_analysis="규칙 기반 분석",
            post_count=len(posts),
            total_likes=likes,
            total_dislikes=dislikes,
        )

    def analyze_batch(
        self,
        stocks_data: list[dict],
        progress_callback=None,
    ) -> list[SentimentResult]:
        """
        배치 심리 분석

        Args:
            stocks_data: [
                {
                    "stock_code": "005930",
                    "stock_name": "삼성전자",
                    "community_posts": [...],
                    "likes": 100,
                    "dislikes": 20
                },
                ...
            ]
            progress_callback: 진행 콜백

        Returns:
            SentimentResult 리스트
        """
        self.logger.info(f"{len(stocks_data)}개 종목 심리 분석 시작")

        results = []
        for i, data in enumerate(stocks_data):
            result = self.analyze(
                stock_code=data.get("stock_code", ""),
                stock_name=data.get("stock_name", ""),
                community_posts=data.get("community_posts"),
                likes=data.get("likes", 0),
                dislikes=data.get("dislikes", 0),
            )
            results.append(result)

            if progress_callback and (i + 1) % 5 == 0:
                progress_callback(i + 1, len(stocks_data))

        self.logger.info(f"심리 분석 완료: {len(results)}개")
        return results

    def summarize(self, results: list[SentimentResult]) -> dict:
        """심리 분석 결과 요약"""
        stage_counts = {s: 0 for s in SentimentStage}
        for r in results:
            stage_counts[r.stage] += 1

        return {
            "total": len(results),
            "stage_distribution": {s.value: c for s, c in stage_counts.items()},
            "fear_stocks": [r.stock_name for r in results if r.stage == SentimentStage.FEAR],
            "doubt_stocks": [r.stock_name for r in results if r.stage == SentimentStage.DOUBT],
            "euphoria_stocks": [r.stock_name for r in results if r.stage == SentimentStage.EUPHORIA],
            "avg_bullish_ratio": sum(r.bullish_ratio for r in results) / len(results) if results else 0,
        }

    def get_stage_description(self, stage: SentimentStage) -> dict:
        """심리 단계 설명 반환"""
        return self.STAGE_CHARACTERISTICS.get(stage, {})
