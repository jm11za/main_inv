"""
Sentiment Reader Persona - 심리 분석가

A+B+C 통합 접근법으로 심리 단계 판정:
- A (뉴스 기반): 뉴스 수량, 헤드라인 어조 분석
- B (토론방 기반): 네이버 금융 토론방 공감/비공감, 글 내용
- C (가격 기반): RSI, 수익률, 거래량 패턴

심리 단계:
- 공포(FEAR): 바닥권 - 모두가 포기, 관심도 최저
- 의심(DOUBT): 초기 - 반신반의, 소수만 관심
- 확신(CONVICTION): 중기 - 점점 확신, 관심도 상승
- 환희(EUPHORIA): 고점 - 모두가 낙관, 과열 (위험)
"""
from dataclasses import dataclass, field
from typing import Any

from src.core.interfaces import SentimentStage
from src.core.logger import get_logger
from src.llm.ollama_client import OllamaClient


@dataclass
class SentimentAnalysis:
    """Sentiment 분석 결과"""
    stock_code: str
    stock_name: str
    sentiment_stage: SentimentStage
    confidence: float  # 0.0 ~ 1.0
    interest_level: float  # 관심도 0.0 ~ 1.0
    tone_score: float  # -1.0 (부정) ~ 1.0 (긍정)
    sample_opinions: list[str] = field(default_factory=list)  # 대표 의견
    reasoning: str = ""  # 판단 근거

    # A+B+C 개별 분석 결과
    news_stage: SentimentStage | None = None  # A: 뉴스 기반
    discussion_stage: SentimentStage | None = None  # B: 토론방 기반
    price_stage: SentimentStage | None = None  # C: 가격 기반

    # 세부 지표
    news_score: float = 0.0  # 뉴스 감성 점수
    discussion_sentiment_ratio: float = 0.0  # 토론방 공감 비율
    rsi: float = 50.0  # RSI
    volume_ratio: float = 1.0  # 거래량 비율


class SentimentReader:
    """
    심리 분석가 페르소나 (A+B+C 통합 분석)

    세 가지 소스를 종합하여 대중 심리 단계를 파악합니다:

    A. 뉴스 기반 (News-based)
       - 뉴스 수량 (관심도 proxy)
       - 헤드라인 어조 (긍정/부정/중립)

    B. 토론방 기반 (Discussion-based)
       - 네이버 금융 토론방 글
       - 공감/비공감 비율
       - 글 내용 분석

    C. 가격 기반 (Price-based)
       - RSI (과매수/과매도)
       - 최근 수익률
       - 거래량 변화

    핵심 지표:
    - 어조 분석 (긍정/부정/중립)
    - 관심도 측정 (글/댓글 수)
    - 과열/침체 판단
    """

    # 심리 단계별 키워드 (규칙 기반 폴백용)
    FEAR_KEYWORDS = [
        "포기", "손절", "물타기", "물렸", "반토막", "떡락",
        "망했", "끝났", "위험", "폭락", "대주주 매도",
        "관리종목", "상폐", "거래정지", "퇴출",
    ]

    DOUBT_KEYWORDS = [
        "저점", "바닥", "반등", "조심", "관망", "기다려",
        "아직", "불안", "확인", "지켜봐", "리스크",
        "매수 타이밍", "분할 매수", "테스트",
    ]

    CONVICTION_KEYWORDS = [
        "상승", "급등", "돌파", "신고가", "대박", "좋아",
        "매수", "올라가", "추천", "갑니다", "가즈아",
        "수익", "익절", "상한가", "5% 상승",
    ]

    EUPHORIA_KEYWORDS = [
        "떡상", "로켓", "우주", "대폭등", "몇 배", "10배",
        "100만", "1000만", "억대", "인생역전", "전재산",
        "빚내서", "영끌", "풀매수", "물량", "개미털기 없음",
        "조정 없이", "무한 상승", "국민주", "국장 자랑",
    ]

    # 부정 어조 키워드
    NEGATIVE_KEYWORDS = [
        "사기", "작전", "세력", "개미털기", "폭락",
        "손해", "물타기", "반토막", "하락", "매도",
    ]

    # 긍정 어조 키워드
    POSITIVE_KEYWORDS = [
        "좋아", "추천", "상승", "급등", "대박", "수익",
        "기대", "호재", "매수", "갑니다", "최고",
    ]

    SYSTEM_PROMPT = """당신은 주식 커뮤니티 심리 분석 전문가입니다.
수천 개의 글을 읽어온 경험으로 대중 심리를 정확히 읽어냅니다.

심리 단계:
- 공포(FEAR): 모두가 포기하고 관심이 없는 상태. 바닥권 신호.
- 의심(DOUBT): 반신반의하며 소수만 관심을 갖는 초기 상태.
- 확신(CONVICTION): 점점 확신이 커지고 관심도가 상승하는 중기.
- 환희(EUPHORIA): 모두가 낙관적이고 과열된 상태. 고점 경고.

분석 시 고려사항:
1. 글의 어조 (긍정/부정/중립)
2. 관심도 (언급량, 댓글 수)
3. 기대감 수준 (현실적 vs 비현실적)
4. 과열 징후 (무리한 목표가, 영끌 언급 등)

반드시 다음 JSON 형식으로 응답하세요:
{
    "stage": "FEAR" or "DOUBT" or "CONVICTION" or "EUPHORIA",
    "confidence": 0.0 ~ 1.0,
    "interest_level": 0.0 ~ 1.0,
    "tone_score": -1.0 ~ 1.0,
    "sample_opinions": ["대표 의견 1", "대표 의견 2"],
    "reasoning": "판단 근거 (2-3문장)"
}"""

    def __init__(self, llm_client: OllamaClient | None = None):
        self.logger = get_logger(self.__class__.__name__)
        self.llm_client = llm_client

    def analyze(
        self,
        stock_code: str,
        stock_name: str,
        # A: 뉴스 기반 입력
        news_headlines: list[str] | None = None,
        news_count: int = 0,
        # B: 토론방 기반 입력
        community_posts: list[str] | None = None,
        discussion_sentiment_ratio: float = 0.0,  # -1 ~ 1
        discussion_likes: int = 0,
        discussion_dislikes: int = 0,
        # C: 가격 기반 입력
        rsi: float = 50.0,
        return_1w: float = 0.0,  # 1주 수익률 (%)
        return_1m: float = 0.0,  # 1개월 수익률 (%)
        volume_ratio: float = 1.0,  # 거래량 비율 (최근/평균)
        # Legacy 호환용 (deprecated)
        blog_posts: list[str] | None = None,
        comments: list[str] | None = None,
        mention_count: int = 0,
    ) -> SentimentAnalysis:
        """
        종목 심리 분석 (A+B+C 통합)

        Args:
            stock_code: 종목코드
            stock_name: 종목명

            # A: 뉴스 기반
            news_headlines: 뉴스 헤드라인 리스트
            news_count: 뉴스 수

            # B: 토론방 기반
            community_posts: 토론방 글 제목/내용
            discussion_sentiment_ratio: 토론방 공감 비율 (-1 ~ 1)
            discussion_likes: 토론방 총 공감 수
            discussion_dislikes: 토론방 총 비공감 수

            # C: 가격 기반
            rsi: RSI 지표 (0~100)
            return_1w: 1주 수익률 (%)
            return_1m: 1개월 수익률 (%)
            volume_ratio: 거래량 비율 (최근/평균)

        Returns:
            SentimentAnalysis
        """
        # Legacy 호환: mention_count가 있으면 news_count로 사용
        if mention_count > 0 and news_count == 0:
            news_count = mention_count

        community_posts = community_posts or []
        news_headlines = news_headlines or []
        blog_posts = blog_posts or []
        comments = comments or []

        # ============================================================
        # A+B+C 개별 분석
        # ============================================================

        # A: 뉴스 기반 분석
        news_stage, news_score = self._analyze_news_based(
            news_headlines, news_count
        )

        # B: 토론방 기반 분석
        discussion_stage, discussion_score = self._analyze_discussion_based(
            community_posts, discussion_sentiment_ratio,
            discussion_likes, discussion_dislikes
        )

        # C: 가격 기반 분석
        price_stage, price_score = self._analyze_price_based(
            rsi, return_1w, return_1m, volume_ratio
        )

        # ============================================================
        # 종합 분석 (가중 평균)
        # ============================================================

        # 데이터 존재 여부에 따른 가중치 결정
        weights = {
            "news": 0.3 if news_headlines or news_count > 0 else 0.0,
            "discussion": 0.4 if community_posts or discussion_likes + discussion_dislikes > 0 else 0.0,
            "price": 0.3,  # 가격 데이터는 항상 존재
        }

        # 가중치 정규화
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        # 종합 점수 계산 (0: FEAR, 1: DOUBT, 2: CONVICTION, 3: EUPHORIA)
        stage_to_score = {
            SentimentStage.FEAR: 0,
            SentimentStage.DOUBT: 1,
            SentimentStage.CONVICTION: 2,
            SentimentStage.EUPHORIA: 3,
        }

        combined_score = (
            stage_to_score[news_stage] * weights["news"] +
            stage_to_score[discussion_stage] * weights["discussion"] +
            stage_to_score[price_stage] * weights["price"]
        )

        # 종합 단계 결정
        if combined_score < 0.75:
            final_stage = SentimentStage.FEAR
        elif combined_score < 1.5:
            final_stage = SentimentStage.DOUBT
        elif combined_score < 2.25:
            final_stage = SentimentStage.CONVICTION
        else:
            final_stage = SentimentStage.EUPHORIA

        # 관심도 계산 (뉴스 수 + 토론 글 수 기반)
        total_posts = news_count + len(community_posts)
        interest_level = min(1.0, total_posts / 50)

        # 어조 점수 (토론방 + 뉴스)
        tone_score = (news_score + discussion_score) / 2

        # 신뢰도 계산
        data_sources = sum([
            1 if news_headlines or news_count > 0 else 0,
            1 if community_posts or discussion_likes + discussion_dislikes > 0 else 0,
            1,  # 가격 데이터
        ])
        confidence = 0.4 + (data_sources * 0.2)

        # 대표 의견 추출
        sample_opinions = self._extract_sample_opinions(
            community_posts + (news_headlines or []),
            final_stage
        )

        # 판단 근거 생성
        reasoning = self._generate_combined_reasoning(
            final_stage, news_stage, discussion_stage, price_stage,
            interest_level, tone_score, rsi, volume_ratio
        )

        return SentimentAnalysis(
            stock_code=stock_code,
            stock_name=stock_name,
            sentiment_stage=final_stage,
            confidence=round(confidence, 2),
            interest_level=round(interest_level, 2),
            tone_score=round(tone_score, 2),
            sample_opinions=sample_opinions,
            reasoning=reasoning,
            # A+B+C 개별 결과
            news_stage=news_stage,
            discussion_stage=discussion_stage,
            price_stage=price_stage,
            news_score=round(news_score, 2),
            discussion_sentiment_ratio=round(discussion_sentiment_ratio, 3),
            rsi=round(rsi, 1),
            volume_ratio=round(volume_ratio, 2),
        )

    def _analyze_news_based(
        self,
        headlines: list[str],
        count: int
    ) -> tuple[SentimentStage, float]:
        """
        A: 뉴스 기반 분석

        Returns:
            (SentimentStage, 어조 점수)
        """
        if not headlines and count == 0:
            # 뉴스 없음 = 관심 없음 = FEAR
            return SentimentStage.FEAR, 0.0

        # 어조 분석
        combined_text = " ".join(headlines).lower()

        positive_count = sum(1 for kw in self.POSITIVE_KEYWORDS if kw in combined_text)
        negative_count = sum(1 for kw in self.NEGATIVE_KEYWORDS if kw in combined_text)

        total = positive_count + negative_count
        if total > 0:
            tone_score = (positive_count - negative_count) / total
        else:
            tone_score = 0.0

        # 관심도 (뉴스 수 기반)
        interest = min(1.0, count / 30)

        # 단계 결정
        if interest < 0.2:
            stage = SentimentStage.FEAR
        elif tone_score < -0.3:
            stage = SentimentStage.FEAR
        elif tone_score < 0.2:
            stage = SentimentStage.DOUBT
        elif tone_score < 0.5:
            stage = SentimentStage.CONVICTION
        else:
            stage = SentimentStage.EUPHORIA

        return stage, tone_score

    def _analyze_discussion_based(
        self,
        posts: list[str],
        sentiment_ratio: float,
        likes: int,
        dislikes: int
    ) -> tuple[SentimentStage, float]:
        """
        B: 토론방 기반 분석

        Returns:
            (SentimentStage, 감성 점수)
        """
        if not posts and likes + dislikes == 0:
            # 데이터 없음 = 관심 없음 = DOUBT (중립)
            return SentimentStage.DOUBT, 0.0

        # 글 내용 분석
        combined_text = " ".join(posts).lower()

        fear_score = sum(1 for kw in self.FEAR_KEYWORDS if kw in combined_text)
        euphoria_score = sum(1 for kw in self.EUPHORIA_KEYWORDS if kw in combined_text)
        conviction_score = sum(1 for kw in self.CONVICTION_KEYWORDS if kw in combined_text)
        doubt_score = sum(1 for kw in self.DOUBT_KEYWORDS if kw in combined_text)

        # 공감 비율 반영
        # sentiment_ratio: -1 (비공감 우세) ~ 1 (공감 우세)

        # 단계 결정 (키워드 + 공감 비율 조합)
        if fear_score > euphoria_score + 2 or sentiment_ratio < -0.3:
            stage = SentimentStage.FEAR
        elif euphoria_score > 3 or sentiment_ratio > 0.6:
            stage = SentimentStage.EUPHORIA
        elif conviction_score > doubt_score or sentiment_ratio > 0.2:
            stage = SentimentStage.CONVICTION
        else:
            stage = SentimentStage.DOUBT

        return stage, sentiment_ratio

    def _analyze_price_based(
        self,
        rsi: float,
        return_1w: float,
        return_1m: float,
        volume_ratio: float
    ) -> tuple[SentimentStage, float]:
        """
        C: 가격 기반 분석

        Returns:
            (SentimentStage, 종합 점수)
        """
        # RSI 기반 과매수/과매도
        # RSI < 30: 과매도 (FEAR)
        # RSI > 70: 과매수 (EUPHORIA 위험)

        # 수익률 + 거래량으로 모멘텀 판단
        momentum = return_1w * 0.4 + return_1m * 0.3

        # 거래량 급증은 관심도 증가 신호
        volume_signal = 0
        if volume_ratio > 2.0:
            volume_signal = 1  # 급증
        elif volume_ratio > 1.5:
            volume_signal = 0.5  # 증가
        elif volume_ratio < 0.5:
            volume_signal = -0.5  # 급감 (관심 저하)

        # 단계 결정
        if rsi < 30:
            # 과매도 = 공포
            stage = SentimentStage.FEAR
        elif rsi > 70:
            # 과매수 = 과열
            if momentum > 5:
                stage = SentimentStage.EUPHORIA
            else:
                stage = SentimentStage.CONVICTION
        elif momentum > 10:
            # 강한 상승 모멘텀
            stage = SentimentStage.CONVICTION if volume_ratio > 1.2 else SentimentStage.DOUBT
        elif momentum < -10:
            # 강한 하락 모멘텀
            stage = SentimentStage.FEAR
        else:
            # 중립
            stage = SentimentStage.DOUBT

        # 종합 점수 (모멘텀 + RSI 기반)
        price_score = (rsi - 50) / 50  # -1 ~ 1

        return stage, price_score

    def _generate_combined_reasoning(
        self,
        final_stage: SentimentStage,
        news_stage: SentimentStage,
        discussion_stage: SentimentStage,
        price_stage: SentimentStage,
        interest_level: float,
        tone_score: float,
        rsi: float,
        volume_ratio: float
    ) -> str:
        """A+B+C 종합 판단 근거 생성"""
        parts = []

        # A: 뉴스
        parts.append(f"[A-뉴스] {news_stage.value}")

        # B: 토론방
        parts.append(f"[B-토론] {discussion_stage.value}")

        # C: 가격
        rsi_desc = "과매도" if rsi < 30 else "과매수" if rsi > 70 else "중립"
        parts.append(f"[C-가격] {price_stage.value} (RSI {rsi:.0f} {rsi_desc})")

        # 관심도
        if interest_level < 0.2:
            parts.append("관심도 매우 낮음.")
        elif interest_level < 0.5:
            parts.append("관심도 보통.")
        else:
            parts.append("관심도 높음.")

        # 거래량
        if volume_ratio > 1.5:
            parts.append("거래량 급증 주의.")

        # 최종 판단
        stage_desc = {
            SentimentStage.FEAR: "공포 단계 - 저점 매수 기회 가능성.",
            SentimentStage.DOUBT: "의심 단계 - 관망 또는 분할 매수 권장.",
            SentimentStage.CONVICTION: "확신 단계 - 상승 추세, 익절 고려.",
            SentimentStage.EUPHORIA: "환희 단계 - 과열 주의, 리스크 관리 필요.",
        }
        parts.append(stage_desc[final_stage])

        return " ".join(parts)

    def analyze_legacy(
        self,
        stock_code: str,
        stock_name: str,
        community_posts: list[str] | None = None,
        blog_posts: list[str] | None = None,
        comments: list[str] | None = None,
        mention_count: int = 0,
    ) -> SentimentAnalysis:
        """
        레거시 분석 (기존 인터페이스 호환용)
        """
        return self.analyze(
            stock_code=stock_code,
            stock_name=stock_name,
            community_posts=community_posts,
            blog_posts=blog_posts,
            comments=comments,
            mention_count=mention_count,
        )

    def _analyze_with_llm(
        self,
        stock_code: str,
        stock_name: str,
        community_posts: list[str],
        blog_posts: list[str],
        comments: list[str],
        mention_count: int
    ) -> SentimentAnalysis:
        """LLM 기반 분석"""
        content_parts = []

        if community_posts:
            content_parts.append(
                f"[커뮤니티 글]\n" + "\n".join(f"- {p[:100]}" for p in community_posts[:15])
            )

        if blog_posts:
            content_parts.append(
                f"[블로그]\n" + "\n".join(f"- {p[:100]}" for p in blog_posts[:10])
            )

        if comments:
            content_parts.append(
                f"[댓글]\n" + "\n".join(f"- {c[:50]}" for c in comments[:20])
            )

        if not content_parts:
            # 데이터 없음 = 관심 없음 = 공포/의심 단계
            return SentimentAnalysis(
                stock_code=stock_code,
                stock_name=stock_name,
                sentiment_stage=SentimentStage.FEAR,
                confidence=0.6,
                interest_level=0.0,
                tone_score=0.0,
                reasoning="커뮤니티 데이터가 없어 관심도가 매우 낮은 것으로 판단됩니다."
            )

        prompt = f"""종목: {stock_name} ({stock_code})
언급량: {mention_count}개

{chr(10).join(content_parts)}

위 커뮤니티 반응을 분석하여 심리 단계를 평가하세요."""

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
    ) -> SentimentAnalysis:
        """LLM 응답 파싱"""
        import json
        import re

        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())

                stage_map = {
                    "FEAR": SentimentStage.FEAR,
                    "DOUBT": SentimentStage.DOUBT,
                    "CONVICTION": SentimentStage.CONVICTION,
                    "EUPHORIA": SentimentStage.EUPHORIA,
                    "공포": SentimentStage.FEAR,
                    "의심": SentimentStage.DOUBT,
                    "확신": SentimentStage.CONVICTION,
                    "환희": SentimentStage.EUPHORIA,
                }

                return SentimentAnalysis(
                    stock_code=stock_code,
                    stock_name=stock_name,
                    sentiment_stage=stage_map.get(
                        data.get("stage", "DOUBT"), SentimentStage.DOUBT
                    ),
                    confidence=float(data.get("confidence", 0.7)),
                    interest_level=float(data.get("interest_level", 0.5)),
                    tone_score=float(data.get("tone_score", 0.0)),
                    sample_opinions=data.get("sample_opinions", []),
                    reasoning=data.get("reasoning", "")
                )
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.warning(f"JSON 파싱 실패: {e}")

        # 파싱 실패 시 텍스트에서 추출
        stage = SentimentStage.DOUBT
        if "FEAR" in response or "공포" in response:
            stage = SentimentStage.FEAR
        elif "EUPHORIA" in response or "환희" in response:
            stage = SentimentStage.EUPHORIA
        elif "CONVICTION" in response or "확신" in response:
            stage = SentimentStage.CONVICTION

        return SentimentAnalysis(
            stock_code=stock_code,
            stock_name=stock_name,
            sentiment_stage=stage,
            confidence=0.5,
            interest_level=0.5,
            tone_score=0.0,
            reasoning=response[:200]
        )

    def _analyze_with_rules(
        self,
        stock_code: str,
        stock_name: str,
        community_posts: list[str],
        blog_posts: list[str],
        comments: list[str],
        mention_count: int
    ) -> SentimentAnalysis:
        """규칙 기반 분석 (폴백)"""
        all_text = " ".join(community_posts + blog_posts + comments).lower()

        # 관심도 계산
        total_posts = len(community_posts) + len(blog_posts) + len(comments)
        interest_level = min(1.0, total_posts / 50)  # 50개 이상이면 최대

        if mention_count > 0:
            interest_level = max(interest_level, min(1.0, mention_count / 100))

        # 어조 점수 계산
        positive_count = sum(1 for kw in self.POSITIVE_KEYWORDS if kw in all_text)
        negative_count = sum(1 for kw in self.NEGATIVE_KEYWORDS if kw in all_text)
        total_tone = positive_count + negative_count

        if total_tone > 0:
            tone_score = (positive_count - negative_count) / total_tone
        else:
            tone_score = 0.0

        # 단계별 키워드 매칭
        fear_score = sum(1 for kw in self.FEAR_KEYWORDS if kw in all_text)
        doubt_score = sum(1 for kw in self.DOUBT_KEYWORDS if kw in all_text)
        conviction_score = sum(1 for kw in self.CONVICTION_KEYWORDS if kw in all_text)
        euphoria_score = sum(1 for kw in self.EUPHORIA_KEYWORDS if kw in all_text)

        # 심리 단계 결정
        scores = {
            SentimentStage.FEAR: fear_score,
            SentimentStage.DOUBT: doubt_score,
            SentimentStage.CONVICTION: conviction_score,
            SentimentStage.EUPHORIA: euphoria_score,
        }

        # 관심도가 매우 낮으면 FEAR
        if interest_level < 0.1:
            stage = SentimentStage.FEAR
            confidence = 0.7
        else:
            # 최대 점수 단계 선택
            max_stage = max(scores, key=scores.get)
            max_score = scores[max_stage]

            if max_score == 0:
                # 키워드 매칭 없으면 관심도 기반 추정
                if interest_level < 0.3:
                    stage = SentimentStage.DOUBT
                elif interest_level < 0.7:
                    stage = SentimentStage.CONVICTION
                else:
                    stage = SentimentStage.EUPHORIA
                confidence = 0.5
            else:
                stage = max_stage
                confidence = min(0.9, 0.5 + max_score * 0.1)

        # 과열 징후 체크 (강제 EUPHORIA 격상)
        if euphoria_score >= 3 or interest_level >= 0.9:
            if tone_score > 0.7:  # 긍정 어조 압도적
                stage = SentimentStage.EUPHORIA
                confidence = 0.8

        # 대표 의견 추출
        sample_opinions = self._extract_sample_opinions(
            community_posts + blog_posts + comments, stage
        )

        reasoning = self._generate_reasoning(stage, interest_level, tone_score)

        return SentimentAnalysis(
            stock_code=stock_code,
            stock_name=stock_name,
            sentiment_stage=stage,
            confidence=confidence,
            interest_level=interest_level,
            tone_score=tone_score,
            sample_opinions=sample_opinions,
            reasoning=reasoning
        )

    def _extract_sample_opinions(
        self,
        texts: list[str],
        stage: SentimentStage
    ) -> list[str]:
        """대표 의견 추출"""
        if not texts:
            return []

        # 단계별 관련 키워드
        stage_keywords = {
            SentimentStage.FEAR: self.FEAR_KEYWORDS,
            SentimentStage.DOUBT: self.DOUBT_KEYWORDS,
            SentimentStage.CONVICTION: self.CONVICTION_KEYWORDS,
            SentimentStage.EUPHORIA: self.EUPHORIA_KEYWORDS,
        }

        keywords = stage_keywords.get(stage, [])
        relevant = []

        for text in texts:
            text_lower = text.lower()
            if any(kw in text_lower for kw in keywords):
                relevant.append(text[:100])

        return relevant[:3]  # 최대 3개

    def _generate_reasoning(
        self,
        stage: SentimentStage,
        interest_level: float,
        tone_score: float
    ) -> str:
        """판단 근거 생성"""
        parts = []

        # 관심도
        if interest_level < 0.2:
            parts.append("관심도가 매우 낮습니다.")
        elif interest_level < 0.5:
            parts.append("관심도가 보통 수준입니다.")
        elif interest_level < 0.8:
            parts.append("관심도가 높은 편입니다.")
        else:
            parts.append("관심도가 매우 높습니다.")

        # 어조
        if tone_score < -0.3:
            parts.append("부정적 의견이 다수입니다.")
        elif tone_score > 0.3:
            parts.append("긍정적 의견이 다수입니다.")
        else:
            parts.append("의견이 혼재되어 있습니다.")

        # 단계별 설명
        stage_desc = {
            SentimentStage.FEAR: "공포 단계로, 바닥권일 가능성이 있습니다.",
            SentimentStage.DOUBT: "의심 단계로, 초기 관심 단계입니다.",
            SentimentStage.CONVICTION: "확신 단계로, 상승 중입니다.",
            SentimentStage.EUPHORIA: "환희 단계로, 과열 주의가 필요합니다.",
        }
        parts.append(stage_desc[stage])

        return " ".join(parts)

    def analyze_batch(
        self,
        stocks_data: list[dict[str, Any]]
    ) -> list[SentimentAnalysis]:
        """
        복수 종목 일괄 분석

        Args:
            stocks_data: 종목 데이터 리스트

        Returns:
            SentimentAnalysis 리스트
        """
        results = []

        for data in stocks_data:
            result = self.analyze(
                stock_code=data["stock_code"],
                stock_name=data.get("stock_name", ""),
                community_posts=data.get("community_posts", []),
                blog_posts=data.get("blog_posts", []),
                comments=data.get("comments", []),
                mention_count=data.get("mention_count", 0)
            )
            results.append(result)

        return results
