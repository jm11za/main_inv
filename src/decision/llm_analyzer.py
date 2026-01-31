"""
LLM Analyzer - Dual Persona 분석 조율자

Skeptic과 Sentiment Reader 페르소나를 조율하여
종목 분석을 수행하는 메인 분석기입니다.

Cloud LLM (Claude)을 기본으로 사용하며,
비용 최적화를 위해 상위 N개 종목만 분석합니다.
"""
from typing import Any

from src.core.interfaces import Tier, TrackType
from src.core.config import get_config
from src.core.logger import get_logger
from src.llm.ollama_client import OllamaClient
from src.decision.personas.skeptic import Skeptic, SkepticAnalysis
from src.decision.personas.sentiment import SentimentReader, SentimentAnalysis
from src.decision.decision_engine import DecisionEngine, DecisionResult


class LLMAnalyzer:
    """
    Dual Persona LLM 분석기

    두 개의 상반된 페르소나를 활용하여 균형 잡힌 투자 판정을 수행합니다.

    - Skeptic: 냉철한 애널리스트 (재료 분석)
    - Sentiment Reader: 심리 분석가 (대중 심리 분석)

    사용법:
        analyzer = LLMAnalyzer()

        # 단일 종목 분석
        result = analyzer.analyze(
            stock_code="005930",
            stock_name="삼성전자",
            tier=Tier.TIER_1,
            track_type=TrackType.TRACK_A,
            total_score=85.0,
            news_headlines=["삼성전자 HBM 대규모 수주"],
            community_posts=["삼성전자 갑니다!"]
        )

        # 일괄 분석
        results = analyzer.analyze_batch(stocks_data)
    """

    def __init__(
        self,
        llm_client: OllamaClient | None = None,
        use_llm: bool = True
    ):
        """
        Args:
            llm_client: 사용할 LLM 클라이언트 (None이면 자동 생성)
            use_llm: LLM 사용 여부 (False면 규칙 기반만 사용)
        """
        self.logger = get_logger(self.__class__.__name__)
        self.config = get_config()

        # LLM 클라이언트 설정
        if use_llm and llm_client is None:
            try:
                self.llm_client = OllamaClient()
            except Exception as e:
                self.logger.warning(f"LLM 클라이언트 초기화 실패: {e}")
                self.llm_client = None
        else:
            self.llm_client = llm_client

        # 페르소나 초기화
        self.skeptic = Skeptic(llm_client=self.llm_client)
        self.sentiment_reader = SentimentReader(llm_client=self.llm_client)

        # Decision Engine
        self.decision_engine = DecisionEngine()

        # 설정
        self.top_n_for_llm = self.config.get("decision.top_n_for_llm", 20)

    def analyze(
        self,
        stock_code: str,
        stock_name: str,
        tier: Tier,
        track_type: TrackType,
        total_score: float,
        # Skeptic 입력
        news_headlines: list[str] | None = None,
        announcements: list[str] | None = None,
        ir_content: str | None = None,
        # Sentiment A+B+C 입력
        community_posts: list[str] | None = None,  # B: 토론방 글
        discussion_sentiment_ratio: float = 0.0,  # B: 토론방 공감 비율
        discussion_likes: int = 0,  # B: 토론방 공감 수
        discussion_dislikes: int = 0,  # B: 토론방 비공감 수
        rsi: float = 50.0,  # C: RSI
        return_1w: float = 0.0,  # C: 1주 수익률
        return_1m: float = 0.0,  # C: 1개월 수익률
        volume_ratio: float = 1.0,  # C: 거래량 비율
        # Legacy 호환용 (deprecated)
        blog_posts: list[str] | None = None,
        comments: list[str] | None = None,
        mention_count: int = 0,
    ) -> DecisionResult:
        """
        단일 종목 분석

        Args:
            stock_code: 종목코드
            stock_name: 종목명
            tier: 섹터 Tier
            track_type: Track 타입
            total_score: 스코어링 점수

            # Skeptic 입력
            news_headlines: 뉴스 헤드라인
            announcements: 공시 목록
            ir_content: IR 자료

            # Sentiment A+B+C 입력
            community_posts: 토론방 글 (B)
            discussion_sentiment_ratio: 토론방 공감 비율 (B)
            discussion_likes: 토론방 공감 수 (B)
            discussion_dislikes: 토론방 비공감 수 (B)
            rsi: RSI 지표 (C)
            return_1w: 1주 수익률 (C)
            return_1m: 1개월 수익률 (C)
            volume_ratio: 거래량 비율 (C)

        Returns:
            DecisionResult
        """
        self.logger.info(f"분석 시작: {stock_name} ({stock_code})")

        # 1. Skeptic 분석 (재료)
        skeptic_result = self.skeptic.analyze(
            stock_code=stock_code,
            stock_name=stock_name,
            news_headlines=news_headlines,
            announcements=announcements,
            ir_content=ir_content,
        )

        self.logger.debug(
            f"Skeptic 결과: {skeptic_result.material_grade.value} "
            f"(신뢰도: {skeptic_result.confidence:.2f})"
        )

        # 2. Sentiment 분석 (A+B+C 통합)
        sentiment_result = self.sentiment_reader.analyze(
            stock_code=stock_code,
            stock_name=stock_name,
            # A: 뉴스 기반
            news_headlines=news_headlines,
            news_count=len(news_headlines) if news_headlines else 0,
            # B: 토론방 기반
            community_posts=community_posts,
            discussion_sentiment_ratio=discussion_sentiment_ratio,
            discussion_likes=discussion_likes,
            discussion_dislikes=discussion_dislikes,
            # C: 가격 기반
            rsi=rsi,
            return_1w=return_1w,
            return_1m=return_1m,
            volume_ratio=volume_ratio,
            # Legacy 호환
            blog_posts=blog_posts,
            comments=comments,
            mention_count=mention_count,
        )

        self.logger.debug(
            f"Sentiment 결과: {sentiment_result.sentiment_stage.value} "
            f"(A:{sentiment_result.news_stage}, B:{sentiment_result.discussion_stage}, "
            f"C:{sentiment_result.price_stage}, 관심도: {sentiment_result.interest_level:.2f})"
        )

        # 3. Decision Engine 판정
        decision = self.decision_engine.decide(
            stock_code=stock_code,
            stock_name=stock_name,
            tier=tier,
            track_type=track_type,
            total_score=total_score,
            skeptic_analysis=skeptic_result,
            sentiment_analysis=sentiment_result,
        )

        self.logger.info(
            f"최종 판정: {stock_name} -> {decision.recommendation.value}"
        )

        return decision

    def analyze_batch(
        self,
        stocks_data: list[dict[str, Any]],
        top_n: int | None = None
    ) -> list[DecisionResult]:
        """
        복수 종목 일괄 분석

        비용 최적화를 위해 점수 상위 N개만 LLM 분석합니다.

        Args:
            stocks_data: 종목 데이터 리스트
                [{
                    stock_code, stock_name, tier, track_type, total_score,
                    # Skeptic
                    news_headlines, announcements, ir_content,
                    # Sentiment A+B+C
                    community_posts, discussion_sentiment_ratio,
                    discussion_likes, discussion_dislikes,
                    rsi, return_1w, return_1m, volume_ratio,
                }]
            top_n: 분석할 상위 종목 수 (None이면 설정값 사용)

        Returns:
            DecisionResult 리스트
        """
        n = top_n or self.top_n_for_llm

        # 점수순 정렬
        sorted_stocks = sorted(
            stocks_data,
            key=lambda x: x.get("total_score", 0),
            reverse=True
        )

        # 상위 N개만 분석
        target_stocks = sorted_stocks[:n]

        self.logger.info(f"분석 대상: {len(target_stocks)}개 (전체 {len(stocks_data)}개 중)")

        results = []
        for data in target_stocks:
            try:
                result = self.analyze(
                    stock_code=data["stock_code"],
                    stock_name=data.get("stock_name", ""),
                    tier=data.get("tier", Tier.TIER_3),
                    track_type=data.get("track_type", TrackType.TRACK_A),
                    total_score=data.get("total_score", 0.0),
                    # Skeptic 입력
                    news_headlines=data.get("news_headlines"),
                    announcements=data.get("announcements"),
                    ir_content=data.get("ir_content"),
                    # Sentiment A+B+C 입력
                    community_posts=data.get("community_posts"),
                    discussion_sentiment_ratio=data.get("discussion_sentiment_ratio", 0.0),
                    discussion_likes=data.get("discussion_likes", 0),
                    discussion_dislikes=data.get("discussion_dislikes", 0),
                    rsi=data.get("rsi", 50.0),
                    return_1w=data.get("return_1w", 0.0),
                    return_1m=data.get("return_1m", 0.0),
                    volume_ratio=data.get("volume_ratio", 1.0),
                    # Legacy 호환
                    blog_posts=data.get("blog_posts"),
                    comments=data.get("comments"),
                    mention_count=data.get("mention_count", 0),
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"분석 실패: {data.get('stock_code')} - {e}")

        return results

    def get_top_picks(
        self,
        results: list[DecisionResult],
        n: int = 5
    ) -> list[DecisionResult]:
        """
        최상위 추천 종목 반환

        Args:
            results: 분석 결과 리스트
            n: 반환 개수

        Returns:
            상위 N개 종목
        """
        return self.decision_engine.get_top_recommendations(results, n=n)

    def summarize(self, results: list[DecisionResult]) -> dict:
        """
        분석 결과 요약

        Args:
            results: 분석 결과 리스트

        Returns:
            요약 정보
        """
        return self.decision_engine.summarize_decisions(results)

    def format_report(self, results: list[DecisionResult]) -> str:
        """
        분석 결과 리포트 포맷팅

        Args:
            results: 분석 결과 리스트

        Returns:
            포맷된 리포트 문자열
        """
        lines = []
        lines.append("=" * 60)
        lines.append("         📊 종목 분석 리포트")
        lines.append("=" * 60)

        summary = self.summarize(results)
        lines.append(f"\n[전체 요약]")
        lines.append(f"  분석 종목: {summary['total']}개")
        lines.append(f"  STRONG_BUY: {summary['strong_buy']}개")
        lines.append(f"  BUY: {summary['buy']}개")
        lines.append(f"  WATCH: {summary['watch']}개")
        lines.append(f"  AVOID: {summary['avoid']}개")

        # STRONG_BUY 종목
        strong_buys = [
            r for r in results
            if r.recommendation.value == "STRONG_BUY"
        ]
        if strong_buys:
            lines.append(f"\n[⭐ STRONG_BUY]")
            for r in strong_buys:
                lines.append(f"  • {r.stock_name} ({r.stock_code})")
                lines.append(f"    재료: {r.material_grade.value}급 / 심리: {r.sentiment_stage.value}")
                lines.append(f"    점수: {r.total_score:.1f} / 신뢰도: {r.confidence:.0%}")
                if r.key_factors:
                    lines.append(f"    핵심: {', '.join(r.key_factors[:3])}")

        # BUY 종목
        buys = [r for r in results if r.recommendation.value == "BUY"]
        if buys:
            lines.append(f"\n[✅ BUY]")
            for r in buys[:5]:  # 상위 5개만
                lines.append(f"  • {r.stock_name} ({r.stock_code})")
                lines.append(f"    재료: {r.material_grade.value}급 / 심리: {r.sentiment_stage.value}")

        # AVOID 경고
        avoids = [r for r in results if r.recommendation.value == "AVOID"]
        if avoids:
            lines.append(f"\n[⚠️ AVOID - 주의 종목]")
            for r in avoids:
                lines.append(f"  • {r.stock_name} ({r.stock_code})")
                lines.append(f"    사유: {r.decision_reasoning}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)
