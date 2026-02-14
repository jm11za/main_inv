"""
리포트 생성기

Claude를 활용한 분석 리포트 생성

새 아키텍처 v2.0:
- 단계별 결과(StageSaver)를 기반으로 종합 리포트 생성
- 섹터 중심의 Top-Down 분석 결과 포맷팅
- 텔레그램 및 텍스트 리포트 지원
"""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.logger import get_logger
from src.core.config import get_config
from src.core.interfaces import Recommendation, MaterialGrade, SentimentStage


@dataclass
class StockRecommendation:
    """종목 추천 정보"""
    stock_code: str
    stock_name: str
    recommendation: Recommendation
    score: float
    tier: int
    material_grade: str
    sentiment_stage: str
    key_factors: list[str] = field(default_factory=list)
    risk_warnings: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class AnalysisReport:
    """분석 리포트"""
    report_id: str
    generated_at: datetime
    market_date: str

    # 추천 종목
    strong_buys: list[StockRecommendation] = field(default_factory=list)
    buys: list[StockRecommendation] = field(default_factory=list)
    watches: list[StockRecommendation] = field(default_factory=list)
    avoids: list[StockRecommendation] = field(default_factory=list)

    # 시장 요약
    market_summary: str = ""
    sector_highlights: list[str] = field(default_factory=list)
    risk_alerts: list[str] = field(default_factory=list)

    # 메타
    total_analyzed: int = 0
    processing_time: float = 0.0

    def get_all_recommendations(self) -> list[StockRecommendation]:
        """모든 추천 종목 반환"""
        return self.strong_buys + self.buys + self.watches

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "market_date": self.market_date,
            "strong_buys": [r.stock_code for r in self.strong_buys],
            "buys": [r.stock_code for r in self.buys],
            "watches": [r.stock_code for r in self.watches],
            "avoids": [r.stock_code for r in self.avoids],
            "market_summary": self.market_summary,
        }


class ReportGenerator:
    """
    Claude 기반 분석 리포트 생성기

    사용법:
        generator = ReportGenerator()

        # 파이프라인 결과로 리포트 생성
        report = generator.generate_report(pipeline_result)

        # 텍스트 리포트
        text = generator.format_text_report(report)

        # Telegram용 요약
        summary = generator.format_telegram_message(report)
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.config = get_config()
        self._claude_client = None

    def _get_claude_client(self):
        """Claude CLI 클라이언트 (Lazy init)"""
        if self._claude_client is None:
            try:
                from src.llm import ClaudeCliClient
                self._claude_client = ClaudeCliClient(timeout=120)
            except Exception as e:
                self.logger.warning(f"Claude 클라이언트 초기화 실패: {e}")
        return self._claude_client

    def generate_report(
        self,
        pipeline_result: Any,
        market_date: str | None = None
    ) -> AnalysisReport:
        """
        파이프라인 결과로 분석 리포트 생성

        Args:
            pipeline_result: Pipeline.run_full() 결과
            market_date: 시장 기준일 (없으면 오늘)

        Returns:
            AnalysisReport
        """
        if market_date is None:
            market_date = datetime.now().strftime("%Y-%m-%d")

        report = AnalysisReport(
            report_id=f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            generated_at=datetime.now(),
            market_date=market_date,
            total_analyzed=len(pipeline_result.stage_results) if pipeline_result else 0,
            processing_time=pipeline_result.duration_seconds if pipeline_result else 0,
        )

        # 추천 종목 분류
        recommendations = pipeline_result.final_recommendations if pipeline_result else []

        for rec in recommendations:
            stock_rec = self._convert_to_recommendation(rec)

            if stock_rec.recommendation == Recommendation.STRONG_BUY:
                report.strong_buys.append(stock_rec)
            elif stock_rec.recommendation == Recommendation.BUY:
                report.buys.append(stock_rec)
            elif stock_rec.recommendation == Recommendation.WATCH:
                report.watches.append(stock_rec)
            else:
                report.avoids.append(stock_rec)

        # Claude로 시장 요약 생성
        report.market_summary = self._generate_market_summary(report)

        return report

    def _convert_to_recommendation(self, rec: dict) -> StockRecommendation:
        """dict를 StockRecommendation으로 변환"""
        action = rec.get("action", rec.get("recommendation", "WATCH"))

        if isinstance(action, str):
            recommendation = Recommendation[action] if action in Recommendation.__members__ else Recommendation.WATCH
        else:
            recommendation = action

        return StockRecommendation(
            stock_code=rec.get("stock_code", ""),
            stock_name=rec.get("stock_name", ""),
            recommendation=recommendation,
            score=rec.get("score", rec.get("total_score", 0)),
            tier=rec.get("tier", 2),
            material_grade=rec.get("material_grade", "C"),
            sentiment_stage=rec.get("sentiment_stage", "의심"),
            key_factors=rec.get("key_factors", []),
            risk_warnings=rec.get("risk_warnings", []),
            reason=rec.get("reason", ""),
        )

    def _generate_market_summary(self, report: AnalysisReport) -> str:
        """Claude로 시장 요약 생성"""
        client = self._get_claude_client()

        if not client:
            return self._generate_fallback_summary(report)

        # 프롬프트 구성
        strong_buy_list = ", ".join([f"{r.stock_name}({r.stock_code})" for r in report.strong_buys[:5]])
        buy_list = ", ".join([f"{r.stock_name}({r.stock_code})" for r in report.buys[:5]])

        prompt = f"""오늘({report.market_date}) 한국 주식 시장 분석 결과를 간결하게 요약해줘.

분석 결과:
- STRONG_BUY 종목: {strong_buy_list or "없음"}
- BUY 종목: {buy_list or "없음"}
- 분석 종목 수: {report.total_analyzed}개

주요 특징과 투자 포인트를 3문장 이내로 요약해줘.
전문적이면서도 명확하게 작성해."""

        try:
            summary = client.generate(prompt)
            return summary.strip()
        except Exception as e:
            self.logger.warning(f"Claude 요약 생성 실패: {e}")
            return self._generate_fallback_summary(report)

    def _generate_fallback_summary(self, report: AnalysisReport) -> str:
        """폴백 요약 생성"""
        parts = []

        if report.strong_buys:
            parts.append(f"STRONG_BUY {len(report.strong_buys)}개 종목 발굴")
        if report.buys:
            parts.append(f"BUY {len(report.buys)}개 종목 발굴")
        if report.watches:
            parts.append(f"WATCH {len(report.watches)}개 종목 관찰 필요")

        if not parts:
            return "오늘 분석에서 특별한 매수 신호가 감지되지 않았습니다."

        return " / ".join(parts)

    def format_text_report(self, report: AnalysisReport) -> str:
        """텍스트 형식 리포트 생성"""
        lines = [
            "=" * 60,
            f"📊 일일 분석 리포트 - {report.market_date}",
            "=" * 60,
            "",
            f"📌 요약: {report.market_summary}",
            "",
        ]

        # STRONG_BUY
        if report.strong_buys:
            lines.append("🔥 STRONG_BUY")
            lines.append("-" * 40)
            for rec in report.strong_buys:
                lines.append(self._format_stock_line(rec))
            lines.append("")

        # BUY
        if report.buys:
            lines.append("✅ BUY")
            lines.append("-" * 40)
            for rec in report.buys:
                lines.append(self._format_stock_line(rec))
            lines.append("")

        # WATCH
        if report.watches:
            lines.append("👀 WATCH")
            lines.append("-" * 40)
            for rec in report.watches[:5]:  # 상위 5개만
                lines.append(self._format_stock_line(rec))
            lines.append("")

        # 메타 정보
        lines.extend([
            "-" * 60,
            f"분석 종목: {report.total_analyzed}개 | 소요 시간: {report.processing_time:.1f}초",
            f"생성 시간: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
        ])

        return "\n".join(lines)

    def _format_stock_line(self, rec: StockRecommendation) -> str:
        """종목 한 줄 포맷"""
        factors = ", ".join(rec.key_factors[:2]) if rec.key_factors else ""
        return f"  • {rec.stock_name}({rec.stock_code}) - 점수:{rec.score:.0f} Tier:{rec.tier} [{factors}]"

    def format_telegram_message(self, report: AnalysisReport) -> str:
        """Telegram용 메시지 포맷"""
        lines = [
            f"📊 *일일 분석 리포트*",
            f"📅 {report.market_date}",
            "",
        ]

        # 요약
        if report.market_summary:
            lines.append(f"📌 {report.market_summary}")
            lines.append("")

        # STRONG_BUY
        if report.strong_buys:
            lines.append("🔥 *STRONG BUY*")
            for rec in report.strong_buys[:3]:
                lines.append(f"  • {rec.stock_name} ({rec.stock_code})")
                if rec.key_factors:
                    lines.append(f"    └ {', '.join(rec.key_factors[:2])}")
            lines.append("")

        # BUY
        if report.buys:
            lines.append("✅ *BUY*")
            for rec in report.buys[:5]:
                lines.append(f"  • {rec.stock_name} ({rec.stock_code})")
            lines.append("")

        # WATCH 요약
        if report.watches:
            lines.append(f"👀 *WATCH*: {len(report.watches)}개 종목")
            lines.append("")

        # 푸터
        lines.append(f"⏱ 분석 {report.total_analyzed}종목 / {report.processing_time:.0f}초")

        return "\n".join(lines)

    def generate_stock_detail_report(
        self,
        stock_code: str,
        analysis_data: dict
    ) -> str:
        """
        개별 종목 상세 리포트 생성 (Claude 활용)

        Args:
            stock_code: 종목코드
            analysis_data: 분석 데이터

        Returns:
            상세 분석 리포트 텍스트
        """
        client = self._get_claude_client()

        if not client:
            return self._generate_fallback_detail(stock_code, analysis_data)

        prompt = f"""다음 종목의 투자 분석 리포트를 작성해줘.

종목코드: {stock_code}
종목명: {analysis_data.get('stock_name', '')}
추천등급: {analysis_data.get('recommendation', '')}
점수: {analysis_data.get('score', 0)}

재료 분석:
- 등급: {analysis_data.get('material_grade', '')}
- 주요 이슈: {analysis_data.get('key_factors', [])}

심리 분석:
- 단계: {analysis_data.get('sentiment_stage', '')}
- 관심도: {analysis_data.get('interest_level', '')}

리스크:
{analysis_data.get('risk_warnings', [])}

전문적인 애널리스트 관점에서 200자 이내로 투자 의견을 작성해줘.
형식: 투자의견, 핵심 포인트, 주의사항 순서로."""

        try:
            detail = client.generate(prompt)
            return detail.strip()
        except Exception as e:
            self.logger.warning(f"상세 리포트 생성 실패: {e}")
            return self._generate_fallback_detail(stock_code, analysis_data)

    def _generate_fallback_detail(self, stock_code: str, data: dict) -> str:
        """폴백 상세 리포트"""
        return f"""
[{data.get('stock_name', stock_code)}] {data.get('recommendation', 'N/A')}
- 점수: {data.get('score', 0):.0f}
- 재료등급: {data.get('material_grade', 'N/A')}
- 심리단계: {data.get('sentiment_stage', 'N/A')}
""".strip()

    # =========================================================================
    # 새 아키텍처 v2.0 지원 메서드
    # =========================================================================

    def generate_from_stages(self, stage_data: dict[str, Any]) -> AnalysisReport:
        """
        단계별 결과로부터 분석 리포트 생성

        Args:
            stage_data: StageSaver.aggregate_all_stages() 결과

        Returns:
            AnalysisReport
        """
        market_date = stage_data.get("date", datetime.now().strftime("%Y-%m-%d"))

        report = AnalysisReport(
            report_id=f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            generated_at=datetime.now(),
            market_date=market_date,
        )

        # 05_stock_verify 결과에서 추천 종목 추출
        verify_stage = stage_data.get("stages", {}).get("05_stock_verify", {})
        verify_results = verify_stage.get("results", [])

        if isinstance(verify_results, list):
            for result in verify_results:
                stock_rec = self._convert_verify_result(result)
                if stock_rec:
                    if stock_rec.recommendation == Recommendation.STRONG_BUY:
                        report.strong_buys.append(stock_rec)
                    elif stock_rec.recommendation == Recommendation.BUY:
                        report.buys.append(stock_rec)
                    elif stock_rec.recommendation == Recommendation.WATCH:
                        report.watches.append(stock_rec)
                    else:
                        report.avoids.append(stock_rec)

        # 요약 정보
        summary = stage_data.get("summary", {})
        report.total_analyzed = summary.get("total_stocks_analyzed", 0)

        # 섹터 하이라이트 추출
        priority_stage = stage_data.get("stages", {}).get("03_sector_priority", {})
        priority_results = priority_stage.get("results", [])
        if isinstance(priority_results, list):
            for result in priority_results[:3]:  # 상위 3개 섹터
                if isinstance(result, dict) and result.get("is_selected"):
                    sector = result.get("theme_name", "") or result.get("sector", "")
                    outlook = result.get("llm_outlook", "")[:50] if result.get("llm_outlook") else ""
                    if sector:
                        report.sector_highlights.append(f"{sector}: {outlook}" if outlook else sector)

        # 시장 요약 생성
        report.market_summary = self._generate_sector_summary(stage_data)

        return report

    def _convert_verify_result(self, result: dict) -> StockRecommendation | None:
        """verify 결과를 StockRecommendation으로 변환"""
        if not isinstance(result, dict):
            return None

        rec_value = result.get("recommendation", "WATCH")
        if isinstance(rec_value, str):
            try:
                recommendation = Recommendation[rec_value]
            except KeyError:
                recommendation = Recommendation.WATCH
        else:
            recommendation = rec_value if isinstance(rec_value, Recommendation) else Recommendation.WATCH

        return StockRecommendation(
            stock_code=result.get("stock_code", ""),
            stock_name=result.get("stock_name", ""),
            recommendation=recommendation,
            score=result.get("total_score", 0),
            tier=1 if result.get("sector_rank", 99) <= 2 else 2,
            material_grade=result.get("material_grade", "C"),
            sentiment_stage=result.get("sentiment_stage", "의심"),
            key_factors=result.get("decision_factors", []),
            risk_warnings=result.get("risk_warnings", []),
            reason=result.get("investment_thesis", ""),
        )

    def _generate_sector_summary(self, stage_data: dict) -> str:
        """섹터 기반 시장 요약 생성"""
        client = self._get_claude_client()

        # 섹터 우선순위 정보
        priority_stage = stage_data.get("stages", {}).get("03_sector_priority", {})
        priority_results = priority_stage.get("results", [])

        selected_sectors = []
        if isinstance(priority_results, list):
            for r in priority_results:
                if isinstance(r, dict) and r.get("is_selected"):
                    selected_sectors.append({
                        "sector": r.get("theme_name", "") or r.get("sector", ""),
                        "rank": r.get("rank", 0),
                        "score": r.get("score", 0),
                    })

        # 추천 종목 정보
        summary = stage_data.get("summary", {})
        rec_counts = summary.get("recommendations", {})

        if not client:
            return self._generate_fallback_sector_summary(selected_sectors, rec_counts)

        sector_list = ", ".join([s["sector"] for s in selected_sectors[:5]])

        prompt = f"""오늘({stage_data.get('date', '')}) 한국 주식 시장 분석 결과를 간결하게 요약해줘.

선정된 상위 섹터: {sector_list or "없음"}
STRONG_BUY: {rec_counts.get('STRONG_BUY', 0)}개
BUY: {rec_counts.get('BUY', 0)}개
WATCH: {rec_counts.get('WATCH', 0)}개

Top-Down 섹터 분석 관점에서 오늘의 투자 포인트를 3문장 이내로 요약해줘.
전문적이면서도 명확하게 작성해."""

        try:
            summary_text = client.generate(prompt)
            return summary_text.strip()
        except Exception as e:
            self.logger.warning(f"Claude 요약 생성 실패: {e}")
            return self._generate_fallback_sector_summary(selected_sectors, rec_counts)

    def _generate_fallback_sector_summary(
        self,
        selected_sectors: list[dict],
        rec_counts: dict
    ) -> str:
        """폴백 섹터 요약"""
        parts = []

        if selected_sectors:
            top_sectors = ", ".join([s["sector"] for s in selected_sectors[:3]])
            parts.append(f"상위 섹터: {top_sectors}")

        if rec_counts.get("STRONG_BUY", 0) > 0:
            parts.append(f"STRONG_BUY {rec_counts['STRONG_BUY']}개 발굴")
        if rec_counts.get("BUY", 0) > 0:
            parts.append(f"BUY {rec_counts['BUY']}개 발굴")

        if not parts:
            return "오늘 분석에서 특별한 매수 신호가 감지되지 않았습니다."

        return " / ".join(parts)

    def format_sector_report(self, stage_data: dict[str, Any]) -> str:
        """섹터 중심 텍스트 리포트 생성"""
        lines = [
            "=" * 70,
            f"📊 섹터 분석 리포트 - {stage_data.get('date', '')}",
            "=" * 70,
            "",
        ]

        # 1. 섹터 우선순위
        priority_stage = stage_data.get("stages", {}).get("03_sector_priority", {})
        priority_results = priority_stage.get("results", [])

        if priority_results:
            lines.append("🏆 섹터 우선순위")
            lines.append("-" * 50)
            for r in priority_results[:10]:
                if isinstance(r, dict):
                    status = "✅" if r.get("is_selected") else "❌" if r.get("is_excluded") else "⏸"
                    sector = r.get("theme_name", "") or r.get("sector", "")
                    rank = r.get("rank", 0)
                    score = r.get("score", 0)
                    lines.append(f"  {status} {rank}위. {sector} (점수: {score:.1f})")
            lines.append("")

        # 2. 선정 종목
        verify_stage = stage_data.get("stages", {}).get("05_stock_verify", {})
        verify_results = verify_stage.get("results", [])

        # 추천별 분류
        strong_buys = [r for r in verify_results if isinstance(r, dict) and r.get("recommendation") == "STRONG_BUY"]
        buys = [r for r in verify_results if isinstance(r, dict) and r.get("recommendation") == "BUY"]
        watches = [r for r in verify_results if isinstance(r, dict) and r.get("recommendation") == "WATCH"]

        if strong_buys:
            lines.append("🔥 STRONG_BUY")
            lines.append("-" * 50)
            for r in strong_buys:
                lines.append(self._format_stock_detail_line(r))
            lines.append("")

        if buys:
            lines.append("✅ BUY")
            lines.append("-" * 50)
            for r in buys:
                lines.append(self._format_stock_detail_line(r))
            lines.append("")

        if watches:
            lines.append("👀 WATCH")
            lines.append("-" * 50)
            for r in watches[:5]:  # 상위 5개만
                lines.append(self._format_stock_detail_line(r))
            lines.append("")

        # 3. 요약 통계
        summary = stage_data.get("summary", {})
        lines.extend([
            "-" * 70,
            f"📈 분석 통계",
            f"  - 분석 종목: {summary.get('total_stocks_analyzed', 0)}개",
            f"  - 분석 섹터: {summary.get('total_sectors_analyzed', 0)}개",
            f"  - 선정 섹터: {summary.get('sectors_selected', 0)}개",
            f"  - 선정 종목: {summary.get('stocks_selected', 0)}개",
            "",
            f"⏰ 생성 시간: {stage_data.get('generated_at', '')}",
            "=" * 70,
        ])

        return "\n".join(lines)

    def _format_stock_detail_line(self, result: dict) -> str:
        """종목 상세 라인 포맷"""
        name = result.get("stock_name", "")
        code = result.get("stock_code", "")
        sector = result.get("sector", "")
        score = result.get("total_score", 0)
        material = result.get("material_grade", "")
        sentiment = result.get("sentiment_stage", "")
        factors = result.get("decision_factors", [])

        factor_str = ", ".join(factors[:2]) if factors else ""
        return f"  • {name}({code}) | {sector} | 점수:{score:.0f} | 재료:{material} 심리:{sentiment}\n    └ {factor_str}"

    def format_telegram_from_stages(self, stage_data: dict[str, Any]) -> str:
        """단계 결과로부터 텔레그램 메시지 생성"""
        lines = [
            f"📊 *섹터 분석 리포트*",
            f"📅 {stage_data.get('date', '')}",
            "",
        ]

        # 상위 섹터
        priority_stage = stage_data.get("stages", {}).get("03_sector_priority", {})
        priority_results = priority_stage.get("results", [])
        selected_sectors = [r for r in priority_results if isinstance(r, dict) and r.get("is_selected")]

        if selected_sectors:
            lines.append("🏆 *상위 섹터*")
            for r in selected_sectors[:3]:
                sector = r.get("theme_name", "") or r.get("sector", "")
                lines.append(f"  • {sector}")
            lines.append("")

        # 추천 종목
        verify_stage = stage_data.get("stages", {}).get("05_stock_verify", {})
        verify_results = verify_stage.get("results", [])

        strong_buys = [r for r in verify_results if isinstance(r, dict) and r.get("recommendation") == "STRONG_BUY"]
        buys = [r for r in verify_results if isinstance(r, dict) and r.get("recommendation") == "BUY"]

        if strong_buys:
            lines.append("🔥 *STRONG BUY*")
            for r in strong_buys[:3]:
                name = r.get("stock_name", "")
                code = r.get("stock_code", "")
                thesis = r.get("investment_thesis", "")[:60] if r.get("investment_thesis") else ""
                lines.append(f"  • {name} (`{code}`)")
                if thesis:
                    lines.append(f"    └ {thesis}")
            lines.append("")

        if buys:
            lines.append("✅ *BUY*")
            for r in buys[:5]:
                name = r.get("stock_name", "")
                code = r.get("stock_code", "")
                lines.append(f"  • {name} (`{code}`)")
            lines.append("")

        # 요약 통계
        summary = stage_data.get("summary", {})
        rec = summary.get("recommendations", {})

        lines.append(f"📈 분석 {summary.get('total_stocks_analyzed', 0)}종목")
        lines.append(f"   STRONG\\_BUY:{rec.get('STRONG_BUY', 0)} BUY:{rec.get('BUY', 0)} WATCH:{rec.get('WATCH', 0)}")

        return "\n".join(lines)
