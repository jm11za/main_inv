"""
Output Layer 테스트

새 아키텍처 v2.0:
- StageSaver 테스트
- ReportGenerator 새 메서드 테스트
- TelegramNotifier 테스트
"""
import os
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.output.stage_saver import StageSaver
from src.output.report_generator import (
    ReportGenerator,
    AnalysisReport,
    StockRecommendation,
)
from src.output.telegram_notifier import TelegramNotifier
from src.core.interfaces import Recommendation


class TestStageSaver:
    """StageSaver 테스트"""

    @pytest.fixture
    def temp_dir(self):
        """임시 디렉토리 생성"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def saver(self, temp_dir):
        """StageSaver 인스턴스"""
        return StageSaver(base_dir=temp_dir, date_str="2026-02-01")

    def test_init(self, saver, temp_dir):
        """초기화 및 디렉토리 생성"""
        assert saver.base_dir == temp_dir
        assert saver.date_str == "2026-02-01"

        # 디렉토리 생성 확인
        assert (temp_dir / "stages").exists()
        assert (temp_dir / "reports" / "daily").exists()
        assert (temp_dir / "reports" / "telegram").exists()
        assert (temp_dir / "logs").exists()

    def test_save_stage_list(self, saver):
        """리스트 결과 저장"""
        results = [
            {"stock_code": "005930", "stock_name": "삼성전자", "primary_sector": "반도체"},
            {"stock_code": "000660", "stock_name": "SK하이닉스", "primary_sector": "반도체"},
        ]

        file_path = saver.save_stage("01_sector_classify", results)

        assert file_path.exists()
        assert "2026-02-01" in file_path.name

        # 저장된 내용 확인
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["stage"] == "01_sector_classify"
        assert data["count"] == 2
        assert len(data["results"]) == 2

    def test_save_stage_dict(self, saver):
        """딕셔너리 결과 저장"""
        results = {
            "selected": ["반도체", "2차전지"],
            "excluded": ["게임"],
            "items": [{"sector": "반도체", "rank": 1}],
        }

        file_path = saver.save_stage("03_sector_priority", results)

        assert file_path.exists()

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "selected" in data["results"]

    def test_save_stage_with_metadata(self, saver):
        """메타데이터 포함 저장"""
        results = [{"stock_code": "005930"}]
        metadata = {"model": "deepseek-r1:14b", "elapsed_time": 30.5}

        file_path = saver.save_stage("02_sector_type", results, metadata=metadata)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["metadata"]["model"] == "deepseek-r1:14b"
        assert data["metadata"]["elapsed_time"] == 30.5

    def test_load_stage(self, saver):
        """저장된 결과 로드"""
        results = [{"stock_code": "005930", "is_selected": True}]
        saver.save_stage("04_stock_selection", results)

        loaded = saver.load_stage("04_stock_selection")

        assert loaded is not None
        assert loaded["count"] == 1
        assert loaded["results"][0]["stock_code"] == "005930"

    def test_load_stage_not_found(self, saver):
        """없는 파일 로드"""
        result = saver.load_stage("99_not_exists")
        assert result is None

    def test_aggregate_all_stages(self, saver):
        """모든 단계 집계"""
        # 테스트 데이터 저장
        saver.save_stage("01_sector_classify", [
            {"stock_code": "005930", "primary_sector": "반도체"},
        ])
        saver.save_stage("03_sector_priority", [
            {"sector": "반도체", "is_selected": True, "rank": 1},
        ])
        saver.save_stage("05_stock_verify", [
            {"stock_code": "005930", "recommendation": "BUY", "is_selected": True},
        ])

        aggregated = saver.aggregate_all_stages()

        assert aggregated["date"] == "2026-02-01"
        assert "stages" in aggregated
        assert "01_sector_classify" in aggregated["stages"]
        assert "summary" in aggregated

    def test_save_daily_report(self, saver):
        """일일 리포트 저장"""
        report_data = {
            "date": "2026-02-01",
            "summary": {"total": 10},
        }

        file_path = saver.save_daily_report(report_data)

        assert file_path.exists()
        assert "full.json" in file_path.name

    def test_save_telegram_message(self, saver):
        """텔레그램 메시지 저장"""
        message = "STRONG_BUY: 삼성전자"

        file_path = saver.save_telegram_message(message)

        assert file_path.exists()
        assert file_path.suffix == ".txt"

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "삼성전자" in content

    def test_list_saved_stages(self, saver):
        """저장된 단계 목록"""
        saver.save_stage("01_sector_classify", [])
        saver.save_stage("03_sector_priority", [])

        saved = saver.list_saved_stages()

        assert "01_sector_classify" in saved
        assert "03_sector_priority" in saved
        assert "02_sector_type" not in saved

    def test_to_serializable_enum(self, saver):
        """Enum 직렬화"""
        from src.core.interfaces import MaterialGrade

        result = saver._to_serializable(MaterialGrade.S)
        assert result == "S"

    def test_to_serializable_datetime(self, saver):
        """datetime 직렬화"""
        dt = datetime(2026, 2, 1, 12, 0, 0)
        result = saver._to_serializable(dt)
        assert "2026-02-01" in result

    def test_to_serializable_dataclass(self, saver):
        """dataclass 직렬화 (to_dict 메서드)"""
        class MockDataclass:
            def to_dict(self):
                return {"key": "value"}

        obj = MockDataclass()
        result = saver._to_serializable(obj)
        assert result == {"key": "value"}


class TestStockRecommendation:
    """StockRecommendation 테스트"""

    def test_create_recommendation(self):
        """추천 생성"""
        rec = StockRecommendation(
            stock_code="005930",
            stock_name="삼성전자",
            recommendation=Recommendation.BUY,
            score=75.5,
            tier=1,
            material_grade="A",
            sentiment_stage="확신",
        )

        assert rec.stock_code == "005930"
        assert rec.recommendation == Recommendation.BUY
        assert rec.score == 75.5


class TestAnalysisReport:
    """AnalysisReport 테스트"""

    def test_create_report(self):
        """리포트 생성"""
        report = AnalysisReport(
            report_id="RPT-001",
            generated_at=datetime.now(),
            market_date="2025-01-31",
        )

        assert report.report_id == "RPT-001"
        assert len(report.strong_buys) == 0

    def test_get_all_recommendations(self):
        """전체 추천 조회"""
        rec1 = StockRecommendation(
            stock_code="005930",
            stock_name="삼성전자",
            recommendation=Recommendation.STRONG_BUY,
            score=85,
            tier=1,
            material_grade="S",
            sentiment_stage="확신",
        )
        rec2 = StockRecommendation(
            stock_code="000660",
            stock_name="SK하이닉스",
            recommendation=Recommendation.BUY,
            score=75,
            tier=2,
            material_grade="A",
            sentiment_stage="의심",
        )

        report = AnalysisReport(
            report_id="RPT-001",
            generated_at=datetime.now(),
            market_date="2025-01-31",
            strong_buys=[rec1],
            buys=[rec2],
        )

        all_recs = report.get_all_recommendations()
        assert len(all_recs) == 2

    def test_to_dict(self):
        """dict 변환"""
        report = AnalysisReport(
            report_id="RPT-001",
            generated_at=datetime.now(),
            market_date="2025-01-31",
            market_summary="테스트 요약",
        )

        d = report.to_dict()

        assert d["report_id"] == "RPT-001"
        assert d["market_summary"] == "테스트 요약"


class TestReportGenerator:
    """ReportGenerator 테스트"""

    def test_init(self):
        """초기화"""
        generator = ReportGenerator()
        assert generator is not None

    def test_convert_to_recommendation(self):
        """dict → StockRecommendation 변환"""
        generator = ReportGenerator()

        rec_dict = {
            "stock_code": "005930",
            "stock_name": "삼성전자",
            "action": "BUY",
            "score": 75.0,
            "tier": 1,
            "material_grade": "A",
            "sentiment_stage": "확신",
            "key_factors": ["HBM 수주", "AI 호황"],
        }

        rec = generator._convert_to_recommendation(rec_dict)

        assert rec.stock_code == "005930"
        assert rec.recommendation == Recommendation.BUY
        assert rec.score == 75.0
        assert len(rec.key_factors) == 2

    def test_format_stock_line(self):
        """종목 라인 포맷"""
        generator = ReportGenerator()

        rec = StockRecommendation(
            stock_code="005930",
            stock_name="삼성전자",
            recommendation=Recommendation.BUY,
            score=75.0,
            tier=1,
            material_grade="A",
            sentiment_stage="확신",
            key_factors=["HBM", "AI"],
        )

        line = generator._format_stock_line(rec)

        assert "삼성전자" in line
        assert "005930" in line
        assert "75" in line

    def test_format_telegram_message(self):
        """Telegram 메시지 포맷"""
        generator = ReportGenerator()

        rec = StockRecommendation(
            stock_code="005930",
            stock_name="삼성전자",
            recommendation=Recommendation.STRONG_BUY,
            score=85.0,
            tier=1,
            material_grade="S",
            sentiment_stage="확신",
            key_factors=["HBM", "AI"],
        )

        report = AnalysisReport(
            report_id="RPT-001",
            generated_at=datetime.now(),
            market_date="2025-01-31",
            strong_buys=[rec],
            market_summary="테스트 요약",
        )

        message = generator.format_telegram_message(report)

        assert "STRONG BUY" in message
        assert "삼성전자" in message
        assert "테스트 요약" in message

    def test_generate_fallback_summary(self):
        """폴백 요약 생성"""
        generator = ReportGenerator()

        rec = StockRecommendation(
            stock_code="005930",
            stock_name="삼성전자",
            recommendation=Recommendation.BUY,
            score=75.0,
            tier=1,
            material_grade="A",
            sentiment_stage="확신",
        )

        report = AnalysisReport(
            report_id="RPT-001",
            generated_at=datetime.now(),
            market_date="2025-01-31",
            buys=[rec],
        )

        summary = generator._generate_fallback_summary(report)

        assert "BUY" in summary
        assert "1" in summary

    @patch.object(ReportGenerator, "_get_claude_client")
    def test_generate_market_summary_with_claude(self, mock_get_claude):
        """Claude로 시장 요약 생성"""
        mock_client = Mock()
        mock_client.generate.return_value = "AI 반도체 섹터 강세. 삼성전자 매수 추천."
        mock_get_claude.return_value = mock_client

        generator = ReportGenerator()

        report = AnalysisReport(
            report_id="RPT-001",
            generated_at=datetime.now(),
            market_date="2025-01-31",
        )

        summary = generator._generate_market_summary(report)

        assert "AI" in summary or "반도체" in summary or len(summary) > 0

    def test_generate_from_stages(self):
        """단계 결과로부터 리포트 생성"""
        generator = ReportGenerator()

        stage_data = {
            "date": "2026-02-01",
            "generated_at": "2026-02-01T12:00:00",
            "stages": {
                "03_sector_priority": {
                    "results": [
                        {"sector": "반도체", "rank": 1, "is_selected": True, "llm_outlook": "AI 수요 급증"},
                        {"sector": "2차전지", "rank": 2, "is_selected": True, "llm_outlook": ""},
                    ]
                },
                "05_stock_verify": {
                    "results": [
                        {
                            "stock_code": "005930",
                            "stock_name": "삼성전자",
                            "recommendation": "STRONG_BUY",
                            "total_score": 85.0,
                            "sector_rank": 1,
                            "material_grade": "S",
                            "sentiment_stage": "의심",
                            "decision_factors": ["HBM 수주", "AI 호황"],
                            "investment_thesis": "반도체 대장주 매수 적기",
                        },
                        {
                            "stock_code": "000660",
                            "stock_name": "SK하이닉스",
                            "recommendation": "BUY",
                            "total_score": 75.0,
                            "material_grade": "A",
                            "sentiment_stage": "확신",
                        },
                    ]
                },
            },
            "summary": {
                "total_stocks_analyzed": 100,
                "recommendations": {"STRONG_BUY": 1, "BUY": 1},
            },
        }

        report = generator.generate_from_stages(stage_data)

        assert report.market_date == "2026-02-01"
        assert len(report.strong_buys) == 1
        assert len(report.buys) == 1
        assert report.strong_buys[0].stock_code == "005930"
        assert report.total_analyzed == 100

    def test_convert_verify_result(self):
        """verify 결과 변환"""
        generator = ReportGenerator()

        result = {
            "stock_code": "005930",
            "stock_name": "삼성전자",
            "recommendation": "STRONG_BUY",
            "total_score": 85.0,
            "sector_rank": 1,
            "material_grade": "S",
            "sentiment_stage": "의심",
            "decision_factors": ["HBM"],
            "risk_warnings": ["환율 리스크"],
            "investment_thesis": "적극 매수",
        }

        rec = generator._convert_verify_result(result)

        assert rec.stock_code == "005930"
        assert rec.recommendation == Recommendation.STRONG_BUY
        assert rec.tier == 1  # sector_rank <= 2 -> tier 1
        assert "HBM" in rec.key_factors

    def test_format_sector_report(self):
        """섹터 중심 리포트 포맷"""
        generator = ReportGenerator()

        stage_data = {
            "date": "2026-02-01",
            "generated_at": "2026-02-01T12:00:00",
            "stages": {
                "03_sector_priority": {
                    "results": [
                        {"sector": "반도체", "rank": 1, "score": 85.0, "is_selected": True},
                        {"sector": "2차전지", "rank": 2, "score": 75.0, "is_selected": True},
                    ]
                },
                "05_stock_verify": {
                    "results": [
                        {
                            "stock_code": "005930",
                            "stock_name": "삼성전자",
                            "recommendation": "STRONG_BUY",
                            "sector": "반도체",
                            "total_score": 85.0,
                            "material_grade": "S",
                            "sentiment_stage": "의심",
                            "decision_factors": ["HBM 수주"],
                        },
                    ]
                },
            },
            "summary": {
                "total_stocks_analyzed": 100,
                "total_sectors_analyzed": 10,
                "sectors_selected": 2,
                "stocks_selected": 5,
            },
        }

        text = generator.format_sector_report(stage_data)

        assert "섹터 분석 리포트" in text
        assert "반도체" in text
        assert "삼성전자" in text
        assert "STRONG_BUY" in text

    def test_format_telegram_from_stages(self):
        """단계 결과로 텔레그램 메시지 생성"""
        generator = ReportGenerator()

        stage_data = {
            "date": "2026-02-01",
            "stages": {
                "03_sector_priority": {
                    "results": [
                        {"sector": "반도체", "is_selected": True},
                    ]
                },
                "05_stock_verify": {
                    "results": [
                        {
                            "stock_code": "005930",
                            "stock_name": "삼성전자",
                            "recommendation": "STRONG_BUY",
                            "investment_thesis": "적극 매수 추천",
                        },
                    ]
                },
            },
            "summary": {
                "total_stocks_analyzed": 50,
                "recommendations": {"STRONG_BUY": 1, "BUY": 2, "WATCH": 5},
            },
        }

        message = generator.format_telegram_from_stages(stage_data)

        assert "섹터 분석 리포트" in message
        assert "반도체" in message
        assert "삼성전자" in message
        assert "STRONG BUY" in message

    def test_generate_fallback_sector_summary(self):
        """폴백 섹터 요약"""
        generator = ReportGenerator()

        selected_sectors = [
            {"sector": "반도체", "rank": 1},
            {"sector": "2차전지", "rank": 2},
        ]
        rec_counts = {"STRONG_BUY": 2, "BUY": 3}

        summary = generator._generate_fallback_sector_summary(selected_sectors, rec_counts)

        assert "반도체" in summary
        assert "STRONG_BUY" in summary


class TestTelegramNotifier:
    """TelegramNotifier 테스트"""

    def test_init(self):
        """초기화"""
        notifier = TelegramNotifier(bot_token="test_token", chat_id="123")
        assert notifier.bot_token == "test_token"
        assert notifier.chat_id == "123"

    def test_is_configured_true(self):
        """설정 완료 확인 (True)"""
        notifier = TelegramNotifier(bot_token="token", chat_id="123")
        assert notifier.is_configured() is True

    @patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""}, clear=False)
    def test_is_configured_false(self):
        """설정 미완료 확인 (False)"""
        notifier = TelegramNotifier(bot_token="", chat_id="")
        notifier.bot_token = ""
        notifier.chat_id = ""
        assert notifier.is_configured() is False

    @patch("src.output.telegram_notifier.requests.Session")
    @patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""}, clear=False)
    def test_send_message_no_token(self, mock_session):
        """토큰 없을 때 발송 건너뜀"""
        notifier = TelegramNotifier(bot_token="", chat_id="123")
        # 명시적으로 토큰 제거
        notifier.bot_token = ""
        result = notifier.send_message("테스트")

        assert result is False

    @patch("src.output.telegram_notifier.requests.Session")
    @patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""}, clear=False)
    def test_send_message_no_chat_id(self, mock_session):
        """Chat ID 없을 때 발송 건너뜀"""
        notifier = TelegramNotifier(bot_token="token", chat_id="")
        notifier.chat_id = ""
        result = notifier.send_message("테스트")

        assert result is False

    @patch.object(TelegramNotifier, "_request")
    def test_send_message_success(self, mock_request):
        """메시지 발송 성공"""
        mock_request.return_value = {"ok": True}

        notifier = TelegramNotifier(bot_token="token", chat_id="123")
        result = notifier.send_message("테스트 메시지")

        assert result is True
        mock_request.assert_called_once()

    @patch.object(TelegramNotifier, "_request")
    def test_send_alert(self, mock_request):
        """알림 발송"""
        mock_request.return_value = {"ok": True}

        notifier = TelegramNotifier(bot_token="token", chat_id="123")
        result = notifier.send_alert("긴급 알림", "내용입니다", level="WARNING")

        assert result is True

    @patch.object(TelegramNotifier, "_request")
    def test_send_stock_alert(self, mock_request):
        """종목 알림 발송"""
        mock_request.return_value = {"ok": True}

        notifier = TelegramNotifier(bot_token="token", chat_id="123")
        result = notifier.send_stock_alert(
            stock_code="005930",
            stock_name="삼성전자",
            alert_type="ENTRY",
            details="매수 진입 시점입니다."
        )

        assert result is True

    @patch.object(TelegramNotifier, "_request")
    def test_test_connection_success(self, mock_request):
        """연결 테스트 성공"""
        mock_request.return_value = {
            "ok": True,
            "result": {"username": "test_bot"}
        }

        notifier = TelegramNotifier(bot_token="token", chat_id="123")
        result = notifier.test_connection()

        assert result is True


class TestOutputIntegration:
    """통합 테스트 (실제 네트워크 호출)"""

    @pytest.mark.skip(reason="실제 네트워크 호출 - 수동 실행")
    def test_real_telegram_connection(self):
        """실제 Telegram 연결 테스트"""
        notifier = TelegramNotifier()

        if notifier.is_configured():
            result = notifier.test_connection()
            print(f"연결 결과: {result}")

    @pytest.mark.skip(reason="실제 네트워크 호출 - 수동 실행")
    def test_get_chat_id(self):
        """Chat ID 조회"""
        notifier = TelegramNotifier()
        chat_id = notifier.get_chat_id_from_updates()
        print(f"Chat ID: {chat_id}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
