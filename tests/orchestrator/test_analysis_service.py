"""
AnalysisService 테스트
"""
import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.orchestrator.analysis_service import AnalysisService


class TestAnalysisServiceInit:
    """AnalysisService 초기화 테스트"""

    @patch("src.orchestrator.analysis_service.get_database")
    def test_init(self, mock_get_db):
        """초기화 테스트"""
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        service = AnalysisService()

        assert service.db == mock_db
        assert service.logger is not None


class TestSaveAnalysisResult:
    """save_analysis_result 테스트"""

    @patch("src.orchestrator.analysis_service.get_database")
    def test_save_analysis_result_success(self, mock_get_db):
        """분석 결과 저장 성공"""
        # Mock 설정
        mock_session = MagicMock()
        mock_db = MagicMock()
        mock_db.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.session.return_value.__exit__ = MagicMock(return_value=None)
        mock_get_db.return_value = mock_db

        # Mock history ID
        mock_history = MagicMock()
        mock_history.id = 1
        mock_history.run_date = datetime.now()
        mock_session.add = MagicMock()
        mock_session.flush = MagicMock()

        # Pipeline result mock
        pipeline_result = MagicMock()
        pipeline_result.stage_results = []
        pipeline_result.final_recommendations = [
            {"stock_code": "005930", "recommendation": "BUY"},
        ]
        pipeline_result.success = True
        pipeline_result.error = None
        pipeline_result.duration_seconds = 10.5

        service = AnalysisService()

        # _save_analysis_history가 mock_history를 반환하도록 설정
        with patch.object(service, "_save_analysis_history", return_value=mock_history):
            result_id = service.save_analysis_result(
                pipeline_result=pipeline_result,
                tier_results={"반도체": {"tier": 1, "s_flow": 0.5}},
                score_results={"005930": {"total_score": 75, "financial_score": 40}},
                decision_results={"005930": {"recommendation": "BUY"}},
            )

        assert result_id == 1

    @patch("src.orchestrator.analysis_service.get_database")
    def test_save_analysis_result_with_empty_data(self, mock_get_db):
        """빈 데이터로 저장"""
        mock_session = MagicMock()
        mock_db = MagicMock()
        mock_db.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.session.return_value.__exit__ = MagicMock(return_value=None)
        mock_get_db.return_value = mock_db

        mock_history = MagicMock()
        mock_history.id = 2
        mock_history.run_date = datetime.now()

        pipeline_result = MagicMock()
        pipeline_result.stage_results = []
        pipeline_result.final_recommendations = []
        pipeline_result.success = True
        pipeline_result.error = None
        pipeline_result.duration_seconds = 5.0

        service = AnalysisService()

        with patch.object(service, "_save_analysis_history", return_value=mock_history):
            result_id = service.save_analysis_result(pipeline_result)

        assert result_id == 2


class TestSaveAnalysisHistory:
    """_save_analysis_history 테스트"""

    @patch("src.orchestrator.analysis_service.get_database")
    def test_save_history_extracts_tiers(self, mock_get_db):
        """Tier 정보 추출 확인"""
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        service = AnalysisService()

        mock_session = MagicMock()
        mock_session.add = MagicMock()
        mock_session.flush = MagicMock()

        pipeline_result = MagicMock()
        pipeline_result.stage_results = [MagicMock(), MagicMock()]
        pipeline_result.final_recommendations = [
            {"stock_code": "005930", "recommendation": "BUY", "name": "삼성전자"},
            {"stock_code": "000660", "recommendation": "WATCH", "name": "SK하이닉스"},
        ]
        pipeline_result.success = True
        pipeline_result.error = None
        pipeline_result.duration_seconds = 15.0

        tier_results = {
            "반도체": {"tier": 1, "s_flow": 0.6},
            "2차전지": {"tier": 2, "s_flow": 0.4},
            "바이오": {"tier": 3, "s_flow": 0.2},
        }

        # Call the method
        from src.core.models import AnalysisHistoryModel

        with patch("src.orchestrator.analysis_service.AnalysisHistoryModel") as MockModel:
            mock_instance = MagicMock()
            mock_instance.id = 1
            MockModel.return_value = mock_instance

            result = service._save_analysis_history(mock_session, pipeline_result, tier_results)

            # Verify the model was created with correct tier data
            call_kwargs = MockModel.call_args.kwargs
            assert "반도체" in call_kwargs["tier1_sectors"]
            assert "2차전지" in call_kwargs["tier2_sectors"]


class TestSaveScoreHistory:
    """_save_score_history 테스트"""

    @patch("src.orchestrator.analysis_service.get_database")
    def test_save_score_history(self, mock_get_db):
        """종목 점수 히스토리 저장"""
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        service = AnalysisService()
        mock_session = MagicMock()

        analysis_date = datetime.now()
        score_results = {
            "005930": {
                "total_score": 80,
                "financial_score": 45,
                "technical_score": 35,
                "breakdown": {"s_flow": 0.5, "s_trend": 0.3},
            },
        }
        decision_results = {
            "005930": {
                "material_grade": "A",
                "sentiment_stage": "의심",
                "recommendation": "BUY",
            }
        }

        from src.core.models import StockScoreHistoryModel

        with patch("src.orchestrator.analysis_service.StockScoreHistoryModel") as MockModel:
            service._save_score_history(
                mock_session, analysis_date, score_results, decision_results, None
            )

            # Verify model was created
            MockModel.assert_called_once()
            call_kwargs = MockModel.call_args.kwargs
            assert call_kwargs["stock_code"] == "005930"
            assert call_kwargs["total_score"] == 80


class TestUpdateStockLatest:
    """_update_stock_latest 테스트"""

    @patch("src.orchestrator.analysis_service.get_database")
    def test_update_stock_latest(self, mock_get_db):
        """종목 테이블 최신 상태 업데이트"""
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        service = AnalysisService()

        mock_session = MagicMock()
        mock_stock = MagicMock()
        mock_session.get.return_value = mock_stock

        score_results = {
            "005930": {
                "total_score": 85,
                "financial_score": 50,
                "technical_score": 35,
                "breakdown": {"s_flow": 0.6},
            }
        }
        decision_results = {
            "005930": {"recommendation": "BUY", "material_grade": "S"}
        }

        service._update_stock_latest(mock_session, score_results, decision_results, None)

        # Stock model should be updated
        assert mock_stock.total_score == 85
        assert mock_stock.recommendation == "BUY"


class TestGetAnalysisHistory:
    """get_analysis_history 테스트"""

    @patch("src.orchestrator.analysis_service.get_database")
    def test_get_history_default_days(self, mock_get_db):
        """기본 30일 히스토리 조회"""
        mock_session = MagicMock()
        mock_db = MagicMock()
        mock_db.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.session.return_value.__exit__ = MagicMock(return_value=None)
        mock_get_db.return_value = mock_db

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        service = AnalysisService()
        result = service.get_analysis_history()

        assert result == []
        mock_session.execute.assert_called_once()


class TestGetStockScoreHistory:
    """get_stock_score_history 테스트"""

    @patch("src.orchestrator.analysis_service.get_database")
    def test_get_stock_score_history(self, mock_get_db):
        """종목 점수 히스토리 조회"""
        mock_session = MagicMock()
        mock_db = MagicMock()
        mock_db.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.session.return_value.__exit__ = MagicMock(return_value=None)
        mock_get_db.return_value = mock_db

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        service = AnalysisService()
        result = service.get_stock_score_history("005930", days=7)

        assert result == []


class TestGetLatestRecommendations:
    """get_latest_recommendations 테스트"""

    @patch("src.orchestrator.analysis_service.get_database")
    def test_get_buy_recommendations(self, mock_get_db):
        """BUY 추천 종목 조회"""
        mock_session = MagicMock()
        mock_db = MagicMock()
        mock_db.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.session.return_value.__exit__ = MagicMock(return_value=None)
        mock_get_db.return_value = mock_db

        mock_stock1 = MagicMock()
        mock_stock1.stock_code = "005930"
        mock_stock1.recommendation = "BUY"

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_stock1]
        mock_session.execute.return_value = mock_result

        service = AnalysisService()
        result = service.get_latest_recommendations("BUY", limit=10)

        assert len(result) == 1
        assert result[0].stock_code == "005930"


class TestGetSectorTierSummary:
    """get_sector_tier_summary 테스트"""

    @patch("src.orchestrator.analysis_service.get_database")
    def test_get_tier_summary(self, mock_get_db):
        """섹터 Tier 요약"""
        mock_session = MagicMock()
        mock_db = MagicMock()
        mock_db.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.session.return_value.__exit__ = MagicMock(return_value=None)
        mock_get_db.return_value = mock_db

        mock_theme1 = MagicMock()
        mock_theme1.name = "반도체"
        mock_theme1.tier = 1
        mock_theme1.s_flow = 0.6
        mock_theme1.s_breadth = 0.5
        mock_theme1.s_trend = 0.4

        mock_theme2 = MagicMock()
        mock_theme2.name = "바이오"
        mock_theme2.tier = 3
        mock_theme2.s_flow = 0.2
        mock_theme2.s_breadth = 0.3
        mock_theme2.s_trend = 0.2

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_theme1, mock_theme2]
        mock_session.execute.return_value = mock_result

        service = AnalysisService()
        result = service.get_sector_tier_summary()

        assert "tier1" in result
        assert "tier3" in result
        assert len(result["tier1"]) == 1
        assert result["tier1"][0]["name"] == "반도체"
