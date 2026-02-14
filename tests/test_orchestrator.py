"""
Orchestrator 테스트

v2.0 PipelineV2 테스트
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.orchestrator.stage_runner import StageRunner, StageResult, StageStatus
from src.orchestrator.pipeline_v2 import PipelineV2, PipelineV2Result


class TestStageResult:
    """StageResult 테스트"""

    def test_create_result(self):
        """결과 생성"""
        result = StageResult(
            stage_name="TestStage",
            status=StageStatus.SUCCESS,
            data={"key": "value"}
        )

        assert result.stage_name == "TestStage"
        assert result.status == StageStatus.SUCCESS
        assert result.data["key"] == "value"

    def test_duration_calculation(self):
        """실행 시간 계산"""
        from datetime import timedelta

        now = datetime.now()
        later = now + timedelta(seconds=5)

        result = StageResult(
            stage_name="TestStage",
            status=StageStatus.SUCCESS,
            started_at=now,
            completed_at=later
        )

        assert abs(result.duration_seconds - 5.0) < 0.1

    def test_to_dict(self):
        """dict 변환"""
        result = StageResult(
            stage_name="TestStage",
            status=StageStatus.FAILED,
            error="Test error"
        )

        d = result.to_dict()

        assert d["stage"] == "TestStage"
        assert d["status"] == "failed"
        assert d["error"] == "Test error"


class TestStageRunner:
    """StageRunner 테스트"""

    def test_init(self):
        """초기화"""
        runner = StageRunner()
        assert runner is not None

    def test_run_stage_failure(self):
        """단계 실패 처리"""
        runner = StageRunner()

        def failing_func():
            raise ValueError("Test error")

        result = runner._run_stage("TestStage", failing_func)

        assert result.status == StageStatus.FAILED
        assert "Test error" in result.error


# =============================================================================
# PipelineV2 테스트 (Top-Down 섹터 중심)
# =============================================================================

class TestPipelineV2Result:
    """PipelineV2Result 테스트"""

    def test_create_result(self):
        """결과 생성"""
        result = PipelineV2Result(
            success=True,
            started_at=datetime.now()
        )

        assert result.success is True
        assert len(result.stages) == 0
        assert len(result.final_decisions) == 0

    def test_duration_calculation(self):
        """실행 시간 계산"""
        from datetime import timedelta

        now = datetime.now()
        later = now + timedelta(seconds=120)

        result = PipelineV2Result(
            success=True,
            started_at=now,
            completed_at=later
        )

        assert abs(result.duration_seconds - 120.0) < 0.1

    def test_to_summary(self):
        """요약 정보 생성"""
        stage1 = StageResult(stage_name="01_sector_classify", status=StageStatus.SUCCESS, data={"results": []})
        stage2 = StageResult(stage_name="02_sector_type", status=StageStatus.SUCCESS, data={"results": []})

        result = PipelineV2Result(
            success=True,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            stages={"01_sector_classify": stage1, "02_sector_type": stage2},
            final_decisions=[{"stock_code": "005930", "recommendation": "BUY"}],
        )

        summary = result.to_summary()

        assert summary["success"] is True
        assert summary["decisions_count"] == 1
        assert "01_sector_classify" in summary["stages"]


class TestPipelineV2:
    """PipelineV2 테스트"""

    def test_init(self):
        """초기화"""
        pipeline = PipelineV2()
        assert pipeline is not None
        assert len(pipeline.STAGES) == 6

    def test_stages_definition(self):
        """단계 정의 확인"""
        expected_stages = [
            "00_data_collect",
            "01_sector_classify",
            "02_sector_type",
            "03_sector_priority",
            "04_stock_selection",
            "05_stock_verify",
        ]

        assert PipelineV2.STAGES == expected_stages

    @patch.object(PipelineV2, "preflight_check")
    @patch.object(PipelineV2, "_run_data_collect")
    def test_run_preflight_failure(self, mock_data_collect, mock_preflight):
        """Preflight 실패 시 중단"""
        from src.core.preflight import PreflightResult

        mock_preflight.return_value = PreflightResult(passed=False)

        pipeline = PipelineV2()
        result = pipeline.run()

        assert result.success is False
        assert "Preflight" in result.error
        mock_data_collect.assert_not_called()

    @patch.object(PipelineV2, "preflight_check")
    @patch.object(PipelineV2, "_run_data_collect")
    @patch.object(PipelineV2, "_run_sector_classify")
    @patch.object(PipelineV2, "_run_sector_type")
    @patch.object(PipelineV2, "_run_sector_priority")
    def test_run_no_selected_sectors(
        self, mock_priority, mock_type, mock_classify, mock_collect, mock_preflight
    ):
        """선정된 섹터 없을 때"""
        from src.core.preflight import PreflightResult

        mock_preflight.return_value = PreflightResult(passed=True)
        mock_collect.return_value = StageResult(
            stage_name="00_data_collect",
            status=StageStatus.SUCCESS,
            data={"theme_count": 1, "stock_count": 10}
        )
        mock_classify.return_value = StageResult(
            stage_name="01_sector_classify",
            status=StageStatus.SUCCESS,
            data={"results": [], "sector_count": 5}
        )
        mock_type.return_value = StageResult(
            stage_name="02_sector_type",
            status=StageStatus.SUCCESS,
            data={"results": []}
        )
        mock_priority.return_value = StageResult(
            stage_name="03_sector_priority",
            status=StageStatus.SUCCESS,
            data={"results": [], "selected_sectors": []}
        )

        pipeline = PipelineV2()
        result = pipeline.run(save_stages=False)

        # 선정된 섹터 없어도 성공으로 종료
        assert result.success is True
        assert len(result.final_decisions) == 0

    def test_generate_summary(self):
        """요약 생성 테스트"""
        pipeline = PipelineV2()

        stage1 = StageResult(
            stage_name="01_sector_classify",
            status=StageStatus.SUCCESS,
            data={"sector_count": 10}
        )
        stage3 = StageResult(
            stage_name="03_sector_priority",
            status=StageStatus.SUCCESS,
            data={"selected_count": 5}
        )
        stage4 = StageResult(
            stage_name="04_stock_selection",
            status=StageStatus.SUCCESS,
            data={"selected_count": 15}
        )
        stage5 = StageResult(
            stage_name="05_stock_verify",
            status=StageStatus.SUCCESS,
            data={"summary": {"recommendation_distribution": {"STRONG_BUY": 2, "BUY": 5}}}
        )

        result = PipelineV2Result(
            success=True,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            stages={
                "01_sector_classify": stage1,
                "03_sector_priority": stage3,
                "04_stock_selection": stage4,
                "05_stock_verify": stage5,
            }
        )

        summary = pipeline._generate_summary(result)

        assert summary["sectors_analyzed"] == 10
        assert summary["sectors_selected"] == 5
        assert summary["stocks_selected"] == 15
        assert summary["recommendations"]["STRONG_BUY"] == 2
        assert summary["recommendations"]["BUY"] == 5

    def test_clear_cache(self):
        """캐시 초기화"""
        pipeline = PipelineV2()
        pipeline._cache["test"] = "value"

        pipeline.clear_cache()

        assert len(pipeline._cache) == 0

    def test_get_cached_data(self):
        """캐시 데이터 반환"""
        pipeline = PipelineV2()
        pipeline._cache["themes"] = ["theme1", "theme2"]

        cached = pipeline.get_cached_data()

        assert "themes" in cached
        assert len(cached["themes"]) == 2


class TestPipelineV2Integration:
    """PipelineV2 통합 테스트"""

    @pytest.mark.skip(reason="실제 네트워크 호출 - 수동 실행")
    def test_run_full_real(self):
        """실제 전체 파이프라인 테스트"""
        pipeline = PipelineV2()
        result = pipeline.run(
            max_themes=3,
            max_stocks=30,
            top_sectors=3,
            top_per_sector=2,
            save_stages=True,
            verbose=True,
        )

        print(f"성공: {result.success}")
        print(f"시간: {result.duration_seconds:.1f}초")
        print(f"최종 판정: {len(result.final_decisions)}개")

        if result.final_decisions:
            for d in result.final_decisions[:5]:
                print(f"  - {d.get('stock_name')}: {d.get('recommendation')}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
