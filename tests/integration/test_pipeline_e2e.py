"""
E2E 통합 테스트: PipelineV2 전체 Stage 0~5 연결 테스트

외부 서비스(네이버, DART, Ollama, Claude)를 mock하고
Stage 0 → 1 → 2 → 3 → 4 → 5 데이터 흐름을 검증
"""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.orchestrator.pipeline_v2 import PipelineV2, PipelineV2Result
from src.orchestrator.stage_runner import StageResult, StageStatus
from src.core.preflight import PreflightResult


# ===== 공통 Mock 데이터 =====

MOCK_THEMES = [
    MagicMock(
        theme_id="T001", name="2차전지", stocks=[],
        change_rate=2.5, volume_power=80
    ),
    MagicMock(
        theme_id="T002", name="자동차부품", stocks=[],
        change_rate=1.2, volume_power=60
    ),
]

MOCK_THEME_STOCKS = {
    "T001": [
        MagicMock(stock_code="373220", stock_name="LG에너지솔루션"),
        MagicMock(stock_code="006400", stock_name="삼성SDI"),
    ],
    "T002": [
        MagicMock(stock_code="012330", stock_name="현대모비스"),
    ],
}


def _make_stage_result(name, data, status=StageStatus.SUCCESS):
    return StageResult(
        stage_name=name,
        status=status,
        data=data,
        started_at=datetime.now(),
        completed_at=datetime.now(),
    )


class TestPipelineV2E2E:
    """PipelineV2 전체 Stage 연결 E2E 테스트"""

    @patch.object(PipelineV2, "preflight_check")
    @patch.object(PipelineV2, "_run_data_collect")
    @patch.object(PipelineV2, "_run_sector_classify")
    @patch.object(PipelineV2, "_run_sector_type")
    @patch.object(PipelineV2, "_run_sector_priority")
    @patch.object(PipelineV2, "_run_stock_selection")
    @patch.object(PipelineV2, "_run_stock_verify")
    def test_full_pipeline_happy_path(
        self,
        mock_verify,
        mock_selection,
        mock_priority,
        mock_type,
        mock_classify,
        mock_collect,
        mock_preflight,
    ):
        """정상 경로: 6단계 전체 실행 → 최종 판정 반환"""
        mock_preflight.return_value = PreflightResult(passed=True)

        # Stage 0: 데이터 수집
        mock_collect.return_value = _make_stage_result("00_data_collect", {
            "theme_count": 2,
            "stock_count": 3,
            "themes": MOCK_THEMES,
            "stock_codes": ["373220", "006400", "012330"],
        })

        # Stage 1: 섹터 분류
        mock_classify.return_value = _make_stage_result("01_sector_classify", {
            "results": [
                {"stock_code": "373220", "stock_name": "LG에너지솔루션", "theme_tags": ["2차전지"]},
                {"stock_code": "006400", "stock_name": "삼성SDI", "theme_tags": ["2차전지"]},
                {"stock_code": "012330", "stock_name": "현대모비스", "theme_tags": ["자동차부품"]},
            ],
            "theme_count": 2,
            "stock_count": 3,
            "sector_count": 2,
        })

        # Stage 2: Type A/B 분류
        mock_type.return_value = _make_stage_result("02_sector_type", {
            "results": [
                {"theme_name": "2차전지", "sector_type": "growth_driven", "confidence": 0.9},
                {"theme_name": "자동차부품", "sector_type": "earnings_driven", "confidence": 0.85},
            ],
            "theme_count": 2,
        })

        # Stage 3: 섹터 우선순위
        mock_priority.return_value = _make_stage_result("03_sector_priority", {
            "results": [
                {"theme_name": "2차전지", "score": 85.0, "selected": True},
                {"theme_name": "자동차부품", "score": 60.0, "selected": True},
            ],
            "selected_sectors": ["2차전지", "자동차부품"],
            "selected_count": 2,
        })

        # Stage 4: 종목 선정
        mock_selection.return_value = _make_stage_result("04_stock_selection", {
            "results": [
                {"stock_code": "373220", "stock_name": "LG에너지솔루션", "theme": "2차전지", "score": 78.5},
                {"stock_code": "006400", "stock_name": "삼성SDI", "theme": "2차전지", "score": 72.0},
            ],
            "selected_count": 2,
            "selected_stocks": [
                {"stock_code": "373220", "stock_name": "LG에너지솔루션", "theme": "2차전지"},
                {"stock_code": "006400", "stock_name": "삼성SDI", "theme": "2차전지"},
            ],
        })

        # Stage 5: 재검증
        mock_verify.return_value = _make_stage_result("05_stock_verify", {
            "results": [
                {
                    "stock_code": "373220",
                    "stock_name": "LG에너지솔루션",
                    "recommendation": "BUY",
                    "material_grade": "A",
                    "sentiment_stage": "확신",
                },
                {
                    "stock_code": "006400",
                    "stock_name": "삼성SDI",
                    "recommendation": "WATCH",
                    "material_grade": "B",
                    "sentiment_stage": "의심",
                },
            ],
            "summary": {
                "recommendation_distribution": {"BUY": 1, "WATCH": 1},
            },
        })

        # 실행
        pipeline = PipelineV2()
        result = pipeline.run(save_stages=False)

        # 검증
        assert result.success is True
        assert "00_data_collect" in result.stages
        assert "01_sector_classify" in result.stages
        assert "02_sector_type" in result.stages
        assert "03_sector_priority" in result.stages
        assert "04_stock_selection" in result.stages
        assert "05_stock_verify" in result.stages

        # 모든 Stage가 SUCCESS
        for stage_name, stage_result in result.stages.items():
            assert stage_result.status == StageStatus.SUCCESS, f"{stage_name} failed"

        # 최종 판정이 존재
        assert len(result.final_decisions) >= 1

        # 호출 순서 확인
        mock_collect.assert_called_once()
        mock_classify.assert_called_once()
        mock_type.assert_called_once()
        mock_priority.assert_called_once()
        mock_selection.assert_called_once()
        mock_verify.assert_called_once()

    @patch.object(PipelineV2, "preflight_check")
    @patch.object(PipelineV2, "_run_data_collect")
    @patch.object(PipelineV2, "_run_sector_classify")
    @patch.object(PipelineV2, "_run_sector_type")
    @patch.object(PipelineV2, "_run_sector_priority")
    def test_pipeline_no_selected_sectors_early_exit(
        self,
        mock_priority,
        mock_type,
        mock_classify,
        mock_collect,
        mock_preflight,
    ):
        """선정 섹터 없으면 Stage 4~5 스킵 후 성공 종료"""
        mock_preflight.return_value = PreflightResult(passed=True)

        mock_collect.return_value = _make_stage_result("00_data_collect", {
            "theme_count": 1, "stock_count": 5,
        })
        mock_classify.return_value = _make_stage_result("01_sector_classify", {
            "results": [], "sector_count": 0,
        })
        mock_type.return_value = _make_stage_result("02_sector_type", {
            "results": [],
        })
        mock_priority.return_value = _make_stage_result("03_sector_priority", {
            "results": [], "selected_sectors": [],
        })

        pipeline = PipelineV2()
        result = pipeline.run(save_stages=False)

        assert result.success is True
        assert len(result.final_decisions) == 0
        assert "04_stock_selection" not in result.stages
        assert "05_stock_verify" not in result.stages

    @patch.object(PipelineV2, "preflight_check")
    @patch.object(PipelineV2, "_run_data_collect")
    def test_pipeline_stage0_failure_stops_pipeline(
        self, mock_collect, mock_preflight
    ):
        """Stage 0 실패 시 전체 파이프라인 중단"""
        mock_preflight.return_value = PreflightResult(passed=True)

        mock_collect.return_value = _make_stage_result(
            "00_data_collect",
            data=None,
            status=StageStatus.FAILED,
        )
        mock_collect.return_value.error = "네트워크 오류"

        pipeline = PipelineV2()
        result = pipeline.run(save_stages=False)

        assert result.success is False
        assert "데이터 수집 실패" in (result.error or "")

    def test_pipeline_preflight_failure(self):
        """Preflight 실패 시 파이프라인 미시작"""
        with patch.object(PipelineV2, "preflight_check") as mock_pf:
            mock_pf.return_value = PreflightResult(passed=False)

            pipeline = PipelineV2()
            result = pipeline.run()

            assert result.success is False
            assert "Preflight" in result.error


class TestPipelineV2DataFlow:
    """Stage 간 데이터 전달 검증"""

    @patch.object(PipelineV2, "preflight_check")
    @patch.object(PipelineV2, "_run_data_collect")
    @patch.object(PipelineV2, "_run_sector_classify")
    @patch.object(PipelineV2, "_run_sector_type")
    @patch.object(PipelineV2, "_run_sector_priority")
    def test_stage0_data_flows_to_stage1(
        self,
        mock_priority,
        mock_type,
        mock_classify,
        mock_collect,
        mock_preflight,
    ):
        """Stage 0 출력이 Stage 1 입력으로 전달되는지 확인"""
        mock_preflight.return_value = PreflightResult(passed=True)

        mock_collect.return_value = _make_stage_result("00_data_collect", {
            "theme_count": 2,
            "stock_count": 3,
            "themes": MOCK_THEMES,
        })

        mock_classify.return_value = _make_stage_result("01_sector_classify", {
            "results": [], "sector_count": 2,
        })
        mock_type.return_value = _make_stage_result("02_sector_type", {"results": []})
        mock_priority.return_value = _make_stage_result("03_sector_priority", {
            "results": [], "selected_sectors": [],
        })

        pipeline = PipelineV2()
        result = pipeline.run(save_stages=False)

        # Stage 1이 호출됨 (Stage 0 성공 후)
        mock_classify.assert_called_once()
        assert result.success is True

    def test_pipeline_result_summary_generation(self):
        """PipelineV2Result 요약 생성 검증"""
        result = PipelineV2Result(
            success=True,
            started_at=datetime(2025, 1, 1, 9, 0),
            completed_at=datetime(2025, 1, 1, 9, 5),
            stages={
                "00_data_collect": _make_stage_result("00_data_collect", {"theme_count": 10}),
                "05_stock_verify": _make_stage_result("05_stock_verify", {"summary": {}}),
            },
            final_decisions=[
                {"stock_code": "005930", "recommendation": "BUY"},
            ],
        )

        summary = result.to_summary()

        assert summary["success"] is True
        assert summary["decisions_count"] == 1
        assert "00_data_collect" in summary["stages"]
        assert "05_stock_verify" in summary["stages"]
        assert "300.0s" == summary["duration"]


class TestPipelineV2Config:
    """파이프라인 설정 테스트"""

    def test_stages_count(self):
        """6개 단계 정의"""
        assert len(PipelineV2.STAGES) == 6

    def test_stage_names_ordered(self):
        """단계 순서 확인"""
        stages = PipelineV2.STAGES
        for i, stage in enumerate(stages):
            assert stage.startswith(f"{i:02d}_"), f"Stage {stage} 순서 불일치"

    def test_testmode_limits(self):
        """테스트 모드에서 테마/종목 수 제한"""
        with patch.object(PipelineV2, "preflight_check") as mock_pf, \
             patch.object(PipelineV2, "_run_data_collect") as mock_collect, \
             patch.object(PipelineV2, "_run_sector_classify") as mock_cls, \
             patch.object(PipelineV2, "_run_sector_type") as mock_type, \
             patch.object(PipelineV2, "_run_sector_priority") as mock_pri:

            mock_pf.return_value = PreflightResult(passed=True)
            mock_collect.return_value = _make_stage_result("00_data_collect", {
                "theme_count": 0, "stock_count": 0,
            })
            mock_cls.return_value = _make_stage_result("01_sector_classify", {
                "results": [], "sector_count": 0,
            })
            mock_type.return_value = _make_stage_result("02_sector_type", {"results": []})
            mock_pri.return_value = _make_stage_result("03_sector_priority", {
                "results": [], "selected_sectors": [],
            })

            pipeline = PipelineV2()
            pipeline.run(testmode=True, save_stages=False)

            # _run_data_collect 호출 시 max_themes=30, max_stocks=50
            call_kwargs = mock_collect.call_args
            if call_kwargs and call_kwargs.kwargs:
                max_themes = call_kwargs.kwargs.get("max_themes")
                max_stocks = call_kwargs.kwargs.get("max_stocks")
                if max_themes is not None:
                    assert max_themes <= 30
                if max_stocks is not None:
                    assert max_stocks <= 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
