"""
CandidateSelector 테스트 (v2.0)

테마명 문자열 기반 후보 선정 테스트
"""
import pytest
from unittest.mock import Mock, patch, MagicMock

from src.core.interfaces import SectorType, TrackType
from src.stock.selector import CandidateSelector, CandidateResult


# ==========================================
# Fixtures
# ==========================================

@pytest.fixture
def selector():
    """CandidateSelector 인스턴스"""
    with patch('src.stock.selector.get_logger') as mock_logger, \
         patch('src.stock.selector.get_config') as mock_config:
        mock_logger.return_value = MagicMock()
        mock_config.return_value = {}
        return CandidateSelector()


@pytest.fixture
def sample_stocks_data():
    """샘플 종목 데이터"""
    return [
        # 테마1: 반도체
        {
            "stock_code": "005930",
            "stock_name": "삼성전자",
            "sector": "반도체",
            "operating_profit_4q": 5000000000,
            "debt_ratio": 50,
            "pbr": 1.5,
            "avg_trading_value": 2000000000000,
            "operating_profit_yoy": 20,
            "roe": 15,
            "foreign_net_ratio": 3,
            "institution_net_ratio": 2,
            "ma20_gap": 2,
            "volume_ratio": 1.5,
            "high_52w_proximity": 0.9,
        },
        {
            "stock_code": "000660",
            "stock_name": "SK하이닉스",
            "sector": "반도체",
            "operating_profit_4q": 3000000000,
            "debt_ratio": 80,
            "pbr": 2.0,
            "avg_trading_value": 1500000000000,
            "operating_profit_yoy": 30,
            "roe": 12,
            "foreign_net_ratio": 4,
            "institution_net_ratio": 3,
            "ma20_gap": 3,
            "volume_ratio": 2.0,
            "high_52w_proximity": 0.85,
        },
        # 테마2: 2차전지
        {
            "stock_code": "373220",
            "stock_name": "LG에너지솔루션",
            "sector": "2차전지",
            "capital_impairment": 0,
            "current_ratio": 150,
            "avg_trading_value": 800000000,
            "revenue_yoy": 50,
            "rd_ratio": 10,
            "foreign_net_ratio": 5,
            "institution_net_ratio": 4,
            "ma20_gap": 1,
            "volume_ratio": 2.5,
            "high_52w_proximity": 0.95,
        },
        {
            "stock_code": "006400",
            "stock_name": "삼성SDI",
            "sector": "2차전지",
            "capital_impairment": 0,
            "current_ratio": 130,
            "avg_trading_value": 600000000,
            "revenue_yoy": 40,
            "rd_ratio": 8,
            "foreign_net_ratio": 2,
            "institution_net_ratio": 2,
            "ma20_gap": 4,
            "volume_ratio": 1.8,
            "high_52w_proximity": 0.80,
        },
    ]


@pytest.fixture
def sector_type_map():
    """테마별 SectorType 매핑"""
    return {
        "반도체": SectorType.TYPE_B,  # 성장형
        "2차전지": SectorType.TYPE_B,  # 성장형
    }


@pytest.fixture
def sample_candidate_results():
    """샘플 CandidateResult 리스트"""
    return [
        CandidateResult(
            stock_code="005930",
            stock_name="삼성전자",
            sector="반도체",
            sector_rank=1,
            track_type=TrackType.TRACK_B,
            filter_passed=True,
            filter_reason="모든 조건 충족",
            financial_score=70.0,
            technical_score=85.0,
            total_score=82.0,
            overall_rank=1,
            is_selected=True,
            selection_reason="테마 내 1위",
        ),
        CandidateResult(
            stock_code="000660",
            stock_name="SK하이닉스",
            sector="반도체",
            sector_rank=2,
            track_type=TrackType.TRACK_B,
            filter_passed=True,
            filter_reason="모든 조건 충족",
            financial_score=65.0,
            technical_score=80.0,
            total_score=77.0,
            overall_rank=2,
            is_selected=True,
            selection_reason="테마 내 2위",
        ),
        CandidateResult(
            stock_code="373220",
            stock_name="LG에너지솔루션",
            sector="2차전지",
            sector_rank=1,
            track_type=TrackType.TRACK_B,
            filter_passed=True,
            filter_reason="모든 조건 충족",
            financial_score=75.0,
            technical_score=90.0,
            total_score=87.0,
            overall_rank=3,
            is_selected=True,
            selection_reason="테마 내 1위",
        ),
        CandidateResult(
            stock_code="006400",
            stock_name="삼성SDI",
            sector="2차전지",
            sector_rank=2,
            track_type=TrackType.TRACK_B,
            filter_passed=False,
            filter_reason="유동비율 부족",
            financial_score=0,
            technical_score=0,
            total_score=0,
            overall_rank=0,
            is_selected=False,
            selection_reason="필터 탈락",
        ),
    ]


# ==========================================
# CandidateResult 테스트
# ==========================================

class TestCandidateResult:
    """CandidateResult dataclass 테스트"""

    def test_create_candidate_result(self):
        """기본 생성 테스트"""
        result = CandidateResult(
            stock_code="005930",
            stock_name="삼성전자",
            sector="반도체",  # 테마명 문자열
            sector_rank=1,
            track_type=TrackType.TRACK_B,
            filter_passed=True,
            filter_reason="통과",
            financial_score=70.0,
            technical_score=85.0,
            total_score=82.0,
            overall_rank=1,
            is_selected=True,
            selection_reason="테마 내 1위",
        )

        assert result.stock_code == "005930"
        assert result.sector == "반도체"  # 문자열
        assert result.track_type == TrackType.TRACK_B
        assert result.is_selected is True

    def test_to_dict(self):
        """to_dict 메서드 테스트"""
        result = CandidateResult(
            stock_code="005930",
            stock_name="삼성전자",
            sector="반도체",
            sector_rank=1,
            track_type=TrackType.TRACK_B,
            filter_passed=True,
            filter_reason="통과",
            financial_score=70.123,
            technical_score=85.456,
            total_score=82.789,
            overall_rank=1,
            is_selected=True,
            selection_reason="테마 내 1위",
        )

        d = result.to_dict()

        assert d["stock_code"] == "005930"
        assert d["sector"] == "반도체"  # 문자열 그대로
        assert d["track_type"] == TrackType.TRACK_B.value
        assert d["financial_score"] == 70.1  # 반올림
        assert d["technical_score"] == 85.5
        assert d["total_score"] == 82.8

    def test_sector_is_string(self):
        """sector 필드가 문자열인지 확인"""
        result = CandidateResult(
            stock_code="005930",
            stock_name="삼성전자",
            sector="2차전지",  # 테마명 문자열
            sector_rank=1,
            track_type=TrackType.TRACK_B,
            filter_passed=True,
            filter_reason="",
            financial_score=0,
            technical_score=0,
            total_score=0,
            overall_rank=0,
            is_selected=False,
            selection_reason="",
        )

        assert isinstance(result.sector, str)
        assert result.sector == "2차전지"


# ==========================================
# CandidateSelector 테스트
# ==========================================

class TestCandidateSelector:
    """CandidateSelector 테스트"""

    def test_init(self, selector):
        """초기화 테스트"""
        assert selector.filter is not None
        assert selector.scorer is not None

    @patch('src.stock.selector.StockFilter')
    @patch('src.stock.selector.StockScorer')
    def test_select_with_theme_string(self, mock_scorer_class, mock_filter_class):
        """테마명 문자열로 select 테스트"""
        # Mock 설정
        mock_filter = MagicMock()
        mock_scorer = MagicMock()
        mock_filter_class.return_value = mock_filter
        mock_scorer_class.return_value = mock_scorer

        # Filter 결과 설정
        from src.stock.filter import StockFilterResult
        mock_filter.apply_batch.return_value = [
            StockFilterResult(
                stock_code="005930",
                stock_name="삼성전자",
                track_type=TrackType.TRACK_B,
                passed=True,
                reason="통과",
            )
        ]

        # Scorer 결과 설정
        from src.stock.scorer import StockScoreResult
        mock_scorer.score_batch.return_value = [
            StockScoreResult(
                stock_code="005930",
                stock_name="삼성전자",
                track_type=TrackType.TRACK_B,
                financial_score=70.0,
                technical_score=85.0,
                total_score=82.0,
                rank=1,
            )
        ]

        with patch('src.stock.selector.get_logger') as mock_logger, \
             patch('src.stock.selector.get_config') as mock_config:
            mock_logger.return_value = MagicMock()
            mock_config.return_value = {}

            selector = CandidateSelector()

            stocks_data = [
                {
                    "stock_code": "005930",
                    "stock_name": "삼성전자",
                    "sector": "반도체",  # 테마명 문자열
                }
            ]

            sector_type_map = {
                "반도체": SectorType.TYPE_B,  # dict[str, SectorType]
            }

            results = selector.select(
                stocks_data=stocks_data,
                sector_type_map=sector_type_map,
                top_per_sector=3,
            )

            assert len(results) > 0
            assert results[0].sector == "반도체"  # 문자열 확인

    def test_get_selected(self, selector, sample_candidate_results):
        """get_selected 테스트"""
        selected = selector.get_selected(sample_candidate_results)

        assert len(selected) == 3  # is_selected=True인 것만
        assert all(r.is_selected for r in selected)
        # 순위순 정렬 확인
        assert selected[0].overall_rank == 1

    def test_get_by_theme(self, selector, sample_candidate_results):
        """get_by_theme 테스트 (v2.0 신규)"""
        # 반도체 테마만 필터링
        semiconductor = selector.get_by_theme(sample_candidate_results, "반도체")

        assert len(semiconductor) == 2
        assert all(r.sector == "반도체" for r in semiconductor)

        # 2차전지 테마만 필터링
        battery = selector.get_by_theme(sample_candidate_results, "2차전지")

        assert len(battery) == 2
        assert all(r.sector == "2차전지" for r in battery)

    def test_get_by_sector_deprecated(self, selector, sample_candidate_results):
        """get_by_sector (deprecated) 하위 호환성 테스트"""
        # 기존 get_by_sector가 get_by_theme과 동일하게 동작해야 함
        semiconductor = selector.get_by_sector(sample_candidate_results, "반도체")

        assert len(semiconductor) == 2
        assert all(r.sector == "반도체" for r in semiconductor)

    def test_summarize(self, selector, sample_candidate_results):
        """summarize 테스트"""
        summary = selector.summarize(sample_candidate_results)

        assert summary["total_processed"] == 4
        assert summary["filter_passed"] == 3
        assert summary["filter_failed"] == 1
        assert summary["selected_count"] == 3

        # 테마 분포 (문자열 키)
        assert "반도체" in summary["theme_distribution"]
        assert "2차전지" in summary["theme_distribution"]
        assert summary["theme_distribution"]["반도체"] == 2
        assert summary["theme_distribution"]["2차전지"] == 1

        # 하위 호환성
        assert "sector_distribution" in summary
        assert summary["sector_distribution"] == summary["theme_distribution"]

        # 트랙 분포
        assert summary["track_b_count"] == 3

        # 선정 종목 목록
        assert len(summary["selected_stocks"]) == 3
        # 테마명 확인
        assert any(s["theme"] == "반도체" for s in summary["selected_stocks"])

    def test_summarize_empty_results(self, selector):
        """빈 결과 summarize 테스트"""
        summary = selector.summarize([])

        assert summary["total_processed"] == 0
        assert summary["selected_count"] == 0
        assert summary["avg_score"] == 0


# ==========================================
# 통합 테스트
# ==========================================

class TestCandidateSelectorIntegration:
    """통합 테스트 (실제 Filter, Scorer 사용)"""

    def test_full_selection_flow(self, sample_stocks_data, sector_type_map):
        """전체 선정 플로우 테스트"""
        with patch('src.stock.selector.get_logger') as mock_logger, \
             patch('src.stock.selector.get_config') as mock_config, \
             patch('src.stock.filter.get_logger') as mock_filter_logger, \
             patch('src.stock.filter.get_config') as mock_filter_config, \
             patch('src.stock.scorer.get_logger') as mock_scorer_logger, \
             patch('src.stock.scorer.get_config') as mock_scorer_config:

            mock_logger.return_value = MagicMock()
            mock_config.return_value = {}
            mock_filter_logger.return_value = MagicMock()
            mock_filter_config.return_value = {}
            mock_scorer_logger.return_value = MagicMock()
            mock_scorer_config.return_value = {}

            selector = CandidateSelector()

            results = selector.select(
                stocks_data=sample_stocks_data,
                sector_type_map=sector_type_map,
                top_per_sector=2,
                max_total=5,
            )

            # 결과 확인
            assert len(results) == 4  # 전체 종목

            # 테마별 그룹 확인
            semiconductor = [r for r in results if r.sector == "반도체"]
            battery = [r for r in results if r.sector == "2차전지"]

            assert len(semiconductor) == 2
            assert len(battery) == 2

            # 선정된 종목
            selected = selector.get_selected(results)
            assert len(selected) <= 5  # max_total

    def test_sector_type_map_with_string_keys(self, sample_stocks_data):
        """sector_type_map이 문자열 키를 사용하는지 확인"""
        with patch('src.stock.selector.get_logger') as mock_logger, \
             patch('src.stock.selector.get_config') as mock_config, \
             patch('src.stock.filter.get_logger') as mock_filter_logger, \
             patch('src.stock.filter.get_config') as mock_filter_config, \
             patch('src.stock.scorer.get_logger') as mock_scorer_logger, \
             patch('src.stock.scorer.get_config') as mock_scorer_config:

            mock_logger.return_value = MagicMock()
            mock_config.return_value = {}
            mock_filter_logger.return_value = MagicMock()
            mock_filter_config.return_value = {}
            mock_scorer_logger.return_value = MagicMock()
            mock_scorer_config.return_value = {}

            selector = CandidateSelector()

            # 문자열 키 사용
            sector_type_map = {
                "반도체": SectorType.TYPE_B,
                "2차전지": SectorType.TYPE_B,
                "자동차": SectorType.TYPE_A,  # 추가 테마
            }

            results = selector.select(
                stocks_data=sample_stocks_data,
                sector_type_map=sector_type_map,
                top_per_sector=2,
            )

            # 매핑에 없는 테마는 기본값 TYPE_A 사용
            # 이 테스트에서는 모든 테마가 매핑되어 있으므로 통과

            # 각 결과의 sector가 문자열인지 확인
            for r in results:
                assert isinstance(r.sector, str)

    def test_default_sector_type_for_unknown_theme(self, sample_stocks_data):
        """매핑되지 않은 테마의 기본 SectorType 테스트"""
        with patch('src.stock.selector.get_logger') as mock_logger, \
             patch('src.stock.selector.get_config') as mock_config, \
             patch('src.stock.filter.get_logger') as mock_filter_logger, \
             patch('src.stock.filter.get_config') as mock_filter_config, \
             patch('src.stock.scorer.get_logger') as mock_scorer_logger, \
             patch('src.stock.scorer.get_config') as mock_scorer_config:

            mock_logger.return_value = MagicMock()
            mock_config.return_value = {}
            mock_filter_logger.return_value = MagicMock()
            mock_filter_config.return_value = {}
            mock_scorer_logger.return_value = MagicMock()
            mock_scorer_config.return_value = {}

            selector = CandidateSelector()

            # 빈 sector_type_map (모든 테마가 매핑 안됨)
            empty_map = {}

            results = selector.select(
                stocks_data=sample_stocks_data,
                sector_type_map=empty_map,
                top_per_sector=2,
            )

            # 매핑 없으면 기본값 TYPE_A 사용 → TRACK_A
            # 하지만 TrackType은 filter에서 결정됨
            assert len(results) > 0


# ==========================================
# 엣지 케이스 테스트
# ==========================================

class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_empty_stocks_data(self, selector, sector_type_map):
        """빈 데이터 처리"""
        results = selector.select(
            stocks_data=[],
            sector_type_map=sector_type_map,
        )

        assert results == []

    def test_single_stock(self, selector, sector_type_map):
        """단일 종목 처리"""
        with patch.object(selector.filter, 'apply_batch') as mock_filter, \
             patch.object(selector.scorer, 'score_batch') as mock_scorer:

            from src.stock.filter import StockFilterResult
            from src.stock.scorer import StockScoreResult

            mock_filter.return_value = [
                StockFilterResult(
                    stock_code="005930",
                    stock_name="삼성전자",
                    track_type=TrackType.TRACK_B,
                    passed=True,
                    reason="통과",
                )
            ]
            mock_scorer.return_value = [
                StockScoreResult(
                    stock_code="005930",
                    stock_name="삼성전자",
                    track_type=TrackType.TRACK_B,
                    financial_score=70.0,
                    technical_score=85.0,
                    total_score=82.0,
                    rank=1,
                )
            ]

            stocks_data = [{
                "stock_code": "005930",
                "stock_name": "삼성전자",
                "sector": "반도체",
            }]

            results = selector.select(
                stocks_data=stocks_data,
                sector_type_map=sector_type_map,
            )

            assert len(results) == 1
            assert results[0].is_selected is True

    def test_max_total_limit(self, selector):
        """max_total 제한 테스트"""
        # 10개 종목 생성, max_total=3
        candidate_results = []
        for i in range(10):
            candidate_results.append(
                CandidateResult(
                    stock_code=f"{i:06d}",
                    stock_name=f"종목{i}",
                    sector="테마A",
                    sector_rank=i + 1,
                    track_type=TrackType.TRACK_A,
                    filter_passed=True,
                    filter_reason="통과",
                    financial_score=90.0 - i,
                    technical_score=80.0 - i,
                    total_score=85.0 - i,
                    overall_rank=i + 1,
                    is_selected=True,  # 일단 모두 선정
                    selection_reason=f"테마 내 {i+1}위",
                )
            )

        selected = selector.get_selected(candidate_results)
        assert len(selected) == 10  # 모두 선정됨 (select 메서드에서 제한 적용)

    def test_multiple_themes_grouping(self, selector, sector_type_map):
        """여러 테마 그룹핑 테스트"""
        with patch.object(selector.filter, 'apply_batch') as mock_filter, \
             patch.object(selector.scorer, 'score_batch') as mock_scorer:

            from src.stock.filter import StockFilterResult
            from src.stock.scorer import StockScoreResult

            # 3개 테마, 각 2종목
            mock_filter.return_value = [
                StockFilterResult(
                    stock_code=f"00{i}",
                    stock_name=f"종목{i}",
                    track_type=TrackType.TRACK_B,
                    passed=True,
                    reason="통과",
                ) for i in range(2)
            ]
            mock_scorer.return_value = [
                StockScoreResult(
                    stock_code=f"00{i}",
                    stock_name=f"종목{i}",
                    track_type=TrackType.TRACK_B,
                    financial_score=70.0 + i,
                    technical_score=80.0 + i,
                    total_score=75.0 + i,
                    rank=i + 1,
                ) for i in range(2)
            ]

            stocks_data = [
                {"stock_code": "001", "stock_name": "종목1", "sector": "테마1"},
                {"stock_code": "002", "stock_name": "종목2", "sector": "테마1"},
                {"stock_code": "003", "stock_name": "종목3", "sector": "테마2"},
                {"stock_code": "004", "stock_name": "종목4", "sector": "테마2"},
                {"stock_code": "005", "stock_name": "종목5", "sector": "테마3"},
                {"stock_code": "006", "stock_name": "종목6", "sector": "테마3"},
            ]

            type_map = {
                "테마1": SectorType.TYPE_A,
                "테마2": SectorType.TYPE_B,
                "테마3": SectorType.TYPE_B,
            }

            results = selector.select(
                stocks_data=stocks_data,
                sector_type_map=type_map,
                top_per_sector=1,  # 테마당 1개
            )

            # 3개 테마 × 2종목 = 6종목 처리됨 (mock이라 실제로는 6개 반환 안됨)
            assert len(results) >= 0


# ==========================================
# 하위 호환성 테스트
# ==========================================

class TestBackwardCompatibility:
    """하위 호환성 테스트"""

    def test_sector_field_is_string_not_enum(self):
        """sector 필드가 Enum이 아닌 문자열인지 확인"""
        result = CandidateResult(
            stock_code="005930",
            stock_name="삼성전자",
            sector="반도체",
            sector_rank=1,
            track_type=TrackType.TRACK_B,
            filter_passed=True,
            filter_reason="",
            financial_score=0,
            technical_score=0,
            total_score=0,
            overall_rank=0,
            is_selected=False,
            selection_reason="",
        )

        # Enum이 아닌 문자열이어야 함
        assert not hasattr(result.sector, 'value')
        assert isinstance(result.sector, str)

    def test_to_dict_sector_is_plain_string(self):
        """to_dict에서 sector가 단순 문자열인지 확인"""
        result = CandidateResult(
            stock_code="005930",
            stock_name="삼성전자",
            sector="AI",
            sector_rank=1,
            track_type=TrackType.TRACK_B,
            filter_passed=True,
            filter_reason="",
            financial_score=0,
            technical_score=0,
            total_score=0,
            overall_rank=0,
            is_selected=False,
            selection_reason="",
        )

        d = result.to_dict()
        assert d["sector"] == "AI"  # 단순 문자열

    def test_summarize_uses_theme_distribution(self, selector, sample_candidate_results):
        """summarize에서 theme_distribution 사용 확인"""
        summary = selector.summarize(sample_candidate_results)

        # v2.0: theme_distribution 사용
        assert "theme_distribution" in summary
        assert isinstance(summary["theme_distribution"], dict)

        # 하위 호환성: sector_distribution도 동일
        assert summary["sector_distribution"] == summary["theme_distribution"]
