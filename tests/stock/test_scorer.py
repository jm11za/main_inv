"""
Stage 4: StockScorer 모듈 테스트
- Track A: 재무 50% + 기술 50%
- Track B: 재무 20% + 기술 80%
"""
import pytest

from src.core.interfaces import TrackType
from src.stock.scorer import StockScorer, StockScoreResult


def _make_stock_data_a(**overrides) -> dict:
    """Track A 종목 기본 데이터"""
    base = {
        "stock_code": "005930",
        "stock_name": "삼성전자",
        # 재무 (Track A)
        "operating_profit_yoy": 30,
        "roe": 15,
        "debt_ratio": 50,
        # 기술 (공통)
        "foreign_net_ratio": 3,
        "institution_net_ratio": 2,
        "ma20_gap": 2.5,
        "volume_ratio": 1.5,
        "high_52w_proximity": 0.8,
    }
    base.update(overrides)
    return base


def _make_stock_data_b(**overrides) -> dict:
    """Track B 종목 기본 데이터"""
    base = {
        "stock_code": "068270",
        "stock_name": "셀트리온",
        # 재무 (Track B)
        "revenue_yoy": 50,
        "rd_ratio": 15,
        # 기술 (공통)
        "foreign_net_ratio": 2,
        "institution_net_ratio": 1,
        "ma20_gap": 3.0,
        "volume_ratio": 2.0,
        "high_52w_proximity": 0.7,
    }
    base.update(overrides)
    return base


class TestStockScorerTrackA:
    """Track A 점수화 테스트"""

    def test_score_basic(self):
        """기본 점수 계산"""
        scorer = StockScorer()
        result = scorer.score(_make_stock_data_a(), TrackType.TRACK_A)

        assert isinstance(result, StockScoreResult)
        assert result.stock_code == "005930"
        assert result.track_type == TrackType.TRACK_A
        assert 0 <= result.financial_score <= 100
        assert 0 <= result.technical_score <= 100
        assert 0 <= result.total_score <= 100

    def test_weight_50_50(self):
        """Track A 가중치 50:50"""
        scorer = StockScorer()
        result = scorer.score(_make_stock_data_a(), TrackType.TRACK_A)
        expected = result.financial_score * 0.5 + result.technical_score * 0.5
        assert abs(result.total_score - expected) < 0.01

    def test_high_debt_lowers_financial_score(self):
        """부채비율 높으면 재무 점수 하락"""
        scorer = StockScorer()
        low_debt = scorer.score(_make_stock_data_a(debt_ratio=30), TrackType.TRACK_A)
        high_debt = scorer.score(_make_stock_data_a(debt_ratio=180), TrackType.TRACK_A)
        assert low_debt.financial_score > high_debt.financial_score

    def test_high_roe_improves_score(self):
        """ROE 높으면 재무 점수 상승"""
        scorer = StockScorer()
        low_roe = scorer.score(_make_stock_data_a(roe=2), TrackType.TRACK_A)
        high_roe = scorer.score(_make_stock_data_a(roe=18), TrackType.TRACK_A)
        assert high_roe.financial_score > low_roe.financial_score

    def test_financial_breakdown_present(self):
        """재무 세부 점수 포함"""
        scorer = StockScorer()
        result = scorer.score(_make_stock_data_a(), TrackType.TRACK_A)
        assert "operating_profit_yoy" in result.financial_breakdown
        assert "roe" in result.financial_breakdown
        assert "debt_penalty" in result.financial_breakdown

    def test_technical_breakdown_present(self):
        """기술 세부 점수 포함"""
        scorer = StockScorer()
        result = scorer.score(_make_stock_data_a(), TrackType.TRACK_A)
        assert "supply_demand" in result.technical_breakdown
        assert "ma20_gap" in result.technical_breakdown
        assert "volume_ratio" in result.technical_breakdown
        assert "high_52w_proximity" in result.technical_breakdown


class TestStockScorerTrackB:
    """Track B 점수화 테스트"""

    def test_weight_20_80(self):
        """Track B 가중치 20:80"""
        scorer = StockScorer()
        result = scorer.score(_make_stock_data_b(), TrackType.TRACK_B)
        expected = result.financial_score * 0.2 + result.technical_score * 0.8
        assert abs(result.total_score - expected) < 0.01

    def test_financial_uses_revenue_and_rd(self):
        """Track B 재무는 매출YoY + R&D"""
        scorer = StockScorer()
        result = scorer.score(_make_stock_data_b(), TrackType.TRACK_B)
        assert "revenue_yoy" in result.financial_breakdown
        assert "rd_ratio" in result.financial_breakdown
        assert "operating_profit_yoy" not in result.financial_breakdown

    def test_high_revenue_yoy_improves_score(self):
        """매출 YoY 높으면 점수 상승"""
        scorer = StockScorer()
        low = scorer.score(_make_stock_data_b(revenue_yoy=0), TrackType.TRACK_B)
        high = scorer.score(_make_stock_data_b(revenue_yoy=80), TrackType.TRACK_B)
        assert high.financial_score > low.financial_score


class TestStockScorerTechnical:
    """기술 점수 공통 테스트"""

    def test_supply_demand_positive(self):
        """외인+기관 순매수 = 수급 점수 상승"""
        scorer = StockScorer()
        weak = scorer.score(
            _make_stock_data_a(foreign_net_ratio=-2, institution_net_ratio=-1),
            TrackType.TRACK_A,
        )
        strong = scorer.score(
            _make_stock_data_a(foreign_net_ratio=5, institution_net_ratio=4),
            TrackType.TRACK_A,
        )
        assert strong.technical_score > weak.technical_score

    def test_ma20_gap_optimal_near_2_5(self):
        """MA20 이격도 2.5% 근처가 최적"""
        scorer = StockScorer()
        optimal = scorer.score(_make_stock_data_a(ma20_gap=2.5), TrackType.TRACK_A)
        far = scorer.score(_make_stock_data_a(ma20_gap=12.0), TrackType.TRACK_A)
        assert optimal.technical_score >= far.technical_score

    def test_high_52w_proximity_improves(self):
        """52주 신고가 근접 -> 점수 상승"""
        scorer = StockScorer()
        low = scorer.score(
            _make_stock_data_a(high_52w_proximity=0.3),
            TrackType.TRACK_A,
        )
        high = scorer.score(
            _make_stock_data_a(high_52w_proximity=0.95),
            TrackType.TRACK_A,
        )
        assert high.technical_score > low.technical_score


class TestStockScorerNormalize:
    """정규화 함수 테스트"""

    def test_normalize_within_range(self):
        """범위 내 값"""
        scorer = StockScorer()
        assert scorer._normalize(50, 0, 100) == 0.5

    def test_normalize_below_min(self):
        """최솟값 이하 → 0"""
        scorer = StockScorer()
        assert scorer._normalize(-10, 0, 100) == 0.0

    def test_normalize_above_max(self):
        """최댓값 이상 → 1"""
        scorer = StockScorer()
        assert scorer._normalize(200, 0, 100) == 1.0

    def test_normalize_equal_min_max(self):
        """min == max → 0.5"""
        scorer = StockScorer()
        assert scorer._normalize(50, 50, 50) == 0.5


class TestStockScorerBatch:
    """배치 & 유틸리티 테스트"""

    def test_score_batch_sorted_and_ranked(self):
        """배치 결과 점수순 정렬 + 순위 부여"""
        scorer = StockScorer()
        stocks = [
            _make_stock_data_a(stock_code="A", operating_profit_yoy=5, roe=5),
            _make_stock_data_a(stock_code="B", operating_profit_yoy=40, roe=18),
        ]
        results = scorer.score_batch(stocks, TrackType.TRACK_A)
        assert results[0].rank == 1
        assert results[1].rank == 2
        assert results[0].total_score >= results[1].total_score

    def test_get_top_n(self):
        """상위 N개 반환"""
        scorer = StockScorer()
        stocks = [
            _make_stock_data_a(stock_code=f"S{i}", operating_profit_yoy=i * 10)
            for i in range(5)
        ]
        results = scorer.score_batch(stocks, TrackType.TRACK_A)
        top2 = scorer.get_top_n(results, 2)
        assert len(top2) == 2

    def test_summarize(self):
        """요약 통계"""
        scorer = StockScorer()
        stocks = [_make_stock_data_a(), _make_stock_data_a(roe=0, debt_ratio=190)]
        results = scorer.score_batch(stocks, TrackType.TRACK_A)
        summary = scorer.summarize(results)
        assert summary["total"] == 2
        assert "avg_score" in summary
        assert "score_distribution" in summary

    def test_summarize_empty(self):
        """빈 결과 요약"""
        scorer = StockScorer()
        assert scorer.summarize([]) == {"total": 0}

    def test_to_dict(self):
        """StockScoreResult.to_dict"""
        scorer = StockScorer()
        result = scorer.score(_make_stock_data_a(), TrackType.TRACK_A)
        d = result.to_dict()
        assert "total_score" in d
        assert "financial_breakdown" in d
        assert "technical_breakdown" in d

    def test_get_weight_config(self):
        """가중치 설정 조회"""
        scorer = StockScorer()
        cfg_a = scorer.get_weight_config(TrackType.TRACK_A)
        cfg_b = scorer.get_weight_config(TrackType.TRACK_B)
        assert cfg_a["track_weights"]["financial"] == 0.5
        assert cfg_b["track_weights"]["financial"] == 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
