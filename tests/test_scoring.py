"""
Layer 4: Scoring 모듈 테스트
"""
import pytest

from src.core.interfaces import TrackType
from src.scoring.stock_scorer import (
    StockScorer,
    ScoreResult,
    FinancialMetrics,
    TechnicalMetrics,
)


class TestFinancialMetrics:
    """재무 지표 테스트"""

    def test_create_metrics(self):
        """지표 생성"""
        metrics = FinancialMetrics(
            operating_profit_yoy=30.0,
            revenue_yoy=20.0,
            roe=15.0,
            rd_ratio=10.0,
            debt_ratio=80.0,
            current_ratio=150.0
        )

        assert metrics.operating_profit_yoy == 30.0
        assert metrics.roe == 15.0

    def test_default_values(self):
        """기본값"""
        metrics = FinancialMetrics()

        assert metrics.operating_profit_yoy == 0.0
        assert metrics.debt_ratio == 0.0


class TestTechnicalMetrics:
    """기술적 지표 테스트"""

    def test_create_metrics(self):
        """지표 생성"""
        metrics = TechnicalMetrics(
            foreign_net_ratio=0.5,
            institution_net_ratio=0.3,
            ma20_gap=5.0,
            volume_ratio=150.0,
            high_52w_proximity=90.0
        )

        assert metrics.foreign_net_ratio == 0.5
        assert metrics.high_52w_proximity == 90.0


class TestStockScorer:
    """종목 스코어러 테스트"""

    def test_score_track_a(self):
        """Track A 점수 계산"""
        scorer = StockScorer()

        financial = FinancialMetrics(
            operating_profit_yoy=30.0,
            roe=15.0,
            debt_ratio=80.0
        )
        technical = TechnicalMetrics(
            foreign_net_ratio=0.5,
            institution_net_ratio=0.3,
            ma20_gap=3.0,
            volume_ratio=120.0,
            high_52w_proximity=85.0
        )

        result = scorer.score(
            stock_code="005930",
            stock_name="삼성전자",
            track_type=TrackType.TRACK_A,
            financial=financial,
            technical=technical
        )

        assert result.stock_code == "005930"
        assert result.track_type == TrackType.TRACK_A
        assert result.financial_weight == 0.5
        assert result.technical_weight == 0.5
        assert 0 <= result.total_score <= 100

    def test_score_track_b(self):
        """Track B 점수 계산"""
        scorer = StockScorer()

        financial = FinancialMetrics(
            revenue_yoy=50.0,
            rd_ratio=15.0
        )
        technical = TechnicalMetrics(
            foreign_net_ratio=0.8,
            institution_net_ratio=0.5,
            ma20_gap=-5.0,  # 눌림목
            volume_ratio=200.0,
            high_52w_proximity=70.0
        )

        result = scorer.score(
            stock_code="000000",
            stock_name="바이오종목",
            track_type=TrackType.TRACK_B,
            financial=financial,
            technical=technical
        )

        assert result.track_type == TrackType.TRACK_B
        assert result.financial_weight == 0.2
        assert result.technical_weight == 0.8
        assert 0 <= result.total_score <= 100

    def test_track_a_vs_b_weights(self):
        """Track A/B 가중치 차이"""
        scorer = StockScorer()

        # 동일 지표로 Track A/B 점수 비교
        financial = FinancialMetrics(
            operating_profit_yoy=20.0,
            revenue_yoy=30.0,
            roe=10.0,
            rd_ratio=8.0
        )
        technical = TechnicalMetrics(
            foreign_net_ratio=0.3,
            institution_net_ratio=0.2,
            ma20_gap=2.0,
            volume_ratio=100.0,
            high_52w_proximity=80.0
        )

        result_a = scorer.score("001", "종목A", TrackType.TRACK_A, financial, technical)
        result_b = scorer.score("002", "종목B", TrackType.TRACK_B, financial, technical)

        # Track A는 재무 50%, Track B는 재무 20%
        assert result_a.financial_weight > result_b.financial_weight
        assert result_a.technical_weight < result_b.technical_weight

    def test_score_from_dict(self):
        """딕셔너리에서 점수 계산"""
        scorer = StockScorer()

        data = {
            "operating_profit_yoy": 25.0,
            "roe": 12.0,
            "debt_ratio": 100.0,
            "foreign_net_ratio": 0.4,
            "institution_net_ratio": 0.2,
            "ma20_gap": 0.0,
            "volume_ratio": 80.0,
            "high_52w_proximity": 75.0
        }

        result = scorer.score_from_dict(
            stock_code="005930",
            stock_name="삼성전자",
            track_type=TrackType.TRACK_A,
            data=data
        )

        assert result.stock_code == "005930"
        assert result.financial_metrics.operating_profit_yoy == 25.0

    def test_score_batch(self):
        """일괄 점수 계산"""
        scorer = StockScorer()

        stocks_data = [
            {
                "stock_code": "005930",
                "stock_name": "삼성전자",
                "track_type": TrackType.TRACK_A,
                "operating_profit_yoy": 30.0,
                "roe": 15.0,
                "foreign_net_ratio": 0.5,
                "ma20_gap": 2.0,
            },
            {
                "stock_code": "000660",
                "stock_name": "SK하이닉스",
                "track_type": TrackType.TRACK_A,
                "operating_profit_yoy": 40.0,
                "roe": 18.0,
                "foreign_net_ratio": 0.6,
                "ma20_gap": 1.0,
            },
        ]

        results = scorer.score_batch(stocks_data)

        assert len(results) == 2
        assert results[0].stock_code == "005930"
        assert results[1].stock_code == "000660"

    def test_score_and_rank(self):
        """점수 계산 및 순위 부여"""
        scorer = StockScorer()

        stocks_data = [
            {
                "stock_code": "001",
                "stock_name": "종목1",
                "track_type": TrackType.TRACK_A,
                "operating_profit_yoy": 10.0,
                "foreign_net_ratio": 0.1,
            },
            {
                "stock_code": "002",
                "stock_name": "종목2",
                "track_type": TrackType.TRACK_A,
                "operating_profit_yoy": 50.0,
                "foreign_net_ratio": 0.8,
            },
            {
                "stock_code": "003",
                "stock_name": "종목3",
                "track_type": TrackType.TRACK_A,
                "operating_profit_yoy": 30.0,
                "foreign_net_ratio": 0.4,
            },
        ]

        results = scorer.score_and_rank(stocks_data)

        # 점수 내림차순 정렬
        assert results[0].rank == 1
        assert results[1].rank == 2
        assert results[2].rank == 3
        assert results[0].total_score >= results[1].total_score
        assert results[1].total_score >= results[2].total_score

    def test_score_and_rank_top_n(self):
        """상위 N개만 반환"""
        scorer = StockScorer()

        stocks_data = [
            {"stock_code": f"00{i}", "stock_name": f"종목{i}", "track_type": TrackType.TRACK_A,
             "operating_profit_yoy": i * 10, "foreign_net_ratio": i * 0.1}
            for i in range(1, 6)
        ]

        results = scorer.score_and_rank(stocks_data, top_n=3)

        assert len(results) == 3
        assert results[0].rank == 1

    def test_get_weight_config(self):
        """가중치 설정 반환"""
        scorer = StockScorer()

        weights_a = scorer.get_weight_config(TrackType.TRACK_A)
        weights_b = scorer.get_weight_config(TrackType.TRACK_B)

        assert weights_a["financial"] == 0.5
        assert weights_a["technical"] == 0.5
        assert weights_b["financial"] == 0.2
        assert weights_b["technical"] == 0.8

    def test_get_top_stocks(self):
        """상위 종목 필터링"""
        scorer = StockScorer()

        stocks_data = [
            {"stock_code": "001", "stock_name": "A종목", "track_type": TrackType.TRACK_A,
             "operating_profit_yoy": 50.0, "foreign_net_ratio": 0.5},
            {"stock_code": "002", "stock_name": "B종목", "track_type": TrackType.TRACK_B,
             "operating_profit_yoy": 30.0, "foreign_net_ratio": 0.8},
            {"stock_code": "003", "stock_name": "C종목", "track_type": TrackType.TRACK_A,
             "operating_profit_yoy": 20.0, "foreign_net_ratio": 0.3},
        ]

        all_results = scorer.score_batch(stocks_data)

        # Track A만 필터
        top_a = scorer.get_top_stocks(all_results, n=5, track_type=TrackType.TRACK_A)
        assert all(r.track_type == TrackType.TRACK_A for r in top_a)

        # 전체
        top_all = scorer.get_top_stocks(all_results, n=2)
        assert len(top_all) == 2


class TestScoreCalculation:
    """점수 계산 로직 테스트"""

    def test_high_financial_score(self):
        """우수 재무 지표"""
        scorer = StockScorer()

        financial = FinancialMetrics(
            operating_profit_yoy=50.0,  # 높은 성장
            roe=20.0,                   # 높은 ROE
            debt_ratio=50.0             # 낮은 부채
        )
        technical = TechnicalMetrics()  # 기본값

        result = scorer.score("001", "우수기업", TrackType.TRACK_A, financial, technical)

        assert result.financial_score > 50  # 재무 점수 양호

    def test_debt_penalty(self):
        """부채비율 패널티"""
        scorer = StockScorer()

        # 부채 낮음
        low_debt = FinancialMetrics(operating_profit_yoy=30.0, debt_ratio=50.0)
        # 부채 높음
        high_debt = FinancialMetrics(operating_profit_yoy=30.0, debt_ratio=250.0)

        result_low = scorer.score("001", "저부채", TrackType.TRACK_A, low_debt, TechnicalMetrics())
        result_high = scorer.score("002", "고부채", TrackType.TRACK_A, high_debt, TechnicalMetrics())

        assert result_low.financial_score > result_high.financial_score

    def test_ma_gap_scoring(self):
        """MA20 이격도 점수"""
        scorer = StockScorer()

        financial = FinancialMetrics()

        # 눌림목 (-5%)
        dip = TechnicalMetrics(ma20_gap=-5.0)
        result_dip = scorer.score("001", "눌림목", TrackType.TRACK_A, financial, dip)

        # 과열 (+15%)
        overheat = TechnicalMetrics(ma20_gap=15.0)
        result_hot = scorer.score("002", "과열", TrackType.TRACK_A, financial, overheat)

        # 눌림목이 과열보다 높은 점수
        assert result_dip.technical_score > result_hot.technical_score

    def test_supply_demand_scoring(self):
        """수급 점수"""
        scorer = StockScorer()

        financial = FinancialMetrics()

        # 강한 수급
        strong = TechnicalMetrics(foreign_net_ratio=0.8, institution_net_ratio=0.5)
        result_strong = scorer.score("001", "수급강", TrackType.TRACK_A, financial, strong)

        # 약한 수급
        weak = TechnicalMetrics(foreign_net_ratio=-0.3, institution_net_ratio=-0.2)
        result_weak = scorer.score("002", "수급약", TrackType.TRACK_A, financial, weak)

        assert result_strong.technical_score > result_weak.technical_score


class TestScoreBreakdown:
    """점수 상세 내역 테스트"""

    def test_breakdown_exists(self):
        """상세 내역 존재"""
        scorer = StockScorer()

        result = scorer.score(
            "005930", "삼성전자", TrackType.TRACK_A,
            FinancialMetrics(operating_profit_yoy=30.0, roe=15.0),
            TechnicalMetrics(foreign_net_ratio=0.5, ma20_gap=2.0)
        )

        assert "financial" in result.score_breakdown
        assert "technical" in result.score_breakdown

    def test_track_a_breakdown(self):
        """Track A 상세 내역"""
        scorer = StockScorer()

        result = scorer.score(
            "005930", "삼성전자", TrackType.TRACK_A,
            FinancialMetrics(operating_profit_yoy=30.0, roe=15.0, debt_ratio=100.0),
            TechnicalMetrics()
        )

        fin_breakdown = result.score_breakdown["financial"]
        assert "영업이익YoY" in fin_breakdown
        assert "ROE" in fin_breakdown

    def test_track_b_breakdown(self):
        """Track B 상세 내역"""
        scorer = StockScorer()

        result = scorer.score(
            "000000", "바이오", TrackType.TRACK_B,
            FinancialMetrics(revenue_yoy=50.0, rd_ratio=15.0),
            TechnicalMetrics()
        )

        fin_breakdown = result.score_breakdown["financial"]
        assert "매출YoY" in fin_breakdown
        assert "R&D비중" in fin_breakdown


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
