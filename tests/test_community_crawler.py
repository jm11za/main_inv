"""
커뮤니티 크롤러 테스트
"""
import pytest
from unittest.mock import Mock, patch, MagicMock

from src.ingest.community_crawler import (
    CommunityCrawler,
    CommunityPost,
    CommunityData,
)


class TestCommunityPost:
    """CommunityPost 데이터클래스 테스트"""

    def test_create_post(self):
        """글 생성"""
        post = CommunityPost(
            stock_code="005930",
            title="삼성전자 언제 오르나요?",
            author="투자자1",
            date="2026.01.31",
            views=100,
            likes=5,
        )

        assert post.stock_code == "005930"
        assert post.title == "삼성전자 언제 오르나요?"
        assert post.views == 100

    def test_post_repr(self):
        """문자열 표현"""
        post = CommunityPost(
            stock_code="005930",
            title="삼성전자 HBM 관련 뉴스가 나왔는데 어떻게 생각하시나요?",
        )

        repr_str = repr(post)
        assert "Post" in repr_str
        assert "삼성전자" in repr_str


class TestCommunityData:
    """CommunityData 집계 테스트"""

    def test_create_data(self):
        """데이터 생성"""
        posts = [
            CommunityPost(stock_code="005930", title="제목1"),
            CommunityPost(stock_code="005930", title="제목2"),
            CommunityPost(stock_code="005930", title="제목3", content="본문3"),
        ]

        data = CommunityData(
            stock_code="005930",
            stock_name="삼성전자",
            posts=posts,
            total_posts=3,
            mention_count=3,
        )

        assert data.stock_code == "005930"
        assert data.total_posts == 3

    def test_titles_property(self):
        """제목 목록"""
        posts = [
            CommunityPost(stock_code="005930", title="제목1"),
            CommunityPost(stock_code="005930", title="제목2"),
        ]

        data = CommunityData(stock_code="005930", posts=posts)

        assert data.titles == ["제목1", "제목2"]

    def test_contents_property(self):
        """본문 목록 (수집된 것만)"""
        posts = [
            CommunityPost(stock_code="005930", title="제목1", content="본문1"),
            CommunityPost(stock_code="005930", title="제목2"),  # 본문 없음
            CommunityPost(stock_code="005930", title="제목3", content="본문3"),
        ]

        data = CommunityData(stock_code="005930", posts=posts)

        assert len(data.contents) == 2
        assert "본문1" in data.contents


class TestCommunityCrawler:
    """CommunityCrawler 테스트"""

    def test_init(self):
        """초기화"""
        crawler = CommunityCrawler(max_posts=30, max_pages=2)

        assert crawler.max_posts == 30
        assert crawler.max_pages == 2
        assert crawler.get_source_name() == "NaverCommunity"

    def test_parse_int(self):
        """정수 파싱"""
        crawler = CommunityCrawler()

        assert crawler._parse_int("1,234") == 1234
        assert crawler._parse_int("100") == 100
        assert crawler._parse_int("") == 0
        assert crawler._parse_int("abc") == 0

    def test_is_ad_or_notice(self):
        """광고/공지 판별"""
        crawler = CommunityCrawler()

        assert crawler._is_ad_or_notice("광고 - 이벤트 참여하세요", "회사")
        assert crawler._is_ad_or_notice("[공지] 게시판 이용안내", "관리자")
        assert crawler._is_ad_or_notice("일반 게시글", "운영자")  # 운영자 작성
        assert not crawler._is_ad_or_notice("삼성전자 언제 오르나요?", "투자자1")

    def test_parse_board_page_empty(self):
        """빈 페이지 파싱"""
        crawler = CommunityCrawler()

        html = "<html><body></body></html>"
        posts = crawler._parse_board_page(html, "005930")

        assert posts == []

    @patch.object(CommunityCrawler, "_fetch_board_page")
    def test_fetch_stock_community(self, mock_fetch):
        """종목 커뮤니티 수집"""
        mock_posts = [
            CommunityPost(
                stock_code="005930",
                title="삼성전자 좋아요",
                views=100,
                likes=5,
            ),
            CommunityPost(
                stock_code="005930",
                title="지금 사도 될까요?",
                views=50,
                likes=2,
            ),
        ]
        mock_fetch.return_value = mock_posts

        crawler = CommunityCrawler(max_pages=1)
        data = crawler.fetch_stock_community("005930", "삼성전자")

        assert data.stock_code == "005930"
        assert data.stock_name == "삼성전자"
        assert data.total_posts == 2
        assert data.avg_views == 75.0
        assert data.avg_likes == 3.5

    @patch.object(CommunityCrawler, "fetch_stock_community")
    def test_get_sentiment_data(self, mock_fetch):
        """Sentiment 분석용 데이터 변환"""
        mock_data = CommunityData(
            stock_code="005930",
            stock_name="삼성전자",
            posts=[
                CommunityPost(stock_code="005930", title="제목1"),
                CommunityPost(stock_code="005930", title="제목2"),
            ],
            mention_count=2,
        )
        mock_fetch.return_value = mock_data

        crawler = CommunityCrawler()
        result = crawler.get_sentiment_data("005930", "삼성전자")

        assert result["stock_code"] == "005930"
        assert result["community_posts"] == ["제목1", "제목2"]
        assert result["mention_count"] == 2


class TestCommunityCrawlerIntegration:
    """통합 테스트 (실제 네트워크 호출)"""

    @pytest.mark.skip(reason="실제 네트워크 호출 - 수동 실행")
    def test_real_fetch(self):
        """실제 수집 테스트"""
        crawler = CommunityCrawler(max_posts=10, max_pages=1)
        data = crawler.fetch_stock_community("005930", "삼성전자")

        print(f"수집된 글 수: {data.total_posts}")
        print(f"제목 샘플: {data.titles[:3]}")

        assert data.total_posts > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
