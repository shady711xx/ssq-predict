"""Tests for crawler error handling and anti-mock safeguards."""
import pytest
from unittest.mock import MagicMock, patch
import requests

from src import crawler, storage


class TestCrawlerFailureSafeguards:
    """Test that crawler failures raise CrawlerError and never fallback to mock data."""

    @pytest.fixture(autouse=True)
    def fast_sleep(self):
        with patch('time.sleep', return_value=None):
            yield

    @patch('src.crawler.requests.get')
    def test_crawler_raises_on_http_error(self, mock_get):
        """When HTTP error occurs, crawler should retry and raise CrawlerError."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError('403 Forbidden')
        mock_get.return_value = mock_resp

        with pytest.raises(crawler.CrawlerError) as excinfo:
            crawler.fetch_history(start_issue='26130')
        assert '抓取双色球历史数据失败' in str(excinfo.value)

    @patch('src.crawler.requests.get')
    def test_crawler_raises_on_missing_table(self, mock_get):
        """When response contains anti-bot challenge or no table, crawler should raise CrawlerError."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.text = '<html><body><h1>Please verify you are human</h1></body></html>'
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        with pytest.raises(crawler.CrawlerError) as excinfo:
            crawler.fetch_history(start_issue='26130')
        assert '抓取双色球历史数据失败' in str(excinfo.value)

    @patch('src.crawler.requests.get')
    def test_crawler_raises_on_connection_timeout(self, mock_get):
        """When requests timeout, crawler should retry and raise CrawlerError."""
        mock_get.side_effect = requests.exceptions.Timeout('Connection timed out')

        with pytest.raises(crawler.CrawlerError) as excinfo:
            crawler.fetch_history(start_issue='26130')
        assert '抓取双色球历史数据失败' in str(excinfo.value)

    @patch('src.crawler.fetch_all_history')
    def test_rebuild_does_not_clear_db_if_crawl_fails(self, mock_fetch, tmp_path, monkeypatch):
        """In update_data(rebuild=True), if crawling fails, existing DB must NOT be cleared."""
        import config
        test_db = tmp_path / "test_crawler.db"
        test_csv = tmp_path / "test_ssq.csv"
        monkeypatch.setattr(config, 'DB_PATH', str(test_db))
        monkeypatch.setattr(config, 'CSV_PATH', str(test_csv))

        mock_fetch.side_effect = crawler.CrawlerError('Network is down')

        storage.init_db()
        conn = storage.get_connection()
        conn.execute('''
            INSERT INTO draws (issue, draw_date, red_1, red_2, red_3, red_4, red_5, red_6, blue)
            VALUES ('26001', '2026-01-01', 1, 2, 3, 4, 5, 6, 7)
        ''')
        conn.commit()
        conn.close()

        initial_count = storage.get_draw_count()
        assert initial_count == 1

        with pytest.raises(crawler.CrawlerError):
            crawler.update_data(rebuild=True)

        # Draw count must remain untouched
        after_count = storage.get_draw_count()
        assert after_count == initial_count


class TestCrawlerStorageDateAndCli:
    def test_get_latest_draw_date(self, tmp_path, monkeypatch):
        import config
        test_db = tmp_path / "test_date.db"
        test_csv = tmp_path / "test_date.csv"
        monkeypatch.setattr(config, 'DB_PATH', str(test_db))
        monkeypatch.setattr(config, 'CSV_PATH', str(test_csv))
        storage.init_db()

        # Empty DB returns None
        assert storage.get_latest_draw_date() is None

        # Insert 2 draws
        storage.insert_draws([
            {'issue': '26001', 'draw_date': '2026-01-01', 'red_1': 1, 'red_2': 2, 'red_3': 3, 'red_4': 4, 'red_5': 5, 'red_6': 6, 'blue': 7},
            {'issue': '26002', 'draw_date': '2026-01-03', 'red_1': 2, 'red_2': 3, 'red_3': 4, 'red_4': 5, 'red_5': 6, 'red_6': 7, 'blue': 8},
        ])
        assert storage.get_latest_draw_date() == '2026-01-03'

    def test_cmd_update_cli_wait_arg(self):
        import main
        parser = main.build_parser()
        args = parser.parse_args(['update', '--wait', '15'])
        assert args.wait == 15
        assert args.rebuild is False

