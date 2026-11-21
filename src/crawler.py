"""Web crawler for SSQ lottery historical data from 500.com."""
import requests
from bs4 import BeautifulSoup
import time
import logging
import os
import sys
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src import storage

logger = logging.getLogger('ssq.crawler')


class CrawlerError(RuntimeError):
    """Exception raised when crawling SSQ data fails."""
    pass


def fetch_history(start_issue: Optional[str] = None, end_issue: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch SSQ draw history from 500.com with retry mechanism."""
    params = {}
    if start_issue:
        params['start'] = start_issue
    if end_issue:
        params['end'] = end_issue
    
    records = []
    last_error = None
    
    for attempt in range(config.CRAWLER_MAX_RETRIES):
        try:
            logger.info(f'Fetching SSQ data (attempt {attempt + 1}/{config.CRAWLER_MAX_RETRIES})...')
            response = requests.get(
                config.CRAWLER_URL,
                params=params,
                headers=config.CRAWLER_HEADERS,
                timeout=config.CRAWLER_TIMEOUT
            )
            response.raise_for_status()
            response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'id': 'tdata'}) or soup.find('table', {'id': 'tablelist'})
            if not table:
                for t in soup.find_all('table'):
                    if len(t.find_all('tr')) > 10:
                        table = t
                        break
            
            if not table:
                last_error = RuntimeError(f'未能找到开奖数据表格，目标页面可能已改版或触发反爬拦截 (HTTP {response.status_code})')
                logger.warning(f'{last_error} (attempt {attempt + 1})')
                if attempt < config.CRAWLER_MAX_RETRIES - 1:
                    time.sleep(config.CRAWLER_RETRY_DELAY)
                    continue
                break
            
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) < 9:
                    continue
                
                try:
                    issue = cells[0].get_text(strip=True)
                    if not issue or not re.match(r'^\d+$', issue):
                        continue
                    
                    reds = []
                    for i in range(1, 7):
                        ball = int(cells[i].get_text(strip=True))
                        reds.append(ball)
                    blue = int(cells[7].get_text(strip=True))
                    reds.sort()
                    
                    draw_date = None
                    for i in range(8, len(cells)):
                        text = cells[i].get_text(strip=True)
                        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', text)
                        if date_match:
                            draw_date = date_match.group(1)
                            break
                    
                    if not draw_date:
                        draw_date = cells[-1].get_text(strip=True) if len(cells) > 8 else ''
                        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', draw_date)
                        draw_date = date_match.group(1) if date_match else ''
                    
                    if all(1 <= r <= 33 for r in reds) and 1 <= blue <= 16 and len(set(reds)) == 6:
                        records.append({
                            'issue': issue,
                            'draw_date': draw_date,
                            'red_1': reds[0], 'red_2': reds[1], 'red_3': reds[2],
                            'red_4': reds[3], 'red_5': reds[4], 'red_6': reds[5],
                            'blue': blue
                        })
                except (ValueError, IndexError) as e:
                    logger.debug(f'Skipping row: {e}')
                    continue
            
            if records:
                logger.info(f'Successfully parsed {len(records)} draw records')
                return records
            else:
                last_error = RuntimeError('目标表格中未能成功解析出任何合法的双色球开奖记录')
                logger.warning(f'{last_error} (attempt {attempt + 1})')
                
        except requests.exceptions.Timeout as e:
            last_error = e
            logger.warning(f'Request timed out (attempt {attempt + 1}): {e}')
        except requests.exceptions.ConnectionError as e:
            last_error = e
            logger.warning(f'Connection error (attempt {attempt + 1}): {e}')
        except requests.exceptions.RequestException as e:
            last_error = e
            logger.error(f'Request error (attempt {attempt + 1}): {e}')
        except Exception as e:
            last_error = e
            logger.error(f'Unexpected error during crawling (attempt {attempt + 1}): {e}')
        
        if attempt < config.CRAWLER_MAX_RETRIES - 1:
            logger.info(f'Retrying in {config.CRAWLER_RETRY_DELAY} seconds...')
            time.sleep(config.CRAWLER_RETRY_DELAY)
    
    err_msg = f'抓取双色球历史数据失败（已重试 {config.CRAWLER_MAX_RETRIES} 次）。原因: {last_error}'
    logger.error(err_msg)
    raise CrawlerError(err_msg)


def fetch_year(year_code: int) -> List[Dict[str, Any]]:
    """Fetch all draws for a specific year (e.g. 3 for 2003, 26 for 2026)."""
    start_issue = f"{year_code:02d}001"
    end_issue = f"{year_code:02d}999"
    return fetch_history(start_issue=start_issue, end_issue=end_issue)


def fetch_all_history(start_year: int = 3, end_year: Optional[int] = None, delay: float = 0.5) -> List[Dict[str, Any]]:
    """Fetch complete history from 2003 to current year in yearly chunks."""
    if end_year is None:
        end_year = datetime.now().year % 100
        
    all_records = []
    logger.info(f'Starting full history crawl from 20{start_year:02d} to 20{end_year:02d}...')
    
    for yr in range(start_year, end_year + 1):
        year_str = f"20{yr:02d}"
        logger.info(f'Fetching {year_str} SSQ draws...')
        records = fetch_year(yr)
        if records:
            all_records.extend(records)
            logger.info(f'  Fetched {len(records)} draws for {year_str}')
        else:
            logger.warning(f'  No draws found for {year_str}')
        time.sleep(delay)
        
    if len(all_records) < 1000:
        err_msg = f'全量历史数据抓取不完整（仅获取到 {len(all_records)} 条记录，预期应大于 1000 条）。已终止操作！'
        logger.error(err_msg)
        raise CrawlerError(err_msg)

    logger.info(f'Completed full crawl: total {len(all_records)} draws fetched')
    return all_records


def update_data(rebuild: bool = False) -> int:
    """Main update function: fetch new data, insert to DB, export CSV."""
    storage.init_db()
    
    if rebuild:
        logger.info('Rebuild requested: fetching full history from 2003 first...')
        records = fetch_all_history()
        logger.info(f'Full history fetched ({len(records)} records). Clearing existing draws and rebuilding...')
        storage.clear_draws()
    else:
        draw_count = storage.get_draw_count()
        if draw_count < 100:
            logger.info(f'DB has {draw_count} records (<100), fetching full history from 2003...')
            records = fetch_all_history()
        else:
            latest_issue = storage.get_latest_issue()
            logger.info(f'Latest issue in DB: {latest_issue}, checking for new draws...')
            records = fetch_history(start_issue=latest_issue)
    
    if records:
        new_count = storage.insert_draws(records)
        storage.export_csv()
        total_count = storage.get_draw_count()
        logger.info(f'Update complete: {new_count} records inserted, total in DB: {total_count}')
        return new_count
    else:
        logger.info('No new records to insert')
        if storage.get_draw_count() > 0:
            storage.export_csv()
        return 0
