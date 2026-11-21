"""SQLite storage and CSV export for SSQ lottery data."""
import sqlite3
import pandas as pd
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger('ssq.storage')


def get_connection():
    """Get SQLite connection."""
    config.ensure_dirs()
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute('PRAGMA journal_mode=WAL')
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS draws (
                issue TEXT PRIMARY KEY,
                draw_date TEXT NOT NULL,
                red_1 INTEGER NOT NULL,
                red_2 INTEGER NOT NULL,
                red_3 INTEGER NOT NULL,
                red_4 INTEGER NOT NULL,
                red_5 INTEGER NOT NULL,
                red_6 INTEGER NOT NULL,
                blue INTEGER NOT NULL
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                predict_date TEXT NOT NULL,
                target_issue TEXT NOT NULL,
                target_date TEXT,
                red_1 INTEGER NOT NULL,
                red_2 INTEGER NOT NULL,
                red_3 INTEGER NOT NULL,
                red_4 INTEGER NOT NULL,
                red_5 INTEGER NOT NULL,
                red_6 INTEGER NOT NULL,
                blue INTEGER NOT NULL,
                score REAL,
                strategy TEXT,
                sum_value INTEGER,
                odd_even_ratio TEXT,
                red_hit INTEGER DEFAULT NULL,
                blue_hit INTEGER DEFAULT NULL,
                prize_level TEXT DEFAULT NULL,
                prize_amount INTEGER DEFAULT 0,
                checked_at TEXT DEFAULT NULL
            )
        ''')
        cursor = conn.execute("PRAGMA table_info(predictions)")
        existing_cols = {row[1] for row in cursor.fetchall()}
        if 'prize_amount' not in existing_cols:
            conn.execute('ALTER TABLE predictions ADD COLUMN prize_amount INTEGER DEFAULT 0')
        if 'checked_at' not in existing_cols:
            conn.execute('ALTER TABLE predictions ADD COLUMN checked_at TEXT DEFAULT NULL')

        conn.execute('CREATE INDEX IF NOT EXISTS idx_draws_date ON draws(draw_date)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_predictions_target ON predictions(target_issue)')
        conn.commit()
    finally:
        conn.close()

    # If draws table is empty and CSV exists, automatically seed database
    load_from_csv_if_empty()
    load_predictions_from_csv_if_empty()


def load_from_csv_if_empty(csv_path: Optional[str] = None) -> int:
    """If database draws table is empty but CSV file exists, load draws from CSV."""
    if get_draw_count() > 0:
        return 0
    if csv_path is None:
        csv_path = config.CSV_PATH
    if not os.path.exists(csv_path):
        return 0
    try:
        df = pd.read_csv(csv_path, dtype={'issue': str})
        if df.empty:
            return 0
        records = df.to_dict('records')
        count = insert_draws(records)
        if count > 0:
            logger.info(f"Initialized database with {count} records loaded from {csv_path}")
        return count
    except Exception as e:
        logger.warning(f"Failed to auto-load from CSV: {e}")
        return 0


def load_predictions_from_csv_if_empty(csv_path: Optional[str] = None) -> int:
    """If predictions table is empty but reports/predictions.csv exists, load into DB."""
    if csv_path is None:
        csv_path = os.path.join(config.REPORTS_DIR, 'predictions.csv')
    if not os.path.exists(csv_path):
        return 0
    try:
        conn = get_connection()
        try:
            cursor = conn.execute('SELECT COUNT(*) FROM predictions')
            if cursor.fetchone()[0] > 0:
                return 0
            df = pd.read_csv(csv_path, dtype={'target_issue': str})
            if df.empty:
                return 0
            df = df.where(pd.notnull(df), None)
            cols = [c for c in df.columns if c != 'id']
            placeholders = ', '.join(['?'] * len(cols))
            col_names = ', '.join(cols)
            sql = f"INSERT INTO predictions ({col_names}) VALUES ({placeholders})"
            cursor.executemany(sql, df[cols].values.tolist())
            conn.commit()
            count = len(df)
            logger.info(f"Initialized predictions table with {count} records loaded from {csv_path}")
            return count
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"Failed to auto-load predictions from CSV: {e}")
        return 0


def get_latest_issue() -> Optional[str]:
    """Get the latest issue number in the database."""
    conn = get_connection()
    try:
        cursor = conn.execute('SELECT MAX(issue) FROM draws')
        row = cursor.fetchone()
        return row[0] if row and row[0] else None
    finally:
        conn.close()


def get_draw_count() -> int:
    """Get total number of draws in the database."""
    conn = get_connection()
    try:
        cursor = conn.execute('SELECT COUNT(*) FROM draws')
        return cursor.fetchone()[0]
    finally:
        conn.close()


def clear_draws():
    """Clear all records from draws table (used for rebuild)."""
    conn = get_connection()
    try:
        conn.execute('DELETE FROM draws')
        conn.commit()
        logger.info('Cleared all draws from database')
    finally:
        conn.close()


def validate_draw(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate and clean a single draw record.
    Returns cleaned dict with 5-digit issue, or None if invalid.
    """
    try:
        raw_issue = str(rec.get('issue', '')).strip()
        if not raw_issue.isdigit():
            return None
        issue = raw_issue.zfill(5)
        
        draw_date = str(rec.get('draw_date', '')).strip()
        if not draw_date:
            return None
        
        reds = [int(rec[f'red_{i}']) for i in range(1, 7)]
        if len(set(reds)) != 6:
            return None
        if not all(1 <= r <= 33 for r in reds):
            return None
        reds.sort()
        
        blue = int(rec['blue'])
        if not (1 <= blue <= 16):
            return None
            
        return {
            'issue': issue,
            'draw_date': draw_date,
            'red_1': reds[0], 'red_2': reds[1], 'red_3': reds[2],
            'red_4': reds[3], 'red_5': reds[4], 'red_6': reds[5],
            'blue': blue
        }
    except (ValueError, TypeError, KeyError):
        return None


def insert_draws(records: List[Dict[str, Any]]) -> int:
    """Insert draw records, ignoring duplicates."""
    if not records:
        return 0
    conn = get_connection()
    inserted = 0
    try:
        cursor = conn.cursor()
        for rec in records:
            clean_rec = validate_draw(rec)
            if not clean_rec:
                continue
            try:
                cursor.execute(
                    '''INSERT OR IGNORE INTO draws 
                       (issue, draw_date, red_1, red_2, red_3, red_4, red_5, red_6, blue)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (clean_rec['issue'], clean_rec['draw_date'],
                     clean_rec['red_1'], clean_rec['red_2'], clean_rec['red_3'],
                     clean_rec['red_4'], clean_rec['red_5'], clean_rec['red_6'], clean_rec['blue'])
                )
                if cursor.rowcount > 0:
                    inserted += 1
            except sqlite3.IntegrityError:
                pass
        conn.commit()
    finally:
        conn.close()
    return inserted


def get_all_draws() -> pd.DataFrame:
    """Get all draw records as a DataFrame, ordered by issue."""
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            'SELECT * FROM draws ORDER BY issue ASC', conn,
            dtype={'issue': str}
        )
        if not df.empty:
            df['issue'] = df['issue'].astype(str).str.zfill(5)
        return df
    finally:
        conn.close()


def get_draws_before(issue: str) -> pd.DataFrame:
    """Get all draws with issue number strictly less than the given issue."""
    conn = get_connection()
    try:
        issue_str = str(issue).zfill(5)
        df = pd.read_sql_query(
            'SELECT * FROM draws WHERE issue < ? ORDER BY issue ASC',
            conn, params=(issue_str,),
            dtype={'issue': str}
        )
        if not df.empty:
            df['issue'] = df['issue'].astype(str).str.zfill(5)
        return df
    finally:
        conn.close()


def get_draws_in_date_range(start_date: str, end_date: str) -> pd.DataFrame:
    """Get draws within a date range [start_date, end_date]."""
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            'SELECT * FROM draws WHERE draw_date >= ? AND draw_date <= ? ORDER BY draw_date ASC',
            conn, params=(start_date, end_date),
            dtype={'issue': str}
        )
        if not df.empty:
            df['issue'] = df['issue'].astype(str).str.zfill(5)
        return df
    finally:
        conn.close()


def export_csv(output_path: Optional[str] = None):
    """Export all draws to CSV file with UTF-8-SIG encoding."""
    if output_path is None:
        output_path = config.CSV_PATH
    df = get_all_draws()
    if not df.empty:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f'Exported {len(df)} records to {output_path}')


def save_predictions(predictions: List[Dict[str, Any]]) -> int:
    """Save prediction records to predictions table."""
    if not predictions:
        return 0
    conn = get_connection()
    inserted = 0
    try:
        cursor = conn.cursor()
        for p in predictions:
            cursor.execute('''
                INSERT INTO predictions 
                (predict_date, target_issue, target_date, red_1, red_2, red_3, red_4, red_5, red_6, blue, score, strategy, sum_value, odd_even_ratio)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                p.get('predict_date', datetime.now().strftime('%Y-%m-%d')),
                str(p.get('target_issue', '')).zfill(5),
                p.get('target_date', ''),
                p['red_1'], p['red_2'], p['red_3'], p['red_4'], p['red_5'], p['red_6'],
                p['blue'],
                p.get('score', 0.0),
                p.get('strategy', 'ensemble'),
                p.get('sum_value', 0),
                p.get('odd_even_ratio', '3:3')
            ))
            inserted += 1
        conn.commit()
    finally:
        conn.close()
    return inserted


def get_predictions_for_issue(target_issue: str) -> pd.DataFrame:
    """Get predictions generated for a specific target issue."""
    conn = get_connection()
    try:
        issue_str = str(target_issue).zfill(5)
        return pd.read_sql_query(
            'SELECT * FROM predictions WHERE target_issue = ? ORDER BY id ASC',
            conn, params=(issue_str,)
        )
    finally:
        conn.close()


def get_all_predictions() -> pd.DataFrame:
    """Get all prediction records."""
    conn = get_connection()
    try:
        return pd.read_sql_query('SELECT * FROM predictions ORDER BY id ASC', conn)
    finally:
        conn.close()


def check_and_update_predictions() -> List[Dict[str, Any]]:
    """Check unverified predictions against draws table and update hit/prize details."""
    conn = get_connection()
    newly_checked = []
    try:
        unverified = pd.read_sql_query(
            'SELECT * FROM predictions WHERE checked_at IS NULL', conn
        )
        if unverified.empty:
            return []

        draws = pd.read_sql_query('SELECT * FROM draws', conn)
        if draws.empty:
            return []
        
        draws_map = {str(row['issue']).zfill(5): row for _, row in draws.iterrows()}
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        cursor = conn.cursor()
        for _, pred in unverified.iterrows():
            target_issue = str(pred['target_issue']).zfill(5)
            if target_issue in draws_map:
                actual = draws_map[target_issue]
                actual_reds = {actual[f'red_{i}'] for i in range(1, 7)}
                actual_blue = actual['blue']
                
                pred_reds = {pred[f'red_{i}'] for i in range(1, 7)}
                pred_blue = pred['blue']
                
                red_hits = len(pred_reds & actual_reds)
                blue_hit = 1 if pred_blue == actual_blue else 0
                
                rule = config.PRIZE_RULES.get((red_hits, blue_hit), {'desc': '未中奖', 'prize': 0})
                prize_level = rule['desc']
                prize_amount = rule['prize']
                
                cursor.execute('''
                    UPDATE predictions 
                    SET red_hit = ?, blue_hit = ?, prize_level = ?, prize_amount = ?, checked_at = ?
                    WHERE id = ?
                ''', (red_hits, blue_hit, prize_level, prize_amount, now_str, pred['id']))
                
                newly_checked.append({
                    'id': pred['id'],
                    'target_issue': target_issue,
                    'red_hit': red_hits,
                    'blue_hit': blue_hit,
                    'blue': pred_blue,
                    'prize_level': prize_level,
                    'prize_amount': prize_amount,
                    'checked_at': now_str,
                })
        conn.commit()
    finally:
        conn.close()
    return newly_checked


def get_checked_predictions_summary() -> Dict[str, Any]:
    """Calculate summary statistics for all verified predictions."""
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            'SELECT * FROM predictions WHERE checked_at IS NOT NULL', conn
        )
        if df.empty:
            return {
                'total_checked': 0,
                'total_investment': 0,
                'total_winnings': 0,
                'net_return': 0,
                'roi_pct': 0.0,
                'prize_counts': {},
            }
        
        total_tickets = len(df)
        total_investment = total_tickets * 2
        total_winnings = int(df['prize_amount'].sum())
        net_return = total_winnings - total_investment
        roi = round((net_return / total_investment) * 100, 2) if total_investment > 0 else 0.0
        prize_counts = df['prize_level'].value_counts().to_dict()

        return {
            'total_predictions': total_tickets,
            'total_checked': total_tickets,
            'total_investment': total_investment,
            'total_winnings': total_winnings,
            'net_return': net_return,
            'roi_pct': roi,
            'prize_counts': prize_counts,
        }
    finally:
        conn.close()
