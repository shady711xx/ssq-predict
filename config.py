import os
import logging

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Directory paths
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
DOCS_DIR = os.path.join(PROJECT_ROOT, 'docs')
DOCS_DATA_DIR = os.path.join(DOCS_DIR, 'data')

# Database & CSV
DB_PATH = os.path.join(DATA_DIR, 'ssq.db')
CSV_PATH = os.path.join(DATA_DIR, 'ssq_history.csv')

# Logging
LOG_FILE = os.path.join(LOGS_DIR, 'ssq.log')
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Crawler settings
CRAWLER_URL = 'http://datachart.500.com/ssq/history/newinc/history.php'
CRAWLER_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'http://datachart.500.com/ssq/',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
}
CRAWLER_TIMEOUT = 30
CRAWLER_MAX_RETRIES = 3
CRAWLER_RETRY_DELAY = 5  # seconds

# SSQ Rules
RED_BALL_MIN = 1
RED_BALL_MAX = 33
BLUE_BALL_MIN = 1
BLUE_BALL_MAX = 16
RED_BALL_COUNT = 6
MAX_RED_OVERLAP = 3

# Model settings
MODEL_WINDOW_SIZES = [5, 10, 20]
MODEL_N_ESTIMATORS = 50
STAT_MODEL_WEIGHT = 0.35
RF_MODEL_WEIGHT = 0.35
LR_MODEL_WEIGHT = 0.30
SMOOTHING_ALPHA = 0.10
SMOOTHING_EPS = 1e-4

# Game theory settings
GAME_THEORY_BETA = 0.35
MAX_BALL_FREQ = 2
CROWD_QUICK_PICK_RATIO = 0.60
CROWD_BIRTHDAY_RATIO = 0.22
CROWD_HOT_FOLLOW_RATIO = 0.10
CROWD_PATTERN_RATIO = 0.08

# Scheduler (双色球每周二、四、日 21:15 开奖，开奖后 10 分钟自动运行)
SCHEDULE_HOUR = 21
SCHEDULE_MINUTE = 25

# Prize rules
PRIZE_RULES = {
    (6, 1): {'desc': '一等奖', 'prize': 5000000},
    (6, 0): {'desc': '二等奖', 'prize': 150000},
    (5, 1): {'desc': '三等奖', 'prize': 3000},
    (5, 0): {'desc': '四等奖', 'prize': 200},
    (4, 1): {'desc': '四等奖', 'prize': 200},
    (4, 0): {'desc': '五等奖', 'prize': 10},
    (3, 1): {'desc': '五等奖', 'prize': 10},
    (2, 1): {'desc': '六等奖', 'prize': 5},
    (1, 1): {'desc': '六等奖', 'prize': 5},
    (0, 1): {'desc': '六等奖', 'prize': 5},
}
SPECIAL_FUYUN_PRIZE = {'desc': '福运奖', 'prize': 5}
SPECIAL_RULE_ACTIVE = False


def ensure_dirs():
    """Ensure all required directories exist."""
    for d in [DATA_DIR, REPORTS_DIR, LOGS_DIR, DOCS_DIR, DOCS_DATA_DIR]:
        os.makedirs(d, exist_ok=True)


ensure_dirs()

# Initialize base logger
logger = logging.getLogger('ssq')
logger.setLevel(LOG_LEVEL)
if not logger.handlers:
    fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
    fh.setLevel(LOG_LEVEL)
    formatter = logging.Formatter(LOG_FORMAT)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
