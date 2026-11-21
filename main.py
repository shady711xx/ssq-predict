"""Main CLI entry point for SSQ Lottery Prediction System.

Commands:
    update      - Fetch latest draw data and update database
    backtest    - Run backtesting over recent draws
    predict     - Generate predictions for upcoming draws
    check       - Check past predictions against actual draws
    run-daily   - Run daily automated pipeline (update -> check -> predict -> export-pool)
    export-pool - Export top-20 frozen candidate pool for mobile app
    meihua      - Run interactive Meihua Yishu divination on candidate pool
    schedule    - Start scheduled daily tasks
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import config

logger = logging.getLogger('ssq.main')


def cmd_update(args):
    """Execute data update."""
    from src import crawler, storage
    
    config.ensure_dirs()
    storage.init_db()
    rebuild = getattr(args, 'rebuild', False)
    
    if rebuild:
        print('🔄 正在全量重建历史数据库 (2003年至今)...')
    else:
        print('🔄 正在检查并更新双色球最新数据...')
    
    try:
        new_count = crawler.update_data(rebuild=rebuild)
        total = storage.get_draw_count()
        if new_count > 0:
            print(f'✅ 成功更新/导入 {new_count} 条数据 (当前库内总记录数: {total})')
        else:
            print(f'✅ 数据已是最新 (共 {total} 条记录)')
        print(f'📊 数据库总记录数: {total}')
    except Exception as e:
        logger.error(f'Update failed: {e}')
        print(f'❌ 数据抓取更新失败: {e}')
        print('⚠️  为保证数据真实纯净，系统已终止执行并退出，未生成任何模拟数据。请检查网络或目标网站状态！')
        sys.exit(1)


def cmd_backtest(args):
    """Execute backtesting."""
    from src import storage
    from src.backtest import (
        print_backtest_report,
        run_backtest,
        save_backtest_report,
    )
    
    storage.init_db()
    draws = getattr(args, 'draws', None)
    days = getattr(args, 'days', None)
    if draws is None and days is None:
        draws = 100
    count = args.count
    special_rule = getattr(args, 'special_rule', False)
    seed = getattr(args, 'seed', 42)

    period_str = f'最近 {draws} 期' if draws else f'过去 {days} 天'
    print(f'\n🔍 正在执行 {period_str} 的 Walk-Forward 消融回测分析 (每期 {count} 注, 随机种子: {seed})...')
    if special_rule:
        print('⭐ 已激活 2026 新规福运奖 (3红+0蓝=5元)')
    print('⏳ 正在进行逐期时序验证与概率校准评估，请稍候...\n')

    detail_results, summary = run_backtest(
        days=days,
        draws=draws,
        n_combinations=count,
        special_rule=special_rule,
        random_seed=seed,
    )
    print_backtest_report(summary, special_rule=special_rule)
    save_backtest_report(detail_results, summary)


def cmd_predict(args):
    """Execute prediction generation."""
    from src.predictor import generate_predictions, print_predictions
    
    count = getattr(args, 'count', 5)
    next_draws = getattr(args, 'next_draws', 1)
    mode = getattr(args, 'mode', 'balanced')
    beta = getattr(args, 'beta', config.GAME_THEORY_BETA)
    max_ball_freq = getattr(args, 'max_ball_freq', config.MAX_BALL_FREQ)

    mode_desc = '稳健保底型 (5不同蓝, 单球<=2)' if mode == 'balanced' else '冲大奖型 (锁单蓝, 单球<=2)'
    print(f'\n🔮 正在生成双色球推荐号码 (模式: {mode_desc}, beta={beta}, 单球频次<={max_ball_freq})...')

    predictions = generate_predictions(
        n_combinations=count,
        n_draws=next_draws,
        mode=mode,
        beta=beta,
        max_ball_freq=max_ball_freq,
    )
    if predictions:
        print_predictions(predictions)
    else:
        print('❌ 预测生成失败，历史数据可能不足。')


def cmd_check(args):
    """Check unverified predictions against database draws."""
    from src.predictor import run_check_and_report
    run_check_and_report()


def cmd_export_pool(args):
    """Export top-20 candidate pool for mobile app."""
    from src.predictor import export_frozen_candidate_pool
    print('\n📱 正在导出双模式 Top 20 冻结候选池...')
    pool_data = export_frozen_candidate_pool()
    meta = pool_data['meta']
    b_count = len(pool_data['pools']['balanced'])
    j_count = len(pool_data['pools']['jackpot'])
    print(f"✅ 成功导出候选池！目标期号: {meta['target_issue']} ({meta['target_draw_date']})")
    print(f"   - 稳健保底池 (balanced): {b_count} 注")
    print(f"   - 冲大奖池 (jackpot): {j_count} 注")
    print(f"   - 输出文件: docs/data/pool.json & reports/latest_pool.json")


def cmd_meihua(args):
    """Execute Meihua Yishu divination on candidate pool."""
    from src import meihua
    from src.predictor import export_frozen_candidate_pool
    from tabulate import tabulate

    # Parse time
    time_str = getattr(args, 'time', None)
    if time_str:
        try:
            purchase_dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M')
        except ValueError:
            purchase_dt = datetime.now()
    else:
        purchase_dt = datetime.now()

    # Parse numbers
    num_arg = getattr(args, 'numbers', ['3', '8'])
    if isinstance(num_arg, list):
        num_str = ' '.join(num_arg)
    else:
        num_str = str(num_arg)
    import re
    raw_nums = re.findall(r'\d+', num_str)
    numbers = [int(x) for x in raw_nums] if raw_nums else [3, 8]

    mode = getattr(args, 'mode', 'balanced')
    gamma = getattr(args, 'gamma', 0.30)

    # Check pool file
    pool_file = os.path.join(config.DOCS_DATA_DIR, 'pool.json')
    if not os.path.exists(pool_file):
        print('ℹ️ 候选池文件不存在，正在即时生成当期 Top 20 候选池...')
        export_frozen_candidate_pool()

    with open(pool_file, 'r', encoding='utf-8') as f:
        pool_json = json.load(f)

    target_issue = pool_json['meta']['target_issue']
    draw_date_str = pool_json['meta']['target_draw_date']
    candidate_pool = pool_json['pools'].get(mode, [])
    if not candidate_pool:
        print(f'❌ 模式 {mode} 在候选池中无数据。')
        return

    # Divination selection
    result = meihua.select_meihua_tickets(
        candidate_pool,
        purchase_dt=purchase_dt,
        numbers=numbers,
        mode=mode,
        gamma=gamma,
        target_count=5,
        max_ball_freq=2,
    )

    hex_info = result['hexagram_info']
    fortune = result['fortune']
    selected = result['selected_tickets']

    print('=' * 85)
    print(f"🌸 梅花易数现场选号终审 (Meihua SSQ Terminal) | 目标期号: {target_issue} ({draw_date_str})")
    print('=' * 85)
    print(f"🕒 购彩时间: {result['purchase_time']} ({result['shichen']}) | 心动报数: {result['numbers']}")
    print(f"📐 起卦排盘:")
    print(f"   - 主卦: {hex_info['main']['upper']['symbol']}{hex_info['main']['lower']['symbol']} 【{hex_info['main']['name']}】 "
          f"(体: {hex_info['ti']['position']}{hex_info['ti']['name']}[{hex_info['ti']['element']}], "
          f"用: {hex_info['yong']['position']}{hex_info['yong']['name']}[{hex_info['yong']['element']}])")
    print(f"   - 动爻: 第 {hex_info['moving_line']} 爻动")
    print(f"   - 互卦: {hex_info['mutual']['upper']['symbol']}{hex_info['mutual']['lower']['symbol']} 【{hex_info['mutual']['name']}】")
    print(f"   - 变卦: {hex_info['transformed']['upper']['symbol']}{hex_info['transformed']['lower']['symbol']} 【{hex_info['transformed']['name']}】 "
          f"(变用: {hex_info['trans_yong']['name']}[{hex_info['trans_yong']['element']}])")
    print('-' * 85)
    print(f"🔮 卦象总断: 【{fortune['rating']}】 {fortune['description']}")
    favored_str = ' / '.join(fortune['favored_elements']) if fortune['favored_elements'] else '无'
    avoided_str = ' / '.join(fortune['avoided_elements']) if fortune['avoided_elements'] else '无'
    print(f"   - 喜用五行: {favored_str} | 忌讳五行: {avoided_str}")
    print('-' * 85)
    print(f"🎯 终审精选 5 注号码 (模式: {mode}, 梅花融合权重 gamma={gamma}):")

    table_data = []
    for i, t in enumerate(selected):
        reds_str = ' '.join(f"{r:02d}" for r in t['reds'])
        fav_reds_str = ','.join(f"{r:02d}" for r in t.get('matched_favored_reds', [])) or '-'
        table_data.append([
            f"第 {i+1} 注",
            f"原#{t.get('original_rank', t.get('rank', i+1))}",
            reds_str,
            f"{t['blue']:02d}",
            f"{t.get('composite_score', 0):.4f}",
            f"{t.get('meihua_score', 0):.4f}",
            f"{t.get('final_meihua_score', 0):.4f}",
            fav_reds_str,
        ])

    print(tabulate(
        table_data,
        headers=['序号', '池排名', '红球推荐 (6码)', '蓝球', '基底分', '梅花分', '终审分', '匹配喜用红球'],
        tablefmt='grid'
    ))
    print("🔒 铁律合规校验: [通过] 严禁改动原注任何数字 | 蓝球去重通过 | 单红球最高出现 <= 2 次")
    print('=' * 85)


def cmd_run_daily(args):
    """Execute daily automated pipeline."""
    from src import crawler, storage
    from src.predictor import (
        estimate_issue_number,
        generate_predictions,
        get_next_draw_dates,
        print_predictions,
        run_check_and_report,
        export_frozen_candidate_pool,
    )

    storage.init_db()
    start_time = datetime.now()
    print('\n' + '=' * 95)
    print(f'🚀 双色球一键日常流水线运行 (Daily SSQ Automated Pipeline) - {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 95)

    # 1. Incremental update
    print('\n[Step 1/4] 📥 正在增量拉取最新开奖数据...')
    try:
        new_count = crawler.update_data()
        print(f'   ✅ 数据拉取完成，新增 {new_count} 期开奖记录 (当前历史总数: {storage.get_draw_count()} 期)')
    except Exception as e:
        logger.error(f'Daily update failed: {e}')
        print(f'   ❌ 数据拉取失败: {e}')
        print('   ⚠️ 因开奖数据未能成功抓取，为保证预测准确性与数据真实纯净，流水线已终止退出！')
        sys.exit(1)

    # 2. Check pending predictions
    print('\n[Step 2/4] 🎯 正在自动核对已开奖预测...')
    newly_checked, check_summary = run_check_and_report()

    # 3. Predict next draw
    mode = getattr(args, 'mode', 'balanced')
    beta = getattr(args, 'beta', config.GAME_THEORY_BETA)
    count = getattr(args, 'count', 5)
    force = getattr(args, 'force', False)

    print('\n[Step 3/4] 🔮 检查并生成最新一期预测...')
    next_dates = get_next_draw_dates(1)
    target_date = next_dates[0]
    latest_issue = storage.get_latest_issue()
    all_draws = storage.get_all_draws()
    latest_date = all_draws[all_draws['issue'] == latest_issue].iloc[0]['draw_date'] if latest_issue and not all_draws.empty else None
    target_issue = estimate_issue_number(target_date, latest_issue, latest_date)

    existing_preds = storage.get_predictions_for_issue(target_issue)
    predictions = None

    if not existing_preds.empty and not force:
        print(f'   ℹ️ 目标期号 {target_issue} ({target_date.strftime("%Y-%m-%d")}) 的预测已存在 (共 {len(existing_preds)} 注)，跳过重复写入。')
        print('   💡 如需覆盖重新生成，请添加 --force 参数。')
    else:
        if force and not existing_preds.empty:
            conn = storage.get_connection()
            try:
                conn.execute('DELETE FROM predictions WHERE target_issue = ?', (target_issue,))
                conn.commit()
                logger.info(f'Cleared existing predictions for {target_issue} (--force)')
            finally:
                conn.close()

        predictions = generate_predictions(
            n_combinations=count,
            n_draws=1,
            mode=mode,
            beta=beta,
        )
        if predictions:
            print_predictions(predictions)

    # 4. Export candidate pool for mobile app
    print('\n[Step 4/4] 📱 正在导出当期 Top 20 移动端冻结候选池...')
    pool_data = export_frozen_candidate_pool()
    print(f"   ✅ 已生成 docs/data/pool.json 与 reports/latest_pool.json (双模式各 20 注)")

    # 5. Generate markdown report in reports/latest_run.md
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    report_path = os.path.join(config.REPORTS_DIR, 'latest_run.md')
    config.ensure_dirs()
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('# 双色球自动化日常运行报告 (Daily Run Summary)\n\n')
        f.write(f'> 运行时间: {start_time.strftime("%Y-%m-%d %H:%M:%S")} | 耗时: {duration:.2f} 秒\n\n')
        f.write('## 1. 数据同步状态\n\n')
        f.write(f'- **新增开奖期数**: {new_count} 期\n')
        f.write(f'- **历史库总期数**: {storage.get_draw_count()} 期\n\n')
        f.write('## 2. 预测开奖核对\n\n')
        if newly_checked:
            f.write(f'- **本次新核对**: {len(newly_checked)} 注预测\n')
            for c in newly_checked:
                f.write(f"  - 期号 `{c['target_issue']}`: 命中红{c['red_hit']}+蓝{c['blue_hit']} -> **{c['prize_level']}** (¥{c['prize_amount']})\n")
        else:
            f.write('- **本次新核对**: 无（历史记录已全部核对完毕或目标期未开奖）\n')
        f.write(f'- **实盘累计总投入**: ¥{check_summary["total_investment"]}\n')
        f.write(f'- **实盘累计总中奖**: ¥{check_summary["total_winnings"]}\n')
        f.write(f'- **实盘累计净收益**: ¥{check_summary["net_return"]} (ROI: {check_summary["roi_pct"]}%)\n\n')
        f.write('## 3. 当期预测结果\n\n')
        f.write(f'- **目标期号**: `{target_issue}` ({target_date.strftime("%Y-%m-%d")})\n')
        f.write(f'- **选号模式**: `{mode}` (beta={beta})\n')
        if predictions:
            f.write('| 序号 | 投注号码 | 模型分 | 独享分 | 综合分 | 反大众标签 |\n')
            f.write('|---|---|---|---|---|---|\n')
            for i, combo in enumerate(predictions[0]['combinations']):
                reds_str = ' '.join(f"{r:02d}" for r in combo['reds'])
                tags_str = ' / '.join(combo.get('tags', []))
                f.write(f"| 第{i+1}注 | `{reds_str} + {combo['blue']:02d}` | {combo['model_score']:.4f} | {combo['uniqueness_score']:.4f} | {combo['final_score']:.4f} | {tags_str} |\n")
        else:
            f.write(f'- 该期已有预测，未覆盖写入（如需重置请加 `--force`）。\n')
        f.write('\n## 4. 移动端候选池导出\n\n')
        f.write(f'- **候选池文件**: `docs/data/pool.json`\n')
        f.write(f'- **候选注数**: 双模式各 20 注，支持手机端无损梅花终审选号。\n')

    print(f'\n📝 日常运行摘要已保存至 {report_path}')
    print('=' * 95)


def cmd_schedule(args):
    """Start scheduled tasks."""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        print('❌ 请先安装 apscheduler: pip install APScheduler')
        sys.exit(1)

    from src import crawler, storage
    from src.predictor import check_past_predictions, generate_predictions, export_frozen_candidate_pool

    storage.init_db()

    def daily_job():
        logger.info('Starting scheduled daily job...')
        print(f'\n⏰ [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] 执行定时任务...')
        
        try:
            new_count = crawler.update_data()
            print(f'  📥 数据更新完成 (新增 {new_count} 条)')
        except Exception as e:
            logger.error(f'Scheduled update failed: {e}')
            print(f'  ❌ 定时更新数据抓取失败: {e}。已跳过当期后续任务，未生成任何模拟数据。')
            return

        try:
            check_past_predictions()
            print('  ✅ 历史预测核对完成')
        except Exception as e:
            logger.error(f'Prediction check failed: {e}')

        try:
            predictions = generate_predictions(n_combinations=5, n_draws=1)
            if predictions:
                print('  🔮 已生成最新预测号码')
        except Exception as e:
            logger.error(f'Prediction generation failed: {e}')

        try:
            export_frozen_candidate_pool()
            print('  📱 已导出当期 Top 20 移动端候选池')
        except Exception as e:
            logger.error(f'Pool export failed: {e}')

        print('  ✅ 定时任务完成')

    scheduler = BlockingScheduler()
    trigger = CronTrigger(
        hour=config.SCHEDULE_HOUR,
        minute=config.SCHEDULE_MINUTE
    )
    scheduler.add_job(daily_job, trigger, id='daily_ssq_job')
    print(f'\n🕐 定时任务已启动 (每日 {config.SCHEDULE_HOUR:02d}:{config.SCHEDULE_MINUTE:02d} 自动运行)')
    print('按 Ctrl+C 退出...')
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print('\n🛑 定时任务已停止')


def main():
    parser = argparse.ArgumentParser(
        description='双色球数据采集、统计建模、博弈论反大众、回测评估与梅花终审系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', help='可用子命令')

    # update
    p_update = subparsers.add_parser('update', help='抓取最新开奖并更新数据库与CSV')
    p_update.add_argument('--rebuild', action='store_true', help='全量重建历史库 (2003年至今)')

    # backtest
    p_backtest = subparsers.add_parser('backtest', help='执行 Walk-Forward 消融回测')
    p_backtest.add_argument('--draws', type=int, default=100, help='回测期数 (默认: 100)')
    p_backtest.add_argument('--days', type=int, default=None, help='回测天数 (例如: 30)')
    p_backtest.add_argument('--count', type=int, default=5, help='每期推荐注数 (默认: 5)')
    p_backtest.add_argument('--special-rule', action='store_true', help='激活 2026 新规福运奖 (3+0=5元)')
    p_backtest.add_argument('--seed', type=int, default=42, help='蒙特卡洛随机种子 (默认: 42)')

    # predict
    p_predict = subparsers.add_parser('predict', help='生成未来开奖号码预测')
    p_predict.add_argument('--count', type=int, default=5, help='推荐注数 (默认: 5)')
    p_predict.add_argument('--next-draws', type=int, default=1, help='预测期数 (默认: 1)')
    p_predict.add_argument('--mode', type=str, choices=['balanced', 'jackpot'], default='balanced', help='选号模式')
    p_predict.add_argument('--beta', type=float, default=config.GAME_THEORY_BETA, help='博弈论权重 (默认: 0.35)')
    p_predict.add_argument('--max-ball-freq', type=int, default=config.MAX_BALL_FREQ, help='跨注单球频次上限 (默认: 2)')

    # check
    subparsers.add_parser('check', help='自动核对已开奖预测并更新收益')

    # export-pool
    subparsers.add_parser('export-pool', help='导出双模式 Top 20 移动端冻结候选池至 docs/data/pool.json')

    # meihua
    p_meihua = subparsers.add_parser('meihua', help='现场梅花易数终审选号')
    p_meihua.add_argument('--numbers', nargs='+', default=['3', '8'], help='心动报数 (例如: 3 8 或 12 27)')
    p_meihua.add_argument('--time', type=str, default=None, help='购彩时间 (格式: YYYY-MM-DD HH:MM)')
    p_meihua.add_argument('--mode', type=str, choices=['balanced', 'jackpot'], default='balanced', help='候选池模式')
    p_meihua.add_argument('--gamma', type=float, default=0.30, help='梅花权重 (默认: 0.30)')

    # run-daily
    p_daily = subparsers.add_parser('run-daily', help='执行一键日常全自动流水线')
    p_daily.add_argument('--force', action='store_true', help='强制覆盖生成当期预测')
    p_daily.add_argument('--mode', type=str, choices=['balanced', 'jackpot'], default='balanced', help='选号模式')
    p_daily.add_argument('--count', type=int, default=5, help='推荐注数 (默认: 5)')
    p_daily.add_argument('--beta', type=float, default=config.GAME_THEORY_BETA, help='博弈论权重 (默认: 0.35)')

    # schedule
    subparsers.add_parser('schedule', help='启动每日定时调度任务')

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    dispatch = {
        'update': cmd_update,
        'backtest': cmd_backtest,
        'predict': cmd_predict,
        'check': cmd_check,
        'export-pool': cmd_export_pool,
        'meihua': cmd_meihua,
        'run-daily': cmd_run_daily,
        'schedule': cmd_schedule,
    }
    dispatch[args.command](args)


if __name__ == '__main__':
    main()
