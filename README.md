# 🎱 双色球预测系统 (SSQ Lottery Prediction System)

基于统计建模、机器学习、博弈论反大众防撞号系统以及**梅花易数河图五行终审**的中国福利彩票双色球分析、云端流水线与手机端离线选号系统。

> ⚠️ **免责声明**: 本系统仅供学习、数据建模与概率研究使用。彩票本质上是极低概率的伪随机事件，任何统计与机器学习模型均无法保证中奖。请理性购彩，切勿沉迷。

---

## 🌟 核心特性与架构

```
               [ 云端低频算力 (GitHub Actions) ]
     每周一、三、五 08:05 定时自动触发 (免开家庭电脑)
           ↓                               ↓
   增量抓取最新开奖 & 自动核对      训练模型 + 博弈论防撞号过滤
                                           ↓
                                导出当期 Top-20 冻结候选池
                             (docs/data/pool.json 静态发布)
                                           ↓
         -------------------------------------------------------
                                           ↓
               [ 手机端高频现场占卜 (GitHub Pages PWA) ]
                纯静态自适应网页，支持“添加到主屏幕”
                                           ↓
               彩票店现场：自动带入购彩时辰 + 输入心动数字
                                           ↓
               【逻辑 A：一念一卦 → 河图五行生克终审】
      从 Top-20 冻结候选池中毫秒级选出最优 5 注 (严禁篡改原注号码)
```

- **全量真实数据基座**: 自动抓取 2003 年双色球发行至今 3,500+ 期数据，SQLite 数据库 + CSV 规范导出。
- **三策略与融合模型**:
  - 策略 A：统计概率加权（基频 + 近期热度 + 遗漏激活 + 和值分布约束）
  - 策略 B：正则化随机森林多标签分类模型（RandomForest）
  - 策略 C：L2 正则化逻辑回归分类模型（LogisticRegression）
  - 融合模型：基于对数似然比（LLR）映射平滑评分（彻底根治 0.0000 坍塌）
- **博弈论“反大众 / 防分奖”系统**:
  - 硬过滤（`AntiPopularityFilter`）：剔除 3,500+ 期历史原号、完整/5连等差数列、4连号、纯月份区扎堆、4同尾号。
  - 独享度软评分（`UniquenessScorer`）：破除生日偏置（32/33加分、<=31扣分）、视觉图形与规律对称惩罚、往期高重叠衰减。
- **标准化梅花易数排盘引擎 (`src/meihua.py` & 前端原生 JS)**:
  - 先天八卦 3-bit 爻象编码（乾1、兑2、离3、震4、巽5、坎6、艮7、坤8）。
  - 时辰自动映射（十二地支子1至亥12）+ 现场心动数字起卦。
  - 主卦、互卦、变卦推演与动爻体用划分（动在1~3上体下用，4~6下体上用）。
  - 河图尾数五行映射（水1/6、火2/7、木3/8、金4/9、土5/0）及五行喜忌生克综合终审。
  - **核心铁律**: 候选池内的 20 注号码均为冻结成品，起卦终审**绝对不修改或替换任何一颗红球或蓝球**；选出的 5 注保证蓝球互不相同、单红球频次严格 $\le 2$。
- **云端全自动免运维**: GitHub Actions 定时执行，更新数据并部署至 GitHub Pages，脱离本地台式机依赖。

---

## 📱 手机端使用指南 (GitHub Pages)

### 1. 开启 GitHub Pages 托管
1. 将本项目推送到你的 GitHub 仓库（设为 Public 或开通 Pages 的 Private 仓库）。
2. 打开仓库页面，进入 **Settings** -> **Pages**。
3. 在 **Build and deployment** 下：
   - Source 选择 **Deploy from a branch**。
   - Branch 选择 `main`，目录选择 `/docs`，点击 **Save**。
4. 几分钟后，你将获得一个在线网址：`https://<你的用户名>.github.io/<仓库名>/`。

### 2. 手机端“添加到主屏幕”（像原生 App 一样打开）
- **iPhone / Safari**:
  1. 在 Safari 中打开 Pages 网址。
  2. 点击底部工具栏的“分享”按钮（向上箭头的方框）。
  3. 下滑选择 **添加到主屏幕 (Add to Home Screen)**。
- **Android / Chrome**:
  1. 在 Chrome 中打开 Pages 网址。
  2. 点击右上角菜单（三个点）。
  3. 选择 **安装应用 (Install app)** 或 **添加到主屏幕**。

### 3. 彩票店现场 3 步锁定号码
1. **核对期号与模式**: 顶部自动同步最新期号与候选池更新时间；可切换 **稳健保底（5蓝去重）** 或 **冲大奖（独享蓝）**。
2. **选择时辰与心动数字**:
   - 默认已自动同步当下时间并换算为十二地支时辰。
   - 输入你当下脑海中浮现的 1~3 个幸运数字（如：`8`、`7 18`、`3 8 15`）。
3. **起卦终审锁定 5 注**:
   - 点击 **“一念一卦 · 河图五行终审”**，毫秒级推演体用五行与喜用倾向。
   - 算法从 Top-20 候选池中自动锁定终审总分最高的 5 注，且单球跨注出现次数 $\le 2$。
   - 点击 **“复制 5 注投注单”**，直接向彩票店老板出示或打票。

---

## 🤖 GitHub Actions 自动云端流水线

项目已内置 `.github/workflows/update_ssq.yml`：
- **触发时间**: 每周一、三、五 **08:05**（北京时间，即双色球开奖次日早晨自动更新）。
- **执行内容**:
  1. `python main.py update`: 增量抓取最新官方开奖并校验入库。
  2. `python main.py check`: 自动对账上期投注并回填中奖奖金与累计 ROI。
  3. `python main.py export-pool`: 重新训练模型，导出当期 Top-20 冻结候选池到 `docs/data/pool.json`。
  4. 自动 `git commit & push` 回仓库，GitHub Pages 自动同步最新候选池。
- **手动触发**: 在 GitHub 仓库的 **Actions** 标签页，选择 *Auto Update SSQ & Export Candidate Pool*，点击 **Run workflow** 即可随时手动触发。

---

## 💻 本地 CLI 命令全览

```bash
# 1. 环境安装
pip install -r requirements.txt

# 2. 导出 Top-20 冻结候选池 (供前端使用)
python main.py export-pool

# 3. 命令行交互式梅花易数终审排盘
python main.py meihua --mode balanced --numbers 7 18

# 4. 一键日常流水线
python main.py run-daily

# 5. 自动对账往期开奖与命中核对
python main.py check

# 6. 数据采集与历史数据全量重建
python main.py update
python main.py update --rebuild

# 7. 长周期消融回测 (100期滚动消融)
python main.py backtest --draws 100 --seed 42

# 8. 运行完整测试套件 (包含梅花易数、防撞号、无泄漏等 78 个测试)
pytest -v
```

---

## 📁 项目目录结构

```
lotto/
├── .github/workflows/
│   └── update_ssq.yml         # GitHub Actions 云端定时任务
├── docs/                      # GitHub Pages 前端静态根目录
│   ├── index.html             # 手机端暗黑质感自适应 PWA 网页
│   ├── manifest.json          # PWA 主屏幕配置文件
│   └── data/
│       └── pool.json          # 当期 Top-20 冻结候选池数据 (Actions 自动更新)
├── main.py                    # CLI 主程序入口
├── config.py                  # 规则、权重、奖级表与路径配置
├── requirements.txt           # 生产与开发依赖
├── src/
│   ├── crawler.py             # 500彩票网分段健壮抓取
│   ├── storage.py             # SQLite 存储、CSV 同步与开奖核对
│   ├── features.py            # 向量化统计特征工程
│   ├── model.py               # 统计/随机森林/逻辑回归及 LLR 平滑评分
│   ├── game_theory.py         # 博弈论硬过滤、独享度软评分与彩民碰撞模拟
│   ├── meihua.py              # 标准化梅花易数排盘、体用五行生克与终审搜索
│   ├── backtest.py            # Walk-Forward 滚动消融回测与蒙特卡洛基准
│   └── predictor.py           # Top-20 候选池导出与预测执行
├── tests/                     # 完整自动化测试套件 (78 passed)
│   ├── test_meihua.py         # 八卦、时辰、体用、河图及选号约束测试
│   ├── test_game_theory.py    # 博弈论过滤与独享度评分测试
│   ├── test_backtest.py       # 回测、奖级与蒙特卡洛测试
│   ├── test_crawler.py        # 爬虫异常抛出与数据保护测试
│   ├── test_features.py       # 特征计算测试
│   ├── test_model.py          # 模型训练、概率平滑与多样性测试
│   └── test_no_data_leakage.py# 滚动回测时序无泄漏严格测试
├── data/                      # 本地数据存储 (自动生成)
│   ├── ssq.db
│   └── ssq_history.csv
└── reports/                   # 预测与回测报告
    ├── latest_pool.json
    └── predictions.csv
```

---

## 📜 许可协议

本项目基于 MIT 协议开源。仅供技术交流与学术研究。
