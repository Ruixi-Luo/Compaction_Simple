# Compaction Simple

独立的五条件、单次压缩实验。主入口只有 `run.py`：准备题目 → 压缩 → 预测 → 指标与报告。

## 2026-10-26 完整 Qwen 重跑

- HotpotQA / QASPER 各100题，固定原来的样本与随机种子。
- Full、No context、SC、LLMLingua-2、Qwen summary；仍是五条件、5×5矩阵。
- 摘要与reader均为 `Qwen/Qwen3-30B-A3B-Instruct-2507`。
- 每份正文每个方法只压缩一次，压缩不看query；prompt和生成参数沿用上轮。50%是原生参数/摘要目标，真实保留率照实报告，不修订或截断。
- 完整主流程现在默认两路API并发，请求启动至少间隔1秒（包括网络重试）；预算共享5元，不是每路5元。费用保护基于估算，不是服务商的人民币硬上限。
- 本地GPU压缩仍按方法顺序运行。并发和请求间隔仅改变调度，不保证服务端输出逐字可复现。

## 本机 CMD 运行

```bat
cd /d "D:\Research\Ph.D\2026.8.4-context compression\Experiments\2026.10.26 Simple Exp\Compaction_Simple"
call .venv\Scripts\activate.bat
set PYTHONHASHSEED=0
set PYTHONIOENCODING=utf-8
python run.py --stage prepare --offline
```

以上只准备题目，不调用模型。确认要付费后执行：

```bat
python -u run.py --stage all --allow-api
```

输出目录：`runs/qwen100_20261026/`。同一目录重启会命中本轮缓存；不要同时启动两个写入同目录的程序。每次新的独立实验应换 `config.json` 的 `run_name`；已有身份冻结后不能修改源码/配置再续跑。

默认不传 `--allow-api` 就不会付费。`--offline` 还禁止缺少缓存时运行GPU，不能与 `--allow-api` 同用。

结果完整后，免费重算同一版本的报告：

```bat
python run.py --stage evaluate --offline
```

## 运行时看到什么

每完成一次压缩或预测条件，终端显示完成量、该次启动累计分钟、API输入和输出token。`telemetry.json` 保存阶段时间、各次启动时间、API用量和费用估计。

- 时间按实际执行阶段记录；暂停后再启动的空闲时间不算执行时长。异常退出未及时记录的尾部时长不会伪造。
- token来自服务商usage，不是本地输入文本长度；输入包括命中缓存部分，二者不重复相加。
- 本轮缓存重放不新增token；缺失usage明确为未知；并发请求耗时之和不是总运行时间。
- 只读查看进度：`python -B tools/monitor_progress.py`。不要持续用PowerShell `Get-Content`读取正在更新的JSON账本。

## 报告内容

`Report.html` 默认英文，可切中文，图和数据内嵌、无需联网：

1. 时间、token、费用概览；Accuracy/Brier/Log loss/实际保留率。
2. 两张双向InfoGap矩阵：行方法相对列方法的最大优势。
3. 相对Full的阈值收益差曲线，保留断点与跳跃。
4. 五等宽箱可靠性图、ECE、每箱样本数和预测概率分布。
5. 实际保留率与Brier、双向InfoGap散点（单个预算点，不是完整前沿）。
6. 常规指标和相对Full差值的95%配对正文簇bootstrap区间。
7. 容忍度—可比性附图；逐题展开的文本与概率。

概率是模型生成的数值预测，不预设为真实校准后验。每数据集100题只是pilot；bootstrap区间是描述性不确定性，不证明Blackwell支配；ECE依赖分箱。InfoGap保存factor-2和unscaled两种尺度，图中用factor-2。

## 代码阅读主线

| 文件 | 作用 |
|---|---|
| `run.py` | 四阶段主流程、顺序保存和独占运行锁 |
| `data.py` | 固定数据筛选和tokenizer |
| `compressors/` | 上游SC/LLMLingua-2的薄封装与单次摘要接口 |
| `reader.py` / `streaming.py` | 实际prompt、API缓存、完整响应检查和预算保护 |
| `tools/concurrent_api.py` | 有界并发、独立worker账本、共享预算和启动限速 |
| `telemetry.py` | 主线程保存时间与usage统计 |
| `metrics.py` | InfoGap、常规分数、校准和配对bootstrap |
| `report.py` / `report_charts.py` | HTML与离线SVG图；不调用模型 |
| `storage.py` | JSON和原子写入重试 |

`tools/replace_summary.py` 是上轮增量替换工具，不是这轮重跑入口。当前 `run.py` 已经直接使用Qwen摘要，无需先跑DeepSeek再替换。

## 原结果与代码备份

全部旧结果已移动到本项目旁边 `../backup_20261026/runs/`，三组运行的相对目录结构保持不变；旧源码在同一备份的 `source/`。`runs_sha256.json` 记录3744个旧文件的哈希。历史manifest中的绝对来源路径保留为当时记录，未改写；查看旧缓存请按备份目录定位。

新版图表的历史数据预览位于 `.cache/report_preview_20261026/Report.html`，显著标注旧数据；它不是这次重跑结果。

## 资源与安全

`resources.json` 仅本机使用，指向D盘共享数据、模型和NLTK缓存；不导入旧项目代码。模型版本见 `models.lock.json`。新电脑需复制 `resources.example.json`、准备资源并安装Python 3.12及对应依赖；仓库不含一键资源下载器。

本机虚拟环境中的大型不可变库可能通过硬链接共享；要升级库时新建环境，不原地升级。`requirements.txt` 是版本记录，CUDA torch与spaCy模型需对应安装源。

密钥读取 `SILICONFLOW_API_KEY` 或本项目 `.env`。Git忽略密钥、资源路径、虚拟环境、缓存、备份和原始结果；它们不随源码推送。

## 离线测试

```bat
set PYTHONHASHSEED=0
python -B -m unittest discover -s tests -v
```

测试用模拟响应，不发API、不加载GPU模型；包括图表数值、分箱边界、配对bootstrap、并发预算、query边界、缓存与Windows文件写入。
