# Compaction Simple

独立的五条件、单次压缩实验项目。**不导入旧仓库的任何实验模块**。旧仓库仅提供 `resources.json` 指定的只读数据、模型和 NLTK 缓存。

## 现在怎么运行

在 Windows CMD 中：

```bat
cd /d "D:\Research\Ph.D\2026.8.4-context compression\Experiments\Compaction_Simple"
call .venv\Scripts\activate.bat
set PYTHONHASHSEED=0
set PYTHONIOENCODING=utf-8
python run.py --stage prepare --offline
```

以上只读取 D 盘已有数据并固定样本，不付费，不运行 GPU。正式跑实验时执行：

```bat
python run.py --stage all --allow-api
```

**这条命令会付费**。预计100+100题约6–10元、3–4.5小时；20元费用预留保持不变。估算沿用历史首稿用量，网络排队、失败重试和新输出长度会改变实际费用和时间。本地已完成 DeepSeek 摘要版和 Qwen 摘要替换版；下面的命令说明面向新的运行。

完整结果已有后，免费重新计算分数和生成HTML：

```bat
python run.py --stage evaluate --offline
```

中断后重新执行同一条 `--stage all --allow-api` 命令，复用**新项目本轮**缓存。不要同时运行两个写入同一目录的进程。默认不传 `--allow-api` 就不能发起付费请求；`--offline` 还禁止本地推理。两者不能一起使用。

## 阅读主线：只从 run.py 开始

```text
config.json
  ↓
run.py: main()
  ├─ prepare()  → data.py：读取、筛选并固定题目
  ├─ compress() → compressors/：每段正文每方法调用一次
  ├─ predict()  → reader.py：文本 + question → p_yes
  └─ evaluate() → metrics.py → report.py：指标、双向矩阵、HTML
```

| 你想理解什么 | 文件 |
|---|---|
| 实验按什么顺序执行、每一步保存什么 | `run.py` |
| 怎样选HotpotQA/QASPER题目 | `data.py` |
| 压缩方法怎样统一调用 | `compressors/__init__.py` |
| 怎样发请求、读缓存、保护费用 | `reader.py` |
| InfoGap与其他分数怎么算 | `metrics.py` |
| 结果怎样显示 | `report.py` |

其余是必要辅助：`storage.py` 管文件；`streaming.py` 处理完整流式响应；压缩器各有薄封装、长文本分块和设备设置。`tests/` 是离线测试，不参与正常实验。`.venv/`、`.cache/` 是环境和临时数据，不是待阅读源码。

## 固定协议

- HotpotQA distractor dev yes/no 与 QASPER train 一致布尔标签、文字证据子集，各100题，seed=20261025。输入保持完整上下文；QASPER证据只定义子集，不裁剪正文。
- 五条件：Full、No context、Selective Context、LLMLingua-2、普通 LLM summary。主流程不含DAC、SALT、Telegrapher；增量摘要替换入口见 `tools/README_summary.md`。
- SC原生 `reduce_ratio=0.5`；LLMLingua-2原生 `rate=0.5`。每个正文只调用一次接口，不做参数试探，不裁剪；算法内部仍可分块推理。
- 摘要沿用原始prompt，目标为原文Qwen token的一半，保留第一份完整输出，不修订。输出超目标也如实保留。不能把截断的响应当完整摘要。
- 压缩器只接收正文与保留率/预算，不能看到question、label或reader反馈。多个问题共享同一正文时只压缩一次。
- Reader：Qwen/Qwen3-30B-A3B-Instruct-2507；摘要：deepseek-ai/DeepSeek-V3.2；temperature=0，thinking=false；原reader prompt不变。
- 五条件配对预测。报告实际保留率和超目标数量，不能称严格等长比较。口头概率不预设为校准后验；两方向InfoGap分别求最大值，保留factor-2和unscaled。
- 没有独立audit阶段。保留样本完整性、配对、概率有效性、请求缓存和费用检查。网络最多三次传输尝试，不保证服务端只生成一次。

## 配置与输出

`config.json`：样本数、种子、模型、费用预留。新实验改 `run_name`，旧结果不覆盖；五条件与比例在当前版本固定，模型替换需重新核对tokenizer和费用。

`resources.json`：指向D盘现有数据目录与NLTK缓存；只共享这些资源，不把旧仓库加入 `sys.path`，不读取旧的 `runs/`。`models.lock.json` 固定两个压缩模型版本。

本轮输出在 **`runs/once100_20261025/`**：

```text
manifest.json                    配置、源码和样本身份
hotpotqa/、qasper/
  examples.json                  固定题目
  compressions/                  单次输出与真实长度
  predictions.json               每题每条件的p_yes
  metrics.json                   指标、双向InfoGap、bootstrap
api/                             真实请求、响应与尝试账本
spending_reservations.json        累计费用预留
Report.html                      默认英文/可切中文，含矩阵和逐题展开
```

只准备数据不产生新的预测，也不自动复制旧实验缓存。两组各100题意味着1000个reader概率、两张5×5矩阵。

## Python环境与API Key

新项目有自己的 `.venv`，激活后不需要旧项目的Python路径。第三方包来自本机已验证环境的独立路径副本，大型DLL等不可变库使用同盘硬链接节省空间；源代码和可写文件使用独立副本。不要在这个环境里原地升级包；要升级时新建环境并重新安装，避免更改共享库文件。

Python 3.12解释器来自本机已安装runtime，数据、模型、环境与结果均位于D盘。`requirements.txt`记录本机第三方版本，CUDA PyTorch及spaCy英语模型需对应安装源；不是建议为了此次运行重装全部依赖。

API Key 读取环境变量 `SILICONFLOW_API_KEY` 或新项目自己的 `.env`，不依赖旧仓库密钥文件；`.env`已经被忽略，不应上传或分享。

## 免费验证

```bat
python -m unittest discover -s tests -v
```

包含单次调用、query边界、缓存身份、无API离线模式、配对结果、InfoGap断点与流式完整性测试。GPU小样本验证也只读取现有权重，不请求LLM API。

## GitHub 版本与本地结果

此仓库保存当前源码、离线测试及配置，不上传 `.env`、虚拟环境、模型/数据缓存、原始 API 响应或 `runs/`。旧项目完整保留。

本机已完成的结果分别在 `runs/once100_20261025_iofix/Report.html`（DeepSeek summary）与 `runs/once100_qwen_summary_20261025/Report.html`（Qwen summary）。摘要替换仍为五条件，复用其余四条件800条预测。增量入口依赖本机已经完成的原运行；新克隆没有这些结果，不能直接执行增量入口。

新电脑需要先创建 Python 3.12 环境、安装依赖、准备数据和模型，并将 `resources.example.json` 复制为 `resources.json` 后修改路径。`requirements.txt` 是本机版本记录，CUDA torch 和 spaCy 模型需要各自对应的安装源；仓库目前不包含自动下载全部资源的一键安装器。模型版本和下载地址见 `models.lock.json`，数据读取布局见 `data.py`。将 `.env.example` 复制为 `.env`，在本地填写密钥。

在本机重跑时，先为新实验设置新的 `run_name`，不要覆盖已有运行。默认 `config.json` 使用 DeepSeek summary；Qwen 替换实验的入口与限制见 [摘要替换说明](tools/README_summary.md)。
