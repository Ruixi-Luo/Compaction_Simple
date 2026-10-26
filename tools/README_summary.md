> 历史增量工具说明：2026-10-26起完整Qwen重跑请使用根目录 `run.py`。旧结果在项目旁边的 `backup_20261026/runs/`，不直接运行本工具恢复旧身份。

# 只替换摘要模型，仍然是五个条件

`replace_summary.py` 是原实验完成后的增量入口，复用本项目现有函数，不改变 `run.py` 的五条件主流程。它只把 DeepSeek summary 换为 Qwen summary；Full、No context、SC、LLMLingua-2 的 800 个预测逐字段复制。

完整 DeepSeek 结果：`runs/once100_20261025_iofix/Report.html`。更早的 `once100_20261025` 因文件占用中断，原目录完整保留；恢复仅修复文件写入，模型与prompt不变。

新结果：`runs/once100_qwen_summary_20261025/Report.html`。

两个报告分别是五条件、两张 5×5 InfoGap 矩阵。原报告、压缩输出、API 记录和概率全部保留；新报告链接回原报告。摘要 prompt、reader prompt、样本、temperature、thinking、输出上限和流式设置不变，只替换摘要模型。逐条核对原始 API 请求后才允许生成。

补跑使用两路并发，每题仍单独发请求，不把多个问题合并到一个prompt。`concurrent_api.py`复用现有API客户端；两路共用5元费用保护和请求启动限速（至少间隔2秒），每路独立账本与缓存，原总尝试次数上限分配到两路而不翻倍。主线程统一保存压缩结果和预测，并恢复原题目顺序。错误发生后停止安排后续任务；已在途的另一条请求仍会收齐并保存。

## 运行

在项目目录和已激活的 `.venv` 中，确认原实验已正常结束、没有补跑进程后执行：

```bat
python tools/replace_summary.py --stage all --allow-api
```

这会付费：197 篇唯一上下文生成 Qwen 首稿，再完成 200 个摘要条件预测。其他四条件不发请求；完全相同的 reader 请求可直接命中原缓存。只保留完整首稿，不修订、不裁剪，实际长度和超目标数量进入报告。

用户已授权本次额外预留 5 元。新运行账本只记录新增调用，不计入原运行费用；重启不会清空账本，超出预留会停止。该预留沿用代码中的保守单价和未知用量估计，不是服务商的人民币扣费硬上限。

如果后台队列已经安排好，不要再次执行上述命令。状态见 `.cache/qwen_summary_queue_state.json`，输出见 `.cache/qwen_summary_extension.log` 和对应 `.err.log`。

不付费检查/重算：

```bat
python tools/replace_summary.py --stage prepare
python tools/replace_summary.py --stage evaluate
```

省略 `--allow-api` 时，缺少 API 缓存会报错，不会付费。`prepare` 要求原实验已完成。代码和配置被冻结后不得修改；原结果文件也逐一校验，保证没有被覆盖。

本扩展没有复制一套算法实现：直接复用 `reader.py` 的请求与费用保护、`compressors.compress_one` 的单次摘要、`metrics.analyze` 的指标以及 `report.build_report` 的可切中英文报告。仅在新进程中把摘要显示名称设为 Qwen summary。

监控请用 `python tools/monitor_progress.py`，不要持续用PowerShell `Get-Content`读取正在写入的JSON文件。该辅助程序以共享读取方式短时取快照，先关闭文件再解析。实时控制台只读取文本日志，不影响实验生命周期。
