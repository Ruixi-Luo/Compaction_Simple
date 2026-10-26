"""生成可离线分享的中英文结果页。所有图和数据都内嵌，不依赖 CDN。"""
import html
import json

from storage import read_json
from report_charts import diagnostics


def bi(english, chinese):
    return f'<span lang="en">{english}</span><span lang="zh">{chinese}</span>'


STYLE = """
:root{font-family:Segoe UI,Microsoft YaHei,Arial,sans-serif;color:#243445;background:#f4f7fa;line-height:1.65}
body{max-width:1180px;margin:auto;padding:28px}header,section{background:white;padding:26px;margin-bottom:22px;border-radius:14px;border:1px solid #dce5ee}
h1{font-size:30px;line-height:1.3}h2{font-size:22px}h3{font-size:17px}button,select{font:inherit;padding:6px 12px;cursor:pointer}
html[data-language=en] [lang=zh],html[data-language=zh] [lang=en]{display:none}
.toolbar{float:right;display:flex;gap:8px}.note{padding:12px 16px;background:#edf4fb;border-left:4px solid #35789a;margin:12px 0}
.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%}th,td{padding:10px;border-bottom:1px solid #dce5ee;text-align:left;white-space:nowrap}th{background:#f0f5f8}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}.card{padding:15px;border:1px solid #dce5ee;border-radius:9px}
pre,code{font-family:Consolas,monospace}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:14px;font-size:13px}details{border-top:1px solid #dce5ee;padding:12px 0}summary{cursor:pointer;font-weight:600}
a{color:#176b97}svg{width:100%;height:auto}svg text{font-family:Segoe UI,Arial,sans-serif}.muted{color:#5b6c7e;font-size:14px}
@media(max-width:650px){body{padding:10px}header,section{padding:16px}.toolbar{float:none}h1{font-size:25px}}
"""


def table(headers, rows):
    return '<div class="tablewrap"><table><thead><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'



LABELS = {'full':'Full','no_context':'No context','selective_context':'Selective Context','llmlingua2':'LLMLingua-2','llm_summary':'LLM summary'}
def page(title, body, script=''):
    """title：标题；body：已转义 HTML 内容；script：本地交互代码。"""
    return ('<!doctype html><html lang="en" data-language="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+
        '</title><style>'+STYLE+'</style><div class="toolbar"><button onclick="switchLanguage(\'en\')">English</button>'
        '<button onclick="switchLanguage(\'zh\')">中文</button></div>'+body+
        '<script>function switchLanguage(l){document.documentElement.dataset.language=l;document.documentElement.lang=l;}'
        +script+'</script></html>')

def heatmap(metrics, methods, maximum, labels=None):
    """metrics：某数据集指标；methods：矩阵顺序；maximum：跨数据集统一色阶。"""
    labels = labels or LABELS
    rows = []
    for a in methods:
        cells = '<th>'+labels[a]+'</th>'
        for b in methods:
            value = metrics['infogap'][a][b]['value']
            strength = value/maximum if maximum else 0
            color = '#'+''.join(f'{round(x+(y-x)*strength):02x}' for x,y in zip((241,246,250),(32,86,133)))
            cells += f'<td style="background:{color};color:{"white" if strength>.55 else "#172b3b"}" title="G({labels[a]}, {labels[b]})={value}">{value:.4f}</td>'
        rows.append('<tr>'+cells+'</tr>')
    return '<div class="tablewrap"><table class="matrix"><tr><th>Row → column</th>'+''.join('<th>'+labels[m]+'</th>' for m in methods)+'</tr>'+''.join(rows)+'</table></div>'

"""简短的双语结果页；默认英文，展示实际长度与两个方向的 InfoGap。"""
import html
from storage import read_json




def build_report(config, run):
    labels = dict(LABELS)
    labels['llm_summary'] = 'Qwen summary' if 'Qwen' in config['summary']['model'] else 'DeepSeek summary'
    body = '<h1>'+bi('Single-call compaction experiment', '单次压缩实验')+'</h1>'
    if config.get('report_note'):
        body += '<div class="note">'+html.escape(config['report_note'])+'</div>'
    body += '<p>Reader: '+html.escape(config['reader']['model'])+'<br>Summary: '+html.escape(config['summary']['model'])+'</p>'
    body += '<p>'+bi('API workers / minimum request-start interval: ', 'API并发数 / 最小请求启动间隔：')+str(config.get('api_workers', 1))+' / '+str(config['reader']['min_interval_seconds'])+' s</p>'
    timing = run/'telemetry.json'
    if timing.exists():
        t = read_json(timing)
        body += '<section><h2>'+bi('Runtime and API token usage', '运行时间与API token用量')+'</h2>'
        body += '<p>'+bi('Recorded execution time (excludes pauses between invocations): ', '已记录执行时长（不含两次启动之间的暂停）：')+f'{t["execution_seconds"]/60:.2f} min</p>'
        body += table(['Session start (UTC)', 'Stage', 'Status', 'Seconds'],
            [[session['started_utc'], stage['name'], stage['status'], f'{stage["elapsed_seconds"]:.2f}']
             for session in t['sessions'] for stage in session['stages']])
        u=t['api_usage']
        body += table(['Role', 'Attempts', 'Input tokens', 'Output tokens', 'Cached input subset', 'Unknown usage', 'In flight'],
            [[role,u[role]['attempts'],u[role]['input_tokens'],u[role]['output_tokens'],u[role]['cached_input_tokens'],u[role]['unknown_usage'],u[role]['in_flight']] for role in ('summary','reader')])
        body += '<p>'+bi('Provider-reported usage only; cache replay adds no new usage. Cached input is included in input tokens. Missing usage is unknown, not free. Concurrent request durations are not added as wall time.',
            '只统计服务商实际返回的用量；缓存重放不新增用量。缓存命中token已包含在输入token内。缺失用量为未知，不视作免费；不把并发请求耗时之和当墙钟时间。')+'</p>'
        body += '<p>'+bi('Accounted cost / outstanding reservations (CNY): ', '已核算费用及未结算预留（元）：')+f'{u["accounted_cny_including_reservations"]:.4f}'+'</p></section>'
    else:
        body += '<p class="muted">'+bi('This historical run has no new-format wall-time telemetry; no timing is fabricated.', '这个历史运行没有新版墙钟统计，不补造时间。')+'</p>'
    body += '<p>'+bi('One native call at retained_ratio=0.5; summary uses the original prompt and first complete draft. No length search, revisions or clipping. This is not an equal-length comparison.',
        '本地方法以保留率0.5调用一次；摘要沿用原始prompt，保留完整首稿。不做长度搜索、修订或裁剪，不是等长比较。')+'</p>'
    body += '<p>'+bi('Probabilities are verbalized model forecasts, not assumed calibrated posteriors. Row → column: maximum advantage of the row predictor. No new independent audit is claimed.',
        '概率是模型口头预测，不预设为校准后验。行→列表示行方法的最大优势。本版本不声称通过独立全量审计。')+'</p>'
    maximum = max(read_json(run/n/'metrics.json')['infogap'][a][b]['value'] for n in config['datasets'] for a in config['methods'] for b in config['methods'])
    for name in config['datasets']:
        metrics = read_json(run/name/'metrics.json')
        rows = read_json(run/name/'predictions.json')
        body += f'<h2>{html.escape(name)} · n={len(rows)}</h2>'
        values = []
        for method in config['methods']:
            s = metrics['scores'][method]
            values.append([labels[method], f'{s["accuracy"]:.3f}', f'{s["brier"]:.4f}',
                           f'{s["log_loss"]:.4f}', f'{s["pooled_retained_ratio"]:.1%}', str(s['over_budget_samples'])])
        body += table(['Method','Accuracy','Brier','Log loss','Actual retention','Above target'], values)
        body += heatmap(metrics, config['methods'], maximum, labels)
        if 'calibration' in metrics:
            body += diagnostics(metrics, config['methods'], labels, bi, table)
        body += '<h3>'+bi('Saved examples', '保存的逐题结果')+'</h3>'
        for row in rows:
            body += '<details><summary>'+html.escape(row['id']+' · '+row['question'])+'</summary>'
            body += '<p>Label: '+str(row['label'])+'</p>'
            for method in config['methods']:
                c = row['conditions'][method]
                body += f'<h4>{labels[method]} · p_yes={c["p_yes"]} · tokens={c["tokens"]}</h4>'
                body += '<pre style="white-space:pre-wrap">'+html.escape(c['text'])+'</pre>'
            body += '</details>'
    (run/'Report.html').write_text(page('Single-call compaction', body), encoding='utf-8')

