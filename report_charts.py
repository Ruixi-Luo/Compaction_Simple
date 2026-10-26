"""Small offline SVG figures; all plotted numbers come from metrics.json."""
import html
import math

COLORS = {'full': '#26384a', 'no_context': '#8a939c', 'selective_context': '#2b8b80',
          'llmlingua2': '#4979bf', 'llm_summary': '#b67728'}


def plot(series, xlabel, ylabel, xlim=(0, 1), ylim=None, diagonal=False, zero=False, xticks=None):
    """series: name/color/points plus optional line, step, point labels and y intervals."""
    all_y = [p[1] for s in series for p in s['points']]
    all_y += [v for s in series for bounds in s.get('intervals', []) for v in bounds]
    if ylim is None:
        lo, hi = min([0, *all_y]), max([0, *all_y])
        pad = max((hi-lo)*.12, .01)
        raw_step = (hi-lo+2*pad)/5
        power = 10**math.floor(math.log10(raw_step))
        step = next(v*power for v in (1, 2, 2.5, 5, 10) if v*power >= raw_step)
        ylim = (math.floor((lo-pad)/step)*step, math.ceil((hi+pad)/step)*step)
        yticks = [ylim[0]+i*step for i in range(round((ylim[1]-ylim[0])/step)+1)]
    else:
        yticks = [ylim[0]+(ylim[1]-ylim[0])*i/5 for i in range(6)]
    x0, x1 = xlim
    y0, y1 = ylim
    if x1 == x0:
        x1 = x0+1
    if y1 == y0:
        y1 = y0+1
    X = lambda x: 72+(x-x0)/(x1-x0)*590
    Y = lambda y: 255-(y-y0)/(y1-y0)*210
    svg = '<svg viewBox="0 0 740 330" role="img" aria-label="'+html.escape(ylabel, quote=True)+'">'
    for y in yticks:
        svg += f'<line x1="72" y1="{Y(y):.2f}" x2="662" y2="{Y(y):.2f}" stroke="#e1e6ed"/>'
        svg += f'<text x="64" y="{Y(y)+4:.2f}" text-anchor="end" font-size="12">{y:.3g}</text>'
    for x in (xticks if xticks is not None else [x0+(x1-x0)*i/5 for i in range(6)]):
        svg += f'<text x="{X(x):.2f}" y="275" text-anchor="middle" font-size="12">{x:.3g}</text>'
    if zero and y0 <= 0 <= y1:
        svg += f'<line x1="72" y1="{Y(0)}" x2="662" y2="{Y(0)}" stroke="#6c7e8e" stroke-dasharray="5 4"/>'
    if diagonal:
        svg += f'<line x1="{X(0)}" y1="{Y(0)}" x2="{X(1)}" y2="{Y(1)}" stroke="#8997a5" stroke-dasharray="5 4"/>'
    svg += '<path d="M72 45 V255 H662" fill="none" stroke="#52677c"/>'
    for s in series:
        color, points = s['color'], s['points']
        if s.get('line', True) and points:
            drawn = []
            for i, p in enumerate(points):
                if s.get('step') and i:
                    drawn.append((p[0], points[i-1][1]))
                drawn.append(p)
            coords = ' '.join(f'{X(x):.2f},{Y(y):.2f}' for x, y in drawn)
            svg += f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>'
        if s.get('markers', not s.get('line', True)):
            for i, (x, y) in enumerate(points):
                label = s.get('labels', [s['name']]*len(points))[i]
                if s.get('intervals'):
                    low, high = s['intervals'][i]
                    svg += f'<line x1="{X(x)}" x2="{X(x)}" y1="{Y(low)}" y2="{Y(high)}" stroke="{color}" stroke-width="2"/>'
                svg += f'<circle cx="{X(x)}" cy="{Y(y)}" r="4" fill="{color}"><title>{html.escape(label)}: ({x:.4f}, {y:.4f})</title></circle>'
                if s.get('show_labels'):
                    svg += f'<text x="{X(x)+6}" y="{Y(y)-7}" font-size="11" fill="{color}">{html.escape(label)}</text>'
    svg += f'<text x="367" y="306" text-anchor="middle" font-size="14">{html.escape(xlabel)}</text>'
    svg += f'<text x="72" y="23" font-size="14">{html.escape(ylabel)}</text></svg>'
    legend = ' · '.join(f'<span style="color:{s["color"]}">{html.escape(s["name"])}</span>' for s in series)
    return svg+'<div class="muted">'+legend+'</div>'


def diagnostics(metrics, methods, labels, bi, table):
    """Return charts and accompanying numeric tables, without reading predictions or API."""
    body = '<section><h3>'+bi('Decision payoff across thresholds', '不同阈值下的决策收益差')+'</h3>'
    body += '<p>'+bi('Above zero favors compression; below zero favors Full. Exact breakpoints and jumps are retained. The upper extreme is G(method, Full); the negative lower extreme is G(Full, method).',
        '零线上方有利于压缩方法，下方有利于Full；保留真实断点和跳跃。上端极值为G(方法,Full)，下端极值的相反数为G(Full,方法)。')+'</p>'
    body += '<div class="grid">'
    for method in methods:
        if method == 'full':
            continue
        points = [(p['threshold'], -p['payoff_gap']) for p in metrics['payoff_curves_vs_full'][method]]
        ab, ba = metrics['infogap'][method]['full'], metrics['infogap']['full'][method]
        body += '<div class="card"><h4>'+labels[method]+'</h4>'
        body += plot([{'name': labels[method]+' − Full', 'color': COLORS[method], 'points': points}],
                     'Decision threshold t', 'Payoff difference', zero=True)
        body += f'<p>G(method, Full)={ab["value"]:.4f}, t={ab["threshold"]:.4g} ({ab["side"]})<br>'
        body += f'G(Full, method)={ba["value"]:.4f}, t={ba["threshold"]:.4g} ({ba["side"]})</p></div>'
    body += '</div></section>'

    body += '<section><h3>'+bi('Calibration and probability distribution', '校准与预测概率分布')+'</h3>'
    body += '<p>'+bi('Five fixed equal-width bins; p=1 belongs to the last bin. Empty bins are omitted. Frequencies are sample estimates, not known true probabilities. n labels show bin support; no smoothing.',
        '五个固定等宽区间，p=1归入最后一箱；空箱不画。纵轴为样本Yes比例，不是真实后验。标注n显示每箱样本量，不平滑。')+'</p><div class="grid">'
    for method in methods:
        cal = metrics['calibration'][method]
        bins = [b for b in cal['bins'] if b['n']]
        body += '<div class="card"><h4>'+labels[method]+f' · ECE={cal["ece"]:.4f}</h4>'
        body += plot([{'name': labels[method], 'color': COLORS[method],
            'points': [(b['mean_probability'], b['yes_frequency']) for b in bins],
            'labels': [f'n={b["n"]}' for b in bins], 'line': False, 'show_labels': True}],
            'Mean predicted p(Yes)', 'Observed Yes frequency', ylim=(0, 1), diagonal=True)
        body += table(['Probability bin', 'Count', 'Mean p', 'Yes frequency'],
            [[f'[{b["low"]:.1f}, {b["high"]:.1f}'+(']' if b['high'] == 1 else ')'), b['n'],
              '—' if b['mean_probability'] is None else f'{b["mean_probability"]:.3f}',
              '—' if b['yes_frequency'] is None else f'{b["yes_frequency"]:.3f}'] for b in cal['bins']])
        distribution = [(float(p), count) for p, count in cal['exact_counts'].items()]
        body += plot([{'name': 'Exact output frequencies', 'color': COLORS[method], 'points': distribution,
                       'line': False, 'show_labels': False}], 'Predicted p(Yes)', 'Number of questions')
        body += '</div>'
    body += '</div></section>'

    body += '<section><h3>'+bi('Actual retention and performance', '实际保留率与性能')+'</h3>'
    body += '<p>'+bi('One observed operating point per method, not a budget-response frontier. Input lengths are measured with the same Qwen tokenizer.',
        '每方法仅一个实际运行点，不是多预算性能曲线；输入长度统一使用Qwen tokenizer计数。')+'</p><div class="grid">'
    for key, title in [('brier', 'Brier (lower is better)'), ('full_over', 'G(Full, method)'), ('method_over', 'G(method, Full)')]:
        series = []
        for m in methods:
            y = metrics['scores'][m]['brier'] if key == 'brier' else metrics['infogap']['full'][m]['value'] if key == 'full_over' else metrics['infogap'][m]['full']['value']
            series.append({'name': labels[m], 'color': COLORS[m],
                           'points': [(metrics['scores'][m]['pooled_retained_ratio'], y)], 'line': False})
        xmax = max(1., *(metrics['scores'][m]['pooled_retained_ratio'] for m in methods))
        body += '<div class="card">'+plot(series, 'Actual retained token fraction', title, xlim=(0, xmax*1.05))+'</div>'
    body += '</div></section>'

    body += '<section><h3>'+bi('Paired metric differences vs Full', '相对Full的配对指标差值')+'</h3>'
    body += '<p>'+bi('95% descriptive paired context-cluster bootstrap percentile intervals. Positive accuracy difference favors the method; negative Brier / log loss / ECE difference favors it. These are not dominance tests or simultaneous confidence bands.',
        '95%描述性配对正文簇bootstrap百分位区间。Accuracy差值正值更好；Brier/Log loss/ECE差值负值更好。不是支配检验，也不是同时置信带。')+'</p><div class="grid">'
    comparisons = [m for m in methods if m != 'full']
    for metric in ('accuracy', 'brier', 'log_loss', 'ece'):
        series = [{'name': f'{i+1}: '+labels[m], 'color': COLORS[m],
                   'points': [(i+1, metrics['scores'][m][metric+'_delta_vs_full'])],
                   'intervals': [metrics['score_intervals'][m]['delta_vs_full'][metric]], 'line': False}
                  for i, m in enumerate(comparisons)]
        body += '<div class="card">'+plot(series, 'Method index (legend order)', 'Delta '+metric,
                                           xlim=(.5, len(comparisons)+.5), zero=True, xticks=list(range(1, len(comparisons)+1)))+'</div>'
    body += '</div>'
    body += table(['Method', 'Metric', 'Value [95% interval]', 'Delta vs Full [95% interval]'],
        [[labels[m], k, f'{metrics["scores"][m][k]:.4f} '+str([round(v,4) for v in metrics['score_intervals'][m]['absolute'][k]]),
          f'{metrics["scores"][m][k+"_delta_vs_full"]:+.4f} '+str([round(v,4) for v in metrics['score_intervals'][m]['delta_vs_full'][k]])]
         for m in methods for k in ('accuracy', 'brier', 'log_loss', 'ece')])
    body += '</section><section><h3>'+bi('Appendix: tolerance and comparability', '附录：容忍度与可比性')+'</h3>'
    curve = metrics['comparability']
    body += '<p>'+bi('Only 10 unordered method pairs per dataset, sharing the same questions. Tolerance s is an allowed payoff difference, not a p-value. Comparable: min(Gab,Gba) ≤ s; strictly comparable also requires max(Gab,Gba) > s.',
        '每个数据集仅10对无序方法，共享同一批题目。s为允许的收益差，不是p值。可比条件min(Gab,Gba)≤s；严格可比还要求max(Gab,Gba)>s。')+'</p>'
    body += plot([{'name': name, 'color': color, 'points': [(x['tolerance'], x[key]) for x in curve], 'step': True}
                  for key, name, color in [('comparable', 'Comparable', '#397e9a'), ('strictly_comparable', 'Strictly comparable', '#b67728')]],
                 'Tolerance s (factor-2 scale)', 'Fraction of method pairs',
                 xlim=(0, max(.1, curve[-1]['tolerance'])), ylim=(0, 1))
    return body+'</section>'
