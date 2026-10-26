"""InfoGap 核心数学：不依赖 LLM、tokenizer 或网络，可单独审计。"""
import math
import random
from collections import defaultdict


def _validate(probabilities, labels):
    if not probabilities or len(probabilities) != len(labels):
        raise ValueError("Nonempty, equal-length prediction and label arrays are required.")
    if any(type(p) not in {int, float} or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Probabilities must be finite numbers in [0, 1].")
    if any(type(y) not in {int, bool} or y not in {0, 1} for y in labels):
        raise ValueError("Labels must be binary.")


def infogap(probabilities_a, probabilities_b, labels):
    """估计 G(A,B)：A 相对 B 的最大决策收益优势，通常不对称。

    h_t(p,y) = (t-p)_+ + (p-y) 1[p<=t] = (t-y) 1[p<=t]
    G(A,B) = 2 sup_t mean(h_t(p_A,y) - h_t(p_B,y)).

    系数 2 采用 arXiv:2507.12094v2 定理 4.1 和定理 5.1 的证明；
    定理 5.1 展示的 estimator 漏写/采用不同尺度，故同时保存 unscaled。
    在全部概率断点及左右极限求 supremum，不使用粗网格或四舍五入。
    """
    _validate(probabilities_a, labels)
    _validate(probabilities_b, labels)
    # 对每个概率断点，记录通过该点时线性函数 D(t)=slope*t+intercept 的变化。
    events = defaultdict(lambda: [0, 0])
    for a, b, y in zip(probabilities_a, probabilities_b, labels):
        events[a][0] += 1
        events[a][1] -= y
        events[b][0] -= 1
        events[b][1] += y
    events[0.0]
    events[1.0]
    slope, intercept = 0, 0
    best = {"unscaled": 0.0, "threshold": 0.0, "side": "constant_action"}
    n = len(labels)
    for threshold in sorted(events):
        # 在断点左边，尚未计入 p==threshold 的样本。
        if threshold > 0:
            value = (slope * threshold + intercept) / n
            if value > best["unscaled"]:
                best = {"unscaled": value, "threshold": threshold, "side": "left_limit"}
        slope += events[threshold][0]
        intercept += events[threshold][1]
        # 此处实现论文的 <= 约定；允许概率完全相同或等于 0/1。
        value = (slope * threshold + intercept) / n
        if value > best["unscaled"]:
            best = {"unscaled": value, "threshold": threshold, "side": "at_threshold"}
    return {"value": 2 * best["unscaled"], **best, "n": n, "normalization_factor": 2}


def forecast_scores(probabilities, labels):
    _validate(probabilities, labels)
    n = len(labels)
    # 仅 Log loss 使用 clipping；InfoGap、Brier 和原始输出均不改概率。
    clipped = [min(1 - 1e-12, max(1e-12, p)) for p in probabilities]
    return {
        "accuracy": sum(int(p >= 0.5) == y for p, y in zip(probabilities, labels)) / n,
        "brier": sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / n,
        "log_loss": -sum(y * math.log(p) + (1 - y) * math.log(1 - p) for p, y in zip(clipped, labels)) / n,
    }


def calibration_summary(probabilities, labels, bins=5):
    """Fixed equal-width bins; empirical frequencies, not known true posteriors."""
    _validate(probabilities, labels)
    buckets = [[] for _ in range(bins)]
    for p, y in zip(probabilities, labels):
        buckets[min(int(p*bins), bins-1)].append((p, y))
    points, ece = [], 0.0
    for i, bucket in enumerate(buckets):
        n = len(bucket)
        mean = sum(p for p, y in bucket)/n if n else None
        frequency = sum(y for p, y in bucket)/n if n else None
        if n:
            ece += n/len(labels)*abs(mean-frequency)
        points.append({'low': i/bins, 'high': (i+1)/bins, 'n': n,
                       'mean_probability': mean, 'yes_frequency': frequency})
    return {'bins': points, 'ece': ece, 'bin_count': bins,
            'exact_counts': {str(p): probabilities.count(p) for p in sorted(set(probabilities))}}


def score_intervals(probabilities, labels, groups, repetitions, seed, bins=5):
    """Paired context-cluster bootstrap: identical sampled indices across methods."""
    members = defaultdict(list)
    for i, group in enumerate(groups):
        members[group].append(i)
    rng, keys = random.Random(seed), list(members)
    metrics = ('accuracy', 'brier', 'log_loss', 'ece')
    draws = {m: {k: [] for k in metrics} for m in probabilities}
    deltas = {m: {k: [] for k in metrics} for m in probabilities}
    for _ in range(repetitions):
        indices = [i for group in rng.choices(keys, k=len(keys)) for i in members[group]]
        y = [labels[i] for i in indices]
        scores = {}
        for method, p in probabilities.items():
            sampled = [p[i] for i in indices]
            scores[method] = {**forecast_scores(sampled, y),
                              'ece': calibration_summary(sampled, y, bins)['ece']}
        for method in probabilities:
            for metric in metrics:
                draws[method][metric].append(scores[method][metric])
                deltas[method][metric].append(scores[method][metric]-scores['full'][metric])

    def interval(values):
        values = sorted(values)
        return [values[int(.025*(len(values)-1))], values[int(.975*(len(values)-1))]]

    return {m: {'absolute': {k: interval(v) for k, v in draws[m].items()},
                'delta_vs_full': {k: interval(v) for k, v in deltas[m].items()}}
            for m in probabilities}


def comparability_curve(matrix, methods):
    """Descriptive tolerance sweep over unordered pairs, not a significance test."""
    pairs = [(matrix[a][b]['value'], matrix[b][a]['value'])
             for i, a in enumerate(methods) for b in methods[i+1:]]
    thresholds = sorted({0., *(v for pair in pairs for v in pair)})
    return [{'tolerance': s, 'pairs': len(pairs),
             'comparable': sum(min(a, b) <= s for a, b in pairs)/len(pairs),
             'strictly_comparable': sum(min(a, b) <= s < max(a, b) for a, b in pairs)/len(pairs)}
            for s in thresholds]


def payoff_curve(a, b, labels):
    """两动作决策的收益差曲线；保留断点左极限和断点值，不平滑跳跃。"""
    _validate(a, labels)
    _validate(b, labels)
    events = defaultdict(lambda: [0, 0])
    for pa, pb, y in zip(a, b, labels):
        events[pa][0] += 1
        events[pa][1] -= y
        events[pb][0] -= 1
        events[pb][1] += y
    events[0.0]
    events[1.0]
    slope, intercept, points = 0, 0, []
    for threshold in sorted(events):
        if threshold > 0:
            points.append({'threshold': threshold, 'side': 'left_limit',
                           'payoff_gap': 2 * (slope * threshold + intercept) / len(labels)})
        slope += events[threshold][0]
        intercept += events[threshold][1]
        points.append({'threshold': threshold, 'side': 'at_threshold',
                       'payoff_gap': 2 * (slope * threshold + intercept) / len(labels)})
    return points


def bootstrap_gaps(a, b, labels, groups, repetitions=500, seed=20261017):
    """按上下文成组、两方法配对重采样；区间仅描述 pilot 波动，不证明支配。"""
    _validate(a, labels)
    _validate(b, labels)
    if len(groups) != len(labels) or repetitions < 2:
        raise ValueError("Invalid bootstrap settings.")
    members = defaultdict(list)
    for index, group in enumerate(groups):
        members[group].append(index)
    keys = list(members)
    rng = random.Random(seed)
    draws_ab, draws_ba = [], []
    for _ in range(repetitions):
        indices = [i for group in rng.choices(keys, k=len(keys)) for i in members[group]]
        sampled_a = [a[i] for i in indices]
        sampled_b = [b[i] for i in indices]
        sampled_y = [labels[i] for i in indices]
        draws_ab.append(infogap(sampled_a, sampled_b, sampled_y)["value"])
        draws_ba.append(infogap(sampled_b, sampled_a, sampled_y)["value"])

    def interval(values):
        values = sorted(values)
        return [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]

    return {"a_over_b": interval(draws_ab), "b_over_a": interval(draws_ba),
            "repetitions": repetitions, "context_clusters": len(keys),
            "interpretation": "Descriptive paired cluster-bootstrap percentile interval, not a dominance test."}

def analyze(rows, config):
    if len(rows) != config["sample_size"] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Incomplete or duplicate samples: refusing to report a silently reduced dataset.")
    methods = config["methods"]
    if any(set(r["conditions"]) != set(methods) for r in rows):
        raise ValueError("Conditions must match the configured methods on every sample.")
    labels = [r["label"] for r in rows]
    groups = [r["context_group"] for r in rows]
    probabilities = {method: [r["conditions"][method]["p_yes"] for r in rows] for method in methods}
    scores = {}
    for method in methods:
        conditions = [r["conditions"][method] for r in rows]
        # 论文的多个问题复用同一压缩结果；耗时不能按题目重复累计。
        unique_compressions = {c['local_compression_hash']: c['compression_seconds']
                               for c in conditions if 'local_compression_hash' in c}
        scores[method] = {
            **forecast_scores(probabilities[method], labels),
            "mean_context_tokens": sum(c["tokens"] for c in conditions) / len(rows),
            "pooled_retained_ratio": sum(c["tokens"] for c in conditions) / sum(r["original_tokens"] for r in rows),
            "over_budget_samples": sum(c["budget_met"] is False for c in conditions),
            "length_repairs": sum(c["length_repair_used"] for c in conditions),
            "unique_compressions": len(unique_compressions),
            "compression_seconds_unique_total": sum(unique_compressions.values()),
        }
    for method in methods:
        for metric in ('accuracy', 'brier', 'log_loss'):
            scores[method][metric + '_delta_vs_full'] = scores[method][metric] - scores['full'][metric]
    matrix = {a: {b: infogap(probabilities[a], probabilities[b], labels) for b in methods} for a in methods}
    uncertainty = {method: bootstrap_gaps(
        probabilities["full"], probabilities[method], labels, groups,
        config["bootstrap_repetitions"], config["bootstrap_seed"],
    ) for method in methods if method != "full"}
    curves = {m: payoff_curve(probabilities['full'], probabilities[m], labels)
              for m in methods if m != 'full'}
    bins = config.get('reliability_bins', 5)
    calibration = {m: calibration_summary(probabilities[m], labels, bins) for m in methods}
    intervals = score_intervals(probabilities, labels, groups,
                               config['bootstrap_repetitions'], config['bootstrap_seed'], bins)
    for m in methods:
        scores[m]['ece'] = calibration[m]['ece']
        scores[m]['ece_delta_vs_full'] = calibration[m]['ece']-calibration['full']['ece']
    subsets = {}
    for task in sorted({r['source_task'] for r in rows if 'source_task' in r}):
        indices = [i for i, r in enumerate(rows) if r.get('source_task') == task]
        y = [labels[i] for i in indices]
        p = {m: [probabilities[m][i] for i in indices] for m in methods}
        subsets[task] = {'n': len(indices), 'positive_labels': sum(y),
                         'scores': {m: forecast_scores(p[m], y) for m in methods},
                         'infogap_vs_full': {m: {'full_over_method': infogap(p['full'], p[m], y),
                                                'method_over_full': infogap(p[m], p['full'], y)}
                                             for m in methods if m != 'full'}}
    return {"n": len(rows), "positive_labels": sum(labels), "scores": scores,
            'calibration': calibration, 'score_intervals': intervals,
            'comparability': comparability_curve(matrix, methods),
            "context_clusters": len(set(groups)), "payoff_curves_vs_full": curves, "source_tasks": subsets,
            "infogap": matrix, "bootstrap_vs_full": uncertainty,
            "normalization": "2 * sup_t mean[(t-y)(1[p_A<=t]-1[p_B<=t])]; also save unscaled.",
            "scope": "Small pilot of verbalized probabilities, not true-posterior recovery or a dominance test."}
