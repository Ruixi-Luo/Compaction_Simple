"""Reuse a finished five-condition run, replacing only its summary with Qwen.

No GPU work and no new sampling. Default is offline; --allow-api permits
only the missing Qwen summaries and their reader predictions (5 CNY reserve).
"""
import argparse
import copy
import html
import os
import sys
import time
from pathlib import Path

PROJECT = Path(os.environ.get('COMPACTION_SIMPLE_PROJECT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT))
import run as pipeline
import report
from compressors import compress_one
from data import ContextTokenizer
from metrics import analyze
from reader import CachedChatClient, predict_one, summary_messages, output_ceiling
from concurrent_api import TwoWorkerClient, network_pacing, bounded_results
from storage import ROOT, read_json, write_json, fingerprint, file_hash

SOURCE_NAME = 'once100_20261025_iofix'
DESTINATION_NAME = 'once100_qwen_summary_20261025'


def snapshot(folder):
    """Record every old result file; refuse to start while its writer holds a lock."""
    if (folder / '.run.lock').exists():
        raise RuntimeError('The original experiment is still locked. Wait for it to finish.')
    return {p.relative_to(folder).as_posix(): file_hash(p)
            for p in sorted(folder.rglob('*')) if p.is_file()}


def verify_source(source):
    manifest = read_json(source / 'manifest.json')
    config = manifest['identity']['config']
    if config['methods'] != pipeline.METHODS or config['retained_ratio'] != .5:
        raise ValueError('Unexpected source protocol.')
    for relative, digest in manifest['identity']['code'].items():
        if file_hash(ROOT / relative) != digest:
            raise ValueError('Original experiment code changed: ' + relative)
    if not (source / 'Report.html').is_file():
        raise ValueError('Original report has not finished.')
    for name in config['datasets']:
        examples = read_json(source / name / 'examples.json')
        rows = read_json(source / name / 'predictions.json')
        metrics = read_json(source / name / 'metrics.json')
        if fingerprint(examples) != manifest['samples'][name]:
            raise ValueError('Original samples changed.')
        if len(rows) != config['sample_size'] or len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Incomplete original predictions.')
        if metrics['n'] != len(rows) or set(metrics['scores']) != set(config['methods']):
            raise ValueError('Incomplete original metrics.')
        for example, row in zip(examples, rows):
            if any(row[k] != v for k, v in example.items()) or set(row['conditions']) != set(config['methods']):
                raise ValueError('Original sample/condition mismatch.')
            for condition in row['conditions'].values():
                p = condition['p_yes']
                if type(p) not in (int, float) or not 0 <= p <= 1:
                    raise ValueError('Invalid original probability.')
    return config


def prepare(source, destination):
    """Freeze the separate run and copy only sample descriptions; old files stay read-only."""
    hashes = snapshot(source)
    original = verify_source(source)
    config = copy.deepcopy(original)
    config['run_name'] = destination.name
    config['summary']['model'] = original['reader']['model']
    config['cost_reserve_cny'] = 5
    identity = {'protocol': 'replace_summary_model_only_v1', 'config': config,
                'source_run': str(source), 'source_files': hashes,
                'extension_sha256': file_hash(__file__),
                'concurrent_api_sha256': file_hash(Path(__file__).with_name('concurrent_api.py')),
                'scheduler': {'workers': 2, 'shared_budget_cny': 5,
                              'request_start_interval_seconds': config['summary']['min_interval_seconds'],
                              'aggregate_attempt_caps_unchanged': True}}
    manifest_path = destination / 'manifest.json'
    if manifest_path.exists():
        if read_json(manifest_path)['identity'] != identity:
            raise ValueError('Frozen extension or source changed. Do not reuse this run.')
    else:
        write_json(manifest_path, {'identity': identity})
        write_json(destination / 'config.json', config)
    for name in config['datasets']:
        for filename in ('examples.json', 'dataset_source.json'):
            value = read_json(source / name / filename)
            target = destination / name / filename
            if target.exists() and read_json(target) != value:
                raise ValueError('Copied samples changed.')
            if not target.exists():
                write_json(target, value)
    return config


def verify_original_prompt(source, name, context, budget, config):
    """Compare with the actual completed DeepSeek request, before any Qwen call."""
    old = pipeline.cached_compression(source / name, context, 'llm_summary', .5)
    if old is None:
        raise ValueError('Missing original summary.')
    request = read_json(source / 'api/summary/api_cache' / (old['request_hash'] + '.json'))['request']
    expected = {'model': 'deepseek-ai/DeepSeek-V3.2',
                'messages': summary_messages(context, budget),
                'temperature': config['summary']['temperature'],
                'max_tokens': output_ceiling(budget),
                'enable_thinking': config['summary']['enable_thinking'],
                'stream': config['summary']['stream']}
    if request != {'base_url': config['summary']['base_url'], 'payload': expected}:
        raise ValueError('Summary prompt/generation settings differ from original request.')


def compress(config, source, destination, tokenizer, allow_api):
    client = TwoWorkerClient(config, 'summary', destination, tokenizer, allow_api)
    for name in config['datasets']:
        examples = read_json(destination / name / 'examples.json')
        contexts = list(dict.fromkeys(e['context'] for e in examples))
        missing = []
        for context in contexts:
            budget = int(tokenizer.count(context) * .5)
            verify_original_prompt(source, name, context, budget, config)
            if pipeline.cached_compression(destination / name, context, 'llm_summary', .5) is not None:
                continue
            missing.append(context)

        def summarize(context):
            start = time.perf_counter()
            result = compress_one(context, 'llm_summary', tokenizer, client=client, retained_ratio=.5)
            result['compression_seconds'] = time.perf_counter() - start
            return result

        completed = len(contexts) - len(missing)
        for context, result in bounded_results(summarize, missing):
            pipeline.cached_compression(destination / name, context, 'llm_summary', .5, result)
            completed += 1
            print(f'Qwen summary {name} {completed}/{len(contexts)}: {result["tokens"]}/{result["budget_tokens"]}', flush=True)


class ReusingReader:
    """Exact request hits may read the old cache; only genuinely new calls use the new budget."""
    def __init__(self, config, source, destination, tokenizer, allow_api):
        self.old = CachedChatClient(config['reader'], source / 'api/reader', offline=True)
        self.new = TwoWorkerClient(config, 'reader', destination, tokenizer, allow_api)

    def complete(self, messages, max_tokens, json_mode=False):
        try:
            return self.old.complete(messages, max_tokens, json_mode)
        except RuntimeError as error:
            if 'Offline replay is missing' not in str(error):
                raise
        return self.new.complete(messages, max_tokens, json_mode)


def predict(config, source, destination, tokenizer, allow_api):
    client = ReusingReader(config, source, destination, tokenizer, allow_api)
    for name in config['datasets']:
        rows = {}
        original_rows = read_json(source / name / 'predictions.json')

        def predict_row(old_row):
            row = copy.deepcopy(old_row)
            condition = pipeline.cached_compression(destination / name, row['context'], 'llm_summary', .5)
            if condition is None:
                raise ValueError('Complete Qwen compression before prediction.')
            forecast = predict_one(condition['text'], row['question'], client, config['reader']['max_tokens'])
            row['conditions']['llm_summary'] = {
                **condition, **forecast,
                'local_compression_hash': fingerprint([row['context'], 'llm_summary']),
                'retained_ratio_actual': condition['tokens'] / row['original_tokens'],
            }
            return row

        for old_row, row in bounded_results(predict_row, original_rows):
            rows[row['id']] = row
            ordered = [rows[old['id']] for old in original_rows if old['id'] in rows]
            write_json(destination / name / 'predictions.json', ordered)
            print(f'Qwen summary reader {name} {len(rows)}/{len(original_rows)}', flush=True)


def evaluate(config, source, destination):
    preserved = 0
    for name in config['datasets']:
        original = read_json(source / name / 'predictions.json')
        rows = read_json(destination / name / 'predictions.json')
        if len(rows) != len(original):
            raise ValueError('Incomplete new predictions.')
        for old, new in zip(original, rows):
            if {k: v for k, v in old.items() if k != 'conditions'} != {k: v for k, v in new.items() if k != 'conditions'}:
                raise ValueError('Sample information changed.')
            for method in config['methods']:
                if method != 'llm_summary':
                    if old['conditions'][method] != new['conditions'][method]:
                        raise ValueError('A reused condition changed: ' + method)
                    preserved += 1
        metrics = analyze(rows, config)
        old_metrics = read_json(source / name / 'metrics.json')
        for a in config['methods'][:-1]:
            if metrics['scores'][a] != old_metrics['scores'][a]:
                raise ValueError('Reused scores changed.')
            for b in config['methods'][:-1]:
                if metrics['infogap'][a][b] != old_metrics['infogap'][a][b]:
                    raise ValueError('Reused InfoGap cell changed.')
        write_json(destination / name / 'metrics.json', metrics)
    report.LABELS['llm_summary'] = 'Qwen summary'
    report.build_report(config, destination)
    target = destination / 'Report.html'
    note = '<section>' + report.bi(
        'Summary model: Qwen3-30B-A3B-Instruct-2507, also used as the reader. Only the summary model changes; prompts and generation settings are unchanged. The other four conditions reuse the previous predictions exactly. Two API workers share one rate limiter and an additional CNY 5 reserve.',
        '摘要与reader均为Qwen3-30B-A3B-Instruct-2507。只替换摘要模型，prompt和生成设置不变；其余四条件逐字段复用旧预测。两路API并发，共用请求限速和新增5元费用预留。')
    note += '<p><a href="../' + html.escape(source.name, quote=True) + '/Report.html">' + report.bi('Original DeepSeek-summary report', '原DeepSeek摘要版报告') + '</a></p></section>'
    target.write_text(target.read_text(encoding='utf-8').replace('</h1>', '</h1>' + note, 1), encoding='utf-8')
    expected = read_json(destination / 'manifest.json')['identity']['source_files']
    if snapshot(source) != expected:
        raise ValueError('Original run files changed.')
    spending_path = destination / 'spending_reservations.json'
    spending = read_json(spending_path) if spending_path.exists() else []
    write_json(destination / 'completion.json', {
        'passed': True, 'source_files_unchanged': True, 'preserved_conditions': preserved,
        'prediction_count': config['sample_size'] * len(config['datasets']) * 5,
        'matrix_cells': len(config['datasets']) * 25,
        'additional_accounted_cny': sum(x['accounted_cny'] for x in spending),
        'completed_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'compress', 'predict', 'evaluate', 'all'], default='all')
    parser.add_argument('--allow-api', action='store_true')
    args = parser.parse_args()
    source, destination = ROOT / 'runs' / SOURCE_NAME, ROOT / 'runs' / DESTINATION_NAME
    with pipeline.run_lock(destination):
        config = prepare(source, destination)
        tokenizer = ContextTokenizer()
        # This process-local wrapper changes scheduling only, never request payloads.
        with network_pacing(config['summary']['min_interval_seconds']):
            if args.stage in ('all', 'compress'):
                compress(config, source, destination, tokenizer, args.allow_api)
            if args.stage in ('all', 'predict'):
                predict(config, source, destination, tokenizer, args.allow_api)
        if args.stage in ('all', 'evaluate'):
            evaluate(config, source, destination)
    print('Output:', destination, flush=True)


if __name__ == '__main__':
    main()
