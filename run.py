"""主线只做四步。缓存和基本校验留在函数内部，没有独立 audit 阶段。"""
import gc
import sys
import time
from storage import ROOT, DATA_ROOT, read_json, write_json, fingerprint, file_hash
from data import ContextTokenizer
from compressors import create_compressor
from data import load_examples
from reader import make_client, predict_one
from metrics import analyze
from compressors import LOCAL_METHODS, compress_one
from report import build_report
from tools.concurrent_api import TwoWorkerClient, bounded_results, network_pacing
from telemetry import Progress

METHODS = ['full', 'no_context', 'selective_context', 'llmlingua2', 'llm_summary']


def prepare(config, run, offline=False):
    """config 设置；run 独立输出目录；offline 禁止下载。固定样本和运行身份。"""
    if offline and not (DATA_ROOT/'tokenizer/qwen3-30b-a3b-instruct-2507.json').exists():
        raise RuntimeError('Offline tokenizer missing.')
    tokenizer = ContextTokenizer()
    files = [*sorted(ROOT.glob('*.py')), *sorted((ROOT/'compressors').glob('*.py')), ROOT/'models.lock.json', ROOT/'resources.json', ROOT/'tools/concurrent_api.py', ROOT/'tools/monitor_progress.py']
    identity = {'protocol': 'single_call_0.5_v1', 'config': config,
                'code': {str(p.relative_to(ROOT)): file_hash(p) for p in files}}
    manifest_path = run/'manifest.json'
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest['identity'] != identity:
            raise ValueError('配置或代码已变化，请使用新的 run_name，保留旧结果。')
        for name, digest in manifest['samples'].items():
            if fingerprint(read_json(run/name/'examples.json')) != digest:
                raise ValueError('Saved samples changed.')
        return tokenizer
    write_json(run/'config.json', config)
    samples = {}
    for name in config['datasets']:
        examples, source = load_examples(name, config['sample_size'], config['sample_seed'], offline)
        write_json(run/name/'examples.json', examples)
        write_json(run/name/'dataset_source.json', source)
        samples[name] = fingerprint(examples)
    write_json(manifest_path, {'identity': identity, 'samples': samples})
    return tokenizer


def cached_compression(folder, context, method, ratio, result=None):
    """只保存/读取本轮的单次压缩；旧长度搜索结果不会自动混入。"""
    request = {'context': context, 'method': method, 'retained_ratio': ratio}
    key = fingerprint(request)
    path = folder/'compressions'/f'{key}.json'
    if result is not None:
        write_json(path, {'request': request, 'result': result, 'sha256': fingerprint(result)})
        return result
    if not path.exists():
        return None
    saved = read_json(path)
    if saved['request'] != request or saved['sha256'] != fingerprint(saved['result']):
        raise ValueError('Compression cache changed.')
    return saved['result']


def api_client(config, role, run, tokenizer, allow_api):
    """Use independent worker ledgers and one shared spending guard within each stage."""
    if config.get('api_workers', 1) == 1:
        return make_client(config, role, run, tokenizer, allow_api)
    return TwoWorkerClient(config, role, run, tokenizer, allow_api)


def compress(config, run, tokenizer, offline=False, allow_api=False, progress=None):
    """Local models run sequentially; summary requests overlap without seeing the query."""
    ratio = config['retained_ratio']
    for method in config['methods']:
        if method in ('full', 'no_context'):
            continue
        engine = client = None
        for name in config['datasets']:
            folder = run/name
            contexts = list(dict.fromkeys(e['context'] for e in read_json(folder/'examples.json')))
            pending = [c for c in contexts if cached_compression(folder, c, method, ratio) is None]
            if pending and offline:
                raise RuntimeError('Missing compression cache; offline mode never runs a model.')
            if pending and method in LOCAL_METHODS and engine is None:
                if sys.flags.hash_randomization:
                    raise RuntimeError('Set PYTHONHASHSEED=0 before starting Python.')
                from compressors.runtime import verify_assets
                verify_assets([method])
                engine = create_compressor(method, config['device'])
            if pending and method == 'llm_summary' and client is None:
                client = api_client(config, 'summary', run, tokenizer, allow_api)

            def task(context):
                start = time.perf_counter()
                result = compress_one(context, method, tokenizer, engine, client, ratio)
                result['compression_seconds'] = time.perf_counter() - start
                return result

            workers = config.get('api_workers', 1) if method == 'llm_summary' else 1
            completed = len(contexts) - len(pending)
            interval = config['summary']['min_interval_seconds'] if method == 'llm_summary' else 0
            with network_pacing(interval):
                for context, result in bounded_results(task, pending, workers=workers):
                    cached_compression(folder, context, method, ratio, result)
                    completed += 1
                    detail = f"tokens={result['tokens']}, target={result['budget_tokens']}"
                    if progress:
                        progress.update('compress/'+method, name, completed, len(contexts), detail)
                    else:
                        print(f'compress {name} {method} {completed}/{len(contexts)} {detail}', flush=True)
            if progress:
                progress.update('compress/'+method, name, completed, len(contexts), 'stage complete')
        del engine, client
        gc.collect()
        if 'torch' in sys.modules and config['device'] == 'cuda':
            sys.modules['torch'].cuda.empty_cache()


def predict(config, run, tokenizer, allow_api=False, progress=None):
    """Each API request is one question/condition. Only the main thread writes rows."""
    client = api_client(config, 'reader', run, tokenizer, allow_api)
    workers = config.get('api_workers', 1)
    for name in config['datasets']:
        folder = run/name
        examples = read_json(folder/'examples.json')

        lengths = [tokenizer.count(e['context']) for e in examples]
        def task(item):
            index, method = item
            example, original_tokens = examples[index], lengths[index]
            context = example['context']
            if method in ('full', 'no_context'):
                text = context if method == 'full' else ''
                condition = {'text': text, 'tokens': tokenizer.count(text), 'budget_met': None,
                             'length_repair_used': False}
            else:
                condition = cached_compression(folder, context, method, config['retained_ratio'])
                if condition is None:
                    raise RuntimeError('Complete the compress stage first.')
                condition = {**condition, 'local_compression_hash': fingerprint([context, method])}
            forecast = predict_one(condition['text'], example['question'], client, config['reader']['max_tokens'])
            return {**condition, **forecast, 'retained_ratio_actual': condition['tokens']/original_tokens}

        conditions = {i: {} for i in range(len(examples))}
        tasks = [(i, m) for i in range(len(examples)) for m in config['methods']]
        with network_pacing(config['reader']['min_interval_seconds']):
            for done, ((index, method), condition) in enumerate(bounded_results(task, tasks, workers=workers), 1):
                conditions[index][method] = condition
                complete = [i for i in range(len(examples)) if len(conditions[i]) == len(config['methods'])]
                rows = [{**examples[i], 'original_tokens': lengths[i],
                         'conditions': {m: conditions[i][m] for m in config['methods']}} for i in complete]
                # Partial questions remain safe in API caches; only complete rows are saved.
                if len(conditions[index]) == len(config['methods']):
                    write_json(folder/'predictions.json', rows)
                if progress:
                    progress.update('predict_conditions', name, done, len(tasks), f'questions={len(complete)}/{len(examples)}')
                else:
                    print(f'predict {name} {done}/{len(tasks)} conditions', flush=True)


def evaluate(config, run):
    """只读概率计算分数和双向 InfoGap，再生成 HTML；不调用模型。"""
    for name in config['datasets']:
        rows = read_json(run/name/'predictions.json')
        examples = read_json(run/name/'examples.json')
        if len(rows) != len(examples) or any(any(r[k] != v for k,v in e.items()) for r,e in zip(rows, examples)):
            raise ValueError('缺题或预测与原始题目不匹配。')
        write_json(run/name/'metrics.json', analyze(rows, config))
    build_report(config, run)

import argparse,json,os,re
from contextlib import contextmanager
from pathlib import Path

def load_config(path):
    """只支持本轮五条件、50%和已验证模型；设置变化请换 run_name。"""
    c=read_json(path)
    if not re.fullmatch(r'[A-Za-z0-9_-]+',c['run_name']):raise ValueError('Invalid run_name')
    if c['methods']!=METHODS or c['retained_ratio']!=.5:raise ValueError('Use the five-condition 0.5 protocol.')
    if not c['datasets'] or len(set(c['datasets']))!=len(c['datasets']) or set(c['datasets'])-{'hotpotqa','qasper'}:raise ValueError('Invalid datasets')
    if type(c['sample_size']) is not int or c['sample_size']<1 or c['bootstrap_repetitions']<2:raise ValueError('Invalid sample/bootstrap size')
    if c['device'] not in ('cuda','cpu') or c['cost_reserve_cny']<=0:raise ValueError('Invalid device/budget')
    if type(c.get('api_workers', 1)) is not int or not 1 <= c.get('api_workers', 1) <= 4:raise ValueError('api_workers must be 1..4')
    for role in ('reader', 'summary'):
        models = {'Qwen/Qwen3-30B-A3B-Instruct-2507'} if role == 'reader' else {'deepseek-ai/DeepSeek-V3.2', 'Qwen/Qwen3-30B-A3B-Instruct-2507'}
        a=c[role]
        if a['model'] not in models or a['base_url']!='https://api.siliconflow.cn/v1' or a['temperature']!=0 or a['enable_thinking'] is not False:raise ValueError('Model/protocol change needs separate validation.')
        if a['max_api_attempts']<c.get('api_workers',1) or a['timeout_seconds']<=0 or a['min_interval_seconds']<0:raise ValueError('Invalid API limits')
    if c['reader']['stream'] or c['reader']['max_tokens']!=128:raise ValueError('Keep reader protocol unchanged.')
    return c

def run_directory(config):
    return ROOT/'runs'/config['run_name']

@contextmanager
def run_lock(run):
    """run：本轮目录。独占锁防止两个终端同时写缓存/记账；正常退出自动删除。"""
    run.mkdir(parents=True, exist_ok=True)
    path = run/'.run.lock'
    try:
        handle = path.open('x', encoding='utf-8')
    except FileExistsError:
        raise RuntimeError(f'Run is already locked: {path}. Check the recorded PID before removing a stale lock.') from None
    with handle:
        json.dump({'pid': os.getpid()}, handle)
    try:
        yield
    finally:
        path.unlink()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--stage', choices=['prepare', 'compress', 'predict', 'evaluate', 'all'], default='all')
    parser.add_argument('--allow-api', action='store_true', help='允许缓存缺失时付费请求；默认禁止。')
    parser.add_argument('--offline', action='store_true', help='只读缓存，不下载、不运行 GPU、不调用 API。')
    args = parser.parse_args()
    if args.offline and args.allow_api:
        parser.error('--offline 与 --allow-api 不能同时使用。')
    config = load_config(args.config)
    if config['methods'] != METHODS or config['retained_ratio'] != .5:
        parser.error('本精简协议固定五条件和 retained_ratio=0.5；请使用 config.json。')
    run = run_directory(config)
    with run_lock(run):
        progress = Progress(run, args.stage)
        try:
            with progress.stage('prepare'):
                tokenizer = prepare(config, run, args.offline)
            if args.stage in ('compress', 'all'):
                with progress.stage('compress'):
                    compress(config, run, tokenizer, args.offline, args.allow_api, progress)
            if args.stage in ('predict', 'all'):
                with progress.stage('predict'):
                    predict(config, run, tokenizer, args.allow_api, progress)
            if args.stage in ('evaluate', 'all'):
                with progress.stage('evaluate'):
                    evaluate(config, run)
        except BaseException:
            progress.finish('failed_or_interrupted')
            raise
        else:
            progress.finish('complete')
            if args.stage in ('evaluate', 'all'):
                build_report(config, run)  # Include final execution time, not a live estimate.
    print('Output:', run)

if __name__=='__main__':
    main()
