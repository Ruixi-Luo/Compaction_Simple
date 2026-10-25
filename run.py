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

METHODS = ['full', 'no_context', 'selective_context', 'llmlingua2', 'llm_summary']


def prepare(config, run, offline=False):
    """config 设置；run 独立输出目录；offline 禁止下载。固定样本和运行身份。"""
    if offline and not (DATA_ROOT/'tokenizer/qwen3-30b-a3b-instruct-2507.json').exists():
        raise RuntimeError('Offline tokenizer missing.')
    tokenizer = ContextTokenizer()
    files = [*sorted(ROOT.glob('*.py')), *sorted((ROOT/'compressors').glob('*.py')), ROOT/'models.lock.json', ROOT/'resources.json']
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


def compress(config, run, tokenizer, offline=False, allow_api=False):
    """按方法遍历唯一正文；本地模型依次加载，摘要沿用原 prompt。"""
    ratio = config['retained_ratio']
    for method in config['methods']:
        if method in ('full', 'no_context'):
            continue
        engine = client = None
        for name in config['datasets']:
            folder = run/name
            contexts = dict.fromkeys(e['context'] for e in read_json(folder/'examples.json'))
            for i, context in enumerate(contexts, 1):
                if cached_compression(folder, context, method, ratio) is not None:
                    continue
                if offline:
                    raise RuntimeError('缺少压缩缓存；offline 模式不运行模型。')
                if method in LOCAL_METHODS and engine is None:
                    if sys.flags.hash_randomization:
                        raise RuntimeError('请先在 CMD 中执行 set PYTHONHASHSEED=0')
                    from compressors.runtime import verify_assets
                    verify_assets([method])
                    engine = create_compressor(method, config['device'])
                if method == 'llm_summary' and client is None:
                    client = make_client(config, 'summary', run, tokenizer, allow_api)
                start = time.perf_counter()
                result = compress_one(context, method, tokenizer, engine, client, ratio)
                result['compression_seconds'] = time.perf_counter() - start
                cached_compression(folder, context, method, ratio, result)
                print(f'compress {name} {method} {i}/{len(contexts)}: {result["tokens"]}/{result["budget_tokens"]}', flush=True)
        del engine, client
        gc.collect()
        if 'torch' in sys.modules and config['device'] == 'cuda':
            sys.modules['torch'].cuda.empty_cache()


def predict(config, run, tokenizer, allow_api=False):
    """每题五条件，同一个 reader。每次请求有缓存，每完成一题保存一行。"""
    client = make_client(config, 'reader', run, tokenizer, allow_api)
    for name in config['datasets']:
        folder = run/name
        examples = read_json(folder/'examples.json')
        rows = []
        for i, example in enumerate(examples, 1):
            context = example['context']
            original_tokens = tokenizer.count(context)
            conditions = {}
            for method in config['methods']:
                if method in ('full', 'no_context'):
                    text = context if method == 'full' else ''
                    condition = {'text': text, 'tokens': tokenizer.count(text), 'budget_met': None,
                                 'length_repair_used': False}
                else:
                    condition = cached_compression(folder, context, method, config['retained_ratio'])
                    if condition is None:
                        raise RuntimeError('请先完成 compress 阶段。')
                    condition = {**condition, 'local_compression_hash': fingerprint([context, method])}
                forecast = predict_one(condition['text'], example['question'], client, config['reader']['max_tokens'])
                conditions[method] = {**condition, **forecast,
                                      'retained_ratio_actual': condition['tokens']/original_tokens}
            rows.append({**example, 'original_tokens': original_tokens, 'conditions': conditions})
            write_json(folder/'predictions.json', rows)
            print(f'predict {name} {i}/{len(examples)}', flush=True)


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
    for role,model in [('reader','Qwen/Qwen3-30B-A3B-Instruct-2507'),('summary','deepseek-ai/DeepSeek-V3.2')]:
        a=c[role]
        if a['model']!=model or a['base_url']!='https://api.siliconflow.cn/v1' or a['temperature']!=0 or a['enable_thinking'] is not False:raise ValueError('Model/protocol change needs separate validation.')
        if a['max_api_attempts']<1 or a['timeout_seconds']<=0 or a['min_interval_seconds']<0:raise ValueError('Invalid API limits')
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
        tokenizer = prepare(config, run, args.offline)
        if args.stage in ('compress', 'all'):
            compress(config, run, tokenizer, args.offline, args.allow_api)
        if args.stage in ('predict', 'all'):
            predict(config, run, tokenizer, args.allow_api)
        if args.stage in ('evaluate', 'all'):
            evaluate(config, run)
    print('Output:', run)

if __name__=='__main__':
    main()
