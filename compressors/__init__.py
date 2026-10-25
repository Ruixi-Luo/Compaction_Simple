"""三个压缩方法：只接收完整 context，单次调用。"""
import math
from reader import generate_summary
def validate_input(context, retained_ratio):
    if not isinstance(context, str):
        raise TypeError("context must be a string, not an example containing a question/label.")
    if isinstance(retained_ratio, bool) or not 0 < retained_ratio <= 1:
        raise ValueError("retained_ratio is the fraction to KEEP, in (0, 1].")

LOCAL_METHODS=('selective_context','llmlingua2')
def create_compressor(method,device='cuda'):
    if method=='selective_context':
        from .selective_context import SelectiveContextCompressor
        return SelectiveContextCompressor(device)
    if method=='llmlingua2':
        from .llmlingua2 import LLMLingua2Compressor
        return LLMLingua2Compressor(device)
    raise ValueError(method)

def compress_one(context, method, tokenizer, engine=None, client=None, retained_ratio=.5):
    """context 原文；method 方法；tokenizer 统一计数；engine 本地模型；client 摘要 API。"""
    if not isinstance(context, str) or not context.strip():
        raise ValueError('压缩器只接收非空正文字符串，不接收 question/label。')
    budget = math.floor(tokenizer.count(context) * retained_ratio)
    if budget < 1:
        raise ValueError('文本太短，无法产生正的长度目标。')
    if method == 'llm_summary':
        result = generate_summary(context, budget, client, tokenizer)
    elif method in LOCAL_METHODS:
        text = engine.compress(context, retained_ratio=retained_ratio)  # 仅此一次
        result = {'text': text, 'tokens': tokenizer.count(text), 'native_rate': retained_ratio}
    else:
        raise ValueError('Unsupported compressor: '+method)
    return {**result, 'budget_tokens': budget, 'budget_met': result['tokens'] <= budget,
            'length_repair_used': False, 'post_clip_used': False, 'matching_trim_used': False,
            'policy': 'single_call_native_ratio', 'compression_seconds': 0.0}
