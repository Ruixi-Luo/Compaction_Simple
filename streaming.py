"""严格组装 SSE 响应；只有收到 DONE 才能交给现有完整性检查。"""
import json
from http.client import IncompleteRead


def assemble_chunks(chunks):
    """按服务端顺序拼接文本，保留最终 finish_reason 和真实 usage。"""
    content, reasoning, usage, finish = [], [], {}, None
    for chunk in chunks:
        if chunk.get('error'):
            raise ValueError('Streaming API returned an error event.')
        if chunk.get('usage'):
            usage = chunk['usage']
        for choice in chunk.get('choices', []):
            if choice.get('index', 0) != 0:
                raise ValueError('Expected one streaming completion.')
            delta = choice.get('delta', {})
            if delta.get('tool_calls'):
                raise ValueError('Unexpected tool call in a text compressor.')
            text = delta.get('content')
            if text:
                if finish is not None or not isinstance(text, str):
                    raise ValueError('Invalid content after a finish event.')
                content.append(text)
            if delta.get('reasoning_content'):
                reasoning.append(delta['reasoning_content'])
            if choice.get('finish_reason') is not None:
                finish = choice['finish_reason']
    return {'choices': [{'message': {'content': ''.join(content),
                                    'reasoning_content': ''.join(reasoning)},
                         'finish_reason': finish}], 'usage': usage}


def read_stream(response):
    """读取服务端 SSE 行；EOF/断连不能当作成功结束，原始事件随缓存保留。"""
    chunks = []
    for raw in response:
        line = raw.decode('utf-8').strip()
        if not line or line.startswith(':'):
            continue
        if not line.startswith('data:'):
            raise ValueError('Unexpected streaming response framing.')
        data = line[5:].strip()
        if data == '[DONE]':
            body = assemble_chunks(chunks)
            body['_stream_chunks'] = chunks
            body['_stream_done'] = True
            return body
        chunks.append(json.loads(data))
    raise IncompleteRead(b'', None)


def verify_stream_cache(body):
    """离线重放时从原始事件再组装，防止缓存里的汇总文本与原事件不符。"""
    if body.get('_stream_done') is not True:
        raise ValueError('Cached stream has no DONE marker.')
    expected = assemble_chunks(body['_stream_chunks'])
    if any(body.get(key) != value for key, value in expected.items()):
        raise ValueError('Cached streaming content differs from raw events.')
