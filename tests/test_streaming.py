"""网络恢复只改变传输方式：完整结束、真实缓存身份、失败计账必须可验证。"""
import io
import json
import tempfile
import unittest
from pathlib import Path
from http.client import RemoteDisconnected, IncompleteRead
from unittest.mock import patch

from reader import CachedChatClient, OutputLimitError
from storage import fingerprint, write_json, read_json
from streaming import read_stream

ROOT = Path(__file__).resolve().parent.parent


def response_stream(finish='stop', done=True):
    chunks = [
        {'choices': [{'index': 0, 'delta': {'content': 'hello '}, 'finish_reason': None}]},
        {'choices': [{'index': 0, 'delta': {'content': '世界'}, 'finish_reason': None}]},
        {'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}]},
        {'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': 3, 'total_tokens': 13}},
    ]
    raw = ': heartbeat\n\n' + ''.join('data: '+json.dumps(c, ensure_ascii=False)+'\n\n' for c in chunks)
    if done:
        raw += 'data: [DONE]\n\n'
    return io.BytesIO(raw.encode('utf-8'))


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'.cache')
        self.addCleanup(self.temp.cleanup)
        self.config = {'base_url': 'https://example.invalid/v1', 'model': 'test-model',
                       'temperature': 0, 'enable_thinking': False, 'stream': True,
                       'api_key_env': 'TEST_ONLY_KEY', 'max_api_attempts': 3,
                       'timeout_seconds': 30, 'min_interval_seconds': 0}
        self.messages = [{'role': 'user', 'content': 'context only'}]

    def test_unicode_finish_and_usage_are_reassembled_and_verified(self):
        body = read_stream(response_stream())
        self.assertEqual(CachedChatClient._extract(body, 'key'), ('hello 世界', 'key'))
        self.assertEqual(body['usage']['total_tokens'], 13)
        body['choices'][0]['message']['content'] = 'forged'
        with self.assertRaises(ValueError):
            CachedChatClient._extract(body, 'key')

    def test_eof_even_after_stop_is_not_a_complete_response(self):
        with self.assertRaises(IncompleteRead):
            read_stream(response_stream(done=False))

    def test_length_response_keeps_existing_strict_failure(self):
        with self.assertRaises(OutputLimitError):
            CachedChatClient._extract(read_stream(response_stream(finish='length')), 'key')

    def test_complete_stream_caches_exact_request_and_replays_without_network(self):
        client = CachedChatClient(self.config, Path(self.temp.name))
        with patch.dict('os.environ', {'TEST_ONLY_KEY': 'test-only'}), patch('reader.urlopen', return_value=response_stream()) as call:
            text, key = client.complete(self.messages, 128)
        payload = json.loads(call.call_args.args[0].data)
        self.assertTrue(payload['stream'])
        self.assertEqual(payload['messages'], self.messages)
        self.assertEqual(payload['max_tokens'], 128)
        cached = read_json(client.cache_dir/f'{key}.json')
        self.assertEqual(cached['request']['payload'], payload)
        with patch('reader.urlopen', side_effect=AssertionError('No network')):
            self.assertEqual(CachedChatClient(self.config, Path(self.temp.name), offline=True).complete(self.messages, 128), (text, key))

    def test_legacy_cache_keeps_original_nonstream_request_identity(self):
        client = CachedChatClient(self.config, Path(self.temp.name), offline=True)
        payload = {'model':'test-model', 'messages':self.messages, 'temperature':0,
                   'max_tokens':128, 'enable_thinking':False, 'stream':False}
        identity = {'base_url':self.config['base_url'], 'payload':payload}
        key = fingerprint(identity)
        write_json(client.cache_dir/f'{key}.json', {'request':identity,
                   'response':{'choices':[{'finish_reason':'stop','message':{'content':'original'}}]}})
        self.assertEqual(client.complete(self.messages,128),('original',key))
        with self.assertRaises(RuntimeError):
            client.complete([{'role':'user','content':'different context'}],128)
        with self.assertRaises(RuntimeError):
            client.complete(self.messages,129)
        self.assertEqual(len(list(client.cache_dir.glob('*.json'))),1)

    def test_disconnect_is_counted_and_bounded_retry_recovers(self):
        client = CachedChatClient(self.config, Path(self.temp.name))
        with patch.dict('os.environ', {'TEST_ONLY_KEY':'test-only'}), patch('reader.time.sleep'), patch('reader.urlopen', side_effect=[RemoteDisconnected(),response_stream()]):
            self.assertEqual(client.complete(self.messages,128)[0],'hello 世界')
        self.assertEqual([e['status'] for e in client.ledger],['network_error','received'])

    def test_incomplete_streams_never_enter_valid_cache_or_reset_attempt_cap(self):
        client = CachedChatClient(self.config, Path(self.temp.name))
        with patch.dict('os.environ', {'TEST_ONLY_KEY':'test-only'}), patch('reader.time.sleep'), patch('reader.urlopen', side_effect=lambda *a,**k: response_stream(done=False)):
            with self.assertRaises(RuntimeError):
                client.complete(self.messages,128)
        self.assertEqual(len(client.ledger),3)
        self.assertEqual(list(client.cache_dir.glob('*.json')),[])
        resumed=CachedChatClient(self.config,Path(self.temp.name))
        with patch.dict('os.environ', {'TEST_ONLY_KEY':'test-only'}), patch('reader.urlopen', side_effect=AssertionError('Cap reached')):
            with self.assertRaisesRegex(RuntimeError,'limit'):
                resumed.complete(self.messages,128)
