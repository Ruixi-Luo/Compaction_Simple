"""Exercise real Windows sharing behavior and response-before-ledger persistence."""
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT=Path(os.environ.get('COMPACTION_SIMPLE_PROJECT',Path(__file__).resolve().parents[1]))
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import storage
import reader
from monitor_progress import open_shared_read

class WindowsIOTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'ledger.json'
        storage.write_json(self.path,{'old':1})

    def test_monitor_snapshot_and_atomic_write_can_overlap(self):
        opened = threading.Event()
        def monitor():
            with open_shared_read(self.path) as handle:
                raw = handle.read()
                opened.set()
                time.sleep(.05)
            self.assertEqual(json.loads(raw),{'old':1})
        thread = threading.Thread(target=monitor)
        thread.start()
        self.assertTrue(opened.wait(2))
        storage.write_json(self.path,{'new':2})
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(storage.read_json(self.path),{'new':2})

    def test_locked_file_retries_disk_only(self):
        handle=open(self.path,'rb')
        try:
            with patch.object(storage.time,'sleep',side_effect=lambda _:handle.close()) as sleep:
                storage.write_json(self.path,{'new':2})
                if os.name=='nt':self.assertGreaterEqual(sleep.call_count,1)
        finally:
            handle.close()
        self.assertEqual(storage.read_json(self.path),{'new':2})

    def test_persistent_failure_keeps_old_and_complete_temp(self):
        with patch.object(storage.os,'replace',side_effect=PermissionError('locked')),patch.object(storage.time,'sleep'):
            with self.assertRaises(PermissionError):storage.write_json(self.path,{'new':2})
        self.assertEqual(storage.read_json(self.path),{'old':1})
        self.assertEqual(storage.read_json(self.path.with_suffix('.json.tmp')),{'new':2})

    def test_received_answer_survives_ledger_write_failure(self):
        config=storage.read_json(ROOT/'config.json')['reader']
        config={**config,'api_key_env':'OFFLINE_TEST_KEY','min_interval_seconds':0}
        client=reader.CachedChatClient(config,Path(self.temp.name)/'api')
        body={'choices':[{'finish_reason':'stop','message':{'content':'{"p_yes":0.7}'}}],
              'usage':{'prompt_tokens':3,'completion_tokens':9}}
        response=MagicMock()
        response.__enter__.return_value=io.BytesIO(json.dumps(body).encode())
        real_write=storage.write_json
        def locked_ledger(path,value):
            if Path(path).name=='api_attempts.json' and value[-1]['status']=='received':
                raise PermissionError('simulated occupied ledger')
            real_write(path,value)
        with patch.dict(os.environ,{'OFFLINE_TEST_KEY':'test-only'}),patch.object(reader,'urlopen',return_value=response) as send,patch.object(reader,'write_json',side_effect=locked_ledger):
            with self.assertRaises(PermissionError):client.complete([{'role':'user','content':'offline test'}],128,True)
            self.assertEqual(send.call_count,1)
        client.offline=True
        with patch.object(reader,'urlopen',side_effect=AssertionError('must reuse cache')):
            text,_=client.complete([{'role':'user','content':'offline test'}],128,True)
        self.assertEqual(text,'{"p_yes":0.7}')

if __name__=='__main__':unittest.main()
