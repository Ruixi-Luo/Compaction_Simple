"""Offline concurrency checks: shared cost, exact cache, retries and bounded work."""
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch, MagicMock

project=Path(os.environ.get('COMPACTION_SIMPLE_PROJECT',Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(project))
local_tools=Path(__file__).resolve().parent
sys.path.insert(0,str(local_tools if (local_tools/'concurrent_api.py').exists() else project/'tools'))
import concurrent_api as concurrent
import reader
from storage import read_json,write_json

class Counter:
    def count(self,text):return len(text.split())

class ParallelTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.run=Path(self.temp.name)
        self.config=read_json(project/'config.json')
        self.config['cost_reserve_cny']=5
        self.config['reader'].update(api_key_env='TEST_ONLY_KEY',min_interval_seconds=0)
        self.env=patch.dict(os.environ,{'TEST_ONLY_KEY':'offline-fake-key'})
        self.env.start();self.addCleanup(self.env.stop)
        self.keys=patch.object(reader,'load_keys');self.keys.start();self.addCleanup(self.keys.stop)

    def test_two_real_client_paths_overlap_and_replay_without_paid_requests(self):
        lock=threading.Lock()
        active=0; maximum=0; sent=[]
        def transport(request,timeout):
            nonlocal active,maximum
            payload=json.loads(request.data)
            with lock:
                sent.append((time.perf_counter(),payload));active+=1;maximum=max(maximum,active)
            time.sleep(.09)
            with lock:active-=1
            response=MagicMock()
            body={'choices':[{'finish_reason':'stop','message':{'content':'{"p_yes":0.7}'}}],
                  'usage':{'prompt_tokens':100,'completion_tokens':9}}
            response.__enter__.return_value=io.BytesIO(json.dumps(body).encode())
            return response
        client=concurrent.TwoWorkerClient(self.config,'reader',self.run,Counter(),True)
        messages=[[{'role':'user','content':f'input {i}'}] for i in range(6)]
        with patch.object(reader,'urlopen',side_effect=transport), concurrent.network_pacing(.02):
            values=list(concurrent.bounded_results(lambda m:client.complete(m,128,True),messages))
        self.assertEqual(len(values),6);self.assertEqual(maximum,2)
        self.assertTrue(all(b[0]-a[0]>=.018 for a,b in zip(sent,sent[1:])))
        for _,payload in sent:
            self.assertEqual(set(payload),{'model','messages','temperature','max_tokens','enable_thinking','stream','response_format'})
            self.assertEqual(payload['temperature'],0)
            self.assertEqual(payload['model'],self.config['reader']['model'])
        ledgers=[read_json(self.run/f'api/reader/worker_{i}/api_attempts.json') for i in (1,2)]
        self.assertEqual(sum(map(len,ledgers)),6)
        self.assertTrue(all(e['status']=='received' for ledger in ledgers for e in ledger))
        spending=read_json(self.run/'spending_reservations.json')
        self.assertEqual(len(spending),6)
        self.assertAlmostEqual(sum(e['accounted_cny'] for e in spending),6*.000127)
        restarted=concurrent.TwoWorkerClient(self.config,'reader',self.run,Counter(),False)
        with patch.object(reader,'urlopen',side_effect=AssertionError('offline replay must not send')):
            self.assertEqual(len(list(concurrent.bounded_results(lambda m:restarted.complete(m,128,True),messages))),6)
        self.assertEqual(read_json(self.run/'spending_reservations.json'),spending)

    def test_duplicate_inputs_make_one_request(self):
        client=concurrent.TwoWorkerClient(self.config,'reader',self.run,Counter(),True)
        response=MagicMock()
        response.__enter__.return_value=io.BytesIO(json.dumps({'choices':[{'finish_reason':'stop','message':{'content':'{"p_yes":0.7}'}}],
                          'usage':{'prompt_tokens':1,'completion_tokens':9}}).encode())
        with patch.object(reader,'urlopen',return_value=response) as transport:
            values=list(concurrent.bounded_results(lambda _:client.complete([{'role':'user','content':'same'}],128,True),[1,2]))
        self.assertEqual(len(values),2);transport.assert_called_once()

    def test_shared_budget_cannot_be_reserved_twice(self):
        client=concurrent.TwoWorkerClient(self.config,'reader',self.run,Counter(),True)
        client.guard.limit=.002 # one reservation fits; two simultaneous reservations do not
        first=threading.Event();release=threading.Event();calls=[]
        def transport(request,timeout):
            calls.append(1);first.set();release.wait(2)
            response=MagicMock()
            response.__enter__.return_value=io.BytesIO(json.dumps({'choices':[{'finish_reason':'stop','message':{'content':'{"p_yes":0.7}'}}],
                            'usage':{'prompt_tokens':1,'completion_tokens':9}}).encode())
            return response
        with patch.object(reader,'urlopen',side_effect=transport),ThreadPoolExecutor(max_workers=2) as pool:
            one=pool.submit(client.complete,[{'role':'user','content':'first'}],128,True)
            self.assertTrue(first.wait(2))
            two=pool.submit(client.complete,[{'role':'user','content':'second'}],128,True)
            try:
                with self.assertRaises(RuntimeError):two.result(timeout=2)
            finally:release.set()
            one.result(timeout=2)
        self.assertEqual(len(calls),1)

    def test_stops_submitting_on_error_but_preserves_finished_peer(self):
        started=[]
        def task(i):
            started.append(i)
            if i==0:raise ValueError('stop')
            time.sleep(.03);return 'complete peer'
        results=[]
        with self.assertRaises(ValueError):
            for item,result in concurrent.bounded_results(task,range(8)):results.append((item,result))
        self.assertEqual(sorted(started),[0,1])
        self.assertEqual(results,[(1,'complete peer')])

    def test_offline_missing_never_sends(self):
        client=concurrent.TwoWorkerClient(self.config,'reader',self.run,Counter(),False)
        with patch.object(reader,'urlopen') as send:
            with self.assertRaises(RuntimeError):client.complete([],128,True)
            send.assert_not_called()

if __name__=='__main__':unittest.main()
