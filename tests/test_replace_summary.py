"""Offline checks for model-only replacement; no API or GPU use."""
import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

project = Path(os.environ.get('COMPACTION_SIMPLE_PROJECT', Path(__file__).resolve().parents[1]))
local_tools = Path(__file__).resolve().parent
sys.path.insert(0, str(local_tools if (local_tools/'replace_summary.py').exists() else project/'tools'))
import replace_summary as extension
from storage import ROOT, fingerprint, read_json, write_json
from reader import CostGuard, summary_messages, output_ceiling


class Counter:
    def count(self, text):
        return len(text.split())


class ReplacementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / 'old'
        self.destination = self.base / 'new'
        self.config = read_json(ROOT / 'config.json')
        self.config['summary']['model'] = 'deepseek-ai/DeepSeek-V3.2'  # Historical replacement fixture
        self.config.update(datasets=['hotpotqa'], sample_size=2, bootstrap_repetitions=4)
        self.examples = [dict(id=str(i), context='one two three four five six',
                              question=f'PRIVATE_QUERY_{i}', label=i, context_group='shared')
                         for i in range(2)]
        self.rows = []
        for example in self.examples:
            conditions = {m: dict(text='old', tokens=1, p_yes=.3, predicted_label=0,
                                  budget_met=True, length_repair_used=False,
                                  prediction_request_hash='old_reader', retained_ratio_actual=1/6)
                          for m in self.config['methods']}
            self.rows.append({**example, 'original_tokens':6, 'conditions':conditions})
        write_json(self.source / 'hotpotqa/examples.json', self.examples)
        write_json(self.source / 'hotpotqa/dataset_source.json', {'source':'synthetic offline test'})
        write_json(self.source / 'hotpotqa/predictions.json', self.rows)
        write_json(self.source / 'manifest.json', {
            'identity': {'config':self.config, 'code':{}},
            'samples': {'hotpotqa':fingerprint(self.examples)}})
        extension.pipeline.evaluate(self.config, self.source)
        old_request = {'base_url':self.config['summary']['base_url'], 'payload':{
            'model':self.config['summary']['model'], 'messages':summary_messages(self.examples[0]['context'], 3),
            'temperature':0.0, 'max_tokens':output_ceiling(3), 'enable_thinking':False, 'stream':True}}
        key = fingerprint(old_request)
        write_json(self.source / 'api/summary/api_cache' / (key+'.json'), {'request':old_request})
        extension.pipeline.cached_compression(self.source / 'hotpotqa', self.examples[0]['context'],
                                              'llm_summary', .5, {'text':'old', 'request_hash':key})

    def test_full_replacement_keeps_four_conditions_and_old_files(self):
        before = extension.snapshot(self.source)
        c = extension.prepare(self.source, self.destination)
        expected = copy.deepcopy(self.config)
        expected.update(run_name='new', cost_reserve_cny=5)
        expected['summary']['model'] = expected['reader']['model']
        self.assertEqual(c, expected)
        summary = Mock()
        summary.complete.return_value = ('first complete output above target is kept', 'new_summary_hash')
        with patch.object(extension, 'TwoWorkerClient', return_value=summary):
            extension.compress(c, self.source, self.destination, Counter(), False)
            extension.compress(c, self.source, self.destination, Counter(), False)
        summary.complete.assert_called_once_with(summary_messages(self.examples[0]['context'], 3), output_ceiling(3))
        self.assertNotIn('PRIVATE_QUERY', str(summary.complete.call_args))
        reader = Mock()
        reader.complete.return_value = ('{"p_yes":0.7}', 'new_reader_hash')
        with patch.object(extension, 'ReusingReader', return_value=reader):
            extension.predict(c, self.source, self.destination, Counter(), False)
        self.assertEqual(reader.complete.call_count, 2)
        extension.evaluate(c, self.source, self.destination)
        rows = read_json(self.destination / 'hotpotqa/predictions.json')
        self.assertFalse(rows[0]['conditions']['llm_summary']['budget_met'])
        for old, new in zip(self.rows, rows):
            for method in c['methods'][:-1]:
                self.assertEqual(old['conditions'][method], new['conditions'][method])
        self.assertEqual(extension.snapshot(self.source), before)
        metrics = read_json(self.destination / 'hotpotqa/metrics.json')
        self.assertEqual(sum(map(len, metrics['infogap'].values())),25)
        page = (self.destination / 'Report.html').read_text(encoding='utf-8')
        self.assertIn('Qwen summary', page)
        self.assertIn('data-language="en"', page)
        self.assertEqual(read_json(self.destination / 'completion.json')['preserved_conditions'],8)

    def test_locked_source_blocks_start(self):
        write_json(self.source / '.run.lock', {'pid':123})
        with self.assertRaises(RuntimeError):
            extension.prepare(self.source, self.destination)

    def test_incomplete_source_blocks_start(self):
        write_json(self.source / 'hotpotqa/predictions.json', self.rows[:1])
        with self.assertRaises(ValueError):
            extension.prepare(self.source, self.destination)

    def test_prompt_change_blocks_new_call(self):
        c = extension.prepare(self.source, self.destination)
        with patch.object(extension, 'summary_messages', return_value=[{'role':'user','content':'CHANGED'}]):
            with self.assertRaises(ValueError):
                extension.verify_original_prompt(self.source, 'hotpotqa', self.examples[0]['context'],3,c)

    def test_changed_reused_prediction_blocks_report(self):
        c = extension.prepare(self.source, self.destination)
        changed = copy.deepcopy(self.rows)
        changed[0]['conditions']['full']['p_yes'] = .9
        write_json(self.destination / 'hotpotqa/predictions.json',changed)
        with self.assertRaises(ValueError):
            extension.evaluate(c,self.source,self.destination)

    def test_old_exact_cache_hit_does_not_call_new_client(self):
        old, new = Mock(), Mock()
        old.complete.return_value=('cached', 'original_hash')
        with patch.object(extension,'CachedChatClient',return_value=old), patch.object(extension,'TwoWorkerClient',return_value=new):
            reader = extension.ReusingReader(self.config,self.source,self.destination,Counter(),True)
            self.assertEqual(reader.complete([],128,True),('cached','original_hash'))
            new.complete.assert_not_called()
            old.complete.side_effect=RuntimeError('Offline replay is missing a cached response.')
            new.complete.return_value=('new','new_hash')
            self.assertEqual(reader.complete([],128,True),('new','new_hash'))
            new.complete.assert_called_once()

    def test_budget_is_cumulative_across_resume(self):
        path=self.destination/'spending_reservations.json'
        write_json(path,[{'accounted_cny':4.9999}])
        guard=CostGuard(path,Counter(),5)
        with self.assertRaises(RuntimeError):
            guard.reserve(self.config['reader']['model'],[{'content':'source'}],512)
        self.assertEqual(read_json(path),[{'accounted_cny':4.9999}])


if __name__ == '__main__':
    unittest.main()
