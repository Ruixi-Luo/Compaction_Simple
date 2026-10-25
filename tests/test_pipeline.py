"""单次协议的核心测试；使用假模型，不发 API、不加载 GPU。"""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from compressors import compress_one
import run as pipeline
from run import load_config
from storage import ROOT, read_json, write_json


class Counter:
    def count(self, text):
        return len(text.split())


class SimpleTests(unittest.TestCase):
    def test_native_called_exactly_once_even_over_target(self):
        for method in ('selective_context', 'llmlingua2'):
            engine = Mock()
            engine.compress.return_value = 'one two three four five'
            result = compress_one('a b c d e f', method, Counter(), engine=engine)
            engine.compress.assert_called_once_with('a b c d e f', retained_ratio=.5)
            self.assertEqual(result['tokens'], 5)
            self.assertFalse(result['budget_met'])
            self.assertFalse(result['post_clip_used'])

    def test_summary_single_call_and_query_blind(self):
        api = Mock()
        api.complete.return_value = ('one two three four five', 'real_hash')
        result = compress_one('a b c d e f', 'llm_summary', Counter(), client=api)
        self.assertEqual(api.complete.call_count, 1)
        self.assertEqual(result['text'], 'one two three four five')
        with self.assertRaises(ValueError):
            compress_one({'context': 'x', 'question': 'leak'}, 'llm_summary', Counter(), client=api)

    def test_cache_identity_and_tamper_detection(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as temp:
            folder = Path(temp)
            pipeline.cached_compression(folder, 'source', 'llmlingua2', .5, {'text':'result'})
            self.assertIsNone(pipeline.cached_compression(folder, 'source', 'llmlingua2', .4))
            path = next((folder/'compressions').glob('*.json'))
            saved = read_json(path); saved['result']['text'] = 'changed'; write_json(path, saved)
            with self.assertRaises(ValueError):
                pipeline.cached_compression(folder, 'source', 'llmlingua2', .5)

    def test_full_pipeline_mocked_and_offline_replay(self):
        config = load_config(ROOT/'config.json')
        config.update(datasets=['hotpotqa'], sample_size=2, bootstrap_repetitions=4, device='cpu')
        examples = [{'id':str(i),'context':'one two three four five six', 'question':f'Question {i}?',
                     'label':i, 'context_group':'shared'} for i in range(2)]
        engine=Mock(); engine.compress.return_value='one two three four'  # over target is accepted
        api=Mock(); api.complete.return_value=('one two three four', 'summary_hash')
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as temp:
            run=Path(temp)
            write_json(run/'hotpotqa/examples.json', examples)
            with patch.object(pipeline,'create_compressor',return_value=engine), patch.object(pipeline,'make_client',return_value=api), patch('compressors.runtime.verify_assets'):
                pipeline.compress(config,run,Counter())
            self.assertEqual(engine.compress.call_count,2)  # one per method, not per question
            self.assertEqual(api.complete.call_count,1)
            pipeline.compress(config,run,Counter(),offline=True)
            with patch.object(pipeline,'make_client',return_value=api), patch.object(pipeline,'predict_one',return_value={'p_yes':.3,'predicted_label':0,'prediction_request_hash':'fake'}):
                pipeline.predict(config,run,Counter())
            pipeline.evaluate(config,run)
            metrics=read_json(run/'hotpotqa/metrics.json')
            self.assertEqual(sum(len(v) for v in metrics['infogap'].values()),25)
            page=(run/'Report.html').read_text(encoding='utf8')
            self.assertIn('data-language="en"',page)
            self.assertIn('Single-call',page)
            self.assertNotIn('DAC',page)
            rows=read_json(run/'hotpotqa/predictions.json');rows[0]['label']=1
            write_json(run/'hotpotqa/predictions.json',rows)
            with self.assertRaises(ValueError):pipeline.evaluate(config,run)

    def test_offline_missing_cache_never_loads_model(self):
        config=load_config(ROOT/'config.json')
        config['datasets']=['hotpotqa']
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as temp:
            run=Path(temp);write_json(run/'hotpotqa/examples.json',[{'context':'one two three four'}])
            with patch.object(pipeline,'create_compressor') as create, patch.object(pipeline,'make_client') as api:
                with self.assertRaises(RuntimeError):pipeline.compress(config,run,Counter(),offline=True)
                create.assert_not_called();api.assert_not_called()


if __name__ == '__main__':
    unittest.main()
