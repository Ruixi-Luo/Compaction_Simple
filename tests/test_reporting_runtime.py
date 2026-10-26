"""Offline checks for diagnostic plots, live usage and the parallel main pipeline."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run as pipeline
from metrics import calibration_summary, score_intervals, comparability_curve, infogap
from reader import prediction_messages, summary_messages
from storage import ROOT, read_json, write_json
from telemetry import Progress, api_usage


class Counter:
    def count(self, text):
        return len(text.split())


class DiagnosticsTests(unittest.TestCase):
    def test_fixed_bins_include_one_and_show_constant_forecasts(self):
        cal = calibration_summary([.5]*100, [1]*71+[0]*29)
        self.assertAlmostEqual(cal['ece'], .21)
        self.assertEqual(cal['exact_counts'], {'0.5': 100})
        self.assertEqual(cal['bins'][2]['n'], 100)
        end = calibration_summary([0., 1.], [0, 1])
        self.assertEqual([b['n'] for b in end['bins']], [1, 0, 0, 0, 1])
        self.assertEqual(end['ece'], 0)

    def test_paired_bootstrap_identical_predictions_have_zero_differences(self):
        args = ({'full': [.2, .8, .4], 'same': [.2, .8, .4]}, [0, 1, 0], ['a', 'a', 'b'], 20, 42)
        result = score_intervals(*args)
        self.assertEqual(result, score_intervals(*args))
        self.assertTrue(all(v == [0, 0] for v in result['same']['delta_vs_full'].values()))

    def test_comparability_is_tolerance_not_a_ranking(self):
        m = {'a': {'a': {'value': 0}, 'b': {'value': .2}},
             'b': {'a': {'value': .1}, 'b': {'value': 0}}}
        curve = comparability_curve(m, ['a', 'b'])
        self.assertEqual([(x['comparable'], x['strictly_comparable']) for x in curve], [(0, 0), (1, 1), (1, 0)])

    def test_token_usage_counts_attempts_once_and_keeps_unknown_usage(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as temp:
            run = Path(temp)
            entries = [{'status': 'received', 'usage': {'prompt_tokens': 100, 'completion_tokens': 9,
                       'prompt_tokens_details': {'cached_tokens': 40}}, 'elapsed_seconds': 2},
                       {'status': 'network_error'}, {'status': 'started'}]
            write_json(run/'api/reader/worker_1/api_attempts.json', entries)
            write_json(run/'api/reader/worker_2/api_attempts.json', [entries[0]])
            write_json(run/'api/reader/worker_1/api_cache/cached.json', {'not_an_attempt': True})
            u = api_usage(run)['reader']
            self.assertEqual((u['attempts'], u['input_tokens'], u['output_tokens'], u['cached_input_tokens']), (4, 200, 18, 80))
            self.assertEqual((u['unknown_usage'], u['in_flight']), (1, 1))
            self.assertEqual(api_usage(run)['reader'], u)
            progress = Progress(run, 'offline_test')
            with progress.stage('predict'):
                progress.update('predict', 'synthetic', 1, 2)
            progress.finish('complete')
            saved = read_json(run/'telemetry.json')
            self.assertEqual(saved['sessions'][0]['status'], 'complete')
            self.assertGreater(saved['execution_seconds'], 0)

    def test_parallel_main_pipeline_preserves_order_and_query_boundary(self):
        c = read_json(ROOT/'config.json')
        c.update(datasets=['hotpotqa'], sample_size=3, api_workers=2, bootstrap_repetitions=8)
        c['summary']['min_interval_seconds'] = c['reader']['min_interval_seconds'] = 0
        examples = [{'id': str(i), 'context': 'a b c d e f', 'context_group': 'one',
                     'question': f'PRIVATE_QUERY_{i}', 'label': i % 2} for i in range(3)]
        requests = []
        class Fake:
            def complete(self, messages, maximum, json_mode=False):
                requests.append((messages, json_mode))
                if not json_mode:
                    self_outer.assertEqual(messages, summary_messages('a b c d e f', 3))
                    return 'a b c', 'offline_summary'
                return '{"p_yes":0.6}', 'offline_reader'
        self_outer = self
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as temp:
            run = Path(temp)
            write_json(run/'hotpotqa/examples.json', examples)
            for m in ('selective_context', 'llmlingua2'):
                pipeline.cached_compression(run/'hotpotqa', examples[0]['context'], m, .5,
                    {'text': 'a b c', 'tokens': 3, 'budget_met': True, 'length_repair_used': False,
                     'compression_seconds': .1})
            with patch.object(pipeline, 'TwoWorkerClient', return_value=Fake()):
                pipeline.compress(c, run, Counter())
                pipeline.predict(c, run, Counter())
            self.assertEqual(sum(not j for _, j in requests), 1)
            self.assertEqual(sum(j for _, j in requests), 15)
            self.assertNotIn('PRIVATE_QUERY', str(next(m for m, j in requests if not j)))
            rows = read_json(run/'hotpotqa/predictions.json')
            self.assertEqual([r['id'] for r in rows], ['0', '1', '2'])
            self.assertTrue(all(list(r['conditions']) == c['methods'] for r in rows))
            pipeline.evaluate(c, run)
            page = (run/'Report.html').read_text(encoding='utf8')
            for marker in ('Qwen summary', 'Decision payoff', 'Calibration and probability',
                           'Actual retention and performance', 'Paired metric', 'tolerance and comparability'):
                self.assertIn(marker, page)


if __name__ == '__main__':
    unittest.main()
