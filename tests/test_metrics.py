import math
import random
import unittest

from metrics import bootstrap_gaps, forecast_scores, infogap


def decision_payoff_gap(a, b, labels, threshold):
    """独立核验：两动作的实际收益，不调用 h_t 或事件扫描算法。"""
    def utility(p, y):
        return threshold - y if p <= threshold else y - threshold
    return sum(utility(pa, y) - utility(pb, y) for pa, pb, y in zip(a, b, labels)) / len(labels)


class InfoGapTests(unittest.TestCase):
    def test_perfect_vs_uninformative_has_expected_scale_and_direction(self):
        self.assertAlmostEqual(infogap([0, 1], [.5, .5], [0, 1])["value"], .5)
        self.assertAlmostEqual(infogap([.5, .5], [0, 1], [0, 1])["value"], 0)

    def test_identical_predictors_zero_even_if_wrong(self):
        self.assertEqual(infogap([1, 0, .2], [1, 0, .2], [0, 1, 1])["value"], 0)

    def test_left_limit_is_not_lost_at_an_atom(self):
        # D(t)=t until t=.9, where it drops to 0; the supremum is approached from below.
        result = infogap([.1], [.9], [0])
        self.assertAlmostEqual(result["value"], 1.8)
        self.assertEqual(result["side"], "left_limit")
        self.assertEqual(result["threshold"], .9)

    def test_endpoint_and_tie_convention(self):
        self.assertAlmostEqual(infogap([1], [0], [1])["value"], 2)
        self.assertAlmostEqual(infogap([0], [1], [1])["value"], 0)

    def test_matches_independently_evaluated_decision_problems(self):
        rng = random.Random(431)
        for _ in range(100):
            a = [rng.choice([0, .1, .4, .5, .9, 1]) for _ in range(15)]
            b = [rng.choice([0, .2, .5, .8, 1]) for _ in range(15)]
            labels = [rng.randrange(2) for _ in a]
            thresholds = {0., 1., *a, *b}
            candidates = thresholds | {math.nextafter(t, -math.inf) for t in thresholds if t > 0}
            expected = max(0., *(decision_payoff_gap(a, b, labels, t) for t in candidates))
            self.assertAlmostEqual(infogap(a, b, labels)["value"], expected, places=12)

    def test_invalid_probabilities_or_labels_are_rejected(self):
        for a, b, y in [([], [], []), ([.1], [.2,.3], [0]), ([math.nan], [.2], [0]), ([1.1], [.2], [0]), ([True], [.2], [0]), ([.1], [.2], [2])]:
            with self.assertRaises(ValueError):
                infogap(a, b, y)

    def test_scores_keep_probability_information(self):
        strong = forecast_scores([.95], [1])
        weak = forecast_scores([.51], [1])
        self.assertEqual(strong["accuracy"], weak["accuracy"])
        self.assertLess(strong["brier"], weak["brier"])
        self.assertLess(strong["log_loss"], weak["log_loss"])

    def test_bootstrap_preserves_pairs_and_is_reproducible(self):
        args = ([0,.8,.2], [0,.8,.2], [0,1,0], ["x","x","y"])
        first = bootstrap_gaps(*args, repetitions=30, seed=2)
        self.assertEqual(first, bootstrap_gaps(*args, repetitions=30, seed=2))
        self.assertEqual(first["a_over_b"], [0,0])
        self.assertEqual(first["context_clusters"], 2)


if __name__ == "__main__":
    unittest.main()

