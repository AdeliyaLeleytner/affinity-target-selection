import unittest
import numpy as np
from study.metrics import pair_order, evaluate


class IntervalOrder(unittest.TestCase):
    def test_exact_open_closed_unknown(self):
        lower = np.array([[7., 6.], [5., -np.inf], [5., -np.inf], [np.nan, 3.], [6., 6.]])
        upper = np.array([[7., 6.], [5., 5.], [5., 5.], [np.nan, 3.], [6., 6.]])
        opened = np.array([[False, False], [False, True], [False, False], [False, False], [False, False]])
        got = pair_order(lower, upper, opened, np.array([[0, 1]]))
        np.testing.assert_array_equal(got[:, 0], [1, 1, 0, 0, 0])

    def test_score_tie_and_reverse(self):
        scores = np.array([[[1., 1.]], [[0., 1.]]])
        exact = np.array([[7., 6.]])
        rows = evaluate(scores, exact, exact, np.zeros_like(exact, bool),
                        ["tie", "reverse"], ["molecule"], "target", "toy", "exact")
        actual = {r["method"]: r["concordance"] for r in rows if r["stratum"] == "both_exact"}
        self.assertEqual(actual, {"tie": 0.5, "reverse": 0.0})


if __name__ == "__main__":
    unittest.main()
