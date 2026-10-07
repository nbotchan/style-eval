import unittest

import numpy as np

from styleval.delta import DeltaModel, burrows_delta, cosine_delta

A = [["the", "the", "cat", "cat"], ["the", "the", "the", "cat"]]
B = [["the", "dog", "dog", "dog"], ["the", "the", "dog", "dog"]]


class DeltaTests(unittest.TestCase):
    def test_formulas_against_hand_calculation(self):
        za, zb = np.array([1.0, -1.0, 2.0]), np.array([0.0, 1.0, 2.0])
        self.assertAlmostEqual(burrows_delta(za, zb), 1.0)  # (1 + 2 + 0) / 3
        self.assertAlmostEqual(cosine_delta(za, zb), 1 - 3 / np.sqrt(30))
        self.assertAlmostEqual(cosine_delta(za, za), 0.0)

    def test_fit_matches_hand_calculation(self):
        model = DeltaModel.fit({"A": A, "B": B}, max_mfw=10, cull=0.5)
        # 'the' is in every chunk; 'cat' and 'dog' in half (author-balanced) and pass cull=0.5.
        # Mean frequencies: the 0.5, dog 0.3125, cat 0.1875, so that is the ranking.
        self.assertEqual(model.vocab, ["the", "dog", "cat"])
        np.testing.assert_allclose(model.mu, [0.5, 0.3125, 0.1875])
        # 'the' per-chunk frequencies are .5 .75 .25 .5 with equal weights:
        # variance = (0 + .0625 + .0625 + 0) / 4 = .03125
        self.assertAlmostEqual(model.sigma[0], np.sqrt(0.03125))

    def test_cull_removes_words_in_too_few_chunks(self):
        model = DeltaModel.fit({"A": A, "B": B}, max_mfw=10, cull=0.75)
        self.assertEqual(model.vocab, ["the"])

    def test_every_author_counts_equally(self):
        # Ten copies of A's chunks must not change the statistics.
        small = DeltaModel.fit({"A": A, "B": B}, max_mfw=10, cull=0.5)
        big = DeltaModel.fit({"A": A * 10, "B": B}, max_mfw=10, cull=0.5)
        np.testing.assert_allclose(small.mu, big.mu)
        np.testing.assert_allclose(small.sigma, big.sigma)

    def test_attribution_of_obvious_chunks(self):
        model = DeltaModel.fit({"A": A, "B": B}, max_mfw=10, cull=0.5)
        Z = model.transform([["cat", "cat", "cat", "the"], ["dog", "dog", "dog", "the"]])
        for method in ("cosine", "burrows"):
            guess = [model.authors[i] for i in model.attribute(Z, 3, method)]
            self.assertEqual(guess, ["A", "B"], method)

    def test_vectorized_distances_match_pairwise_functions(self):
        rng = np.random.default_rng(0)
        model = DeltaModel(
            authors=["x", "y", "z"],
            vocab=["w%d" % i for i in range(20)],
            mu=np.zeros(20),
            sigma=np.ones(20),
            profiles=rng.normal(size=(3, 20)),
        )
        Z = rng.normal(size=(5, 20))
        for n in (5, 20):
            b = model.distances(Z, n, "burrows")
            c = model.distances(Z, n, "cosine")
            for i in range(5):
                for j in range(3):
                    self.assertAlmostEqual(b[i, j], burrows_delta(Z[i, :n], model.profiles[j, :n]))
                    self.assertAlmostEqual(c[i, j], cosine_delta(Z[i, :n], model.profiles[j, :n]))

    def test_unknown_method_rejected(self):
        model = DeltaModel.fit({"A": A, "B": B}, max_mfw=10, cull=0.5)
        with self.assertRaises(ValueError):
            model.distances(model.transform(A), 3, "manhattan")


if __name__ == "__main__":
    unittest.main()
