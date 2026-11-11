"""Unit tests for irdiv/diversify.py (no LLM, no data files, runs in about a second).

  lazy greedy == plain greedy          (same picks when there are no exact ties; same objective value always)
  greedy >= (1 - 1/e) * exact optimum  (the guarantee for monotone submodular maximisation, checked by brute force on small pools)

Run from the repo root or from backend/ (the path to backend/ is resolved from this file, not from the working directory):
    python -m unittest discover -s tests -p "test_diversify.py" -v        # from the repo root
    python -m unittest discover -s ../tests -p "test_diversify.py" -v     # from backend/
"""
import math, os, sys, unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
from irdiv.diversify import exact, greedy, lazy_greedy, mmr, objective

LAMS = (0.0, 0.3, 0.7, 1.0)
EPS = 1e-9


def make_pool(rng, n, dim=6, dup=False):
    """Random relevance r in [0,1] and a cosine-style similarity matrix S >= 0 with unit diagonal (what the real pipeline produces).
    dup=True copies the first half of the pool into the second half, creating exact ties between documents."""
    r = rng.random(n)
    X = rng.random((n, dim))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    if dup:
        half = n // 2
        X[half:], r[half:] = X[:n - half], r[:n - half]
    return r, X @ X.T


def cases(n_seeds=60, dup=False):
    """Yields (r, S, k, lam, weighted) over small pools: n in 8..12, k in 2..4."""
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        n, k = int(rng.integers(8, 13)), int(rng.integers(2, 5))
        r, S = make_pool(rng, n, dup=dup)
        for lam in LAMS:
            for weighted in (True, False):
                yield r, S, k, lam, weighted


class LazyGreedyMatchesGreedy(unittest.TestCase):
    def test_same_selection_without_ties(self):
        for r, S, k, lam, w in cases():
            g, _ = greedy(r, S, k, lam, w)
            l, _ = lazy_greedy(r, S, k, lam, w)
            self.assertEqual(g, l, f"k={k} lam={lam} weighted={w}")

    def test_same_objective_with_exact_duplicates(self):
        """Duplicate documents are exact ties; the two implementations may pick different twins, but the value of the set must be identical."""
        for r, S, k, lam, w in cases(dup=True):
            g, _ = greedy(r, S, k, lam, w)
            l, _ = lazy_greedy(r, S, k, lam, w)
            self.assertAlmostEqual(objective(g, r, S, k, lam, w), objective(l, r, S, k, lam, w), delta=EPS, msg=f"k={k} lam={lam} weighted={w}")

    def test_lazy_needs_no_more_evaluations_than_plain_greedy(self):
        for r, S, k, lam, w in cases(n_seeds=20):
            _, evals = lazy_greedy(r, S, k, lam, w)
            self.assertLessEqual(evals, len(r) * k)           # plain greedy evaluates every candidate in each of the k rounds
            self.assertGreaterEqual(evals, len(r))            # the initial pass touches every candidate once

    def test_selection_is_valid(self):
        for r, S, k, lam, w in cases(n_seeds=20):
            for sel in (greedy(r, S, k, lam, w)[0], lazy_greedy(r, S, k, lam, w)[0], mmr(r, S, k, lam)):
                self.assertEqual(len(sel), k)
                self.assertEqual(len(set(sel)), k, "no document may be picked twice")
                self.assertTrue(all(0 <= i < len(r) for i in sel))


class GreedyApproximationGuarantee(unittest.TestCase):
    def test_objective_is_monotone_submodular(self):
        """The premise of the (1-1/e) bound, checked directly: f(A+x)-f(A) >= f(B+x)-f(B) for A subset of B, and f(A+x) >= f(A)."""
        rng = np.random.default_rng(0)
        for _ in range(200):
            r, S = make_pool(rng, 10)
            lam, w = float(rng.choice(LAMS)), bool(rng.integers(2))
            f = lambda sel: objective(sel, r, S, 5, lam, w) if sel else 0.0
            perm = [int(i) for i in rng.permutation(10)]
            A, B, x = perm[:2], perm[:5], perm[5]
            self.assertGreaterEqual(f(A + [x]), f(A) - EPS)
            self.assertGreaterEqual(f(A + [x]) - f(A), f(B + [x]) - f(B) - EPS)

    def test_greedy_reaches_1_minus_1_over_e_of_exact_optimum(self):
        bound, worst = 1 - 1 / math.e, 1.0
        for r, S, k, lam, w in cases():
            got = objective(greedy(r, S, k, lam, w)[0], r, S, k, lam, w)
            opt = objective(exact(r, S, k, lam, w), r, S, k, lam, w)
            self.assertGreaterEqual(got, bound * opt - EPS, f"k={k} lam={lam} weighted={w}: greedy {got:.4f} vs optimum {opt:.4f}")
            self.assertLessEqual(got, opt + EPS, "greedy cannot beat the exact optimum")
            if opt > 0:
                worst = min(worst, got / opt)
        print(f"\n  worst greedy/optimum ratio over all cases: {worst:.3f} (bound {bound:.3f})")

    def test_lazy_greedy_also_meets_the_bound(self):
        bound = 1 - 1 / math.e
        for r, S, k, lam, w in cases(n_seeds=30):
            got = objective(lazy_greedy(r, S, k, lam, w)[0], r, S, k, lam, w)
            opt = objective(exact(r, S, k, lam, w), r, S, k, lam, w)
            self.assertGreaterEqual(got, bound * opt - EPS)

    def test_exact_is_really_optimal(self):
        """Guards the oracle the bound test relies on: no k-subset may beat exact()."""
        import itertools
        r, S = make_pool(np.random.default_rng(1), 8)
        best = objective(exact(r, S, 3, 0.7), r, S, 3, 0.7)
        for c in itertools.combinations(range(8), 3):
            self.assertLessEqual(objective(c, r, S, 3, 0.7), best + EPS)

    def test_gains_telescope_to_the_objective(self):
        """greedy() reports the marginal gain of each pick; they must sum to the objective of the final set."""
        for r, S, k, lam, w in cases(n_seeds=20):
            sel, gains = greedy(r, S, k, lam, w)
            self.assertAlmostEqual(sum(gains), objective(sel, r, S, k, lam, w), delta=EPS)
            self.assertTrue(all(a >= b - EPS for a, b in zip(gains, gains[1:])), "gains must be non-increasing (diminishing returns)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
