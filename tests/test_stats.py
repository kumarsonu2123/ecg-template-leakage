"""Fast checks of the statistics and metrics (no data or GPU needed): pytest -q"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ecgleak import leakage as LK, evaluate as EV, analysis as AN


def test_bootstrap_finds_known_gain():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(500), np.zeros(500)]
    strong = y + rng.normal(0, 0.8, len(y))
    weak = y + rng.normal(0, 1.6, len(y))
    boot = LK.paired_bootstrap(y, {"a": strong, "b": weak}, n_boot=300, seed=0)
    d = boot["a"] - boot["b"]
    assert np.percentile(d, 2.5) > 0


def test_permutation_null_is_not_significant():
    rng = np.random.default_rng(1)
    y = np.r_[np.ones(300), np.zeros(300)]
    s = y + rng.normal(0, 1, len(y))
    assert LK.paired_permutation(y, s, s + rng.normal(0, 1e-6, len(y)), n_perm=200, seed=0) > 0.05


def test_shuffled_labels_score_near_chance():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(1000, 8))
    y = rng.permutation(np.r_[np.ones(500), np.zeros(500)]).astype(int)
    auc, _ = EV.cv_auroc(X, y, np.arange(1000) % 5, [1.0])
    assert 0.4 < auc < 0.6


def test_eer_perfect_and_random():
    assert EV.eer(np.ones(50), np.zeros(50)) == 0.0
    rng = np.random.default_rng(3)
    assert 0.35 < EV.eer(rng.normal(size=2000), rng.normal(size=2000)) < 0.65


def test_bh_matches_definition():
    q = AN.bh([0.01, 0.04, 0.03, 0.2])
    assert np.allclose(q, [0.04, 0.16 / 3, 0.16 / 3, 0.2])  # 0.01*4/1; 0.04*4/3 for both middle values; 0.2
