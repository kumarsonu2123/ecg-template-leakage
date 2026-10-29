import warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

def fit_and_score(X_dev, y_dev, folds, X_test, C_grid, pca=None, seed=0):
    """Choose C by CV over the probe-dev patient folds, refit on all probe-dev, score the test set once."""
    cv = [(np.where(folds != f)[0], np.where(folds == f)[0]) for f in np.unique(folds)]
    steps = [StandardScaler()]
    if pca:
        steps.append(PCA(n_components=min(pca, X_dev.shape[1]), random_state=seed))
    steps.append(LogisticRegressionCV(Cs=list(C_grid), cv=cv, scoring="roc_auc", max_iter=5000))
    model = make_pipeline(*steps).fit(X_dev, y_dev)
    return model.predict_proba(X_test)[:, 1], float(model[-1].C_[0])

def auc(y, s):
    return float(roc_auc_score(y, s))

def paired_bootstrap(y, scores, n_boot=2000, seed=0):
    """scores: dict name -> test-set scores. Resamples patients (one record each) and recomputes every AUROC."""
    rng = np.random.default_rng(seed)
    n, out = len(y), {k: [] for k in scores}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        for k, s in scores.items():
            out[k].append(roc_auc_score(y[i], s[i]))
    return {k: np.array(v) for k, v in out.items()}

def percentile_rank(s):
    return (np.argsort(np.argsort(s)) + 0.5) / len(s)

def paired_permutation(y, s_a, s_b, n_perm=1000, seed=0):
    """Two-sided p-value for AUROC(a) - AUROC(b): swap the two scores within each patient at random."""
    rng = np.random.default_rng(seed)
    a, b = percentile_rank(s_a), percentile_rank(s_b)
    obs = roc_auc_score(y, a) - roc_auc_score(y, b)
    null = np.empty(n_perm)
    for j in range(n_perm):
        swap = rng.random(len(y)) < 0.5
        null[j] = roc_auc_score(y, np.where(swap, b, a)) - roc_auc_score(y, np.where(swap, a, b))
    return float((1 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1))
