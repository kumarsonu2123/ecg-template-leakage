import warnings
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from . import leakage as LK
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

def fit_and_score(X_dev, y_dev, folds, X_test, C_grid, attacker="logreg", pca=None, seed=0):
    """Same protocol as Phase 3, with a choice of attacker."""
    if attacker == "logreg":
        return LK.fit_and_score(X_dev, y_dev, folds, X_test, C_grid, pca=pca, seed=seed)[0]
    cv = [(np.where(folds != f)[0], np.where(folds == f)[0]) for f in np.unique(folds)]
    if attacker == "mlp":
        model = make_pipeline(StandardScaler(), MLPClassifier((256,), alpha=1e-3, early_stopping=True,
                                                              max_iter=300, random_state=seed))
        model.fit(X_dev, y_dev)
    elif attacker == "knn":
        model = GridSearchCV(make_pipeline(StandardScaler(), KNeighborsClassifier(weights="distance")),
                             {"kneighborsclassifier__n_neighbors": [5, 15, 31, 63]}, cv=cv, scoring="roc_auc")
        model.fit(X_dev, y_dev)
    else:
        raise ValueError(attacker)
    return model.predict_proba(X_test)[:, 1]

def gains(y, scores, n_boot, n_perm, seed=0):
    """Delta_rand / Delta_raw / Delta_demo with paired-bootstrap 95% CIs and permutation p-values (as Phase 3)."""
    aucs = {k: LK.auc(y, s) for k, s in scores.items()}
    boot = LK.paired_bootstrap(y, scores, n_boot, seed)
    untr = [k for k in scores if k.startswith("untrained_")]
    rows = []
    def add(name, est, b, a, bb):
        lo, hi = np.percentile(b, [2.5, 97.5])
        rows.append({"gain": name, "estimate": est, "ci_low": lo, "ci_high": hi, "ci_halfwidth": (hi - lo) / 2,
                     "p_perm": LK.paired_permutation(y, a, bb, n_perm, seed)})
    pooled = np.mean([LK.percentile_rank(scores[k]) for k in untr], axis=0)
    add("Delta_rand", aucs["identity_template"] - np.mean([aucs[k] for k in untr]),
        boot["identity_template"] - np.mean([boot[k] for k in untr], axis=0), scores["identity_template"], pooled)
    if "raw_median_beat" in scores:
        add("Delta_raw", aucs["identity_template"] - aucs["raw_median_beat"],
            boot["identity_template"] - boot["raw_median_beat"], scores["identity_template"], scores["raw_median_beat"])
    if "age_sex" in scores:
        add("Delta_demo", aucs["template_plus_age_sex"] - aucs["age_sex"],
            boot["template_plus_age_sex"] - boot["age_sex"], scores["template_plus_age_sex"], scores["age_sex"])
    return pd.DataFrame(rows).set_index("gain"), aucs

def bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float); n = len(p); o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1)
    return out

def integrated_gradients(f, x, steps=32):
    """IG from a zero baseline; f maps a batch of windows to one number per window."""
    total = torch.zeros_like(x)
    for a in torch.linspace(1.0 / steps, 1.0, steps, device=x.device):
        xi = (a * x).detach().requires_grad_(True)
        g, = torch.autograd.grad(f(xi).sum(), xi)
        total += g
    return (x * total / steps).detach()

def identity_score_fn(model, x_full_windows):
    """Output to explain for an identity encoder: similarity of window 1 to the record's own template."""
    with torch.no_grad():
        ref = F.normalize(model(x_full_windows.reshape(-1, x_full_windows.shape[-1]))
                          .reshape(x_full_windows.shape[0], x_full_windows.shape[1], -1).mean(1), dim=1)
    return lambda xi: (model(xi) * ref).sum(1)

def overlap(a, b):
    """Per-record Spearman rank correlation of |attribution| over time (raw values, no min-max offset)."""
    return np.array([spearmanr(np.abs(u), np.abs(v))[0] for u, v in zip(a, b)])
