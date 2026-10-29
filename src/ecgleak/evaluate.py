import warnings
import numpy as np, torch, torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

@torch.no_grad()
def embed(model, signals, window, device, batch=256):
    """Template = mean of the embeddings of non-overlapping windows, L2-normalised."""
    model.eval()
    n_win = signals.shape[1] // window
    out = []
    for i in range(0, len(signals), batch):
        x = torch.from_numpy(signals[i:i + batch, :n_win * window].astype(np.float32)).to(device)
        e = model(x.reshape(-1, window)).reshape(-1, n_win, model.emb_dim).mean(1)
        out.append(F.normalize(e, dim=1).cpu().numpy())
    return np.concatenate(out)

def unit_rows(X):
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

def eer(genuine, impostor):
    y = np.r_[np.ones(len(genuine)), np.zeros(len(impostor))]
    fpr, tpr, _ = roc_curve(y, np.r_[genuine, impostor])
    i = np.nanargmin(np.abs((1 - tpr) - fpr))
    return float((fpr[i] + 1 - tpr[i]) / 2)

def verification(emb, patient_ids, dates):
    """Enroll each patient's earliest record; verify their later records against every enrolled patient."""
    order = np.lexsort((dates.astype("datetime64[ns]").astype(np.int64), patient_ids))
    seen, enroll, probe = set(), [], []
    for i in order:
        (probe if patient_ids[i] in seen else enroll).append(i)
        seen.add(patient_ids[i])
    E, P = emb[enroll], emb[probe]
    S = P @ E.T
    same = patient_ids[probe][:, None] == patient_ids[enroll][None, :]
    rank1 = float(np.mean(patient_ids[enroll][S.argmax(1)] == patient_ids[probe]))
    return {"eer": eer(S[same], S[~same]), "rank1": rank1, "n_enrolled": len(enroll), "n_probes": len(probe)}

def task_data(labels, names, cls):
    """Positives carry the class; negatives are NORM-only records."""
    names = list(names)
    pos = labels[:, names.index(cls)] == 1
    neg = (labels[:, names.index("NORM")] == 1) & (labels.sum(1) == 1)
    keep = pos | neg
    return keep, pos[keep].astype(int)

def demo_features(age, sex):
    a = np.where(np.isfinite(age), np.clip(age, 0, 90), np.nan)
    a = np.where(np.isnan(a), np.nanmedian(a), a)
    s = np.where(np.isfinite(sex), sex, 0.5)
    return np.c_[a, s]

def cv_auroc(X, y, folds, C_grid, pca=None, seed=0):
    """Out-of-fold AUROC over the given patient folds; C chosen by inner 3-fold CV on training folds only."""
    oof = np.full(len(y), np.nan)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        steps = [StandardScaler()]
        if pca:
            steps.append(PCA(n_components=min(pca, X.shape[1], int(tr.sum()) - 1), random_state=seed))
        steps.append(LogisticRegressionCV(Cs=list(C_grid), cv=3, scoring="roc_auc", max_iter=3000))
        model = make_pipeline(*steps).fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof)), oof
