import glob, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from .models import build_encoder, ArcFaceHead

def _schedule(opt, total, warm):
    def f(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)

def latest_checkpoint(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "epoch_*.pt")))
    return files[-1] if files else None

def _crops(signals, idx, offs, scale, window, rng):
    xb = np.stack([signals[i, o:o + window] for i, o in zip(idx, offs)]).astype(np.float32)
    return (xb * scale[:, None] + rng.normal(0, 0.05, xb.shape)).astype(np.float32)

def train_identity(cfg, signals, patient_ids, run_dir, device, epochs=None, eval_fn=None, crops_per_record=4,
                   seed=None, stop_after=None, max_steps_per_epoch=None, log=print, loss="arcface",
                   patience=None, snapshot_epochs=(), margin=0.3):
    """Identity training (ArcFace, or a Siamese contrastive loss) with automatic resume from run_dir."""
    t, seed = cfg["train"], cfg["seed"] if seed is None else seed
    epochs, patience = epochs or t["epochs"], patience or t["patience"]
    window = int(cfg["preprocess"]["window_s"] * cfg["data"]["fs"])
    classes, y_all = np.unique(patient_ids, return_inverse=True)
    members = [np.where(y_all == c)[0] for c in range(len(classes))]
    n, L, B = len(signals), signals.shape[1], t["batch"]
    steps = math.ceil(n * crops_per_record / B)
    if max_steps_per_epoch:
        steps = min(steps, max_steps_per_epoch)

    torch.manual_seed(seed)
    model = build_encoder(cfg).to(device)
    head = ArcFaceHead(cfg["model"]["emb_dim"], len(classes), cfg["arcface"]["s"], cfg["arcface"]["m"]).to(device) \
        if loss == "arcface" else None
    params = list(model.parameters()) + (list(head.parameters()) if head is not None else [])
    opt = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"])
    sched = _schedule(opt, steps * epochs, int(t["warmup_frac"] * steps * epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "snapshots"), exist_ok=True)

    start, history, best, bad = 0, [], float("inf"), 0
    ck = latest_checkpoint(run_dir)
    if ck:
        st = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        if head is not None:
            head.load_state_dict(st["head"])
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        torch.set_rng_state(st["torch_rng"].cpu())
        if device == "cuda" and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda_rng"]])
        start, history, best, bad = st["epoch"] + 1, st["history"], st["best"], st["bad"]
        log(f"Resumed from {os.path.basename(ck)} -> continuing at epoch {start + 1}")
        if bad >= patience or (stop_after is not None and start >= stop_after):
            log("This run had already finished.")
            return model, history

    for epoch in range(start, epochs):
        t0 = time.time()
        model.train()
        if head is not None:
            head.train()
        rng = np.random.default_rng(seed * 1000 + epoch)
        idx = np.repeat(np.arange(n), crops_per_record); rng.shuffle(idx)
        tot_loss = tot_acc = 0.0
        for b in range(steps):
            ii = idx[b * B:(b + 1) * B]
            m = len(ii)
            if loss == "arcface":
                xb = _crops(signals, ii, rng.integers(0, L - window + 1, m), rng.uniform(0.9, 1.1, m), window, rng)
                yb = torch.from_numpy(y_all[ii]).long().to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                    emb = model(torch.from_numpy(xb).to(device))
                logits = head(emb.float(), yb)
                L_b = F.cross_entropy(logits, yb)
                acc = (logits.argmax(1) == yb).float().mean().item()
            else:
                pos = np.array([rng.choice(members[y_all[i]][members[y_all[i]] != i])
                                if len(members[y_all[i]]) > 1 else i for i in ii])
                neg = rng.integers(0, n, m)
                clash = y_all[neg] == y_all[ii]
                while clash.any():
                    neg[clash] = rng.integers(0, n, clash.sum()); clash = y_all[neg] == y_all[ii]
                allx = np.concatenate([_crops(signals, j, rng.integers(0, L - window + 1, m), rng.uniform(0.9, 1.1, m), window, rng)
                                       for j in (ii, pos, neg)])
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                    e = model(torch.from_numpy(allx).to(device)).float()
                a, p, q = e[:m], e[m:2 * m], e[2 * m:]
                cap, can = (a * p).sum(1), (a * q).sum(1)
                L_b = ((1 - cap) ** 2).mean() + (F.relu(can - margin) ** 2).mean()
                acc = (cap > can).float().mean().item()
            opt.zero_grad(set_to_none=True)
            scaler.scale(L_b).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot_loss += L_b.item(); tot_acc += acc
        rec = {"epoch": epoch + 1, "loss": tot_loss / steps, "train_acc": tot_acc / steps,
               "lr": sched.get_last_lr()[0], "seconds": round(time.time() - t0, 1)}
        if eval_fn is not None:
            rec.update(eval_fn(model))
        if rec.get("eer", -epoch) < best:
            best, bad = rec.get("eer", -epoch), 0
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "eer": rec.get("eer")},
                       os.path.join(run_dir, "best.pt"))
        else:
            bad += 1
        if epoch + 1 in snapshot_epochs:
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "eer": rec.get("eer")},
                       os.path.join(run_dir, "snapshots", f"epoch_{epoch + 1:03d}.pt"))
        history.append(rec)
        path = os.path.join(run_dir, "checkpoints", f"epoch_{epoch + 1:03d}.pt")
        torch.save({"epoch": epoch, "model": model.state_dict(), "head": head.state_dict() if head is not None else None,
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "history": history, "best": best, "bad": bad, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None}, path + ".tmp")
        os.replace(path + ".tmp", path)
        for old in sorted(glob.glob(os.path.join(run_dir, "checkpoints", "epoch_*.pt")))[:-3]:
            os.remove(old)
        log(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()))
        if stop_after is not None and epoch + 1 >= stop_after:
            break
        if bad >= patience:
            log("Early stopping: identity EER stopped improving")
            break
    return model, history

def load_best(cfg, run_dir, device, name="best.pt"):
    model = build_encoder(cfg).to(device)
    st = torch.load(os.path.join(run_dir, name), map_location=device, weights_only=False)
    model.load_state_dict(st["model"])
    return model.eval(), st

class DiagNet(nn.Module):
    """Same encoder + one linear output: a supervised diagnosis model (used only for attributions, H4)."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder, self.out = encoder, nn.Linear(encoder.emb_dim, 1)
    def forward(self, x):
        return self.out(self.encoder(x)).squeeze(1)

def train_diagnosis(cfg, signals, y, path, device, epochs=15, seed=0, log=print):
    window = int(cfg["preprocess"]["window_s"] * cfg["data"]["fs"])
    torch.manual_seed(seed)
    net = DiagNet(build_encoder(cfg)).to(device)
    if os.path.isfile(path):
        net.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        log(f"Loaded existing {os.path.basename(path)}")
        return net.eval()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pw = torch.tensor([(1 - y).sum() / max(1, y.sum())], device=device, dtype=torch.float32)
    n, L, B = len(signals), signals.shape[1], cfg["train"]["batch"]
    for epoch in range(epochs):
        net.train()
        rng = np.random.default_rng(seed * 1000 + epoch)
        idx = np.repeat(np.arange(n), 2); rng.shuffle(idx)
        tot = 0.0
        for b in range(math.ceil(len(idx) / B)):
            ii = idx[b * B:(b + 1) * B]
            xb = _crops(signals, ii, rng.integers(0, L - window + 1, len(ii)), rng.uniform(0.9, 1.1, len(ii)), window, rng)
            logit = net(torch.from_numpy(xb).to(device))
            l = F.binary_cross_entropy_with_logits(logit, torch.from_numpy(y[ii]).float().to(device), pos_weight=pw)
            opt.zero_grad(set_to_none=True); l.backward(); opt.step(); tot += l.item()
        log(f"  diag epoch {epoch + 1}: loss {tot / math.ceil(len(idx) / B):.4f}")
    torch.save(net.state_dict(), path)
    return net.eval()
