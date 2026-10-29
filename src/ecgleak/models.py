import math
import torch, torch.nn as nn, torch.nn.functional as F

class ECGEncoder(nn.Module):
    """1-lead ECG window (B x 1250) -> 256-d unit vector (the template)."""
    def __init__(self, channels=(32, 64, 128), kernel=7, layers=2, heads=4, emb_dim=256, max_len=1024):
        super().__init__()
        blocks, c_in = [], 1
        for c in channels:
            blocks += [nn.Conv1d(c_in, c, kernel, padding=kernel // 2), nn.BatchNorm1d(c), nn.GELU(), nn.MaxPool1d(2)]
            c_in = c
        self.stem = nn.Sequential(*blocks)
        d = channels[-1]
        self.pos = nn.Parameter(torch.randn(1, max_len, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d, dropout=0.1,
                                           activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.proj = nn.Linear(d, emb_dim)
        self.emb_dim = emb_dim

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.stem(x).transpose(1, 2)
        h = self.transformer(h + self.pos[:, :h.size(1)])
        return F.normalize(self.proj(h.mean(dim=1)), dim=1)

class ArcFaceHead(nn.Module):
    """Additive angular margin: pushes each person's templates into a tight cone."""
    def __init__(self, emb_dim, n_classes, s=30.0, m=0.5):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, emb_dim) * 0.01)
        self.s, self.m = s, m
        self.th, self.mm = math.cos(math.pi - m), math.sin(math.pi - m) * m

    def forward(self, emb, y):
        cos = F.linear(emb, F.normalize(self.W, dim=1)).clamp(-1 + 1e-7, 1 - 1e-7)
        t = cos.gather(1, y[:, None])
        with_margin = torch.cos(torch.acos(t) + self.m)
        t_new = torch.where(t > self.th, with_margin, t - self.mm)
        return self.s * cos.scatter(1, y[:, None], t_new)

def build_encoder(cfg):
    m = cfg["model"]
    return ECGEncoder(m["cnn_channels"], m["kernel"], m["transformer_layers"], m["heads"], m["emb_dim"])

def untrained_copies(cfg, seeds, device):
    """Same architecture, random weights, never trained: the baseline for Delta_rand."""
    out = []
    for s in seeds:
        torch.manual_seed(s)
        out.append(build_encoder(cfg).to(device).eval())
    return out
