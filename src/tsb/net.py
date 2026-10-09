import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

LEVELS = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

class QNet(nn.Module):
    """Context encoder + shared per-step head.  Predicts 9 monotone deciles (relative to the weekly mean level)."""
    def __init__(self, n_ctx, K, n_lag_f, n_step_f, hid=256, emb=64):
        super().__init__()
        self.cfg = dict(n_ctx=n_ctx, K=K, n_lag_f=n_lag_f, n_step_f=n_step_f, hid=hid, emb=emb)
        self.enc = nn.Sequential(nn.Linear(n_ctx, hid), nn.GELU(), nn.Linear(hid, emb), nn.GELU())
        self.head = nn.Sequential(nn.Linear(emb + K * n_lag_f + n_step_f, hid), nn.GELU(),
                                  nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 9))
        with torch.no_grad():
            self.head[-1].weight.mul_(0.1)
            self.head[-1].bias.zero_()

    def forward(self, ctx, lag, step, base):
        """ctx [B,n_ctx], lag [B,H,K,F], step [B,H,S], base [B,H] -> q [B,H,9] (value-1 units)."""
        B, H = lag.shape[:2]
        e = self.enc(ctx)
        x = torch.cat([e[:, None, :].expand(B, H, -1), lag.reshape(B, H, -1), step], -1)
        o = self.head(x)
        med = base + o[..., 0]
        g = F.softplus(o[..., 1:] + 0.0) * 0.01 + 1e-4
        lo = torch.cumsum(g[..., :4], -1)          # distances below the median for 0.4,0.3,0.2,0.1
        hi = torch.cumsum(g[..., 4:], -1)          # distances above for 0.6..0.9
        q = torch.cat([med[..., None] - lo.flip(-1), med[..., None], med[..., None] + hi], -1)
        return q

def pinball(q, y, ymask, w):
    """q [B,H,9], y [B,H], ymask [B,H], w [B] -> scalar (mean over valid points, weighted per sample)."""
    lv = LEVELS.to(q.dtype)
    d = y[..., None] - q
    loss = torch.maximum(lv * d, (lv - 1.0) * d).mean(-1)         # mean over levels
    num = (loss * ymask).sum(1)
    den = ymask.sum(1).clamp(min=1)
    per = num / den
    return (per * w).sum() / w.sum().clamp(min=1e-9)

def n_params(m):
    return sum(p.numel() for p in m.parameters())
