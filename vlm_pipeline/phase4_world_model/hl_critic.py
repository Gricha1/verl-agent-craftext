"""HL-Gauss classification critic of the SLM planner (user 2026-11-21: "обучи на 1й стадии такого же критика как у них вместо
Value нашей и потом дообучение его в РЛ стадии"), the critic of the ZeroHero crafter-vision-12m PPO
(scripts/experiments/crafter_qwen_ppo/README.md of iameteron/ZeroHero, branch crafter-vision-12m):

  * a separate copy of the SLM (adapter, backbone, readouts; no weights shared with the acting SLM) that reads the same prompt
    (System + Z_t + Task) and takes the hidden state h of the last prompt position "Assistant:\\n";
  * head = MLP(d -> d -> 101) over bins c_1..c_101 uniform in symlog space on [-3, 3] (DreamerV3 symlog(x) = sign(x) ln(1 + |x|),
    raw range about -19..19);
  * a target return G becomes the Gaussian N(symlog(G), sigma^2), sigma = 0.75 bin width, integrated over the bins
    p_i = Phi((e_{i+1} - y) / sigma) - Phi((e_i - y) / sigma) (e the bin edges, y = symlog G clipped to [-3, 3], renormalised);
    loss = cross-entropy H(p, softmax(logits))  (HL-Gauss, Farebrother et al. 2024, "Stop Regressing");
  * V(s) = sum_i softmax(logits)_i symexp(c_i).

`HLGaussValue(h)` returns V with shape (B, 1), so every `value_head(h).squeeze(-1)` of the SLM code reads the expected value of a
critic checkpoint unchanged. A critic checkpoint is an SLMReasonAct checkpoint with a "critic_hl" dict (the bin config);
`attach_for(model, state)` swaps the head before load_state_dict.

Real returns (stage 2): along a played episode x_0..x_{n-1} with rewards r_t of the transition x_t -> x_{t+1} (t <= n - 2),
    G^lambda_t = r_t + gamma [ (1 - lambda) V(x_{t+1}) + lambda G^lambda_{t+1} ],   G^lambda_{n-1} = v_end
with v_end = the value of the final frame: 0 at a death, 1 at a finished wood task (the SLM value G_t = gamma^(T-1-t) of a
success has V = 1 in its last frame, and its rewards are 0), V(x_{n-1}) at a timeout (bootstrap). These are the GAE(lambda)
returns of PPO (A_t = G^lambda_t - V(x_t)).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def symlog(x):
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x):
    return torch.sign(x) * torch.expm1(x.abs())


class HLGaussValue(nn.Module):
    def __init__(self, d, bins=101, lo=-3.0, hi=3.0, sigma_bins=0.75):
        super().__init__()
        self.cfg = dict(bins=bins, lo=lo, hi=hi, sigma_bins=sigma_bins)
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, bins))
        edges = torch.linspace(lo, hi, bins + 1)
        self.register_buffer("edges", edges, persistent=False)
        self.register_buffer("centers", (edges[1:] + edges[:-1]) / 2, persistent=False)
        self.sigma = sigma_bins * (hi - lo) / bins

    def logits(self, h):
        return self.net(h).float()

    def mean(self, logits):
        return (logits.softmax(-1) * symexp(self.centers)).sum(-1)

    def forward(self, h):
        return self.mean(self.logits(h))[:, None]

    def target_probs(self, G):
        y = symlog(G.float()).clamp(self.cfg["lo"], self.cfg["hi"])
        cdf = 0.5 * (1 + torch.erf((self.edges[None] - y[:, None]) / (self.sigma * math.sqrt(2))))
        p = cdf[:, 1:] - cdf[:, :-1]
        return p / p.sum(-1, keepdim=True).clamp_min(1e-8)

    def loss(self, logits, G, mask=None):
        """mean cross-entropy over the rows (mask: weights per row)."""
        ce = -(self.target_probs(G) * logits.float().log_softmax(-1)).sum(-1)
        if mask is None:
            return ce.mean()
        return (ce * mask).sum() / mask.sum().clamp_min(1)


def attach(model, cfg=None):
    """Replace the SLM's scalar value head by an HL-Gauss head (fresh weights)."""
    d = model.value_head[0].in_features if isinstance(model.value_head, nn.Sequential) else model.value_head.net[0].in_features
    dev = next(model.parameters()).device
    model.value_head = HLGaussValue(d, **(cfg or {})).to(dev)
    return model


def attach_for(model, state):
    """Before load_state_dict(state["model"]): give the model the head type of the checkpoint."""
    if state.get("critic_hl") is not None:
        attach(model, state["critic_hl"])
    return model


def lambda_returns(v, r, v_end, gamma=0.99, lam=0.95):
    """One episode: v [n] critic values of its frames, r [n] rewards (r[t] of x_t -> x_{t+1}, r[n-1] unused), v_end the value
    of the last frame -> G^lambda [n] (G[n-1] = v_end)."""
    n = len(v)
    G = torch.empty(n, dtype=torch.float32)
    G[n - 1] = v_end
    for t in range(n - 2, -1, -1):
        G[t] = r[t] + gamma * ((1 - lam) * v[t + 1] + lam * G[t + 1])
    return G


def end_value(e, v_last, game):
    """Value of the last saved frame of a played episode (episodes.json row e of eval_slm_d4_env.py --save-traj)."""
    if e["end"] in ("timeout", "other"):
        return float(v_last)                                   # cut by --max-steps: bootstrap
    return 1.0 if (game == "wood" and e["success"]) else 0.0   # wood task done by the next action: V = 1; a death: 0


def online_targets(v, rewards, starts, lengths, eps, game, gamma=0.99, lam=0.95):
    """G^lambda of every frame of a traj dir. v [N] critic values, rewards [N] (traj rewards.npy; the wood critic's return is
    the task's, so its rewards are 0)."""
    G = torch.empty(len(v), dtype=torch.float32)
    for s, n, e in zip(starts, lengths, eps):
        vv = v[s:s + n].float().cpu()
        rr = torch.zeros(n) if game == "wood" else torch.as_tensor(rewards[s:s + n], dtype=torch.float32)
        G[s:s + n] = lambda_returns(vv, rr, end_value(e, vv[-1], game), gamma, lam)
    return G
