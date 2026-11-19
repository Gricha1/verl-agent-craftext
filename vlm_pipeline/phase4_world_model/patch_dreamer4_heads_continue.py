#!/usr/bin/env python3
"""Patch a 4ku/dreamer4 (3489e13) checkout (after patch_dreamer4_heads_noproprio.py and patch_dreamer4_heads_text_task.py) with
a CONTINUE head, so phase 2 can learn the environment's native reward (user, 2026-11-19: the reward must not be changed).

Dreamer 4 reads "the episode ended" off the reward head (terminal_bins: the dataset's rule on the reward value). With the
native wood reward (Craftax reward / 50: 0.02 per achievement, 0.002 per health point gained or lost) neither the task
success nor a death has a reward of its own, so the end has to be predicted separately, as in DreamerV3:

agent_heads.py
  AgentHeads(cont_head=True): + cont head on h_t, one logit of "the slot's frame is terminal" (the episode ended with the
  transition into it: task done or death).
train_heads.py
  --cont_head 1: sample_clips also returns term (B, T) = terminals of the transition that arrived with the slot's frame;
  loss term "cont" = binary cross-entropy on the valid reward slots, weight --cont_weight in the RMS-normalised sum; the
  held-out terminal precision / recall / F1 are measured on the cont head (sigmoid > --term_prob) instead of the reward bins.

  python patch_dreamer4_heads_continue.py <dreamer4 checkout>/dreamer4
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])


def patch(path, edits):
    src = path.read_text()
    for old, new in edits:
        if new in src:
            continue
        assert src.count(old) == 1, (path.name, "not found once", old[:70])
        src = src.replace(old, new)
    path.write_text(src)
    print("patched", path)


patch(root / "models" / "agent_heads.py", [
    ("""                 task_text_ids=None, text_vocab: int = 0, text_words=None):
        super().__init__()
        d_in = n_agent * d_model""", """                 task_text_ids=None, text_vocab: int = 0, text_words=None, cont_head: bool = False):
        super().__init__()
        d_in = n_agent * d_model"""),
    ("""        self.value = TwoHotHead(d_in, mtp=0, d_hidden=d_hidden, num_bins=value_bins,
                                bin_low=value_bin_low, bin_high=value_bin_high)
""", """        self.value = TwoHotHead(d_in, mtp=0, d_hidden=d_hidden, num_bins=value_bins,
                                bin_low=value_bin_low, bin_high=value_bin_high)
        if cont_head:   # patched: logit of "the slot's frame is terminal"
            self.cont = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, 1))
"""),
])

patch(root / "train" / "train_heads.py", [
    ("""    ret = np.zeros((B, T), np.float32)
""", """    ret = np.zeros((B, T), np.float32)
    term = np.zeros((B, T), np.float32)   # patched: terminal of the transition that arrived with the slot's frame
"""),
    ("""        valid_r[b] = (rew_here & fresh[b]).astype(np.float32)
""", """        valid_r[b] = (rew_here & fresh[b]).astype(np.float32)
        if "terminals" in ep:   # patched
            term[b] = np.where(rew_here, ep["terminals"][np.maximum(s - 1, 0)], False)
"""),
    ("""            "task": as_t(task), "ret": as_t(ret), "valid_v": as_t(fresh.astype(np.float32))}""",
     """            "task": as_t(task), "ret": as_t(ret), "valid_v": as_t(fresh.astype(np.float32)), "term": as_t(term)}"""),
    ("""            clip["reward"].shape).to(device) < 0.5)
""", """            clip["reward"].shape).to(device) < 0.5)
        if hasattr(heads, "cont"):   # patched: terminals from the continue head
            is_term = torch.sigmoid(heads.cont(h.float()).squeeze(-1)) > term_prob
            true = clip["term"] > 0.5
"""),
    ("""    ap.add_argument("--instr_weight", type=float, default=1.0)
""", """    ap.add_argument("--instr_weight", type=float, default=1.0)
    ap.add_argument("--cont_head", type=int, default=0, help="patched: 1 = continue head on the dataset's terminals "
                    "(for rewards that do not mark the end of an episode, e.g. the native Craftax reward)")
    ap.add_argument("--cont_weight", type=float, default=1.0)
"""),
    ("""    if cfg.text_tokenizer:   # patched: instruction text -> compact token ids
""", """    if cfg.cont_head:   # patched
        agent_cfg["cont_head"] = True
    if cfg.text_tokenizer:   # patched: instruction text -> compact token ids
"""),
    ("""                             **({"instr": cfg.instr_weight} if cfg.text_tokenizer else {})},""",
     """                             **({"instr": cfg.instr_weight} if cfg.text_tokenizer else {}),
                             **({"cont": cfg.cont_weight} if cfg.cont_head else {})},"""),
    ("""                terms["instr"] = instr_loss(heads, h.float(), clip["task"])
""", """                terms["instr"] = instr_loss(heads, h.float(), clip["task"])
            if cfg.cont_head:   # patched: binary cross-entropy of the terminal flag on the valid reward slots
                vr = clip["valid_r"]
                bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    heads.cont(h.float()).squeeze(-1), clip["term"], reduction="none")
                terms["cont"] = (bce * vr).sum() / vr.sum().clamp_min(1.0)
"""),
])
