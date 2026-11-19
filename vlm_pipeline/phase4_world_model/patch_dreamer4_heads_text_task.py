#!/usr/bin/env python3
"""Patch a 4ku/dreamer4 (3489e13) checkout (after patch_dreamer4_heads_noproprio.py) so that phase 2 gets the task as TEXT
and must reconstruct it (user, 2026-11-19).

agent_heads.py
  AgentTokenEncoder: with task_text_ids (per task: token ids of the instruction, remapped to a compact vocabulary of the
    instruction tokens, padded with V) the task embedding is MLP(mean of the word embeddings of the instruction) instead
    of a one-hot lookup; the paper's own remark "text embeddings could easily be used".
  AgentHeads: + instr head on h_t: per position of the instruction a V+1-way softmax (V words + pad), i.e. the agent
    output must carry the instruction text; instr_loss(h, task) = mean cross-entropy, instr_exact(h, task) = share of
    slots whose argmax sequence equals the instruction.
train_heads.py
  --text_tokenizer <dir with tokenizer.json>: tokenizes the dataset's manifest["task_names"] (index = task id) and turns
  the above on; --instr_weight (default 1) weights the instr term in the RMS-normalised sum; the held-out exact match is
  printed with every evaluation.

  python patch_dreamer4_heads_text_task.py <dreamer4 checkout>/dreamer4
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
    ("""    def __init__(self, d_model: int, *, n_agent: int = 1, n_tasks: int = 1):
        super().__init__()
        if n_agent < 1:
            raise ValueError(f"n_agent must be >= 1, got {n_agent}")
        self.d_model = d_model
        self.n_agent = n_agent
        self.n_tasks = n_tasks
        self.task = nn.Embedding(n_tasks, d_model)
        nn.init.normal_(self.task.weight, std=0.02)
""", """    def __init__(self, d_model: int, *, n_agent: int = 1, n_tasks: int = 1, task_text_ids=None, text_vocab: int = 0):
        super().__init__()
        if n_agent < 1:
            raise ValueError(f"n_agent must be >= 1, got {n_agent}")
        self.d_model = d_model
        self.n_agent = n_agent
        self.n_tasks = n_tasks
        self.text = task_text_ids is not None          # patched: task as text (instruction tokens)
        if self.text:
            self.register_buffer("text_ids", torch.tensor(task_text_ids, dtype=torch.long), persistent=False)
            self.word = nn.Embedding(text_vocab + 1, d_model, padding_idx=text_vocab)
            nn.init.normal_(self.word.weight, std=0.02)
            self.text_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            self.text_vocab = text_vocab
        else:
            self.task = nn.Embedding(n_tasks, d_model)
            nn.init.normal_(self.task.weight, std=0.02)
"""),
    ("""        tok = self.task(task_ids.to(dev)).unsqueeze(2)            # (B, T, 1, D)
""", """        if self.text:   # patched: mean of the instruction's word embeddings -> MLP
            ids = self.text_ids[task_ids.to(dev)]                                     # (B, T, L)
            keep = (ids != self.text_vocab).float().unsqueeze(-1)
            mean = (self.word(ids) * keep).sum(-2) / keep.sum(-2).clamp_min(1.0)
            tok = self.text_mlp(mean).unsqueeze(2)                                   # (B, T, 1, D)
        else:
            tok = self.task(task_ids.to(dev)).unsqueeze(2)            # (B, T, 1, D)
"""),
    ("""                 reward_bin_high: float = 1.0, value_bins: int = 255,
                 value_bin_low: float = -1.0, value_bin_high: float = 1.0):
        super().__init__()
        d_in = n_agent * d_model
        self.d_model = d_model
        self.n_agent = n_agent
        self.d_in = d_in
        self.mtp = mtp
        self.tokens = AgentTokenEncoder(d_model, n_agent=n_agent, n_tasks=n_tasks)
""", """                 reward_bin_high: float = 1.0, value_bins: int = 255,
                 value_bin_low: float = -1.0, value_bin_high: float = 1.0,
                 task_text_ids=None, text_vocab: int = 0, text_words=None):
        super().__init__()
        d_in = n_agent * d_model
        self.d_model = d_model
        self.n_agent = n_agent
        self.d_in = d_in
        self.mtp = mtp
        self.tokens = AgentTokenEncoder(d_model, n_agent=n_agent, n_tasks=n_tasks,
                                        task_text_ids=task_text_ids, text_vocab=text_vocab)
        if task_text_ids is not None:   # patched: instruction reconstruction head on h_t
            self.instr_len = len(task_text_ids[0])
            self.instr_classes = text_vocab + 1
            self.instr = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(),
                                       nn.Linear(d_hidden, self.instr_len * self.instr_classes))
"""),
    ("""def build_agent_heads(cfg: Dict) -> AgentHeads:""", """def instr_logits(heads, h):
    \"\"\"(..., d_in) -> (..., L, V + 1) logits of the instruction tokens (patched).\"\"\"
    return heads.instr(h).view(*h.shape[:-1], heads.instr_len, heads.instr_classes)


def instr_loss(heads, h, task):
    \"\"\"h (B, T, d_in), task (B,) -> mean cross-entropy of the instruction tokens over slots and positions.\"\"\"
    tgt = heads.tokens.text_ids[task.long()]                                   # (B, L)
    lg = instr_logits(heads, h)                                                # (B, T, L, C)
    tgt = tgt[:, None].expand(-1, lg.shape[1], -1)
    return torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tgt.reshape(-1))


@torch.no_grad()
def instr_exact(heads, h, task):
    \"\"\"Share of slots whose argmax token sequence is exactly the instruction.\"\"\"
    tgt = heads.tokens.text_ids[task.long()][:, None]
    return float((instr_logits(heads, h).argmax(-1) == tgt).all(-1).float().mean())


def build_agent_heads(cfg: Dict) -> AgentHeads:"""),
])

patch(root / "train" / "train_heads.py", [
    ("""from dreamer4.models.agent_heads import (agent_tokens_are_write_only, build_agent_heads,""",
     """from dreamer4.models.agent_heads import instr_exact, instr_loss   # patched
from dreamer4.models.agent_heads import (agent_tokens_are_write_only, build_agent_heads,"""),
    ("""    ap.add_argument("--n_tasks", type=int, default=1,""", """    ap.add_argument("--text_tokenizer", default="", help="patched: dir with tokenizer.json; the task enters as the "
                    "instruction text (dataset manifest task_names) and an instr head must reconstruct it")
    ap.add_argument("--instr_weight", type=float, default=1.0)
    ap.add_argument("--n_tasks", type=int, default=1,"""),
    ("""    heads = build_agent_heads(agent_cfg).to(dev)
""", """    if cfg.text_tokenizer:   # patched: instruction text -> compact token ids
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(str(Path(cfg.text_tokenizer) / "tokenizer.json"))
        names = ds.manifest["task_names"]
        raw = [tk.encode(n, add_special_tokens=False).ids for n in names]
        vocab = sorted({i for r in raw for i in r})
        L = max(map(len, raw))
        agent_cfg["task_text_ids"] = [[vocab.index(i) for i in r] + [len(vocab)] * (L - len(r)) for r in raw]
        agent_cfg["text_vocab"] = len(vocab)
        agent_cfg["text_words"] = [tk.id_to_token(i) for i in vocab]
        print(f"task as text: {names} -> {L} tokens over {len(vocab)} words", flush=True)
    heads = build_agent_heads(agent_cfg).to(dev)
"""),
    ("""                             "bootstrap": cfg.dyn_weight * 0.5, "value": cfg.value_weight},""",
     """                             "bootstrap": cfg.dyn_weight * 0.5, "value": cfg.value_weight,
                             **({"instr": cfg.instr_weight} if cfg.text_tokenizer else {})},"""),
    ("""            if cfg.dyn_weight > 0:
                # The video-prediction loss""", """            if cfg.text_tokenizer:   # patched
                terms["instr"] = instr_loss(heads, h.float(), clip["task"])
            if cfg.dyn_weight > 0:
                # The video-prediction loss"""),
    ("""            n_ev = cfg.final_eval_n if s == cfg.steps else cfg.eval_n""", """            n_ev = cfg.final_eval_n if s == cfg.steps else cfg.eval_n
            if cfg.text_tokenizer:   # patched: held-out instruction reconstruction
                heads.eval(); world.dyn.eval()
                with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=bool(cfg.amp)):
                    vc = sample_clips(episodes, val_ids, cfg.batch, T, rng, dev)
                    m["instr_exact"] = instr_exact(heads, agent_forward(world, heads, vc, **fwd).float(), vc["task"])
                heads.train(); world.dyn.train()
                print(f"instr exact heldout {m['instr_exact']:.4f}", flush=True)"""),
])
