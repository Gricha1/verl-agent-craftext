"""Closed-loop / offline evaluation of a materialised visual-SLM checkpoint (one mode per process).

offline  -- teacher-corpus states (frame + true action history) -> greedy action accuracy
teacher  -- closed loop in the teacher/collector env: InstructionWrapper(Craftax-Classic-Pixels-v1);
            the phase-2 gate before RL
adapter  -- closed loop in the veRL adapter env (CraftextJaxVectorEnv, CMDP wrapper), plus a per-step
            check of the adapter frame against the Craftax pixel observation of the same state

Paths default to this checkout: --repo = vlm_pipeline/phase3_verl (prompt/parser of the veRL plugin),
--teacher-root = caged_craftext, --verl-root = the repository root (agent_system).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ACTIONS = ("NOOP", "LEFT", "RIGHT", "UP", "DOWN", "DO", "SLEEP", "PLACE_STONE", "PLACE_TABLE",
           "PLACE_FURNACE", "PLACE_PLANT", "MAKE_WOOD_PICKAXE", "MAKE_STONE_PICKAXE",
           "MAKE_IRON_PICKAXE", "MAKE_WOOD_SWORD", "MAKE_STONE_SWORD", "MAKE_IRON_SWORD")


class Policy:
    def __init__(self, model_path: str, repo: str, no_history: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        sys.path.insert(0, repo)
        from craftext_integration.slm_vision.prompt import (VISUAL_SLM_ACTION_HISTORY, parse_slm_action,
                                                             vision_image_only_prompt)
        self.torch, self.prompt, self.parse, self.h = torch, vision_image_only_prompt, parse_slm_action, VISUAL_SLM_ACTION_HISTORY
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).cuda().eval()
        self.markers = list(self.model.config.crafter_visual_token_ids)
        self.eos = int(self.model.config.eos_token_id)
        self.no_history = no_history  # prompt protocol vision_image_only_v1: no history line at all

    def act(self, frame01: np.ndarray, task: str, history: list[str]):
        """frame01: [63,63,3] float in [0,1]. Greedy, no KV cache (8 tokens at most)."""
        torch = self.torch
        ids = self.markers + self.tok.encode(self.prompt(task, None if self.no_history else history[-self.h:]), add_special_tokens=False)
        x = torch.tensor([ids], device="cuda")
        pix = torch.from_numpy(np.ascontiguousarray(np.moveaxis(frame01.astype(np.float32), -1, 0)))[None].cuda()
        out, first_probs = [], None
        with torch.no_grad():
            for _ in range(8):
                logits = self.model(input_ids=x, pixel_values=pix).logits[0, -1]
                if first_probs is None:
                    first_probs = torch.softmax(logits.float(), -1)
                nxt = int(logits.argmax())
                if nxt == self.eos:
                    break
                out.append(nxt)
                x = torch.cat([x, torch.tensor([[nxt]], device="cuda")], 1)
        text = self.tok.decode(out, skip_special_tokens=True)
        return self.parse(text).action_name, text


def run_offline(args, pol):
    import pyarrow.parquet as pq
    root = Path(args.corpus)
    rows = []
    for split in ("train", "val"):
        for r in pq.read_table(root / f"{split}.parquet").to_pylist():
            e = r["extra_info"]
            rows.append((e["episode_id"], e["step"], split, e["instruction"], e["teacher_action"],
                         Path(e["observation_path"]).name))
    rows.sort()
    hist, last_ep, stats = [], None, collections.defaultdict(lambda: [0, 0])
    pred_counts = collections.Counter()
    for ep, step, split, instr, label, img in rows:
        if ep != last_ep:
            hist, last_ep = [], ep
        if split == "val" and stats["val"][1] < args.offline_limit:
            with np.load(root / "images" / img) as a:
                frame = a["observation"].astype(np.float32)
            if frame.max() > 1:
                frame = frame / 255.0
            pred, _ = pol.act(frame, instr, hist)
            pred_counts[pred] += 1
            stats["val"][0] += int(pred == label)
            stats["val"][1] += 1
        hist.append(label)
    acc = stats["val"][0] / max(1, stats["val"][1])
    return {"mode": "offline", "val_rows": stats["val"][1], "accuracy": acc, "pred_counts": dict(pred_counts)}


def run_teacher(args, pol):
    T = Path(args.teacher_root)
    sys.path[:0] = [str(T), str(T / "Craftax")]
    import jax
    import jax.numpy as jnp
    from craftax.craftax_env import make_craftax_env_from_name
    from craftext.environment.craftext_wrapper import InstructionWrapper
    import craftax
    base = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", False)
    params = base.default_params
    env = InstructionWrapper(base, args.settings)
    instrs = list(env.scenario_handler.scenario_data.instructions_list)
    step = jax.jit(env.step)
    episodes = []
    for task in range(len(instrs)):
        for seed in args.seeds:
            key = jax.random.PRNGKey(seed)
            obs, state = env.reset(key, params, instruction_idx=task)
            hist, acts, ret, success = [], collections.Counter(), 0.0, False
            for t in range(args.max_steps):
                a, _ = pol.act(np.asarray(obs), instrs[task], hist)
                acts[a] += 1
                key, k = jax.random.split(key)
                obs, state, r, done, _ = step(k, state, jnp.int32(ACTIONS.index(a)), params)
                ret += float(r)
                hist.append(a)
                success = bool(state.instruction_done)
                if bool(done):
                    break
            episodes.append({"task": instrs[task], "seed": seed, "steps": t + 1, "success": success,
                             "return": ret, "top_actions": acts.most_common(4)})
            print(json.dumps(episodes[-1]), flush=True)
    return {"mode": "teacher", "craftax_file": craftax.__file__, "instructions": instrs, "episodes": episodes,
            "success_rate": sum(e["success"] for e in episodes) / len(episodes)}


def run_adapter(args, pol):
    os.environ["CRAFTEXT_LEGACY_ROOT"] = args.verl_root
    sys.path.insert(0, args.repo)
    from craftext_integration.env_adapter import CraftextJaxVectorEnv
    from craftext_integration.actions import ParsedAction
    import jax
    env = CraftextJaxVectorEnv(num_envs=args.lanes, seed=args.seeds[0], max_steps=args.max_steps,
                               config_name=args.settings, use_pixel_obs=True)
    import craftax  # importable only after the adapter put the legacy tree on sys.path
    from agent_system.environments.env_package.caged_craftext import envs as legacy_envs
    inner = env._env
    _, frames, infos = env.reset()
    tasks = [str(i.get("instruction")) for i in infos]
    hist = [[] for _ in range(args.lanes)]
    acts = [collections.Counter() for _ in range(args.lanes)]
    ret = np.zeros(args.lanes)
    active = [True] * args.lanes
    success = [False] * args.lanes
    steps = [0] * args.lanes
    max_diff, frame_checks = 0.0, 0

    def parity(fr):
        nonlocal max_diff, frame_checks
        st = jax.device_get(inner.state)
        for i in range(args.lanes):
            s_i = jax.tree_util.tree_map(lambda x: x[i], st.env_state)
            ref = np.asarray(inner.wrapper.env.get_obs(s_i))  # Craftax pixel obs the teacher sees, [0,1]
            max_diff = max(max_diff, float(np.abs(fr[i].astype(np.float32) / 255.0 - ref).max()))
            frame_checks += 1

    parity(frames)
    for t in range(args.max_steps):
        chosen = []
        for i in range(args.lanes):
            a, _ = pol.act(frames[i].astype(np.float32) / 255.0, tasks[i], hist[i]) if active[i] else ("NOOP", "")
            chosen.append(ParsedAction(action_id=ACTIONS.index(a), action_name=a, reasoning="", valid=True))
        _, frames, rewards, dones, infos = env.step(chosen)
        if t < 20:
            parity(frames)
        for i in range(args.lanes):
            if not active[i]:
                continue
            acts[i][chosen[i].action_name] += 1
            hist[i].append(chosen[i].action_name)
            ret[i] += rewards[i]
            steps[i] = t + 1
            success[i] = success[i] or bool(infos[i].get("instruction_done", False))
            if dones[i]:
                active[i] = False
        if not any(active):
            break
    eps = [{"task": tasks[i], "steps": steps[i], "success": success[i], "return": float(ret[i]),
            "top_actions": acts[i].most_common(4)} for i in range(args.lanes)]
    for e in eps:
        print(json.dumps(e), flush=True)
    return {"mode": "adapter", "craftax_file": craftax.__file__, "legacy_envs": legacy_envs.__file__,
            "adapter_vs_get_obs_max_abs_diff": max_diff, "frame_checks": frame_checks,
            "info_keys": sorted(infos[0].keys()), "episodes": eps,
            "success_rate": sum(success) / args.lanes}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("offline", "teacher", "adapter"), required=True)
    p.add_argument("--model", required=True)
    checkout = Path(__file__).resolve().parents[3]
    p.add_argument("--repo", default=str(checkout / "vlm_pipeline" / "phase3_verl"))
    p.add_argument("--corpus")
    p.add_argument("--teacher-root", default=str(checkout / "caged_craftext"))
    p.add_argument("--verl-root", default=str(checkout))
    p.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=[20261300])
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--lanes", type=int, default=8)
    p.add_argument("--offline-limit", type=int, default=600)
    p.add_argument("--no-history", action="store_true")
    p.add_argument("--settings", default="achievements_wood", help="CrafText settings of the teacher / adapter env")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    t0 = time.time()
    pol = Policy(args.model, args.repo, args.no_history)
    result = {"offline": run_offline, "teacher": run_teacher, "adapter": run_adapter}[args.mode](args, pol)
    result["seconds"] = time.time() - t0
    result["no_history"] = args.no_history
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}), flush=True)


if __name__ == "__main__":
    main()
