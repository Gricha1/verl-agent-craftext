"""Stage 2 through veRL (safe_rl_nlp.md, rules for agents 18-21): planning in Dreamer 4 + GRPO, in one veRL run.

The policy is the Stage-1 SLM over Dreamer 4 latents (slm_vision/materialize_d4_checkpoint.py). At every real state of
every lane the vector worker
  1. asks vLLM for n plans of the same prompt (pre text | latent window + readouts as markers | task), the latent window
     coming from dreamer_service.py (phase4_world_model), which encodes the real frames with the Dreamer 4 tokenizer;
  2. has dreamer_service score every plan with the Stage-1 formula (dream + reward / continue heads + critic V);
  3. plays the first action of the best plan in the real game (re-planning at every step, as in Stage 1).
Every ``train_every``-th state its n plans become one GRPO group: n PPO items with the dream score as the reward and the
uid ``<prompt uid>-t<turn>`` (veRL's compute_grpo_outcome_advantage groups by uid). veRL owns the optimiser, the lr
schedule, the checkpoints and their resume; the training environments persist across rollouts (episodes are long), a lane
whose episode ends starts a new world.

Validation plays fresh worlds with the same planner (n plans, best by the dream) to the end or ``val_max_steps``.

Config (``+actor_rollout_ref.rollout.agent.craftext.<key>``): game (crafter | wood), dreamer_socket, n_plans, plan_len,
train_every, steps_per_rollout, episode_cap, val_max_steps, seed, validation_seed, visual_tokens, settings (wood),
teacher_root (wood: caged_craftext with Craftax).
"""
from __future__ import annotations

import asyncio
import json
import sys
from copy import deepcopy
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import ray
import torch
import transfer_queue as tq

from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ
from verl.utils.model import compute_position_id_with_mask
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

from .agent_loop import CraftextAgentLoopOutput
from .vector_manager import CraftextVectorAgentLoopWorker, _value_at
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "phase4_world_model"))
from slm_plan_text import NAMES, parse_plan, prompt_parts  # noqa: E402


class DreamerClient:
    """The agent loop's side of phase4_world_model/dreamer_service.py. Blocking; called through asyncio.to_thread."""

    def __init__(self, socket: str) -> None:
        self.connection = Client(socket, family="AF_UNIX", authkey=b"dreamer")

    def _call(self, *message: Any) -> Any:
        self.connection.send(message)
        reply = self.connection.recv()
        if isinstance(reply, Exception):
            raise RuntimeError("dreamer_service failed") from reply
        return reply

    def first(self, frames, tasks, learn=False):
        return self._call("first", list(frames), list(tasks), bool(learn))

    def reset(self, lanes, frames, tasks):
        return self._call("reset", list(lanes), list(frames), list(tasks))

    def step(self, lanes, actions, frames, rewards=None, ends=None):
        return self._call("step", list(lanes), [int(a) for a in actions], list(frames),
                          [float(r) for r in (rewards or [0.0] * len(lanes))], list(ends or [""] * len(lanes)))

    def update(self):
        return self._call("update")

    def refresh(self):
        return self._call("refresh")

    def score(self, lanes, plans):
        return self._call("score", list(lanes), [list(p) for p in plans])

    def close(self) -> None:
        self.connection.close()


class CrafterVectorEnv:
    """Stock Crafter worlds (the ZeroHero protocol of eval_slm_d4_env.py --game crafter): 64x64 frames cropped to 63x63,
    one world seed per lane; ``reset_lane`` starts a new world with the next seed."""

    def __init__(self, num_envs: int, seed: int, episode_cap: int, task: str):
        import crafter
        self.crafter, self.L, self.next_seed, self.cap, self.task = crafter, num_envs, seed, episode_cap, task
        self.envs, self.frames = [None] * num_envs, [None] * num_envs
        self.steps, self.unlocked, self.ret, self.died = [0] * num_envs, [set() for _ in range(num_envs)], [0.0] * num_envs, [False] * num_envs
        for lane in range(num_envs):
            self.reset_lane(lane)

    def reset_lane(self, lane):
        e = self.crafter.Env(size=(64, 64), seed=self.next_seed, length=10 ** 9)
        self.next_seed += 1
        self.envs[lane] = e
        self.frames[lane] = np.ascontiguousarray(e.reset()[:63, :63])
        self.steps[lane], self.unlocked[lane], self.ret[lane], self.died[lane] = 0, set(), 0.0, False
        return self.frames[lane]

    def tasks(self):
        return [self.task] * self.L

    def step(self, lanes, actions):
        out = []
        for lane, a in zip(lanes, actions):
            o, r, d, info = self.envs[lane].step(int(a))
            self.frames[lane] = np.ascontiguousarray(o[:63, :63])
            self.steps[lane] += 1; self.ret[lane] += float(r)
            self.unlocked[lane] |= {k for k, v in info["achievements"].items() if v > 0}
            died = bool(d) and info["inventory"]["health"] <= 0
            self.died[lane] |= died
            cap = self.steps[lane] >= self.cap
            out.append((float(r), bool(d) or cap, False, "terminal" if d else ("truncated" if cap else "")))
        return out

    def episode_stats(self, lane):
        return dict(achievements=float(len(self.unlocked[lane])), ret=self.ret[lane], steps=self.steps[lane],
                    success=0.0, died=float(self.died[lane]))


class CraftaxWoodVectorEnv:
    """The wood teacher env of eval_slm_d4_env.py: InstructionWrapper(Craftax-Classic-Pixels-v1, settings), lanes cycling
    over the instructions, frames round(obs * 255)."""

    def __init__(self, num_envs: int, seed: int, episode_cap: int, settings: str, teacher_root: str):
        import os
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        sys.path[:0] = [teacher_root, f"{teacher_root}/Craftax"]
        import jax
        import jax.numpy as jnp
        from craftax.craftax_env import make_craftax_env_from_name
        from craftext.environment.craftext_wrapper import InstructionWrapper
        self.jax, self.jnp = jax, jnp
        base = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", False)
        self.params = base.default_params
        self.env = InstructionWrapper(base, settings)
        self.instrs = list(self.env.scenario_handler.scenario_data.instructions_list)
        self.L, self.next_seed, self.cap = num_envs, seed, episode_cap
        env, params = self.env, self.params

        def step_one(key, st, act):
            key, k = jax.random.split(key)
            obs, st, r, done, _ = env.step(k, st, act, params)
            return key, obs, st, r, done, st.instruction_done

        self.step_v = jax.jit(jax.vmap(step_one))
        self.reset_one = jax.jit(lambda k, i: env.reset(k, params, instruction_idx=i))
        self.task_idx = [lane % len(self.instrs) for lane in range(num_envs)]
        self.keys, self.states, self.frames = [None] * num_envs, [None] * num_envs, [None] * num_envs
        self.steps, self.ret, self.success = [0] * num_envs, [0.0] * num_envs, [False] * num_envs
        for lane in range(num_envs):
            self.reset_lane(lane)

    def reset_lane(self, lane):
        k = self.jax.random.PRNGKey(self.next_seed); self.next_seed += 1
        obs, st = self.reset_one(k, self.jnp.int32(self.task_idx[lane]))
        self.keys[lane], self.states[lane] = k, st
        self.frames[lane] = np.round(np.asarray(obs) * 255).astype(np.uint8)
        self.steps[lane], self.ret[lane], self.success[lane] = 0, 0.0, False
        return self.frames[lane]

    def tasks(self):
        return [self.instrs[i] for i in self.task_idx]

    def step(self, lanes, actions):
        jax = self.jax
        keys = self.jnp.stack([self.keys[l] for l in lanes])
        sts = jax.tree_util.tree_map(lambda *x: self.jnp.stack(x), *[self.states[l] for l in lanes])
        keys, obs, sts, r, done, ok = self.step_v(keys, sts, self.jnp.asarray(np.asarray(actions, np.int32)))
        obs, r, done, ok = np.asarray(obs), np.asarray(r), np.asarray(done), np.asarray(ok)
        out = []
        for i, l in enumerate(lanes):
            self.keys[l] = keys[i]; self.states[l] = jax.tree_util.tree_map(lambda x: x[i], sts)
            self.frames[l] = np.round(obs[i] * 255).astype(np.uint8)
            self.steps[l] += 1; self.ret[l] += float(r[i]); self.success[l] |= bool(ok[i])
            cap = self.steps[l] >= self.cap
            end = ("success" if ok[i] else "terminal") if done[i] else ("truncated" if cap else "")
            out.append((float(r[i]), bool(done[i]) or cap, bool(ok[i]), end))
        return out

    def episode_stats(self, lane):
        return dict(achievements=0.0, ret=self.ret[lane], steps=self.steps[lane], success=float(self.success[lane]), died=0.0)


class CraftextPlanVectorAgentLoopWorker(CraftextVectorAgentLoopWorker):
    """One Ray actor: the vector env(s), the dreamer_service connection(s), n plans per state, GRPO groups."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        cfg = self.config.get("craftext", {})
        self.game = str(cfg.get("game", "crafter"))
        self.dreamer_socket = str(cfg.get("dreamer_socket"))
        self.n_plans = int(cfg.get("n_plans", 8))
        self.plan_len = int(cfg.get("plan_len", 16))
        self.train_every = int(cfg.get("train_every", 4))
        self.steps_per_rollout = int(cfg.get("steps_per_rollout", 32))
        self.episode_cap = int(cfg.get("episode_cap", 1000))
        self.val_max_steps = int(cfg.get("val_max_steps", 1000))
        self.settings = str(cfg.get("settings", "achievements_wood"))
        self.teacher_root = str(cfg.get("teacher_root", ""))
        n_vis = int(cfg.get("visual_tokens", 320))
        self.visual_marker_ids = [int(self.tokenizer.convert_tokens_to_ids(f"<|crafter_visual_{i}|>")) for i in range(n_vis)]
        if any(t is None or t < 0 or t == self.tokenizer.unk_token_id for t in self.visual_marker_ids):
            raise RuntimeError("the Dreamer-latent SLM tokenizer lacks its marker tokens")
        self.n_chunks = int(cfg.get("n_chunks", 4))
        self._train = None   # persistent training env + dreamer connections + latent windows
        self.turn_counter = 0

    def _make_env(self, num_envs, seed, cap):
        if self.game == "crafter":
            return CrafterVectorEnv(num_envs, seed, cap, "Unlock as many achievements as possible.")
        return CraftaxWoodVectorEnv(num_envs, seed, cap, self.settings, self.teacher_root)

    def _prompt(self, task):
        pre, post = prompt_parts(task, self.plan_len)
        enc = lambda s: self.tokenizer.encode(s, add_special_tokens=False)
        return enc(pre) + self.visual_marker_ids + enc(post)

    async def _plans(self, lanes, windows, tasks, validate):
        """n plans per lane from vLLM: (prompt_ids, [result]*n) per lane."""
        params = deepcopy(self._sampling_params(validate=False))   # the planner samples n candidates also in validation
        params["max_tokens"] = min(self.per_turn_max_tokens, self.codec.rollout_config.response_length)
        params["detokenize"] = False   # plans are decoded here; vLLM's frontend detokenizer is a single Python thread
        jobs, keys = [], []
        prompts = {}
        for lane in lanes:
            prompts[lane] = self._prompt(tasks[lane])
            for n in range(self.n_plans):
                keys.append((lane, n))
                jobs.append(self.llm_client.generate(request_id=f"plan-{uuid4().hex}", prompt_ids=prompts[lane],
                                                     sampling_params=params,
                                                     image_data=[windows[lane][:, None, None]], priority=lane))
        results = await asyncio.gather(*jobs)
        out = {lane: [None] * self.n_plans for lane in lanes}
        for (lane, n), res in zip(keys, results):
            out[lane][n] = res
        return prompts, out

    def _plan_of(self, result):
        text = self.tokenizer.decode(result.token_ids, skip_special_tokens=True)
        plan = [x for x in (parse_plan(text) or []) if x is not None]
        return (plan or [0]), bool(plan), text

    async def _turn(self, env, client, windows, lanes, tasks, validate, make_items, uid_of, off=0, timing=None):
        """One real step of the given lanes (global ids; ``client`` serves lanes off..off+n-1 as 0..n-1); returns (items,
        per-lane (reward, done, ok, end), actions). ``timing`` accumulates seconds per phase."""
        import time as _time
        t0 = _time.time()
        prompts, results = await self._plans(lanes, windows, tasks, validate)
        t1 = _time.time()
        rows_lane, rows_plan, parsed = [], [], []
        for lane in lanes:
            for res in results[lane]:
                plan, ok, _ = self._plan_of(res)
                rows_lane.append(lane - off); rows_plan.append(plan); parsed.append(ok)
        scores = await asyncio.to_thread(client.score, rows_lane, rows_plan)
        t2 = _time.time()
        items, acts = [], []
        for j, lane in enumerate(lanes):
            sc = [float(s) for s in scores[j * self.n_plans:(j + 1) * self.n_plans]]
            best = int(np.argmax(sc))
            acts.append(rows_plan[j * self.n_plans + best][0])
            if make_items:
                for n, res in enumerate(results[lane]):
                    rid = list(res.token_ids)
                    if not rid:
                        continue
                    items.append(dict(lane=lane, uid=uid_of(lane), plan=n, prompt_ids=prompts[lane], response_ids=rid,
                                      logprobs=list(res.log_probs) if res.log_probs else None, reward=sc[n],
                                      window=windows[lane], parsed=parsed[j * self.n_plans + n], chosen=float(n == best)))
        outs = await asyncio.to_thread(env.step, lanes, acts)
        t3 = _time.time()
        new_w = await asyncio.to_thread(client.step, [l - off for l in lanes], acts, [env.frames[l] for l in lanes],
                                        [o[0] for o in outs], [o[3] for o in outs])
        for i, l in enumerate(lanes):
            windows[l] = new_w[i]
        if timing is not None:
            for k, v in (("generate", t1 - t0), ("dream_score", t2 - t1), ("env", t3 - t2), ("encode", _time.time() - t3)):
                timing[k] = timing.get(k, 0.0) + v
        return items, outs, acts

    def _chunks(self, L):
        """Contiguous lane groups that run as independent coroutines, so vLLM generates for one group while dreamer_service
        scores another (n_chunks connections to the service, one per group)."""
        n = max(1, min(self.n_chunks, L))
        size = (L + n - 1) // n
        return [list(range(s, min(L, s + size))) for s in range(0, L, size)]

    def _output(self, it, prompt, global_steps, extra):
        mask = [1] * len(it["response_ids"])
        out = CraftextAgentLoopOutput(
            prompt_ids=it["prompt_ids"], response_ids=it["response_ids"], response_mask=mask,
            response_logprobs=it["logprobs"], reward_score=it["reward"], num_turns=1, metrics=AgentLoopMetrics(),
            extra_fields={"craftext_turn_end_indices": [len(mask) - 1], "craftext_turn_rewards": [it["reward"]],
                          "craftext_action_valid": float(it["parsed"]), "craftext_action_invalid": float(not it["parsed"]),
                          "craftext_plan_chosen": it["chosen"], **extra})
        field = out.as_dict()
        field.update(prompt)
        field.pop("multi_modal_data", None)
        field["uid"] = it["uid"]                                   # GRPO group: the n plans of one state
        prompt_ids = torch.tensor(it["prompt_ids"], dtype=torch.int64)
        response_ids = torch.tensor(it["response_ids"], dtype=torch.int64)
        input_ids = torch.cat([prompt_ids, response_ids], 0)
        field["loss_mask"] = field["response_mask"]
        field["input_ids"] = input_ids
        field["position_ids"] = compute_position_id_with_mask(torch.ones_like(input_ids).unsqueeze(0)).squeeze(0)
        field["multi_modal_inputs"] = {"pixel_values": torch.from_numpy(np.asarray(it["window"], np.float32).copy()).unsqueeze(0)}
        tag = {"status": "success", "prompt_len": prompt_ids.size(0), "response_len": response_ids.size(0),
               "seq_len": input_ids.size(0), "global_steps": global_steps, "min_global_steps": int(global_steps),
               "max_global_steps": int(global_steps)}
        return field, tag

    async def _run_vector_batch(self, batch, *, validate: bool) -> None:
        import time as _time
        prompts = [{k: _value_at(v, lane) for k, v in batch.items()} for lane in range(len(batch))]
        uids = [str(p["uid"]) for p in prompts]
        partition = "val" if validate else "train"
        for uid in uids:
            await tq.async_kv_put(key=uid, partition_id=partition, tag={"status": "running"})
        L = len(batch)
        gs = int(prompts[0]["global_steps"])
        keys, fields, tags = [], [], []
        clients = []
        timing = {}
        t_start = _time.time()

        async def connect(env, tasks, learn):
            """One dreamer_service connection per lane group; the service keeps lane state per connection (local ids)."""
            chunks, windows = [], {}
            for lanes in self._chunks(L):
                c = await asyncio.to_thread(DreamerClient, self.dreamer_socket)
                clients.append(c)
                w = await asyncio.to_thread(c.first, [env.frames[l] for l in lanes], [tasks[l] for l in lanes], learn)
                for i, l in enumerate(lanes):
                    windows[l] = np.asarray(w[i])
                chunks.append(dict(lanes=lanes, off=lanes[0], client=c))
            return chunks, windows

        try:
            if validate:   # fresh worlds, played to the end with the same planner
                env = self._make_env(L, self.validation_seed, self.val_max_steps)
                tasks = env.tasks()
                chunks, windows = await connect(env, tasks, False)
                stats, w_last = {}, {}

                async def play(ch):
                    active, turn = list(ch["lanes"]), 0
                    while active and turn < self.val_max_steps:
                        _, outs, _ = await self._turn(env, ch["client"], windows, active, tasks, True, False, None,
                                                      off=ch["off"], timing=timing)
                        still = []
                        for l, (r, done, ok, _) in zip(active, outs):
                            if done:
                                stats[l] = env.episode_stats(l)
                            else:
                                still.append(l)
                        active = still; turn += 1
                    for l in active:
                        stats[l] = env.episode_stats(l)
                    w = await asyncio.to_thread(ch["client"].reset, [l - ch["off"] for l in ch["lanes"]],
                                                [env.frames[l] for l in ch["lanes"]], [tasks[l] for l in ch["lanes"]])
                    for i, l in enumerate(ch["lanes"]):
                        w_last[l] = np.asarray(w[i])

                await asyncio.gather(*[play(ch) for ch in chunks])
                # one item per lane carries its episode (the trainer reads reward_extra_info of validation items)
                prompts_, results = await self._plans(list(range(L)), w_last, tasks, True)
                for l in range(L):
                    st = stats[l]
                    res = results[l][0]
                    it = dict(uid=uids[l], plan=0, prompt_ids=prompts_[l], response_ids=list(res.token_ids) or [2],
                              logprobs=None, reward=float(st["achievements"] if self.game == "crafter" else st["success"]),
                              window=w_last[l], parsed=1.0, chosen=1.0)
                    extra = {"craftext_episode_final": True, "reward_extra_info": {
                        "craftext_episode_success": st["success"], "craftext_episode_return": st["ret"],
                        "craftext_episode_steps": float(st["steps"]), "craftext_episode_achievements": st["achievements"],
                        "craftext_episode_died": st["died"]}}
                    f, t = self._output(it, prompts[l], gs, extra)
                    keys.append(f"{uids[l]}_0_0"); fields.append(f); tags.append(t)
            else:
                if self._train is None or self._train["L"] != L:
                    env = self._make_env(L, self.seed, self.episode_cap)
                    tasks = env.tasks()
                    chunks, windows = await connect(env, tasks, True)
                    self._train = dict(L=L, env=env, chunks=chunks, tasks=tasks, windows=windows)
                tr = self._train
                env, chunks, tasks, windows = tr["env"], tr["chunks"], tr["tasks"], tr["windows"]
                session = [0] * L
                last_field = {}   # lane -> index of its newest PPO item (carries a finished episode's figures)
                base = self.turn_counter

                async def roll(ch):
                    for s in range(self.steps_per_rollout):
                        turn = base + s
                        make = turn % self.train_every == 0
                        items, outs, _ = await self._turn(env, ch["client"], windows, ch["lanes"], tasks, False, make,
                                                          lambda lane, turn=turn: f"{uids[lane]}-t{turn}",
                                                          off=ch["off"], timing=timing)
                        for it in items:
                            f, t = self._output(it, prompts[it["lane"]], gs, {})
                            keys.append(f"{uids[it['lane']]}_{session[it['lane']]}_0"); session[it["lane"]] += 1
                            last_field[it["lane"]] = len(fields)
                            fields.append(f); tags.append(t)
                        ended = [l for l, (r, done, ok, _) in zip(ch["lanes"], outs) if done]
                        for l in ended:
                            st = env.episode_stats(l)
                            if l in last_field:   # train/success_rate, episode return / length, achievements (metrics.py)
                                fields[last_field[l]]["extra_fields"].update(
                                    craftext_episode_final=True, craftext_episode_success=st["success"],
                                    craftext_episode_return=st["ret"], craftext_episode_steps=float(st["steps"]),
                                    craftext_episode_achievements=st["achievements"])
                            env.reset_lane(l)
                        if ended:
                            now = env.tasks()
                            for l in ended:
                                tasks[l] = now[l]
                            w = await asyncio.to_thread(ch["client"].reset, [l - ch["off"] for l in ended],
                                                        [env.frames[l] for l in ended], [tasks[l] for l in ended])
                            for i, l in enumerate(ended):
                                windows[l] = np.asarray(w[i])

                await asyncio.gather(*[roll(ch) for ch in chunks])
                self.turn_counter = base + self.steps_per_rollout
                info = await asyncio.to_thread(chunks[0]["client"].update)   # the critic learns from the real returns
                if info.get("world_updates"):
                    refreshed = await asyncio.gather(*(asyncio.to_thread(ch["client"].refresh) for ch in chunks))
                    for ch, encoded in zip(chunks, refreshed, strict=True):
                        for lane, window in zip(ch["lanes"], encoded, strict=True):
                            windows[lane] = np.asarray(window)
                if "world_updates" in info and fields:
                    fields[0]["extra_fields"].update({f"craftext_{k}": float(v) for k, v in info.items()})
                if info.get("critic_updates") and fields:
                    fields[0]["extra_fields"].update({f"craftext_{k}": float(v) for k, v in info.items()})
            wall = _time.time() - t_start
            if fields:   # seconds per phase summed over lane groups; overlap = their sum / wall time
                fields[0]["extra_fields"].update({f"craftext_time_{k}": float(v) for k, v in timing.items()})
                fields[0]["extra_fields"]["craftext_time_rollout_wall"] = float(wall)
            print(json.dumps(dict(plan_rollout=partition, lanes=L, chunks=len(self._chunks(L)), wall=round(wall, 1),
                                  **{k: round(v, 1) for k, v in timing.items()})), flush=True)
            if not fields:
                raise RuntimeError("plan rollout produced no PPO items")
            await tq.async_kv_batch_put(keys=keys, fields=list_of_dict_to_tensordict(fields), tags=tags, partition_id=partition)
            for uid in uids:
                await tq.async_kv_put(key=uid, partition_id=partition, tag={"status": "finished"})
        except Exception:
            for uid in uids:
                await tq.async_kv_put(key=uid, partition_id=partition, tag={"status": "failure"})
            raise
        finally:
            if validate:
                for c in clients:
                    await asyncio.to_thread(c.close)


class CraftextPlanVectorAgentLoopManager(AgentLoopManagerTQ):
    """V1 manager scheduling one planning vector worker."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.agent_loop_workers_class = ray.remote(CraftextPlanVectorAgentLoopWorker)
