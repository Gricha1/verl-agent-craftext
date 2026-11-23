"""Update the Dreamer scorer from real Crafter transitions between veRL rollouts."""

import os
from collections import deque
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F


class OnlineDreamer:
    def __init__(self, service, world):
        from copy import deepcopy

        from dreamer4.models.tokenizer import patchify
        from dreamer4.train.common import LossCombiner
        from dreamer4.train.dynamics_objectives import clean_context_loss

        self.service = service
        self.patchify = patchify
        self.dynamics_loss = clean_context_loss
        self.model_cfg, self.objective = world.model_cfg, world.objective
        self.replay = deque(maxlen=service.a.world_buffer)
        self.version = 0
        self.anchor = deepcopy(service.tok.tok.encoder).eval().requires_grad_(False)
        modules = (service.tok.tok, service.dyn, service.heads)
        for module in modules:
            module.requires_grad_(True)
        self.params = [p for module in modules for p in module.parameters()]
        self.opt = torch.optim.AdamW(self.params, lr=service.a.world_lr)
        # Match the Stage-1 convention of RMS-normalizing different Dreamer
        # objectives. Reconstruction and the coordinate anchor update only the codec.
        self.losses = LossCombiner(
            {"flow": 1., "bootstrap": .5, "reward": 1., "terminal": 1., "value": 1., "clone": 1.},
            normalize=True, device=service.dev,
        )
        self.critic_params = list(service.critics[0].parameters())
        service.critics[0].requires_grad_(True)
        self.critic_opt = torch.optim.AdamW(self.critic_params, lr=service.a.critic_lr)

    def add(self, frames, actions, reward, end, task):
        self.add_batch([(frames, actions, reward, end, task)])

    def add_batch(self, samples):
        # Eight frames give five consecutive four-frame encodings. Keep pixels on
        # host RAM so every replay sample can be encoded under the current tokenizer.
        clips = torch.stack([torch.stack(frames) for frames, *_ in samples]).cpu()
        self.replay.extend((clip.clone(), tuple(actions), reward, end == "terminal", task)
                           for clip, (_, actions, reward, end, task) in zip(clips, samples, strict=True))

    def update(self):
        s = self.service
        if len(self.replay) < s.a.world_batch:
            return {"world_updates": 0, "world_replay": len(self.replay)}
        started = perf_counter()
        totals = {}
        for _ in range(s.a.world_ft_steps):
            picks = torch.randint(len(self.replay), (s.a.world_batch,)).tolist()
            frame_rows, action_rows, rewards, terminal, tasks = zip(*(self.replay[i] for i in picks))
            frames = torch.stack(frame_rows).to(s.dev)
            acts = torch.tensor(action_rows, device=s.dev)
            reward = torch.tensor(rewards, device=s.dev)
            done = torch.tensor(terminal, device=s.dev)
            gi = torch.tensor(tasks, device=s.dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # Train the encoder and decoder on real clips. Anchor the coordinate
                # system for the SLM and stop the dynamics target from chasing itself.
                pixels = frames[:, -4:].permute(0, 1, 4, 2, 3).float() / 255
                patches = self.patchify(pixels.contiguous(), s.tok.patch_size)
                encoded, _ = s.tok.tok.encoder(patches)
                recon = F.mse_loss(s.tok.tok.decoder(encoded).float(), patches.float())
                with torch.no_grad():
                    anchor, _ = self.anchor(patches)
                    z = s.tok.encode_frames(frames)[:, 3:].float()
                latent_anchor = F.mse_loss(encoded.float(), anchor.float())

                # The first token is an action sentinel. Each subsequent action
                # leads into that slot, as in Dreamer scoring and pretraining.
                seq = F.one_hot(acts, 18).float()
                sentinel = F.one_hot(torch.full_like(acts[:, :1], 17), 18).float()
                aligned = torch.cat([sentinel, seq], dim=1)
                old_z, new_z = z[:, :-1], z[:, 1:]
                dyn_terms, _ = self.dynamics_loss(
                    s.dyn, new_z, torch.cat([sentinel, seq[:, 1:]], dim=1),
                    p=3, k_max=s.k_max, K=self.model_cfg["K"], tau_ctx=s.tau_ctx,
                    ctx_noise_min=self.objective.get("ctx_noise_min") or None,
                    ctx_noise_max=self.objective.get("ctx_noise_max") or None,
                    do_sched=float(torch.rand(())) < self.objective.get("sched_sample_prob", 0),
                    bootstrap_frac=float(self.objective.get("bootstrap_frac", 0) or 0),
                )
                prediction = sum(t for t in dyn_terms.values() if t is not None)
                head_task = torch.tensor([s.head_task[s.tasks[t]] for t in tasks], device=s.dev)
                step = torch.full_like(acts, s._log2_int(s.k_max))
                sig = torch.full_like(acts, min(round((1 - s.tau_ctx) * s.k_max), s.k_max))
                tokens = s.heads.tokens(step.shape, task_ids=head_task[:, None].expand_as(step), device=s.dev)

                def features(latents, actions):
                    out = s.dyn(actions, step, sig, (1 - s.tau_ctx) * latents, agent_tokens=tokens)
                    return s.flatten_agent_out(out[-1][:, -1].float())

                before = features(old_z, aligned[:, :4])
                after = features(new_z, torch.cat([sentinel, seq[:, 1:]], dim=1))
                reward_loss = s.heads.reward.loss(s.heads.reward.at(after, 0), reward)
                cont_loss = reward_loss.new_zeros(())
                if s.cont:
                    cont_loss = F.binary_cross_entropy_with_logits(s.heads.cont(after).squeeze(-1).float(), done.float())
                with torch.no_grad():
                    next_v = s._critic_mean(s._critic_out(s.critics[0], new_z, gi))
                    target = reward + s.a.gamma * (~done) * next_v
                value_loss = s.heads.value.loss(s.heads.value.at(before, 0), target)
                clone_loss = F.cross_entropy(s.heads.policy.at(before, 0).float(), acts[:, -1])
                critic = s._critic_out(s.critics[0], old_z, gi)
                critic_loss = s.critics[0].value_head.loss(critic, target) if s.hl else F.mse_loss(critic, target)
                # Separate gradient paths: RL loss belongs to veRL; auxiliary
                # objectives train the scorer only on real transitions.
                world_loss, _ = self.losses(dict(
                    flow=dyn_terms.get("flow"), bootstrap=dyn_terms.get("bootstrap"),
                    reward=reward_loss, terminal=cont_loss if s.cont else None,
                    value=value_loss, clone=clone_loss,
                ))
                loss = recon + latent_anchor + world_loss
            terms = dict(recon=recon, anchor=latent_anchor, dynamics=prediction, reward=reward_loss,
                         terminal=cont_loss, value=value_loss, clone=clone_loss, slm_value=critic_loss)
            if not torch.stack([v.detach() for v in terms.values()]).isfinite().all():
                raise FloatingPointError("non-finite online Dreamer loss")
            self.opt.zero_grad(set_to_none=True)
            self.critic_opt.zero_grad(set_to_none=True)
            loss.backward()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, 1., error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(self.critic_params, 1., error_if_nonfinite=True)
            self.opt.step()
            self.critic_opt.step()
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.) + float(value.detach())
        self.version += 1
        return dict(world_updates=s.a.world_ft_steps, world_replay=len(self.replay),
                    world_seconds=perf_counter() - started,
                    **{f"world_loss_{k}": v / s.a.world_ft_steps for k, v in totals.items()})

    def save(self, directory, step):
        s = self.service
        path = Path(directory) / "dreamer_online.pt"
        tmp = path.with_suffix(".tmp")
        torch.save(dict(tokenizer=s.tok.tok.state_dict(), dynamics=s.dyn.state_dict(), heads=s.heads.state_dict(),
                        critic=s.critics[0].state_dict(), optimizer=self.opt.state_dict(),
                        critic_optimizer=self.critic_opt.state_dict(), loss_normalizer=self.losses.state_dict(),
                        replay=list(self.replay),
                        version=self.version, step=step, rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state(s.dev)), tmp)
        os.replace(tmp, path)
