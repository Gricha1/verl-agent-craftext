"""DreamerV3 (official code, all hyperparameters of the chosen config) trained on stored Crafter episodes.

The same as the wood2200 run (train_wood.py, 2026-11-13) with the data source made a flag: embodied/run/train.py
unchanged except two things. The environment plays the stored episodes one after another in a fixed random order (each
frame once, like an online run that met these frames), and the action written to the replay and fed to the dynamics is
the action from the data, not the policy's sample. The run length is the config's own run.steps (crafter: 1.1e6 env
steps, train_ratio 512, batch 16 x 64) unless the data has fewer frames, then one pass over the data.

Data (build_qwen_crafter64.py or build_wood_v3.py): frames.npy [N, 64, 64, 3] uint8, actions.npy, manifest.json
(starts, lengths), a reward file (reward that led to frame t) and a terminal file (frame t is a death).

  python dreamerv3_offline_train.py --data <dir> --rewards rewards_in --terminals terminal_in [--reward-scale 1] \
      --configs crafter --logdir <dir> --replay.online False --jax.prealloc False
"""
import json, pathlib, sys

root = pathlib.Path("/d/dreamerv3")
sys.path.insert(0, str(root)); sys.path.insert(1, str(root / "dreamerv3"))
import elements, embodied, jax, jax.numpy as jnp, numpy as np, ruamel.yaml as yaml
from dreamerv3 import main as v3main


def pop_flag(argv, name, default=None):
    if name not in argv:
        return default
    i = argv.index(name)
    value = argv[i + 1]
    del argv[i:i + 2]
    return value


argv = sys.argv[1:]
data_dir = pathlib.Path(pop_flag(argv, "--data"))
reward_file = pop_flag(argv, "--rewards", "rewards_in")
terminal_file = pop_flag(argv, "--terminals", "terminal_in")
reward_scale = float(pop_flag(argv, "--reward-scale", "1"))
configs = yaml.YAML(typ="safe").load((root / "dreamerv3" / "configs.yaml").read_text())
parsed, other = elements.Flags(configs=["defaults"]).parse_known(argv)
config = elements.Config(configs["defaults"])
for name in parsed.configs:
    config = config.update(configs[name])
config = elements.Flags(config).parse(other)

frames = np.load(data_dir / "frames.npy", mmap_mode="r")
actions = np.load(data_dir / "actions.npy")
rewards = reward_scale * np.load(data_dir / f"{reward_file}.npy").astype(np.float32)
terminals = np.load(data_dir / f"{terminal_file}.npy")
manifest = json.loads((data_dir / "manifest.json").read_text())
order = np.random.default_rng(config.seed).permutation(len(manifest["starts"]))   # episodes in a fixed random order
TOTAL = int(sum(manifest["lengths"]))
STEPS = int(min(TOTAL, config.run.steps))
config = config.update({"run.steps": STEPS})
logdir = elements.Path(config.logdir); logdir.mkdir(); config.save(logdir / "config.yaml")
print("data", data_dir, "episodes", len(order), "frames", TOTAL, "run.steps", STEPS, "train_ratio", config.run.train_ratio, flush=True)


class StoredEnv(embodied.Env):
    """Plays the stored episodes in order; obs and spaces exactly as embodied.envs.crafter.Crafter."""

    def __init__(self):
        self.episode, self.t, self.done = -1, 0, True
        self.next_action = 0

    @property
    def obs_space(self):
        return {"image": elements.Space(np.uint8, (64, 64, 3)), "reward": elements.Space(np.float32),
                "is_first": elements.Space(bool), "is_last": elements.Space(bool), "is_terminal": elements.Space(bool),
                "log/reward": elements.Space(np.float32)}

    @property
    def act_space(self):
        return {"action": elements.Space(np.int32, (), 0, 17), "reset": elements.Space(bool)}

    def step(self, action):
        if action["reset"] or self.done:
            self.episode = (self.episode + 1) % len(order)
            self.t, self.done = 0, False
        else:
            self.t += 1
        i = order[self.episode]
        first, length = manifest["starts"][i], manifest["lengths"][i]
        g = first + self.t
        last = self.t == length - 1
        self.done = last
        self.next_action = int(actions[g])
        return {"image": np.asarray(frames[g]), "reward": np.float32(rewards[g]), "is_first": self.t == 0, "is_last": last,
                "is_terminal": bool(terminals[g]), "log/reward": np.float32(rewards[g])}


env = StoredEnv()
agent = v3main.make_agent(config)
replay = v3main.make_replay(config, "replay")
logger = v3main.make_logger(config)
args = elements.Config(**config.run, replica=config.replica, replicas=config.replicas, logdir=config.logdir,
                       batch_size=config.batch_size, batch_length=config.batch_length, report_length=config.report_length,
                       consec_train=config.consec_train, consec_report=config.consec_report, replay_context=config.replay_context)

step = logger.step
train_agg, train_fps = elements.Agg(), elements.FPS()
batch_steps = args.batch_size * args.batch_length
should_train = elements.when.Ratio(args.train_ratio / batch_steps)
should_log, should_report = embodied.LocalClock(args.log_every), embodied.LocalClock(args.report_every)
should_save = embodied.LocalClock(args.save_every)

driver = embodied.Driver([lambda: env], parallel=False)
driver.on_step(lambda tran, _: step.increment())
driver.on_step(replay.add)
stream_train = iter(agent.stream(v3main.make_stream(config, replay, "train")))
stream_report = iter(agent.stream(v3main.make_stream(config, replay, "report")))
carry_train = [agent.init_train(args.batch_size)]
carry_report = agent.init_report(args.batch_size)


def trainfn(tran, worker):
    if len(replay) < args.batch_size * args.batch_length:
        return
    for _ in range(should_train(step)):
        batch = next(stream_train)
        carry_train[0], outs, mets = agent.train(carry_train[0], batch)
        train_fps.step(batch_steps)
        if "replay" in outs:
            replay.update(outs["replay"])
        train_agg.add(mets, prefix="train")


driver.on_step(trainfn)
cp = elements.Checkpoint(logdir / "ckpt")
cp.step = step
cp.agent = agent
cp.load_or_save()


def policy(carry, obs):
    """The agent's own policy step (latents, replay context), then the stored action in place of its sample."""
    carry, acts, outs = agent.policy(carry, obs, mode="train")
    a = np.array([env.next_action], np.int32)
    acts = {**acts, "action": a}
    enc, dyn, dec, prev = carry
    old = prev["action"][0]
    carry = (enc, dyn, dec, {**prev, "action": [jax.device_put(a[0].astype(old.dtype), old.sharding)]})  # explicit transfer: the agent forbids implicit ones
    return carry, acts, outs


milestones = [int(STEPS * q) for q in (0.25, 0.5, 0.75)]   # extra copies of the checkpoint for evaluation over training


def snapshot():
    import shutil
    cp.save()
    latest = sorted(p for p in (logdir / "ckpt").glob("*") if p.is_dir())[-1]
    shutil.copytree(str(latest), str(logdir / f"snap_{int(step):07d}"))


print("Start training loop", flush=True)
driver.reset(agent.init_policy)
while step < args.steps:
    driver(policy, steps=10)
    if should_report(step) and len(replay):
        agg = elements.Agg()
        for _ in range(args.consec_report * args.report_batches):
            carry_report, mets = agent.report(carry_report, next(stream_report))
            agg.add(mets)
        logger.add(agg.result(), prefix="report")
    if should_log(step):
        logger.add(train_agg.result()); logger.add(replay.stats(), prefix="replay")
        logger.add({"fps/train": train_fps.result()}); logger.write()
    if milestones and step >= milestones[0]:
        milestones.pop(0); snapshot()
    elif should_save(step):
        cp.save()
snapshot()
logger.close()
print("TRAIN_DONE", int(step), flush=True)
