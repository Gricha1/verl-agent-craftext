import argparse
import os
import sys

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import time

from typing import Optional, Any, Tuple

from flax.linen.initializers import (
    constant, 
    orthogonal,
)

import wandb
from flax.training import (
    orbax_utils,
)
import flax
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

from typing import NamedTuple, Dict
import distrax
import functools

from wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)

from logz.batch_logging import create_log_dict, batch_log

from craftext.environment.scenarious.manager import create_scenarios_with_dataset
from craftext.environment.encoders.craftext_distilbert_model_encoder import make_encoder
from craftax.craftax_env import make_craftax_env_from_name
from craftext.environment.craftext_wrapper import InstructionWrapper
from rnn_network import ScannedRNN, ActorCriticTextVisualRNN
from analysis.inference_rnn import Experiment, ExperimentArgs

@flax.struct.dataclass
class TransitionScheme:
    done        : jax.Array
    action      : jax.Array
    value       : jax.Array
    reward      : jax.Array
    log_prob    : jax.Array
    obs         : jax.Array
    info        : jax.Array
    instruction : jax.Array


def make_train(config, network_params):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # Create environment
    env = make_craftax_env_from_name(
        config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"]
    )
    env_params = env.default_params
    
    if config["USE_PLANS"]:
        encoder = make_encoder(n_splits=5)
        scenarious_loader = create_scenarios_with_dataset(True)
        env = InstructionWrapper(env, config["CRAFTEXT_SETTINGS"], 
                                 encode_model_class=encoder, 
                                 scenario_handler_class=scenarious_loader)
    else:
        encoder = make_encoder(n_splits=config["EXPAND_EMB"])
        env = InstructionWrapper(env, config["CRAFTEXT_SETTINGS"], encode_model_class=encoder)
    # Wrap with some extra logging
    env = LogWrapper(env)

    # Wrap with a batcher, maybe using optimistic resets
    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
    else:
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCriticTextVisualRNN(env.action_space(env_params).n, config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["LAYER_SIZE"]
        )

        
        # encoded_input_expanded = jnp.expand_dims(env.encoded_instruction, axis=1)
        # print("encoded_input_expanded.shape", encoded_input_expanded.shape)
        # encoded_input_tiled = jnp.tile(encoded_input_expanded, (1,  config["NUM_ENVS"], 1))
        # print("encoded_input_tiled.shape", encoded_input_tiled.shape)
        # print(config["NUM_ENVS"])
        encoded = env.encoded_instruction           # (768,)
        encoded = jnp.expand_dims(encoded, axis=0)  # (1, 768)
        encoded = jnp.expand_dims(encoded, axis=1)  # (1, 1, 768)
        encoded_input_tiled = jnp.tile(encoded,
                                (1, config["NUM_ENVS"], 1))
        network_params_alt = network.init(_rng, init_hstate, init_x, encoded_input_tiled)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            
        if network_params is None:
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params_alt,
                tx=tx,
            )
        else:
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
            )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    hstate,
                    rng,
                    update_step,
                ) = runner_state
                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                print("-->", env_state.env_state.instruction.shape)
                print("->", last_obs[np.newaxis, :].shape)
                print("->", ac_in[0].shape)
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in, env_state.env_state.instruction[np.newaxis, :])
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = env.step(
                    _rng, env_state, action, env_params
                )
                transition = TransitionScheme(
                    last_done, action, value, reward, log_prob, last_obs, info,  instruction=env_state.env_state.instruction
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    done,
                    hstate,
                    rng,
                    update_step,
                )
                return runner_state, transition

            initial_hstate = runner_state[-3]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                rng,
                update_step,
            ) = runner_state
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in, env_state.env_state.instruction[np.newaxis, :])
            last_val = last_val.squeeze(0)

            def _calculate_gae(traj_batch, last_val, last_done):
                def _get_advantages(carry, transition):
                    gae, next_value, next_done = carry
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = (
                        reward + config["GAMMA"] * next_value * (1 - next_done) - value
                    )
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
                    )
                    return (gae, value, done), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val, last_done),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params, init_hstate[0], (traj_batch.obs, traj_batch.done),traj_batch.instruction
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (init_hstate, traj_batch, advantages, targets)

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            init_hstate = initial_hstate[None, :]  # TBH
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = jax.tree.map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum()
                / traj_batch.info["returned_episode"].sum(),
                traj_batch.info,
            )
            rng = update_state[-1]
            if config["DEBUG"] and config["USE_WANDB"]:

                def callback(metric, update_step):
                    to_log = create_log_dict(metric, config)
                    batch_log(update_step, to_log, config)

                jax.debug.callback(callback, metric, update_step)

            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                rng,
                update_step + 1,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            init_hstate,
            _rng,
            0,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metric": metric}

    return train



def run_ppo(config):
    config = {k.upper(): v for k, v in config.__dict__.items()}
    config["PATH_TO_CHECKPOINT"] = 'None'
    base_timestamps = config['TOTAL_TIMESTEPS']
    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=config["ENV_NAME"]
            + "-PPO_RNN-"
            + str(int(config["TOTAL_TIMESTEPS"] // 1e6))
            + "M",
        )

    num_restarts = 5
    for restart in range(num_restarts):
        rng = jax.random.PRNGKey(config["SEED"])
        print(f"Starting training iteration {restart + 1}/{num_restarts}")

        # Reload weights from the checkpoint
        if os.path.exists(config["PATH_TO_CHECKPOINT"]):
            print(f"Loading weights from checkpoint: {config['PATH_TO_CHECKPOINT']}")
            orbax_checkpointer = PyTreeCheckpointer()
            checkpoint_manager = CheckpointManager(
                config["PATH_TO_CHECKPOINT"],
                orbax_checkpointer,
                CheckpointManagerOptions(max_to_keep=1, create=False),
            )
            with jax.disable_jit():
                if restart == 0:
                    train_state = checkpoint_manager.restore(60000)
                    network_params = train_state['runner_state'][0]["params"]
                else:
                    train_state = checkpoint_manager.restore(int(config['TOTAL_TIMESTEPS']))
                    network_params = train_state['runner_state'][0]["params"]

            print("Weights successfully loaded from checkpoint.")
        else:
            print("No valid checkpoint found, using default initialization.")

            network_params = None  # Initialize or handle default weights
        config['TOTAL_TIMESTEPS'] = base_timestamps * (restart + 1)

        # Split RNG for this training iteration
        rng, current_rng = jax.random.split(rng)

        # Prepare the training function
        train_jit = jax.jit(make_train(config, network_params))

        # Run the training
        t0 = time.time()
        train_state = train_jit(current_rng)
        t1 = time.time()

        # Print performance metrics
        print(f"Iteration {restart + 1} completed.")
        print("Time to run experiment:", t1 - t0)
        print("SPS:", config["TOTAL_TIMESTEPS"] / (t1 - t0))
        
        time.sleep(20)
        # Save checkpoint after this iteration
        checkpoint_dir = f"checkpoint_restart_{restart + 1}"
        checkpoint_path = os.path.join(
            wandb.run.dir if config["USE_WANDB"] else ".", checkpoint_dir
        )

        orbax_checkpointer = PyTreeCheckpointer()
        options = CheckpointManagerOptions(max_to_keep=1, create=True)
        checkpoint_manager = CheckpointManager(checkpoint_path, orbax_checkpointer, options)

        # Save the current train state
        save_args = orbax_utils.save_args_from_target(train_state)
        checkpoint_manager.save(
            config["TOTAL_TIMESTEPS"],
            train_state,
            save_kwargs={"save_args": save_args},
        )
        print(f"Saved checkpoint to {checkpoint_path}")

        # INFERENCE
        common_args = {
            "num_envs": 1024,
            "experiment_name": wandb.run.dir,
            "ratio": min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
            "checkpoint_num": checkpoint_dir,
            "env_name": config['ENV_NAME'],
            "max_grad_norm": config['MAX_GRAD_NORM'],
            "lr": config['LR'],
            "layer_size": config['LAYER_SIZE'],
            "total_timesteps": config['TOTAL_TIMESTEPS'],
            "path": wandb.run.dir,
            "view": False,
            "use_plans": config["USE_PLANS"],
            "inference_step":config["INFERENCE_STEP"],
            "expand_emb": config["EXPAND_EMB"]
        }

        # INFERENCE ON TRAIN
        train_args = ExperimentArgs(**common_args, craftext_settings=config['CRAFTEXT_SETTINGS'])
        train_experiment = Experiment(train_args)
        inference, mean_by_tasks = train_experiment.run()
        wandb.log({"train":mean_by_tasks})

        # INFERENCE ON TEST
        test_args = ExperimentArgs(**common_args, craftext_settings=config['CRAFTEXT_SETTINGS'] + "_test_other_params")
        test_experiment = Experiment(test_args)
        inference, mean_by_tasks = test_experiment.run()
        wandb.log({"test":mean_by_tasks})
        
        # Update PATH_TO_CHECKPOINT for the next iteration
        config["PATH_TO_CHECKPOINT"] = checkpoint_path

    print("All training iterations completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Craftax-Classic-Symbolic-v1")
    parser.add_argument("--craftext_settings", type=str, default=None)
    parser.add_argument("--expand_emb", type=int, default=1)
    parser.add_argument(
        "--num_envs",
        type=int,
        default=16,
    )
    parser.add_argument("--use_plans", type=bool, default=False)
    parser.add_argument("--total_timesteps", type=lambda x: int(float(x)), default=250000000)
    parser.add_argument("--inference_step", type=lambda x: int(float(x)), default=2000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.8)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument(
        "--anneal_lr", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=np.random.randint(2**31))
    parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--save_policy", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument(
        "--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.seed is None:
        args.seed = np.random.randint(2**31)

    if args.jit:
        run_ppo(args)
    else:
        with jax.disable_jit():
            run_ppo(args)