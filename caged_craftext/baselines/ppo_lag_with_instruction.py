import argparse
import os
import sys
import time

import logging

import jax
import jax.numpy as jnp
import numpy as np
import optax
from craftax.craftax_env import make_craftax_env_from_name

#import wandb
import comet_ml
from typing import NamedTuple

from flax.training import orbax_utils
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)


from logz.batch_logging_cmdp import batch_log, create_log_dict
from models.actor_critic import (
    ActorCritic,
    ActorCriticConv)
from models.actor_critic_with_text_constraints import (
    ActorCriticConvWithBERTCMDP
)
from models.icm import ICMEncoder, ICMForward, ICMInverse
from wrappers_cmdp import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
)

from craftext.environment.craftext_wrapper_cmdp import CMDPInstructionWrapper

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    cost_value: jnp.ndarray
    reward_e: jnp.ndarray
    reward_i: jnp.ndarray
    reward: jnp.ndarray
    cost: jnp.ndarray # CMDP
    episode_cost: jnp.ndarray # CMDP
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray
    instruction: jnp.ndarray
    textual_constraint: jnp.ndarray


def make_train(config, network_params, experiment=None):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    assert config["NUM_UPDATES"] >= config["SAVE_FREQ"]
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    # If add CrafText extantion
    env_name = config["ENV_NAME"].replace("-Text", "")
    env = make_craftax_env_from_name(
        env_name, not config["USE_OPTIMISTIC_RESETS"]
    )
    env_params = env.default_params
    env = CMDPInstructionWrapper(env, config["CRAFTEXT_SETTINGS"])
    env = LogWrapper(env)
    env = OptimisticResetVecEnvWrapper(
            env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )

    def create_unique_checkpoint_dir(base_dir=f"{args.dir_path}/baselines/checkpoints/{args.algo_name}", prefix="exp_"):
        os.makedirs(base_dir, exist_ok=True)
        existing = [d for d in os.listdir(base_dir) if d.startswith(prefix) and os.path.isdir(os.path.join(base_dir, d))]
        indices = []
        for d in existing:
            try:
                idx = int(d[len(prefix):])
                indices.append(idx)
            except ValueError:
                pass
        next_idx = max(indices) + 1 if indices else 0
        new_dir = os.path.join(base_dir, f"{prefix}{next_idx}")
        os.makedirs(new_dir)
        return new_dir

    checkpoint_dir = create_unique_checkpoint_dir()
    config["PATH_TO_CHECKPOINT"] = checkpoint_dir
    orbax_checkpointer = PyTreeCheckpointer()
    checkpoint_manager = CheckpointManager(
        checkpoint_dir,
        orbax_checkpointer,
        CheckpointManagerOptions(max_to_keep=2),
    )
    #wandb.config.update({"PATH_TO_CHECKPOINT": checkpoint_dir}, allow_val_change=True) 
    experiment.log_parameter("PATH_TO_CHECKPOINT", checkpoint_dir)
    
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        if "Symbolic" in config["ENV_NAME"]:
            assert 1 == 0, "didnt implemented for CMDP"
            network = ActorCritic(env.action_space(env_params).n, config["LAYER_SIZE"])
            encoded_input_tiled = env.encoded_instruction           # (768,)
        elif "Text" in config["ENV_NAME"]:
            encoded = env.encoded_instruction           # (768,)  # (1, 768)
            encoded = jnp.expand_dims(encoded, axis=0)  # 1, 768)
            encoded_input_tiled = jnp.tile(encoded,
                                    (config["NUM_ENVS"], 1))
            encoded_constraint = env.encoded_textual_constraint           # (768,)  # (1, 768)
            encoded_constraint = jnp.expand_dims(encoded_constraint, axis=0)  # 1, 768)
            encoded_constraint_tiled = jnp.tile(encoded_constraint,
                                    (config["NUM_ENVS"], 1))
            network = ActorCriticConvWithBERTCMDP(
                env.action_space(env_params).n, config["LAYER_SIZE"]
            )
        else:
            assert 1 == 0, "didnt implemented for CMDP"
            encoded_input_tiled = env.encoded_instruction
            network = ActorCriticConv(
                env.action_space(env_params).n, config["LAYER_SIZE"]
            )

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(
                (config["NUM_ENVS"], *env.observation_space(env_params).shape)
            )

        print(init_x.shape)
        
        network_params_alt = network.init(_rng, init_x, encoded_input_tiled, encoded_constraint_tiled)
        
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5) if not config["NO_LR_SCHEDULING"] else optax.adam(learning_rate=config["LR"], eps=1e-5),
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
        
        # Set up optimizers for policy and value function
        lambda_init = {'lambda': jnp.array(config["INIT_LAMBDA"], dtype=jnp.float32)}  # Начальное значение lambda
        lambda_optimizer = optax.adam(learning_rate=3e-4)  # Оптимизатор для lambda
        lambda_state = TrainState.create(
            apply_fn=lambda x: x['lambda'],  # Просто возвращаем параметр
            params=lambda_init,
            tx=lambda_optimizer,
        )
         

        # Exploration state
        ex_state = {
            "icm_encoder": None,
            "icm_forward": None,
            "icm_inverse": None,
            "e3b_matrix": None,
        }

        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            print("HII")
            def _env_step(runner_state, unused):
                (
                    train_state,
                    lambda_state,
                    env_state,
                    last_obs,
                    ex_state,
                    rng,
                    update_step,
                    global_steps,
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                print("!!! OBS !!!!", last_obs.shape)
                pi, value, cost_value = network.apply(train_state.params, last_obs, 
                                          env_state.env_state.instruction,
                                          env_state.env_state.textual_constraint)
                
        
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
    
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward_e, done, info = env.step(
                    _rng, env_state, action, env_params
                )

                global_steps += config["NUM_ENVS"]
                cost = info["cost"]
                episode_cost = info["episode_cost"]
                #cost = info["cost"] # CMDP
                
               # print(reward_e)
                reward_i = jnp.zeros(config["NUM_ENVS"])
                reward = reward_e + reward_i
              
               # instruction = jnp.repeat(env.encoded_instruction, done.shape[0], axis=0)
               # print(instruction.shape)
                
                transition = Transition(
                    done=done,
                    action=action,
                    value=value,
                    cost_value=cost_value, # CMDP
                    reward=reward,
                    reward_i=reward_i,
                    reward_e=reward_e,
                    cost=cost, # CMDP
                    episode_cost=episode_cost,
                    log_prob=log_prob,
                    obs=last_obs,
                    next_obs=obsv,
                    info=info,
                    instruction=env_state.env_state.instruction, # env.encoded_instruction[0]
                    textual_constraint=env_state.env_state.textual_constraint
                )
                runner_state = (
                    train_state,
                    lambda_state,
                    env_state,
                    obsv,
                    ex_state,
                    rng,
                    update_step,
                    global_steps,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            mean_episode_cost = (traj_batch.episode_cost * traj_batch.info["returned_episode"]).sum() / traj_batch.info["returned_episode"].sum()

            # CALCULATE ADVANTAGE
            (
                train_state,
                lambda_state,
                env_state,
                last_obs,
                ex_state,
                rng,
                update_step,
                global_steps,
            ) = runner_state
            _, last_val, cost_last_val = network.apply(train_state.params, last_obs, 
                                        env_state.env_state.instruction, 
                                        env_state.env_state.textual_constraint)
          #  exit()
            def _calculate_gae_reward(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                        _get_advantages,
                         (jnp.zeros_like(last_val), last_val),
                        traj_batch,
                        reverse=True,
                        unroll=16,
                    )

                return advantages, advantages + traj_batch.value

            def _calculate_gae_cost(traj_batch, cost_last_val):
                def _get_cost_advantages(gae_and_cost_next_value, transition):
                    cost_gae, next_cost_value = gae_and_cost_next_value
                    done, cost_value, cost = (
                        transition.done,
                        transition.cost_value,
                        transition.cost,
                    )
                    cost_delta = cost + config["GAMMA"] * next_cost_value * (1 - done) - cost_value
                    cost_gae = (
                        cost_delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * cost_gae
                    )
                    return (cost_gae, cost_value), cost_gae

                _, cost_advantages = jax.lax.scan(
                        _get_cost_advantages,
                         (jnp.zeros_like(cost_last_val), cost_last_val),
                        traj_batch,
                        reverse=True,
                        unroll=16,
                    )

                return cost_advantages, cost_advantages + traj_batch.cost_value

            advantages, targets = _calculate_gae_reward(traj_batch, last_val)
            cost_advantages, cost_targets = _calculate_gae_cost(traj_batch, cost_last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_lambda_state, batch_info):
                    train_state, lambda_state = train_lambda_state
                    traj_batch, advantages, targets, cost_advantages, cost_targets = batch_info

                    # UPDATE LAMBDA
                    def lambda_loss(current_lambda):
                        cost_violation = mean_episode_cost - config["COST_THRESHOLD"]
                        return -current_lambda * cost_violation
                    
                    lambda_grad = jax.grad(lambda_loss)(lambda_state.params['lambda'])
                    lambda_state = lambda_state.apply_gradients(grads={'lambda': lambda_grad})

                    # Policy/value network
                    def _loss_fn(params, traj_batch, gae, targets, cost_gae, cost_targets):
                        # RERUN NETWORK
                        pi, value, cost_value = network.apply(params, traj_batch.obs, 
                                                  traj_batch.instruction,
                                                  traj_batch.textual_constraint)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE REWARD VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])

                        ### ERROR WITH MEAN
                        print(value.shape)
                        print(targets.shape)
                        
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE COST VALUE LOSS
                        cost_value_pred_clipped = traj_batch.cost_value + (
                            cost_value - traj_batch.cost_value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])

                        ### ERROR WITH MEAN
                        print(cost_value.shape)
                        print(cost_targets.shape)
                        
                        cost_value_losses = jnp.square(cost_value - cost_targets)
                        cost_value_losses_clipped = jnp.square(cost_value_pred_clipped - cost_targets)
                        cost_value_loss = (
                            0.5 * jnp.maximum(cost_value_losses, cost_value_losses_clipped).mean()
                        )

                        # CALCULATE REWARD ACTOR LOSS
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
                        # CALCULATE COST ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        cost_gae = (cost_gae - cost_gae.mean()) / (cost_gae.std() + 1e-8)
                        cost_loss_actor1 = ratio * cost_gae
                        cost_loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * cost_gae
                        )

                        current_lambda = jax.nn.softplus(lambda_state.params['lambda'])
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        #cost_loss_actor = jnp.minimum(cost_loss_actor1, cost_loss_actor2)
                        cost_loss_actor = cost_loss_actor1
                        loss_actor = loss_actor.mean()
                        loss_actor += current_lambda * cost_loss_actor.mean()
                        loss_actor = loss_actor / (1 + current_lambda)
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            + config["VF_COEF"] * cost_value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, cost_value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets, cost_advantages, cost_targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)

                    losses = (total_loss, 0)
                    return (train_state, lambda_state), losses

                (
                    train_state,
                    lambda_state,
                    traj_batch,
                    advantages,
                    targets,
                    cost_advantages,
                    cost_targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets, cost_advantages, cost_targets)
                print( config["MINIBATCH_SIZE"], config["NUM_MINIBATCHES"])

                print(traj_batch.instruction.shape)
                print(traj_batch.textual_constraint.shape)
                batch = jax.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_lambda_state, losses = jax.lax.scan(
                    _update_minbatch, (train_state, lambda_state), minibatches
                )
                train_state, lambda_state = train_lambda_state

                update_state = (
                    train_state,
                    lambda_state,
                    traj_batch,
                    advantages,
                    targets,
                    cost_advantages,
                    cost_targets,
                    rng,
                )
                return update_state, losses

            update_state = (
                train_state,
                lambda_state,
                traj_batch,
                advantages,
                targets,
                cost_advantages,
                cost_targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )

            train_state = update_state[0]
            lambda_state = update_state[1]
            metric = jax.tree_map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum()
                / traj_batch.info["returned_episode"].sum(),
                traj_batch.info,
            )

            rng = update_state[-1]

            metric["episode_cost"] = mean_episode_cost
            metric["global_steps"] = global_steps
            metric["lambda"] = lambda_state.params['lambda']

            # wandb logging
            if config["DEBUG"] and config["USE_WANDB"]:

                def callback(metric, update_step):
                    to_log = create_log_dict(metric, config)
                    batch_log(update_step, to_log, config, experiment)

                jax.debug.callback(
                    callback,
                    metric,
                    update_step,
                )

            runner_state = (
                train_state,
                lambda_state,
                env_state,
                last_obs,
                ex_state,
                rng,
                update_step + 1,
                global_steps,
            )

            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            lambda_state,
            env_state,
            obsv,
            ex_state,
            _rng,
            0,
            0,
        )
        
        def _maybe_save(runner_state):
            train_state, lambda_state, env_state, last_obs, ex_state, rng, update_step, global_steps = runner_state
            
            def _save_callback(train_state, global_steps):
                current_step = jax.device_get(global_steps)
                print(f"Saving weights at step {current_step}")
                save_args = orbax_utils.save_args_from_target(train_state)
                checkpoint_manager.save(
                    int(current_step),
                    {"train_state": train_state},
                    save_kwargs={"save_args": {"train_state": save_args}},
                )
            
            should_save = update_step % config["SAVE_FREQ"] == 0
            jax.lax.cond(
                should_save,
                lambda: jax.debug.callback(_save_callback, train_state, global_steps) or (),
                lambda: ()
            )
                
            return runner_state

        def _scan_update(runner_state, unused):
            runner_state, metric = _update_step(runner_state, unused)
            
            # Добавляем шаг сохранения
            runner_state = _maybe_save(runner_state)
            
            return runner_state, metric
        
        runner_state, metric = jax.lax.scan(
            _scan_update, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}  # , "info": metric}

    return train

    
def run_ppo(config):
    # Convert config keys to uppercase for consistency
    config = {k.upper(): v for k, v in config.__dict__.items()}
    config["PATH_TO_CHECKPOINT"] = 'None'# base_checkpoint_path  # Initialize with no checkpoint
    base_timestamps = config['TOTAL_TIMESTEPS']
    # Initialize WandB if enabled
    #if config["USE_WANDB"]:
    #    wandb.init(
    #        project=config["WANDB_PROJECT"],
    #        entity=config["WANDB_ENTITY"],
    #        config=config,
    #        name=config["ENV_NAME"]
    #        + "-"
    #        + str(int(config["TOTAL_TIMESTEPS"] // 1e6))
    #        + "M",
    #    )
    if config["USE_COMET"]:
        comet_ml.login()
        experiment = comet_ml.start(project_name="ppo_lag_craftext")
        experiment.log_parameters(config)

    # Initialize random keys
    rng = jax.random.PRNGKey(config["SEED"])

    # Define the number of restarts
    num_restarts = 1  # Hyperparameter for the number of restarts
    for restart in range(num_restarts):
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
                    #print(train_state.keys())
          #  network_params = train_state["params"]
            print("Weights successfully loaded from checkpoint.")
        else:
            print("No valid checkpoint found, using default initialization.")
           # exit()
            network_params = None  # Initialize or handle default weights
        config['TOTAL_TIMESTEPS'] = base_timestamps * (restart + 1)

        # Split RNG for this training iteration
        rng, current_rng = jax.random.split(rng)

        # Prepare the training function
        train_jit = jax.jit(make_train(config, network_params, experiment))

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

        # Update PATH_TO_CHECKPOINT for the next iteration
        config["PATH_TO_CHECKPOINT"] = checkpoint_path

    print("All training iterations completed.")




if __name__ == "__main__":
    #--env_name "Craftax-Pixels-v1-Text"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str, default="/usr/home/workspace")
    parser.add_argument("--algo_name", type=str, default="PPO_LAG")
    parser.add_argument("--env_name", type=str, default="Craftax-Pixels-v1-Text")
    parser.add_argument("--craftext_settings", type=str, default=None)
    parser.add_argument(
        "--num_envs",
        type=int,
        default=256,#1024,
    )
    parser.add_argument(
        "--total_timesteps", type=lambda x: int(float(x)), default=1250000000 
    )  # Allow scientific notation
    parser.add_argument("--save_freq", type=int, default=10) # при env_num=512, сохраняет при 512000, x2, x3, ...

    # PPO
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--no_lr_scheduling", action="store_true", default=False)
    parser.add_argument("--num_steps", type=int, default=100)
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
        "--anneal_lr", default=True
    )

    # PPO LAG
    parser.add_argument("--cost_threshold", type=float, default=5.0)
    parser.add_argument("--init_lambda", type=float, default=1.0)

    # debug
    parser.add_argument("--debug", default=True)
    parser.add_argument("--jit", default=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--use_wandb", default=True
    )
    parser.add_argument(
        "--use_comet", default=True
    )
    parser.add_argument("--save_policy", action="store_true")
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--wandb_project", type=str, default="cmdp_craftext")
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument(
        "--use_optimistic_resets", default=True
    )
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)

    # EXPLORATION
    parser.add_argument("--exploration_update_epochs", type=int, default=4)
    # ICM
    parser.add_argument("--icm_reward_coeff", type=float, default=1.0)
    parser.add_argument("--train_icm", action="store_true")
    parser.add_argument("--icm_lr", type=float, default=3e-4)
    parser.add_argument("--icm_forward_loss_coef", type=float, default=1.0)
    parser.add_argument("--icm_inverse_loss_coef", type=float, default=1.0)
    parser.add_argument("--icm_layer_size", type=int, default=256)
    parser.add_argument("--icm_latent_size", type=int, default=32)
    # E3B
    parser.add_argument("--e3b_reward_coeff", type=float, default=1.0)
    parser.add_argument("--use_e3b", action="store_true")
    parser.add_argument("--e3b_lambda", type=float, default=0.1)

    args, rest_args = parser.parse_known_args(sys.argv[1:])

    # Отключить логи уровня INFO и ниже для orbax
    logging.getLogger("orbax").setLevel(logging.WARNING)

    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.use_e3b:
        assert args.train_icm
        assert args.icm_reward_coeff == 0
    if args.seed is None:
        args.seed = np.random.randint(2**31)

    if args.jit:
        run_ppo(args)
    else:
        with jax.disable_jit():
            run_ppo(args)