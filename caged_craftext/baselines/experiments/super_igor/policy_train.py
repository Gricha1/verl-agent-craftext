import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from craftax.craftax_env import make_craftax_env_from_name

import wandb
from typing import NamedTuple

from flax.training import orbax_utils
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

from craftext.encoders.craftext_base_model_encoder import EncodeForm
from craftext.enviroment.craftext_wrapper import InstructionWrapper

from baselines.experiments.super_igor.craftext_wrappers.encoder import make_encoder_with_planning
from baselines.experiments.super_igor.craftext_wrappers.env_wrapper import SIPlanning
from baselines.experiments.super_igor.craftext_wrappers.scenarius_loader import create_scenarios_with_super_dataset
from baselines.experiments.super_igor.super_dataset import SuperDataset

from baselines.logz.batch_logging import batch_log, create_log_dict
from baselines.models.actor_critic_with_text import create_actor_critic

from baselines.models.icm import ICMEncoder, ICMForward, ICMInverse
from baselines.wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward_e: jnp.ndarray
    reward_i: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray
    instruction: jnp.ndarray


def make_train(config, network_params):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    # If add CrafText extantion
    env_name = config["ENV_NAME"].replace("-Text", "")
    env = make_craftax_env_from_name(
        env_name, not config["USE_OPTIMISTIC_RESETS"]
    )
    env_params = env.default_params
    #REPLACE INTO DATASET
   # EncodeModel = QwenModelWrapper(config["LLM_PATH"], num_return_sequences=20, split_into_steps=True, make_one_hot=True)
    load_preinited = False
    if config['PLANER_TYPE']=="sd":
        super_dataset = SuperDataset.load_from_json(config["SUPER_DATASET"])
        config['PLANER_CONFIG']['super_dataset'] = super_dataset
        load_preinited = True
            
    EncodeModel = make_encoder_with_planning(planer_type = config['PLANER_TYPE'],
                                             planer_config=config['PLANER_CONFIG'],
                                             embedding_source=config['EMBEDDING_SOURCE'],
                                             step_by_step=config['STEP_BY_STEP'])
    
    ScenariosClass = create_scenarios_with_super_dataset(config['SUPER_DATASET'], update_sd=False, load_preinited=load_preinited)
    env = InstructionWrapper(env, config["CRAFTEXT_SETTINGS"],
                             encode_model_class=EncodeModel,
                            scenario_handler_class=ScenariosClass,
                            encode_form=EncodeForm.EMBEDDING)
    
    env = SIPlanning(env)
    env = LogWrapper(env)
    env = OptimisticResetVecEnvWrapper(
            env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
   
        
    
    

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = create_actor_critic(ac_type=config["AC_TYPE"],
                                     vision_type=config["AC_VISION_TYPE"],
                                     text_encoder_type=config["AC_TEXT_ENCODER_TYPE"],
                                     text_mlp_sizes=config['AC_TEXT_MLP_SIZES'],
                                     nonlinearity=config['AC_NONLINEARITY'],
                                     vision_mlp_sizes=config['AC_VISION_MLP_SIZES'],
                                     layer_width=config["LAYER_SIZE"],
                                     action_dim=env.action_space(env_params).n)

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
        network_params_alt = network.init(_rng, init_x, env.encoded_instruction)
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

        # Exploration state
        ex_state = {
            "icm_encoder": None,
            "icm_forward": None,
            "icm_inverse": None,
            "e3b_matrix": None,
        }

        if config["TRAIN_ICM"]:
            obs_shape = env.observation_space(env_params).shape
            assert len(obs_shape) == 1, "Only configured for 1D observations"
            obs_shape = obs_shape[0]

            # Encoder
            icm_encoder_network = ICMEncoder(
                num_layers=3,
                output_dim=config["ICM_LATENT_SIZE"],
                layer_size=config["ICM_LAYER_SIZE"],
            )
            rng, _rng = jax.random.split(rng)
            icm_encoder_network_params = icm_encoder_network.init(
                _rng, jnp.zeros((1, obs_shape))
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["ICM_LR"], eps=1e-5),
            )
            ex_state["icm_encoder"] = TrainState.create(
                apply_fn=icm_encoder_network.apply,
                params=icm_encoder_network_params,
                tx=tx,
            )

            # Forward
            icm_forward_network = ICMForward(
                num_layers=3,
                output_dim=config["ICM_LATENT_SIZE"],
                layer_size=config["ICM_LAYER_SIZE"],
                num_actions=env.num_actions,
            )
            rng, _rng = jax.random.split(rng)
            icm_forward_network_params = icm_forward_network.init(
                _rng, jnp.zeros((1, config["ICM_LATENT_SIZE"])), jnp.zeros((1,))
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["ICM_LR"], eps=1e-5),
            )
            ex_state["icm_forward"] = TrainState.create(
                apply_fn=icm_forward_network.apply,
                params=icm_forward_network_params,
                tx=tx,
            )

            # Inverse
            icm_inverse_network = ICMInverse(
                num_layers=3,
                output_dim=env.num_actions,
                layer_size=config["ICM_LAYER_SIZE"],
            )
            rng, _rng = jax.random.split(rng)
            icm_inverse_network_params = icm_inverse_network.init(
                _rng,
                jnp.zeros((1, config["ICM_LATENT_SIZE"])),
                jnp.zeros((1, config["ICM_LATENT_SIZE"])),
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["ICM_LR"], eps=1e-5),
            )
            ex_state["icm_inverse"] = TrainState.create(
                apply_fn=icm_inverse_network.apply,
                params=icm_inverse_network_params,
                tx=tx,
            )

            if config["USE_E3B"]:
                ex_state["e3b_matrix"] = (
                    jnp.repeat(
                        jnp.expand_dims(
                            jnp.identity(config["ICM_LATENT_SIZE"]), axis=0
                        ),
                        config["NUM_ENVS"],
                        axis=0,
                    )
                    / config["E3B_LAMBDA"]
                )


        
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            print("HII")
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    ex_state,
                    rng,
                    update_step,
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                print("!!! OBS !!!!", last_obs.shape)
                pi, value = network.apply(train_state.params, last_obs, env_state.env_state.step_embedding)
                
        
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
    
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward_e, done, info = env.step(
                    _rng, env_state, action, env_params
                )

               # print(reward_e)
                reward_i = jnp.zeros(config["NUM_ENVS"])

                if config["TRAIN_ICM"]:
                    latent_obs = ex_state["icm_encoder"].apply_fn(
                        ex_state["icm_encoder"].params, last_obs
                    )
                    latent_next_obs = ex_state["icm_encoder"].apply_fn(
                        ex_state["icm_encoder"].params, obsv
                    )

                    latent_next_obs_pred = ex_state["icm_forward"].apply_fn(
                        ex_state["icm_forward"].params, latent_obs, action
                    )
                    error = (latent_next_obs - latent_next_obs_pred) * (
                        1 - done[:, None]
                    )
                    mse = jnp.square(error).mean(axis=-1)

                    reward_i = mse * config["ICM_REWARD_COEFF"]

                    if config["USE_E3B"]:
                        # Embedding is (NUM_ENVS, 128)
                        # e3b_matrix is (NUM_ENVS, 128, 128)
                        us = jax.vmap(jnp.matmul)(ex_state["e3b_matrix"], latent_obs)
                        bs = jax.vmap(jnp.dot)(latent_obs, us)

                        def update_c(c, b, u):
                            return c - (1.0 / (1 + b)) * jnp.outer(u, u)

                        updated_cs = jax.vmap(update_c)(ex_state["e3b_matrix"], bs, us)
                        new_cs = (
                            jnp.repeat(
                                jnp.expand_dims(
                                    jnp.identity(config["ICM_LATENT_SIZE"]), axis=0
                                ),
                                config["NUM_ENVS"],
                                axis=0,
                            )
                            / config["E3B_LAMBDA"]
                        )
                        ex_state["e3b_matrix"] = jnp.where(
                            done[:, None, None], new_cs, updated_cs
                        )

                        e3b_bonus = jnp.where(
                            done, jnp.zeros((config["NUM_ENVS"],)), bs
                        )

                        reward_i = e3b_bonus * config["E3B_REWARD_COEFF"]

                reward = reward_e + reward_i
              
               # instruction = jnp.repeat(env.encoded_instruction, done.shape[0], axis=0)
               # print(instruction.shape)
                
                transition = Transition(
                    done=done,
                    action=action,
                    value=value,
                    reward=reward,
                    reward_i=reward_i,
                    reward_e=reward_e,
                    log_prob=log_prob,
                    obs=last_obs,
                    next_obs=obsv,
                    info=info,
                    instruction=env_state.env_state.step_embedding# env.encoded_instruction[0]
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    ex_state,
                    rng,
                    update_step,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
           # exit()
            # print(len(traj_batch))
            # print(traj_batch)
            # exit()

            # CALCULATE ADVANTAGE
            (
                train_state,
                env_state,
                last_obs,
                ex_state,
                rng,
                update_step,
            ) = runner_state
            _, last_val = network.apply(train_state.params, last_obs, env_state.env_state.step_embedding)
          #  exit()
            def _calculate_gae(traj_batch, last_val):
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

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    # Policy/value network
                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs, traj_batch.instruction)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
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
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)

                    losses = (total_loss, 0)
                    return train_state, losses

                (
                    train_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                print( config["MINIBATCH_SIZE"], config["NUM_MINIBATCHES"])

                print(traj_batch.instruction.shape)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, losses = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, losses

            update_state = (
                train_state,
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

            # UPDATE EXPLORATION STATE
            def _update_ex_epoch(update_state, unused):
                def _update_ex_minbatch(ex_state, traj_batch):
                    def _inverse_loss_fn(
                        icm_encoder_params, icm_inverse_params, traj_batch
                    ):
                        latent_obs = ex_state["icm_encoder"].apply_fn(
                            icm_encoder_params, traj_batch.obs
                        )
                        latent_next_obs = ex_state["icm_encoder"].apply_fn(
                            icm_encoder_params, traj_batch.next_obs
                        )

                        action_pred_logits = ex_state["icm_inverse"].apply_fn(
                            icm_inverse_params, latent_obs, latent_next_obs
                        )
                        true_action = jax.nn.one_hot(
                            traj_batch.action, num_classes=action_pred_logits.shape[-1]
                        )

                        bce = -jnp.mean(
                            jnp.sum(
                                action_pred_logits
                                * true_action
                                * (1 - traj_batch.done[:, None]),
                                axis=1,
                            )
                        )

                        return bce * config["ICM_INVERSE_LOSS_COEF"]

                    inverse_grad_fn = jax.value_and_grad(
                        _inverse_loss_fn,
                        has_aux=False,
                        argnums=(
                            0,
                            1,
                        ),
                    )
                    inverse_loss, grads = inverse_grad_fn(
                        ex_state["icm_encoder"].params,
                        ex_state["icm_inverse"].params,
                        traj_batch,
                    )
                    icm_encoder_grad, icm_inverse_grad = grads
                    ex_state["icm_encoder"] = ex_state["icm_encoder"].apply_gradients(
                        grads=icm_encoder_grad
                    )
                    ex_state["icm_inverse"] = ex_state["icm_inverse"].apply_gradients(
                        grads=icm_inverse_grad
                    )

                    def _forward_loss_fn(icm_forward_params, traj_batch):
                        latent_obs = ex_state["icm_encoder"].apply_fn(
                            ex_state["icm_encoder"].params, traj_batch.obs
                        )
                        latent_next_obs = ex_state["icm_encoder"].apply_fn(
                            ex_state["icm_encoder"].params, traj_batch.next_obs
                        )

                        latent_next_obs_pred = ex_state["icm_forward"].apply_fn(
                            icm_forward_params, latent_obs, traj_batch.action
                        )

                        error = (latent_next_obs - latent_next_obs_pred) * (
                            1 - traj_batch.done[:, None]
                        )
                        return (
                            jnp.square(error).mean() * config["ICM_FORWARD_LOSS_COEF"]
                        )

                    forward_grad_fn = jax.value_and_grad(
                        _forward_loss_fn, has_aux=False
                    )
                    forward_loss, icm_forward_grad = forward_grad_fn(
                        ex_state["icm_forward"].params, traj_batch
                    )
                    ex_state["icm_forward"] = ex_state["icm_forward"].apply_gradients(
                        grads=icm_forward_grad
                    )

                    losses = (inverse_loss, forward_loss)
                    return ex_state, losses

                (ex_state, traj_batch, rng) = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)

                exit()
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), traj_batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                ex_state, losses = jax.lax.scan(
                    _update_ex_minbatch, ex_state, minibatches
                )
                update_state = (ex_state, traj_batch, rng)
                return update_state, losses

            if config["TRAIN_ICM"]:
                ex_update_state = (ex_state, traj_batch, rng)
                ex_update_state, ex_loss = jax.lax.scan(
                    _update_ex_epoch,
                    ex_update_state,
                    None,
                    config["EXPLORATION_UPDATE_EPOCHS"],
                )
                metric["icm_inverse_loss"] = ex_loss[0].mean()
                metric["icm_forward_loss"] = ex_loss[1].mean()
                metric["reward_i"] = traj_batch.reward_i.mean()
                metric["reward_e"] = traj_batch.reward_e.mean()

                ex_state = ex_update_state[0]
                rng = ex_update_state[-1]

            # wandb logging
            if config["DEBUG"] and config["USE_WANDB"]:

                def callback(metric, update_step):
                    to_log = create_log_dict(metric, config)
                    batch_log(update_step, to_log, config)

                jax.debug.callback(
                    callback,
                    metric,
                    update_step,
                )

            runner_state = (
                train_state,
                env_state,
                last_obs,
                ex_state,
                rng,
                update_step + 1,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            ex_state,
            _rng,
            0,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}  # , "info": metric}

    return train


def extract_planer_config(config):
    planner_config = {'model_config': \
                {
                'original_model_path': config['original_model_path'.upper()],
                'peft_weights_path': config['peft_weights_path'.upper()],
                },
              'generation_config':\
                {
                'num_paraphrases': config['num_paraphrases'.upper()],
                'beam_groups': config['beam_groups'.upper()],
                'beams_count': config['beams_count'.upper()],
                'max_new_tokens': config['max_new_tokens'.upper()],
                'prompt_template': config['prompt_template'.upper()],
                },
                'super_dataset':None,
                'augment':config['augment'.upper()]
             }
    return planner_config

def run_ppo(config):
    # Convert config keys to uppercase for consistency
    config = {k.upper(): v for k, v in config.__dict__.items()}
    config['PLANER_CONFIG'] = extract_planer_config(config)
    if config['start_checkpoint_path'.upper()] != "None":
        start_checkpoint_path = config['start_checkpoint_path'.upper()] # os.path.abspath("./wandb" + config['start_checkpoint_path'.upper()].split("wandb")[1])
    else:
        start_checkpoint_path = "None"
    base_checkpoint_path = start_checkpoint_path #os.path.abspath("./wandb/achivments_v2/files/checkpoint_restart_1")
    config["PATH_TO_CHECKPOINT"] = base_checkpoint_path  # Initialize with no checkpoint
    base_timestamps = config['TOTAL_TIMESTEPS']
    # Initialize WandB if enabled
    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=config["ENV_NAME"]
            + "-"
            + str(int(config["TOTAL_TIMESTEPS"] // 1e6))
            + "M",
        )
    if config['ENCODE_FORM_NAME'] == 'EMBEDDING':
        config['ENCODE_FORM'] = EncodeForm.EMBEDDING
    elif config['ENCODE_FORM_NAME'] == 'WEIGHTED_MEAN':
        config['ENCODE_FORM'] = EncodeForm.WEIGHTED_MEAN
    elif config['ENCODE_FORM_NAME'] == 'TOKENS':
        config['ENCODE_FORM'] = EncodeForm.TOKEN
    elif config['ENCODE_FORM_NAME'] == 'EMBED_CLS_FOR_SPLITS':
         config['ENCODE_FORM'] = EncodeForm.EMBED_CLS_FOR_SPLITS

    # Initialize random keys
    rng = jax.random.PRNGKey(config["SEED"])

    # Define the number of restarts
    num_restarts = 1  # Hyperparameter for the number of restarts
    for restart in range(0, num_restarts):
        print(f"Starting training iteration {restart + 1}/{num_restarts}")

        # Reload weights from the checkpoint
        if os.path.exists(config["PATH_TO_CHECKPOINT"]):
            print("---- " * 80)
            print()
            print(f"Loading weights from checkpoint: {config['PATH_TO_CHECKPOINT']}")
            print()
            print("---- " * 80)
            orbax_checkpointer = PyTreeCheckpointer()
            checkpoint_manager = CheckpointManager(
                config["PATH_TO_CHECKPOINT"],
                orbax_checkpointer,
                CheckpointManagerOptions(max_to_keep=1, create=False),
            )
            with jax.disable_jit():
                # if restart == 0:
                #     train_state = checkpoint_manager.restore(60000)
                #     network_params = train_state['runner_state'][0]["params"]
                # else:
                    timesteps = 250000000 #int(config['TOTAL_TIMESTEPS'])
                    train_state = checkpoint_manager.restore(timesteps)
                    network_params = train_state['runner_state'][0]["params"]
                    #print(train_state.keys())
          #  network_params = train_state["params"]
            print("Weights successfully loaded from checkpoint.")
        elif config["PATH_TO_CHECKPOINT"]=="None" or config["PATH_TO_CHECKPOINT"] is None:
            print("---- " * 80)
            print()
            print(config["PATH_TO_CHECKPOINT"])
            print(f"No valid checkpoint found, using default initialization.")
            print()
            print("---- " * 80)
            network_params = None
        else:
            print("---- " * 80)
            print()
            print(config["PATH_TO_CHECKPOINT"])
            print(f"No valid checkpoint found, using default initialization.")
            print()
            print("---- " * 80)
            exit()

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
        
       # time.sleep(20)
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

    os.makedirs(config['EXPERIMENT_NAME'], exist_ok=True)
    with open(f"{config['EXPERIMENT_NAME']}/path_to_last_checkpoint.txt", "w", encoding="utf-8") as file:
        file.write(checkpoint_path)

    print("All training iterations completed.")
    wandb.finish()





if __name__ == "__main__":
    #--env_name "Craftax-Pixels-v1-Text"
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Craftax-Pixels-v1-Text")
    parser.add_argument("--start_checkpoint_path", type=str, default=None)
    parser.add_argument("--llm_path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--super_dataset", type=str, default=None)
    parser.add_argument("--craftext_settings", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default="experiment")
    parser.add_argument("--encode_form_name", type=str, default="EMBEDDING")
    
    #Plan generation config
    parser.add_argument("--planer_type", type=str, default="llm")
    parser.add_argument("--embedding_source", type=int, default=1)
    parser.add_argument("--step_by_step", type=bool, default=True)
    parser.add_argument("--original_model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--peft_weights_path", type=str, default=None)
    parser.add_argument("--num_paraphrases", type=int, default=15)
    parser.add_argument("--beam_groups", type=int, default=15)
    parser.add_argument("--beams_count", type=int, default=15)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--prompt_template", type=int, default=2)
    parser.add_argument("--augment", type=bool, default=False)
    
    #Observation Encoder config
    parser.add_argument("--ac_type", type=str, default="ac_model")
    parser.add_argument("--ac_vision_type", type=str, default="resnet_impala")
    parser.add_argument("--ac_text_encoder_type", type=str, default="mlp")
    parser.add_argument("--ac_text_mlp_sizes", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--ac_nonlinearity", type=str, default="relu")
    parser.add_argument("--ac_vision_mlp_sizes",type=int, nargs="+", default=[256,])


    parser.add_argument(
        "--num_envs",
        type=int,
        default=1024,#1024,
    )
    parser.add_argument(
        "--total_timesteps", type=lambda x: int(float(x)), default=250000000 
    )  # Allow scientific notation
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.8)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.1) #0.01
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument(
        "--anneal_lr", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--save_policy", action="store_true")
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument(
        "--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--optimistic_reset_ratio", type=int, default=1) #16

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
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.use_e3b:
        assert args.train_icm
        assert args.icm_reward_coeff == 0
    if args.seed is None:
        args.seed = 42

    if args.jit:
        run_ppo(args)
    else:
        with jax.disable_jit():
            run_ppo(args)
