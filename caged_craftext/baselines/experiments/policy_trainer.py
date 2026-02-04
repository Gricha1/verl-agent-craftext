import jax
import jax.numpy as jnp
import numpy as np
import optax
from craftax.craftax_env import make_craftax_env_from_name

from typing import NamedTuple

from flax.training.train_state import TrainState

from baselines.experiments.super_igor.encoder import  EncodeForm, QwenModelWrapper
from baselines.experiments.super_igor.scenarius_loader import create_scenarios_with_super_dataset

from baselines.logz.batch_logging import batch_log, create_log_dict
from baselines.models.actor_critic import (
    ActorCritic,
    ActorCriticConv,
    ActorCriticConvWithBERT,
)
from baselines.models.icm import ICMEncoder, ICMForward, ICMInverse
from baselines.wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
)
from craftext.environment.craftext_wrapper import InstructionWrapper


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
    EncodeModel = QwenModelWrapper(config["LLM_PATH"], num_return_sequences=1)
    ScenariosClass = create_scenarios_with_super_dataset(config['SUPER_DATASET'])
    env = InstructionWrapper(env, config["CRAFTEXT_SETTINGS"],
                             encode_model_class=EncodeModel,
                            scenario_handler_class=ScenariosClass,
                            encode_form=EncodeForm.WEIGHTED_MEAN)
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
        if "Symbolic" in config["ENV_NAME"]:
            network = ActorCritic(env.action_space(env_params).n, config["LAYER_SIZE"])
        elif "Text" in config["ENV_NAME"]:
            network = ActorCriticConvWithBERT(
                env.action_space(env_params).n, config["LAYER_SIZE"]
            )
        else:
            network = ActorCriticConv(
                env.action_space(env_params).n, config["LAYER_SIZE"]
            )

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
                pi, value = network.apply(train_state.params, last_obs, env_state.env_state.instruction)
                
        
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
                    instruction=env_state.env_state.instruction# env.encoded_instruction[0]
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
            _, last_val = network.apply(train_state.params, last_obs, env_state.env_state.instruction)
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
