"""Implementations of algorithms for continuous control."""
import os
from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import optax
import flax
import pickle
from flax.training.train_state import TrainState
from flax import struct
import numpy as np
from jaxrl5.agents.agent import Agent
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.networks import MLP, Ensemble, StateActionValue, StateValue, Relu_StateActionValue, Relu_StateValue, AdvCondDDPM, FourierFeatures, cosine_beta_schedule, adv_cond_ddpm_sampler, MLPResNet, get_weight_decay_mask, vp_beta_schedule
from jaxrl5.networks.diffusion import adv_cond_dpm_solver_sampler_1st, vp_sde_schedule, AdvReturnPredictor, adv_cond_comp_ddpm_sampler, AdvCondCompDDPM, AdvGuidedCompDDPM, adv_cond_comp_flow_sampler, AdvGuidedSingleCompDDPM, adv_cond_single_comp_ddpm_sampler, adv_cond_single_comp_flow_sampler


def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

def safe_expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff < 0, expectile, (1 - expectile))
    return weight * (diff**2)

@partial(jax.jit, static_argnames=('critic_fn'))
def compute_q(critic_fn, critic_params, observations, actions):
    q_values = critic_fn({'params': critic_params}, observations, actions)
    q_values = q_values.min(axis=0)
    return q_values

@partial(jax.jit, static_argnames=('value_fn'))
def compute_v(value_fn, value_params, observations):
    v_values = value_fn({'params': value_params}, observations)
    return v_values

@partial(jax.jit, static_argnames=('adv_return_predictor_fn'))
def compute_adv_r(adv_return_predictor_fn, adv_return_predictor_params, observations, adv_c):
    adv_r = adv_return_predictor_fn({'params': adv_return_predictor_params}, observations, adv_c)
    return adv_r

@partial(jax.jit, static_argnames=('safe_critic_fn'))
def compute_safe_q(safe_critic_fn, safe_critic_params, observations, actions):
    safe_q_values = safe_critic_fn({'params': safe_critic_params}, observations, actions)
    safe_q_values = safe_q_values.max(axis=0)
    return safe_q_values

def mish(x):
    return x * jnp.tanh(nn.softplus(x))

class Co3GSingleComp(Agent):
    score_model: TrainState
    target_score_model: TrainState
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    safe_critic: TrainState
    safe_target_critic: TrainState
    safe_value: TrainState
    discount: float
    tau: float
    actor_tau: float
    critic_hyperparam: float
    cost_critic_hyperparam: float
    reverse_cost_temp: float
    reverse_reward_temp: float
    reward_advantage_predictor_hyperparam: float
    critic_objective: str = struct.field(pytree_node=False)
    critic_type: str = struct.field(pytree_node=False)
    actor_type: str = struct.field(pytree_node=False)
    actor_objective: str = struct.field(pytree_node=False)
    sampling_method: str = struct.field(pytree_node=False)
    extract_method: str = struct.field(pytree_node=False)
    act_dim: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)
    N: int #How many samples per observation
    M: int = struct.field(pytree_node=False) #How many repeat last steps
    clip_sampler: bool = struct.field(pytree_node=False)
    ddpm_temperature: float
    cost_temperature: float
    reward_temperature: float
    qc_thres: float
    cost_ub: float
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_hats: jnp.ndarray
    wr_init: float
    wc_init: float
    eps_l_reward: float
    eps_l_reward_lower: float
    eps_l_cost: float

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        actor_architecture: str = 'mlp',
        actor_lr: Union[float, optax.Schedule] = 3e-4,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_hidden_dims: Sequence[int] = (256, 256),
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        critic_hyperparam: float = 0.8,
        cost_critic_hyperparam: float = 0.8,
        reward_advantage_predictor_hyperparam: float = 0.8,
        reverse_cost_temp: float = 0.5,
        reverse_reward_temp: float = 0.5,
        ddpm_temperature: float = 1.0,
        num_qs: int = 2,
        actor_num_blocks: int = 2,
        actor_weight_decay: Optional[float] = None,
        actor_tau: float = 0.001,
        actor_dropout_rate: Optional[float] = None,
        actor_layer_norm: bool = False,
        value_layer_norm: bool = False,
        cost_temperature: float = 3.0,
        reward_temperature: float = 3.0,
        T: int = 5,
        time_dim: int = 64,
        N: int = 64,
        M: int = 0,
        clip_sampler: bool = True,
        actor_objective: str = 'bc',
        critic_objective: str = 'expectile',
        critic_type: str = 'hj',
        actor_type: str = 'flow',
        sampling_method: str = 'ddpm',
        beta_schedule: str = 'vp',
        decay_steps: Optional[int] = int(2e6),
        extract_method: bool = False,
        cost_limit: float = 10.,
        env_max_steps: int = 1000,
        cost_ub: float = 200.,
        use_adamw: bool = False,
        wr_init: float = 5.0,
        wc_init: float = 5.0,
        eps_l_reward: float = 0.0,
        eps_l_reward_lower: float = -0.1,
        eps_l_cost: float = 0.6,
    ):

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key, safe_critic_key, safe_value_key = jax.random.split(rng, 6)
        actions = action_space.sample()
        observations = observation_space.sample()
        action_dim = action_space.shape[0]

        qc_thres = cost_limit * (1 - discount**env_max_steps) / (
            1 - discount) / env_max_steps
        
        preprocess_time_cls = partial(FourierFeatures,
                                      output_size=time_dim,
                                      learnable=True)

        cond_model_cls = partial(MLP,
                                hidden_dims=(128, 128),
                                activations=mish,
                                activate_final=False)
        
        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        if actor_architecture == 'mlp':
            base_model_cls = partial(MLP,
                                    hidden_dims=tuple(list(actor_hidden_dims) + [action_dim]),
                                    activations=mish,
                                    use_layer_norm=actor_layer_norm,
                                    activate_final=False)
            adv_r_cond_model_cls = partial(MLP,
                                hidden_dims=(128, 128),
                                activations=mish,
                                activate_final=False)
            adv_c_cond_model_cls = partial(MLP,
                                hidden_dims=(128, 128),
                                activations=mish,
                                activate_final=False)
            

            actor_def = AdvGuidedSingleCompDDPM(time_preprocess_cls=preprocess_time_cls,
                            cond_encoder_cls=cond_model_cls,
                            reverse_encoder_cls=base_model_cls,
                            r_cond_reverse_encoder_cls=base_model_cls,
                            c_cond_reverse_encoder_cls=base_model_cls,
                            adv_r_encoder_cls=adv_r_cond_model_cls,
                            adv_c_encoder_cls=adv_c_cond_model_cls)


        elif actor_architecture == 'ln_resnet':

            base_model_cls = partial(MLPResNet,
                                     use_layer_norm=actor_layer_norm,
                                     num_blocks=actor_num_blocks,
                                     dropout_rate=actor_dropout_rate,
                                     out_dim=action_dim,
                                     activations=mish)
            adv_r_cond_model_cls = partial(MLP,
                                hidden_dims=(128, 128),
                                activations=mish,
                                activate_final=False)
            adv_c_cond_model_cls = partial(MLP,
                                hidden_dims=(128, 128),
                                activations=mish,
                                activate_final=False)
            
            actor_def = AdvGuidedSingleCompDDPM(time_preprocess_cls=preprocess_time_cls,
                            cond_encoder_cls=cond_model_cls,
                            reverse_encoder_cls=base_model_cls,
                            r_cond_reverse_encoder_cls=base_model_cls,
                            c_cond_reverse_encoder_cls=base_model_cls,
                            adv_r_encoder_cls=adv_r_cond_model_cls,
                            adv_c_encoder_cls=adv_c_cond_model_cls)


        else:
            raise ValueError(f'Invalid actor architecture: {actor_architecture}')
        
        time = jnp.zeros((1, 1))
        adv_r = jnp.zeros((1,1))
        adv_c = jnp.zeros((1,1))
        observations = jnp.expand_dims(observations, axis = 0)
        actions = jnp.expand_dims(actions, axis = 0)
        actor_params = actor_def.init(actor_key, observations, adv_r, adv_c, actions,
                                        time)['params']

        score_model = TrainState.create(apply_fn=actor_def.apply,
                                        params=actor_params,
                                        tx=optax.adamw(learning_rate=actor_lr, 
                                                       weight_decay=actor_weight_decay if actor_weight_decay is not None else 0.0,
                                                       mask=get_weight_decay_mask,))
        
        target_score_model = TrainState.create(apply_fn=actor_def.apply,
                                               params=actor_params,
                                               tx=optax.GradientTransformation(
                                                    lambda _: None, lambda _: None))
        
        # adv_return_predictor_base_cls = partial(MLP, hidden_dims=tuple(list(critic_hidden_dims) + [1]), activations=jnp.tanh, activate_final=True)
        # adv_c_cond_model_cls_new = partial(MLP,
        #                         hidden_dims=(128, 128),
        #                         activations=mish,
        #                         activate_final=False)
        # adv_return_predictor_def = AdvReturnPredictor(adv_c_encoder_cls=adv_c_cond_model_cls_new, adv_r_predictor_cls=adv_return_predictor_base_cls)
        # adv_return_predictor_params = adv_return_predictor_def.init(adv_return_predictor_key, observations, adv_c)["params"]
        # adv_return_predictor_optimiser = optax.adam(learning_rate=critic_lr)
        # adv_return_predictor = TrainState.create(
        #     apply_fn=adv_return_predictor_def.apply, params=adv_return_predictor_params, tx=adv_return_predictor_optimiser
        # )

        critic_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True)
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        if use_adamw:
            critic_optimiser = optax.adamw(learning_rate=critic_lr, weight_decay=1e-2)
        else:
            critic_optimiser = optax.adam(learning_rate=critic_lr)
        critic = TrainState.create(
            apply_fn=critic_def.apply, params=critic_params, tx=critic_optimiser
        )
        target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        if critic_type == 'qc':
            critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
            critic_def = Ensemble(critic_cls, num=num_qs)

            # critic_cls = partial(Relu_StateActionValue, base_cls=critic_base_cls)
            # critic_def = Ensemble(critic_cls, num=num_qs)

        safe_critic_params = critic_def.init(safe_critic_key, observations, actions)["params"]
        safe_critic = TrainState.create(
            apply_fn=critic_def.apply, params=safe_critic_params, tx=critic_optimiser
        )
        safe_target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=safe_critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )


        value_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True, use_layer_norm=value_layer_norm)
        value_def = StateValue(base_cls=value_base_cls)
        value_params = value_def.init(value_key, observations)["params"]
        if use_adamw:
            value_optimiser = optax.adamw(learning_rate=value_lr, weight_decay=1e-4)
        else:
            value_optimiser = optax.adam(learning_rate=value_lr)

        value = TrainState.create(apply_fn=value_def.apply,
                                  params=value_params,
                                  tx=value_optimiser)

        if critic_type == 'qc':
            value_def = StateValue(base_cls=value_base_cls)
            # value_def = Relu_StateValue(base_cls=value_base_cls)

        safe_value_params = value_def.init(safe_value_key, observations)["params"]

        safe_value = TrainState.create(apply_fn=value_def.apply,
                                  params=safe_value_params,
                                  tx=value_optimiser)

        if beta_schedule == 'cosine':
            betas = jnp.array(cosine_beta_schedule(T))
        elif beta_schedule == 'linear':
            betas = jnp.linspace(1e-4, 2e-2, T)
        elif beta_schedule == 'vp':
            betas = jnp.array(vp_beta_schedule(T))
        else:
            raise ValueError(f'Invalid beta schedule: {beta_schedule}')

        alphas = 1 - betas
        alpha_hat = jnp.array([jnp.prod(alphas[:i + 1]) for i in range(T)])

        return cls(
            actor=None, # Base class attribute
            score_model=score_model,
            # adv_return_predictor=adv_return_predictor,
            target_score_model=target_score_model,
            critic=critic,
            target_critic=target_critic,
            value=value,
            safe_critic=safe_critic,
            safe_target_critic=safe_target_critic,
            safe_value=safe_value,
            tau=tau,
            discount=discount,
            rng=rng,
            betas=betas,
            alpha_hats=alpha_hat,
            act_dim=action_dim,
            T=T,
            N=N,
            M=M,
            alphas=alphas,
            ddpm_temperature=ddpm_temperature,
            actor_tau=actor_tau,
            actor_objective=actor_objective,
            sampling_method=sampling_method,
            critic_objective=critic_objective,
            critic_type=critic_type,
            actor_type=actor_type,
            critic_hyperparam=critic_hyperparam,
            cost_critic_hyperparam=cost_critic_hyperparam,
            reward_advantage_predictor_hyperparam=reward_advantage_predictor_hyperparam,
            reverse_cost_temp=reverse_cost_temp,
            reverse_reward_temp=reverse_reward_temp,
            clip_sampler=clip_sampler,
            cost_temperature=cost_temperature,
            reward_temperature=reward_temperature,
            extract_method=extract_method,
            qc_thres=qc_thres,
            cost_ub=cost_ub,
            wr_init=wr_init,
            wc_init=wc_init,
            eps_l_reward=eps_l_reward,
            eps_l_reward_lower=eps_l_reward_lower,
            eps_l_cost=eps_l_cost,
        )

    def update_v(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch["observations"],
            batch["actions"],
        )
        q = qs.min(axis=0)

        def value_loss_fn(value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            v = agent.value.apply_fn({"params": value_params}, batch["observations"])

            if agent.critic_objective == 'expectile':
                value_loss = expectile_loss(q - v, agent.critic_hyperparam).mean()
            else:
                raise ValueError(f'Invalid critic objective: {agent.critic_objective}')

            return value_loss, {"value_loss": value_loss, "v": v.mean()}

        grads, info = jax.grad(value_loss_fn, has_aux=True)(agent.value.params)
        value = agent.value.apply_gradients(grads=grads)

        agent = agent.replace(value=value)

        return agent, info
    
    def update_q(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        next_v = agent.value.apply_fn(
            {"params": agent.value.params}, batch["next_observations"]
        )

        target_q = batch["rewards"] + agent.discount * batch["masks"] * next_v

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = agent.critic.apply_fn(
                {"params": critic_params}, batch["observations"], batch["actions"]
            )
            critic_loss = ((qs - target_q) ** 2).mean()

            return critic_loss, {
                "critic_loss": critic_loss,
                "q": qs.mean(),
            }

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(agent.critic.params)
        critic = agent.critic.apply_gradients(grads=grads)

        agent = agent.replace(critic=critic)

        target_critic_params = optax.incremental_update(
            critic.params, agent.target_critic.params, agent.tau
        )
        target_critic = agent.target_critic.replace(params=target_critic_params)

        new_agent = agent.replace(critic=critic, target_critic=target_critic)
        return new_agent, info
    
    def update_vc(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        qcs = agent.safe_target_critic.apply_fn(
            {"params": agent.safe_target_critic.params},
            batch["observations"],
            batch["actions"],
        )
        qc = qcs.max(axis=0)

        def safe_value_loss_fn(safe_value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            vc = agent.safe_value.apply_fn({"params": safe_value_params}, batch["observations"])

            if agent.critic_type == "hj":
                safe_value_loss = safe_expectile_loss(qc - vc, agent.cost_critic_hyperparam).mean()
            elif agent.critic_type == "qc":
                safe_value_loss = expectile_loss(qc - vc, 0.7).mean()
            else:
                assert False

            return safe_value_loss, {"safe_value_loss": safe_value_loss, "vc": vc.mean(), "vc_max": vc.max(), "vc_min": vc.min()}

        grads, info = jax.grad(safe_value_loss_fn, has_aux=True)(agent.safe_value.params)
        safe_value = agent.safe_value.apply_gradients(grads=grads)

        agent = agent.replace(safe_value=safe_value)

        return agent, info
    
    def update_qc(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        next_vc = agent.safe_value.apply_fn(
            {"params": agent.safe_value.params}, batch["next_observations"]
        )
        if agent.critic_type == "hj":
            qc_nonterminal = (1. - agent.discount) * batch["costs"] + agent.discount * jnp.maximum(batch["costs"], next_vc)
            target_qc = qc_nonterminal * batch["masks"] + batch["costs"] * (1 - batch["masks"])
        elif agent.critic_type == 'qc':
            target_qc = batch["costs"] + agent.discount * batch["masks"] * next_vc
        else:
            raise ValueError(f'Invalid critic type: {agent.critic_type}')

            
        def safe_critic_loss_fn(safe_critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qcs = agent.safe_critic.apply_fn(
                {"params": safe_critic_params}, batch["observations"], batch["actions"]
            )
            safe_critic_loss = ((qcs - target_qc) ** 2).mean()

            return safe_critic_loss, {
                "safe_critic_loss": safe_critic_loss,
                "qc": qcs.mean(),
                "qc_max": qcs.max(),
                "qc_min": qcs.min(),
                "costs": batch["costs"].mean()
            }

        grads, info = jax.grad(safe_critic_loss_fn, has_aux=True)(agent.safe_critic.params)
        safe_critic = agent.safe_critic.apply_gradients(grads=grads)

        agent = agent.replace(safe_critic=safe_critic)

        safe_target_critic_params = optax.incremental_update(
            safe_critic.params, agent.safe_target_critic.params, agent.tau
        )
        safe_target_critic = agent.safe_target_critic.replace(params=safe_target_critic_params)

        new_agent = agent.replace(safe_critic=safe_critic, safe_target_critic=safe_target_critic)
        return new_agent, info

    def update_actor(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        rng = agent.rng
        key, rng = jax.random.split(rng, 2)

        if agent.actor_type=="flow":
            x1 = batch['actions']
            t = jax.random.randint(key, (x1.shape[0],), 0, agent.T + 1).astype(jnp.float32) / agent.T
            key, rng = jax.random.split(rng, 2)
            x0 = jax.random.normal(key, x1.shape)
            tv = t[..., None]
            x_t = x0 * (1 - tv) + x1 * tv
            vel = (x1 - x0)
        elif agent.sampling_method == 'dpm_solver-1':
            # continuous time SDE
            eps = 1e-3
            time = jax.random.uniform(key, (batch['actions'].shape[0], )) * (1. - eps) + eps
            key, rng = jax.random.split(rng, 2)
            noise_sample = jax.random.normal(key, (batch['actions'].shape[0], agent.act_dim))
            alpha_t, sigma_t = vp_sde_schedule(time)
            time = jnp.expand_dims(time, axis=1)
            noisy_actions = alpha_t[:, None] * batch['actions'] + sigma_t[:, None] * noise_sample
        elif agent.sampling_method == 'ddpm':
            # discrete time SDE
            time = jax.random.randint(key, (batch['actions'].shape[0], ), 0, agent.T)
            key, rng = jax.random.split(rng, 2)
            noise_sample = jax.random.normal(key, (batch['actions'].shape[0], agent.act_dim))
            
            alpha_hats = agent.alpha_hats[time]
            time = jnp.expand_dims(time, axis=1)
            alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
            alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
            noisy_actions = alpha_1 * batch['actions'] + alpha_2 * noise_sample
        else:
            raise ValueError(f'Invalid samplint method: {agent.sampling_method}')

        key, rng = jax.random.split(rng, 2)
        
        '''
        reward adv
        '''
        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch["observations"],
            batch["actions"],
        )

        q = qs.min(axis=0)

        v = agent.value.apply_fn(
            {"params": agent.value.params}, batch["observations"]
        )
        
        '''
        cost reward
        '''
        qcs = agent.safe_target_critic.apply_fn(
            {"params": agent.safe_target_critic.params},
            batch["observations"],
            batch["actions"],
        )

        qc = qcs.max(axis=0)

        vc = agent.safe_value.apply_fn(
                {"params": agent.safe_value.params}, batch["observations"]
            )
        
        weights = jnp.ones(qc.shape)

        eps = 0. if agent.critic_type != 'qc' else 0.
        
        unsafe_condition = jnp.where( vc >  0. - eps, 1, 0)
        safe_condition = jnp.where(vc <= 0. - eps, 1, 0) * jnp.where(qc<=0. - eps, 1, 0)

        # cost_exp_adv = jnp.exp((vc-qc) * agent.cost_temperature)
        # reward_exp_adv = jnp.exp((q - v) * agent.reward_temperature)
        
        # unsafe_weights = unsafe_condition * jnp.clip(cost_exp_adv, 0, agent.cost_ub) ## ignore vc >0, qc>vc
        # safe_weights = safe_condition * jnp.clip(reward_exp_adv, 0, 100)
        
        # reward_adv_weights = unsafe_weights + safe_weights
        
        cost_tanh_adv = jnp.tanh((vc - qc) * agent.reverse_cost_temp)
        reward_tanh_adv = jnp.tanh((q - v) * agent.reverse_reward_temp)

        if agent.critic_type == "hj":
            safe_quantile = jnp.quantile(cost_tanh_adv, agent.eps_l_cost)
            unsafe_quantile = jnp.maximum(jnp.quantile(cost_tanh_adv, 0.8), 0.0)
            # cost_adv_classifier = safe_condition * jnp.where(cost_adv > safe_quantile, 1, 0) + unsafe_condition * jnp.where(cost_adv > unsafe_quantile, 1, 0)
            cost_adv_classifier = jnp.where(
                safe_condition,
                jnp.where(cost_tanh_adv > safe_quantile, 1, 0),
                jnp.where(unsafe_condition * (cost_tanh_adv > unsafe_quantile), 1, 0),
            )

            cost_safe_positive_ratio = ((safe_condition * jnp.where(cost_tanh_adv > safe_quantile, 1, 0))!=0).sum()/safe_condition.shape[0]
            cost_unsafe_positive_ratio = ((unsafe_condition * jnp.where(cost_tanh_adv > unsafe_quantile, 1, 0))!=0).sum()/unsafe_condition.shape[0]

        elif agent.critic_type == "qc":
            # safe_quantile = jnp.maximum(jnp.quantile(cost_tanh_adv, agent.eps_l_cost), 0.0)
            safe_quantile = agent.eps_l_cost
            cost_adv_classifier = jnp.where(cost_tanh_adv > safe_quantile, 1, 0)
            cost_safe_positive_ratio = 0.0
            cost_unsafe_positive_ratio = 0.0
        else:
            assert False
        reward_adv_classifier = jnp.where(reward_tanh_adv > agent.eps_l_reward, 1.0, jnp.where(reward_tanh_adv > agent.eps_l_reward_lower, 0.1, 0.0))

        total_classifier = cost_adv_classifier * reward_adv_classifier
        
        def actor_loss_fn(
                score_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            eps, eps_cond = agent.score_model.apply_fn({'params': score_model_params},
                                    batch['observations'],
                                    reward_adv_classifier,
                                    cost_adv_classifier,
                                    x_t,
                                    t,
                                    rngs={'dropout': key},
                                    training=True)
            uncond_loss = ((eps - vel) ** 2).mean(axis = -1).mean()
            cond_loss = (((eps_cond - vel) ** 2).mean(axis = -1) * total_classifier).mean()
            actor_loss =  uncond_loss + cond_loss
            
            return actor_loss, {'actor_loss': actor_loss, 'uncond_loss':uncond_loss, 'cond_loss':cond_loss, 'reward_positive_ratio' : (reward_adv_classifier!=0).sum()/reward_adv_classifier.shape[0], 'total_positive_ratio' : (total_classifier!=0).sum()/total_classifier.shape[0], 'cost_positive_ratio' : (cost_adv_classifier!=0).sum()/cost_adv_classifier.shape[0], 'cost_safe_positive_ratio' : cost_safe_positive_ratio, 'cost_unsafe_positive_ratio' : cost_unsafe_positive_ratio}
            
        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.score_model.params)
        score_model = agent.score_model.apply_gradients(grads=grads)

        agent = agent.replace(score_model=score_model)

        target_score_params = optax.incremental_update(
            score_model.params, agent.target_score_model.params, agent.actor_tau
        )

        target_score_model = agent.target_score_model.replace(params=target_score_params)
        new_agent = agent.replace(score_model=score_model, target_score_model=target_score_model, rng=rng)

        return new_agent, info

    def eval_actions(self, observations: jnp.ndarray):
        rng = self.rng

        if len(observations.shape) == 1:
            observations = jnp.expand_dims(observations, axis=0)
        batch_size = observations.shape[0]
        observations = jax.device_put(observations)
        observations = jnp.expand_dims(observations, axis = 0).repeat(self.N, axis = 0)
        observations = observations.reshape((batch_size * self.N, -1))
        # wr = jnp.where(vc > 0, 0, self.wr_init).reshape((batch_size * self.N, -1))
        w = jnp.ones((batch_size * self.N, 1))
        # wr = self.wr_init * jnp.ones((batch_size * self.N, 1))
        # wc = jnp.where(vc > 0, 10, self.wc_init).reshape((batch_size * self.N, -1))
        adv_c_classifier = jnp.ones((batch_size * self.N, 1))
        adv_r_classifier = jnp.ones((batch_size * self.N, 1))

        score_params = self.target_score_model.params
        if self.actor_type=="flow":
            actions, rng = adv_cond_single_comp_flow_sampler(self.score_model.apply_fn, score_params, self.T, rng, self.act_dim, observations, adv_r_classifier, adv_c_classifier, w, self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M, self.clip_sampler)
        else:
            actions, rng = adv_cond_single_comp_ddpm_sampler(self.score_model.apply_fn, score_params, self.T, rng, self.act_dim, observations, adv_r_classifier, adv_c_classifier, w, self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M, self.clip_sampler)
        
        rng, key = jax.random.split(rng, 2)
        qcs = compute_safe_q(self.safe_target_critic.apply_fn, self.safe_target_critic.params, observations, actions).reshape((self.N, batch_size, -1))

        # if self.critic_type == "qc":
        #     qcs = qcs - self.qc_thres
        
        idx = jnp.argmin(jnp.squeeze(qcs, axis=-1), axis=0)  # shape (b,)
        b_idx = jnp.arange(batch_size)
        actions = actions.reshape((self.N, batch_size, -1))
        action = actions[idx, b_idx]
           

        new_rng = rng

        return np.array(action.squeeze()), self.replace(rng=new_rng)
    
    def eval_actions_given_w(self, observations, w):
        rng = self.rng

        # assert len(observations.shape) == 1
        if len(observations.shape) == 1:
            observations = jnp.expand_dims(observations, axis=0)
        batch_size = observations.shape[0]
        observations = jax.device_put(observations)
        observations = jnp.expand_dims(observations, axis = 0).repeat(self.N, axis = 0)
        # print(observations.shape)
        observations = observations.reshape((batch_size * self.N, -1))
        vc = self.safe_value.apply_fn(
            {"params": self.safe_value.params}, observations
        )
        # wr = jnp.where(vc > 0, 0, wr).reshape((batch_size * self.N, -1))
        w = w * jnp.ones((batch_size * self.N, 1))
        adv_c_classifier = jnp.ones((batch_size * self.N, 1))
        adv_r_classifier = jnp.ones((batch_size * self.N, 1))

        score_params = self.target_score_model.params
        if self.actor_type=="flow":
            actions, rng = adv_cond_single_comp_flow_sampler(self.score_model.apply_fn, score_params, self.T, rng, self.act_dim, observations, adv_r_classifier, adv_c_classifier, w, self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M, self.clip_sampler)
        else:
            actions, rng = adv_cond_single_comp_ddpm_sampler(self.score_model.apply_fn, score_params, self.T, rng, self.act_dim, observations, adv_r_classifier, adv_c_classifier, w, self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M, self.clip_sampler)
        
        
        rng, key = jax.random.split(rng, 2)
        qcs = compute_safe_q(self.safe_target_critic.apply_fn, self.safe_target_critic.params, observations, actions).reshape((self.N, batch_size, -1))

        # if self.critic_type == "qc":
        #     qcs = qcs - self.qc_thres
        idx = jnp.argmin(jnp.squeeze(qcs, axis=-1), axis=0)  # shape (b,)
        b_idx = jnp.arange(batch_size)
        actions = actions.reshape((self.N, batch_size, -1))
        action = actions[idx, b_idx]
        new_rng = rng

        return np.array(action.squeeze()), self.replace(rng=new_rng)
    
    def eval_actions_rs(self, observations, w):
        rng = self.rng

        # assert len(observations.shape) == 1
        if len(observations.shape) == 1:
            observations = jnp.expand_dims(observations, axis=0)

        w_num = w.shape[0]
        
        batch_size = observations.shape[0]
        observations = jax.device_put(observations)
        observations = jnp.expand_dims(observations, axis = 0).repeat(self.N * w_num, axis = 0)
        # print(observations.shape)
        observations = observations.reshape((batch_size * self.N * w_num, -1))
        # vc = self.safe_value.apply_fn(
        #     {"params": self.safe_value.params}, observations
        # )
        # wr = jnp.where(vc > 0, 0, wr).reshape((batch_size * self.N, -1))
        # wr = wr * jnp.ones((batch_size * self.N, 1))
        # wc = wc * jnp.ones((batch_size * self.N, 1))
        w = w.repeat(self.N, axis=0)
        w = jnp.expand_dims(w, axis=1).repeat(batch_size, axis=1)
        w = w.reshape((batch_size * self.N * w_num, 1))
        adv_c_classifier = jnp.ones((batch_size * self.N * w_num, 1))
        adv_r_classifier = jnp.ones((batch_size * self.N * w_num, 1))

        score_params = self.target_score_model.params
        if self.actor_type == "flow":
            actions, rng = adv_cond_single_comp_flow_sampler(self.score_model.apply_fn, score_params, self.T, rng, self.act_dim, observations, adv_r_classifier, adv_c_classifier, w, self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M, self.clip_sampler)
        else:
            actions, rng = adv_cond_single_comp_ddpm_sampler(self.score_model.apply_fn, score_params, self.T, rng, self.act_dim, observations, adv_r_classifier, adv_c_classifier, w, self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M, self.clip_sampler)
        
        
        rng, key = jax.random.split(rng, 2)
        qcs = compute_safe_q(self.safe_target_critic.apply_fn, self.safe_target_critic.params, observations, actions).reshape((self.N * w_num, batch_size, -1))
        qs = compute_q(self.target_critic.apply_fn, self.target_critic.params, observations, actions).reshape((self.N * w_num, batch_size, -1))
        vc = self.safe_value.apply_fn(
            {"params": self.safe_value.params}, observations
        ).reshape((self.N * w_num, batch_size, -1))
        threshold = - 0.95
        any_nonpos = jnp.any(qcs <= threshold, axis=0)
        masked_qs = jnp.where(qcs <= threshold, qs, -jnp.inf) # shape (N, bs, D)
        idxs_from_qs = jnp.argmax(masked_qs, axis=0)  # shape (bs, D), int
        idxs_from_qcs = jnp.argmin(qcs, axis=0)      # shape (bs, D), int
        chosen_idxs = jnp.where(any_nonpos, idxs_from_qs, idxs_from_qcs)  # shape (bs, D)
        chosen_idxs_exp = chosen_idxs[None, ...]      # shape (1, bs, D)
        actions = actions.reshape((self.N * w_num, batch_size, -1))
        action = jnp.take_along_axis(actions, chosen_idxs_exp, axis=0).squeeze(0)  # shape (bs, D)
        new_rng = rng

        return np.array(action.squeeze()), self.replace(rng=new_rng)
    
    
    def actor_loss_no_grad(agent, batch: DatasetDict):
        rng = agent.rng
        key, rng = jax.random.split(rng, 2)
        time = jax.random.randint(key, (batch['actions'].shape[0], ), 0, agent.T)
        key, rng = jax.random.split(rng, 2)
        noise_sample = jax.random.normal(
            key, (batch['actions'].shape[0], agent.act_dim))
        
        alpha_hats = agent.alpha_hats[time]
        time = jnp.expand_dims(time, axis=1)
        alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
        alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
        noisy_actions = alpha_1 * batch['actions'] + alpha_2 * noise_sample

        key, rng = jax.random.split(rng, 2)


        def actor_loss_fn(score_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            eps_pred = agent.score_model.apply_fn({'params': score_model_params},
                                       batch['observations'],
                                       noisy_actions,
                                       time,
                                       training=False)
            
            actor_loss = (((eps_pred - noise_sample) ** 2).sum(axis = -1)).mean()

            return actor_loss, {'actor_loss': actor_loss}

        _, info = jax.grad(actor_loss_fn, has_aux=True)(agent.score_model.params)
        new_agent = agent.replace(rng=rng)

        return new_agent, info
    

    @jax.jit
    def actor_update(self, batch: DatasetDict):
        new_agent = self
        new_agent, actor_info = new_agent.update_actor(batch)
        return new_agent, actor_info
    
    @jax.jit
    def eval_loss(self, batch: DatasetDict):
        assert False
        new_agent = self
        new_agent, actor_info = new_agent.actor_loss_no_grad(batch)
        return new_agent, actor_info
    
    @jax.jit
    def critic_update(self, batch: DatasetDict):
        def slice(x):
            return x[:256]

        new_agent = self
        
        mini_batch = jax.tree_util.tree_map(slice, batch)
        new_agent, critic_info = new_agent.update_v(mini_batch)
        new_agent, value_info = new_agent.update_q(mini_batch)

        return new_agent, {**critic_info, **value_info}

    @jax.jit
    def update(self, batch: DatasetDict, real_batch=None):
        new_agent = self
        batch_size = int(batch['observations'].shape[0]/2)

        def first_half(x):
            return x[:batch_size]
        
        def second_half(x):
            return x[batch_size:]
        
        if real_batch is not None:
            first_batch = jax.tree_util.tree_map(first_half, real_batch)
            second_batch = jax.tree_util.tree_map(second_half, real_batch)
        else:
            first_batch = jax.tree_util.tree_map(first_half, batch)
            second_batch = jax.tree_util.tree_map(second_half, batch)

        new_agent, _ = new_agent.update_actor(first_batch)
        new_agent, actor_info = new_agent.update_actor(second_batch)

        def slice(x):
            return x[:256]
        
        mini_batch = jax.tree_util.tree_map(slice, batch)
        if real_batch is not None:
            mini_batch_reward = jax.tree_util.tree_map(slice, real_batch)
            new_agent, critic_info = new_agent.update_v(mini_batch_reward)
            new_agent, value_info = new_agent.update_q(mini_batch_reward)
        else:
            new_agent, critic_info = new_agent.update_v(mini_batch)
            new_agent, value_info = new_agent.update_q(mini_batch)
        new_agent, safe_critic_info = new_agent.update_vc(mini_batch)
        new_agent, safe_value_info = new_agent.update_qc(mini_batch)

        return new_agent, {**actor_info, **critic_info, **value_info, **safe_critic_info, **safe_value_info}
    
    def save(self, modeldir, save_time):
        file_name = 'model' + str(save_time) + '.pickle'
        state_dict = flax.serialization.to_state_dict(self)
        pickle.dump(state_dict, open(os.path.join(modeldir, file_name), 'wb'))

    def load(self, model_location):
        pkl_file = pickle.load(open(model_location, 'rb'))
        new_agent = flax.serialization.from_state_dict(target=self, state=pkl_file)
        return new_agent