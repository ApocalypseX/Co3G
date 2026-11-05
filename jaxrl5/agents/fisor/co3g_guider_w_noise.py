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
from jaxrl5.agents import FISOR, Co3G
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.networks import MLP, Ensemble, StateActionValue, StateValue, Relu_StateActionValue, Relu_StateValue, AdvCondDDPM, FourierFeatures, cosine_beta_schedule, adv_cond_ddpm_sampler, MLPResNet, get_weight_decay_mask, vp_beta_schedule
from jaxrl5.networks.diffusion import adv_cond_dpm_solver_sampler_1st, vp_sde_schedule, AdvReturnPredictor, adv_cond_comp_ddpm_sampler, AdvCondCompDDPM, adv_cond_comp_ddpm_sampler_given_noise, adv_cond_comp_ddpm_sampler_return_noise, adv_cond_comp_flow_sampler_return_noise, adv_cond_comp_flow_sampler_given_noise
from jaxrl5.networks.mlp import GuidanceActorWithNoise, PIDParams, PIDState

def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

def safe_expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff < 0, expectile, (1 - expectile))
    return weight * (diff**2)

def pid_control_step(params: PIDParams, state: PIDState, qc: jnp.ndarray):
    error_new = jnp.mean(qc - params.thres)
    # error_new = jnp.quantile(qc - params.thres, 0.8)
    error_diff = jnp.maximum(0.0, error_new - state.error_old)
    error_integral = jnp.mean(jnp.maximum(0.0, state.error_integral + error_new))
    new_state = PIDState(error_old=error_new, error_integral=error_integral)

    multiplier = jnp.maximum(
        0.0,
        params.KP * jnp.maximum(0.0, error_new) +
        params.KI * error_integral +
        params.KD * error_diff
    )
    return multiplier, new_state

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

class Co3GGuiderWithNoise(Agent):
    diffusion_policy: Co3G
    actor: TrainState
    target_actor: TrainState
    critic: TrainState
    critic_a: TrainState
    target_critic_a: TrainState
    safe_critic: TrainState
    safe_critic_a: TrainState
    target_safe_critic_a: TrainState
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
    N: int = struct.field(pytree_node=False)
    M: int = struct.field(pytree_node=False) #How many repeat last steps
    clip_sampler: bool = struct.field(pytree_node=False)
    use_org_respo: bool = struct.field(pytree_node=False)
    use_org_cpq: bool = struct.field(pytree_node=False)
    distill_org_qr: bool = struct.field(pytree_node=False)
    distill_org_qc: bool = struct.field(pytree_node=False)
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
    eps_l_cost: float
    guidance_bound: float
    value_ensemble_weight: float
    actor_loss_type: str = struct.field(pytree_node=False)
    pid_params: PIDParams
    pid_state: PIDState
    cpq_quantile: float

    @classmethod
    def create(
        cls,
        diffusion_policy: Co3G,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        actor_architecture: str = 'mlp',
        actor_lr: Union[float, optax.Schedule] = 1e-4,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_hidden_dims: Sequence[int] = (512, 512, 512, 512),
        actor_hidden_dims: Sequence[int] = (512, 512, 512, 512),
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
        eps_l_reward: float = 0.8,
        eps_l_cost: float = 0.8,
        guidance_bound: float = 5.0,
        noise_bound: float = 1.0,
        value_ensemble_weight: float = 0.75,
        actor_update_freq: int = 2,
        actor_loss_type: str = 'cpq',
        pid: Sequence[float] = (0.1, 0.003, 0.001),
        use_org_respo: bool = False,
        use_org_cpq: bool = False,
        distill_org_qr: bool = False,
        distill_org_qc: bool = False,
        cpq_quantile: float = 0.0,
    ):

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, critic_a_key, safe_critic_key, safe_critic_a_key = jax.random.split(rng, 6)
        actions = action_space.sample()
        actions_org = action_space.sample()
        actions_org = jnp.expand_dims(actions_org, axis = 0)
        actions = jnp.expand_dims(actions, axis = 0)
        w_r = jnp.zeros((1, 1))
        w_c = jnp.zeros((1, 1))
        actions = jnp.concatenate([w_c, w_r, actions], axis=-1)

        observations = observation_space.sample()
        action_dim = 2 + action_space.shape[0]
        # action_dim = 2
        # action_dim = action_space.shape[0]

        qc_thres = cost_limit * (1 - discount**env_max_steps) / (
            1 - discount) / env_max_steps
        
        
        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        actor_def = GuidanceActorWithNoise(
            hidden_dims=actor_hidden_dims,
            action_dim=action_dim,
            guidance_bound=guidance_bound,
            noise_bound=noise_bound,
        )
        
        observations = jnp.expand_dims(observations, axis = 0)
        actor_params = actor_def.init(actor_key, observations)['params']

        actor = TrainState.create(apply_fn=actor_def.apply,
                                        params=actor_params,
                                        tx=optax.adamw(learning_rate=actor_lr, 
                                                       weight_decay=actor_weight_decay if actor_weight_decay is not None else 0.0,
                                                       mask=get_weight_decay_mask,))
        
        target_actor = TrainState.create(apply_fn=actor_def.apply,
                                        params=actor_params,
                                        tx=optax.GradientTransformation(
                                                    lambda _: None, lambda _: None))
        

        critic_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True, use_layer_norm=True)
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

        critic_a_params = critic_def.init(critic_a_key, observations, actions_org)["params"]
        # critic_a_params = diffusion_policy.critic.params
        critic_a_apply = critic_def.apply
        # critic_a_apply = diffusion_policy.critic.apply_fn
        critic_a = TrainState.create(
            apply_fn=critic_a_apply, params=critic_a_params, tx=critic_optimiser
        )
        target_critic_a = TrainState.create(
            apply_fn=critic_a_apply,
            params=critic_a_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        if critic_type == 'qc':
            critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
            critic_def = Ensemble(critic_cls, num=num_qs)

        safe_critic_params = critic_def.init(safe_critic_key, observations, actions)["params"]
        safe_critic = TrainState.create(
            apply_fn=critic_def.apply, params=safe_critic_params, tx=critic_optimiser
        )

        safe_critic_a_params = critic_def.init(safe_critic_a_key, observations, actions_org)["params"]
        # safe_critic_a_params = diffusion_policy.safe_critic.params
        safe_critic_a_apply = critic_def.apply
        # safe_critic_a_apply = diffusion_policy.safe_critic.apply_fn
        safe_critic_a = TrainState.create(
            apply_fn=safe_critic_a_apply, params=safe_critic_a_params, tx=critic_optimiser
        )
        target_safe_critic_a = TrainState.create(
            apply_fn=safe_critic_a_apply,
            params=safe_critic_a_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

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

        kp, ki, kd = pid
        pid_params = PIDParams(KP=kp, KI=ki, KD=kd, thres=0.0)
        pid_state = PIDState(error_old=0.0, error_integral=0.0)

        return cls(
            diffusion_policy=diffusion_policy,
            actor=actor, # Base class attribute
            target_actor=target_actor,
            critic=critic,
            critic_a=critic_a,
            target_critic_a=target_critic_a,
            safe_critic=safe_critic,
            safe_critic_a=safe_critic_a,
            target_safe_critic_a=target_safe_critic_a,
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
            use_org_respo=(use_org_respo and actor_loss_type=="respo"),
            use_org_cpq=(use_org_cpq and actor_loss_type=="cpq"),
            distill_org_qr=distill_org_qr,
            distill_org_qc=distill_org_qc,
            cost_temperature=cost_temperature,
            reward_temperature=reward_temperature,
            extract_method=extract_method,
            qc_thres=qc_thres,
            cost_ub=cost_ub,
            wr_init=wr_init,
            wc_init=wc_init,
            eps_l_reward=eps_l_reward,
            eps_l_cost=eps_l_cost,
            guidance_bound=guidance_bound,
            value_ensemble_weight=value_ensemble_weight,
            actor_loss_type=actor_loss_type,
            pid_params=pid_params,
            pid_state=pid_state,
            cpq_quantile=cpq_quantile
        )
    
    def update_q(agent, batch: DatasetDict, actions, wr, wc, noise) -> Tuple[Agent, Dict[str, float]]:
        if agent.distill_org_qr:
            target_q = agent.diffusion_policy.critic.apply_fn(
                {"params": agent.diffusion_policy.critic.params}, batch["observations"], actions
            ).min(axis=0)
        else:
            target_q = agent.target_critic_a.apply_fn(
                {"params": agent.target_critic_a.params}, batch["observations"], actions
            ).min(axis=0)

        new_actions = jnp.concatenate([wc, wr, noise], axis=-1)

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = agent.critic.apply_fn(
                {"params": critic_params}, batch["observations"], new_actions
            )
            critic_loss = ((qs - target_q) ** 2).mean()

            return critic_loss, {
                "critic_loss": critic_loss,
                "q": qs.mean(),
            }

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(agent.critic.params)
        critic = agent.critic.apply_gradients(grads=grads)

        new_agent = agent.replace(critic=critic)
        return new_agent, info
    
    def update_qa(agent, batch: DatasetDict, next_actions) -> Tuple[Agent, Dict[str, float]]:

        next_qa = agent.target_critic_a.apply_fn(
            {"params": agent.target_critic_a.params}, batch["next_observations"], next_actions,
        )
        next_qa = 0.75 * next_qa.min(axis=0) + 0.25 * next_qa.max(axis=0)
        target_qa = batch["rewards"] + agent.discount * batch["masks"] * next_qa

        def critic_a_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qas = agent.critic_a.apply_fn(
                {"params": critic_params}, batch["observations"], batch["actions"]
            )
            critic_a_loss = ((qas - target_qa) ** 2).mean()

            return critic_a_loss, {
                "critic_a_loss": critic_a_loss,
                "qa": qas.mean(),
            }

        grads, info = jax.grad(critic_a_loss_fn, has_aux=True)(agent.critic_a.params)
        critic_a = agent.critic_a.apply_gradients(grads=grads)
        agent = agent.replace(critic_a=critic_a)

        target_critic_a_params = optax.incremental_update(
            critic_a.params, agent.target_critic_a.params, agent.tau
        )
        target_critic_a = agent.target_critic_a.replace(params=target_critic_a_params)
        new_agent = agent.replace(critic_a=critic_a, target_critic_a=target_critic_a)

        return new_agent, info
    
    
    def update_qc(agent, batch: DatasetDict, actions, wr, wc, noise) -> Tuple[Agent, Dict[str, float]]:
        if agent.distill_org_qc:
            target_qc = agent.diffusion_policy.safe_critic.apply_fn(
                {"params": agent.diffusion_policy.safe_critic.params}, batch["observations"], actions
            ).max(axis=0)
        else:
            target_qc = agent.target_safe_critic_a.apply_fn(
                {"params": agent.target_safe_critic_a.params}, batch["observations"], actions
            ).max(axis=0)

        new_actions = jnp.concatenate([wc, wr, noise], axis=-1)
            
        def safe_critic_loss_fn(safe_critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qcs = agent.safe_critic.apply_fn(
                {"params": safe_critic_params}, batch["observations"], new_actions
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

        new_agent = agent.replace(safe_critic=safe_critic)
        return new_agent, info
    
    def update_qca(agent, batch: DatasetDict, next_actions) -> Tuple[Agent, Dict[str, float]]:

        next_qca = agent.target_safe_critic_a.apply_fn(
            {"params": agent.target_safe_critic_a.params}, batch["next_observations"], next_actions,
        )
        # next_qca = next_qca.max(axis=0)
        # 0.75
        next_qca = (1-agent.value_ensemble_weight) * next_qca.min(axis=0) + agent.value_ensemble_weight * next_qca.max(axis=0)
        if agent.use_org_respo:
            target_qca = batch["costs_org"] + agent.discount * batch["masks"] * next_qca
        else:
            next_qca = jnp.clip(next_qca, a_max=25.0)
            qca_nonterminal = (1. - agent.discount) * batch["costs"] + agent.discount * jnp.maximum(batch["costs"], next_qca)
            target_qca = qca_nonterminal * batch["masks"] + batch["costs"] * (1 - batch["masks"])

        def safe_critic_a_loss_fn(safe_critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qcas = agent.safe_critic_a.apply_fn(
                {"params": safe_critic_params}, batch["observations"], batch["actions"]
            )
            safe_critic_a_loss = ((qcas - target_qca) ** 2).mean()

            return safe_critic_a_loss, {
                "safe_critic_a_loss": safe_critic_a_loss,
                "qca": qcas.mean(),
                "qca_max": qcas.max(),
                "qca_min": qcas.min(),
            }

        grads, info = jax.grad(safe_critic_a_loss_fn, has_aux=True)(agent.safe_critic_a.params)
        safe_critic_a = agent.safe_critic_a.apply_gradients(grads=grads)
        agent = agent.replace(safe_critic_a=safe_critic_a)

        target_safe_critic_a_params = optax.incremental_update(
            safe_critic_a.params, agent.target_safe_critic_a.params, agent.tau
        )
        target_safe_critic_a = agent.target_safe_critic_a.replace(params=target_safe_critic_a_params)
        new_agent = agent.replace(safe_critic_a=safe_critic_a, target_safe_critic_a=target_safe_critic_a)

        return new_agent, info

    def update_actor(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:        
        vc = agent.diffusion_policy.safe_value.apply_fn(
                {"params": agent.diffusion_policy.safe_value.params}, batch["observations"]
            )

        eps = 0.
        
        unsafe_condition = jnp.where( vc >  0. - eps, 1, 0)
        safe_condition = jnp.where(vc <= 0. - eps, 1, 0)
        
        
        def actor_loss_fn(
                actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            new_action = agent.actor.apply_fn(
                {'params': actor_params},
                batch['observations'],
                training=True
            )
            q = agent.critic.apply_fn({"params": agent.critic.params}, batch["observations"], new_action).mean(axis = 0)
            # qc = agent.safe_critic.apply_fn({"params": agent.safe_critic.params}, batch["observations"], new_action).mean(axis = 0)
            qc = agent.safe_critic.apply_fn({"params": agent.safe_critic.params}, batch["observations"], new_action).mean(axis = 0)
            if agent.actor_loss_type == "cpq":
                threshold = jnp.where(agent.cpq_quantile == 0.0, 0.0, jnp.quantile(qc, agent.cpq_quantile))
                # if agent.cpq_quantile == 0.0:
                #     safe_action_mask = jnp.where(qc<=0, 1.0, 0.0)
                #     unsafe_action_mask = jnp.where(qc>0, 1.0, 0.0)
                # else:
                #     threshold = jnp.quantile(qc, agent.cpq_quantile)
                safe_action_mask = jnp.where(qc<=threshold, 1.0, 0.0)
                unsafe_action_mask = jnp.where(qc>threshold, 1.0, 0.0)
                if agent.use_org_cpq:
                    actor_loss = (unsafe_condition * qc -safe_condition * safe_action_mask * q).mean()
                else:
                    actor_loss = (unsafe_condition * qc -safe_condition * safe_action_mask * q + safe_condition * unsafe_action_mask * qc).mean()
                pid_state = agent.pid_state
            elif agent.actor_loss_type == "lagrange":
                multiplier, pid_state = jax.lax.stop_gradient(pid_control_step(agent.pid_params, agent.pid_state, safe_condition * qc))
                actor_loss = (unsafe_condition * qc + safe_condition * (qc * multiplier - q)).mean()
            elif agent.actor_loss_type == "respo":
                if agent.use_org_respo:
                    indicator = jnp.clip(vc, -1.0, 25.0)
                    indicator = (indicator + 1.0) / 26.0
                    multiplier, pid_state = jax.lax.stop_gradient(pid_control_step(agent.pid_params, agent.pid_state, qc*(1-indicator)))
                    actor_loss = (-q * (1-indicator) + qc * (multiplier * (1-indicator) + indicator)).mean()
                else:
                    indicator = jnp.clip(vc, -1.0, 0.0) + 1.0
                    multiplier, pid_state = jax.lax.stop_gradient(pid_control_step(agent.pid_params, agent.pid_state, safe_condition*qc*(1-indicator)))
                    actor_loss = (unsafe_condition * qc + safe_condition * (-q * (1-indicator) + qc * (multiplier * (1-indicator) + indicator))).mean()
            else:
                assert False
            if agent.actor_loss_type == "lagrange" or agent.actor_loss_type == "respo":
                return actor_loss, ({'actor_loss': actor_loss, "multiplier": multiplier}, pid_state)
            else:
                return actor_loss, ({'actor_loss': actor_loss}, pid_state)
            
        grads, (info, new_pid_state) = jax.grad(actor_loss_fn, has_aux=True)(agent.actor.params)
        actor = agent.actor.apply_gradients(grads=grads)

        agent = agent.replace(actor=actor)

        target_actor_params = optax.incremental_update(
            actor.params, agent.target_actor.params, agent.actor_tau
        )
        target_actor = agent.target_actor.replace(params=target_actor_params)
        new_agent = agent.replace(actor=actor, target_actor=target_actor, pid_state=new_pid_state)

        return new_agent, info
    
    def sample_actions(self, observations: jnp.ndarray):
        rng = self.rng

        batch_size = observations.shape[0]
        observations = jax.device_put(observations)
        # observations = jnp.expand_dims(observations, axis = 0).repeat(self.N, axis = 0)
        # observations = observations.reshape((batch_size * self.N, -1))

        key, rng = jax.random.split(rng, 2)
        wr = jax.random.uniform(key, shape=(batch_size, 1), minval=0.0, maxval=self.guidance_bound)
        # wr_expand = jnp.expand_dims(wr, axis = 0).repeat(self.N, axis = 0)
        # wr_expand = wr_expand.reshape((batch_size * self.N, -1))
        # wr = jax.device_put(wr)
        key, rng = jax.random.split(rng, 2)
        wc = jax.random.uniform(key, shape=(batch_size, 1), minval=0.0, maxval=self.guidance_bound)
        # wc_expand = jnp.expand_dims(wc, axis = 0).repeat(self.N, axis = 0)
        # wc_expand = wc_expand.reshape((batch_size * self.N, -1))
        # wc = jax.device_put(wc)
        adv_c_classifier = jnp.ones((batch_size, 1))
        adv_r_classifier = jnp.ones((batch_size, 1))

        score_params = self.diffusion_policy.score_model.params
        if self.actor_type == "flow":
            actions, noise, rng = adv_cond_comp_flow_sampler_return_noise(self.diffusion_policy.score_model.apply_fn, score_params, self.T, rng, self.act_dim-2, observations, adv_r_classifier, adv_c_classifier, wr, wc, self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M, self.clip_sampler)
        else:
            actions, noise, rng = adv_cond_comp_ddpm_sampler_return_noise(self.diffusion_policy.score_model.apply_fn, score_params, self.T, rng, self.act_dim-2, observations, adv_r_classifier, adv_c_classifier, wr, wc, self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M, self.clip_sampler)
        new_rng = rng

        return actions.squeeze(), wr, wc, noise, self.replace(rng=new_rng)

    def sample_actions_agent(self, observations: jnp.ndarray):
        rng = self.rng
        batch_size = observations.shape[0]
        observations = jax.device_put(observations)

        latent_action = self.actor.apply_fn(
            {"params": self.target_actor.params}, observations
        )

        wc = latent_action[:,:1]
        wr = latent_action[:,1:2]
        noise = latent_action[:, 2:]

        adv_c_classifier = jnp.ones((batch_size, 1))
        adv_r_classifier = jnp.ones((batch_size, 1))

        score_params = self.diffusion_policy.score_model.params
        if self.actor_type == "flow":
            actions, rng = adv_cond_comp_flow_sampler_given_noise(self.diffusion_policy.score_model.apply_fn, score_params, self.T, rng, self.act_dim-2, observations, adv_r_classifier, adv_c_classifier, wr, wc, self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M, self.clip_sampler, noise)
        else:
            actions, rng = adv_cond_comp_ddpm_sampler_given_noise(self.diffusion_policy.score_model.apply_fn, score_params, self.T, rng, self.act_dim-2, observations, adv_r_classifier, adv_c_classifier, wr, wc, self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M, self.clip_sampler, noise)
        new_rng = rng

        return actions.squeeze(), self.replace(rng=new_rng)

    def eval_actions(self, observations: jnp.ndarray):
        rng = self.rng

        # assert len(observations.shape) == 1
        if len(observations.shape) == 1:
            observations = jnp.expand_dims(observations, axis=0)
        batch_size = observations.shape[0]
        observations = jax.device_put(observations)
        # observations = jnp.expand_dims(observations, axis = 0).repeat(self.N, axis = 0)
        # observations = observations.reshape((batch_size * self.N, -1))

        latent_action = self.actor.apply_fn(
            {"params": self.actor.params}, observations
        )
        wc = latent_action[:,:1]
        wr = latent_action[:,1:2]
        noise = latent_action[:, 2:]

        adv_c_classifier = jnp.ones((batch_size, 1))
        adv_r_classifier = jnp.ones((batch_size, 1))

        score_params = self.diffusion_policy.score_model.params
        if self.actor_type == "flow":
            actions, rng = adv_cond_comp_flow_sampler_given_noise(self.diffusion_policy.score_model.apply_fn, score_params, self.T, rng, self.act_dim-2, observations, adv_r_classifier, adv_c_classifier, wr, wc, self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M, self.clip_sampler, noise)
        else:
            actions, rng = adv_cond_comp_ddpm_sampler_given_noise(self.diffusion_policy.score_model.apply_fn, score_params, self.T, rng, self.act_dim-2, observations, adv_r_classifier, adv_c_classifier, wr, wc, self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M, self.clip_sampler, noise)
        new_rng = rng
        return np.array(actions.squeeze()), self.replace(rng=new_rng)

    @jax.jit
    def update(self, batch: DatasetDict, update_actor: bool, real_batch=None):
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

        # if update_actor:
        #     new_agent, _ = new_agent.update_actor(first_batch)
        #     new_agent, actor_info = new_agent.update_actor(second_batch)
        # else:
        #     actor_info = {}
        def actor_update_fn(agent_and_batches):
            agent, first_batch, second_batch = agent_and_batches
            agent, _ = agent.update_actor(first_batch)
            agent, actor_info = agent.update_actor(second_batch)
            return agent, actor_info

        def actor_skip_fn(agent_and_batches):
            agent, _, _ = agent_and_batches
            if self.actor_loss_type == "lagrange" or self.actor_loss_type == "respo":
                out_info = {'multiplier': jnp.array(0.0), 'actor_loss': jnp.array(0.0)}
            else:
                out_info = {'actor_loss': jnp.array(0.0)}
            return agent, out_info

        new_agent, actor_info = jax.lax.cond(
            update_actor,
            actor_update_fn,
            actor_skip_fn,
            operand=(new_agent, first_batch, second_batch)
        )

        def slice(x):
            return x[:256]
        
        mini_batch = jax.tree_util.tree_map(slice, batch)
        actions, wr, wc, noise, new_agent = new_agent.sample_actions(mini_batch["observations"])
        next_actions, new_agent = new_agent.sample_actions_agent(mini_batch["next_observations"])

        new_agent, critic_a_info = new_agent.update_qa(mini_batch, next_actions)
        new_agent, safe_critic_a_info = new_agent.update_qca(mini_batch, next_actions)

        new_agent, value_info = new_agent.update_q(mini_batch, actions, wr, wc, noise)
        new_agent, safe_value_info = new_agent.update_qc(mini_batch, actions, wr, wc, noise)

        return new_agent, {**actor_info, **value_info, **safe_value_info, **critic_a_info, **safe_critic_a_info}
    
    def save(self, modeldir, save_time):
        file_name = 'model' + str(save_time) + '.pickle'
        state_dict = flax.serialization.to_state_dict(self)
        pickle.dump(state_dict, open(os.path.join(modeldir, file_name), 'wb'))

    def load(self, model_location):
        pkl_file = pickle.load(open(model_location, 'rb'))
        new_agent = flax.serialization.from_state_dict(target=self, state=pkl_file)
        return new_agent