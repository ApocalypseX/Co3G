from typing import Callable, Optional, Sequence
import flax.linen as nn
import jax.numpy as jnp
import flax

default_init = nn.initializers.xavier_uniform

def get_weight_decay_mask(params):
    flattened_params = flax.traverse_util.flatten_dict(
        flax.core.frozen_dict.unfreeze(params))

    def decay(k, v):
        if any([(key == 'bias' or 'Input' in key or 'Output' in key)
                for key in k]):
            return False
        else:
            return True

    return flax.core.frozen_dict.freeze(
        flax.traverse_util.unflatten_dict(
            {k: decay(k, v)
             for k, v in flattened_params.items()}))

from typing import NamedTuple

class PIDState(NamedTuple):
    error_old: jnp.ndarray
    error_integral: jnp.ndarray

class PIDParams(NamedTuple):
    KP: jnp.ndarray
    KI: jnp.ndarray
    KD: jnp.ndarray
    thres: jnp.ndarray

# class LagrangianPIDController:
#     """
#     Lagrangian multiplier controller (JAX, imperative style)

#     Args:
#         KP (float): proportional gain
#         KI (float): integral gain
#         KD (float): derivative gain
#         thres (float or array-like): setpoint
#     """
#     def __init__(self, KP, KI, KD, thres):
#         self.KP = float(KP)
#         self.KI = float(KI)
#         self.KD = float(KD)
#         # 将 thres 保持为标量或数组（依输入而定）
#         self.thres = jnp.array(thres)
#         # 状态（标量）
#         self.error_old = 0.0
#         self.error_integral = 0.0

#     def control(self, qc):
#         """
#         qc: array-like with shape [batch,] (或任何可求 mean 的数组)
#         返回: scalar (jnp.float32)
#         """
#         qc = jnp.array(qc)
#         # mean 会返回 0-d array (标量)
#         error_new = jnp.mean(qc - self.thres)
#         # ReLU for positive diff only
#         error_diff = jnp.maximum(0.0, error_new - self.error_old)
#         # accumulate integral but apply relu and then mean (与原逻辑一致)
#         self.error_integral = float(jnp.mean(jnp.maximum(0.0, self.error_integral + error_new)))
#         self.error_old = float(error_new)

#         multiplier = jnp.maximum(
#             0.0,
#             self.KP * jnp.maximum(0.0, error_new) +
#             self.KI * self.error_integral +
#             self.KD * error_diff
#         )
#         return jnp.mean(multiplier)

class GuidanceActorWoNoise(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    scale_final: Optional[float] = None
    guidance_bound: float = 5.0
    noise_bound: float = 1.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            if i + 1 == len(self.hidden_dims) and self.scale_final is not None:
                x = nn.Dense(size, kernel_init=default_init(self.scale_final))(x)
            else:
                x = nn.Dense(size, kernel_init=default_init())(x)
            x = nn.LayerNorm()(x)
            x = self.activations(x)
        wr = nn.Dense(1, kernel_init=default_init())(x)
        wr = (jnp.tanh(wr) + 1) / 2 * self.guidance_bound
        wc = nn.Dense(1, kernel_init=default_init())(x)
        wc = (jnp.tanh(wc) + 1) / 2 * self.guidance_bound
        # noise = nn.Dense(self.action_dim-2, kernel_init=default_init())(x)
        # noise = jnp.tanh(noise) * self.noise_bound
        # output = jnp.concatenate([wc, wr, noise], axis=-1)
        output = jnp.concatenate([wc, wr], axis=-1)
        return output

class GuidanceActorWithNoise(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    scale_final: Optional[float] = None
    guidance_bound: float = 5.0
    noise_bound: float = 1.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            if i + 1 == len(self.hidden_dims) and self.scale_final is not None:
                x = nn.Dense(size, kernel_init=default_init(self.scale_final))(x)
            else:
                x = nn.Dense(size, kernel_init=default_init())(x)
            x = nn.LayerNorm()(x)
            x = self.activations(x)
        wr = nn.Dense(1, kernel_init=default_init())(x)
        wr = (jnp.tanh(wr) + 1) / 2 * self.guidance_bound
        wc = nn.Dense(1, kernel_init=default_init())(x)
        wc = (jnp.tanh(wc) + 1) / 2 * self.guidance_bound
        noise = nn.Dense(self.action_dim-2, kernel_init=default_init())(x)
        noise = jnp.tanh(noise) * self.noise_bound
        output = jnp.concatenate([wc, wr, noise], axis=-1)
        # output = jnp.concatenate([wc, wr], axis=-1)
        return output

class GuidanceActorOnlyNoise(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    scale_final: Optional[float] = None
    guidance_bound: float = 5.0
    noise_bound: float = 1.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            if i + 1 == len(self.hidden_dims) and self.scale_final is not None:
                x = nn.Dense(size, kernel_init=default_init(self.scale_final))(x)
            else:
                x = nn.Dense(size, kernel_init=default_init())(x)
            x = nn.LayerNorm()(x)
            x = self.activations(x)
        noise = nn.Dense(self.action_dim, kernel_init=default_init())(x)
        noise = jnp.tanh(noise) * self.noise_bound
        output = noise
        # output = jnp.concatenate([wc, wr], axis=-1)
        return output

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    use_layer_norm: bool = False
    scale_final: Optional[float] = None
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            if i + 1 == len(self.hidden_dims) and self.scale_final is not None:
                x = nn.Dense(size, kernel_init=default_init(self.scale_final))(x)
            else:
                x = nn.Dense(size, kernel_init=default_init())(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
                x = self.activations(x)
        return x