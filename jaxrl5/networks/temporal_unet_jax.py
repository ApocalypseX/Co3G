# temporal_unet_flax.py
import math
from typing import Sequence, Optional, Tuple, List

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers
# from jaxrl5.networks import default_init
default_init = nn.initializers.xavier_uniform

def mish(x):
    return x * jnp.tanh(nn.softplus(x))


# class SinusoidalPosEmb(nn.Module):
#     dim: int

#     @nn.compact
#     def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
#         # x: (batch,) or (batch,1) scalar times
#         # output: (batch, dim)
#         half_dim = self.dim // 2
#         emb = math.log(10000) / (half_dim - 1)
#         # create vector shape (half_dim,)
#         freq = jnp.exp(jnp.arange(half_dim) * -emb)
#         # x shape ensure (batch,)
#         x_ = x.reshape(-1)
#         args = jnp.outer(x_, freq)  # (batch, half_dim)
#         emb_sin = jnp.sin(args)
#         emb_cos = jnp.cos(args)
#         emb = jnp.concatenate([emb_sin, emb_cos], axis=-1)
#         return emb

class SinusoidalPosEmb(nn.Module):
    dim: int

    def setup(self):
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        # precompute freq once and store as a python attribute (constant)
        self.freq = jnp.exp(jnp.arange(half_dim) * -emb)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # ensure x -> shape (batch,)
        x_ = jnp.reshape(x, (x.shape[0],))
        # (batch, half_dim) via outer product implemented as multiply + broadcast
        args = x_[:, None] * self.freq[None, :]  # broadcasting; cheaper than jnp.outer
        sin = jnp.sin(args)
        cos = jnp.cos(args)
        return jnp.concatenate([sin, cos], axis=-1)

# class Conv1dBlock(nn.Module):
#     out_channels: int
#     kernel_size: int = 3
#     mish_act: bool = True
#     n_groups: int = 8

#     @nn.compact
#     def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
#         """
#         x: (batch, length, channels)
#         returns: (batch, length, out_channels)
#         """
#         conv = nn.Conv(features=self.out_channels,
#                        kernel_size=(self.kernel_size,),
#                        padding='SAME',
#                        kernel_init=initializers.lecun_normal())
#         x = conv(x)
#         # GroupNorm operates on channels axis = -1
#         x = nn.GroupNorm(num_groups=self.n_groups)(x)
#         if self.mish_act:
#             x = mish(x)
#         else:
#             x = nn.silu(x)
#         return x

class Conv1dBlock(nn.Module):
    out_channels: int
    kernel_size: int = 3
    mish_act: bool = True
    n_groups: int = 8

    def setup(self):
        self.conv = nn.Conv(features=self.out_channels,
                            kernel_size=(self.kernel_size,),
                            padding='SAME',
                            kernel_init=initializers.lecun_normal())
        self.gn = nn.GroupNorm(num_groups=self.n_groups, epsilon=1e-6)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.conv(x)
        x = self.gn(x)
        if self.mish_act:
            return mish(x)
        else:
            return nn.silu(x)

# class Downsample1d(nn.Module):
#     dim: int

#     @nn.compact
#     def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
#         # stride=2 conv
#         conv = nn.Conv(features=self.dim, kernel_size=(3,), strides=(2,), padding='SAME')
#         return conv(x)
class Downsample1d(nn.Module):
    dim: int

    def setup(self):
        self.conv = nn.Conv(features=self.dim, kernel_size=(3,), strides=(2,), padding='SAME')

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.conv(x)


# class Upsample1d(nn.Module):
#     dim: int
#     kernel_size: int = 4
#     stride: int = 2

#     @nn.compact
#     def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
#         # ConvTranspose1d equivalent
#         convt = nn.ConvTranspose(features=self.dim,
#                                  kernel_size=(self.kernel_size,),
#                                  strides=(self.stride,),
#                                  padding='SAME')
#         return convt(x)

class Upsample1d(nn.Module):
    dim: int
    kernel_size: int = 4
    stride: int = 2

    def setup(self):
        self.convt = nn.ConvTranspose(features=self.dim,
                                      kernel_size=(self.kernel_size,),
                                      strides=(self.stride,),
                                      padding='SAME')

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.convt(x)

class Residual(nn.Module):
    fn: nn.Module

    def __call__(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class ResidualTemporalBlock(nn.Module):
    in_channels: int
    out_channels: int
    embed_dim: int
    kernel_size: int = 5
    mish: bool = True

    def setup(self):
        self.block1 = Conv1dBlock(out_channels=self.out_channels, kernel_size=self.kernel_size, mish_act=self.mish)
        self.block2 = Conv1dBlock(out_channels=self.out_channels, kernel_size=self.kernel_size, mish_act=self.mish)
        if self.in_channels != self.out_channels:
            # 1x1 conv to match channels
            self.res_conv = nn.Conv(features=self.out_channels, kernel_size=(1,), padding='SAME')
        else:
            self.res_conv = None

        # time MLP: act -> Dense(out_channels)  (we will expand to length dim)
        # self.time_dense1 = None  # placeholder for compact style (we'll use functional in __call__)
        self.time_dense = nn.Dense(self.out_channels, kernel_init=default_init())

    # @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        x: (batch, length, in_channels)
        t: (batch, embed_dim)
        returns: (batch, length, out_channels)
        """
        # First conv block
        out = self.block1(x)  # -> (batch, length, out_channels)

        # time mlp: act -> Dense(out_channels)
        if self.mish:
            act = mish
        else:
            act = nn.silu

        # dense1 = nn.Dense(self.out_channels, kernel_init=default_init())
        t_proj = self.time_dense(act(t))  # (batch, out_channels)
        # expand over length axis
        t_proj = t_proj[:, None, :]  # (batch, 1, out_channels)
        out = out + t_proj  # broadcast

        # second block
        out = self.block2(out)

        # residual connection (adjust channels if needed)
        if self.res_conv is not None:
            res = self.res_conv(x)
        else:
            res = x
        return out + res

class ResidualBlock(nn.Module):
    in_channels: int
    out_channels: int
    embed_dim: int
    kernel_size: int = 5
    mish: bool = True

    def setup(self):
        self.block1 = Conv1dBlock(out_channels=self.out_channels, kernel_size=self.kernel_size, mish_act=self.mish)
        self.block2 = Conv1dBlock(out_channels=self.out_channels, kernel_size=self.kernel_size, mish_act=self.mish)
        if self.in_channels != self.out_channels:
            # 1x1 conv to match channels
            self.res_conv = nn.Conv(features=self.out_channels, kernel_size=(1,), padding='SAME')
        else:
            self.res_conv = None

    # @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (batch, length, in_channels)
        t: (batch, embed_dim)
        returns: (batch, length, out_channels)
        """
        # First conv block
        out = self.block1(x)  # -> (batch, length, out_channels)

        # second block
        out = self.block2(out)

        # residual connection (adjust channels if needed)
        if self.res_conv is not None:
            res = self.res_conv(x)
        else:
            res = x
        return out + res

class TemporalUNet(nn.Module):
    horizon: int
    transition_dim: int
    dim: int = 128
    dim_mults: Sequence[int] = (1, 2, 4, 8)
    kernel_size: int = 5

    def setup(self):
        dims = [self.transition_dim, *[self.dim * m for m in self.dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))

        use_mish = True
        act_fn = mish  # function

        # time MLP
        self.time_mlp = nn.Sequential([
            SinusoidalPosEmb(self.dim),
            nn.Dense(self.dim * 4, kernel_init=default_init()),
            act_fn,
            nn.Dense(self.dim, kernel_init=default_init())
        ])
        # returns mlp (expects returns shape (batch,1) or (batch,))
        self.returns_mlp_layers = nn.Sequential([
            nn.Dense(self.dim, kernel_init=default_init()),
            act_fn,
            nn.Dense(self.dim * 4, kernel_init=default_init()),
            act_fn,
            nn.Dense(self.dim, kernel_init=default_init())
        ])
        self.costs_mlp_layers = nn.Sequential([
            nn.Dense(self.dim, kernel_init=default_init()),
            act_fn,
            nn.Dense(self.dim * 4, kernel_init=default_init()),
            act_fn,
            nn.Dense(self.dim, kernel_init=default_init())
        ])
        self.cond_mlp_layers = nn.Sequential([
            nn.Dense(self.dim, kernel_init=default_init()),
            act_fn,
            nn.Dense(self.dim * 4, kernel_init=default_init()),
            act_fn,
            nn.Dense(self.dim, kernel_init=default_init())
        ])
        embed_dim = 4 * self.dim

        # build downs
        downs = []
        cur_horizon = self.horizon
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            block_a = ResidualTemporalBlock(dim_in, dim_out, embed_dim=embed_dim, kernel_size=self.kernel_size, mish=use_mish)
            block_b = ResidualTemporalBlock(dim_out, dim_out, embed_dim=embed_dim, kernel_size=self.kernel_size, mish=use_mish)
            down = Downsample1d(dim_out) if not is_last else None
            downs.append((block_a, block_b, down))
            if not is_last:
                cur_horizon = cur_horizon // 2
        self.downs = downs

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, kernel_size=self.kernel_size, mish=use_mish)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, kernel_size=self.kernel_size, mish=use_mish)

        # build ups (reverse, skipping the first dims[0])
        ups = []
        cur_horizon_up = cur_horizon
        rev_in_out = list(reversed(in_out))
        for ind, (dim_in, dim_out) in enumerate(rev_in_out):
            is_last = ind >= (len(rev_in_out) - 1)
            # note: in original code they used dim_out * 2 -> dim_in (due to concat)
            if not is_last:
                block_a = ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=embed_dim, kernel_size=self.kernel_size, mish=use_mish)
                block_b = ResidualTemporalBlock(dim_in, dim_in, embed_dim=embed_dim, kernel_size=self.kernel_size, mish=use_mish)
            else:
                block_a = ResidualTemporalBlock(dim_out * 2, dim_out, embed_dim=embed_dim, kernel_size=self.kernel_size, mish=use_mish)
                block_b = ResidualTemporalBlock(dim_out, dim_out, embed_dim=embed_dim, kernel_size=self.kernel_size, mish=use_mish)
            up = Upsample1d(dim_in) if not is_last else None
            ups.append((block_a, block_b, up))
            if not is_last:
                cur_horizon_up = cur_horizon_up * 2
        self.ups = ups

        # final convs
        self.final_block_conv = Conv1dBlock(out_channels=self.dim, kernel_size=self.kernel_size, mish_act=use_mish)
        self.final_conv_1x1 = nn.Conv(features=self.transition_dim, kernel_size=(1,), padding='SAME')

    def _apply_time_mlp(self, time: jnp.ndarray):
        # time: (batch,) or (batch,1)
        # x = time
        # x = self.time_mlp[0](x)
        # x = self.time_mlp[1](x)
        # x = self.time_mlp[2](x)
        # x = self.time_mlp[3](x)
        # return x  # (batch, dim)
        return self.time_mlp(time)

    def _apply_returns_mlp(self, returns: jnp.ndarray):
        # x = returns
        # for layer in self.returns_mlp_layers:
        #     x = layer(x)
        # return x
        return self.returns_mlp_layers(returns)
    
    def _apply_costs_mlp(self, costs: jnp.ndarray):
        # x = costs
        # for layer in self.costs_mlp_layers:
        #     x = layer(x)
        # return x
        return self.costs_mlp_layers(costs)
    
    def _apply_cond_mlp(self, cond: jnp.ndarray):
        # x = cond
        # for layer in self.cond_mlp_layers:
        #     x = layer(x)
        # return x
        return self.cond_mlp_layers(cond)

    def __call__(self,
                 x: jnp.ndarray,
                 cond: jnp.ndarray,
                 time: jnp.ndarray,
                 returns: jnp.ndarray,
                 costs: jnp.ndarray) -> jnp.ndarray:
        """
        x: (batch, horizon, transition)
        cond: currently unused in this implementation (kept for API parity)
        time: (batch,) or (batch,1)
        returns: (batch,1) or None
        """

        # keep input for possible external use
        x_inp = x

        # time embedding
        t = self._apply_time_mlp(time)  # (batch, dim)

        returns_embed = self._apply_returns_mlp(returns)  # (batch, dim)
        costs_embed = self._apply_costs_mlp(costs)
        cond_embed = self._apply_cond_mlp(cond)
        # concat to time embedding
        t = jnp.concatenate([t, cond_embed, costs_embed, returns_embed], axis=-1)  # (batch, embed_dim)

        h_skips: List[jnp.ndarray] = []

        # down path
        cur = x  # (batch, length, channels)
        for (res1, res2, down) in self.downs:
            cur = res1(cur, t)
            cur = res2(cur, t)
            h_skips.append(cur)
            if down is not None:
                cur = down(cur)

        # mid
        cur = self.mid_block1(cur, t)
        cur = self.mid_block2(cur, t)

        # up path
        for (res1, res2, up) in self.ups:
            skip = h_skips.pop()
            # concat on channels axis (-1)
            cur = jnp.concatenate([cur, skip], axis=-1)
            cur = res1(cur, t)
            cur = res2(cur, t)
            if up is not None:
                cur = up(cur)

        # final conv
        cur = self.final_block_conv(cur)
        cur = self.final_conv_1x1(cur)  # (batch, length, transition_dim)

        # output shape (batch, horizon, transition) consistent with input
        return cur

class TemporalValue(nn.Module):
    horizon: int
    transition_dim: int
    cond_dim: int = 128
    dim: int = 32
    time_dim: Optional[int] = None
    out_dim: int = 1
    dim_mults: Sequence[int] = (1, 2, 4, 8)

    def setup(self):
        dims = [self.transition_dim, *[self.dim * m for m in self.dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))

        td = self.time_dim or self.dim

        act_fn = mish
        self.cond_mlp_layers = nn.Sequential([
            nn.Dense(self.cond_dim, kernel_init=default_init()),
            act_fn,
            nn.Dense(self.cond_dim * 4, kernel_init=default_init()),
            act_fn,
            nn.Dense(self.cond_dim, kernel_init=default_init())
        ])

        # build blocks
        blocks = []
        cur_h = self.horizon
        for dim_in, dim_out in in_out:
            b1 = ResidualBlock(dim_in, dim_out, embed_dim=td, kernel_size=5, mish=True)
            b2 = ResidualBlock(dim_out, dim_out, embed_dim=td, kernel_size=5, mish=True)
            down = Downsample1d(dim_out)
            blocks.append((b1, b2, down))
            cur_h = max(1, cur_h // 2)
        self.blocks = blocks

        self.fc_dim = dims[-1] * max(cur_h, 1)
        self.final_fc1 = nn.Dense(self.fc_dim // 2, kernel_init=default_init())
        self.final_fc2 = nn.Dense(self.out_dim, kernel_init=default_init())
    
    def _apply_cond_mlp(self, cond: jnp.ndarray):
        # x = cond
        # for layer in self.cond_mlp_layers:
        #     x = layer(x)
        # return x
        return self.cond_mlp_layers(cond)

    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray):
        """
        x: (batch, horizon, transition)
        """
        cur = x

        for b1, b2, down in self.blocks:
            cur = b1(cur)
            cur = b2(cur)
            cur = down(cur)

        # flatten and combine with time
        cur = cur.reshape((cur.shape[0], -1))
        cond_emb = self._apply_cond_mlp(cond)
        out = self.final_fc1(jnp.concatenate([cond_emb, cur], axis=-1))
        out = mish(out)
        out = self.final_fc2(out)
        return jnp.squeeze(out, -1)

if __name__=="__main__":
    batch = 4
    horizon = 32
    transition_dim = 6
    cond_dim = 3
    model = TemporalUNet(horizon=horizon, transition_dim=transition_dim)
    # model = TemporalValue(horizon=horizon, transition_dim=transition_dim, cond_dim=cond_dim)
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (batch, horizon, transition_dim))
    cond = jax.random.normal(rng, (batch, transition_dim))
    time = jnp.linspace(0., 1., batch)  # shape (batch,)
    returns = jax.random.normal(rng, (batch, 1))
    costs = jax.random.normal(rng, (batch, 1))
    variables = model.init({'params': rng}, x, cond, time, returns, costs)
    params = variables['params']
    out = model.apply({'params': params}, x, cond, time, returns, costs)
    print(out.shape)  # (batch, horizon, transition_dim)