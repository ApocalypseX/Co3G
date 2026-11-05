from typing import Dict

import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
import time
from jaxrl5.data.dsrl_datasets import DSRLDataset
from tqdm.auto import trange  # noqa


def evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False
) -> Dict[str, float]:

    episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []
    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs, info = env.reset()
        episode_ret, episode_cost, episode_len= 0.0, 0.0, 0
        while True:
            if render:
                env.render()
                time.sleep(1e-3)
            action, agent = agent.eval_actions(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            cost = info["cost"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if terminated or truncated:
                break
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)

    return {"return": np.mean(episode_rets), "episode_len": np.mean(episode_lens), "cost": np.mean(episode_costs)}

def multi_scale_evaluate_simple(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False, test_mode=False
) -> Dict[str, float]:
    
    scale_ls = [(0.0,1.0),(0.2,1.0)]
    total_result = {}
    for scale in scale_ls:
        wr, wc = scale
        episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []
        for _ in trange(num_episodes, desc="Evaluating", leave=False):
            obs, info = env.reset()
            episode_ret, episode_cost, episode_len= 0.0, 0.0, 0
            while True:
                if render:
                    env.render()
                    time.sleep(1e-3)
                action, agent = agent.eval_actions_given_w(obs, wr, wc)
                obs, reward, terminated, truncated, info = env.step(action)
                cost = info["cost"]
                episode_ret += reward
                episode_len += 1
                episode_cost += cost
                if terminated or truncated:
                    break
            episode_rets.append(episode_ret)
            episode_lens.append(episode_len)
            episode_costs.append(episode_cost)
        print(f"{(wr,wc)} return {np.mean(episode_rets)}, cost {np.mean(episode_costs)}")
        new_result = {f"return_{(wr,wc)}": np.mean(episode_rets), f"episode_len_{(wr,wc)}": np.mean(episode_lens), f"cost_{(wr,wc)}": np.mean(episode_costs)}
        total_result.update(new_result)

    return total_result

def multi_scale_evaluate_simple_single_comp(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False, test_mode=False
) -> Dict[str, float]:
    
    scale_ls = [0.0, 1.0]
    total_result = {}
    for scale in scale_ls:
        w = scale
        episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []
        for _ in trange(num_episodes, desc="Evaluating", leave=False):
            obs, info = env.reset()
            episode_ret, episode_cost, episode_len= 0.0, 0.0, 0
            while True:
                if render:
                    env.render()
                    time.sleep(1e-3)
                action, agent = agent.eval_actions_given_w(obs, w)
                obs, reward, terminated, truncated, info = env.step(action)
                cost = info["cost"]
                episode_ret += reward
                episode_len += 1
                episode_cost += cost
                if terminated or truncated:
                    break
            episode_rets.append(episode_ret)
            episode_lens.append(episode_len)
            episode_costs.append(episode_cost)
        print(f"{(w)} return {np.mean(episode_rets)}, cost {np.mean(episode_costs)}")
        new_result = {f"return_{w}": np.mean(episode_rets), f"episode_len_{w}": np.mean(episode_lens), f"cost_{w}": np.mean(episode_costs)}
        total_result.update(new_result)

    return total_result

def multi_scale_evaluate_rs_max_reward(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False
) -> Dict[str, float]:
    
    # wr_ls = [0.0, 0.2, 0.5, 0.8, 1.0]
    # wc_ls = [0.0, 0.2, 0.5, 0.8, 1.0]
    # wr_ls = [0.0, 0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 1.8, 2.0, 2.3, 2.5, 2.8, 3.0]
    # wc_ls = [0.0, 0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 1.8, 2.0, 2.3, 2.5, 2.8, 3.0]
    wr_ls = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    wc_ls = [0.0]
    
    wr = jnp.array(wr_ls)
    wc = jnp.array(wc_ls)
    K = wr.shape[0]
    KC = wc.shape[0]
    wr = jnp.repeat(wr, KC)
    wc = jnp.tile(wc, K)
    episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []
    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs, info = env.reset()
        episode_ret, episode_cost, episode_len= 0.0, 0.0, 0
        while True:
            if render:
                env.render()
                time.sleep(1e-3)
            action, agent = agent.eval_actions_rs_max_reward(obs, wr, wc)
            obs, reward, terminated, truncated, info = env.step(action)
            cost = info["cost"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if terminated or truncated:
                break
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)

    return {"return": np.mean(episode_rets), "episode_len": np.mean(episode_lens), "cost": np.mean(episode_costs)}

def multi_scale_evaluate_rs(
    agent, env: gym.Env, num_episodes: int, num_heads=13, save_video: bool = False, render: bool = False
) -> Dict[str, float]:
    
    # wr_ls = [0.0, 0.2, 0.5, 0.8, 1.0]
    # wc_ls = [0.0, 0.2, 0.5, 0.8, 1.0]
    if num_heads == 13:
        wr_ls = [0.0, 0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 1.8, 2.0, 2.3, 2.5, 2.8, 3.0]
        wc_ls = [0.0, 0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 1.8, 2.0, 2.3, 2.5, 2.8, 3.0]
    elif num_heads == 7:
        wr_ls = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        wc_ls = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    elif num_heads == 4:
        wr_ls = [0.0, 1.0, 2.0, 3.0]
        wc_ls = [0.0, 1.0, 2.0, 3.0]
    elif num_heads == 3:
        wr_ls = [0.0, 1.5, 3.0]
        wc_ls = [0.0, 1.5, 3.0]
    elif num_heads == 2:
        wr_ls = [0.0, 3.0]
        wc_ls = [0.0, 3.0]
    
    wr = jnp.array(wr_ls)
    wc = jnp.array(wc_ls)
    K = wr.shape[0]
    KC = wc.shape[0]
    wr = jnp.repeat(wr, KC)
    wc = jnp.tile(wc, K)
    episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []
    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs, info = env.reset()
        episode_ret, episode_cost, episode_len= 0.0, 0.0, 0
        while True:
            if render:
                env.render()
                time.sleep(1e-3)
            action, agent = agent.eval_actions_rs(obs, wr, wc)
            obs, reward, terminated, truncated, info = env.step(action)
            cost = info["cost"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if terminated or truncated:
                break
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)

    return {"return": np.mean(episode_rets), "episode_len": np.mean(episode_lens), "cost": np.mean(episode_costs)}

def multi_scale_evaluate_rs_single_comp(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False
) -> Dict[str, float]:
    
    # wr_ls = [0.0, 0.2, 0.5, 0.8, 1.0]
    # wc_ls = [0.0, 0.2, 0.5, 0.8, 1.0]
    # wr_ls = [0.0, 0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 1.8, 2.0, 2.3, 2.5, 2.8, 3.0]
    # wc_ls = [0.0, 0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 1.8, 2.0, 2.3, 2.5, 2.8, 3.0]
    w_ls = [0.0, 0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 1.8, 2.0, 2.3, 2.5, 2.8, 3.0]
    
    w = jnp.array(w_ls)
    episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []
    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs, info = env.reset()
        episode_ret, episode_cost, episode_len= 0.0, 0.0, 0
        while True:
            if render:
                env.render()
                time.sleep(1e-3)
            action, agent = agent.eval_actions_rs(obs, w)
            obs, reward, terminated, truncated, info = env.step(action)
            cost = info["cost"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if terminated or truncated:
                break
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)

    return {"return": np.mean(episode_rets), "episode_len": np.mean(episode_lens), "cost": np.mean(episode_costs)}

def multi_scale_evaluate_rs_soft(
    agent, env: gym.Env, num_episodes: int, env_max_steps: int, save_video: bool = False, render: bool = False
) -> Dict[str, float]:
    
    # wr_ls = [0.0, 0.2, 0.5, 0.8, 1.0]
    # wc_ls = [0.0, 0.2, 0.5, 0.8, 1.0]
    # wr_ls = [0.0, 0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 1.8, 2.0, 2.3, 2.5, 2.8, 3.0]
    # wc_ls = [0.0, 0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 1.8, 2.0, 2.3, 2.5, 2.8, 3.0]
    wr_ls = [0.0, 0.5, 1.0, 2.0]
    wc_ls = [0.0, 0.5, 1.0, 2.0]
    # wr_ls = [0.0, 1.0]
    # wc_ls = [0.0, 1.0]

    wr = jnp.array(wr_ls)
    wc = jnp.array(wc_ls)
    K = wr.shape[0]
    KC = wc.shape[0]
    wr = jnp.repeat(wr, KC)
    wc = jnp.tile(wc, K)
    total_result = {}
    avg_cost = 0
    avg_return = 0
    avg_len = 0
    for cost_threshold in [20,40,80]:
        episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []
        for _ in trange(num_episodes, desc="Evaluating", leave=False):
            obs, info = env.reset()
            episode_ret, episode_cost, episode_len= 0.0, 0.0, 0
            cost_limit = cost_threshold
            while True:
                if render:
                    env.render()
                    time.sleep(1e-3)
                disc_cost_limit = max(cost_limit,0)
                action, agent = agent.eval_actions_rs_soft(obs, wr, wc, disc_cost_limit)
                obs, reward, terminated, truncated, info = env.step(action)
                cost = info["cost"]
                episode_ret += reward
                episode_len += 1
                episode_cost += cost
                cost_limit -= cost
                if terminated or truncated:
                    break
            episode_rets.append(episode_ret)
            episode_lens.append(episode_len)
            episode_costs.append(episode_cost)
        result = {f"return_{cost_threshold}": np.mean(episode_rets), f"episode_len_{cost_threshold}": np.mean(episode_lens), f"cost_{cost_threshold}": np.mean(episode_costs)}
        avg_cost += np.mean(episode_costs) / cost_threshold
        avg_return += np.mean(episode_rets)
        avg_len += np.mean(episode_lens)
        total_result.update(result)
    avg_result = {"return_avg": avg_return/3, "episode_len_avg": avg_len/3, "cost_avg":avg_cost/3}
    total_result.update(avg_result)
    return total_result

def multi_scale_evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False, test_mode=False
) -> Dict[str, float]:
    
    wr_ls = [0.0, 0.3, 0.5, 0.8, 1.0]
    wc_ls = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    total_result = {}
    # for scale in scale_ls:/
    for wr in wr_ls:
        for wc in wc_ls:
            episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []
            for _ in trange(num_episodes, desc="Evaluating", leave=False):
                obs, info = env.reset()
                episode_ret, episode_cost, episode_len= 0.0, 0.0, 0
                while True:
                    if render:
                        env.render()
                        time.sleep(1e-3)
                    action, agent = agent.eval_actions_given_w(obs, wr, wc)
                    obs, reward, terminated, truncated, info = env.step(action)
                    cost = info["cost"]
                    episode_ret += reward
                    episode_len += 1
                    episode_cost += cost
                    if terminated or truncated:
                        break
                episode_rets.append(episode_ret)
                episode_lens.append(episode_len)
                episode_costs.append(episode_cost)
            print(f"{(wr,wc)} return {np.mean(episode_rets)}, cost {np.mean(episode_costs)}")
            new_result = {f"return_{(wr,wc)}": np.mean(episode_rets), f"episode_len_{(wr,wc)}": np.mean(episode_lens), f"cost_{(wr,wc)}": np.mean(episode_costs)}
            total_result.update(new_result)
    return total_result

def multi_scale_evaluate_faster(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False, test_mode=False
) -> Dict[str, float]:
    
    wr_ls = [0.0, 0.3, 0.5, 0.8, 1.0]
    wc_ls = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    p=0
    q=0
    total_result = {}
    while True:
        wr = wr_ls[p]
        wc = wc_ls[q]
        last_p = p
        last_q = q
        episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []
        for _ in trange(num_episodes, desc="Evaluating", leave=False):
            obs, info = env.reset()
            episode_ret, episode_cost, episode_len= 0.0, 0.0, 0
            while True:
                if render:
                    env.render()
                    time.sleep(1e-3)
                action, agent = agent.eval_actions_given_w(obs, wr, wc)
                obs, reward, terminated, truncated, info = env.step(action)
                cost = info["cost"]
                episode_ret += reward
                episode_len += 1
                episode_cost += cost
                if terminated or truncated:
                    break
            episode_rets.append(episode_ret)
            episode_lens.append(episode_len)
            episode_costs.append(episode_cost)
        print(f"{(wr,wc)} return {np.mean(episode_rets)}, cost {np.mean(episode_costs)}")
        new_result = {f"return_{(wr,wc)}": np.mean(episode_rets), f"episode_len_{(wr,wc)}": np.mean(episode_lens), f"cost_{(wr,wc)}": np.mean(episode_costs)}
        total_result.update(new_result)
        if np.mean(episode_costs) <= 10:
            p += 1
        else:
            q += 1
        if p >= len(wr_ls):
            p = len(wr_ls)-1
        if q >= len(wc_ls):
            q = len(wc_ls)-1
        if p == last_p and q == last_q:
            break
        
    return total_result

def multi_scale_evaluate_and_save(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False, result_save_dir=None,
) -> Dict[str, float]:
    
    wr_ls = [0.0, 0.3, 0.5, 0.8, 1.0, 1.5]
    wc_ls = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    total_result = {}
    # for scale in scale_ls:/
    for wr in wr_ls:
        for wc in wc_ls:
            obs_ls = []
            episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []
            for _ in trange(num_episodes, desc="Evaluating", leave=False):
                obs, info = env.reset()
                episode_ret, episode_cost, episode_len= 0.0, 0.0, 0
                while True:
                    if render:
                        env.render()
                        time.sleep(1e-3)
                    action, agent = agent.eval_actions_given_w(obs, wr, wc)
                    obs_ls.append(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    cost = info["cost"]
                    episode_ret += reward
                    episode_len += 1
                    episode_cost += cost
                    if terminated or truncated:
                        break
                episode_rets.append(episode_ret)
                episode_lens.append(episode_len)
                episode_costs.append(episode_cost)
            print(f"{(wr,wc)} return {np.mean(episode_rets)}, cost {np.mean(episode_costs)}")
            np.savez(result_save_dir+f"_{wr}_{wc}.npz", obs=np.array(obs_ls))
            new_result = {f"return_{(wr,wc)}": np.mean(episode_rets), f"episode_len_{(wr,wc)}": np.mean(episode_lens), f"cost_{(wr,wc)}": np.mean(episode_costs)}
            total_result.update(new_result)

    return total_result

def multi_scale_evaluate_single_comp(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False, test_mode=False
) -> Dict[str, float]:
    
    w_ls = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    total_result = {}
    # for scale in scale_ls:/
    for w in w_ls:
        episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []
        for _ in trange(num_episodes, desc="Evaluating", leave=False):
            obs, info = env.reset()
            episode_ret, episode_cost, episode_len= 0.0, 0.0, 0
            while True:
                if render:
                    env.render()
                    time.sleep(1e-3)
                action, agent = agent.eval_actions_given_w(obs, w)
                obs, reward, terminated, truncated, info = env.step(action)
                cost = info["cost"]
                episode_ret += reward
                episode_len += 1
                episode_cost += cost
                if terminated or truncated:
                    break
            episode_rets.append(episode_ret)
            episode_lens.append(episode_len)
            episode_costs.append(episode_cost)
        print(f"{(w)} return {np.mean(episode_rets)}, cost {np.mean(episode_costs)}")
        new_result = {f"return_{w}": np.mean(episode_rets), f"episode_len_{w}": np.mean(episode_lens), f"cost_{w}": np.mean(episode_costs)}
        total_result.update(new_result)

    return total_result


def evaluate_and_save(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False, result_save_dir=None,
) -> Dict[str, float]:

    episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []
    obs_ls = []
    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs, info = env.reset()
        episode_ret, episode_cost, episode_len= 0.0, 0.0, 0
        while True:
            if render:
                env.render()
                time.sleep(1e-3)
            action, agent = agent.eval_actions(obs)
            obs_ls.append(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            cost = info["cost"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if terminated or truncated:
                break
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)
    np.savez(result_save_dir, obs=np.array(obs_ls))
    return {"return": np.mean(episode_rets), "episode_len": np.mean(episode_lens), "cost": np.mean(episode_costs)}

def evaluate_pr(
    agent, env: gym.Env, num_episodes: int) -> Dict[str, float]:

    episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []

    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs = env.reset()
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        while True:
            action, agent = agent.eval_actions(obs)
            obs, reward, done, info = env.step(action)
            cost = info["violation"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if done or episode_len == env._max_episode_steps:
                break
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)

    return {"return": np.mean(episode_rets), "episode_len": np.mean(episode_lens), "cost": np.mean(episode_costs), "no_safe": np.mean(episode_no_safes)}
