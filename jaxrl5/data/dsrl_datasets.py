import os
import gymnasium as gym
import dsrl
import numpy as np
from jaxrl5.data.dataset import Dataset
import h5py
import math
from jaxrl5.data.dataset import _check_lengths
from typing import *


class DSRLDataset(Dataset):
    def __init__(self, env: gym.Env, clip_to_eps: bool = True, eps: float = 1e-5, critic_type="qc", data_location=None, cost_scale=1., ratio = 1.0, 
                 safe_only=False, env_name=None, conservative_cost_f=False, separate_buffer=False, unsafe_percent=-1.0, require_org_cost=False, max_length=None):

        if data_location is not None:
            # Point Robot
            dataset_dict = {}
            print('=========Data loading=========')
            print('Load point robot data from:', data_location)
            f = h5py.File(data_location, 'r')
            dataset_dict["observations"] = np.array(f['state'])
            dataset_dict["actions"] = np.array(f['action'])
            dataset_dict["next_observations"] = np.array(f['next_state'])
            dataset_dict["rewards"] = np.array(f['reward'])
            dataset_dict["dones"] = np.array(f['done'])
            # dataset_dict['costs'] = np.array(f['h'])
            dataset_dict['costs'] = np.array(f['cost'])

            if critic_type == "hj":
                if require_org_cost:
                    dataset_dict['costs_org'] = dataset_dict['costs']
                dataset_dict['costs'] = np.where(dataset_dict['costs']>0, 1*cost_scale, -1)
                

            violation = np.array(f['cost'])
            print('env_max_episode_steps', env._max_episode_steps)
            print('mean_episode_reward', env._max_episode_steps * np.mean(dataset_dict['rewards']))
            print('mean_episode_cost', env._max_episode_steps * np.mean(violation))

        else:
            # DSRL
            if ratio == 1.0:
                dataset_dict = env.get_dataset()
            else:
                _, dataset_name = os.path.split(env.dataset_url)
                file_list = dataset_name.split('-')
                ratio_num = int(float(file_list[-1].split('.')[0]) * ratio)
                dataset_ratio = '-'.join(file_list[:-1]) + '-' + str(ratio_num) + '-' + str(ratio) + '.hdf5'
                dataset_dict = env.get_dataset(os.path.join('data', dataset_ratio))
            print('max_episode_reward', env.max_episode_reward, 
                'min_episode_reward', env.min_episode_reward,
                'mean_episode_reward', env._max_episode_steps * np.mean(dataset_dict['rewards']))
            print('max_episode_cost', env.max_episode_cost, 
                'min_episode_cost', env.min_episode_cost,
                'mean_episode_cost', env._max_episode_steps * np.mean(dataset_dict['costs']))
            print('data_num', dataset_dict['actions'].shape[0])
            dataset_dict['dones'] = np.logical_or(dataset_dict["terminals"],
                                                dataset_dict["timeouts"]).astype(np.float32)
            del dataset_dict["terminals"]
            del dataset_dict['timeouts']


            self.require_org_cost = require_org_cost
            if critic_type == "hj":
                if require_org_cost:
                    dataset_dict['costs_org'] = dataset_dict['costs']
                dataset_dict['costs'] = np.where(dataset_dict['costs']>0, 1*cost_scale, -1)
        
        if max_length is not None:
            self.max_length = max_length
        
        self.cost_scale = cost_scale
        self.critic_type = critic_type
        self.clip_to_eps = clip_to_eps
        self.eps = eps
        if clip_to_eps:
            lim = 1 - eps
            dataset_dict["actions"] = np.clip(dataset_dict["actions"], -lim, lim)

        for k, v in dataset_dict.items():
            dataset_dict[k] = v.astype(np.float32)

        dataset_dict["masks"] = 1.0 - dataset_dict['dones']
        del dataset_dict['dones']

        self.separate_buffer = separate_buffer
        if self.separate_buffer:
            self.reward_mean = np.mean(dataset_dict['rewards'])
            self.reward_std = np.std(dataset_dict['rewards'])
            for key in dataset_dict.keys():
                dataset_dict[key] = None
        super().__init__(dataset_dict)
    
    def add_and_norm(self, added_data, max_episode_reward: float, min_episode_reward: float, scaling: float = 1000):

        added_data['rewards'] = added_data['rewards'].reshape((-1,))
        added_data['rewards'] = added_data['rewards'] * scaling / (max_episode_reward - min_episode_reward)
        added_data['masks'] = 1.0 - added_data['dones'].reshape((-1,))
        if self.critic_type == "hj":
            added_data['costs'] = np.where(added_data['costs']>0, 1*self.cost_scale, -1).reshape((-1,))

        if self.dataset_dict['observations'] is not None:
            self.dataset_dict['observations'] = np.concatenate((self.dataset_dict['observations'], added_data['observations']), axis=0)
            self.dataset_dict['actions'] = np.concatenate((self.dataset_dict['actions'], added_data['actions']), axis=0)
            self.dataset_dict['next_observations'] = np.concatenate((self.dataset_dict['next_observations'], added_data['next_observations']), axis=0)
            self.dataset_dict['rewards'] = np.concatenate((self.dataset_dict['rewards'], added_data['rewards']), axis=0)
            self.dataset_dict['masks'] = np.concatenate((self.dataset_dict['masks'], added_data['masks']), axis=0)
            self.dataset_dict['costs'] = np.concatenate((self.dataset_dict['costs'], added_data['costs']), axis=0)
        else:
            self.dataset_dict['observations'] = np.array(added_data['observations'])
            self.dataset_dict['actions'] = np.array(added_data['actions'])
            self.dataset_dict['next_observations'] = np.array(added_data['next_observations'])
            self.dataset_dict['rewards'] = np.array(added_data['rewards'])
            self.dataset_dict['masks'] = np.array(added_data['masks'])
            self.dataset_dict['costs'] = np.array(added_data['costs'])

        if self.clip_to_eps:
            lim = 1 - self.eps
            self.dataset_dict["actions"] = np.clip(self.dataset_dict["actions"], -lim, lim)
        
        for k, v in self.dataset_dict.items():
            self.dataset_dict[k] = v.astype(np.float32)
        
        self.dataset_len = _check_lengths(self.dataset_dict)
        print("New Dataset Length: ", self.dataset_len)
    
    def add_and_normalize(self, added_data):

        added_data['rewards'] = added_data['rewards'].reshape((-1,))
        added_data['rewards'] = (added_data['rewards'] - self.reward_mean) / (self.reward_std + 1e-8)
        added_data['masks'] = 1.0 - added_data['dones'].reshape((-1,))
        if self.require_org_cost:
            added_data['costs_org'] = added_data['costs'].reshape((-1,))
        if self.critic_type == "hj":
            added_data['costs'] = np.where(added_data['costs']>0, 1*self.cost_scale, -1).reshape((-1,))

        if self.dataset_dict['observations'] is not None:
            self.dataset_dict['observations'] = np.concatenate((self.dataset_dict['observations'], added_data['observations']), axis=0)
            self.dataset_dict['actions'] = np.concatenate((self.dataset_dict['actions'], added_data['actions']), axis=0)
            self.dataset_dict['next_observations'] = np.concatenate((self.dataset_dict['next_observations'], added_data['next_observations']), axis=0)
            self.dataset_dict['rewards'] = np.concatenate((self.dataset_dict['rewards'], added_data['rewards']), axis=0)
            self.dataset_dict['masks'] = np.concatenate((self.dataset_dict['masks'], added_data['masks']), axis=0)
            self.dataset_dict['costs'] = np.concatenate((self.dataset_dict['costs'], added_data['costs']), axis=0)
            if self.require_org_cost:
                self.dataset_dict['costs_org'] = np.concatenate((self.dataset_dict['costs_org'], added_data['costs_org']), axis=0)
        else:
            self.dataset_dict['observations'] = np.array(added_data['observations'])
            self.dataset_dict['actions'] = np.array(added_data['actions'])
            self.dataset_dict['next_observations'] = np.array(added_data['next_observations'])
            self.dataset_dict['rewards'] = np.array(added_data['rewards'])
            self.dataset_dict['masks'] = np.array(added_data['masks'])
            self.dataset_dict['costs'] = np.array(added_data['costs'])
            if self.require_org_cost:
                self.dataset_dict['costs_org'] = np.array(added_data['costs_org'])

        if self.clip_to_eps:
            lim = 1 - self.eps
            self.dataset_dict["actions"] = np.clip(self.dataset_dict["actions"], -lim, lim)
        
        for k, v in self.dataset_dict.items():
            self.dataset_dict[k] = v.astype(np.float32)
        
        self.dataset_len = _check_lengths(self.dataset_dict)
        if self.dataset_len > self.max_length:
            self.dataset_len = self.max_length
            for k in self.dataset_dict.keys():
                self.dataset_dict[k] = self.dataset_dict[k][-self.max_length:]
            self.dataset_len = _check_lengths(self.dataset_dict)
        print("New Dataset Length: ", self.dataset_len)
    
    def add(self, added_data):
        self.dataset_dict['observations'] = np.concatenate((self.dataset_dict['observations'], added_data['observations']), axis=0)
        self.dataset_dict['actions'] = np.concatenate((self.dataset_dict['actions'], added_data['actions']), axis=0)
        self.dataset_dict['next_observations'] = np.concatenate((self.dataset_dict['next_observations'], added_data['next_observations']), axis=0)

        added_data['rewards'] = added_data['rewards'].reshape((-1,))
        self.dataset_dict['rewards'] = np.concatenate((self.dataset_dict['rewards'], added_data['rewards']), axis=0)

        added_data['masks'] = 1.0 - added_data['dones'].reshape((-1,))
        self.dataset_dict['masks'] = np.concatenate((self.dataset_dict['masks'], added_data['masks']), axis=0)

        if self.critic_type == "hj":
            added_data['costs'] = np.where(added_data['costs']>0, 1*self.cost_scale, -1).reshape((-1,))
        self.dataset_dict['costs'] = np.concatenate((self.dataset_dict['costs'], added_data['costs']), axis=0)

        if self.clip_to_eps:
            lim = 1 - self.eps
            self.dataset_dict["actions"] = np.clip(self.dataset_dict["actions"], -lim, lim)
        
        for k, v in self.dataset_dict.items():
            self.dataset_dict[k] = v.astype(np.float32)
        
        self.dataset_len = _check_lengths(self.dataset_dict)
        print("New Dataset Length: ", self.dataset_len)

