import os
os.environ["WANDB_API_KEY"] = ""
os.environ["SWANLAB_API_KEY"] = ""
import sys
sys.path.append('.')
import random
import numpy as np
from absl import app, flags
import datetime
import yaml
from ml_collections import config_flags, ConfigDict
import wandb
import swanlab
from tqdm.auto import trange  # noqa
import gymnasium as gym
from env.env_list import env_list
from env.point_robot import PointRobot
from jaxrl5.wrappers import wrap_gym
from jaxrl5.agents import FISOR, Co3G
from jaxrl5.data.dsrl_datasets import DSRLDataset
from jaxrl5.evaluation import evaluate, evaluate_pr, multi_scale_evaluate_simple
import json


FLAGS = flags.FLAGS
flags.DEFINE_integer('env_id', 30, 'Choose env')
flags.DEFINE_integer('max_steps', 2000001, 'max steps')
flags.DEFINE_integer('conservative_cost_f', 0, 'whether to use conservative cost function')
flags.DEFINE_float('ratio', 1.0, 'dataset ratio')
flags.DEFINE_string('project', '', 'project name for wandb')
flags.DEFINE_string('experiment_name', '', 'experiment name for wandb')
flags.DEFINE_string('actor_type', 'flow', 'experiment name for wandb')
flags.DEFINE_string('critic_type', 'hj', 'experiment name for wandb')
flags.DEFINE_integer('cost_limit', 10, 'cost limit')
flags.DEFINE_float('tau', 0.9, 'reverse expectile parameter')
flags.DEFINE_float('reward_adv_tau', 0.8, 'reverse expectile parameter')
flags.DEFINE_float('eps_l_reward', 0.0, 'threshold for reward classifier')
flags.DEFINE_float('eps_l_reward_lower', 0.0, 'threshold for reward classifier')
flags.DEFINE_float('eps_l_cost', 0.0, 'threshold for cost classifier')
flags.DEFINE_float('wr_init', 5.0, 'threshold for cost classifier')
flags.DEFINE_float('wc_init', 5.0, 'threshold for cost classifier')
flags.DEFINE_integer('N', 16, 'whether to use conservative cost function')
flags.DEFINE_integer('T', 5, 'whether to use conservative cost function')
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

def to_dict(config):
    if isinstance(config, ConfigDict):
        return {k: to_dict(v) for k, v in config.items()}
    return config


def call_main(details):
    details['agent_kwargs']['cost_scale'] = details['dataset_kwargs']['cost_scale']
    # wandb.init(project=details['project'], name=details['experiment_name'], group=details['group'], config=details['agent_kwargs'])

    if details['env_name'] == 'PointRobot':
        assert details['dataset_kwargs']['pr_data'] is not None, "No data for Point Robot"
        env = eval(details['env_name'])(id=0, seed=0)
        env_max_steps = env._max_episode_steps
        ds = DSRLDataset(env, critic_type=details['agent_kwargs']['critic_type'], data_location=details['dataset_kwargs']['pr_data'], safe_only=False,
                          env_name=details['env_name'], conservative_cost_f=details['conservative_cost_f'])
    else:
        env = gym.make(details['env_name'])
        ds = DSRLDataset(env, critic_type=details['agent_kwargs']['critic_type'], cost_scale=details['dataset_kwargs']['cost_scale'], ratio=details['ratio'], 
                         safe_only=False, env_name=details['env_name'], conservative_cost_f=details['conservative_cost_f'])
        env_max_steps = env._max_episode_steps
        env = wrap_gym(env, cost_limit=details['cost_limit'])
        ds.normalize_returns(env.max_episode_reward, env.min_episode_reward, env_max_steps)
    ds.seed(details["seed"])

    config_dict = dict(details['agent_kwargs'])
    config_dict['env_max_steps'] = env_max_steps

    model_cls = config_dict.pop("model_cls") 
    config_dict.pop("cost_scale") 
    agent = globals()[model_cls].create(
        details['seed'], env.observation_space, env.action_space, **config_dict
    )

    if details['agent_kwargs']['actor_type'] == 'flow':
        if details['agent_kwargs']['critic_type'] == 'qc':
            swanlab.init(project=details['project'], experiment_name=details['experiment_name'], config=config_dict, mode="local",logdir="./swanlog_comp_flow_0424_soft")
        else:
            swanlab.init(project=details['project'], experiment_name=details['experiment_name'], config=config_dict, mode="local",logdir="./swanlog_comp_flow_0424")
        prefix = "_comp_flow_0424"
    else:
        if details['agent_kwargs']['critic_type'] == 'qc':
            swanlab.init(project=details['project'], experiment_name=details['experiment_name'], config=config_dict, mode="local",logdir="./swanlog_comp_diff_0424_soft")
        else:
            swanlab.init(project=details['project'], experiment_name=details['experiment_name'], config=config_dict, mode="local",logdir="./swanlog_comp_diff_0424")
        prefix = "_comp_diff_0424"
    
    if details['agent_kwargs']['critic_type'] == 'qc':
        prefix = "_comp_flow_0424_soft"



    save_time = 1
    for i in trange(details['max_steps'], smoothing=0.1, desc=details['experiment_name']):
        sample = ds.sample_jax(details['batch_size'])
        agent, info = agent.update(sample)
        
        if i % details['log_interval'] == 0:
            swanlab.log({f"train/{k}": v for k, v in info.items()}, step=i)

        # if i % details['eval_interval'] == 0 and i > 0:
        if i % details['eval_interval'] == 0:
            agent.save(f"./results{prefix}/{details['group']}/{details['experiment_name']}", save_time)
            save_time += 1
            if details['env_name'] == 'PointRobot':
                eval_info = evaluate_pr(agent, env, details['eval_episodes'])
            else:
                eval_info = multi_scale_evaluate_simple(agent, env, details['eval_episodes'])
            eval_info_keys = eval_info.keys()
            for key in eval_info_keys:
                if "return" in key:
                    post_key = key[6:]
                    cost_key = "cost"+post_key
                    pre_return = eval_info[key]
                    pre_cost = eval_info[cost_key]
                    post_retun, post_cost = env.get_normalized_score(pre_return, pre_cost)
                    eval_info[key] = post_retun
                    eval_info[cost_key] = post_cost
            swanlab.log({f"eval/{k}": v for k, v in eval_info.items()}, step=i)


def main(_):
    parameters = FLAGS.config
    if FLAGS.project != '':
        parameters['project'] = FLAGS.project
    parameters['env_name'] = env_list[FLAGS.env_id]
    parameters['max_steps'] = FLAGS.max_steps
    parameters['ratio'] = FLAGS.ratio
    parameters['conservative_cost_f'] = FLAGS.conservative_cost_f
    parameters['cost_limit'] = FLAGS.cost_limit
    parameters['agent_kwargs']['cost_limit'] = FLAGS.cost_limit
    parameters['group'] = parameters['env_name']

    parameters['agent_kwargs']['critic_hyperparam'] = FLAGS.tau
    parameters['agent_kwargs']['cost_critic_hyperparam'] = FLAGS.tau
    parameters['agent_kwargs']['reward_advantage_predictor_hyperparam'] = FLAGS.reward_adv_tau
    parameters['agent_kwargs']['eps_l_reward'] = FLAGS.eps_l_reward
    parameters['agent_kwargs']['eps_l_reward_lower'] = FLAGS.eps_l_reward_lower
    parameters['agent_kwargs']['eps_l_cost'] = FLAGS.eps_l_cost
    parameters['agent_kwargs']['wr_init'] = FLAGS.wr_init
    parameters['agent_kwargs']['wc_init'] = FLAGS.wc_init
    parameters['agent_kwargs']['N'] = FLAGS.N
    parameters['agent_kwargs']['T'] = FLAGS.T
    parameters['agent_kwargs']['actor_type'] = FLAGS.actor_type
    parameters['agent_kwargs']['critic_type'] = FLAGS.critic_type

    if parameters['agent_kwargs']['actor_type'] == 'flow':
        prefix = "_comp_flow_0424"
    else:
        prefix = "_comp_diff_0424"
    
    if parameters['agent_kwargs']['critic_type'] == 'qc':
        prefix = "_comp_flow_0424_soft"

    if FLAGS.conservative_cost_f:
        pre = "safer_llm_"
    else:
        pre = ""

    parameters['experiment_name'] = pre + parameters['env_name']
    parameters['experiment_name'] += '_' + str(datetime.date.today()) + '_s' + str(parameters['seed']) + '_c' + str(parameters['cost_limit']) + '_' + str(random.randint(0,1000))

    if parameters['env_name'] == 'PointRobot':
        parameters['max_steps'] = 100001
        parameters['batch_size'] = 1024
        parameters['eval_interval'] = 25000
        parameters['agent_kwargs']['cost_temperature'] = 2
        parameters['agent_kwargs']['reward_temperature'] = 5
        parameters['agent_kwargs']['cost_ub'] = 150
        parameters['agent_kwargs']['N'] = 8

    print(parameters)

    if not os.path.exists(f"./results{prefix}/{parameters['group']}/{parameters['experiment_name']}"):
        os.makedirs(f"./results{prefix}/{parameters['group']}/{parameters['experiment_name']}")
    with open(f"./results{prefix}/{parameters['group']}/{parameters['experiment_name']}/config.json", "w") as f:
        json.dump(to_dict(parameters), f, indent=4)
    
    call_main(parameters)


if __name__ == '__main__':
    app.run(main)
