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
from jaxrl5.agents import FISOR, Co3G, Co3GGuiderWithNoiseReward
from jaxrl5.data.dsrl_datasets import DSRLDataset
from jaxrl5.evaluation import evaluate, evaluate_pr
import json

diffusor_list = [
]

FLAGS = flags.FLAGS
flags.DEFINE_integer('env_id', 30, 'Choose env')
flags.DEFINE_integer('max_steps', 2000001, 'max steps')
flags.DEFINE_integer('conservative_cost_f', 0, 'whether to use conservative cost function')
flags.DEFINE_float('ratio', 1.0, 'dataset ratio')
flags.DEFINE_string('project', '', 'project name for wandb')
flags.DEFINE_string('experiment_name', '', 'experiment name for wandb')
flags.DEFINE_integer('cost_limit', 10, 'cost limit')
flags.DEFINE_float('tau', 0.9, 'reverse expectile parameter')
flags.DEFINE_float('reward_adv_tau', 0.8, 'reverse expectile parameter')
flags.DEFINE_float('eps_l_reward', 0.0, 'threshold for reward classifier')
flags.DEFINE_float('eps_l_reward_lower', 0.0, 'threshold for reward classifier')
flags.DEFINE_float('eps_l_cost', 0.0, 'threshold for cost classifier')
flags.DEFINE_float('wr_init', 5.0, 'threshold for cost classifier')
flags.DEFINE_float('wc_init', 5.0, 'threshold for cost classifier')
flags.DEFINE_float('guidance_bound', 5.0, 'threshold for cost classifier')
flags.DEFINE_float('noise_bound', 1.0, 'threshold for cost classifier')
flags.DEFINE_integer('actor_update_freq', 2, 'actor_update_freq')
flags.DEFINE_integer('num_qs', 2, 'num_qs')
flags.DEFINE_string('actor_loss_type', 'cpq', 'actor_loss_type')
flags.DEFINE_integer('steering_noise', 1, 'whether to steer noise')
flags.DEFINE_integer('use_org_respo', 0, 'whether to use_org_respo')
flags.DEFINE_integer('use_org_cpq', 0, 'whether to use_org_cpq')
flags.DEFINE_string('actor_type', 'flow', 'experiment name for wandb')
flags.DEFINE_float('value_ensemble_weight', 0.75, 'threshold for cost classifier')
flags.DEFINE_integer('distill_org_qr', 0, 'whether to use_org_qr')
flags.DEFINE_integer('distill_org_qc', 0, 'whether to use_org_qc')
flags.DEFINE_float('cpq_quantile', 0.0, 'threshold for cost classifier')

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
                          env_name=details['env_name'], conservative_cost_f=details['conservative_cost_f'], require_org_cost=True)
    else:
        env = gym.make(details['env_name'])
        ds = DSRLDataset(env, critic_type=details['agent_kwargs']['critic_type'], cost_scale=details['dataset_kwargs']['cost_scale'], ratio=details['ratio'], 
                         safe_only=False, env_name=details['env_name'], conservative_cost_f=details['conservative_cost_f'], require_org_cost=True)
        env_max_steps = env._max_episode_steps
        env = wrap_gym(env, cost_limit=details['cost_limit'])
        # ds.normalize_returns(env.max_episode_reward, env.min_episode_reward, env_max_steps)
        ds.normalize_rewards()
    ds.seed(details["seed"])

    config_dict = dict(details['agent_kwargs'])
    config_dict['env_max_steps'] = env_max_steps

    diffusor_dict = dict(details['diffusor_kwargs'])
    diffusor_dict['env_max_steps'] = env_max_steps

    config_dict.pop("model_cls") 
    config_dict.pop("cost_scale")

    diffusor_dict.pop("model_cls")

    diffusion_policy = Co3G.create(
        details['seed'], env.observation_space, env.action_space, **diffusor_dict
    )
    diffusion_policy = diffusion_policy.load(details['diffusor_path'])

    agent = Co3GGuiderWithNoiseReward.create(
        diffusion_policy, details['seed'], env.observation_space, env.action_space, **config_dict
    )
    
    logdir = "./swanlog_guider_reward"
    swanlab.init(project=details['project'], experiment_name=details['experiment_name'], config=config_dict, mode="local", logdir=logdir)


    save_time = 1
    for i in trange(details['max_steps'], smoothing=0.1, desc=details['experiment_name']):
        sample = ds.sample_jax(details['batch_size'])

        update_actor = False
        if i % details['actor_update_freq'] == 0:
            update_actor = True
        agent, info = agent.update(sample, update_actor)
        
        if i % details['log_interval'] == 0:
            swanlab.log({f"train/{k}": v for k, v in info.items()}, step=i)

        # if i % details['eval_interval'] == 0 and i > 0:
        if i % details['eval_interval'] == 0:
            # agent.save(f"./results/{details['group']}/{details['experiment_name']}", save_time)/
            save_time += 1
            if details['env_name'] == 'PointRobot':
                eval_info = evaluate_pr(agent, env, details['eval_episodes'])
            else:
                eval_info = evaluate(agent, env, details['eval_episodes'])
            if details['env_name'] != 'PointRobot':
                eval_info["normalized_return"], eval_info["normalized_cost"] = env.get_normalized_score(eval_info["return"], eval_info["cost"])
            swanlab.log({f"eval/{k}": v for k, v in eval_info.items()}, step=i)


def main(_):
    parameters = FLAGS.config
    if FLAGS.project != '':
        parameters['project'] = FLAGS.project
    parameters['env_name'] = env_list[FLAGS.env_id]
    parameters['diffusor_path'] = diffusor_list[FLAGS.env_id]
    parameters['max_steps'] = FLAGS.max_steps
    parameters['ratio'] = FLAGS.ratio
    parameters['conservative_cost_f'] = FLAGS.conservative_cost_f
    parameters['cost_limit'] = FLAGS.cost_limit
    parameters['agent_kwargs']['cost_limit'] = FLAGS.cost_limit
    parameters['group'] = parameters['env_name']
    parameters['steering_noise'] = FLAGS.steering_noise

    parameters['agent_kwargs']['critic_hyperparam'] = FLAGS.tau
    parameters['agent_kwargs']['cost_critic_hyperparam'] = FLAGS.tau
    parameters['agent_kwargs']['reward_advantage_predictor_hyperparam'] = FLAGS.reward_adv_tau
    parameters['agent_kwargs']['eps_l_reward'] = FLAGS.eps_l_reward
    parameters['agent_kwargs']['eps_l_cost'] = FLAGS.eps_l_cost
    parameters['agent_kwargs']['wr_init'] = FLAGS.wr_init
    parameters['agent_kwargs']['wc_init'] = FLAGS.wc_init
    

    parameters['agent_kwargs']['guidance_bound'] = FLAGS.guidance_bound
    parameters['agent_kwargs']['value_ensemble_weight'] = FLAGS.value_ensemble_weight
    parameters['agent_kwargs']['noise_bound'] = FLAGS.noise_bound
    parameters['agent_kwargs']['actor_update_freq'] = FLAGS.actor_update_freq
    parameters['agent_kwargs']['actor_loss_type'] = FLAGS.actor_loss_type
    parameters['agent_kwargs']['num_qs'] = FLAGS.num_qs
    parameters['agent_kwargs']['use_org_respo'] = FLAGS.use_org_respo
    parameters['agent_kwargs']['use_org_cpq'] = FLAGS.use_org_cpq
    parameters['agent_kwargs']['actor_type'] = FLAGS.actor_type
    parameters['agent_kwargs']['distill_org_qr'] = FLAGS.distill_org_qr
    parameters['agent_kwargs']['distill_org_qc'] = FLAGS.distill_org_qc
    parameters['agent_kwargs']['cpq_quantile'] = FLAGS.cpq_quantile

    parameters['diffusor_kwargs']['critic_hyperparam'] = FLAGS.tau
    parameters['diffusor_kwargs']['cost_critic_hyperparam'] = FLAGS.tau
    parameters['diffusor_kwargs']['reward_advantage_predictor_hyperparam'] = FLAGS.reward_adv_tau
    parameters['diffusor_kwargs']['eps_l_reward'] = FLAGS.eps_l_reward
    parameters['diffusor_kwargs']['eps_l_reward_lower'] = FLAGS.eps_l_reward_lower
    parameters['diffusor_kwargs']['eps_l_cost'] = FLAGS.eps_l_cost
    parameters['diffusor_kwargs']['wr_init'] = FLAGS.wr_init
    parameters['diffusor_kwargs']['wc_init'] = FLAGS.wc_init
    parameters['diffusor_kwargs']['cost_limit'] = FLAGS.cost_limit
    parameters['diffusor_kwargs']['actor_type'] = FLAGS.actor_type

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

    if not os.path.exists(f"./results/{parameters['group']}/{parameters['experiment_name']}"):
        os.makedirs(f"./results/{parameters['group']}/{parameters['experiment_name']}")
    with open(f"./results/{parameters['group']}/{parameters['experiment_name']}/config.json", "w") as f:
        json.dump(to_dict(parameters), f, indent=4)
    
    call_main(parameters)


if __name__ == '__main__':
    app.run(main)
