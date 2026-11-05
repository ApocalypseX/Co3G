from ml_collections import ConfigDict
import numpy as np

def get_config(config_string):
    base_real_config = dict(
        project='FISOR',
        seed=-1,
        max_steps=2000001,
        eval_episodes=20,
        batch_size=1024, #Actor batch size x 2 (so really 1024), critic is fixed to 256 2048
        log_interval=1000,
        eval_interval=250000,
        normalize_returns=True,
        dynamics_hidden_dims = [256,256,256,256],
        num_ensemble = 7,
        num_elites = 5,
        dynamic_weight_decays = [2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4],
        with_cost = False,
        dynamics_lr = 1e-3,
        simple_scaler = True,
        use_scheduler = False,
        use_delta_obs = True,
        reward_scale = 1.0,
        cost_scale = 1.0,
        cost_coef = 0.5,
        rollout_batch_size = 10000,
        model_buffer_size = 1000000,
        rollout_interval=250000,
        pid = [0.1, 0.003, 0.001],
        actor_update_freq = 2,
    )

    if base_real_config["seed"] == -1:
        base_real_config["seed"] = np.random.randint(1000)

    base_data_config = dict(
        cost_scale=25,
        pr_data='~/point_robot-expert-random-100k.hdf5', # The location of point_robot data
    )

    possible_structures = {
        "fisor": ConfigDict(
            dict(
                agent_kwargs=dict(
                    model_cls="FISOR",
                    cost_limit=80,
                    actor_lr=3e-4,
                    critic_lr=3e-4,
                    value_lr=3e-4,
                    cost_temperature=5,
                    reward_temperature=3,
                    T=5,
                    N=16,
                    M=0,
                    clip_sampler=True,
                    actor_dropout_rate=0.1,
                    actor_num_blocks=3,
                    actor_weight_decay=None,
                    decay_steps=int(3e6),
                    actor_layer_norm=True,
                    value_layer_norm=False,
                    actor_tau=0.001,
                    actor_architecture='ln_resnet',
                    critic_objective='expectile',
                    critic_hyperparam = 0.9,
                    cost_critic_hyperparam = 0.9,
                    critic_type="hj", #[hj, qc]
                    cost_ub=150,
                    beta_schedule='vp',
                    actor_objective="feasibility", 
                    sampling_method="ddpm", 
                    extract_method="minqc", 
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_real_config,
            )
        ),
        "co3g": ConfigDict(
            dict(
                agent_kwargs=dict(
                    model_cls="Co3G",
                    cost_limit=80,
                    actor_lr=3e-4,
                    critic_lr=3e-4,
                    value_lr=3e-4,
                    cost_temperature=5,
                    reward_temperature=3,
                    T=5,
                    N=16,
                    M=0,
                    clip_sampler=True,
                    actor_dropout_rate=0.1,
                    actor_num_blocks=3,
                    actor_weight_decay=None,
                    decay_steps=int(3e6),
                    actor_layer_norm=True,
                    value_layer_norm=False,
                    actor_tau=0.001,
                    actor_architecture='ln_resnet',
                    critic_objective='expectile',
                    critic_hyperparam = 0.9,
                    cost_critic_hyperparam = 0.9,
                    reward_advantage_predictor_hyperparam = 0.8,
                    critic_type="hj", #[hj, qc]
                    cost_ub=150,
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_real_config,
            )
        ),
        "co3g_guider": ConfigDict(
            dict(
                diffusor_kwargs=dict(
                    model_cls="Co3G",
                    cost_limit=80,
                    actor_lr=3e-4,
                    critic_lr=3e-4,
                    value_lr=3e-4,
                    cost_temperature=5,
                    reward_temperature=3,
                    T=5,
                    N=16,
                    M=0,
                    clip_sampler=True,
                    actor_dropout_rate=0.1,
                    actor_num_blocks=3,
                    actor_weight_decay=None,
                    decay_steps=int(3e6),
                    actor_layer_norm=True,
                    value_layer_norm=False,
                    actor_tau=0.001,
                    actor_architecture='ln_resnet',
                    critic_objective='expectile',
                    critic_hyperparam = 0.9,
                    cost_critic_hyperparam = 0.9,
                    critic_type="hj", #[hj, qc]
                    cost_ub=150,
                ),
                agent_kwargs=dict(
                    cost_limit=80,
                    actor_lr=1e-4,
                    critic_lr=3e-4,
                    cost_temperature=5,
                    reward_temperature=3,
                    T=5,
                    N=16,
                    M=0,
                    clip_sampler=True,
                    actor_dropout_rate=0.1,
                    actor_num_blocks=3,
                    actor_weight_decay=None,
                    decay_steps=int(3e6),
                    actor_layer_norm=True,
                    value_layer_norm=True,
                    actor_tau=0.001,
                    actor_architecture='ln_resnet',
                    critic_objective='expectile',
                    critic_hyperparam = 0.9,
                    cost_critic_hyperparam = 0.9,
                    reward_advantage_predictor_hyperparam = 0.8,
                    critic_type="hj", #[hj, qc]
                    cost_ub=150,
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_real_config,
            )
        ),
    }
    return possible_structures[config_string]