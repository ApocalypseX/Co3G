export XLA_PYTHON_CLIENT_PREALLOCATE=False
# export JAX_PLATFORM_NAME=cpu
for env_id in {0..20}
do
    CUDA_VISIBLE_DEVICES=0 python launcher/examples/train_co3g.py --env_id $env_id --config configs/train_config.py:co3g --actor_type "flow" --project "Co3G" &
    CUDA_VISIBLE_DEVICES=1 python launcher/examples/train_co3g.py --env_id $env_id --config configs/train_config.py:co3g --actor_type "flow" --project "Co3G" &
    CUDA_VISIBLE_DEVICES=2 python launcher/examples/train_co3g.py --env_id $env_id --config configs/train_config.py:co3g --actor_type "flow" --project "Co3G" &
    wait
done


