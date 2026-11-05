export XLA_PYTHON_CLIENT_PREALLOCATE=False
# export JAX_PLATFORM_NAME=cpu
for env_id in {0..20}
do
    CUDA_VISIBLE_DEVICES=0 python launcher/examples/train_co3g_guider.py --env_id $env_id --config configs/train_config.py:co3g_guider --num_qs 2 --steering_noise 1 --distill_org_qr 0 --guidance_bound 5.0 --value_ensemble_weight 0.75 --use_org_respo 1 --actor_loss_type "respo" --actor_type "flow" --tau 0.9 --cost_limit 10 --project "Co3GGuider" &
    CUDA_VISIBLE_DEVICES=1 python launcher/examples/train_co3g_guider.py --env_id $env_id --config configs/train_config.py:co3g_guider --num_qs 2 --steering_noise 1 --distill_org_qr 0 --guidance_bound 5.0 --value_ensemble_weight 0.75 --use_org_respo 1 --actor_loss_type "respo" --actor_type "flow" --tau 0.9 --cost_limit 10 --project "Co3GGuider" &
    CUDA_VISIBLE_DEVICES=2 python launcher/examples/train_co3g_guider.py --env_id $env_id --config configs/train_config.py:co3g_guider --num_qs 2 --steering_noise 1 --distill_org_qr 0 --guidance_bound 5.0 --value_ensemble_weight 0.75 --use_org_respo 1 --actor_loss_type "respo" --actor_type "flow" --tau 0.9 --cost_limit 10 --project "Co3GGuider" &
    wait
done





