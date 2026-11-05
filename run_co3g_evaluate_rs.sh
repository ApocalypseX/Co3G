export XLA_PYTHON_CLIENT_PREALLOCATE=False

for env_id in {0..20}
do
    CUDA_VISIBLE_DEVICES=0 python launcher/examples/evaluate_co3g_rs.py --env_id $env_id --config configs/train_config.py:co3g --num_heads 13 --cost_limit 10 --project "Co3GEvalRS"
    # wait
done


