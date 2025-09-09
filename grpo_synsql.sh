set -x

export RAY_automatic_object_spilling_enabled=false
export RAY_ROTATION_MAX_BYTES=0
export RAY_ROTATION_BACKUP_COUNT=0

export HYDRA_FULL_ERROR=1

projname='grpo'
expname='synsql_omnisql7b_grpo_00003'

model_path=path_to/OmniSQL-7B
train_files=path_to/train_syn.parquet
test_files=path_to/val_syn.parquet
reward_function=path_to/accuracy_reward.py

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1000 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.actor.optim.lr=5e-5 \
    actor_rollout_ref.actor.optim.lr_constant_steps_ratio=0.0 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.04 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=10 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.temperature=0.8 \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path="$reward_function" \
    custom_reward_function.name='accuracy_reward' \
    trainer.default_local_dir=path_to_save_folder \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=$projname \
    trainer.experiment_name=$expname \
    trainer.n_gpus_per_node=6 \
    trainer.nnodes=1 \
    trainer.save_freq=9 \
    trainer.test_freq=9 \
    trainer.total_epochs=20 $@
