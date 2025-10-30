#!/bin/bash

set -x

export HYDRA_FULL_ERROR=1
export VERL_LOGGING_LEVEL=DEBUG


PROJECT_NAME="vqa_scanqa_images_multiturn_multi_choice_comprehensive_metric"
EXPERIMENT_NAME="local_reward_0_pretrained_3"

BASEDIR=/workspace/git/verl_0.6
SAVE_CHECKPOINT_DIR=${BASEDIR}/checkpoints

DATASET_TRAIN=/workspace/data/verl/scanqa_images_64_336x224_672x448_multiturn_format_update_1/train.parquet
DATASET_VAL=/workspace/data/verl/scanqa_images_64_336x224_672x448_multiturn_format_update_1/test.parquet

REF_MODEL_PATH=pretrained/scannet_multiframe_qwen_chkpt_0_rererun/scanqa_qwen_cot_verl_3

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-path=${BASEDIR}/recipe/scanqa_multurn_multi_choice_comprehensive_metric/configs \
    --config-name='deepeyes_multiturn_grpo' \
    data.train_files=${DATASET_TRAIN} \
    data.val_files=[${DATASET_VAL}] \
    data.train_batch_size=8 \
    data.max_prompt_length=16384 \
    data.max_response_length=16384 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=False \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=2 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=recipe/scanqa_multurn_multi_choice_comprehensive_metric/configs/image_resize_tool_config.yaml \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb','tensorboard'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=-1 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    trainer.rollout_data_dir=/workspace/experiments/verl/rollout/scanqa_multurn_multi_choice_comprehensive_metric/custom_checkpoint_rerun_reward_0 \
    trainer.total_epochs=1 2>&1 | tee ./logs/${EXPERIMENT_NAME}.log

    # actor_rollout_ref.rollout.multi_turn.tool_config_path=recipe/debug/configs/image_zoom_in_tool_config.yaml \