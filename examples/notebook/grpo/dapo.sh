# =============================================================================
# GRPO + DAPO：在准确率约束下缩短推理长度 — 分阶段参数说明
#
# 【必须同步】评测脚本 SamplingConfig 与下列训练长度对齐，否则长度指标不可比：
#   - max_tokens 应与 max_completion_length 一致（或略大 32～64 留 tokenizer 余量）
#   - 例：本脚本 Phase1 使用 max_completion_length=512 → 评测 max_tokens=512
#
# Phase 1（当前启用）：对齐训推 + 加强软长度惩罚
#   - 降低 completion 硬顶；soft 起罚点低于硬顶，避免「贴墙」
#   - 略提高 soft_overlong 权重
#
# Phase 2（若 Phase1 准确率仍够、长度仍偏长）：在 Phase1 基础上
#   - max_completion_length 384
#   - reward_weights 1.0 1.8
#   - soft_max_length 640, soft_cache_length 256  （expected_len=384，与 completion 上限对齐惩罚区间）
#
# Phase 3（若组内 reward 拉不开）：temperature 0.8~0.9 或检查 num_generations / 数据
# =============================================================================

source /data3/zhangwanyu/miniconda3/bin/activate vllm_0.8.5
# 设置 wandb 存储目录
export WANDB_DIR=/data3/zhangwanyu/.wandb
export WANDB_CACHE_DIR=/data3/zhangwanyu/.wandb/cache

# 同时重新设置 ModelScope/HF 缓存（避免之前的问题）
export MODELSCOPE_CACHE=/data3/zhangwanyu/.cache/modelscope
export HF_HOME=/data3/zhangwanyu/.cache/huggingface

# 创建目录（确保权限）
mkdir -p /data3/zhangwanyu/.wandb/cache

export NCCL_IB_DISABLE=1
export VLLM_USE_V1=0
export NCCL_P2P_LEVEL=SYS
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501
# 新增这个很重要，强制 vLLM 内部也用 loopback
export VLLM_HOST_IP=127.0.0.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_TRITON_FLASH_ATTN=0
# 若 plugin 需要 E2B：运行前 export，勿把真实密钥写入仓库
# export E2B_API_KEY="your_key"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift rlhf \
    --rlhf_type grpo \
    --model /data3/huangxinyi/models/Qwen/Qwen3-4B/ \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs accuracy soft_overlong \
    --overlong_filter true \
    --reward_weights 1.0 1.5 \
    --vllm_mode colocate \
    --soft_cache_length 384 \
    --soft_max_length 768 \
    --vllm_gpu_memory_utilization 0.50 \
    --sleep_level 0 \
    --offload_model true \
    --vllm_enforce_eager \
    --vllm_enable_prefix_caching false \
    --vllm_max_model_len 1536 \
    --use_vllm true \
    --lora_rank 16 \
    --lora_alpha 32 \
    --torch_dtype float16 \
    --dataset 'renwulei_rl' \
    --load_from_cache_file true \
    --max_completion_length 512 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 4 \
    --eval_steps 20 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 1024 \
    --output_dir output/output_qwen3_dapo_lora_gsm8k_chinese_dapo \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 0.75 \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero2 \
    --log_completions false \
    --report_to wandb \
    --generation_batch_size 8 \
    --loss_type dapo \
    --epsilon 0.2 \
    --epsilon_high 0.28
