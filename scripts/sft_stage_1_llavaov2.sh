#!/usr/bin/env bash
set -euo pipefail

# ---- Video pixel + frame budget (overridable per call) ----
# OneVision's vision tower uses patch_size=14 (vs Qwen2.5-VL's 14*2=28),
# so the original VIDEO_*_PIXELS budgets carry over to "visual tokens after spatial_merge",
# but the per-frame raw pixel count is different. We start with the same numbers as
# the Qwen2.5-VL script and tune via env vars on smaller GPUs.
export VIDEO_MIN_PIXELS=${VIDEO_MIN_PIXELS:-78400}        # ~100 visual tokens/frame after merge
export FPS_MAX_FRAMES=${FPS_MAX_FRAMES:-480}              # 4 min @ 2 FPS
export VIDEO_MAX_PIXELS=${VIDEO_MAX_PIXELS:-19267584}     # ~24k visual tokens total per clip

text_sink=${TEXT_SINK:-512}
TEXT_SLIDING_WINDOW=${TEXT_SLIDING_WINDOW:-512}

# Default DATASET_PATH to the local softlinked Inf-Stream-Train tree.
export DATASET_PATH=${DATASET_PATH:-/data/v-kaichen/streaming-vlm/data/Inf-Stream-Train}

if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: DATASET_PATH=$DATASET_PATH does not exist" >&2
    exit 1
fi

# ---- Train hyperparameters ----
epoch_num=${EPOCH_NUM:-1}
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-64}
learning_rate=${LEARNING_RATE:-1e-5}
model_name=${MODEL_NAME:-/data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct}

# ---- Distributed config (default 4 GPUs to match the local A6000 dev box) ----
NPROC_PER_NODE=${NPROC_PER_NODE:-4}

# ---- W&B / output ----
WANDB_API_KEY=${WANDB_API_KEY:-your-wandb-api-key}
WANDB_ENTITY=${WANDB_ENTITY:-your-wandb-entity}
WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-StreamingVLM_LlavaOV2_SFT_stage_1}

timestamp=$(date +%Y%m%d_%H%M%S)
export RUN_NAME="${WANDB_PROJECT_NAME}_e${epoch_num}_lr${learning_rate}_ps${text_sink}_pw${TEXT_SLIDING_WINDOW}"
export OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints}

# ---- Datasets ----
TRAIN_DATASET_NAMES=(
    "train_s12w24_with_seeks.jsonl"
    "train_s12w24_with_seeks.jsonl"
    "train_livecc_with_seeks.jsonl"
)
VALID_DATASET_NAMES=(
    "valid_s12w24_with_seeks.jsonl"
    "valid_s12w24_with_seeks.jsonl"
    "valid_livecc_with_seeks.jsonl"
)

TRAIN_FILES=("${TRAIN_DATASET_NAMES[@]/#/$DATASET_PATH/}")
VALID_FILES=("${VALID_DATASET_NAMES[@]/#/$DATASET_PATH/}")

# ---- Run ----
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_API_KEY=$WANDB_API_KEY \
WANDB_ENTITY=$WANDB_ENTITY \
WANDB_PROJECT=$WANDB_PROJECT_NAME \
TOKENIZERS_PARALLELISM=false \
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE train.py \
    --deepspeed ./scripts/zero3.json \
    --overwrite_output_dir True \
    --output_dir "${OUTPUT_DIR}/${RUN_NAME}_${timestamp}" \
    --run_name "$RUN_NAME" \
    --save_on_each_node True \
    --do_train True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate $learning_rate \
    --warmup_ratio 0.03 \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --num_train_epochs $epoch_num \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --pretrained_model_name_or_path "$model_name" \
    --train_annotation_paths "${TRAIN_FILES[@]}" \
    --dataloader_num_workers 32 \
    --report_to wandb \
    --ignore_data_skip False \
    --save_strategy steps \
    --save_steps 20 \
    --save_total_limit 10 \
    --load_best_model_at_end False \
    --greater_is_better False \
    --prediction_loss_only true \
    --eval_steps 100 \
    --metric_for_best_model eval_loss \
    --eval_strategy steps \
    --per_device_eval_batch_size 1 \
    --eval_annotation_paths "${VALID_FILES[@]}" \
    --text_sink $text_sink \
    --text_sliding_window $TEXT_SLIDING_WINDOW
