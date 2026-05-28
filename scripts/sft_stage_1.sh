export VIDEO_MIN_PIXELS=78400 # 100*28*28. the minimum visual frame tokens sent to llm is 100
export FPS_MAX_FRAMES=480 # maximum number of frames for each video (480/60/2 = 4min)

export VIDEO_MAX_PIXELS=19267584 # 24576*28*28. the maximum overall video tokens sent to llm is 24k (leave 8k for language)

text_sink=512
TEXT_SLIDING_WINDOW=512

export DATASET_PATH=/path/to/your/Inf-Stream-Train
########################################################
# Key parameters
epoch_num=1
gradient_accumulation_steps=64
learning_rate=1e-5 
model_name="Qwen/Qwen2.5-VL-7B-Instruct"
########################################################

# WANDB_API_KEY=/your/wandb/api/key
# WANDB_ENTITY=/your/wandb/entity

WANDB_API_KEY=your-wandb-api-key
WANDB_ENTITY=your-wandb-entity
WANDB_PROJECT_NAME=StreamingVLM_SFT_stage_1

timestamp=$(date +%Y%m%d_%H%M%S)

export RUN_NAME="${WANDB_PROJECT_NAME}_e${epoch_num}_lr${learning_rate}_ps${text_sink}_pw${TEXT_SLIDING_WINDOW}"
export OUTPUT_DIR="./checkpoints" # wo a / at the end

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

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_API_KEY=$WANDB_API_KEY WANDB_ENTITY=$WANDB_ENTITY WANDB_PROJECT=$WANDB_PROJECT_NAME TOKENIZERS_PARALLELISM=false torchrun --standalone --nproc_per_node=8 train.py --deepspeed ./scripts/zero3.json --output_dir "${OUTPUT_DIR}/${RUN_NAME}_${timestamp}" --run_name $RUN_NAME --save_on_each_node True --do_train True --per_device_train_batch_size 1 --gradient_accumulation_steps $gradient_accumulation_steps --learning_rate $learning_rate --warmup_ratio 0.03 --optim adamw_torch --lr_scheduler_type cosine --num_train_epochs $epoch_num --logging_steps 1 --bf16 True --tf32 True --gradient_checkpointing True --pretrained_model_name_or_path $model_name --train_annotation_paths "${TRAIN_FILES[@]}"  --dataloader_num_workers 32  --use_liger_kernel True --report_to wandb   --ignore_data_skip False --save_strategy steps --save_steps 20 --save_total_limit 10 --load_best_model_at_end False   --greater_is_better False --prediction_loss_only true --eval_steps 100 --metric_for_best_model eval_loss --eval_strategy steps --per_device_eval_batch_size 1 --eval_annotation_paths "${VALID_FILES[@]}" --text_sink $text_sink --text_sliding_window $TEXT_SLIDING_WINDOW

