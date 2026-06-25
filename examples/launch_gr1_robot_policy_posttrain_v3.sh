#!/usr/bin/env bash
# GR1 robot-policy posttraining v3 launcher.
#
# Changes from v2:
#   - 29D action (filter zero parts, per-dataset normalization)
#   - Keeps: loss_scale=10, encode_exact_durations=[17], image aug
set -uo pipefail
export HF_HUB_OFFLINE=1
export WANDB_ENTITY="lixing11177-nan"

SITE=$(python -c "import sysconfig; print(sysconfig.get_path(\"purelib\"))")
NVIDIA_DIR="$SITE/nvidia"

ls -l "$NVIDIA_DIR/cuda_runtime/lib/libcudart.so.12"
ln -sfn cuda_runtime "$NVIDIA_DIR/cudart"

TOML_FILE="examples/toml/sft_config/gr1_robot_policy_posttrain_v3.toml"
: "${GR1_DATA_ROOT:=/root/workspace/mengya/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot/}"
: "${COSMOS3_NANO_HF_PATH:=/root/.cache/huggingface/hub/models--nvidia--Cosmos3-Nano/snapshots/03c14e74a6ddb51985d614b75d70f2443efc6a05}"
: "${BASE_CHECKPOINT_PATH:=/root/workspace/mengya/cosmos-framework/examples/checkpoints/Cosmos3-Nano-DCP}"
: "${WAN_VAE_PATH:=/root/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth}"
: "${NPROC_PER_NODE:=8}"
: "${MASTER_PORT:=50031}"
: "${GR1_DEBUG_INPUT:=0}"
: "${GR1_DEBUG_LOSS:=0}"
: "${GR1_DEBUG_LIMIT:=0}"
: "${GR1_DEBUG_TIME:=0}"
: "${GR1_DEBUG_TIME_EVERY:=1}"
: "${GR1_DEBUG_TIME_LIMIT:=0}"
: "${GR1_DEBUG_TIME_SYNC:=1}"
: "${TRAIN_ITERS:=20000}"
: "${WANDB_MODE:=online}"
: "${WANDB_ENV_FILE:=examples/gr1_wandb.env}"
: "${CUDA_VISIBLE_DEVICES:=4,5,6,7,0,1,2,3}"

WANDB_API_KEY="8c7410f7bcb3dbb1917b410cf7583e2f34c10d2e"

if [[ -f "$WANDB_ENV_FILE" ]]; then
  set -a
  source "$WANDB_ENV_FILE"
  set +a
fi

TAIL_OVERRIDES=(
  "trainer.max_iter=${TRAIN_ITERS}"
  "job.wandb_mode=${WANDB_MODE}"
  "scheduler.cycle_lengths=[${TRAIN_ITERS}]"
  "dataloader_train.num_workers=${NUM_WORKERS:-4}"
  "dataloader_train.persistent_workers=false"
  "dataloader_train.max_batch_size=128"
  "dataloader_train.pool_size=128"
  "dataloader_train.prefetch_factor=4"
)

if [[ -d "$SITE/nvidia" ]]; then
  AV_LIBS="$SITE/av.libs"
  export LD_LIBRARY_PATH="${AV_LIBS}:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/npp/lib:${LD_LIBRARY_PATH:-}"
fi

EXTRA_DATASET_CHECK='[[ -f "$BASE_CHECKPOINT_PATH/model/.metadata" ]] || { echo "ERROR: BASE_CHECKPOINT_PATH is not a DCP checkpoint root: $BASE_CHECKPOINT_PATH" >&2; exit 1; }'

export WANDB_MODE CUDA_VISIBLE_DEVICES
[[ -n "${WANDB_API_KEY:-}" ]] && export WANDB_API_KEY
export GR1_DATA_ROOT BASE_CHECKPOINT_PATH WAN_VAE_PATH COSMOS3_NANO_HF_PATH EXTRA_DATASET_CHECK GR1_DEBUG_INPUT GR1_DEBUG_LOSS GR1_DEBUG_LIMIT GR1_DEBUG_TIME GR1_DEBUG_TIME_EVERY GR1_DEBUG_TIME_LIMIT GR1_DEBUG_TIME_SYNC WANDB_MODE WANDB_ENV_FILE

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
