#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# ============================================================================
# Structured-TOML launch for GR1 action-policy SFT on Cosmos3-Nano (8B MoT).
# Drives cosmos_framework.scripts.train against
# examples/toml/sft_config/gr1_robot_policy_merge.toml (selects the
# registered `gr1_robot_policy_merge` experiment; res256, GR1 29D joint
# policy + use_state, ego_view, trains the generation + action heads).
#
# Env vars (override for your filesystem):
#   GR1_DATA_ROOT         GR1 LeRobot root (single dataset OR parent of datasets)
#   BASE_CHECKPOINT_PATH  DCP of nvidia/Cosmos3-Nano (convert_model_to_dcp; see docs)
#   WAN_VAE_PATH          Wan2.2 VAE .pth (Wan-AI/Wan2.2-TI2V-5B)
#   WANDB_API_KEY         for online logging (TOML wandb_mode="online")
#   NPROC_PER_NODE        torchrun --nproc_per_node (default 8)
#   EXTRA_TAIL_OVERRIDES  space-separated Hydra overrides
#
# Single-node smoke (config/data sanity, a few iters):
#   export EXTRA_TAIL_OVERRIDES="trainer.max_iter=10 checkpoint.save_iter=10 \
#                                dataloader_train.max_samples_per_batch=8 \
#                                dataloader_train.dataloader.num_workers=0"
#   bash examples/launch_gr1_robot_policy_merge.sh
#
# Multi-node: launch on every worker; the trainer reads torchrun's
# --nnodes/--node_rank. For HSDP set
# model.parallelism.data_parallel_replicate_degree = <num_nodes> (shard stays 8).
# ============================================================================

TOML_FILE="examples/toml/sft_config/gr1_robot_policy_merge.toml"
: "${DATASET_PATH:=/root/workspace/mengya/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot/}"
: "${BASE_CHECKPOINT_PATH:=/root/workspace/mengya/cosmos-framework/examples/checkpoints/Cosmos3-Nano-DCP}"
: "${WAN_VAE_PATH:=/root/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth}"
export WAN_VAE_PATH

# Base checkpoint, VAE and tokenizer are local; default to HF offline so no run-time
# Hub network calls are attempted (set HF_HUB_OFFLINE=0 if you need a download).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# W&B API key is a SECRET — keep it OUT of version control. Put it in
# examples/gr1_wandb.env (gitignored via *.env), one line:  WANDB_API_KEY=xxxx
# It is sourced here if present; otherwise set WANDB_API_KEY in your shell, or
# export WANDB_MODE=disabled to skip logging.
WANDB_ENV_FILE="${WANDB_ENV_FILE:-examples/gr1_wandb.env}"
if [[ -f "$WANDB_ENV_FILE" ]]; then set -a; source "$WANDB_ENV_FILE"; set +a; fi

# The experiment reads ${oc.env:GR1_DATA_ROOT}; bridge the launcher's DATASET_PATH to it.
export GR1_DATA_ROOT="${GR1_DATA_ROOT:-$DATASET_PATH}"

# Some environments don't expose the venv's bundled ffmpeg + CUDA NPP/runtime libs on
# the system loader path, which torchcodec needs to decode the LeRobot videos. Add them
# if present (no-op otherwise).
SITE="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))' 2>/dev/null || true)"
if [[ -n "${SITE:-}" && -d "$SITE/nvidia" ]]; then
    export LD_LIBRARY_PATH="$SITE/av.libs:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/npp/lib:${LD_LIBRARY_PATH:-}"
fi

# Accept either a single LeRobot dataset (meta/info.json) or a parent directory of
# datasets (*/meta/info.json).
EXTRA_DATASET_CHECK='[[ -f "$GR1_DATA_ROOT/meta/info.json" ]] || compgen -G "$GR1_DATA_ROOT/*/meta/info.json" > /dev/null || { echo "ERROR: no GR1 LeRobot dataset under $GR1_DATA_ROOT (expected meta/info.json or */meta/info.json)" >&2; exit 1; }'

# Dataloader knobs (env-overridable). NOTE: the new dataloader is
# PackingDataLoader -> RankPartitionedDataLoader, so worker knobs live under the
# nested `dataloader_train.dataloader.*` path; the per-rank batch is
# `dataloader_train.max_samples_per_batch` (the old DataPacker `max_batch_size`;
# the old `pool_size` no longer exists).
: "${MAX_SAMPLES_PER_BATCH:=256}"
: "${NUM_WORKERS:=20}"
: "${PREFETCH_FACTOR:=8}"
: "${PERSISTENT_WORKERS:=true}"

# Run name -> job.name (drives BOTH the output dir and the W&B run name). Give each
# experiment a distinct RUN_NAME to get a fresh output dir + new W&B run (no resume,
# no manual rm). Reusing the same RUN_NAME while keeping its output dir resumes that
# run (and continues its W&B run). Default keeps the recipe's base name.
: "${RUN_NAME:=gr1_robot_policy_merge}"

# Extra Hydra overrides from the environment: a space-separated string word-split into
# the TAIL_OVERRIDES array (an exported string survives `bash <wrapper>`).
TAIL_OVERRIDES=(
    "job.name=${RUN_NAME}"
    "dataloader_train.max_samples_per_batch=${MAX_SAMPLES_PER_BATCH}"
    "dataloader_train.dataloader.num_workers=${NUM_WORKERS}"
    "dataloader_train.dataloader.prefetch_factor=${PREFETCH_FACTOR}"
    "dataloader_train.dataloader.persistent_workers=${PERSISTENT_WORKERS}"
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
