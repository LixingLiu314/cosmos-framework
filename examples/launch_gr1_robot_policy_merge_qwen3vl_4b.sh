#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# ============================================================================
# Structured-TOML launch for GR1 action-policy SFT with Qwen3-VL-4B init.
# Drives cosmos_framework.scripts.train against
# examples/toml/sft_config/gr1_robot_policy_merge_qwen3vl_4b.toml.
#
# Env vars (override for your filesystem):
#   GR1_DATA_ROOT         GR1 LeRobot root (single dataset OR parent of datasets)
#   DATASET_PATH          alias for GR1_DATA_ROOT when GR1_DATA_ROOT is unset
#   QWEN_4B_MODEL_PATH    local Qwen3-VL-4B-Instruct snapshot or HF id
#   WAN_VAE_PATH          Wan2.2 VAE .pth (Wan-AI/Wan2.2-TI2V-5B)
#   WANDB_API_KEY         for online logging (TOML wandb_mode="online")
#   NPROC_PER_NODE        torchrun --nproc_per_node (default 8)
#   EXTRA_TAIL_OVERRIDES  space-separated Hydra overrides
#
# This recipe intentionally does not set BASE_CHECKPOINT_PATH: it fresh-inits
# from Qwen 4B HF weights, copies reasoner -> generator, and leaves adapters at
# their normal initial weights.
# ============================================================================

TOML_FILE="examples/toml/sft_config/gr1_robot_policy_merge_qwen3vl_4b.toml"
: "${DATASET_PATH:=/root/workspace/mengya/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot/}"
: "${QWEN_4B_MODEL_PATH:=/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17/}"
: "${WAN_VAE_PATH:=/root/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth}"
export QWEN_4B_MODEL_PATH WAN_VAE_PATH

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x /root/workspace/mengya/cosmos-framework/.venv/bin/python ]]; then
        PYTHON_BIN=/root/workspace/mengya/cosmos-framework/.venv/bin/python
    else
        PYTHON_BIN=python
    fi
fi
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "ERROR: python executable not found: $PYTHON_BIN" >&2; exit 1; }

# Some conda/env installs provide torch.distributed.run but not the torchrun
# console script. The shared launcher invokes `torchrun`, so provide a local
# function fallback before sourcing it.
if ! command -v torchrun >/dev/null 2>&1; then
    "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1 || { echo "ERROR: torchrun not found and $PYTHON_BIN -m torch.distributed.run is unavailable" >&2; exit 1; }
import torch.distributed.run
PY
    torchrun() {
        "$PYTHON_BIN" -m torch.distributed.run "$@"
    }
fi

# Local path guard. Qwen/Qwen3.5-4B has model_type=qwen3_5 and is not
# compatible with the Qwen3-VL MoT implementation used by this recipe.
if [[ -d "$QWEN_4B_MODEL_PATH" ]]; then
    [[ -f "$QWEN_4B_MODEL_PATH/config.json" ]] || { echo "ERROR: QWEN_4B_MODEL_PATH has no config.json: $QWEN_4B_MODEL_PATH" >&2; exit 1; }
    MODEL_TYPE="$("$PYTHON_BIN" - "$QWEN_4B_MODEL_PATH/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(json.load(f).get("model_type", ""))
PY
)"
    [[ "$MODEL_TYPE" == "qwen3_vl" ]] || { echo "ERROR: QWEN_4B_MODEL_PATH must be Qwen3-VL 4B compatible; found model_type=$MODEL_TYPE" >&2; exit 1; }
fi

# Base Qwen/VAE/tokenizer assets are local by default. Set HF_HUB_OFFLINE=0 if
# you intentionally want Hugging Face downloads during launch.
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
SITE="$("$PYTHON_BIN" -c 'import sysconfig; print(sysconfig.get_path("purelib"))' 2>/dev/null || true)"
if [[ -n "${SITE:-}" && -d "$SITE/nvidia" ]]; then
    export LD_LIBRARY_PATH="$SITE/av.libs:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/npp/lib:${LD_LIBRARY_PATH:-}"
fi

# Accept either a single LeRobot dataset (meta/info.json) or a parent directory of
# datasets (*/meta/info.json).
EXTRA_DATASET_CHECK='[[ -f "$GR1_DATA_ROOT/meta/info.json" ]] || compgen -G "$GR1_DATA_ROOT/*/meta/info.json" > /dev/null || { echo "ERROR: no GR1 LeRobot dataset under $GR1_DATA_ROOT (expected meta/info.json or */meta/info.json)" >&2; exit 1; }'

# Dataloader knobs (env-overridable). The nested path matches
# PackingDataLoader -> RankPartitionedDataLoader.
: "${MAX_SAMPLES_PER_BATCH:=128}"
: "${NUM_WORKERS:=16}"
: "${PREFETCH_FACTOR:=4}"
: "${PERSISTENT_WORKERS:=False}"

# Run name -> job.name (drives BOTH the output dir and the W&B run name).
: "${RUN_NAME:=gr1_robot_policy_merge_qwen3vl_4b}"

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

# Capture git provenance so cosmos_framework.utils.launch records commit/branch/diff
# into the W&B run config (JOB_INFO/*). launch.py only reads these files if present
# in CWD, so write them at the repo root before the trainer starts.
if git rev-parse --git-dir > /dev/null 2>&1; then
    git rev-parse HEAD > git_commit.txt
    git rev-parse --abbrev-ref HEAD > git_branch.txt
    git diff > git_diff.txt
fi

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
