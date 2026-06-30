#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Parallel RoboCasa-GR1 evaluation launcher for the 29D Cosmos3 policy
# (experiment `gr1_robot_policy_merge`, trained via launch_gr1_robot_policy_merge.sh).
#
# Architecture (websocket, two python envs):
#   [Cosmos3 policy server]  <-- ws -->  [RoboCasa sim client]
#    lixing cosmos_framework               lzd starVLA
#    mengya shared .venv                   robocasa conda env
#
# The server is lixing's standalone gr1_29 server (action_policy_server_robocasa_gr1.py):
# 29D action, per-dataset minmax denorm for BOTH action and state, use_state history.
# These are resolved per-example from {ACTION_STATE_STATS_ROOT}/{dataset}/meta/stats.json
# by env name -> matches 29D training (per-dataset, dual action/state stats).
#
# Common usage (local ckpt + cached VAE, offline):
#   GPU_IDS="0 1 2 3 4 5 6 7" N_EPISODES=50 \
#     bash examples/run_robocasa_parallel_eval_gr1_29.sh
#
# Fast planning check:
#   DRY_RUN=1 GPU_IDS="6 7" bash examples/run_robocasa_parallel_eval_gr1_29.sh
#
# Override the checkpoint:
#   CHECKPOINT_PATH=/abs/path/to/checkpoints/iter_000016000 \
#     GPU_IDS="0" bash examples/run_robocasa_parallel_eval_gr1_29.sh
# ============================================================================

# --- repos / interpreters ---------------------------------------------------
STARVLA_REPO="${STARVLA_REPO:-/root/workspace/lzd/starVLA}"
COSMOS_REPO="${COSMOS_REPO:-/root/workspace/lixing/cosmos-framework}"
# Shared venv (no lixing venv exists). The gr1_29 logic lives in the server
# script itself, which is run by path from COSMOS_REPO, so the per-dataset
# 29D denorm/state behaviour comes from lixing's code regardless of the
# venv editable-install resolution.
COSMOS_PYTHON="${COSMOS_PYTHON:-/root/workspace/mengya/cosmos-framework/.venv/bin/python}"
ROBOCASA_PYTHON="${ROBOCASA_PYTHON:-/root/miniconda3/envs/robocasa/bin/python}"
COSMOS_SERVER="${COSMOS_SERVER:-${COSMOS_REPO}/cosmos_framework/scripts/action_policy_server_robocasa_gr1.py}"

# --- checkpoint (local, offline by default) ---------------------------------
CHECKPOINT_REPO="${CHECKPOINT_REPO:-nvidia/Cosmos3-Nano-Policy-DROID}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${COSMOS_REPO}/outputs/train/cosmos3/gr1_robot_policy/gr1_bs128_worker16_perfetch4_persisfalse_episodeencodecache_hue/checkpoints/iter_000020000}"
PRETRAINED_PATH="${PRETRAINED_PATH:-${CHECKPOINT_PATH:-${CHECKPOINT_REPO}}}"
ALLOW_DOWNLOAD="${ALLOW_DOWNLOAD:-0}"   # 0 = local ckpt + cached VAE, no Hub calls

# --- networking / orchestration --------------------------------------------
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"   # 8-GPU parallel: 24 tasks round-robin -> 3 tasks/server
BASE_PORT="${BASE_PORT:-5682}"
DIST_BASE_PORT="${DIST_BASE_PORT:-41000}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
CLIENT_HOST="${CLIENT_HOST:-127.0.0.1}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-600}"
START_SERVERS="${START_SERVERS:-1}"
KEEP_SERVERS_ON_EXIT="${KEEP_SERVERS_ON_EXIT:-0}"
DRY_RUN="${DRY_RUN:-0}"
FAIL_FAST="${FAIL_FAST:-0}"

# --- rollout knobs ----------------------------------------------------------
N_EPISODES="${N_EPISODES:-50}"
N_ENVS="${N_ENVS:-1}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
N_ACTION_STEPS="${N_ACTION_STEPS:-16}"
ACTION_HORIZON="${ACTION_HORIZON:-16}"

# --- 29D schema (locked defaults; match training) ---------------------------
ACTION_SCHEMA="${ACTION_SCHEMA:-gr1_29}"
ACTION_DENORM="${ACTION_DENORM:-auto}"     # auto -> minmax (per-dataset)
ACTION_HISTORY="${ACTION_HISTORY:-auto}"   # auto -> state
# Root holding per-dataset stats.json; BOTH action and state stats for gr1_29
# are resolved per-example from {root}/{dataset}/meta/stats.json by env name.
ACTION_STATE_STATS_ROOT="${ACTION_STATE_STATS_ROOT:-${GR1_DATA_ROOT:-/root/workspace/mengya/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot}}"
ACTION_STATE_STATS_PATH="${ACTION_STATE_STATS_PATH:-}"  # explicit override (skip per-example)
ACTION_STATS_PATH="${ACTION_STATS_PATH:-}"              # explicit override (skip per-example)
RAW_ACTION_DIM="${RAW_ACTION_DIM:-}"  # empty -> schema default (29)
DOMAIN_ID="${DOMAIN_ID:-}"            # empty -> schema default (31)

# --- sampling ---------------------------------------------------------------
NUM_STEPS="${NUM_STEPS:-30}"
GUIDANCE="${GUIDANCE:-1.0}"
SHIFT="${SHIFT:-5.0}"
SEED="${SEED:-0}"
DETERMINISTIC_SEED="${DETERMINISTIC_SEED:-0}"
ACTION_CLAMP="${ACTION_CLAMP:-0}"
DECODE_VIDEO="${DECODE_VIDEO:-0}"   # 0 -> --no-decode-video (faster eval)
USE_EMA_WEIGHTS="${USE_EMA_WEIGHTS:-0}"
ACTION_DEBUG="${ACTION_DEBUG:-0}"
ACTION_DEBUG_LIMIT="${ACTION_DEBUG_LIMIT:-5}"

# --- image / obs ------------------------------------------------------------
RESOLUTION="${RESOLUTION:-256}"
MUJOCO_GL="${MUJOCO_GL:-egl}"
RESIZE_SIZE="${RESIZE_SIZE:-256 256}"
IMAGE_OBS_KEY="${IMAGE_OBS_KEY:-video.ego_view_bg_crop_pad_res256_freq20}"
IMAGE_RESIZE_MODE="${IMAGE_RESIZE_MODE:-preserve}"
POLICY_INPUT_EXPECTED_SIZE="${POLICY_INPUT_EXPECTED_SIZE:-256 256}"

# --- HF offline (local ckpt + cached Wan2.2 VAE) ----------------------------
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"

# --- outputs ----------------------------------------------------------------
RUN_ID="${RUN_ID:-eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${COSMOS_REPO}/outputs/cosmos3_robocasa_eval_gr1_29/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
VIDEO_ROOT="${VIDEO_ROOT:-${OUTPUT_ROOT}/videos}"
POLICY_VIDEO_ROOT="${POLICY_VIDEO_ROOT:-${OUTPUT_ROOT}/policy_videos}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/summary.tsv}"
TASKS_FILE="${TASKS_FILE:-}"
TASK_LIMIT="${TASK_LIMIT:-0}"

# All 24 GR1 tasks (full RoboCasa env names; server maps each to its
# gr1_unified.<Task> stats dir via the _GR1 split rule). Verified 1:1 against
# the LeRobot dataset dirs and the starVLA env registry (2026-06-28).
# Override with TASKS_FILE=<file> (one env name per line) or TASK_LIMIT=<n>.
ENV_NAMES=(
  gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
)

declare -a GPU_LIST SERVER_GPUS SERVER_PORTS SERVER_PIDS SERVER_LOGS
declare -a TASKS EVAL_PIDS EVAL_LOGS EVAL_ENVS EVAL_GPUS EVAL_PORTS EVAL_VIDEOS EVAL_POLICY_VIDEOS
declare -a RESIZE_ARGS POLICY_INPUT_EXPECTED_SIZE_ARGS
FAILED=0

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

sanitize_name() {
  local value="$1"; value="${value//\//_}"; value="${value// /_}"; printf '%s' "$value"
}

load_tasks() {
  TASKS=()
  if [[ -n "${TASKS_FILE}" ]]; then
    [[ -f "${TASKS_FILE}" ]] || { echo "TASKS_FILE does not exist: ${TASKS_FILE}" >&2; exit 2; }
    while IFS= read -r line || [[ -n "${line}" ]]; do
      line="${line%%#*}"; line="${line#"${line%%[![:space:]]*}"}"; line="${line%"${line##*[![:space:]]}"}"
      [[ -z "${line}" ]] && continue
      TASKS+=("${line}")
    done < "${TASKS_FILE}"
  else
    TASKS=("${ENV_NAMES[@]}")
  fi
  if [[ "${TASK_LIMIT}" != "0" ]]; then
    (( TASK_LIMIT >= 1 )) || { echo "TASK_LIMIT must be 0 or positive, got ${TASK_LIMIT}" >&2; exit 2; }
    TASKS=("${TASKS[@]:0:${TASK_LIMIT}}")
  fi
  (( ${#TASKS[@]} > 0 )) || { echo "No RoboCasa tasks selected." >&2; exit 2; }
}

configure_servers() {
  read -r -a GPU_LIST <<< "${GPU_IDS}"
  (( ${#GPU_LIST[@]} > 0 )) || { echo "GPU_IDS must contain at least one GPU id." >&2; exit 2; }
  SERVER_GPUS=(); SERVER_PORTS=()
  for idx in "${!GPU_LIST[@]}"; do
    SERVER_GPUS[$idx]="${GPU_LIST[$idx]}"
    SERVER_PORTS[$idx]="$((BASE_PORT + idx))"
  done
}

configure_resize() {
  read -r -a RESIZE_ARGS <<< "${RESIZE_SIZE}"
  (( ${#RESIZE_ARGS[@]} == 2 )) || { echo "RESIZE_SIZE must be two ints, got: ${RESIZE_SIZE}" >&2; exit 2; }
  read -r -a POLICY_INPUT_EXPECTED_SIZE_ARGS <<< "${POLICY_INPUT_EXPECTED_SIZE}"
  (( ${#POLICY_INPUT_EXPECTED_SIZE_ARGS[@]} == 2 )) || { echo "POLICY_INPUT_EXPECTED_SIZE must be two ints, got: ${POLICY_INPUT_EXPECTED_SIZE}" >&2; exit 2; }
}

print_plan() {
  echo "RoboCasa Cosmos3 gr1_29 parallel eval plan"
  echo "run_id=${RUN_ID}"
  echo "server_script=${COSMOS_SERVER}"
  echo "cosmos_python=${COSMOS_PYTHON}"
  echo "robocasa_python=${ROBOCASA_PYTHON}"
  echo "checkpoint_path=${CHECKPOINT_PATH:-<unset>}"
  echo "allow_download=${ALLOW_DOWNLOAD}"
  echo "episodes_per_task=${N_EPISODES}  n_envs=${N_ENVS}  max_episode_steps=${MAX_EPISODE_STEPS}  n_action_steps=${N_ACTION_STEPS}"
  echo "action_schema=${ACTION_SCHEMA}  action_denorm=${ACTION_DENORM}  action_history=${ACTION_HISTORY}"
  echo "action_state_stats_root=${ACTION_STATE_STATS_ROOT}"
  echo "raw_action_dim=${RAW_ACTION_DIM:-<schema-default 29>}  domain_id=${DOMAIN_ID:-<schema-default 31>}"
  echo "num_steps=${NUM_STEPS}  guidance=${GUIDANCE}  shift=${SHIFT}  seed=${SEED}  ema=${USE_EMA_WEIGHTS}"
  echo "decode_video=${DECODE_VIDEO}  resolution=${RESOLUTION}  image_obs_key=${IMAGE_OBS_KEY}"
  echo "log_dir=${LOG_DIR}"
  echo
  echo "Servers (${#SERVER_GPUS[@]})"
  for idx in "${!SERVER_GPUS[@]}"; do
    echo "  server[${idx}]: gpu=${SERVER_GPUS[$idx]} port=${SERVER_PORTS[$idx]} dist_port=$((DIST_BASE_PORT + idx))"
  done
  echo "Tasks (${#TASKS[@]})"
  for idx in "${!TASKS[@]}"; do
    local slot=$((idx % ${#SERVER_GPUS[@]}))
    echo "  task[${idx}]: port=${SERVER_PORTS[$slot]} gpu=${SERVER_GPUS[$slot]} env=${TASKS[$idx]}"
  done
  if [[ "${DRY_RUN}" == "1" ]]; then echo; echo "DRY_RUN=1; no processes launched."; fi
}

checkpoint_args() {
  if [[ -n "${CHECKPOINT_PATH}" ]]; then
    printf '%s\n' "--checkpoint-path" "${CHECKPOINT_PATH}"
  else
    printf '%s\n' "--checkpoint-repo" "${CHECKPOINT_REPO}"
  fi
  [[ "${ALLOW_DOWNLOAD}" == "1" ]] && printf '%s\n' "--allow-download"
}

wait_for_port() {
  local host="$1" port="$2" timeout_s="$3"
  "${COSMOS_PYTHON}" - "${host}" "${port}" "${timeout_s}" <<'PY'
import socket, sys, time
host, port, timeout_s = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
deadline = time.time() + timeout_s; last = None
while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=2.0):
            sys.exit(0)
    except OSError as exc:
        last = exc; time.sleep(2.0)
print(f"Timed out waiting for {host}:{port}: {last}", file=sys.stderr); sys.exit(1)
PY
}

start_server() {
  local idx="$1" gpu="${SERVER_GPUS[$1]}" port="${SERVER_PORTS[$1]}"
  local dist_port=$((DIST_BASE_PORT + idx))
  local log_file="${LOG_DIR}/server_gpu${gpu}_port${port}.log"
  local ckpt_args=(); mapfile -t ckpt_args < <(checkpoint_args)

  local action_args=(--action-schema "${ACTION_SCHEMA}" --action-denorm "${ACTION_DENORM}" --action-history "${ACTION_HISTORY}")
  [[ -n "${ACTION_STATE_STATS_ROOT}" ]] && action_args+=(--action-state-stats-root "${ACTION_STATE_STATS_ROOT}")
  [[ -n "${ACTION_STATE_STATS_PATH}" ]] && action_args+=(--action-state-stats-path "${ACTION_STATE_STATS_PATH}")
  [[ -n "${ACTION_STATS_PATH}" ]] && action_args+=(--action-stats-path "${ACTION_STATS_PATH}")
  [[ -n "${RAW_ACTION_DIM}" ]] && action_args+=(--raw-action-dim "${RAW_ACTION_DIM}")
  [[ -n "${DOMAIN_ID}" ]] && action_args+=(--domain-id "${DOMAIN_ID}")
  [[ "${DETERMINISTIC_SEED}" == "1" ]] && action_args+=(--deterministic-seed)
  [[ "${ACTION_CLAMP}" == "1" ]] && action_args+=(--action-clamp)
  [[ "${USE_EMA_WEIGHTS}" == "1" ]] && action_args+=(--use-ema-weights)
  if [[ "${DECODE_VIDEO}" == "1" ]]; then action_args+=(--decode-video); else action_args+=(--no-decode-video); fi

  log "Starting gr1_29 server idx=${idx} gpu=${gpu} port=${port} schema=${ACTION_SCHEMA} denorm=${ACTION_DENORM} history=${ACTION_HISTORY}"
  (
    cd "${COSMOS_REPO}"
    export STARVLA_REPO COSMOS_REPO
    # COSMOS first so lixing's package is preferred; STARVLA provides deployment.model_server.
    export PYTHONPATH="${COSMOS_REPO}:${STARVLA_REPO}:${PYTHONPATH:-}"
    export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PATH="/root/.local/bin:/root/miniconda3/bin:${PATH}"
    export MASTER_ADDR="127.0.0.1" MASTER_PORT="${dist_port}" WORLD_SIZE="1" RANK="0" LOCAL_RANK="0"
    export ACTION_DEBUG="${ACTION_DEBUG}" ACTION_DEBUG_LIMIT="${ACTION_DEBUG_LIMIT}"
    # torchcodec needs the venv's bundled ffmpeg + CUDA NPP/runtime libs on the loader path.
    SITE="$("${COSMOS_PYTHON}" -c 'import sysconfig; print(sysconfig.get_path("purelib"))' 2>/dev/null || true)"
    if [[ -n "${SITE:-}" && -d "$SITE/nvidia" ]]; then
      export LD_LIBRARY_PATH="$SITE/av.libs:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/npp/lib:${LD_LIBRARY_PATH:-}"
    fi
    exec "${COSMOS_PYTHON}" "${COSMOS_SERVER}" \
      --host "${SERVER_HOST}" --port "${port}" \
      "${ckpt_args[@]}" "${action_args[@]}" \
      --num-steps "${NUM_STEPS}" --guidance "${GUIDANCE}" --shift "${SHIFT}" --seed "${SEED}" \
      --resolution "${RESOLUTION}" --action-horizon "${ACTION_HORIZON}"
  ) > "${log_file}" 2>&1 &
  SERVER_PIDS[$idx]=$!
  SERVER_LOGS[$idx]="${log_file}"
}

start_servers() {
  if [[ "${START_SERVERS}" != "1" ]]; then log "START_SERVERS=0; using already running servers."; return; fi
  for idx in "${!SERVER_GPUS[@]}"; do start_server "${idx}"; done
  for idx in "${!SERVER_GPUS[@]}"; do
    local pid="${SERVER_PIDS[$idx]}" port="${SERVER_PORTS[$idx]}" log_file="${SERVER_LOGS[$idx]}"
    log "Waiting for server idx=${idx} pid=${pid} port=${port}"
    if ! wait_for_port "${CLIENT_HOST}" "${port}" "${SERVER_START_TIMEOUT}"; then
      echo "Server failed to become ready: idx=${idx} pid=${pid} log=${log_file}" >&2
      tail -n 120 "${log_file}" >&2 || true; exit 1
    fi
    kill -0 "${pid}" 2>/dev/null || { echo "Server exited after opening port: idx=${idx} log=${log_file}" >&2; tail -n 120 "${log_file}" >&2 || true; exit 1; }
    log "Server ready idx=${idx} port=${port}"
  done
}

write_summary_header() {
  mkdir -p "$(dirname "${SUMMARY_PATH}")"
  printf 'task\tgpu\tport\texit_code\tsuccess_rate\tlog\tvideo_dir\tpolicy_video_dir\n' > "${SUMMARY_PATH}"
}

wait_eval_slot() {
  local slot="$1" pid="${EVAL_PIDS[$1]:-}"
  [[ -z "${pid}" ]] && return 0
  local env_name="${EVAL_ENVS[$slot]}" gpu="${EVAL_GPUS[$slot]}" port="${EVAL_PORTS[$slot]}"
  local log_file="${EVAL_LOGS[$slot]}" video_dir="${EVAL_VIDEOS[$slot]}" policy_video_dir="${EVAL_POLICY_VIDEOS[$slot]:-}"
  local exit_code=0 success_rate="NA" success_line=""
  if wait "${pid}"; then exit_code=0; else exit_code=$?; FAILED=1; fi
  success_line="$(grep -E 'Success rate:' "${log_file}" 2>/dev/null | tail -n 1 || true)"
  [[ -n "${success_line}" ]] && success_rate="${success_line##*Success rate: }"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${env_name}" "${gpu}" "${port}" "${exit_code}" "${success_rate}" "${log_file}" "${video_dir}" "${policy_video_dir}" >> "${SUMMARY_PATH}"
  if (( exit_code == 0 )); then
    log "Eval finished gpu=${gpu} port=${port} success_rate=${success_rate} env=${env_name}"
  else
    log "Eval FAILED exit=${exit_code} gpu=${gpu} port=${port} env=${env_name}; log=${log_file}"
    tail -n 120 "${log_file}" >&2 || true
    [[ "${FAIL_FAST}" == "1" ]] && exit "${exit_code}"
  fi
  unset "EVAL_PIDS[$slot]" "EVAL_LOGS[$slot]" "EVAL_ENVS[$slot]" "EVAL_GPUS[$slot]" "EVAL_PORTS[$slot]" "EVAL_VIDEOS[$slot]" "EVAL_POLICY_VIDEOS[$slot]"
}

run_eval_async() {
  local task_idx="$1" slot=$(( $1 % ${#SERVER_GPUS[@]} ))
  local env_name="${TASKS[$task_idx]}" gpu="${SERVER_GPUS[$slot]}" port="${SERVER_PORTS[$slot]}"
  local task_name="${env_name##*/}" safe_name; safe_name="$(sanitize_name "${task_name}")"
  local log_file="${LOG_DIR}/eval_${safe_name}_gpu${gpu}_port${port}.log"
  local video_dir="${VIDEO_ROOT}/${safe_name}" policy_video_dir="${POLICY_VIDEO_ROOT}/${safe_name}"

  wait_eval_slot "${slot}"
  mkdir -p "${video_dir}"
  [[ "${DECODE_VIDEO}" == "1" ]] && mkdir -p "${policy_video_dir}"
  log "Launching eval task=${task_idx} gpu=${gpu} port=${port} env=${env_name}"
  (
    cd "${STARVLA_REPO}"
    export PYTHONPATH="${STARVLA_REPO}:${PYTHONPATH:-}"
    export MUJOCO_GL="${MUJOCO_GL}"
    exec "${ROBOCASA_PYTHON}" examples/Robocasa_tabletop/eval_files/simulation_env.py \
      --args.env_name "${env_name}" \
      --args.host "${CLIENT_HOST}" --args.port "${port}" \
      --args.n_episodes "${N_EPISODES}" --args.n_envs "${N_ENVS}" \
      --args.max_episode_steps "${MAX_EPISODE_STEPS}" --args.n_action_steps "${N_ACTION_STEPS}" \
      --args.resize_size "${RESIZE_ARGS[@]}" \
      --args.image_obs_key "${IMAGE_OBS_KEY}" --args.image_resize_mode "${IMAGE_RESIZE_MODE}" \
      --args.policy_input_expected_size "${POLICY_INPUT_EXPECTED_SIZE_ARGS[@]}" \
      --args.video_out_path "${video_dir}" --args.policy_video_out_path "${policy_video_dir}" \
      --args.pretrained_path "${PRETRAINED_PATH}"
  ) > "${log_file}" 2>&1 &
  EVAL_PIDS[$slot]=$!
  EVAL_LOGS[$slot]="${log_file}"; EVAL_ENVS[$slot]="${env_name}"; EVAL_GPUS[$slot]="${gpu}"
  EVAL_PORTS[$slot]="${port}"; EVAL_VIDEOS[$slot]="${video_dir}"; EVAL_POLICY_VIDEOS[$slot]="${policy_video_dir}"
}

wait_all_evals() { for slot in "${!SERVER_GPUS[@]}"; do wait_eval_slot "${slot}"; done; }

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  for pid in "${EVAL_PIDS[@]}"; do [[ -n "${pid:-}" ]] && kill "${pid}" 2>/dev/null || true; done
  if [[ "${START_SERVERS}" == "1" && "${KEEP_SERVERS_ON_EXIT}" != "1" ]]; then
    for pid in "${SERVER_PIDS[@]}"; do [[ -n "${pid:-}" ]] && kill "${pid}" 2>/dev/null || true; done
  fi
  exit "${exit_code}"
}

main() {
  load_tasks; configure_servers; configure_resize; print_plan
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  mkdir -p "${LOG_DIR}" "${VIDEO_ROOT}"
  if [[ "${DECODE_VIDEO}" == "1" ]]; then mkdir -p "${POLICY_VIDEO_ROOT}"; fi
  write_summary_header
  trap cleanup EXIT INT TERM
  start_servers
  for idx in "${!TASKS[@]}"; do run_eval_async "${idx}"; done
  wait_all_evals
  log "Summary written to ${SUMMARY_PATH}"
  (( FAILED == 0 )) || exit 1
}

main "$@"
