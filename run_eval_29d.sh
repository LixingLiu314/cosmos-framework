#!/bin/bash
# Cosmos3 GR1 29D evaluation launcher
# Evaluates the new 29D posttrain experiment
set -e

EVAL_ROOT="/root/workspace/lixing/eval_results"
COSMOS_DIR="/root/workspace/lixing/cosmos-framework"
STARVLA_DIR="/root/workspace/lixing/starVLA"
VENV="/root/workspace/mengya/cosmos-framework/.venv/bin/activate"

# --- 29D checkpoint (update ITER as new checkpoints become available) ---
ITER="iter_000005000"
CKPT_29D="/root/workspace/mengya/cosmos-framework/outputs/train/cosmos3/gr1_robot_policy_29d/gr1_robot_policy_posttrain_29d/checkpoints/${ITER}/"

# --- GPU / port ---
GPU_29D=0
PORT_29D=5684

# --- Output directories ---
EXP_OUT="${EVAL_ROOT}/gr1_robot_policy_posttrain_29d/${ITER}"
mkdir -p "${EXP_OUT}"/{policy_videos,client_debug,server_debug}

# --- Verify checkpoint exists ---
if [ ! -d "${CKPT_29D}" ]; then
    echo "ERROR: Checkpoint not found: ${CKPT_29D}"
    echo "Training is at ~iter 3494, save_iter=5000. First checkpoint not yet saved."
    echo "Check progress: tail -f /root/workspace/mengya/cosmos-framework/outputs/train/cosmos3/gr1_robot_policy_29d/gr1_robot_policy_posttrain_29d/wandb/run-*/files/output.log"
    exit 1
fi

# --- Server script ---
cat > /tmp/exp_29d_server.sh << EOF
#!/bin/bash
source ${VENV}
cd ${COSMOS_DIR}
CUDA_VISIBLE_DEVICES=${GPU_29D} HF_HUB_OFFLINE=1 python cosmos_framework/scripts/action_policy_server_robocasa_gr1.py \
    --checkpoint-path ${CKPT_29D} \
    --action-schema gr1_29 \
    --port ${PORT_29D} \
    --model-input-debug-dir ${EXP_OUT}/server_debug \
    2>&1 | tee ${EXP_OUT}/server_log.txt
EOF
chmod +x /tmp/exp_29d_server.sh

# --- Kill old session if any ---
tmux kill-session -t exp_29d_server 2>/dev/null || true

# --- Launch server ---
tmux new-session -d -s exp_29d_server "bash /tmp/exp_29d_server.sh; bash"

echo "============================================"
echo "29D Server launched:"
echo "  exp_29d_server: GPU ${GPU_29D}, port ${PORT_29D}, ${ITER}"
echo "  action-schema: gr1_29 (29D)"
echo "============================================"
echo "Waiting 120s for model to load..."
sleep 120

# --- Send client command into existing exp_29d_client tmux session ---
# NOTE: exp_29d_client must already have robocasa env activated manually
CLIENT_CMD="cd ${STARVLA_DIR} && PYTHONPATH=. python examples/Robocasa_tabletop/eval_files/simulation_env.py \
    --args.port ${PORT_29D} \
    --args.video-out-path ${EXP_OUT} \
    --args.policy-video-out-path ${EXP_OUT}/policy_videos \
    --args.policy-input-debug-dir ${EXP_OUT}/client_debug \
    2>&1 | tee ${EXP_OUT}/client_log.txt"

tmux send-keys -t exp_29d_client "${CLIENT_CMD}" Enter

echo ""
echo "Client command sent to exp_29d_client."
echo "Monitor with:"
echo "  tmux attach -t exp_29d_server"
echo "  tmux attach -t exp_29d_client"
