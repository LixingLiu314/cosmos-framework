#!/bin/bash
# Cosmos3 GR1 v3 evaluation launcher
# Evaluates v3 posttrain: 29D action, per-dataset norm, image aug
set -e

EVAL_ROOT="/root/workspace/lixing/eval_results"
COSMOS_DIR="/root/workspace/lixing/cosmos-framework"
STARVLA_DIR="/root/workspace/lixing/starVLA"
VENV="/root/workspace/mengya/cosmos-framework/.venv/bin/activate"

# --- v3 checkpoint (update ITER as new checkpoints become available) ---
ITER="iter_000005000"
CKPT_V3="${COSMOS_DIR}/outputs/train/cosmos3/gr1_robot_policy_v3/gr1_robot_policy_posttrain_v3/checkpoints/${ITER}/"

# --- GPU / port ---
GPU_V3=0
PORT_V3=5684

# --- Output directories ---
EXP_OUT="${EVAL_ROOT}/gr1_robot_policy_posttrain_v3/${ITER}"
mkdir -p "${EXP_OUT}"/{policy_videos,client_debug,server_debug}

# --- Verify checkpoint exists ---
if [ ! -d "${CKPT_V3}" ]; then
    echo "ERROR: Checkpoint not found: ${CKPT_V3}"
    echo "Available checkpoints:"
    ls "${COSMOS_DIR}/outputs/train/cosmos3/gr1_robot_policy_v3/gr1_robot_policy_posttrain_v3/checkpoints/" 2>/dev/null || echo "  directory not found"
    exit 1
fi

# --- Server script ---
# NOTE: We use 'python -c' with sys.path.insert to override the editable install
# in mengya's venv (.pth file) that forces mengya's cosmos-framework to load first.
cat > /tmp/exp_v3_server.sh << EOF
#!/bin/bash
source ${VENV}
cd ${COSMOS_DIR}
CUDA_VISIBLE_DEVICES=${GPU_V3} HF_HUB_OFFLINE=1 python -c "
import sys; sys.path.insert(0, '${COSMOS_DIR}'); import cosmos_framework
import runpy; runpy.run_path('cosmos_framework/scripts/action_policy_server_robocasa_gr1.py', run_name='__main__')
" \
    --checkpoint-path ${CKPT_V3} \
    --action-schema gr1_29 \
    --port ${PORT_V3} \
    --model-input-debug-dir ${EXP_OUT}/server_debug \
    2>&1 | tee ${EXP_OUT}/server_log.txt
EOF
chmod +x /tmp/exp_v3_server.sh

# --- Kill old session if any ---
tmux kill-session -t exp_v3_server 2>/dev/null || true

# --- Launch server ---
tmux new-session -d -s exp_v3_server "bash /tmp/exp_v3_server.sh; bash"

echo "============================================"
echo "V3 Server launched:"
echo "  exp_v3_server: GPU ${GPU_V3}, port ${PORT_V3}, ${ITER}"
echo "  action-schema: gr1_29 (29D, per-dataset norm)"
echo "============================================"
echo "Waiting 120s for model to load..."
sleep 120

# --- Send client command into existing exp_v3_client tmux session ---
# NOTE: exp_v3_client must already have robocasa env activated manually
CLIENT_CMD="cd ${STARVLA_DIR} && PYTHONPATH=. python examples/Robocasa_tabletop/eval_files/simulation_env.py \
    --args.port ${PORT_V3} \
    --args.video-out-path ${EXP_OUT} \
    --args.policy-video-out-path ${EXP_OUT}/policy_videos \
    --args.policy-input-debug-dir ${EXP_OUT}/client_debug \
    2>&1 | tee ${EXP_OUT}/client_log.txt"

tmux send-keys -t exp_v3_client "${CLIENT_CMD}" Enter

echo ""
echo "Client command sent to exp_v3_client."
echo "Monitor with:"
echo "  tmux attach -t exp_v3_server"
echo "  tmux attach -t exp_v3_client"
