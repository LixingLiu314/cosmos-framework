#!/bin/bash
# Cosmos3 GR1 v2 evaluation launcher
# Evaluates the v2 posttrain experiment (44D, loss_scale=10, image aug)
set -e

EVAL_ROOT="/root/workspace/lixing/eval_results"
COSMOS_DIR="/root/workspace/lixing/cosmos-framework"
STARVLA_DIR="/root/workspace/lixing/starVLA"
VENV="/root/workspace/mengya/cosmos-framework/.venv/bin/activate"

# --- v2 checkpoint (update ITER as new checkpoints become available) ---
ITER="iter_000005000"
CKPT_V2="${COSMOS_DIR}/outputs/train/cosmos3/gr1_robot_policy_v2_policy/gr1_robot_policy_posttrain_v2_policy/checkpoints/${ITER}/"

# --- GPU / port ---
GPU_V2=0
PORT_V2=5684

# --- Output directories ---
EXP_OUT="${EVAL_ROOT}/gr1_robot_policy_posttrain_v2/${ITER}"
mkdir -p "${EXP_OUT}"/{policy_videos,client_debug,server_debug}

# --- Verify checkpoint exists ---
if [ ! -d "${CKPT_V2}" ]; then
    echo "ERROR: Checkpoint not found: ${CKPT_V2}"
    echo "Available checkpoints:"
    ls "${COSMOS_DIR}/outputs/train/cosmos3/gr1_robot_policy_v2_policy/gr1_robot_policy_posttrain_v2_policy/checkpoints/" 2>/dev/null || echo "  directory not found"
    exit 1
fi

# --- Server script ---
# NOTE: We use 'python -c' with sys.path.insert to override the editable install
# in mengya's venv (.pth file) that forces mengya's cosmos-framework to load first.
cat > /tmp/exp_v2_server.sh << EOF
#!/bin/bash
source ${VENV}
cd ${COSMOS_DIR}
CUDA_VISIBLE_DEVICES=${GPU_V2} HF_HUB_OFFLINE=1 python -c "
import sys; sys.path.insert(0, '${COSMOS_DIR}'); import cosmos_framework
import runpy; runpy.run_path('cosmos_framework/scripts/action_policy_server_robocasa_gr1.py', run_name='__main__')
" \
    --checkpoint-path ${CKPT_V2} \
    --action-schema gr1_44 \
    --port ${PORT_V2} \
    --model-input-debug-dir ${EXP_OUT}/server_debug \
    2>&1 | tee ${EXP_OUT}/server_log.txt
EOF
chmod +x /tmp/exp_v2_server.sh

# --- Kill old session if any ---
tmux kill-session -t exp_v2_server 2>/dev/null || true

# --- Launch server ---
tmux new-session -d -s exp_v2_server "bash /tmp/exp_v2_server.sh; bash"

echo "============================================"
echo "V2 Server launched:"
echo "  exp_v2_server: GPU ${GPU_V2}, port ${PORT_V2}, ${ITER}"
echo "  action-schema: gr1_44 (44D, matching v2 training)"
echo "============================================"
echo "Waiting 120s for model to load..."
sleep 120

# --- Send client command into existing exp_v2_client tmux session ---
# NOTE: exp_v2_client must already have robocasa env activated manually
CLIENT_CMD="cd ${STARVLA_DIR} && PYTHONPATH=. python examples/Robocasa_tabletop/eval_files/simulation_env.py \
    --args.port ${PORT_V2} \
    --args.video-out-path ${EXP_OUT} \
    --args.policy-video-out-path ${EXP_OUT}/policy_videos \
    --args.policy-input-debug-dir ${EXP_OUT}/client_debug \
    2>&1 | tee ${EXP_OUT}/client_log.txt"

tmux send-keys -t exp_v2_client "${CLIENT_CMD}" Enter

echo ""
echo "Client command sent to exp_v2_client."
echo "Monitor with:"
echo "  tmux attach -t exp_v2_server"
echo "  tmux attach -t exp_v2_client"
