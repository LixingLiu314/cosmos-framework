#!/bin/bash
# Cosmos3 GR1 evaluation launcher
# Starts two servers, waits for model loading, then sends client commands to existing tmux sessions.
set -e

EVAL_ROOT="/root/workspace/lixing/eval_results"
COSMOS_DIR="/root/workspace/lixing/cosmos-framework"
STARVLA_DIR="/root/workspace/lixing/starVLA"

CKPT_POSTTRAIN="/root/workspace/mengya/cosmos-framework/outputs/train/cosmos3/gr1_robot_policy/gr1_robot_policy_posttrain/checkpoints/iter_000040000/"
CKPT_ALLW="/root/workspace/mengya/cosmos-framework/outputs/train/cosmos3/gr1_robot_policy/gr1_robot_policy_posttrain_allw/checkpoints/iter_000045000/"

GPU_POSTTRAIN=0
GPU_ALLW=2
PORT_POSTTRAIN=5682
PORT_ALLW=5683

EXP1_OUT="${EVAL_ROOT}/gr1_robot_policy_posttrain/iter_000040000"
EXP2_OUT="${EVAL_ROOT}/gr1_robot_policy_posttrain_allw/iter_000045000"

mkdir -p "${EXP1_OUT}"/{policy_videos,client_debug,server_debug}
mkdir -p "${EXP2_OUT}"/{policy_videos,client_debug,server_debug}

# --- Server helper scripts ---
cat > /tmp/exp1_server.sh << EOF
#!/bin/bash
source /root/workspace/mengya/cosmos-framework/.venv/bin/activate
cd ${COSMOS_DIR}
CUDA_VISIBLE_DEVICES=${GPU_POSTTRAIN} HF_HUB_OFFLINE=1 python cosmos_framework/scripts/action_policy_server_robocasa_gr1.py --checkpoint-path ${CKPT_POSTTRAIN} --port ${PORT_POSTTRAIN} --model-input-debug-dir ${EXP1_OUT}/server_debug 2>&1 | tee ${EXP1_OUT}/server_log.txt
EOF

cat > /tmp/exp2_server.sh << EOF
#!/bin/bash
source /root/workspace/mengya/cosmos-framework/.venv/bin/activate
cd ${COSMOS_DIR}
CUDA_VISIBLE_DEVICES=${GPU_ALLW} HF_HUB_OFFLINE=1 python cosmos_framework/scripts/action_policy_server_robocasa_gr1.py --checkpoint-path ${CKPT_ALLW} --port ${PORT_ALLW} --model-input-debug-dir ${EXP2_OUT}/server_debug 2>&1 | tee ${EXP2_OUT}/server_log.txt
EOF

chmod +x /tmp/exp1_server.sh /tmp/exp2_server.sh

# --- Kill old sessions if any ---
tmux kill-session -t exp1_server 2>/dev/null || true
tmux kill-session -t exp2_server 2>/dev/null || true

# --- Launch servers ---
tmux new-session -d -s exp1_server "bash /tmp/exp1_server.sh; bash"
tmux new-session -d -s exp2_server "bash /tmp/exp2_server.sh; bash"

echo "============================================"
echo "Servers launched:"
echo "  exp1_server: GPU ${GPU_POSTTRAIN}, port ${PORT_POSTTRAIN}, posttrain iter_40000"
echo "  exp2_server: GPU ${GPU_ALLW}, port ${PORT_ALLW}, posttrain_allw iter_45000"
echo "============================================"
echo "Waiting 120s for models to load..."
sleep 120

# --- Send client commands into existing exp1_client / exp2_client tmux sessions ---
# NOTE: exp1_client and exp2_client must already have robocasa env activated manually
CLIENT_CMD_1="cd ${STARVLA_DIR} && PYTHONPATH=. python examples/Robocasa_tabletop/eval_files/simulation_env.py --args.port ${PORT_POSTTRAIN} --args.video-out-path ${EXP1_OUT} --args.policy-video-out-path ${EXP1_OUT}/policy_videos --args.policy-input-debug-dir ${EXP1_OUT}/client_debug 2>&1 | tee ${EXP1_OUT}/client_log.txt"

CLIENT_CMD_2="cd ${STARVLA_DIR} && PYTHONPATH=. python examples/Robocasa_tabletop/eval_files/simulation_env.py --args.port ${PORT_ALLW} --args.video-out-path ${EXP2_OUT} --args.policy-video-out-path ${EXP2_OUT}/policy_videos --args.policy-input-debug-dir ${EXP2_OUT}/client_debug 2>&1 | tee ${EXP2_OUT}/client_log.txt"

tmux send-keys -t exp1_client "${CLIENT_CMD_1}" Enter
tmux send-keys -t exp2_client "${CLIENT_CMD_2}" Enter

echo ""
echo "Client commands sent to exp1_client and exp2_client."
echo "Monitor with:"
echo "  tmux attach -t exp1_server"
echo "  tmux attach -t exp2_server"
echo "  tmux attach -t exp1_client"
echo "  tmux attach -t exp2_client"
