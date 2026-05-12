#!/bin/bash
# Run all HelpSteer3 experiments: BT, MeanVar, and anchor variants
# Phase 1: all training runs (serial, full GPU)
# Phase 2: all evaluations (parallel, 1 task per GPU, batches of NUM_GPUS)
# Hardware: 4x NVIDIA A800 80GB PCIe  (GPU 0 excluded — see GPU_IDS below)
# Usage: bash scripts/run_anchor_helpsteer3_split.sh

set -e

# echo "Sleeping for 2 hours before starting..."
# sleep 1200

SCRIPT_START_TIME=$(date +%s)

# ====== Configuration ======
MODEL_NAMES=(
    # "NousResearch/Meta-Llama-3.1-8B-Instruct"
    "unsloth/Llama-3.2-1B-Instruct"
    # "unsloth/Llama-3.2-3B-Instruct"
)
SEEDS=(1)
ANCHOR_MODEL_NAMES=(
    # "Skywork/Skywork-Reward-V2-Llama-3.2-3B"
    # "deepseek-v4-pro"
    "deepseek-v4-flash"
    # "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
)
ANCHOR_LAMBDAS=(0.1)
LEARNING_RATES=(1e-5)

# GPUs to use (GPU 0 excluded — in use by another process)
GPU_IDS=(0 1 2 3)
NUM_GPUS=${#GPU_IDS[@]}
CUDA_DEVICES=$(IFS=,; echo "${GPU_IDS[*]}")
# ===========================

# ====== Helper: checkpoint dir ======
ckpt_dir() {
    local run_name="$1" model_short="$2" seed="$3" lr="$4" anchor_short="$5" anchor_lam="$6"
    if [ -n "$anchor_short" ]; then
        echo "checkpoints/${run_name}_${model_short}_anchor-${anchor_short}_lam${anchor_lam}_lr${lr}_seed${seed}"
    else
        echo "checkpoints/${run_name}_${model_short}_lr${lr}_seed${seed}"
    fi
}

# ====== Helper: flush EVAL_QUEUE dynamically, keep GPUs busy ======
flush_eval_queue() {
    local next_task=0
    local num_tasks=${#EVAL_QUEUE[@]}

    local -a pids=()
    local -a pid_gpus=()

    launch_task() {
        local gpu_id="$1"
        local cmd="${EVAL_QUEUE[$next_task]}"

        echo "  [GPU ${gpu_id}] ${cmd}"
        CUDA_VISIBLE_DEVICES=${gpu_id} bash -c "${cmd}" &

        pids+=($!)
        pid_gpus+=("${gpu_id}")
        next_task=$((next_task + 1))
    }

    # launch initial tasks, one per GPU
    for gpu_id in "${GPU_IDS[@]}"; do
        if [ "${next_task}" -lt "${num_tasks}" ]; then
            launch_task "${gpu_id}"
        fi
    done

    # whenever one finishes, launch the next task on that freed GPU
    while [ ${#pids[@]} -gt 0 ]; do
        local finished_index=-1

        for i in "${!pids[@]}"; do
            local pid="${pids[$i]}"

            if ! kill -0 "${pid}" 2>/dev/null; then
                wait "${pid}"
                finished_index="${i}"
                break
            fi
        done

        if [ "${finished_index}" -ge 0 ]; then
            local freed_gpu="${pid_gpus[$finished_index]}"

            unset 'pids[finished_index]'
            unset 'pid_gpus[finished_index]'

            # compact arrays after unset
            pids=("${pids[@]}")
            pid_gpus=("${pid_gpus[@]}")

            if [ "${next_task}" -lt "${num_tasks}" ]; then
                launch_task "${freed_gpu}"
            fi
        else
            sleep 5
        fi
    done

    EVAL_QUEUE=()
}

# ====== EVAL_QUEUE accumulator ======
EVAL_QUEUE=()

echo "============================================"
echo "  HelpSteer3 Experiment Suite (BT / MV / Anchor)"
echo "  Models:         ${MODEL_NAMES[*]}"
echo "  Seeds:          ${SEEDS[*]}"
echo "  Learning rates: ${LEARNING_RATES[*]}"
echo "  Anchor models:  ${ANCHOR_MODEL_NAMES[*]}"
echo "  Anchor lambdas: ${ANCHOR_LAMBDAS[*]}"
echo "  GPUs:           ${GPU_IDS[*]}  (CUDA_VISIBLE_DEVICES=${CUDA_DEVICES})"
echo "============================================"


# # ====== Step 0: Data Preparation ======
# echo ""
# echo "=== Step 0: Data Preparation ==="
# python data/prepare_helpsteer3.py
# python data/prepare_rewardbench.py
# python data/prepare_ppe_p.py


# # ====== Step 1: Generate Anchor Scores ======
# echo ""
# echo "=== Step 1: Generate Anchor Scores ==="
# for ANCHOR_MODEL_NAME in "${ANCHOR_MODEL_NAMES[@]}"; do
#     echo "--- Scoring with ${ANCHOR_MODEL_NAME} ---"
#     CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python scripts/generate_skywork_rewards.py \
#         --model_name "${ANCHOR_MODEL_NAME}" \
#         --train_data data/helpsteer3_train.json \
#         --val_data data/helpsteer3_heldout.json \
#         --test_data data/helpsteer3_heldout.json
# done


# # ====== Step 1.5: Evaluate Anchor Model as Baseline ======
# echo ""
# echo "=== Step 1.5: Evaluate Anchor Model as Baseline ==="
# for ANCHOR_MODEL_NAME in "${ANCHOR_MODEL_NAMES[@]}"; do
#     echo "--- Evaluating ${ANCHOR_MODEL_NAME} ---"
#     CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python scripts/evaluate_anchor_model.py \
#         --model_name "${ANCHOR_MODEL_NAME}" \
#         --test_data data/helpsteer3_heldout.json \
#         --rewardbench_data data/rewardbench.json \
#         --dataset_name helpsteer3 \
#         --sample_map data/helpsteer3_sample_map.json
# done


# ====== Phase 1: All Training ======
echo ""
echo "============================================"
echo "  Phase 1: All Training Runs"
echo "============================================"

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    MODEL_SHORT=$(echo "${MODEL_NAME##*/}" | tr '[:upper:]' '[:lower:]')

    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "  Model: ${MODEL_NAME}  |  Seed: ${SEED}"

        for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
            LR_KEY=$(python -c "print(float('${LEARNING_RATE}'))")
            echo "  LR: ${LEARNING_RATE} (key: ${LR_KEY})"

            # # --- BT (no anchor dependency, run once per LR/seed) ---
            # echo "=== Train: BT on HelpSteer3 ==="
            # CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} accelerate launch --config_file configs/accelerate_config.yaml \
            #     --num_processes ${NUM_GPUS} \
            #     -m src.train --config configs/bt_helpsteer3.yaml \
            #     --model_name ${MODEL_NAME} \
            #     --learning_rate ${LEARNING_RATE} \
            #     --seed ${SEED}
            # EVAL_QUEUE+=("python -m src.evaluate \
            #     --config configs/bt_helpsteer3.yaml \
            #     --checkpoint $(ckpt_dir bt_helpsteer3 ${MODEL_SHORT} ${SEED} ${LR_KEY})")

            # # --- MeanVar (no anchor dependency, run once per LR/seed) ---
            # echo "=== Train: MV on HelpSteer3 ==="
            # CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} accelerate launch --config_file configs/accelerate_config.yaml \
            #     --num_processes ${NUM_GPUS} \
            #     -m src.train --config configs/mv_helpsteer3.yaml \
            #     --model_name ${MODEL_NAME} \
            #     --learning_rate ${LEARNING_RATE} \
            #     --seed ${SEED}
            # EVAL_QUEUE+=("python -m src.evaluate \
            #     --config configs/mv_helpsteer3.yaml \
            #     --checkpoint $(ckpt_dir mv_helpsteer3 ${MODEL_SHORT} ${SEED} ${LR_KEY})")

            # --- Anchor variants: loop over anchor models × lambdas ---
            for ANCHOR_MODEL_NAME in "${ANCHOR_MODEL_NAMES[@]}"; do
                ANCHOR_MODEL_KEY=$(echo "${ANCHOR_MODEL_NAME##*/}" | tr '[:upper:]' '[:lower:]')

                for ANCHOR_LAMBDA in "${ANCHOR_LAMBDAS[@]}"; do
                    echo "............................................"
                    echo "  Anchor: ${ANCHOR_MODEL_KEY}  |  λ: ${ANCHOR_LAMBDA}"
                    echo "............................................"

                    # # MV + 1anchor
                    # echo "=== Train: MV+1anchor on HelpSteer3 ==="
                    # CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} accelerate launch --config_file configs/accelerate_config.yaml \
                    #     --num_processes ${NUM_GPUS} \
                    #     -m src.train --config configs/mv_1anchor_helpsteer3.yaml \
                    #     --model_name ${MODEL_NAME} \
                    #     --anchor_model ${ANCHOR_MODEL_KEY} \
                    #     --anchor_lambda ${ANCHOR_LAMBDA} \
                    #     --learning_rate ${LEARNING_RATE} \
                    #     --seed ${SEED}
                    # EVAL_QUEUE+=("python -m src.evaluate \
                    #     --config configs/mv_1anchor_helpsteer3.yaml \
                    #     --checkpoint $(ckpt_dir mv_1anchor_helpsteer3 ${MODEL_SHORT} ${SEED} ${LR_KEY} ${ANCHOR_MODEL_KEY} ${ANCHOR_LAMBDA})")

                    # MV + 2anchor
                    echo "=== Train: MV+2anchor on HelpSteer3 ==="
                    CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} accelerate launch --config_file configs/accelerate_config.yaml \
                        --num_processes ${NUM_GPUS} \
                        -m src.train --config configs/mv_2anchor_helpsteer3.yaml \
                        --model_name ${MODEL_NAME} \
                        --anchor_model ${ANCHOR_MODEL_KEY} \
                        --anchor_lambda ${ANCHOR_LAMBDA} \
                        --learning_rate ${LEARNING_RATE} \
                        --seed ${SEED}
                    EVAL_QUEUE+=("python -m src.evaluate \
                        --config configs/mv_2anchor_helpsteer3.yaml \
                        --checkpoint $(ckpt_dir mv_2anchor_helpsteer3 ${MODEL_SHORT} ${SEED} ${LR_KEY} ${ANCHOR_MODEL_KEY} ${ANCHOR_LAMBDA})")

                done  # ANCHOR_LAMBDA
            done  # ANCHOR_MODEL_NAME

        done  # LEARNING_RATE
    done  # SEED
done  # MODEL_NAME


# ====== Phase 2: All Evaluation (NUM_GPUS GPUs parallel) ======
echo ""
echo "============================================"
echo "  Phase 2: All Evaluation  (${#EVAL_QUEUE[@]} tasks, ${NUM_GPUS}-GPU parallel)"
echo "============================================"
flush_eval_queue


# ====== Generate Result Tables ======
echo ""
echo "=== Generating Result Tables ==="
python scripts/generate_tables.py

SCRIPT_END_TIME=$(date +%s)
TOTAL_TIME=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

echo ""
echo "============================================"
echo "  All HelpSteer3 experiments complete!"
echo "  Results saved to results/"
echo "  Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "============================================"
