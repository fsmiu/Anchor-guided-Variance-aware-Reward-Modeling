#!/bin/bash
# Best-of-N (BoN) anchor-comparison experiment driver.
#
# Phase 1 (bon_generate.py)     — incremental: re-running with a larger
#   --max_n only generates the additional samples.
# Phase 2a (bon_golden_score.py) — multi-GPU data-parallel: each GPU loads its
#   own copy of the gold RM and scores a disjoint subset of prompts. Result is
#   cached to a .npy sidecar; subsequent runs are no-ops.
# Phase 2b (bon_proxy_score.py)  — one process per GPU, each handling a
#   round-robin subset of proxy RMs. Per-RM JSONs are written independently.
# Phase 2c (bon_combine.py)      — aggregates per-RM JSONs into the combined
#   JSON consumed by plot_bon.py.
# Phase 3 (plot_bon.py)          — BT/BT-hard/Gaussian/three two-anchor q50 plot.
#
# Usage:
#   bash scripts/run_bon_anchor_compare.sh
#
# Edit the EDIT-THESE block below before running. All phases run sequentially
# but Phase 2a and Phase 2b can each be re-run independently.

set -e

START_TIME=$(date +%s)

format_duration() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))
    printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
}

# ============================================================
# EDIT THESE
# ============================================================

SETTING="ood"

# --- Step 1: Generation ---
POLICY_MODELS=(
    "unsloth/Llama-3.2-1B-Instruct"
)
POLICY_SHORTS=(
    "llama-3.2-1b-instruct"
)
EVAL_PROMPTS="data/ultrafeedback_heldout.json"   # for ood
# EVAL_PROMPTS="data/helpsteer2_heldout.json"    # for id
NUM_PROMPTS=500
MAX_N=64
GEN_TEMPERATURES=(1.0)
GEN_TOP_PS=(1.0)
MAX_NEW_TOKENS=512
GEN_BATCH_SIZE=64
GEN_MAX_BATCH=64

# --- Step 2: Scoring ---
GOLD_RM_MODELS=(
    # "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
    # "allenai/Llama-3.1-8B-Instruct-RM-RB2"
    "openbmb/UltraRM-13b"
)
GOLD_RM_SHORTS=(
    # "skywork-reward-v2-llama-3.1-8b"
    # "allenai-llama-3.1-8b-instruct-rm-rb2"
    "ultrarm-13b"
)
SCORE_BATCH_SIZE=16
NUM_SELECTION_SEEDS=4
# N_VALUES="1 2 4 8 16 32 64 128"   # uncomment for explicit list

# RM identifiers (must match checkpoint naming convention)
RM_MODEL_NAMES=(
    "unsloth/Llama-3.2-3B-Instruct"
)
ANCHOR_MODEL_NAMES=(
    "skywork-reward-v2-llama-3.2-3b"
    "deepseek-v4-pro"
    "deepseek-v4-flash"
)
RM_DATASETS=(
    # "helpsteer2"
    # "helpsteer3"
    "multipref"
    "helpsteer2"
    # "personalllm"
)
RM_SEEDS=(1)
BT_RM_LRS=("1e-05")
BT_HARD_RM_LRS=("1e-05")
MV_RM_LRS=("1e-05")
MV_1ANCHOR_RM_LRS=("5e-06")
MV_2ANCHOR_RM_LRS=("1e-05")
MV_1ANCHOR_LAMBDAS=("0.1")
MV_2ANCHOR_LAMBDAS=("0.1")

# Anchor comparison only needs the central two-anchor reward mode.
MV_REWARD_MODES="q50"

# Misc
SEED=42
CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GENERATE_WORKERS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')

if [ ${#POLICY_MODELS[@]} -ne ${#POLICY_SHORTS[@]} ]; then
    echo "ERROR: POLICY_MODELS and POLICY_SHORTS must have the same length."
    exit 1
fi

if [ ${#GOLD_RM_MODELS[@]} -ne ${#GOLD_RM_SHORTS[@]} ]; then
    echo "ERROR: GOLD_RM_MODELS and GOLD_RM_SHORTS must have the same length."
    exit 1
fi

run_proxy_score_and_combine() {
    local run_tag="$1"
    local output_dir="$2"

    mkdir -p "${output_dir}"

    # Fail fast on missing local checkpoints for enabled proxy RMs.
    # HF Hub models are skipped by this local-directory check.
    for ckpt in "${PROXY_RM_PATHS[@]}"; do
        if [[ "${ckpt}" == checkpoints* ]] && [ ! -d "${ckpt}" ]; then
            echo "ERROR: missing checkpoint directory: ${ckpt}"
            echo "       Update RM_DATASETS / RM_MODEL_NAMES / *_LRS / *_LAMBDAS / RM_SEEDS,"
            echo "       or train the missing checkpoint first."
            exit 1
        fi
    done

    # Optional explicit n schedule.
    local n_values_arg=()
    if [ -n "${N_VALUES:-}" ]; then
        n_values_arg=(--n_values ${N_VALUES})
    fi

    # ============================================================
    # PHASE 2b: PROXY SCORING (one process per GPU, RMs round-robin'd)
    # ============================================================
    echo "=== Phase 2b: Proxy scoring (${run_tag}) ==="

    local gpu_array=()
    IFS=',' read -ra gpu_array <<< "${CUDA_VISIBLE_DEVICES}"
    local num_gpus=${#gpu_array[@]}
    local num_proxies=${#PROXY_RM_PATHS[@]}
    echo "  ${num_proxies} proxy RM(s) across ${num_gpus} GPU(s): [${CUDA_VISIBLE_DEVICES}]"

    local proxy_log_dir="${output_dir}/proxy_logs"
    mkdir -p "${proxy_log_dir}"

    local proxy_pids=()
    local g proxy_i
    for ((g=0; g<num_gpus; g++)); do
        local this_paths=()
        local this_types=()
        local this_modes=()
        local this_labels=()
        for ((proxy_i=g; proxy_i<num_proxies; proxy_i+=num_gpus)); do
            this_paths+=("${PROXY_RM_PATHS[$proxy_i]}")
            this_types+=("${PROXY_RM_TYPES[$proxy_i]}")
            this_modes+=("${PROXY_RM_MODES[$proxy_i]}")
            this_labels+=("${PROXY_RM_LABELS[$proxy_i]}")
        done
        if [ ${#this_paths[@]} -eq 0 ]; then
            continue
        fi
        local log_file="${proxy_log_dir}/gpu${gpu_array[$g]}_${run_tag}.log"
        echo "  GPU ${gpu_array[$g]} -> ${this_labels[*]}  (log: ${log_file})"
        CUDA_VISIBLE_DEVICES=${gpu_array[$g]} python scripts/bon_proxy_score.py \
            --responses_cache       "${RESPONSES_CACHE}" \
            --gold_rm_model         "${GOLD_RM_MODEL}" \
            --gold_scores_path      "${GOLD_SCORES_PATH}" \
            --rm_max_length         2048 \
            --score_batch_size      ${SCORE_BATCH_SIZE} \
            --proxy_rm_path         "${this_paths[@]}" \
            --proxy_rm_type         "${this_types[@]}" \
            --reward_mode           "${this_modes[@]}" \
            --proxy_rm_label        "${this_labels[@]}" \
            --num_selection_seeds   ${NUM_SELECTION_SEEDS} \
            --output_dir            "${output_dir}" \
            --run_tag               "${run_tag}" \
            --seed                  ${SEED} \
            --device                cuda:0 \
            "${n_values_arg[@]}" \
            > "${log_file}" 2>&1 &
        proxy_pids+=($!)
    done

    local proxy_failed=0
    local pid
    for pid in "${proxy_pids[@]}"; do
        if ! wait $pid; then
            echo "ERROR: bon_proxy_score.py worker (pid=${pid}) failed."
            proxy_failed=1
        fi
    done
    if [ ${proxy_failed} -ne 0 ]; then
        echo "One or more proxy workers failed. See ${proxy_log_dir}/gpu*_${run_tag}.log"
        exit 1
    fi

    # ============================================================
    # PHASE 2c: COMBINE per-RM JSONs into bon_combined_*.json
    # ============================================================
    echo "=== Phase 2c: Combine per-RM JSONs (${run_tag}) ==="
    python scripts/bon_combine.py \
        --output_dir       "${output_dir}" \
        --run_tag          "${run_tag}" \
        --proxy_rm_label   "${PROXY_RM_LABELS[@]}"
}

# ============================================================
# AUTO-ASSEMBLED (do not edit)
# ============================================================

for policy_idx in "${!POLICY_MODELS[@]}"; do
    POLICY_MODEL="${POLICY_MODELS[$policy_idx]}"
    POLICY_SHORT="${POLICY_SHORTS[$policy_idx]}"

    for GEN_TEMPERATURE in "${GEN_TEMPERATURES[@]}"; do
        for GEN_TOP_P in "${GEN_TOP_PS[@]}"; do
            GEN_RUN_TAG="${POLICY_SHORT}_${SETTING}_t${GEN_TEMPERATURE}_tp${GEN_TOP_P}_n${NUM_PROMPTS}_maxn${MAX_N}_seed${SEED}"
            GEN_OUTPUT_DIR="results_bon/generation/${GEN_RUN_TAG}"
            RESPONSES_CACHE="${GEN_OUTPUT_DIR}/responses_cache.json"

            mkdir -p "${GEN_OUTPUT_DIR}"

# ============================================================
# PHASE 1: GENERATION (incremental — safe to re-run)
# Runs once per policy + prompt set + temperature/top_p + max_n + seed.
# ============================================================
echo "=== Phase 1: Generation (${GEN_RUN_TAG}) ==="
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python scripts/bon_generate.py \
    --setting           "${SETTING}" \
    --eval_prompts      "${EVAL_PROMPTS}" \
    --num_prompts       ${NUM_PROMPTS} \
    --policy_model      "${POLICY_MODEL}" \
    --max_n             ${MAX_N} \
    --max_prompt_length 1536 \
    --max_new_tokens    ${MAX_NEW_TOKENS} \
    --gen_temperature   ${GEN_TEMPERATURE} \
    --gen_top_p         ${GEN_TOP_P} \
    --gen_batch_size    ${GEN_BATCH_SIZE} \
    --gen_max_batch     ${GEN_MAX_BATCH} \
    --num_workers       ${NUM_GENERATE_WORKERS} \
    --worker_devices    "${CUDA_VISIBLE_DEVICES}" \
    --output            "${RESPONSES_CACHE}" \
    --seed              ${SEED}

            for gold_idx in "${!GOLD_RM_MODELS[@]}"; do
                GOLD_RM_MODEL="${GOLD_RM_MODELS[$gold_idx]}"
                GOLD_RM_SHORT="${GOLD_RM_SHORTS[$gold_idx]}"
                GOLD_RUN_TAG="${GEN_RUN_TAG}_gold-${GOLD_RM_SHORT}"
                GOLD_OUTPUT_DIR="results_bon/gold/${GOLD_RUN_TAG}"
                GOLD_TAG=$(echo "${GOLD_RM_MODEL}" | tr '/ ' '__')
                GOLD_SCORES_PATH="${GOLD_OUTPUT_DIR}/gold_scores_${GOLD_TAG}_${GOLD_RUN_TAG}.npy"

                mkdir -p "${GOLD_OUTPUT_DIR}"

# ============================================================
# PHASE 2a: GOLD SCORING (multi-GPU data-parallel)
# Runs once per responses_cache + gold RM.
# ============================================================
echo "=== Phase 2a: Gold scoring (${GOLD_RUN_TAG}) ==="
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python scripts/bon_golden_score.py \
    --responses_cache   "${RESPONSES_CACHE}" \
    --gold_rm_model     "${GOLD_RM_MODEL}" \
    --rm_max_length     2048 \
    --score_batch_size  ${SCORE_BATCH_SIZE} \
    --num_workers       ${NUM_GENERATE_WORKERS} \
    --worker_devices    "${CUDA_VISIBLE_DEVICES}" \
    --output_dir        "${GOLD_OUTPUT_DIR}" \
    --run_tag           "${GOLD_RUN_TAG}"

                for RM_DATASET in "${RM_DATASETS[@]}"; do
                    for RM_MODEL_NAME in "${RM_MODEL_NAMES[@]}"; do
                        RM_SHORT=$(echo "${RM_MODEL_NAME##*/}" | tr '[:upper:]' '[:lower:]')

                        for RM_SEED in "${RM_SEEDS[@]}"; do
                            BT_RM_LR="${BT_RM_LRS[0]}"
                            BT_HARD_RM_LR="${BT_HARD_RM_LRS[0]}"
                            MV_RM_LR="${MV_RM_LRS[0]}"
                            MV_1ANCHOR_RM_LR="${MV_1ANCHOR_RM_LRS[0]}"
                            MV_1ANCHOR_LAMBDA="${MV_1ANCHOR_LAMBDAS[0]}"

BT_RM_PATH="checkpoints_V1_5.3/bt_${RM_DATASET}_${RM_SHORT}_lr${BT_RM_LR}_seed${RM_SEED}"
BT_HARD_RM_PATH="checkpoints/bt_${RM_DATASET}_hard_${RM_SHORT}_lr${BT_HARD_RM_LR}_seed${RM_SEED}"
MV_RM_PATH="checkpoints_V1_5.3/mv_${RM_DATASET}_${RM_SHORT}_lr${MV_RM_LR}_seed${RM_SEED}"

BASELINE_RUN_TAG="${RM_DATASET}_${RM_SHORT}_seed${RM_SEED}_btlr${BT_RM_LR}_bthardlr${BT_HARD_RM_LR}_mvlr${MV_RM_LR}_${GEN_RUN_TAG}_baseline_gold-${GOLD_RM_SHORT}"
BASELINE_OUTPUT_DIR="results_bon/proxy/${BASELINE_RUN_TAG}"
BASELINE_COMBINED_INPUT="${BASELINE_OUTPUT_DIR}/bon_combined_${BASELINE_RUN_TAG}.json"

echo "=== BoN shared baseline run: ${BASELINE_RUN_TAG} ==="
PROXY_RM_PATHS=("${BT_RM_PATH}" "${BT_HARD_RM_PATH}" "${MV_RM_PATH}")
PROXY_RM_TYPES=("bt" "bt" "meanvar")
PROXY_RM_MODES=("q50" "q50" "q50")
PROXY_RM_LABELS=("bt" "bt_hard" "mv_q50")
run_proxy_score_and_combine "${BASELINE_RUN_TAG}" "${BASELINE_OUTPUT_DIR}"

                            for MV_2ANCHOR_RM_LR in "${MV_2ANCHOR_RM_LRS[@]}"; do
                                for MV_2ANCHOR_LAMBDA in "${MV_2ANCHOR_LAMBDAS[@]}"; do
                                    ANCHOR_COMBINED_INPUTS=()

                                    for ANCHOR_MODEL_NAME in "${ANCHOR_MODEL_NAMES[@]}"; do
                                        ANCHOR_SHORT=$(echo "${ANCHOR_MODEL_NAME##*/}" | tr '[:upper:]' '[:lower:]')
                                        RUN_TAG="${RM_DATASET}_${RM_SHORT}_seed${RM_SEED}_mv2lr${MV_2ANCHOR_RM_LR}_${GEN_RUN_TAG}_${ANCHOR_SHORT}_mv2lam${MV_2ANCHOR_LAMBDA}_gold-${GOLD_RM_SHORT}"
                                        OUTPUT_DIR="results_bon/proxy/${RUN_TAG}"
                                        COMBINED_INPUT="${OUTPUT_DIR}/bon_combined_${RUN_TAG}.json"

                                        echo "=== BoN anchor proxy run: ${RUN_TAG} ==="

MV_1ANCHOR_RM_PATH="checkpoints/mv_1anchor_${RM_DATASET}_${RM_SHORT}_anchor-${ANCHOR_SHORT}_lam${MV_1ANCHOR_LAMBDA}_lr${MV_1ANCHOR_RM_LR}_seed${RM_SEED}"
MV_2ANCHOR_RM_PATH="checkpoints/mv_2anchor_${RM_DATASET}_${RM_SHORT}_anchor-${ANCHOR_SHORT}_lam${MV_2ANCHOR_LAMBDA}_lr${MV_2ANCHOR_RM_LR}_seed${RM_SEED}"
ANCHOR_AS_PROXY_PATH="${ANCHOR_MODEL_NAME}"

# # Anchor as proxy
# PROXY_RM_PATHS=("${ANCHOR_AS_PROXY_PATH}")
# PROXY_RM_TYPES=("anchor")
# PROXY_RM_MODES=("q50")
# PROXY_RM_LABELS=("anchor_proxy")

# # MV+1anchor variants
# PROXY_RM_PATHS=("${MV_1ANCHOR_RM_PATH}")
# PROXY_RM_TYPES=("meanvar")
# PROXY_RM_MODES=("q50")
# PROXY_RM_LABELS=("mv_1anchor_q50")

# MV+2anchor variants
PROXY_RM_PATHS=("${MV_2ANCHOR_RM_PATH}")
PROXY_RM_TYPES=("meanvar")
PROXY_RM_MODES=("q50")
PROXY_RM_LABELS=("mv_2anchor_q50")
run_proxy_score_and_combine "${RUN_TAG}" "${OUTPUT_DIR}"

                                        if [ -f "${COMBINED_INPUT}" ]; then
                                            ANCHOR_COMBINED_INPUTS+=("${COMBINED_INPUT}")
                                        fi
                                    done

                                    # ============================================================
                                    # STEP 3: PLOT
                                    # ============================================================
                                    echo "=== Step 3: Plot all-anchor comparison ==="
                                    if [ -f "${BASELINE_COMBINED_INPUT}" ] && [ ${#ANCHOR_COMBINED_INPUTS[@]} -eq ${#ANCHOR_MODEL_NAMES[@]} ]; then
                                        ANCHOR_COMPARE_RUN_TAG="${RM_DATASET}_${RM_SHORT}_seed${RM_SEED}_mv2lr${MV_2ANCHOR_RM_LR}_${GEN_RUN_TAG}_allanchors_mv2lam${MV_2ANCHOR_LAMBDA}_gold-${GOLD_RM_SHORT}"
                                        ANCHOR_COMPARE_OUTPUT_DIR="results_bon/proxy/${ANCHOR_COMPARE_RUN_TAG}"
                                        mkdir -p "${ANCHOR_COMPARE_OUTPUT_DIR}"

                                        PLOT_INPUTS=("${BASELINE_COMBINED_INPUT}" "${ANCHOR_COMBINED_INPUTS[@]}")
                                        python scripts/plot_bon.py \
                                            --input   "${PLOT_INPUTS[@]}" \
                                            --output  "${ANCHOR_COMPARE_OUTPUT_DIR}/bon_curve_bt_bt-hard_gaussian_two_anchor_q50_${ANCHOR_COMPARE_RUN_TAG}.pdf" \
                                            --include bt bt_hard mv_q50 mv_2anchor_q50 \
                                            --paper_style \
                                            --hide_baseline
                                        echo "Done. Results in ${ANCHOR_COMPARE_OUTPUT_DIR}"
                                    else
                                        echo "Skip all-anchor q50 plot: baseline found=$([ -f "${BASELINE_COMBINED_INPUT}" ] && echo 1 || echo 0), anchors found ${#ANCHOR_COMBINED_INPUTS[@]} / ${#ANCHOR_MODEL_NAMES[@]}."
                                    fi
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))

echo "Total runtime: $(format_duration ${TOTAL_SECONDS}) (${TOTAL_SECONDS}s)"
