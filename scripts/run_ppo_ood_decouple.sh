#!/bin/bash
# Decoupled PPO Experiment Suite — OOD setting.
# Trains with proxy rewards only, saves periodic policy checkpoints, then runs
# offline gold-RM eval over those checkpoints and optionally deletes periodic
# checkpoints after successful eval.
#
# Convention for RM checkpoint paths follows scripts/run_anchor_multipref_split.sh
# (its ckpt_dir helper):
#   checkpoints/{run_name}_{model_short}_[anchor-{anchor_short}_lam{lam}_]lr{lr}_seed{seed}
#
# IMPORTANT: the ${lr} segment uses Python's float repr (e.g. 5e-6 → "5e-06").
# To derive the correct string from your CLI lr, run:
#   python -c "print(float('5e-6'))"   # → 5e-06

set -e

# echo "Sleeping for 2 hours before starting..."
# sleep 6000

SCRIPT_START_TIME=$(date +%s)

# ==========================================================================
# ===================== EDIT BEFORE RUNNING ================================
# ==========================================================================

# --- Policy model short name (only affects display/logs) ---
POLICY_SHORT="llama-3.2-1b-instruct"

# --- RM base model short name (MUST match the base used for RM training) ---
# Set equal to POLICY_SHORT for the common case (RM and policy share a base).
# Override for cross-size supervision, e.g. 3B RM → 1B policy:
RM_SHORT="llama-3.2-3b-instruct"
RM_SHORT="${RM_SHORT:-${POLICY_SHORT}}"

# --- Optional explicit RM tokenizer source ---
# Usually leave empty: src.ppo_train will prefer the RM checkpoint's recorded
# base model name, and only fall back to the checkpoint directory if needed.
# Set this only if you need to override tokenizer loading manually.
RM_TOKENIZER="${RM_TOKENIZER:-}"

# --- Anchor models/signals used during anchor RM training ---
# Full HF Hub names or signal keys. ANCHOR_SHORT is derived from each entry
# and used in checkpoint names.
ANCHOR_MODEL_NAMES=(
    # "Skywork/Skywork-Reward-V2-Llama-3.2-3B"
    "deepseek-v4-flash"
    # "deepseek-v4-pro"
)
RM_SEED=1

# --- Gold RM used for evaluation/scoring ---
# Overrides configs/ppo.yaml::gold_rm_model for offline eval. You can also run:
#   GOLD_RM_MODEL="your/model-or-local-path" bash scripts/run_ppo_ood_decouple.sh
# GOLD_RM_MODEL="${GOLD_RM_MODEL:-Skywork/Skywork-Reward-V2-Llama-3.1-8B}"
GOLD_RM_MODEL="${GOLD_RM_MODEL:-openbmb/UltraRM-13b}"


BT_RM_LR="1e-05"
BT_HARD_RM_LR="1e-05"
MV_RM_LR="1e-05"
MV_1ANCHOR_RM_LR="5e-06"
MV_2ANCHOR_RM_LR="1e-05"

MV_1ANCHOR_LAMBDA="0.1"
MV_2ANCHOR_LAMBDA="0.1"

SEED=1
# REWARD_MODES=(q50 random)
REWARD_MODES=(q10 q30 q70)
PPO_EPOCHS=(1)
NUM_EPOCHS=(1)
KL_COEFS=(0.01)
TRAIN_TEMPERATURES=(0.7)
TRAIN_TOP_PS=(0.9)
ROLLOUT_BATCH_SIZES=(64)
LEARNING_RATES=(1e-6)
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-50}"
CLEANUP_EVAL_CHECKPOINTS="${CLEANUP_EVAL_CHECKPOINTS:-1}"
OFFLINE_EVAL_DEVICE="${OFFLINE_EVAL_DEVICE:-cuda}"
OFFLINE_EVAL_BATCH_SIZE="${OFFLINE_EVAL_BATCH_SIZE:-16}"
OFFLINE_EVAL_WORKERS="${OFFLINE_EVAL_WORKERS:-4}"
OFFLINE_EVAL_WORKER_DEVICES="${OFFLINE_EVAL_WORKER_DEVICES:-0,1,2,3}"
OFFLINE_EVAL_NUM_PROMPTS="${OFFLINE_EVAL_NUM_PROMPTS:-500}"

# --- RM training datasets (selects which RM checkpoints to load) ---
# Each value must match the dataset used to train the RM checkpoints. The
# RM checkpoint directory names are of the form
#   {bt,mv,mv_1anchor,mv_2anchor}_${rm_dataset}_...
# (see scripts/run_anchor_{multipref,personalllm,helpsteer2,helpsteer3}_split.sh).
# Allowed values: multipref | personalllm | helpsteer2 | helpsteer3
RM_DATASETS=(multipref)

for _rm_dataset in "${RM_DATASETS[@]}"; do
    case "${_rm_dataset}" in
        multipref|personalllm|helpsteer2|helpsteer3) ;;
        *)
            echo "ERROR: RM_DATASETS entry must be one of: multipref, personalllm, helpsteer2, helpsteer3 (got '${_rm_dataset}')"
            exit 1
            ;;
    esac
done
unset _rm_dataset

# ==========================================================================
# ===================== AUTO-ASSEMBLED (do not edit) =======================
# ==========================================================================

SETTING="ood"
PPO_PROMPTS="data/ultrafeedback_train.json"
EVAL_PROMPTS="data/ultrafeedback_heldout.json"

echo "============================================"
echo "  PPO Experiment Suite — OOD (UltraFeedback)"
echo "  Policy base:        ${POLICY_SHORT}"
echo "  RM base:            ${RM_SHORT}"
if [ "${POLICY_SHORT}" != "${RM_SHORT}" ]; then
    echo "  (cross-size: RM base differs from policy base)"
fi
echo "  RM datasets:        ${RM_DATASETS[*]}"
echo "  Gold RM:            ${GOLD_RM_MODEL}"
echo "  Anchor models:      ${ANCHOR_MODEL_NAMES[*]}"
echo "  Reward modes:       ${REWARD_MODES[*]}"
echo "  PPO epochs:         ${PPO_EPOCHS[*]}"
echo "  Num epochs:         ${NUM_EPOCHS[*]}"
echo "  KL coefs:           ${KL_COEFS[*]}"
echo "  Train temperatures: ${TRAIN_TEMPERATURES[*]}"
echo "  Train top_p:        ${TRAIN_TOP_PS[*]}"
echo "  Rollout batch size: ${ROLLOUT_BATCH_SIZES[*]}"
echo "  Learning rates:     ${LEARNING_RATES[*]}"
echo "  Save every steps:   ${SAVE_EVERY_STEPS}"
echo "  Cleanup checkpoints:${CLEANUP_EVAL_CHECKPOINTS}"
echo "  Offline eval batch: ${OFFLINE_EVAL_BATCH_SIZE}"
echo "  Offline eval workers:${OFFLINE_EVAL_WORKERS}"
echo "  Offline eval devices:${OFFLINE_EVAL_WORKER_DEVICES}"
echo "  Seed:               ${SEED}"
echo "============================================"

# if [ ! -f "${PPO_PROMPTS}" ] || [ ! -f "${EVAL_PROMPTS}" ]; then
#     echo ""
#     echo "ERROR: UltraFeedback JSONs not found. Run first:"
#     echo "  python data/prepare_ultrafeedback.py"
#     echo "expected at:"
#     echo "  ${PPO_PROMPTS}"
#     echo "  ${EVAL_PROMPTS}"
#     exit 1
# fi

sanitize_tag() {
    local raw="$1"
    raw="${raw//\//_}"
    raw="${raw// /_}"
    echo "${raw}"
}

launch_ppo_decoupled() {
    local proxy_path="$1"
    local proxy_type="$2"
    local reward_mode="$3"
    shift 3

    local rm_dirname
    rm_dirname=$(basename "${proxy_path}")
    local output_root="checkpoints/ppo_decoupled/policy-$(sanitize_tag "${POLICY_SHORT}")/rm-$(sanitize_tag "${rm_dirname}")/${SETTING}_${reward_mode}_lr${lr}_kl${kl_coef}_ep${ppo_epoch}_ne${num_epoch}_t${train_temp}_tp${train_top_p}_rbs${rollout_bs}_seed${SEED}"
    local manifest="${output_root}/manifest.json"

    local extra_args=(
        --policy_short "${POLICY_SHORT}"
        --rm_short "${RM_SHORT}"
    )

    if [ -n "${RM_TOKENIZER}" ]; then
        extra_args+=(--rm_tokenizer "${RM_TOKENIZER}")
    fi

    accelerate launch --config_file configs/accelerate_config.yaml \
        -m src.ppo_train_decoupled \
        --config configs/ppo.yaml \
        --proxy_rm_path "${proxy_path}" \
        --proxy_rm_type "${proxy_type}" \
        --reward_mode "${reward_mode}" \
        --setting "${SETTING}" \
        --ppo_prompts "${PPO_PROMPTS}" \
        --eval_prompts "${EVAL_PROMPTS}" \
        --gold_rm_model "${GOLD_RM_MODEL}" \
        --output_root "${output_root}" \
        --save_every_steps "${SAVE_EVERY_STEPS}" \
        --seed "${SEED}" \
        "${extra_args[@]}" \
        "$@"

    local cleanup_args=()
    if [ "${CLEANUP_EVAL_CHECKPOINTS}" = "1" ]; then
        cleanup_args+=(--cleanup_periodic)
    fi

    local prompt_args=()
    if [ -n "${OFFLINE_EVAL_NUM_PROMPTS}" ]; then
        prompt_args+=(--num_eval_prompts "${OFFLINE_EVAL_NUM_PROMPTS}")
    fi
    if [ -n "${OFFLINE_EVAL_BATCH_SIZE}" ]; then
        prompt_args+=(--eval_batch_size "${OFFLINE_EVAL_BATCH_SIZE}")
    fi

    python scripts/ppo_eval_checkpoints.py \
        --manifest "${manifest}" \
        --gold_rm_model "${GOLD_RM_MODEL}" \
        --device "${OFFLINE_EVAL_DEVICE}" \
        --num_workers "${OFFLINE_EVAL_WORKERS}" \
        --worker_devices "${OFFLINE_EVAL_WORKER_DEVICES}" \
        "${prompt_args[@]}" \
        "${cleanup_args[@]}"
}

# --- Pre-training MV RM variance inspection ---
# Single-GPU diagnostic. Verifies that each MV RM's sigma has meaningful
# spread (and that mu-diff is reasonable) before burning GPU hours on PPO.
# Skips automatically if an existing summary.json is already on disk for
# that checkpoint — if you already ran run_ppo_id.sh it won't repeat.
# Override with RUN_INSPECTION=0 to skip entirely.
# RUN_INSPECTION=${RUN_INSPECTION:-1}
# if [ "${RUN_INSPECTION}" = "1" ]; then
#     INSPECT_OUT_DIR="results/rm_variance"
#     for ckpt in "${MV_RM_PATH}" "${MV_1ANCHOR_RM_PATH}" "${MV_2ANCHOR_RM_PATH}"; do
#     for ckpt in "${MV_RM_PATH}" "${MV_1ANCHOR_RM_PATH}" "${MV_2ANCHOR_RM_PATH}"; do

#         rm_name=$(basename "${ckpt}")
#         out_dir="${INSPECT_OUT_DIR}/${rm_name}"
#         if [ -f "${out_dir}/rm_variance_summary.json" ]; then
#             echo "[skip] ${rm_name}: already inspected → ${out_dir}"
#             continue
#         fi
#         echo ""
#         echo "=== Inspecting RM: ${rm_name} ==="
#         python scripts/inspect_rm_variance.py \
#             --checkpoint "${ckpt}" \
#             --model_type meanvar \
#             --heldout_data "data/multipref_heldout.json" \
#             --ultrafeedback_prompts "data/ultrafeedback_heldout.json" \
#             --output_dir "${out_dir}"
#     done
# fi

i=1
for rm_dataset in "${RM_DATASETS[@]}"; do

BT_RM_PATH="checkpoints_V1_5.3/bt_${rm_dataset}_${RM_SHORT}_lr${BT_RM_LR}_seed${RM_SEED}"
BT_HARD_RM_PATH="checkpoints/bt_${rm_dataset}_hard_${RM_SHORT}_lr${BT_HARD_RM_LR}_seed${RM_SEED}"
MV_RM_PATH="checkpoints_V1_5.3/mv_${rm_dataset}_${RM_SHORT}_lr${MV_RM_LR}_seed${RM_SEED}"

echo ""
echo "############################################"
echo "  RM dataset: ${rm_dataset}"
echo "    BT RM: ${BT_RM_PATH}"
echo "    BT hard RM: ${BT_HARD_RM_PATH}"
echo "    MV RM: ${MV_RM_PATH}"
echo "############################################"

for ppo_epoch in "${PPO_EPOCHS[@]}"; do
for num_epoch in "${NUM_EPOCHS[@]}"; do
for kl_coef in "${KL_COEFS[@]}"; do
for train_temp in "${TRAIN_TEMPERATURES[@]}"; do
for train_top_p in "${TRAIN_TOP_PS[@]}"; do
for rollout_bs in "${ROLLOUT_BATCH_SIZES[@]}"; do
for lr in "${LEARNING_RATES[@]}"; do

COMMON_HP_ARGS=(
    --ppo_epochs "${ppo_epoch}"
    --num_epochs "${num_epoch}"
    --kl_coef "${kl_coef}"
    --train_temperature "${train_temp}"
    --train_top_p "${train_top_p}"
    --rollout_batch_size "${rollout_bs}"
    --learning_rate "${lr}"
)

HP_DESC_BASE="rm_dataset=${rm_dataset}  ppo_epochs=${ppo_epoch}  num_epochs=${num_epoch}  kl_coef=${kl_coef}  temp=${train_temp}  top_p=${train_top_p}  rbs=${rollout_bs}  lr=${lr}"

# # # --- 1× BT (q50 only) ---
# echo ""
# echo "=== [${i}] BT  mode=q50  ${HP_DESC_BASE} ==="
# launch_ppo_decoupled "${BT_RM_PATH}" bt q50 "${COMMON_HP_ARGS[@]}"
# i=$((i + 1))

# # --- 1× BT hard (q50 only) ---
echo ""
echo "=== [${i}] BT hard  mode=q50  ${HP_DESC_BASE} ==="
launch_ppo_decoupled "${BT_HARD_RM_PATH}" bt q50 "${COMMON_HP_ARGS[@]}"
i=$((i + 1))

# # # --- MV (no anchor) ---
# echo ""
# echo "=== [${i}] MV  mode=q50  ${HP_DESC_BASE} ==="
# launch_ppo_decoupled "${MV_RM_PATH}" meanvar q50  "${COMMON_HP_ARGS[@]}"
# i=$((i + 1))

# # --- MV (no anchor) ---
# for mode in "${REWARD_MODES[@]}"; do
#     echo ""
#     echo "=== [${i}] MV  mode=${mode}  ${HP_DESC_BASE} ==="
#     launch_ppo_decoupled "${MV_RM_PATH}" meanvar "${mode}" "${COMMON_HP_ARGS[@]}"
#     i=$((i + 1))
# done

for ANCHOR_MODEL_NAME in "${ANCHOR_MODEL_NAMES[@]}"; do
    ANCHOR_SHORT=$(echo "${ANCHOR_MODEL_NAME##*/}" | tr '[:upper:]' '[:lower:]')

MV_1ANCHOR_RM_PATH="checkpoints/mv_1anchor_${rm_dataset}_${RM_SHORT}_anchor-${ANCHOR_SHORT}_lam${MV_1ANCHOR_LAMBDA}_lr${MV_1ANCHOR_RM_LR}_seed${RM_SEED}"
MV_2ANCHOR_RM_PATH="checkpoints/mv_2anchor_${rm_dataset}_${RM_SHORT}_anchor-${ANCHOR_SHORT}_lam${MV_2ANCHOR_LAMBDA}_lr${MV_2ANCHOR_RM_LR}_seed${RM_SEED}"
HP_DESC="${HP_DESC_BASE}  anchor=${ANCHOR_SHORT}"

echo ""
echo "############################################"
echo "  RM dataset: ${rm_dataset}"
echo "  Anchor model/signal: ${ANCHOR_MODEL_NAME}"
echo "    MV+1anchor RM: ${MV_1ANCHOR_RM_PATH}"
echo "    MV+2anchor RM: ${MV_2ANCHOR_RM_PATH}"
echo "############################################"

# # --- 1× Anchor-as-proxy (q50 only; external Skywork RM guides PPO) ---
# echo ""
# echo "=== [${i}] AnchorProxy  mode=q50  ${HP_DESC} ==="
# launch_ppo_decoupled "${ANCHOR_MODEL_NAME}" anchor q50 "${COMMON_HP_ARGS[@]}"
# i=$((i + 1))

# # --- MV + 1anchor ---
# for mode in "${REWARD_MODES[@]}"; do
#     echo ""
#     echo "=== [${i}] MV+1anchor (lam=${MV_1ANCHOR_LAMBDA})  mode=${mode}  ${HP_DESC} ==="
#     launch_ppo_decoupled "${MV_1ANCHOR_RM_PATH}" meanvar "${mode}" \
#         --anchor_method 1anchor --anchor_lambda "${MV_1ANCHOR_LAMBDA}" \
#         "${COMMON_HP_ARGS[@]}"
#     i=$((i + 1))
# done

# --- MV + 2anchor ---
for mode in "${REWARD_MODES[@]}"; do
    echo ""
    echo "=== [${i}] MV+2anchor (lam=${MV_2ANCHOR_LAMBDA})  mode=${mode}  ${HP_DESC} ==="
    launch_ppo_decoupled "${MV_2ANCHOR_RM_PATH}" meanvar "${mode}" \
        --anchor_method 2anchor --anchor_lambda "${MV_2ANCHOR_LAMBDA}" \
        "${COMMON_HP_ARGS[@]}"
    i=$((i + 1))
done

done  # anchor_model
done  # lr
done  # rollout_bs
done  # train_top_p
done  # train_temp
done  # kl_coef
done  # num_epoch
done  # ppo_epoch
done  # rm_dataset

SCRIPT_END_TIME=$(date +%s)
TOTAL_TIME=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECS=$((TOTAL_TIME % 60))

echo ""
echo "============================================"
echo "  All decoupled OOD PPO runs complete!"
echo "  Checkpoints: checkpoints/ppo_decoupled/policy-<policy-short>/rm-<rm-dirname>/${SETTING}_*"
echo "  Total runtime: ${HOURS}h ${MINUTES}m ${SECS}s"
echo "============================================"
