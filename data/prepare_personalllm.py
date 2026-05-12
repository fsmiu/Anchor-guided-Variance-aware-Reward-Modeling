"""Prepare PersonalLLM dataset (namkoong-lab/PersonalLLM) for reward model training."""

import json
import os
import random
from collections import defaultdict
from itertools import combinations

from datasets import load_dataset


# The 10 reward models used as "annotators"
REWARD_MODELS = [
    "gemma_2b",
    "gemma_7b",
    "mistral_raft",
    "mistral_ray",
    "mistral_weqweasdas",
    "llama3_sfairx",
    "oasst_deberta_v3",
    "beaver_7b",
    "oasst_pythia_7b",
    "oasst_pythia_1b",
]


def extract_soft_label(score_a, score_b):
    """Extract soft label from 10 reward model scores for a pair of responses.

    For each RM:
      score_a > score_b  -> A wins
      score_a < score_b  -> B wins
      score_a == score_b -> Tie (counts as 0.5 vote for each)

    Returns (chosen, rejected, soft_label, chosen_idx, rejected_idx) or None if soft tie.
    chosen/rejected are response indices (1-based).
    """
    votes_a = 0
    votes_b = 0
    votes_tie = 0

    for rm in REWARD_MODELS:
        sa = score_a[rm]
        sb = score_b[rm]
        if sa > sb:
            votes_a += 1
        elif sa < sb:
            votes_b += 1
        else:
            votes_tie += 1

    votes_a_eff = votes_a + 0.5 * votes_tie
    votes_b_eff = votes_b + 0.5 * votes_tie
    total_eff = votes_a_eff + votes_b_eff

    if total_eff == 0:
        return None

    soft_a = votes_a_eff / total_eff

    if soft_a > 0.5:
        return soft_a, "a"
    elif soft_a < 0.5:
        return 1.0 - soft_a, "b"
    else:
        return None


def process_split(ds, split_name):
    """Process a PersonalLLM split and return (processed, original_sample_map)."""
    processed = []
    original_sample_map = defaultdict(list)
    skipped_no_label = 0
    skipped_soft_tie = 0
    total_pairs = 0

    response_indices = list(range(1, 9))  # 1..8

    for idx, sample in enumerate(ds):
        prompt = sample.get("prompt", "")
        if not prompt:
            continue

        # Collect responses and their RM scores
        responses = {}
        scores = {}
        for ri in response_indices:
            resp = sample.get(f"response_{ri}", "")
            if not resp:
                continue
            rm_scores = {}
            valid = True
            for rm in REWARD_MODELS:
                s = sample.get(f"response_{ri}_{rm}")
                if s is None:
                    valid = False
                    break
                rm_scores[rm] = s
            if valid:
                responses[ri] = resp
                scores[ri] = rm_scores

        # Generate all pairs from available responses
        available = sorted(responses.keys())
        if len(available) < 2:
            continue

        for ri, rj in combinations(available, 2):
            total_pairs += 1
            result = extract_soft_label(scores[ri], scores[rj])
            if result is None:
                skipped_soft_tie += 1
                continue

            soft_label, winner = result

            if winner == "a":
                chosen = responses[ri]
                rejected = responses[rj]
            else:
                chosen = responses[rj]
                rejected = responses[ri]

            instance = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "original_sample_id": idx,
                "soft_label": soft_label,
            }
            processed.append(instance)
            original_sample_map[idx].append(len(processed) - 1)

    print(f"\nProcessing summary [{split_name}]:")
    print(f"  Total raw samples: {len(ds)}")
    print(f"  Total pairs evaluated: {total_pairs}")
    print(f"  Skipped (soft label tie): {skipped_soft_tie}")
    print(f"  Final instances: {len(processed)}")
    print(f"  Unique original samples kept: {len(original_sample_map)}")

    return processed, original_sample_map


def main():
    print("Loading PersonalLLM dataset (namkoong-lab/PersonalLLM)...")
    ds = load_dataset("namkoong-lab/PersonalLLM")
    ds_train = ds["train"]
    ds_test = ds["test"]
    print(f"Raw train size: {len(ds_train)}, Raw test size: {len(ds_test)}")
    print(f"Columns: {ds_train.column_names}")

    # Print a raw sample to understand structure
    print("\n--- Raw sample (keys & truncated values) ---")
    sample = ds_train[0]
    for key in sample:
        val = sample[key]
        val_str = str(val)
        if len(val_str) > 200:
            print(f"  {key}: {val_str[:200]}...")
        else:
            print(f"  {key}: {val_str}")

    # Process both splits independently
    train_data, train_sample_map = process_split(ds_train, "train")
    heldout_data, heldout_sample_map = process_split(ds_test, "heldout")

    print(f"\n  -> {len(train_data)} train / {len(heldout_data)} heldout instances")

    # Subsample: 40k train, 2k heldout
    random.seed(42)
    if len(train_data) > 40000:
        train_data = random.sample(train_data, 40000)
        print(f"  Subsampled train to {len(train_data)}")
    if len(heldout_data) > 2000:
        heldout_data = random.sample(heldout_data, 2000)
        print(f"  Subsampled heldout to {len(heldout_data)}")

    # Save
    os.makedirs("data", exist_ok=True)
    with open("data/personalllm_train.json", "w") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open("data/personalllm_heldout.json", "w") as f:
        json.dump(heldout_data, f, indent=2, ensure_ascii=False)

    # Save total count (train only, excluding heldout) for size matching
    with open("data/personalllm_count.txt", "w") as f:
        f.write(str(len(train_data)))

    # Print 3 random samples
    print("\n--- 3 random training samples ---")
    for s in random.sample(train_data, min(3, len(train_data))):
        print(f"  Prompt:   {s['prompt'][:120]}...")
        print(f"  Chosen:   {s['chosen'][:120]}...")
        print(f"  Rejected: {s['rejected'][:120]}...")
        print(f"  Soft label: {s['soft_label']:.4f}")
        print()

    # Dataset stats
    print(f"Dataset statistics:")
    print(f"  Train:   {len(train_data)}")
    print(f"  Heldout: {len(heldout_data)} (official test split)")

    # Save original_sample_map for stratified analysis
    sample_map_serializable = {str(k): v for k, v in train_sample_map.items()}
    with open("data/personalllm_sample_map.json", "w") as f:
        json.dump(sample_map_serializable, f)

    print("\nDone! Files saved to data/")


if __name__ == "__main__":
    main()