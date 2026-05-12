"""Prepare HelpSteer3 dataset (nvidia/HelpSteer3) for reward model training."""

import json
import os
import random
from collections import defaultdict

from datasets import load_dataset


def is_single_turn(context):
    """Check if a context is single-turn (only one user message, no history)."""
    return len(context) == 1


def extract_soft_label(individual_preference, response1, response2):
    """Extract soft label from individual preference annotations.

    HelpSteer3 individual_preference scores:
      > 0  -> B wins (response2 is better)
      < 0  -> A wins (response1 is better)
      == 0 -> Tie (counts as 0.5 vote for each)

    Returns (chosen, rejected, soft_label) or None if soft tie.
    """
    votes_a = 0
    votes_b = 0
    votes_tie = 0

    for ann in individual_preference:
        score = ann.get("score", 0)
        if score > 0:
            votes_b += 1
        elif score < 0:
            votes_a += 1
        else:
            votes_tie += 1

    votes_a_eff = votes_a + 0.5 * votes_tie
    votes_b_eff = votes_b + 0.5 * votes_tie
    total_eff = votes_a_eff + votes_b_eff

    if total_eff == 0:
        return None

    soft_a = votes_a_eff / total_eff  # response1 win rate

    if soft_a > 0.5:
        return response1, response2, soft_a
    elif soft_a < 0.5:
        return response2, response1, 1.0 - soft_a
    else:
        # soft label == 0.5: no signal, discard
        return None


def process_split(ds, split_name):
    """Process a HelpSteer3 split and return (processed, original_sample_map)."""
    processed = []
    original_sample_map = defaultdict(list)
    skipped_multi_turn = 0
    skipped_no_label = 0
    skipped_soft_tie = 0
    total_annotations = 0

    for idx, sample in enumerate(ds):
        context = sample.get("context", [])
        if not context:
            continue

        # Filter: single-turn only
        if not is_single_turn(context):
            skipped_multi_turn += 1
            continue

        # Get responses
        response1 = sample.get("response1", "")
        response2 = sample.get("response2", "")
        if not response1 or not response2:
            continue

        # Get individual preference annotations
        individual_preference = sample.get("individual_preference", [])
        if not individual_preference:
            skipped_no_label += 1
            continue

        total_annotations += len(individual_preference)

        result = extract_soft_label(individual_preference, response1, response2)
        if result is None:
            skipped_soft_tie += 1
            continue

        chosen, rejected, soft_label = result

        # Use the single user message as prompt
        prompt = context[0]["content"]

        instance = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "original_sample_id": idx,
            "soft_label": soft_label,
            "domain": sample.get("domain", ""),
            "language": sample.get("language", ""),
        }
        processed.append(instance)
        original_sample_map[idx].append(len(processed) - 1)

    print(f"\nProcessing summary [{split_name}]:")
    print(f"  Total raw samples: {len(ds)}")
    print(f"  Skipped (multi-turn): {skipped_multi_turn}")
    print(f"  Skipped (no labels): {skipped_no_label}")
    print(f"  Total annotations processed: {total_annotations}")
    print(f"  Skipped (soft label tie): {skipped_soft_tie}")
    print(f"  Final instances: {len(processed)}")
    print(f"  Unique original samples kept: {len(original_sample_map)}")

    return processed, original_sample_map


def main():
    print("Loading HelpSteer3 dataset (nvidia/HelpSteer3, preference)...")
    ds_train   = load_dataset("nvidia/HelpSteer3", "preference", split="train")
    ds_heldout = load_dataset("nvidia/HelpSteer3", "preference", split="validation")
    print(f"Raw train size: {len(ds_train)}, Raw heldout size: {len(ds_heldout)}")
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

    # Process both splits independently — the official HF train/validation
    # splits are used as-is for train/heldout (no synthetic subdivision).
    train_data,   train_sample_map   = process_split(ds_train,   "train")
    heldout_data, heldout_sample_map = process_split(ds_heldout, "heldout")

    print(f"\n  -> {len(train_data)} train / {len(heldout_data)} heldout instances")

    # Save
    os.makedirs("data", exist_ok=True)
    with open("data/helpsteer3_train.json", "w") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open("data/helpsteer3_heldout.json", "w") as f:
        json.dump(heldout_data, f, indent=2, ensure_ascii=False)

    # Save total count (train only, excluding heldout) for size matching
    with open("data/helpsteer3_count.txt", "w") as f:
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
    print(f"  Heldout: {len(heldout_data)} (official validation split)")

    # Save original_sample_map for stratified analysis
    sample_map_serializable = {str(k): v for k, v in train_sample_map.items()}
    with open("data/helpsteer3_sample_map.json", "w") as f:
        json.dump(sample_map_serializable, f)

    print("\nDone! Files saved to data/")


if __name__ == "__main__":
    main()