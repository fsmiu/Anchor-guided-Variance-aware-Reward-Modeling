"""Prepare HelpSteer2 dataset (nvidia/HelpSteer2, preference) for reward model training."""

import json
import os
import random
from collections import defaultdict, Counter

from datasets import load_dataset


def extract_soft_label(all_preferences_unprocessed, response1, response2):
    """Extract soft label from all_preferences_unprocessed annotations.

    HelpSteer2 preference strength scores:
      > 0  -> Response 2 is better (B wins)
      < 0  -> Response 1 is better (A wins)
      == 0 -> Tie (counts as 0.5 vote for each)

    Returns (chosen, rejected, soft_label) or None if soft tie.
    """
    votes_a = 0
    votes_b = 0
    votes_tie = 0

    for ann in all_preferences_unprocessed:
        strength = ann.get("strength", 0)
        if strength < 0:
            votes_a += 1
        elif strength > 0:
            votes_b += 1
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


def process_split(samples, split_name):
    """Process a list of HelpSteer2-preference samples and return (processed, original_sample_map)."""
    processed = []
    original_sample_map = defaultdict(list)

    skipped_empty_prompt = 0
    skipped_empty_response = 0
    skipped_no_label = 0
    skipped_soft_tie = 0
    total_annotations = 0

    for idx, sample in enumerate(samples):
        prompt = sample.get("prompt", "")
        if not prompt:
            skipped_empty_prompt += 1
            continue

        # Get responses
        response1 = sample.get("response_1", "")
        response2 = sample.get("response_2", "")
        if not response1 or not response2:
            skipped_empty_response += 1
            continue

        # Get all annotator preferences (unprocessed)
        all_prefs = sample.get("all_preferences_unprocessed", [])
        if not all_prefs:
            skipped_no_label += 1
            continue

        total_annotations += len(all_prefs)

        result = extract_soft_label(all_prefs, response1, response2)
        if result is None:
            skipped_soft_tie += 1
            continue

        chosen, rejected, soft_label = result

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
    print(f"  Total raw samples: {len(samples)}")
    print(f"  Skipped (empty prompt): {skipped_empty_prompt}")
    print(f"  Skipped (empty response): {skipped_empty_response}")
    print(f"  Skipped (no labels): {skipped_no_label}")
    print(f"  Total annotations processed: {total_annotations}")
    print(f"  Skipped (soft label tie): {skipped_soft_tie}")
    print(f"  Final instances: {len(processed)}")
    print(f"  Unique original samples kept: {len(original_sample_map)}")

    return processed, original_sample_map


def main():
    print("Loading HelpSteer2 dataset (nvidia/HelpSteer2, preference)...")
    ds = load_dataset("nvidia/HelpSteer2", data_dir="preference")["train"]
    print(f"Raw dataset size: {len(ds)}")
    print(f"Columns: {ds.column_names}")

    # Print a raw sample to understand structure
    print("\n--- Raw sample (keys & truncated values) ---")
    sample = ds[0]
    for key in sample:
        val = sample[key]
        val_str = str(val)
        if len(val_str) > 200:
            print(f"  {key}: {val_str[:200]}...")
        else:
            print(f"  {key}: {val_str}")

    print("\nUnique split values:", sorted(set(ds["split"])))
    print("Split counts:", Counter(ds["split"]))

    # Split by the 'split' field already present in the dataset
    train_samples = [s for s in ds if s.get("split") == "train"]
    val_samples = [s for s in ds if s.get("split") == "val"]
    print(f"\nSplit by 'split' field: {len(train_samples)} train, {len(val_samples)} validation")

    # Process both splits independently
    train_data, train_sample_map = process_split(train_samples, "train")
    heldout_data, heldout_sample_map = process_split(val_samples, "heldout")

    print(f"\n  -> {len(train_data)} train / {len(heldout_data)} heldout instances")

    # Save
    os.makedirs("data", exist_ok=True)
    with open("data/helpsteer2_train.json", "w") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open("data/helpsteer2_heldout.json", "w") as f:
        json.dump(heldout_data, f, indent=2, ensure_ascii=False)

    # Save total count (train only, excluding heldout) for size matching
    with open("data/helpsteer2_count.txt", "w") as f:
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
    with open("data/helpsteer2_sample_map.json", "w") as f:
        json.dump(sample_map_serializable, f)

    print("\nDone! Files saved to data/")


if __name__ == "__main__":
    main()