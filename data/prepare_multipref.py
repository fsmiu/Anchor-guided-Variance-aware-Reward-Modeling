"""Prepare MultiPref dataset (allenai/multipref) for reward model training."""

import json
import os
import random
from collections import defaultdict

from datasets import load_dataset


# Labels where A is preferred
A_WINS = {"A-is-clearly-better", "A-is-slightly-better"}
# Labels where B is preferred
B_WINS = {"B-is-clearly-better", "B-is-slightly-better"}
# Tie label — discard
TIES = {"Tie"}


def extract_annotations(sample):
    """Extract all annotator overall_pref labels from a sample.

    allenai/multipref has:
      - normal_worker_annotations: list of 2 annotation dicts
      - expert_worker_annotations: list of 2 annotation dicts
    Each dict has an 'overall_pref' field.

    Returns list of overall_pref strings (up to 4).
    """
    labels = []
    for field in ["normal_worker_annotations", "expert_worker_annotations"]:
        annotations = sample.get(field, [])
        if annotations is None:
            continue
        for ann in annotations:
            if isinstance(ann, dict) and "overall_pref" in ann:
                labels.append(ann["overall_pref"])
    return labels


def main():
    print("Loading MultiPref dataset (allenai/multipref)...")
    ds = load_dataset("allenai/multipref", "default", split="train")
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

    # Process samples
    processed = []
    original_sample_map = defaultdict(list)  # sample_id -> list of instance indices
    skipped_no_label = 0
    skipped_soft_tie = 0
    total_annotations = 0

    for idx, sample in enumerate(ds):
        # Get prompt text
        prompt_text = sample.get("text", "")
        if not prompt_text:
            continue

        # 保留所有样本，不再过滤 multi-turn

        # Get responses
        completion_a = sample.get("completion_a", "")
        completion_b = sample.get("completion_b", "")
        if not completion_a or not completion_b:
            continue

        # Get all annotator labels (up to 4: 2 normal + 2 expert)
        labels = extract_annotations(sample)
        if not labels:
            skipped_no_label += 1
            continue

        # Count votes for A vs B (and ties) across all annotators
        votes_a = 0
        votes_b = 0
        votes_tie = 0

        for label in labels:
            total_annotations += 1
            if label in A_WINS:
                votes_a += 1
            elif label in B_WINS:
                votes_b += 1
            elif label in TIES:
                votes_tie += 1
            # unknown labels are ignored

        votes_a_eff = votes_a + 0.5 * votes_tie
        votes_b_eff = votes_b + 0.5 * votes_tie
        total_eff = votes_a_eff + votes_b_eff

        if total_eff == 0:
            continue

        soft_a = votes_a_eff / total_eff

        if soft_a > 0.5:
            chosen = completion_a
            rejected = completion_b
            soft_label = soft_a
        elif soft_a < 0.5:
            chosen = completion_b
            rejected = completion_a
            soft_label = 1.0 - soft_a
        else:
            # soft_a == 0.5: equal effective votes, no signal
            skipped_soft_tie += 1
            continue

        instance = {
            "prompt": prompt_text,
            "chosen": chosen,
            "rejected": rejected,
            "original_sample_id": idx,
            "soft_label": soft_label,
        }
        processed.append(instance)
        original_sample_map[idx].append(len(processed) - 1)

    print(f"\nProcessing summary:")
    print(f"  Total raw samples: {len(ds)}")
    print(f"  Skipped (no labels): {skipped_no_label}")
    print(f"  Total annotations processed: {total_annotations}")
    print(f"  Skipped (soft label tie): {skipped_soft_tie}")
    print(f"  Final expanded instances: {len(processed)}")
    print(f"  Unique original samples kept: {len(original_sample_map)}")

    # Build prompt-level groups so that all instances sharing the same prompt
    # text end up in the same split (prevents prompt leakage).
    prompt_group_map = defaultdict(list)  # prompt_text -> [instance_indices]
    for i, inst in enumerate(processed):
        prompt_group_map[inst["prompt"]].append(i)

    print(f"  Unique prompts: {len(prompt_group_map)}")

    # Shuffle and split 90/10 at the *prompt* level.
    # The heldout set serves as both val (best-checkpoint selection) and test (ID analysis).
    random.seed(42)
    prompt_keys = list(prompt_group_map.keys())
    random.shuffle(prompt_keys)

    split_point = int(0.95 * len(prompt_keys))
    train_prompts = set(prompt_keys[:split_point])
    heldout_prompts = set(prompt_keys[split_point:])

    train_data = [inst for inst in processed if inst["prompt"] in train_prompts]
    heldout_data = [inst for inst in processed if inst["prompt"] in heldout_prompts]

    print(f"\nSplit (by prompt): {len(train_prompts)} train / "
          f"{len(heldout_prompts)} heldout prompts")
    print(f"  -> {len(train_data)} train / {len(heldout_data)} heldout instances")

    # Save
    os.makedirs("data", exist_ok=True)
    with open("data/multipref_train.json", "w") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open("data/multipref_heldout.json", "w") as f:
        json.dump(heldout_data, f, indent=2, ensure_ascii=False)

    # Save total count for Skywork size matching
    with open("data/multipref_count.txt", "w") as f:
        f.write(str(len(processed)))

    # Print 3 random samples
    print("\n--- 3 random training samples ---")
    for s in random.sample(train_data, min(3, len(train_data))):
        print(f"  Prompt: {s['prompt'][:120]}...")
        print(f"  Chosen: {s['chosen'][:120]}...")
        print(f"  Rejected: {s['rejected'][:120]}...")
        print()

    # Dataset stats
    print(f"Dataset statistics:")
    print(f"  Total expanded samples: {len(processed)}")
    print(f"  Train: {len(train_data)}")
    print(f"  Heldout: {len(heldout_data)}")
    print(f"  Unique prompts: {len(prompt_group_map)}")
    print(f"  Unique original samples: {len(original_sample_map)}")

    # Save original_sample_map for stratified analysis
    sample_map_serializable = {str(k): v for k, v in original_sample_map.items()}
    with open("data/multipref_sample_map.json", "w") as f:
        json.dump(sample_map_serializable, f)

    print("\nDone! Files saved to data/")


if __name__ == "__main__":
    main()