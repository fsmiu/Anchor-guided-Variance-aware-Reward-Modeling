"""Prepare UltraFeedback prompts for OOD PPO training and evaluation.

Produces two JSON files of the form ``[{"prompt": str}, ...]``:

- ``data/ultrafeedback_train.json``
- ``data/ultrafeedback_heldout.json``

The train/heldout sizes are matched to the existing MultiPref splits by reading:

- ``data/multipref_train.json``
- ``data/multipref_heldout.json``

We load ``allenai/ultrafeedback_binarized_cleaned`` (a well-maintained
variant). If that dataset is unavailable in the user's environment, we fall
back to ``HuggingFaceH4/ultrafeedback_binarized`` — the variants share the
``prompt`` field, so no other changes are required.

Determinism: sampling uses ``random.seed(42)`` so the train/heldout split is
reproducible across runs.
"""

import json
import os
import random

from datasets import load_dataset


DATASET_CANDIDATES = [
    # (repo, split). First one that loads is used.
    ("allenai/ultrafeedback_binarized_cleaned", "train_prefs"),
    ("allenai/ultrafeedback_binarized_cleaned", "train"),
    ("HuggingFaceH4/ultrafeedback_binarized", "train_prefs"),
    ("HuggingFaceH4/ultrafeedback_binarized", "train_sft"),
]

MULTIPREF_TRAIN_PATH = "data/multipref_train.json"
MULTIPREF_HELDOUT_PATH = "data/multipref_heldout.json"


def load_uf_dataset():
    last_err = None
    for repo, split in DATASET_CANDIDATES:
        try:
            print(f"Trying {repo} split={split} ...")
            ds = load_dataset(repo, split=split)
            print(f"  loaded: {len(ds)} rows, columns={ds.column_names}")
            return ds, repo, split
        except Exception as e:  # noqa: BLE001
            print(f"  failed: {e}")
            last_err = e
    raise RuntimeError(f"Could not load any UltraFeedback variant: {last_err}")


def load_target_sizes():
    with open(MULTIPREF_TRAIN_PATH, "r", encoding="utf-8") as f:
        multipref_train = json.load(f)
    with open(MULTIPREF_HELDOUT_PATH, "r", encoding="utf-8") as f:
        multipref_heldout = json.load(f)

    train_size = len(multipref_train)
    heldout_size = len(multipref_heldout)

    print("Using MultiPref reference sizes:")
    print(f"  {MULTIPREF_TRAIN_PATH}   ({train_size} prompts)")
    print(f"  {MULTIPREF_HELDOUT_PATH} ({heldout_size} prompts)")

    return train_size, heldout_size


def main():
    train_size, heldout_size = load_target_sizes()
    total_needed = train_size + heldout_size

    ds, repo, split = load_uf_dataset()
    print(f"\nUsing dataset: {repo} [{split}]")
    print(f"Sample[0] keys: {list(ds[0].keys())}")

    # Deduplicate prompts globally first, so train/heldout cannot overlap.
    seen = set()
    prompts = []
    for sample in ds:
        p = sample.get("prompt") or sample.get("instruction") or ""
        if not p:
            continue
        if p in seen:
            continue
        seen.add(p)
        prompts.append({"prompt": p})

    print(f"Unique prompts: {len(prompts)}")
    if len(prompts) < total_needed:
        raise RuntimeError(
            "UltraFeedback has only "
            f"{len(prompts)} unique prompts, need {total_needed} "
            f"({train_size} train + {heldout_size} heldout)."
        )

    random.seed(42)
    random.shuffle(prompts)

    # Both splits are random, disjoint samples from the shuffled unique pool.
    train = prompts[:train_size]
    heldout = prompts[train_size : train_size + heldout_size]

    # Explicit safety check: no prompt overlap between train and heldout.
    train_prompts = {x["prompt"] for x in train}
    heldout_prompts = {x["prompt"] for x in heldout}
    overlap = train_prompts & heldout_prompts
    if overlap:
        raise RuntimeError(f"Found {len(overlap)} overlapping prompts between train and heldout.")

    os.makedirs("data", exist_ok=True)
    train_path = "data/ultrafeedback_train.json"
    heldout_path = "data/ultrafeedback_heldout.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2, ensure_ascii=False)
    with open(heldout_path, "w", encoding="utf-8") as f:
        json.dump(heldout, f, indent=2, ensure_ascii=False)

    print("\nSaved:")
    print(f"  {train_path}   ({len(train)} prompts)")
    print(f"  {heldout_path} ({len(heldout)} prompts)")

    print("\nThree random train samples:")
    for sample in random.sample(train, min(3, len(train))):
        preview = sample["prompt"].replace("\n", " ")[:160]
        print(f"  - {preview}...")


if __name__ == "__main__":
    main()