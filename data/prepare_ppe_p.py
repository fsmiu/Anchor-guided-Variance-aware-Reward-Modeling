"""Prepare PPE-P (PPE Human Preference V1) for evaluation."""

import json
import os
from datasets import load_dataset


def main():
    print("Loading lmarena-ai/PPE-Human-Preference-V1...")
    ds_dict = load_dataset("lmarena-ai/PPE-Human-Preference-V1")
    print(f"Available splits: {list(ds_dict.keys())}")
    ds = ds_dict[list(ds_dict.keys())[0]]
    print(f"Raw dataset size: {len(ds)}")
    print(f"Columns: {ds.column_names}")
    print(f"Sample[0]: {ds[0]}")

    processed = []
    skipped = 0

    for sample in ds:
        # Inspect the actual column names from the print above and adapt accordingly.
        # Expected columns: prompt, response_a, response_b, winner (or similar).
        # The column names below are placeholders — update after inspecting the sample.
        prompt = sample.get("prompt", "")
        # winner: "model_a" means response_1 is preferred; "model_b" means response_2 is preferred
        winner = sample.get("winner", "")
        response_1 = sample.get("response_1", "")
        response_2 = sample.get("response_2", "")

        if not prompt or not response_1 or not response_2 or not winner:
            skipped += 1
            continue

        if winner in ("model_a", "A", "a"):
            chosen = response_1
            rejected = response_2
        elif winner in ("model_b", "B", "b"):
            chosen = response_2
            rejected = response_1
        else:
            # tie or unknown — skip
            skipped += 1
            continue

        processed.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })

    print(f"Skipped: {skipped}")
    print(f"Final samples: {len(processed)}")

    os.makedirs("data", exist_ok=True)
    with open("data/ppe_p_test.json", "w") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)
    print("Saved to data/ppe_p_test.json")


if __name__ == "__main__":
    main()
