"""Load and prepare RewardBench for OOD evaluation."""

import json
import os

from datasets import load_dataset


def main():
    print("Loading RewardBench dataset...")
    ds = load_dataset("allenai/reward-bench", split="filtered")
    print(f"Dataset size: {len(ds)}")
    print(f"Columns: {ds.column_names}")

    # Print a raw sample
    print("\n--- Raw sample ---")
    sample = ds[0]
    for key in sample:
        val = sample[key]
        if isinstance(val, str) and len(val) > 200:
            print(f"  {key}: {val[:200]}...")
        else:
            print(f"  {key}: {val}")

    # Organize by category
    categories = {}
    processed = []

    for sample in ds:
        prompt = sample.get("prompt", "")
        chosen = sample.get("chosen", "")
        rejected = sample.get("rejected", "")
        # RewardBench has a 'subset' field indicating the category
        subset = sample.get("subset", "unknown")

        # Map subsets to high-level categories
        category = map_subset_to_category(subset)

        instance = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "subset": subset,
            "category": category,
        }
        processed.append(instance)

        if category not in categories:
            categories[category] = 0
        categories[category] += 1

    print(f"\nCategory distribution:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    # Save
    os.makedirs("data", exist_ok=True)
    with open("data/rewardbench.json", "w") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    print(f"\nTotal samples: {len(processed)}")
    print("Done! File saved to data/rewardbench.json")


def map_subset_to_category(subset):
    """Map RewardBench subset names to high-level categories."""
    subset_lower = subset.lower()

    chat_hard_subsets = {
        "mt-bench-hard", "llmbar-natural", "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst", "llmbar-adver-GPTOut", "llmbar-adver-manual",
    }
    safety_subsets = {
        "refusals-dangerous", "refusals-offensive", "xstest-should-refuse",
        "xstest-should-respond", "do not answer",
    }
    reasoning_subsets = {
        "math-prm-human", "hep-cpp", "hep-go", "hep-java", "hep-js",
        "hep-python", "hep-rust",
    }
    chat_subsets = {
        "alpacaeval-easy", "alpacaeval-length", "alpacaeval-hard",
        "mt-bench-easy", "mt-bench-med",
    }

    if subset in chat_hard_subsets:
        return "Chat Hard"
    elif subset in safety_subsets:
        return "Safety"
    elif subset in reasoning_subsets:
        return "Reasoning"
    elif subset in chat_subsets:
        return "Chat"
    else:
        # Fallback heuristics
        if "safety" in subset_lower or "refus" in subset_lower or "xstest" in subset_lower:
            return "Safety"
        elif "math" in subset_lower or "hep" in subset_lower or "code" in subset_lower:
            return "Reasoning"
        elif "hard" in subset_lower or "llmbar" in subset_lower:
            return "Chat Hard"
        else:
            return "Chat"


if __name__ == "__main__":
    main()
