"""Score multipref data with Skywork-Reward-V2-Llama-3.1-8B and compute metrics.

Loads the Skywork reward model, scores chosen and rejected responses for each
entry in multipref_train.json and multipref_heldout.json, writes the scores back
into the JSON files, and prints accuracy / brier / log-loss metrics.

Usage
-----
python scripts/generate_skywork_rewards.py
python scripts/generate_skywork_rewards.py --batch_size 4 --max_length 2048
"""

import argparse
import json
import math
import os
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm



def load_reward_model(model_name, device="cuda"):
    """Load a HuggingFace reward model (AutoModelForSequenceClassification)."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        num_labels=1,
        use_cache=False,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


@torch.no_grad()
def score_responses(prompts, responses, model, tokenizer, batch_size=8, max_length=2048):
    """Score a list of (prompt, response) pairs using the reward model.

    Returns a list of float scores, one per pair.
    """
    # Pre-format all texts using chat template
    texts = []
    for prompt, response in zip(prompts, responses):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)

    all_scores = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Scoring"):
        batch_texts = texts[start : start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        output = model(**inputs)
        scores = output.logits[:, 0].float().cpu().tolist()
        all_scores.extend(scores)

    return all_scores


def process_file(data_path, model, tokenizer, batch_size, max_length, anchor_key=None,
                 norm_mean=None):
    """Score chosen/rejected in a JSON file and return updated data + scores + labels.

    If norm_mean is provided, it is subtracted from every score (mean-centering).
    The scale (std) is intentionally kept intact so that anchor thresholds and
    loss magnitudes stay in the original RM's natural units.  When norm_mean is
    None (train split), raw scores are returned so the caller can compute the
    mean and then overwrite the file.
    """
    with open(data_path) as f:
        data = json.load(f)

    prompts = [d["prompt"] for d in data]
    chosens = [d["chosen"] for d in data]
    rejecteds = [d["rejected"] for d in data]
    soft_labels = [d.get("soft_label", 1.0) for d in data]

    print(f"\nScoring chosen responses ({len(data)} samples)...")
    chosen_scores = score_responses(prompts, chosens, model, tokenizer, batch_size, max_length)

    print(f"Scoring rejected responses ({len(data)} samples)...")
    rejected_scores = score_responses(prompts, rejecteds, model, tokenizer, batch_size, max_length)

    # Mean-center scores if train mean is provided (no std division)
    if norm_mean is not None:
        chosen_scores = [s - norm_mean for s in chosen_scores]
        rejected_scores = [s - norm_mean for s in rejected_scores]

    # Write scores back into data using nested anchor_scores format
    for i, entry in enumerate(data):
        if anchor_key:
            entry.setdefault("anchor_scores", {})[anchor_key] = {
                "chosen_score": chosen_scores[i],
                "rejected_score": rejected_scores[i],
            }
        else:
            entry["chosen_score"] = chosen_scores[i]
            entry["rejected_score"] = rejected_scores[i]

    # Save updated JSON
    with open(data_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved scores to {data_path}")

    return chosen_scores, rejected_scores, soft_labels


def compute_train_mean(chosen_scores, rejected_scores):
    """Compute mean over all train scores (chosen + rejected combined)."""
    all_scores = chosen_scores + rejected_scores
    return sum(all_scores) / len(all_scores)


def _sigmoid(x):
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def print_metrics(split_name, chosen_scores, rejected_scores, soft_labels):
    """Compute and print metrics for a split using soft labels."""
    n = len(chosen_scores)

    # Accuracy (hard: chosen_score > rejected_score)
    acc = sum(c > r for c, r in zip(chosen_scores, rejected_scores)) / n if n > 0 else 0.0

    # Brier: mean((p - y)^2) where p = sigmoid(chosen - rejected)
    probs = [_sigmoid(c - r) for c, r in zip(chosen_scores, rejected_scores)]
    brier = sum((p - y) ** 2 for p, y in zip(probs, soft_labels)) / n if n > 0 else 0.0

    # Log-loss: -mean(y * log(p) + (1-y) * log(1-p))
    ll = 0.0
    for p, y in zip(probs, soft_labels):
        p_clamped = max(min(p, 1 - 1e-7), 1e-7)
        ll += -(y * math.log(p_clamped) + (1 - y) * math.log(1 - p_clamped))
    ll = ll / n if n > 0 else 0.0

    print(f"\n{'=' * 50}")
    print(f"  Split:    {split_name}")
    print(f"  N pairs:  {n}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Brier:    {brier:.4f}")
    print(f"  Log-loss: {ll:.4f}")
    print(f"{'=' * 50}")

    return {"accuracy": acc, "brier": brier, "logloss": ll}


def main():
    parser = argparse.ArgumentParser(description="Generate Skywork reward scores for multipref data")
    parser.add_argument("--model_name", type=str, default="Skywork/Skywork-Reward-V2-Llama-3.2-3B")
    parser.add_argument("--train_data", type=str, default="data/multipref_train.json")
    parser.add_argument("--val_data", type=str, default="data/multipref_heldout.json")
    parser.add_argument("--test_data", type=str, default="data/multipref_heldout.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dataset_name", type=str, default=None,
        help="Dataset identifier used in output file names. "
             "Defaults to the stem of --train_data before the first '_' "
             "(e.g. 'multipref' from 'data/multipref_train.json').",
    )
    args = parser.parse_args()

    # Derive anchor key from model name (e.g. "skywork-reward-v2-llama-3.1-8b")
    anchor_key = args.model_name.split("/")[-1].lower()
    dataset_name = args.dataset_name or os.path.basename(args.train_data).split("_")[0]
    run_key = f"{anchor_key}_{dataset_name}"

    print(f"Loading model: {args.model_name}")
    print(f"Anchor key: {anchor_key}")
    print(f"Dataset: {dataset_name}")
    model, tokenizer = load_reward_model(args.model_name, args.device)
    print("Model loaded.\n")

    all_metrics = {}

    # ------------------------------------------------------------------
    # Step 1: score the train split and derive the mean for centering
    # ------------------------------------------------------------------
    norm_mean = None

    if os.path.exists(args.train_data):
        split_name = os.path.basename(args.train_data)
        print(f"\n{'=' * 50}")
        print(f"Processing (train): {args.train_data}")
        print(f"{'=' * 50}")

        # Score without centering first so we can compute the mean
        chosen_scores, rejected_scores, soft_labels = process_file(
            args.train_data, model, tokenizer, args.batch_size, args.max_length,
            anchor_key=anchor_key, norm_mean=None,
        )
        norm_mean = compute_train_mean(chosen_scores, rejected_scores)
        print(f"\nTrain score mean (will be subtracted): {norm_mean:.4f}")

        # Overwrite the file with mean-centred scores
        chosen_scores_norm = [s - norm_mean for s in chosen_scores]
        rejected_scores_norm = [s - norm_mean for s in rejected_scores]

        with open(args.train_data) as f:
            data = json.load(f)
        for i, entry in enumerate(data):
            if anchor_key:
                entry.setdefault("anchor_scores", {})[anchor_key] = {
                    "chosen_score": chosen_scores_norm[i],
                    "rejected_score": rejected_scores_norm[i],
                }
            else:
                entry["chosen_score"] = chosen_scores_norm[i]
                entry["rejected_score"] = rejected_scores_norm[i]
        with open(args.train_data, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved mean-centred scores to {args.train_data}")

        metrics = print_metrics(split_name, chosen_scores_norm, rejected_scores_norm, soft_labels)
        all_metrics[split_name] = metrics
    else:
        print(f"[skip] File not found: {args.train_data}")

    # ------------------------------------------------------------------
    # Step 2: score val / test using train normalisation statistics
    # ------------------------------------------------------------------
    extra_splits = []
    if args.val_data:
        extra_splits.append(args.val_data)
    extra_splits.append(args.test_data)

    for data_path in extra_splits:
        if not os.path.exists(data_path):
            print(f"[skip] File not found: {data_path}")
            continue

        split_name = os.path.basename(data_path)
        print(f"\n{'=' * 50}")
        print(f"Processing: {data_path}")
        print(f"{'=' * 50}")

        chosen_scores, rejected_scores, soft_labels = process_file(
            data_path, model, tokenizer, args.batch_size, args.max_length,
            anchor_key=anchor_key, norm_mean=norm_mean,
        )
        metrics = print_metrics(split_name, chosen_scores, rejected_scores, soft_labels)
        all_metrics[split_name] = metrics

    # Save metrics summary + normalisation statistics
    os.makedirs("results", exist_ok=True)
    metrics_path = f"results/{run_key}_reward_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    if norm_mean is not None:
        norm_stats = {"mean": norm_mean}
        norm_path = f"results/{run_key}_norm_stats.json"
        with open(norm_path, "w") as f:
            json.dump(norm_stats, f, indent=2)
        print(f"Norm stats saved to {norm_path}")


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    main()
