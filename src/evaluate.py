"""Evaluation script for reward models on ID test sets and RewardBench OOD."""

import argparse
import json
import math
import os
import re
from collections import defaultdict

import torch
import yaml
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm

from src.model import BTRewardModel, MeanVarRewardModel, create_reward_model
from src.dataset import PreferenceDataset, RewardBenchDataset, collate_fn, rewardbench_collate_fn, ppe_p_collate_fn


def brier_score(probs, soft_labels):
    """Mean squared error between predicted P(chosen wins) and soft label."""
    return sum((p - y) ** 2 for p, y in zip(probs, soft_labels)) / len(probs)


def ce_loss(probs, soft_labels):
    """Cross-entropy with soft labels: -[y*log(p) + (1-y)*log(1-p)]."""
    total = 0.0
    for p, y in zip(probs, soft_labels):
        p = max(min(p, 1 - 1e-7), 1e-7)
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(probs)


# def load_model(checkpoint_dir, model_type, device="cuda"):
#     """Load a trained reward model from checkpoint."""
#     config_path = os.path.join(checkpoint_dir, "train_config.json")
#     if os.path.exists(config_path):
#         with open(config_path) as f:
#             train_config = json.load(f)
#         model_name = train_config["model_name"]
#     else:
#         model_name = "NousResearch/Meta-Llama-3.1-8B-Instruct"

#     config = AutoConfig.from_pretrained(checkpoint_dir)

#     if model_type == "bt":
#         model = BTRewardModel(config)
#     else:
#         model = MeanVarRewardModel(config)

#     # Load weights from checkpoint
#     state_dict_path = os.path.join(checkpoint_dir, "model.safetensors")
#     if os.path.exists(state_dict_path):
#         from safetensors.torch import load_file
#         state_dict = load_file(state_dict_path)
#         model.load_state_dict(state_dict)
#     else:
#         # Try pytorch_model.bin
#         bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
#         if os.path.exists(bin_path):
#             state_dict = torch.load(bin_path, map_location="cpu")
#             model.load_state_dict(state_dict)
#         else:
#             # Try sharded safetensors
#             from safetensors.torch import load_file
#             import glob
#             shard_files = sorted(glob.glob(os.path.join(checkpoint_dir, "model-*.safetensors")))
#             if shard_files:
#                 state_dict = {}
#                 for sf in shard_files:
#                     state_dict.update(load_file(sf))
#                 model.load_state_dict(state_dict)
#             else:
#                 raise FileNotFoundError(f"No model weights found in {checkpoint_dir}")

#     model = model.to(torch.bfloat16).to(device)
#     model.eval()
#     return model, model_name


def load_model(checkpoint_dir, model_type, device="cuda"):
    """Load a trained reward model from checkpoint."""
    config_path = os.path.join(checkpoint_dir, "train_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            train_config = json.load(f)
        model_name = train_config["model_name"]
    else:
        model_name = "NousResearch/Meta-Llama-3.1-8B-Instruct"

    # Build model without fully re-initializing the backbone
    model = create_reward_model(
        model_name_or_path=model_name,
        model_type=model_type,
        torch_dtype=torch.bfloat16,
    )

    # Load reward-model checkpoint weights
    state_dict_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(state_dict_path):
        from safetensors.torch import load_file
        state_dict = load_file(state_dict_path)
        model.load_state_dict(state_dict)
    else:
        bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            from safetensors.torch import load_file
            import glob

            shard_files = sorted(glob.glob(os.path.join(checkpoint_dir, "model-*.safetensors")))
            if shard_files:
                state_dict = {}
                for sf in shard_files:
                    state_dict.update(load_file(sf))
                model.load_state_dict(state_dict)
            else:
                raise FileNotFoundError(f"No model weights found in {checkpoint_dir}")

    model = model.to(torch.bfloat16).to(device)
    model.eval()
    return model, model_name


@torch.no_grad()
def evaluate_id(model, model_type, dataloader, device="cuda"):
    correct = 0
    total = 0
    all_probs = []
    all_soft_labels = []

    for batch in tqdm(dataloader, desc="Evaluating ID"):
        input_ids_chosen = batch["input_ids_chosen"].to(device)
        attention_mask_chosen = batch["attention_mask_chosen"].to(device)
        input_ids_rejected = batch["input_ids_rejected"].to(device)
        attention_mask_rejected = batch["attention_mask_rejected"].to(device)
        soft_label = batch.get("soft_label")

        if model_type == "bt":
            r_w = model(input_ids=input_ids_chosen, attention_mask=attention_mask_chosen)
            r_l = model(input_ids=input_ids_rejected, attention_mask=attention_mask_rejected)
            margin = r_w - r_l
            probs = torch.sigmoid(margin).cpu().tolist()
        else:
            r_w, var_w = model(input_ids=input_ids_chosen, attention_mask=attention_mask_chosen)
            r_l, var_l = model(input_ids=input_ids_rejected, attention_mask=attention_mask_rejected)
            margin = r_w - r_l
            std = torch.sqrt(var_w + var_l)
            normal = torch.distributions.Normal(0, 1)
            probs = normal.cdf(margin / std).cpu().tolist()

        correct += (r_w > r_l).sum().item()
        total += r_w.size(0)

        if isinstance(probs, float):
            probs = [probs]
        all_probs.extend(probs)

        if soft_label is not None:
            all_soft_labels.extend(soft_label.cpu().tolist())

    accuracy = correct / total if total > 0 else 0.0
    metrics = {"accuracy": accuracy}

    if all_soft_labels:
        metrics["brier"] = brier_score(all_probs, all_soft_labels)
        metrics["ce_loss"] = ce_loss(all_probs, all_soft_labels)

    return metrics


@torch.no_grad()
def evaluate_rewardbench(model, model_type, dataloader, device="cuda"):
    category_correct = defaultdict(int)
    category_total = defaultdict(int)
    all_probs = []

    for batch in tqdm(dataloader, desc="Evaluating RewardBench"):
        input_ids_chosen = batch["input_ids_chosen"].to(device)
        attention_mask_chosen = batch["attention_mask_chosen"].to(device)
        input_ids_rejected = batch["input_ids_rejected"].to(device)
        attention_mask_rejected = batch["attention_mask_rejected"].to(device)
        categories = batch["category"]

        if model_type == "bt":
            r_w = model(input_ids=input_ids_chosen, attention_mask=attention_mask_chosen)
            r_l = model(input_ids=input_ids_rejected, attention_mask=attention_mask_rejected)
            margin = r_w - r_l
            probs = torch.sigmoid(margin).cpu().tolist()
        else:
            r_w, var_w = model(input_ids=input_ids_chosen, attention_mask=attention_mask_chosen)
            r_l, var_l = model(input_ids=input_ids_rejected, attention_mask=attention_mask_rejected)
            margin = r_w - r_l
            std = torch.sqrt(var_w + var_l)
            normal = torch.distributions.Normal(0, 1)
            probs = normal.cdf(margin / std).cpu().tolist()

        if isinstance(probs, float):
            probs = [probs]
        all_probs.extend(probs)

        is_correct = (r_w > r_l)
        for i, cat in enumerate(categories):
            category_correct[cat] += is_correct[i].item()
            category_total[cat] += 1

    results = {}
    total_correct = 0
    total_count = 0
    for cat in sorted(category_total.keys()):
        acc = category_correct[cat] / category_total[cat]
        results[cat] = acc
        total_correct += category_correct[cat]
        total_count += category_total[cat]

    results["Overall"] = total_correct / total_count if total_count > 0 else 0.0

    # Hard label = 1.0 for all RewardBench samples
    hard_labels = [1.0] * len(all_probs)
    results["brier"] = brier_score(all_probs, hard_labels)
    results["ce_loss"] = ce_loss(all_probs, hard_labels)

    return results


@torch.no_grad()
def evaluate_ppe_p(model, model_type, dataloader, device="cuda"):
    """Evaluate on PPE-P (PPE Human Preference V1). Hard label = 1.0."""
    correct = 0
    total = 0
    all_probs = []

    for batch in tqdm(dataloader, desc="Evaluating PPE-P"):
        input_ids_chosen = batch["input_ids_chosen"].to(device)
        attention_mask_chosen = batch["attention_mask_chosen"].to(device)
        input_ids_rejected = batch["input_ids_rejected"].to(device)
        attention_mask_rejected = batch["attention_mask_rejected"].to(device)

        if model_type == "bt":
            r_w = model(input_ids=input_ids_chosen, attention_mask=attention_mask_chosen)
            r_l = model(input_ids=input_ids_rejected, attention_mask=attention_mask_rejected)
            margin = r_w - r_l
            probs = torch.sigmoid(margin).cpu().tolist()
        else:
            r_w, var_w = model(input_ids=input_ids_chosen, attention_mask=attention_mask_chosen)
            r_l, var_l = model(input_ids=input_ids_rejected, attention_mask=attention_mask_rejected)
            margin = r_w - r_l
            std = torch.sqrt(var_w + var_l)
            normal = torch.distributions.Normal(0, 1)
            probs = normal.cdf(margin / std).cpu().tolist()

        if isinstance(probs, float):
            probs = [probs]
        all_probs.extend(probs)
        correct += (r_w > r_l).sum().item()
        total += r_w.size(0)

    hard_labels = [1.0] * len(all_probs)
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "brier": brier_score(all_probs, hard_labels),
        "ce_loss": ce_loss(all_probs, hard_labels),
    }


@torch.no_grad()
def evaluate_stratified_multipref(model, model_type, test_data_path, sample_map_path,
                                   tokenizer, max_length=2048, batch_size=32, device="cuda"):
    """Stratified evaluation on MultiPref by annotator agreement.

    High-agreement: all valid annotators chose the same side
    Diverging: annotators disagreed
    """
    with open(test_data_path) as f:
        test_data = json.load(f)
    with open(sample_map_path) as f:
        sample_map = json.load(f)

    # Build reverse map: instance_idx -> original_sample_id
    instance_to_original = {}
    for orig_id, instance_indices in sample_map.items():
        for inst_idx in instance_indices:
            instance_to_original[inst_idx] = orig_id

    # Group test instances by original sample ID
    # We need to figure out which test instances correspond to which original samples
    # The test data preserves original_sample_id
    original_groups = defaultdict(list)
    for i, sample in enumerate(test_data):
        orig_id = sample.get("original_sample_id")
        if orig_id is not None:
            original_groups[orig_id].append(i)

    # For each original sample, check agreement
    # High-agreement: all annotations point same direction
    # Diverging: mixed directions
    high_agreement_indices = []
    diverging_indices = []

    for orig_id, indices in original_groups.items():
        # All instances from this original sample in the test set
        # Since we binarized, "chosen" is always the winner
        # To check agreement, we need to see if all instances agree on which response is better
        # Since the prompt is the same, check if chosen/rejected are consistent
        if len(indices) <= 1:
            # Only one annotation, can't determine agreement
            high_agreement_indices.extend(indices)
            continue

        # Check if all instances have the same chosen response
        chosens = set()
        for idx in indices:
            chosens.add(test_data[idx]["chosen"][:100])  # use prefix as key

        if len(chosens) == 1:
            high_agreement_indices.extend(indices)
        else:
            diverging_indices.extend(indices)

    # Evaluate each group
    dataset = PreferenceDataset.__new__(PreferenceDataset)
    dataset.tokenizer = tokenizer
    dataset.max_length = max_length

    results = {}

    for group_name, indices in [("High-Agreement", high_agreement_indices),
                                 ("Diverging", diverging_indices)]:
        if not indices:
            results[group_name] = 0.0
            continue

        dataset.data = [test_data[i] for i in indices]
        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4)
        id_metrics = evaluate_id(model, model_type, loader, device)
        acc = id_metrics["accuracy"]
        results[group_name] = acc
        print(f"  {group_name}: {acc:.4f} ({len(indices)} samples)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate reward model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_scope", choices=("all", "id", "ood"), default="all",
                        help="Evaluation scope: all preserves the default behavior; id skips OOD; ood skips ID.")
    args = parser.parse_args()

    config = load_config(args.config)
    model_type = config["model_type"]
    dataset_name = "ood" if args.eval_scope == "ood" else config.get("dataset_name", "unknown")
    max_length = config.get("max_length", 2048)

    # Prefer run_name from checkpoint's train_config (has CLI overrides baked in)
    train_config_path = os.path.join(args.checkpoint, "train_config.json")
    if os.path.exists(train_config_path):
        with open(train_config_path) as f:
            train_cfg = json.load(f)
        run_name = train_cfg.get("run_name", config.get("run_name", "model"))
    else:
        run_name = config.get("run_name", "model")

    print(f"=== Evaluating {run_name} ===")
    print(f"  Model type: {model_type}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Eval scope: {args.eval_scope}")

    # Load model
    model, model_name = load_model(args.checkpoint, model_type, args.device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    results = {"run_name": run_name, "model_type": model_type, "dataset": dataset_name}

    # Record checkpoint step and anchor tau values from train config
    if os.path.exists(train_config_path):
        checkpoint_step = train_cfg.get("checkpoint_step")
        if checkpoint_step is not None:
            results["checkpoint_step"] = checkpoint_step
        for key in ("anchor_method", "resolved_anchor_tau",
                     "resolved_anchor_tau1", "resolved_anchor_tau2",
                     "anchor_sample_percent", "anchor_sample_count",
                     "anchor_sample_seed"):
            val = train_cfg.get(key)
            if val is not None:
                results[key] = val

    # Fallback: parse step from checkpoint path
    if "checkpoint_step" not in results:
        m = re.search(r"checkpoint-(\d+)/?$", args.checkpoint)
        if m:
            results["checkpoint_step"] = int(m.group(1))

    if "checkpoint_step" in results:
        print(f"  Checkpoint step: {results['checkpoint_step']}")
    if "resolved_anchor_tau" in results:
        print(f"  Anchor tau: {results['resolved_anchor_tau']}")
    if "resolved_anchor_tau1" in results:
        print(f"  Anchor tau1: {results['resolved_anchor_tau1']}")
    if "resolved_anchor_tau2" in results:
        print(f"  Anchor tau2: {results['resolved_anchor_tau2']}")
    if "anchor_sample_percent" in results:
        print(f"  Anchor sample percent: {results['anchor_sample_percent']:g}%")
    if "anchor_sample_count" in results:
        print(f"  Anchor sample count: {results['anchor_sample_count']}")

    # 1. In-distribution evaluation
    test_data_path = config.get("test_data")
    if args.eval_scope == "ood":
        print("\n--- In-Distribution Evaluation: skipped (eval_scope=ood) ---")
    elif test_data_path and os.path.exists(test_data_path):
        print(f"\n--- In-Distribution Evaluation: {test_data_path} ---")
        test_dataset = PreferenceDataset(test_data_path, tokenizer, max_length=max_length)
        test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size,
                                  collate_fn=collate_fn, num_workers=4)
        id_metrics = evaluate_id(model, model_type, test_loader, args.device)
        results["id_accuracy"] = id_metrics["accuracy"]
        if "brier" in id_metrics:
            results["id_brier"] = id_metrics["brier"]
            results["id_ce_loss"] = id_metrics["ce_loss"]
        print(f"  Accuracy : {id_metrics['accuracy']:.4f}")
        if "brier" in id_metrics:
            print(f"  Brier    : {id_metrics['brier']:.4f}")
            print(f"  CE loss  : {id_metrics['ce_loss']:.4f}")

    # 2. RewardBench OOD evaluation
    rb_path = "data/rewardbench.json"
    if args.eval_scope == "id":
        print("\n--- RewardBench OOD Evaluation: skipped (eval_scope=id) ---")
    elif os.path.exists(rb_path):
        print(f"\n--- RewardBench OOD Evaluation ---")
        rb_dataset = RewardBenchDataset(rb_path, tokenizer, max_length=max_length)
        rb_loader = DataLoader(rb_dataset, batch_size=args.eval_batch_size,
                                collate_fn=rewardbench_collate_fn, num_workers=4)
        rb_results = evaluate_rewardbench(model, model_type, rb_loader, args.device)
        results["rewardbench"] = rb_results
        for k, v in rb_results.items():
            print(f"  {k}: {v:.4f}")

    # 3. PPE-P evaluation
    ppe_p_path = "data/ppe_p_test.json"
    if args.eval_scope == "id":
        print("\n--- PPE-P: skipped (eval_scope=id) ---")
    elif os.path.exists(ppe_p_path):
        from src.dataset import PPEPDataset
        print(f"\n--- PPE-P ---")
        ppe_p_dataset = PPEPDataset(ppe_p_path, tokenizer, max_length=max_length)
        ppe_p_loader = DataLoader(ppe_p_dataset, batch_size=args.eval_batch_size,
                                  collate_fn=ppe_p_collate_fn, num_workers=4)
        ppe_p_results = evaluate_ppe_p(model, model_type, ppe_p_loader, args.device)
        results["ppe_p"] = ppe_p_results
        for k, v in ppe_p_results.items():
            print(f"  {k}: {v:.4f}")

    # Save results
    os.makedirs("results", exist_ok=True)
    result_file = os.path.join("results", f"{run_name}_results.json")
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {result_file}")

    return results


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
