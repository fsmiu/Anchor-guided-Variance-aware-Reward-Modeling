"""Training script for BT and Mean-Var reward models using TRL."""

import argparse
import json
import os
import random
import re
import sys

import numpy as np
import torch
import yaml
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)
from trl import RewardTrainer, RewardConfig
from torch.distributions import Normal

from src.model import create_reward_model
from src.dataset import PreferenceDataset
from src.loss import bt_loss, mean_var_loss, anchor_1_loss, anchor_2_loss


class PreTokenizedRewardTrainer(RewardTrainer):
    """Base class that skips TRL's dataset preparation for pre-tokenized data."""

    def _prepare_dataset(self, dataset, *args, **kwargs):
        return dataset

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to bridge custom compute_loss outputs to HF Trainer eval loop."""
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        r_w = outputs["r_w"]
        r_l = outputs["r_l"]
        # logits = torch.cat([r_w, r_l], dim=-1)  # (batch, 2)
        logits = torch.stack([r_w, r_l], dim=-1)  # (batch, 2)
        labels = torch.ones(r_w.size(0), device=r_w.device)  # dummy
        return loss.detach(), logits.detach(), labels.detach()


class BTRewardTrainer(PreTokenizedRewardTrainer):
    """RewardTrainer with standard Bradley-Terry loss."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids_chosen = inputs["input_ids_chosen"]
        attention_mask_chosen = inputs["attention_mask_chosen"]
        input_ids_rejected = inputs["input_ids_rejected"]
        attention_mask_rejected = inputs["attention_mask_rejected"]
        soft_label = inputs.get("soft_label")

        r_w = model(input_ids=input_ids_chosen, attention_mask=attention_mask_chosen)
        r_l = model(input_ids=input_ids_rejected, attention_mask=attention_mask_rejected)

        loss = bt_loss(r_w, r_l, soft_label=soft_label)

        if self.state.is_world_process_zero:
            self.log({
                "rewards/chosen": r_w.detach().mean().item(),
                "rewards/rejected": r_l.detach().mean().item(),
                "rewards/accuracies": (r_w > r_l).float().detach().mean().item(),
                "rewards/margins": (r_w - r_l).detach().mean().item(),
            })

        if return_outputs:
            return loss, {"r_w": r_w, "r_l": r_l}
        return loss


class MeanVarRewardTrainer(PreTokenizedRewardTrainer):
    """RewardTrainer with Thurstonian/Gaussian loss for Mean-Var model."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids_chosen = inputs["input_ids_chosen"]
        attention_mask_chosen = inputs["attention_mask_chosen"]
        input_ids_rejected = inputs["input_ids_rejected"]
        attention_mask_rejected = inputs["attention_mask_rejected"]
        soft_label = inputs.get("soft_label")

        r_w, var_w = model(input_ids=input_ids_chosen, attention_mask=attention_mask_chosen)
        r_l, var_l = model(input_ids=input_ids_rejected, attention_mask=attention_mask_rejected)

        loss = mean_var_loss(r_w, r_l, var_w, var_l, soft_label=soft_label)

        if self.state.is_world_process_zero:
            self.log({
                "rewards/chosen": r_w.detach().mean().item(),
                "rewards/rejected": r_l.detach().mean().item(),
                "rewards/accuracies": (r_w > r_l).float().detach().mean().item(),
                "rewards/margins": (r_w - r_l).detach().mean().item(),
                "rewards/var_chosen": var_w.detach().mean().item(),
                "rewards/var_rejected": var_l.detach().mean().item(),
            })

        if return_outputs:
            return loss, {"r_w": r_w, "r_l": r_l, "var_w": var_w, "var_l": var_l}
        return loss


def anchor_binary_accuracy(r_hat, var, r_tilde, tau, sample_weight=None):
    r_hat = r_hat.float()
    var = var.float().clamp(min=1e-6, max=1e6)
    std = torch.sqrt(var)
    r_tilde = r_tilde.float()

    normal = Normal(
        torch.tensor(0.0, device=r_hat.device, dtype=torch.float32),
        torch.tensor(1.0, device=r_hat.device, dtype=torch.float32),
    )

    z = ((r_hat - float(tau)) / std).clamp(min=-6.0, max=6.0)
    q = normal.cdf(z)

    pred_anchor = (q >= 0.5).float()
    true_anchor = (r_tilde >= float(tau)).float()
    correct = (pred_anchor == true_anchor).float()
    if sample_weight is None:
        return correct.mean()

    weight = sample_weight.to(device=correct.device, dtype=correct.dtype)
    denom = weight.sum()
    denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    return (correct * weight).sum() / denom


class AnchorMeanVarRewardTrainer(PreTokenizedRewardTrainer):
    """Gaussian reward trainer with anchor loss from an external RM.

    Supports two anchor methods:
      - "1anchor": single binary threshold (BCE on CDF at tau)
      - "2anchor": two binary thresholds (BCE on CDF at tau1, tau2)

    Total loss: L_pref + anchor_lambda * L_anchor
    """

    def __init__(self, anchor_method, anchor_lambda=0.1,
                 anchor_tau=0.0, anchor_tau1=0.0, anchor_tau2=0.0, **kwargs):
        super().__init__(**kwargs)
        self.anchor_method = anchor_method
        self.anchor_lambda = anchor_lambda
        self.anchor_tau = anchor_tau
        self.anchor_tau1 = anchor_tau1
        self.anchor_tau2 = anchor_tau2

    # def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    #     input_ids_chosen = inputs["input_ids_chosen"]
    #     attention_mask_chosen = inputs["attention_mask_chosen"]
    #     input_ids_rejected = inputs["input_ids_rejected"]
    #     attention_mask_rejected = inputs["attention_mask_rejected"]
    #     soft_label = inputs.get("soft_label")

    #     r_w, var_w = model(input_ids=input_ids_chosen, attention_mask=attention_mask_chosen)
    #     r_l, var_l = model(input_ids=input_ids_rejected, attention_mask=attention_mask_rejected)

    #     # Preference loss (same as MeanVarRewardTrainer)
    #     loss_pref = mean_var_loss(r_w, r_l, var_w, var_l, soft_label=soft_label)

    #     # Anchor loss (applied to each response independently)
    #     chosen_score = inputs["chosen_score"]   # (batch, 1)
    #     rejected_score = inputs["rejected_score"]  # (batch, 1)

    #     if self.anchor_method == "1anchor":
    #         loss_anchor = (anchor_1_loss(r_w, var_w, chosen_score, self.anchor_tau) +
    #                        anchor_1_loss(r_l, var_l, rejected_score, self.anchor_tau))
    #     elif self.anchor_method == "2anchor":
    #         loss_anchor = (anchor_2_loss(r_w, var_w, chosen_score, self.anchor_tau1, self.anchor_tau2) +
    #                        anchor_2_loss(r_l, var_l, rejected_score, self.anchor_tau1, self.anchor_tau2))
    #     else:
    #         raise ValueError(f"Unknown anchor_method: {self.anchor_method}")

    #     loss = loss_pref + self.anchor_lambda * loss_anchor

    #     if self.state.is_world_process_zero:
    #         self.log({
    #             "rewards/chosen": r_w.detach().mean().item(),
    #             "rewards/rejected": r_l.detach().mean().item(),
    #             "rewards/accuracies": (r_w > r_l).float().detach().mean().item(),
    #             "rewards/margins": (r_w - r_l).detach().mean().item(),
    #             "rewards/var_chosen": var_w.detach().mean().item(),
    #             "rewards/var_rejected": var_l.detach().mean().item(),
    #             "loss/pref": loss_pref.detach().item(),
    #             "loss/anchor": loss_anchor.detach().item(),
    #         })

    #     if return_outputs:
    #         return loss, {"r_w": r_w, "r_l": r_l, "var_w": var_w, "var_l": var_l}
    #     return loss
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids_chosen = inputs["input_ids_chosen"]
        attention_mask_chosen = inputs["attention_mask_chosen"]
        input_ids_rejected = inputs["input_ids_rejected"]
        attention_mask_rejected = inputs["attention_mask_rejected"]
        soft_label = inputs.get("soft_label")

        r_w, var_w = model(
            input_ids=input_ids_chosen,
            attention_mask=attention_mask_chosen,
        )
        r_l, var_l = model(
            input_ids=input_ids_rejected,
            attention_mask=attention_mask_rejected,
        )

        loss_pref = mean_var_loss(r_w, r_l, var_w, var_l, soft_label=soft_label)

        if "chosen_score" not in inputs or "rejected_score" not in inputs:
            raise ValueError("Anchor training requires chosen_score and rejected_score in inputs.")

        chosen_score = inputs["chosen_score"]      # (batch,)
        rejected_score = inputs["rejected_score"]  # (batch,)
        anchor_weight = inputs.get("anchor_weight")
        if anchor_weight is not None:
            anchor_weight = anchor_weight.to(device=r_w.device, dtype=torch.float32)

        log_dict = {
            "rewards/chosen": r_w.detach().mean().item(),
            "rewards/rejected": r_l.detach().mean().item(),
            "rewards/accuracies": (r_w > r_l).float().detach().mean().item(),
            "rewards/margins": (r_w - r_l).detach().mean().item(),
            "rewards/var_chosen": var_w.detach().mean().item(),
            "rewards/var_rejected": var_l.detach().mean().item(),
            "loss/pref": loss_pref.detach().item(),
        }

        if self.anchor_method == "1anchor":
            loss_anchor = (
                anchor_1_loss(r_w, var_w, chosen_score, self.anchor_tau, sample_weight=anchor_weight) +
                anchor_1_loss(r_l, var_l, rejected_score, self.anchor_tau, sample_weight=anchor_weight)
            )
            # loss_anchor = (
            #     anchor_1_loss(r_w, var_w.detach(), chosen_score, self.anchor_tau) +
            #     anchor_1_loss(r_l, var_l.detach(), rejected_score, self.anchor_tau)
            # )

            acc_chosen = anchor_binary_accuracy(r_w, var_w, chosen_score, self.anchor_tau, sample_weight=anchor_weight)
            acc_rejected = anchor_binary_accuracy(r_l, var_l, rejected_score, self.anchor_tau, sample_weight=anchor_weight)
            acc_anchor = 0.5 * (acc_chosen + acc_rejected)

            log_dict.update({
                "anchor/accuracy": acc_anchor.detach().item(),
                "anchor/accuracy_chosen": acc_chosen.detach().item(),
                "anchor/accuracy_rejected": acc_rejected.detach().item(),
            })

        elif self.anchor_method == "2anchor":
            loss_anchor = (
                anchor_2_loss(r_w, var_w, chosen_score, self.anchor_tau1, self.anchor_tau2,
                              sample_weight=anchor_weight) +
                anchor_2_loss(r_l, var_l, rejected_score, self.anchor_tau1, self.anchor_tau2,
                              sample_weight=anchor_weight)
            )
            # loss_anchor = (
            #         anchor_2_loss(r_w, var_w.detach(), chosen_score, self.anchor_tau1, self.anchor_tau2) +
            #         anchor_2_loss(r_l, var_l.detach(), rejected_score, self.anchor_tau1, self.anchor_tau2)
            #     )

            acc_tau1_chosen = anchor_binary_accuracy(r_w, var_w, chosen_score, self.anchor_tau1,
                                                     sample_weight=anchor_weight)
            acc_tau1_rejected = anchor_binary_accuracy(r_l, var_l, rejected_score, self.anchor_tau1,
                                                       sample_weight=anchor_weight)
            acc_tau2_chosen = anchor_binary_accuracy(r_w, var_w, chosen_score, self.anchor_tau2,
                                                     sample_weight=anchor_weight)
            acc_tau2_rejected = anchor_binary_accuracy(r_l, var_l, rejected_score, self.anchor_tau2,
                                                       sample_weight=anchor_weight)

            acc_tau1 = 0.5 * (acc_tau1_chosen + acc_tau1_rejected)
            acc_tau2 = 0.5 * (acc_tau2_chosen + acc_tau2_rejected)
            acc_anchor = 0.5 * (acc_tau1 + acc_tau2)

            log_dict.update({
                "anchor/accuracy": acc_anchor.detach().item(),
                "anchor/accuracy_tau1": acc_tau1.detach().item(),
                "anchor/accuracy_tau2": acc_tau2.detach().item(),
                "anchor/accuracy_tau1_chosen": acc_tau1_chosen.detach().item(),
                "anchor/accuracy_tau1_rejected": acc_tau1_rejected.detach().item(),
                "anchor/accuracy_tau2_chosen": acc_tau2_chosen.detach().item(),
                "anchor/accuracy_tau2_rejected": acc_tau2_rejected.detach().item(),
            })

        else:
            raise ValueError(f"Unknown anchor_method: {self.anchor_method}")

        loss = loss_pref + self.anchor_lambda * loss_anchor
        log_dict["loss/anchor"] = loss_anchor.detach().item()
        if anchor_weight is not None:
            log_dict["anchor/batch_weight_sum"] = anchor_weight.detach().sum().item()

        if self.state.is_world_process_zero:
            self.log(log_dict)

        if return_outputs:
            return loss, {"r_w": r_w, "r_l": r_l, "var_w": var_w, "var_l": var_l}
        return loss
    


def compute_anchor_thresholds(data_path, anchor_model=None):
    """Compute anchor thresholds from external RM scores in the training data.

    Returns (median, p25, p75) of all chosen_score and rejected_score values.
    """
    with open(data_path) as f:
        data = json.load(f)
    scores = []
    for d in data:
        # Read from nested anchor_scores if anchor_model is specified
        if anchor_model and "anchor_scores" in d:
            s = d["anchor_scores"].get(anchor_model, {})
            if "chosen_score" in s:
                scores.append(s["chosen_score"])
            if "rejected_score" in s:
                scores.append(s["rejected_score"])
        else:
            if "chosen_score" in d:
                scores.append(d["chosen_score"])
            if "rejected_score" in d:
                scores.append(d["rejected_score"])
    scores = np.array(scores)
    median = float(np.median(scores))
    p25 = float(np.percentile(scores, 25))
    p75 = float(np.percentile(scores, 75))
    print(f"  Anchor thresholds — median: {median:.4f}, p25: {p25:.4f}, p75: {p75:.4f}")
    return median, p25, p75

def inspect_anchor_label_distribution(data_path, tau1=None, tau2=None, anchor_model=None):
    with open(data_path) as f:
        data = json.load(f)

    chosen_scores = []
    rejected_scores = []

    for d in data:
        if anchor_model and "anchor_scores" in d:
            s = d["anchor_scores"].get(anchor_model, {})
            if "chosen_score" in s:
                chosen_scores.append(float(s["chosen_score"]))
            if "rejected_score" in s:
                rejected_scores.append(float(s["rejected_score"]))
        else:
            if "chosen_score" in d:
                chosen_scores.append(float(d["chosen_score"]))
            if "rejected_score" in d:
                rejected_scores.append(float(d["rejected_score"]))

    chosen_scores = np.array(chosen_scores, dtype=np.float32)
    rejected_scores = np.array(rejected_scores, dtype=np.float32)

    chosen_scores = chosen_scores[np.isfinite(chosen_scores)]
    rejected_scores = rejected_scores[np.isfinite(rejected_scores)]

    print("  --- Anchor score distribution ---")
    print(f"  chosen count   : {len(chosen_scores)}")
    print(f"  rejected count : {len(rejected_scores)}")
    if len(chosen_scores) > 0:
        print(f"  chosen min/max : {chosen_scores.min():.4f} / {chosen_scores.max():.4f}")
    if len(rejected_scores) > 0:
        print(f"  rejected min/max : {rejected_scores.min():.4f} / {rejected_scores.max():.4f}")

    if tau1 is not None:
        print(f"  P(chosen >= tau1={tau1:.4f})   = {(chosen_scores >= tau1).mean():.4f}")
        print(f"  P(rejected >= tau1={tau1:.4f}) = {(rejected_scores >= tau1).mean():.4f}")

    if tau2 is not None:
        print(f"  P(chosen >= tau2={tau2:.4f})   = {(chosen_scores >= tau2).mean():.4f}")
        print(f"  P(rejected >= tau2={tau2:.4f}) = {(rejected_scores >= tau2).mean():.4f}")


class PreferenceCollator:
    """Data collator that passes through pre-tokenized preference pairs."""

    def __call__(self, features):
        batch = {
            "input_ids_chosen": torch.stack([f["input_ids_chosen"] for f in features]),
            "attention_mask_chosen": torch.stack([f["attention_mask_chosen"] for f in features]),
            "input_ids_rejected": torch.stack([f["input_ids_rejected"] for f in features]),
            "attention_mask_rejected": torch.stack([f["attention_mask_rejected"] for f in features]),
        }
        if "soft_label" in features[0]:
            batch["soft_label"] = torch.stack([f["soft_label"] for f in features])
        if "chosen_score" in features[0]:
            batch["chosen_score"] = torch.stack([f["chosen_score"] for f in features])
        if "rejected_score" in features[0]:
            batch["rejected_score"] = torch.stack([f["rejected_score"] for f in features])
        if "anchor_weight" in features[0]:
            batch["anchor_weight"] = torch.stack([f["anchor_weight"] for f in features])
        return batch


def accuracy_metrics(eval_pred):
    """Compute preference accuracy from eval predictions."""
    logits, labels = eval_pred
    r_w = logits[:, 0]
    r_l = logits[:, 1]
    accuracy = (r_w > r_l).astype(float).mean().item()
    return {"accuracy": accuracy}


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def format_anchor_percent(percent):
    pct = float(percent)
    if pct.is_integer():
        return str(int(pct))
    return str(pct).replace(".", "p")


def main():
    parser = argparse.ArgumentParser(description="Train reward model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--model_name", type=str, default=None, help="Override model name from config")
    parser.add_argument("--anchor_model", type=str, default=None, help="Override anchor model key from config")
    parser.add_argument("--anchor_lambda", type=float, default=None, help="Override anchor_lambda from config")
    parser.add_argument("--anchor_sample_percent", type=float, default=None,
                        help="Percentage of train samples that participate in anchor loss")
    parser.add_argument("--anchor_sample_seed", type=int, default=None,
                        help="Seed for deterministic anchor-sample selection")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning_rate from config")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    seed = args.seed
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    config = load_config(args.config)

    # Extract config values (CLI overrides config)
    model_name = args.model_name or config["model_name"]
    model_type = config["model_type"]  # "bt" or "meanvar"
    train_data_path = config["train_data"]
    max_length = config.get("max_length", 2048)
    run_name = config.get("run_name", "reward_model")
    anchor_model = args.anchor_model or config.get("anchor_model")
    anchor_sample_percent = (
        args.anchor_sample_percent
        if args.anchor_sample_percent is not None
        else config.get("anchor_sample_percent", 100.0)
    )
    anchor_sample_seed = (
        args.anchor_sample_seed
        if args.anchor_sample_seed is not None
        else config.get("anchor_sample_seed", 0)
    )

    if float(anchor_sample_percent) != 100.0:
        run_name = f"{run_name}_anchorpct{format_anchor_percent(anchor_sample_percent)}"

    # Auto-adjust run_name when model is overridden
    if args.model_name:
        model_short = args.model_name.split("/")[-1].lower()
        run_name = f"{run_name}_{model_short}"
    # Include anchor model identifier and lambda so different anchors don't collide
    if anchor_model:
        anchor_short = anchor_model.split("/")[-1].lower()
        anchor_lambda_val = args.anchor_lambda if args.anchor_lambda is not None else config.get("anchor_lambda", 0.1)
        run_name = f"{run_name}_anchor-{anchor_short}_lam{anchor_lambda_val}"
    # Include learning rate in run_name when overridden so different LRs don't collide
    if args.learning_rate is not None:
        run_name = f"{run_name}_lr{args.learning_rate}"
    # Add seed to run name so wandb names and checkpoint dirs differ across seeds
    run_name = f"{run_name}_seed{seed}"
    output_dir = f"checkpoints/{run_name}"

    print(f"=== Training {model_type.upper()} Reward Model ===")
    print(f"  Model: {model_name}")
    print(f"  Data: {train_data_path}")
    print(f"  Output: {output_dir}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load datasets
    print("Loading dataset...")
    train_dataset = PreferenceDataset(train_data_path, tokenizer, max_length=max_length,
                                       anchor_model=anchor_model,
                                       anchor_sample_percent=anchor_sample_percent,
                                       anchor_sample_seed=anchor_sample_seed)
    print(f"  Train samples: {len(train_dataset)}")
    if anchor_model:
        print(f"  Anchor sample percent: {float(anchor_sample_percent):g}%")
        print(f"  Anchor sample count: {train_dataset.anchor_sample_count}")
        print(f"  Anchor sample seed: {anchor_sample_seed}")

    eval_dataset = None
    eval_data_path = config.get("val_data")
    if eval_data_path and os.path.exists(eval_data_path):
        eval_strategy = config.get("eval_strategy", "no")
        if eval_strategy != "no":
            eval_dataset = PreferenceDataset(eval_data_path, tokenizer, max_length=max_length,
                                              anchor_model=anchor_model)
            print(f"  Eval samples: {len(eval_dataset)}")

    # Create model
    print("Creating model...")
    model = create_reward_model(model_name, model_type=model_type, torch_dtype=torch.bfloat16)

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable({"use_reentrant": False})

    # Training arguments
    training_args = RewardConfig(
        output_dir=output_dir,
        num_train_epochs=config.get("epochs", 1),
        per_device_train_batch_size=config.get("per_device_batch_size", 16),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        learning_rate=float(args.learning_rate if args.learning_rate is not None else config.get("learning_rate", 2e-6)),
        lr_scheduler_type=config.get("lr_scheduler", "cosine"),
        warmup_ratio=config.get("warmup_ratio", 0.05),
        weight_decay=float(config.get("weight_decay", 1e-4)),
        bf16=True,
        logging_steps=config.get("logging_steps", 50),
        eval_strategy=config.get("eval_strategy", "no"),
        eval_steps=config.get("eval_steps", None),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size",
                                              config.get("per_device_batch_size", 16)),
        save_strategy=config.get("eval_strategy", "epoch"),
        save_steps=config.get("eval_steps", None),
        save_total_limit=2,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="accuracy" if eval_dataset else None,
        greater_is_better=True if eval_dataset else None,
        report_to=config.get("report_to", "wandb"),
        run_name=run_name,
        seed=seed,
        dataloader_num_workers=4,
        deepspeed=config.get("deepspeed", None),
        gradient_checkpointing=True,
        remove_unused_columns=False,
        max_length=max_length,
    )

    # Select trainer class and create trainer
    anchor_method = config.get("anchor_method", None)
    resolved_anchor_tau = None
    resolved_anchor_tau1 = None
    resolved_anchor_tau2 = None

    common_trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=accuracy_metrics if eval_dataset else None,
        processing_class=tokenizer,
        data_collator=PreferenceCollator(),
    )

    if model_type == "bt":
        trainer = BTRewardTrainer(**common_trainer_kwargs)
    elif model_type == "meanvar" and anchor_method is None:
        trainer = MeanVarRewardTrainer(**common_trainer_kwargs)
    elif model_type == "meanvar" and anchor_method in ("1anchor", "2anchor"):
        # Resolve anchor thresholds
        anchor_lambda = float(args.anchor_lambda if args.anchor_lambda is not None else config.get("anchor_lambda", 0.1))
        anchor_tau = config.get("anchor_tau", 0.0)
        anchor_tau1 = config.get("anchor_tau1", 0.0)
        anchor_tau2 = config.get("anchor_tau2", 0.0)

        # Auto-compute thresholds from training data if set to "auto"
        if anchor_tau == "auto" or anchor_tau1 == "auto" or anchor_tau2 == "auto":
            median, p25, p75 = compute_anchor_thresholds(train_data_path, anchor_model=anchor_model)
            if anchor_tau == "auto":
                anchor_tau = median
            if anchor_tau1 == "auto":
                anchor_tau1 = p25
            if anchor_tau2 == "auto":
                anchor_tau2 = p75

        anchor_tau = float(anchor_tau)
        anchor_tau1 = float(anchor_tau1)
        anchor_tau2 = float(anchor_tau2)
        resolved_anchor_tau = anchor_tau
        resolved_anchor_tau1 = anchor_tau1
        resolved_anchor_tau2 = anchor_tau2

        inspect_anchor_label_distribution(
                train_data_path,
                tau1=anchor_tau1 if anchor_method == "2anchor" else anchor_tau,
                tau2=anchor_tau2 if anchor_method == "2anchor" else None,
                anchor_model=anchor_model,
            )


        print(f"  Anchor method: {anchor_method}")
        print(f"  Anchor lambda: {anchor_lambda}")
        if anchor_method == "1anchor":
            print(f"  Anchor tau: {anchor_tau}")
        elif anchor_method == "2anchor":
            print(f"  Anchor tau1: {anchor_tau1}, tau2: {anchor_tau2}")

        trainer = AnchorMeanVarRewardTrainer(
            anchor_method=anchor_method,
            anchor_lambda=anchor_lambda,
            anchor_tau=anchor_tau,
            anchor_tau1=anchor_tau1,
            anchor_tau2=anchor_tau2,
            **common_trainer_kwargs,
        )
    else:
        raise ValueError(f"Unknown model_type/anchor_method: {model_type}/{anchor_method}")

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Extract checkpoint step info from trainer state
    checkpoint_step = trainer.state.global_step
    best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
    if best_ckpt:
        m = re.search(r"checkpoint-(\d+)$", best_ckpt)
        if m:
            checkpoint_step = int(m.group(1))

    # Save config (with any CLI overrides applied)
    save_config = dict(config)
    save_config["model_name"] = model_name
    if anchor_model:
        save_config["anchor_model"] = anchor_model
    if args.anchor_lambda is not None:
        save_config["anchor_lambda"] = float(args.anchor_lambda)
    if args.learning_rate is not None:
        save_config["learning_rate"] = float(args.learning_rate)
    save_config["run_name"] = run_name
    save_config["output_dir"] = output_dir
    save_config["checkpoint_step"] = checkpoint_step
    save_config["anchor_sample_percent"] = float(anchor_sample_percent)
    save_config["anchor_sample_count"] = train_dataset.anchor_sample_count
    save_config["anchor_sample_seed"] = int(anchor_sample_seed)
    if resolved_anchor_tau is not None:
        save_config["resolved_anchor_tau"] = resolved_anchor_tau
    if resolved_anchor_tau1 is not None:
        save_config["resolved_anchor_tau1"] = resolved_anchor_tau1
    if resolved_anchor_tau2 is not None:
        save_config["resolved_anchor_tau2"] = resolved_anchor_tau2
    with open(os.path.join(output_dir, "train_config.json"), "w") as f:
        json.dump(save_config, f, indent=2)

    print("Training complete!")


if __name__ == "__main__":
    main()
