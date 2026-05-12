"""Preference dataset for reward model training and evaluation."""

import json
import random

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class PreferenceDataset(Dataset):
    """Dataset for preference pairs (prompt, chosen, rejected)."""

    def __init__(self, data_path, tokenizer, max_length=2048, anchor_model=None,
                 anchor_sample_percent=None, anchor_sample_seed=0):
        """
        Args:
            data_path: Path to JSON file with list of {prompt, chosen, rejected}.
            tokenizer: HuggingFace tokenizer.
            max_length: Max sequence length for tokenization.
            anchor_model: Anchor model key for reading nested anchor_scores (e.g. "skywork-reward-v2-llama-3.1-8b").
            anchor_sample_percent: Percentage of samples that participate in anchor loss.
            anchor_sample_seed: Seed for deterministic anchor-sample selection.
        """
        with open(data_path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.anchor_model = anchor_model
        self.anchor_sample_percent = anchor_sample_percent
        self.anchor_sample_seed = anchor_sample_seed
        self.anchor_sample_indices = None
        self.anchor_sample_count = len(self.data)

        if anchor_sample_percent is not None:
            pct = float(anchor_sample_percent)
            if pct < 0.0 or pct > 100.0:
                raise ValueError(f"anchor_sample_percent must be in [0, 100], got {pct}")

            n = len(self.data)
            count = int(round(n * pct / 100.0))
            count = max(0, min(n, count))

            indices = list(range(n))
            rng = random.Random(int(anchor_sample_seed))
            rng.shuffle(indices)
            self.anchor_sample_indices = set(indices[:count])
            self.anchor_sample_count = count

        self.column_names = [
            "input_ids_chosen", "attention_mask_chosen",
            "input_ids_rejected", "attention_mask_rejected",
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        prompt = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        # Build chat-format inputs
        chosen_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen},
        ]
        rejected_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected},
        ]

        # Apply chat template
        chosen_text = self.tokenizer.apply_chat_template(
            chosen_messages, tokenize=False, add_generation_prompt=False
        )
        rejected_text = self.tokenizer.apply_chat_template(
            rejected_messages, tokenize=False, add_generation_prompt=False
        )

        # Tokenize
        chosen_enc = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        soft_label = sample.get("soft_label", 1.0)

        item = {
            "input_ids_chosen": chosen_enc["input_ids"].squeeze(0),
            "attention_mask_chosen": chosen_enc["attention_mask"].squeeze(0),
            "input_ids_rejected": rejected_enc["input_ids"].squeeze(0),
            "attention_mask_rejected": rejected_enc["attention_mask"].squeeze(0),
            "soft_label": torch.tensor(soft_label, dtype=torch.float32),
        }

        # Anchor scores from external RM (e.g. Skywork-Reward-V2)
        if self.anchor_model and "anchor_scores" in sample:
            scores = sample["anchor_scores"].get(self.anchor_model, {})
            if "chosen_score" in scores:
                item["chosen_score"] = torch.tensor(scores["chosen_score"], dtype=torch.float32)
            if "rejected_score" in scores:
                item["rejected_score"] = torch.tensor(scores["rejected_score"], dtype=torch.float32)
        elif "chosen_score" in sample:
            item["chosen_score"] = torch.tensor(sample["chosen_score"], dtype=torch.float32)
            if "rejected_score" in sample:
                item["rejected_score"] = torch.tensor(sample["rejected_score"], dtype=torch.float32)

        if self.anchor_sample_indices is not None:
            item["anchor_weight"] = torch.tensor(
                1.0 if idx in self.anchor_sample_indices else 0.0,
                dtype=torch.float32,
            )

        return item


class RewardBenchDataset(Dataset):
    """Dataset for RewardBench evaluation."""

    def __init__(self, data_path, tokenizer, max_length=2048):
        with open(data_path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        prompt = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        chosen_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen},
        ]
        rejected_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected},
        ]

        chosen_text = self.tokenizer.apply_chat_template(
            chosen_messages, tokenize=False, add_generation_prompt=False
        )
        rejected_text = self.tokenizer.apply_chat_template(
            rejected_messages, tokenize=False, add_generation_prompt=False
        )

        chosen_enc = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids_chosen": chosen_enc["input_ids"].squeeze(0),
            "attention_mask_chosen": chosen_enc["attention_mask"].squeeze(0),
            "input_ids_rejected": rejected_enc["input_ids"].squeeze(0),
            "attention_mask_rejected": rejected_enc["attention_mask"].squeeze(0),
            "category": sample.get("category", "unknown"),
            "subset": sample.get("subset", "unknown"),
        }


class PPEPDataset(Dataset):
    """Dataset for PPE-P (PPE Human Preference V1) evaluation.
    Hard label only — no soft_label field.
    Interface identical to RewardBenchDataset.
    """

    def __init__(self, data_path, tokenizer, max_length=2048):
        with open(data_path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        prompt = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        chosen_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen},
        ]
        rejected_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected},
        ]

        chosen_text = self.tokenizer.apply_chat_template(
            chosen_messages, tokenize=False, add_generation_prompt=False
        )
        rejected_text = self.tokenizer.apply_chat_template(
            rejected_messages, tokenize=False, add_generation_prompt=False
        )

        chosen_enc = self.tokenizer(
            chosen_text, max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            rejected_text, max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )

        return {
            "input_ids_chosen": chosen_enc["input_ids"].squeeze(0),
            "attention_mask_chosen": chosen_enc["attention_mask"].squeeze(0),
            "input_ids_rejected": rejected_enc["input_ids"].squeeze(0),
            "attention_mask_rejected": rejected_enc["attention_mask"].squeeze(0),
        }


def collate_fn(batch):
    """Collate function for preference datasets."""
    return {
        "input_ids_chosen": torch.stack([b["input_ids_chosen"] for b in batch]),
        "attention_mask_chosen": torch.stack([b["attention_mask_chosen"] for b in batch]),
        "input_ids_rejected": torch.stack([b["input_ids_rejected"] for b in batch]),
        "attention_mask_rejected": torch.stack([b["attention_mask_rejected"] for b in batch]),
        "soft_label": torch.stack([b["soft_label"] for b in batch]),
    }


def rewardbench_collate_fn(batch):
    """Collate function for RewardBench that preserves category info."""
    return {
        "input_ids_chosen": torch.stack([b["input_ids_chosen"] for b in batch]),
        "attention_mask_chosen": torch.stack([b["attention_mask_chosen"] for b in batch]),
        "input_ids_rejected": torch.stack([b["input_ids_rejected"] for b in batch]),
        "attention_mask_rejected": torch.stack([b["attention_mask_rejected"] for b in batch]),
        "category": [b["category"] for b in batch],
        "subset": [b["subset"] for b in batch],
    }

def ppe_p_collate_fn(batch):
    """Collate function for PPE-P."""
    return {
        "input_ids_chosen": torch.stack([b["input_ids_chosen"] for b in batch]),
        "attention_mask_chosen": torch.stack([b["attention_mask_chosen"] for b in batch]),
        "input_ids_rejected": torch.stack([b["input_ids_rejected"] for b in batch]),
        "attention_mask_rejected": torch.stack([b["attention_mask_rejected"] for b in batch]),
    }


