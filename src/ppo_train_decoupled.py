"""PPO training with periodic checkpoints and no online gold-RM eval."""

from __future__ import annotations

import argparse
import json
import os
import random
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
import wandb
from accelerate import PartialState
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainerCallback,
    set_seed,
)

from trl.experimental.ppo import PPOConfig, PPOTrainer

from src.ppo_reward import ProxyRewardWrapper, VALID_REWARD_MODES


def fmt_num(x) -> str:
    if x is None:
        return "none"
    return f"{float(x):g}"


def sanitize_name(x: Optional[str]) -> str:
    if x is None:
        return "none"
    return str(x).replace("/", "_")


def build_rm_tag(
    rm_type: str,
    anchor_method: Optional[str],
    anchor_lambda: Optional[float],
) -> str:
    if rm_type == "bt":
        return "bt"
    if rm_type == "anchor":
        return "anchor"
    if anchor_method is None:
        return "mv"
    return f"mv-{anchor_method}-lam{fmt_num(anchor_lambda)}"


def build_run_name(
    setting: str,
    reward_mode: str,
    policy_tag: str,
    rm_dirname: str,
    lr: float,
    kl: float,
    ppo_epochs: int,
    num_epochs: int,
    seed: int,
    train_temperature: float,
    train_top_p: float,
    rollout_batch_size: int,
) -> str:
    return (
        f"ppo_{setting}_{reward_mode}"
        f"_policy-{sanitize_name(policy_tag)}"
        f"_rm-{sanitize_name(rm_dirname)}"
        f"_lr{fmt_num(lr)}_kl{fmt_num(kl)}_ep{ppo_epochs}_ne{num_epochs}"
        f"_t{fmt_num(train_temperature)}_tp{fmt_num(train_top_p)}"
        f"_rbs{rollout_batch_size}_seed{seed}"
    )


def build_output_dir(
    policy_tag: str,
    rm_dirname: str,
    setting: str,
    reward_mode: str,
    lr: float,
    kl: float,
    ppo_epochs: int,
    num_epochs: int,
    seed: int,
    train_temperature: float,
    train_top_p: float,
    rollout_batch_size: int,
) -> str:
    return (
        f"checkpoints/ppo_decoupled/"
        f"policy-{sanitize_name(policy_tag)}/"
        f"rm-{sanitize_name(rm_dirname)}/"
        f"{setting}_{reward_mode}_lr{fmt_num(lr)}_kl{fmt_num(kl)}_ep{ppo_epochs}_ne{num_epochs}"
        f"_t{fmt_num(train_temperature)}_tp{fmt_num(train_top_p)}"
        f"_rbs{rollout_batch_size}_seed{seed}"
    )


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_local_device() -> torch.device:
    """Pick the current process's CUDA device under accelerate/torchrun."""
    if torch.cuda.is_available():
        if "LOCAL_RANK" in os.environ:
            return torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def is_global_main_process() -> bool:
    rank = os.environ.get("RANK")
    local_rank = os.environ.get("LOCAL_RANK")
    if rank is not None:
        return int(rank) == 0
    if local_rank is not None:
        return int(local_rank) == 0
    return True


class _IdentityBackbone(nn.Module):
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, **kwargs):
        if input_ids is None:
            raise ValueError("_IdentityBackbone requires input_ids")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        carrier = torch.stack(
            [input_ids.to(torch.float32), attention_mask.to(torch.float32)], dim=-1
        )
        return SimpleNamespace(hidden_states=(carrier,))


class ProxyRewardModule(nn.Module):
    base_model_prefix = "shim"

    def __init__(
        self,
        proxy_rm: ProxyRewardWrapper,
        policy_tokenizer,
        proxy_tokenizer,
        rm_max_length: int = 2048,
    ):
        super().__init__()
        self.proxy_rm = proxy_rm
        self.policy_tok = policy_tokenizer
        self.proxy_tok = proxy_tokenizer
        self.rm_max_length = rm_max_length
        self.shim = _IdentityBackbone()

        self._user_sentinel = "__OAI_USER_SENTINEL__"
        self._assistant_sentinel = "__OAI_ASSISTANT_SENTINEL__"
        self._chat_parse_spec = self._build_chat_parse_spec()

        self._is_main = is_global_main_process()
        self._current_global_step = 0
        self._rollout_buffer = {"reward": [], "rm_len": [], "response_len": []}

    def _build_chat_parse_spec(self):
        rendered = self.policy_tok.apply_chat_template(
            [
                {"role": "user", "content": self._user_sentinel},
                {"role": "assistant", "content": self._assistant_sentinel},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        u = rendered.find(self._user_sentinel)
        a = rendered.find(self._assistant_sentinel)
        if u == -1 or a == -1 or u >= a:
            return None
        return {
            "prefix": rendered[:u],
            "middle": rendered[u + len(self._user_sentinel):a],
            "suffix": rendered[a + len(self._assistant_sentinel):],
        }

    def _extract_prompt_response(self, decoded_text: str):
        spec = self._chat_parse_spec
        if spec is not None:
            prefix = spec["prefix"]
            middle = spec["middle"]
            suffix = spec["suffix"]

            start = 0
            if prefix:
                prefix_pos = decoded_text.find(prefix)
                if prefix_pos != -1:
                    start = prefix_pos + len(prefix)

            middle_pos = decoded_text.find(middle, start) if middle else -1
            if middle_pos != -1:
                prompt = decoded_text[start:middle_pos]
                response_start = middle_pos + len(middle)
                if suffix:
                    suffix_pos = decoded_text.find(suffix, response_start)
                    if suffix_pos != -1:
                        response = decoded_text[response_start:suffix_pos]
                    else:
                        response = decoded_text[response_start:]
                else:
                    response = decoded_text[response_start:]
                return prompt, response

        clean_text = self.policy_tok.decode(
            self.policy_tok.encode(decoded_text, add_special_tokens=False),
            skip_special_tokens=True,
        )
        return "", clean_text

    def _accumulate_rollout_stats(self, rewards, rm_attention_mask, response_lengths):
        if not self._is_main:
            return
        self._rollout_buffer["reward"].append(rewards.detach().float().cpu())
        self._rollout_buffer["rm_len"].append(
            rm_attention_mask.sum(dim=1).detach().float().cpu()
        )
        self._rollout_buffer["response_len"].append(
            torch.tensor(response_lengths, dtype=torch.float32)
        )

    def flush_rollout_stats(self):
        if not self._is_main or wandb.run is None:
            self._rollout_buffer = {"reward": [], "rm_len": [], "response_len": []}
            return
        if not self._rollout_buffer["reward"]:
            return
        reward = torch.cat(self._rollout_buffer["reward"])
        rm_len = torch.cat(self._rollout_buffer["rm_len"])
        resp_len = torch.cat(self._rollout_buffer["response_len"])
        stats = {
            "train/rollout/reward_mean": reward.mean().item(),
            "train/rollout/reward_std":  reward.std(unbiased=False).item(),
            "train/rollout/reward_min":  reward.min().item(),
            "train/rollout/reward_max":  reward.max().item(),
            "train/rollout/rm_len_mean": rm_len.mean().item(),
            "train/rollout/rm_len_std":  rm_len.std(unbiased=False).item(),
            "train/rollout/rm_len_min":  rm_len.min().item(),
            "train/rollout/rm_len_max":  rm_len.max().item(),
            "train/rollout/response_len_mean": resp_len.mean().item(),
            "train/rollout/response_len_std":  resp_len.std(unbiased=False).item(),
            "train/rollout/response_len_min":  resp_len.min().item(),
            "train/rollout/response_len_max":  resp_len.max().item(),
        }
        try:
            wandb.log(stats, step=self._current_global_step)
        except Exception as e:
            print(f"[warn] wandb.log failed in rollout stats flush: {e}")
        self._rollout_buffer = {"reward": [], "rm_len": [], "response_len": []}

    @torch.no_grad()
    def score(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                f"Expected hidden_states shape (B, S, H), got {tuple(hidden_states.shape)}"
            )
        if hidden_states.size(-1) < 2:
            raise ValueError(
                f"Expected hidden_states last dim >= 2, got {tuple(hidden_states.shape)}"
            )

        policy_input_ids = hidden_states[..., 0].long().detach().cpu()
        policy_attention_mask = hidden_states[..., 1].long().detach().cpu()

        rm_texts = []
        response_lengths = []
        last_valid_indices = []
        for input_ids_row, attention_mask_row in zip(policy_input_ids, policy_attention_mask):
            valid_ids = input_ids_row[attention_mask_row.bool()]
            if valid_ids.numel() == 0:
                valid_ids = input_ids_row[-1:].clone()

            decoded = self.policy_tok.decode(valid_ids.tolist(), skip_special_tokens=False)
            prompt_text, response_text = self._extract_prompt_response(decoded)

            if prompt_text:
                rm_text = self.proxy_tok.apply_chat_template(
                    [
                        {"role": "user", "content": prompt_text},
                        {"role": "assistant", "content": response_text},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            else:
                rm_text = response_text

            rm_texts.append(rm_text)

            resp_ids = self.proxy_tok.encode(response_text, add_special_tokens=False)
            response_lengths.append(len(resp_ids))

            nz = attention_mask_row.nonzero(as_tuple=False)
            last_valid_indices.append(int(nz[-1].item()) if nz.numel() > 0 else input_ids_row.numel() - 1)

        enc = self.proxy_tok(
            rm_texts,
            padding=True,
            truncation=True,
            max_length=self.rm_max_length,
            return_tensors="pt",
        )
        rm_input_ids = enc["input_ids"].to(self.proxy_rm.device)
        rm_attention_mask = enc["attention_mask"].to(self.proxy_rm.device)

        rewards = self.proxy_rm.score_for_ppo(rm_input_ids, rm_attention_mask)
        rewards = rewards.detach().to(hidden_states.device, dtype=torch.float32)

        self._accumulate_rollout_stats(
            rewards=rewards,
            rm_attention_mask=rm_attention_mask,
            response_lengths=response_lengths,
        )

        batch_size, seq_len, _ = hidden_states.shape
        token_rewards = torch.zeros(batch_size, seq_len, 1, device=hidden_states.device, dtype=torch.float32)
        for i, last_idx in enumerate(last_valid_indices):
            token_rewards[i, last_idx, 0] = rewards[i]
        return token_rewards


class PPODebugLogCallback(TrainerCallback):
    def __init__(self):
        self._is_main = is_global_main_process()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self._is_main or not logs:
            return control

        keys = [
            "objective/kl",
            "objective/entropy",
            "objective/scores",
            "objective/non_score_reward",
            "policy/clipfrac_avg",
            "loss/policy_avg",
            "loss/value_avg",
            "lr",
            "eps",
        ]
        msg = []
        for k in keys:
            if k in logs and isinstance(logs[k], (int, float)):
                msg.append(f"{k}={logs[k]:.4f}")
        if msg:
            print(f"[PPO DEBUG][trainer log][step={state.global_step}] " + " | ".join(msg))
        return control


class RewardModuleStepSyncCallback(TrainerCallback):
    def __init__(self, reward_module):
        self.m = reward_module

    def on_step_begin(self, args, state, control, **kwargs):
        self.m._current_global_step = int(state.global_step)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        self.m._current_global_step = int(state.global_step)
        self.m.flush_rollout_stats()
        return control


def unwrap_policy_model(model, accelerator=None):
    if accelerator is not None:
        try:
            model = accelerator.unwrap_model(model)
        except Exception:
            pass
    for attr in ("policy", "pretrained_model", "model"):
        candidate = getattr(model, attr, None)
        if candidate is not None and hasattr(candidate, "save_pretrained"):
            return candidate
    return model


def save_policy_checkpoint(model, tokenizer, output_dir: str, accelerator=None) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = unwrap_policy_model(model, accelerator=accelerator)
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


class PeriodicPolicySaveCallback(TrainerCallback):
    def __init__(self, output_root: str, tokenizer, every: int):
        self.output_root = output_root
        self.tokenizer = tokenizer
        self.every = int(every)
        self.saved = []

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.every <= 0 or not state.global_step:
            return control
        if state.global_step % self.every != 0:
            return control
        if not is_global_main_process():
            return control
        if model is None:
            return control
        checkpoint_dir = os.path.join(
            self.output_root, "checkpoints", f"step-{int(state.global_step)}"
        )
        save_policy_checkpoint(
            model, self.tokenizer, checkpoint_dir, accelerator=getattr(self, "accelerator", None)
        )
        self.saved.append(
            {"kind": "periodic", "step": int(state.global_step), "path": checkpoint_dir}
        )
        print(f"[checkpoint] saved periodic PPO policy -> {checkpoint_dir}")
        return control


def build_train_dataset(ppo_prompts: list, policy_tokenizer, max_prompt_length: int) -> Dataset:
    rows = []
    for item in ppo_prompts:
        prompt_text = policy_tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )

        enc = policy_tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_prompt_length,
            padding=False,
            return_attention_mask=True,
        )

        rows.append(
            {
                "query": prompt_text,
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
            }
        )

    ds = Dataset.from_list(rows)
    ds = ds.with_format("python")
    return ds


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--proxy_rm_path", type=str, required=True)
    p.add_argument("--proxy_rm_type", type=str, required=True, choices=["bt", "meanvar", "anchor"])
    p.add_argument(
        "--reward_mode",
        type=str,
        required=True,
        choices=list(VALID_REWARD_MODES),
    )
    p.add_argument("--setting", type=str, required=True, choices=["id", "ood"])
    p.add_argument("--ppo_prompts", type=str, required=True)
    p.add_argument("--eval_prompts", type=str, required=True)
    p.add_argument("--gold_rm_model", type=str, default=None)
    p.add_argument("--output_root", type=str, default=None)
    p.add_argument("--save_every_steps", type=int, default=30)
    p.add_argument("--anchor_method", type=str, default=None, choices=[None, "1anchor", "2anchor"])
    p.add_argument("--anchor_lambda", type=float, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--kl_coef", type=float, default=None)
    p.add_argument("--ppo_epochs", type=int, default=None)
    p.add_argument("--num_epochs", type=int, default=None)
    p.add_argument("--train_temperature", type=float, default=None)
    p.add_argument("--train_top_p", type=float, default=None)
    p.add_argument("--rollout_batch_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--policy_short", type=str, default=None)
    p.add_argument("--rm_short", type=str, default=None)
    p.add_argument("--rm_tokenizer", type=str, default=None)
    p.add_argument("--debug", action="store_true", default=False)
    p.add_argument("--debug_print_limit", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    lr = args.learning_rate if args.learning_rate is not None else float(cfg["learning_rate"])
    kl = args.kl_coef if args.kl_coef is not None else float(cfg["kl_coef"])
    ppo_epochs = args.ppo_epochs if args.ppo_epochs is not None else int(cfg["ppo_epochs"])
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    num_epochs = args.num_epochs if args.num_epochs is not None else int(cfg["num_epochs"])
    train_temperature = (
        args.train_temperature if args.train_temperature is not None else float(cfg["train_temperature"])
    )
    train_top_p = (
        args.train_top_p if args.train_top_p is not None else float(cfg.get("train_top_p", 1.0))
    )
    rollout_batch_size = (
        args.rollout_batch_size if args.rollout_batch_size is not None else int(cfg["rollout_batch_size"])
    )
    gold_rm_model = args.gold_rm_model or cfg.get("gold_rm_model")
    lr_scheduler_type = cfg.get("lr_scheduler_type", "cosine")
    warmup_ratio = cfg.get("warmup_ratio", None)
    warmup_steps = cfg.get("warmup_steps", None)
    if warmup_ratio is not None and warmup_steps is not None:
        raise ValueError("Specify only one of warmup_ratio or warmup_steps, not both.")

    seed_everything(seed)

    policy_model_name = cfg["policy_model"]
    policy_tag = args.policy_short or os.path.basename(policy_model_name.rstrip("/"))
    rm_dirname = os.path.basename(args.proxy_rm_path.rstrip("/"))
    rm_tag = build_rm_tag(args.proxy_rm_type, args.anchor_method, args.anchor_lambda)
    run_name = build_run_name(
        args.setting, args.reward_mode, policy_tag, rm_dirname,
        lr, kl, ppo_epochs, num_epochs, seed, train_temperature, train_top_p, rollout_batch_size,
    )
    output_root = args.output_root or build_output_dir(
        policy_tag, rm_dirname, args.setting, args.reward_mode,
        lr, kl, ppo_epochs, num_epochs, seed, train_temperature, train_top_p, rollout_batch_size,
    )
    final_dir = os.path.join(output_root, "final")
    manifest_path = os.path.join(output_root, "manifest.json")

    os.environ["WANDB_PROJECT"] = cfg["wandb_project"]

    pstate = PartialState()
    local_device = get_local_device()
    if local_device.type == "cuda":
        torch.cuda.set_device(local_device)
    print(
        f"[rank={pstate.process_index}] local_device={local_device} "
        f"current_device={torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'}"
    )

    policy_tokenizer = AutoTokenizer.from_pretrained(policy_model_name)
    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token
    policy_tokenizer.padding_side = "left"

    policy_model = AutoModelForCausalLM.from_pretrained(
        policy_model_name,
        dtype=torch.bfloat16,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        policy_model_name,
        dtype=torch.bfloat16,
    )

    for m in (policy_model, ref_model):
        if getattr(m, "generation_config", None) is not None:
            m.generation_config.max_length = None
            m.generation_config.max_new_tokens = int(cfg["max_new_tokens"])

    value_model = AutoModelForSequenceClassification.from_pretrained(
        policy_model_name,
        num_labels=1,
        dtype=torch.bfloat16,
    )

    proxy_rm = ProxyRewardWrapper(
        checkpoint_dir=args.proxy_rm_path,
        rm_type=args.proxy_rm_type,
        reward_mode=args.reward_mode,
        device=str(local_device),
    )

    rm_tokenizer_source = (
        args.rm_tokenizer
        or getattr(proxy_rm, "base_model_name", None)
        or args.proxy_rm_path
    )
    proxy_tokenizer = AutoTokenizer.from_pretrained(rm_tokenizer_source)
    if proxy_tokenizer.pad_token is None:
        proxy_tokenizer.pad_token = proxy_tokenizer.eos_token
    proxy_tokenizer.padding_side = "right"

    reward_model = ProxyRewardModule(
        proxy_rm=proxy_rm,
        policy_tokenizer=policy_tokenizer,
        proxy_tokenizer=proxy_tokenizer,
        rm_max_length=int(cfg.get("rm_max_length", 2048)),
    )

    ppo_prompts = load_json(args.ppo_prompts)
    train_dataset = build_train_dataset(
        ppo_prompts=ppo_prompts,
        policy_tokenizer=policy_tokenizer,
        max_prompt_length=int(cfg["max_prompt_length"]),
    )

    world_size = pstate.num_processes
    rollout_bs_global = int(rollout_batch_size)
    if rollout_bs_global % world_size != 0:
        raise ValueError(
            f"rollout_batch_size ({rollout_bs_global}) must be divisible by world_size ({world_size})."
        )

    per_rank_bs = rollout_bs_global // world_size
    mini_bs = int(cfg["mini_batch_size"])
    if per_rank_bs % mini_bs != 0:
        raise ValueError(
            f"per-rank rollout size ({per_rank_bs}) must be divisible by mini_batch_size ({mini_bs})."
        )

    num_mini_batches = per_rank_bs // mini_bs
    total_episodes = num_epochs * len(ppo_prompts)

    ppo_config = PPOConfig(
        output_dir=output_root,
        run_name=run_name,
        seed=seed,
        learning_rate=lr,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=float(warmup_ratio) if warmup_ratio is not None else 0.0,
        warmup_steps=int(warmup_steps) if warmup_steps is not None else 0,
        per_device_train_batch_size=mini_bs,
        gradient_accumulation_steps=1,
        num_mini_batches=num_mini_batches,
        num_ppo_epochs=ppo_epochs,
        total_episodes=total_episodes,
        response_length=int(cfg["max_new_tokens"]),
        stop_token_id=policy_tokenizer.eos_token_id,
        temperature=float(train_temperature),
        # top_p=float(train_top_p),
        kl_coef=kl,
        cliprange=float(cfg["clip_range"]),
        cliprange_value=float(cfg["value_clip_range"]),
        gamma=float(cfg["gamma"]),
        lam=float(cfg["lam"]),
        report_to="wandb",
        save_strategy="no",
        eval_strategy="no",
    )

    save_callback = PeriodicPolicySaveCallback(
        output_root=output_root,
        tokenizer=policy_tokenizer,
        every=int(args.save_every_steps),
    )
    callbacks = [
        save_callback,
        PPODebugLogCallback(),
        RewardModuleStepSyncCallback(reward_model),
    ]

    def data_collator(features):
        token_features = [
            {
                "input_ids": f["input_ids"],
                "attention_mask": f["attention_mask"],
            }
            for f in features
        ]
        batch = policy_tokenizer.pad(
            token_features,
            padding=True,
            return_tensors="pt",
        )
        batch["query"] = [f["query"] for f in features]
        return batch

    trainer = PPOTrainer(
        args=ppo_config,
        processing_class=policy_tokenizer,
        model=policy_model,
        ref_model=ref_model,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    try:
        save_callback.accelerator = trainer.accelerator
    except Exception:
        pass

    is_main = True
    try:
        is_main = trainer.accelerator.is_main_process
    except Exception:
        pass

    wb_rm_type = {
        "bt": "bt",
        "anchor": "anchor",
        "mv": "mv",
        "mv-1anchor": "mv-1anchor",
        "mv-2anchor": "mv-2anchor",
    }.get(rm_tag.split("-lam")[0], rm_tag.split("-lam")[0])

    if is_main and wandb.run is not None:
        try:
            wandb.run.name = run_name
            wandb.run.tags = (
                args.setting,
                wb_rm_type,
                args.reward_mode,
                sanitize_name(policy_tag),
                sanitize_name(args.rm_short or rm_dirname),
                "decoupled",
            )
            wandb.config.update(
                {
                    "setting": args.setting,
                    "proxy_rm_type": wb_rm_type,
                    "proxy_rm_path": args.proxy_rm_path,
                    "proxy_rm_dirname": rm_dirname,
                    "proxy_rm_base_model": getattr(proxy_rm, "base_model_name", None),
                    "anchor_method": args.anchor_method,
                    "anchor_lambda": args.anchor_lambda,
                    "reward_mode": args.reward_mode,
                    "quantile": (
                        None
                        if args.reward_mode == "random"
                        else int(args.reward_mode[1:]) / 100.0
                    ),
                    "kl_coef": kl,
                    "ppo_epochs": ppo_epochs,
                    "learning_rate": lr,
                    "lr_scheduler_type": lr_scheduler_type,
                    "warmup_ratio": float(warmup_ratio) if warmup_ratio is not None else None,
                    "warmup_steps": int(warmup_steps) if warmup_steps is not None else None,
                    "train_temperature": train_temperature,
                    "train_top_p": train_top_p,
                    "rollout_batch_size": rollout_batch_size,
                    "policy_model": policy_model_name,
                    "policy_short": policy_tag,
                    "rm_short": args.rm_short,
                    "rm_tokenizer": rm_tokenizer_source,
                    "gold_rm": gold_rm_model,
                    "output_root": output_root,
                    "save_every_steps": int(args.save_every_steps),
                    "seed": seed,
                },
                allow_val_change=True,
            )
        except Exception as e:
            print(f"[warn] W&B metadata update failed: {e}")

    trainer.train()

    if is_main:
        save_policy_checkpoint(
            trainer.model,
            policy_tokenizer,
            final_dir,
            accelerator=getattr(trainer, "accelerator", None),
        )
        checkpoints = list(save_callback.saved)
        checkpoints.append({"kind": "final", "step": int(trainer.state.global_step), "path": final_dir})

        manifest = {
            "run_name": run_name,
            "output_root": output_root,
            "manifest_path": manifest_path,
            "setting": args.setting,
            "rm_tag": rm_tag,
            "policy_model": policy_model_name,
            "policy_short": policy_tag,
            "rm_short": args.rm_short,
            "proxy_rm_path": args.proxy_rm_path,
            "proxy_rm_dirname": rm_dirname,
            "proxy_rm_base_model": getattr(proxy_rm, "base_model_name", None),
            "proxy_rm_type": args.proxy_rm_type,
            "rm_tokenizer": rm_tokenizer_source,
            "gold_rm": gold_rm_model,
            "reward_mode": args.reward_mode,
            "anchor_method": args.anchor_method,
            "anchor_lambda": args.anchor_lambda,
            "learning_rate": lr,
            "lr_scheduler_type": lr_scheduler_type,
            "warmup_ratio": float(warmup_ratio) if warmup_ratio is not None else None,
            "warmup_steps": int(warmup_steps) if warmup_steps is not None else None,
            "kl_coef": kl,
            "ppo_epochs": ppo_epochs,
            "num_epochs": num_epochs,
            "train_temperature": train_temperature,
            "train_top_p": train_top_p,
            "rollout_batch_size": rollout_batch_size,
            "seed": seed,
            "ppo_prompts": args.ppo_prompts,
            "eval_prompts": args.eval_prompts,
            "max_prompt_length": int(cfg["max_prompt_length"]),
            "max_new_tokens": int(cfg["max_new_tokens"]),
            "eval_batch_size": int(cfg["eval_batch_size"]),
            "save_every_steps": int(args.save_every_steps),
            "checkpoints": checkpoints,
        }
        os.makedirs(output_root, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        with open(os.path.join(final_dir, "ppo_run.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"[manifest] wrote {manifest_path}")

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
