"""Periodic evaluation during PPO training.

Aligned with RM training semantics:
- Responses are extracted per-sample using each prompt's true (unpadded) length.
- RM scoring rebuilds chat-formatted (prompt, response) inputs with
  apply_chat_template(..., add_generation_prompt=False), matching RM training.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import wandb


def _chunks(seq: List, size: int) -> Iterable[List]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


class PPOEvaluator:
    """Periodic rollout + proxy/gold scoring for PPO training.

    Eval is data-parallel across all ranks: each rank greedily generates
    responses for its slice of ``eval_prompts`` and scores them with the
    proxy and gold RMs locally. Scores are then all-gathered so the main
    process can compute aggregate metrics and log them once to W&B.
    """

    def __init__(
        self,
        eval_prompts: list,
        proxy_rm,                     # ProxyRewardWrapper
        proxy_tokenizer,
        gold_rm,                      # GoldRewardScorer
        gold_tokenizer,
        policy_tokenizer,
        accelerator=None,             # accelerate.Accelerator; used for gather + rank info
        max_prompt_length: int = 1536,
        max_new_tokens: int = 512,
        batch_size: int = 16,
        rm_max_length: int = 2048,
        device: str = "cuda",
        ref_model=None,
        debug_enabled: bool = False,
        debug_print_limit: int = 2,
    ):
        self.eval_prompts = eval_prompts
        self.proxy_rm = proxy_rm
        self.proxy_tokenizer = proxy_tokenizer
        self.gold_rm = gold_rm
        self.gold_tokenizer = gold_tokenizer
        self.policy_tokenizer = policy_tokenizer
        self.accelerator = accelerator
        self.max_prompt_length = max_prompt_length
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.rm_max_length = rm_max_length
        self.device = device
        self.ref_model = ref_model
        self.debug_enabled = debug_enabled
        self.debug_print_limit = debug_print_limit

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------

    def _score_pairs(self, prompts: List[str], responses: List[str],
                     tokenizer, score_fn, formatter=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize (prompt, response) pairs with dynamic padding and score them.

        This matches RM training semantics:
        [{user: prompt}, {assistant: response}] ->
        apply_chat_template(..., add_generation_prompt=False)

        Returns (scores, rm_lens) both on CPU.
        """
        if formatter is not None:
            texts = formatter(prompts, responses)
        else:
            texts = []
            for p, r in zip(prompts, responses):
                messages = [
                    {"role": "user", "content": p},
                    {"role": "assistant", "content": r},
                ]
                texts.append(tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                ))
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.rm_max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        scores = score_fn(input_ids, attention_mask)  # (batch,)
        rm_lens = attention_mask.sum(dim=1).detach().float().cpu()
        return scores.detach().float().cpu(), rm_lens

    @torch.no_grad()
    def _gold_score(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.gold_rm.score_input_ids(input_ids=input_ids, attention_mask=attention_mask)

    # ------------------------------------------------------------------
    # Greedy generation
    # ------------------------------------------------------------------

    def _trim_response_ids(self, row: torch.Tensor, tok) -> torch.Tensor:
        """Trim a generated response sequence for text decoding / length.

        Stops at first EOS if present; otherwise strips trailing pad tokens.
        """
        eos_id = tok.eos_token_id
        pad_id = tok.pad_token_id

        if row.numel() == 0:
            return row

        # Stop at the first EOS token, inclusive.
        if eos_id is not None:
            eos_pos = (row == eos_id).nonzero(as_tuple=False)
            if eos_pos.numel() > 0:
                return row[: int(eos_pos[0].item()) + 1]

        # Otherwise strip trailing pads only.
        if pad_id is not None:
            non_pad = (row != pad_id).nonzero(as_tuple=False)
            if non_pad.numel() == 0:
                return row[:0]
            return row[: int(non_pad[-1].item()) + 1]

        return row

    def _generate_greedy(
        self, policy_model, prompts: List[str]
    ) -> Tuple[List[str], List[int], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate greedy responses for a batch of prompts.

        Returns:
            response_texts, response_token_lengths,
            full_ids (B, S) — padded prompt prefix + generated continuation,
            full_attention_mask (B, S) — 1 on [true_prompt, response] positions,
            response_mask (B, S) — 1 on response positions only.

        Important:
        For left-padded batches, each sample's true prompt length is given by
        attention_mask.sum(). We slice generated tokens per-sample using that
        length rather than the batch-padded width.
        """
        tok = self.policy_tokenizer
        saved_padding_side = tok.padding_side
        tok.padding_side = "left"
        try:
            prompt_texts = [
                tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in prompts
            ]
            enc = tok(
                prompt_texts,
                padding=True,
                truncation=True,
                max_length=self.max_prompt_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            gen = policy_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,  # ignored when do_sample=False
                top_p=1.0,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

            response_texts: List[str] = []
            response_lengths: List[int] = []

            input_width = input_ids.shape[1]
            full_ids = gen  # (B, input_width + gen_len)
            full_attention_mask = torch.zeros_like(full_ids, dtype=torch.long)
            response_mask = torch.zeros_like(full_ids, dtype=torch.long)

            true_prompt_lens = attention_mask.sum(dim=1).tolist()
            for i, row in enumerate(gen):
                prompt_len_i = int(true_prompt_lens[i])

                # gen includes the full padded input row prefix. For left padding,
                # the actual prompt occupies the rightmost prompt_len_i tokens of
                # the input prefix, so the generated continuation starts after the
                # full input width used by generate(), not after prompt_len_i.
                #
                # However, the continuation region is always the suffix beyond the
                # original input width for every row in the batch.
                continuation = row[input_width:]

                # Fallback: if a backend ever returns shorter-than-expected rows,
                # recover by slicing relative to the sample's true prompt length.
                if continuation.numel() == 0 and row.numel() > prompt_len_i:
                    continuation = row[prompt_len_i:]

                trimmed = self._trim_response_ids(continuation, tok)
                resp_len = int(trimmed.numel())
                response_texts.append(tok.decode(trimmed, skip_special_tokens=True))
                response_lengths.append(resp_len)

                prompt_start = input_width - prompt_len_i
                full_attention_mask[i, prompt_start:input_width + resp_len] = 1
                response_mask[i, input_width:input_width + resp_len] = 1
        finally:
            tok.padding_side = saved_padding_side

        return response_texts, response_lengths, full_ids, full_attention_mask, response_mask

    @torch.no_grad()
    def _response_logprob_sum(
        self, model, full_ids: torch.Tensor,
        attention_mask: torch.Tensor, response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-sample summed log-prob of response tokens under `model`.

        logits[:, t, :] predicts the token at position t+1, so the log-prob
        of a response token at position r is taken from logits[:, r-1, :].
        Computed via gather + logsumexp to avoid materializing the full
        (B, S, V) softmax tensor in fp32.
        """
        model_device = next(model.parameters()).device
        ids = full_ids.to(model_device)
        mask = attention_mask.to(model_device)
        logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1, :]
        targets = ids[:, 1:]
        gathered = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1).float()
        log_z = torch.logsumexp(logits, dim=-1).float()
        target_logp = gathered - log_z
        resp_target_mask = response_mask.to(model_device)[:, 1:].float()
        return (target_logp * resp_target_mask).sum(dim=1).detach().cpu()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def _rank_slice(self) -> List:
        """Split eval_prompts contiguously across ranks."""
        if self.accelerator is None:
            return list(self.eval_prompts)
        rank = self.accelerator.process_index
        world = self.accelerator.num_processes
        n = len(self.eval_prompts)
        step = (n + world - 1) // world
        start = rank * step
        end = min(n, start + step)
        return list(self.eval_prompts[start:end])

    def _gather_cat(self, arr: np.ndarray) -> np.ndarray:
        """All-gather a 1-D numpy array across ranks -> concatenated array."""
        if self.accelerator is None or self.accelerator.num_processes == 1:
            return arr

        local = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
        local_len = torch.tensor([local.numel()], device=self.device)
        all_lens = self.accelerator.gather(local_len).detach().cpu().numpy().tolist()
        max_len = int(max(all_lens))

        pad = torch.full((max_len - local.numel(),), float("nan"),
                         dtype=torch.float32, device=self.device)
        padded = torch.cat([local, pad], dim=0)
        gathered = self.accelerator.gather(padded).detach().cpu().numpy()
        return gathered[~np.isnan(gathered)]

    @torch.no_grad()
    def evaluate(self, policy_model, global_step: int) -> Optional[dict]:
        was_training = policy_model.training
        policy_model.eval()
        ref_was_training = None
        if self.ref_model is not None:
            ref_was_training = self.ref_model.training
            self.ref_model.eval()
        try:
            local_prompts = self._rank_slice()

            local_proxy, local_gold, local_lens, local_rm_lens = [], [], [], []
            local_kls: List[torch.Tensor] = []
            cached_prompts: List[str] = []
            cached_responses: List[str] = []
            for batch in _chunks(local_prompts, self.batch_size):
                prompts = [b["prompt"] for b in batch]
                responses, lengths, full_ids, full_attn, resp_mask = \
                    self._generate_greedy(policy_model, prompts)
                local_lens.extend(lengths)

                if self.debug_enabled and not cached_prompts:
                    k = min(self.debug_print_limit, len(prompts))
                    cached_prompts = list(prompts[:k])
                    cached_responses = list(responses[:k])

                proxy_scores, proxy_rm_lens = self._score_pairs(
                    prompts, responses, self.proxy_tokenizer,
                    self.proxy_rm.score_for_eval,
                )
                gold_scores, _ = self._score_pairs(
                    prompts, responses, self.gold_tokenizer,
                    self._gold_score,
                    formatter=self.gold_rm.format_pairs,
                )
                local_proxy.append(proxy_scores)
                local_gold.append(gold_scores)
                local_rm_lens.append(proxy_rm_lens)

                if self.ref_model is not None:
                    policy_logp = self._response_logprob_sum(
                        policy_model, full_ids, full_attn, resp_mask
                    )
                    ref_logp = self._response_logprob_sum(
                        self.ref_model, full_ids, full_attn, resp_mask
                    )
                    local_kls.append(policy_logp - ref_logp)

            local_proxy_np = (torch.cat(local_proxy).numpy() if local_proxy
                              else np.zeros((0,), dtype=np.float32))
            local_gold_np = (torch.cat(local_gold).numpy() if local_gold
                             else np.zeros((0,), dtype=np.float32))
            local_lens_np = np.asarray(local_lens, dtype=np.float32)
            local_rm_lens_np = (torch.cat(local_rm_lens).numpy() if local_rm_lens
                                else np.zeros((0,), dtype=np.float32))
            local_kls_np = (torch.cat(local_kls).numpy() if local_kls
                            else np.zeros((0,), dtype=np.float32))

            proxy_scores = self._gather_cat(local_proxy_np)
            gold_scores = self._gather_cat(local_gold_np)
            lens = self._gather_cat(local_lens_np)
            rm_lens = self._gather_cat(local_rm_lens_np)
            kls = self._gather_cat(local_kls_np)

            is_main = (self.accelerator is None
                       or self.accelerator.is_main_process)
            if not is_main:
                return None

            if proxy_scores.size == 0:
                proxy_scores = np.array([0.0])
            if gold_scores.size == 0:
                gold_scores = np.array([0.0])
            if lens.size == 0:
                lens = np.array([0.0])
            if rm_lens.size == 0:
                rm_lens = np.array([0.0])

            metrics = {
                "eval/proxy_score_mean":     float(proxy_scores.mean()),
                "eval/proxy_score_std":      float(proxy_scores.std()),
                "eval/proxy_score_min":      float(proxy_scores.min()),
                "eval/proxy_score_max":      float(proxy_scores.max()),
                "eval/gold_score_mean":      float(gold_scores.mean()),
                "eval/gold_score_std":       float(gold_scores.std()),
                "eval/proxy_gold_gap":       float(proxy_scores.mean() - gold_scores.mean()),
                "eval/response_length_mean": float(lens.mean()),
                "eval/response_length_std":  float(lens.std()),
                "eval/response_length_min":  float(lens.min()),
                "eval/response_length_max":  float(lens.max()),
                "eval/rm_len_mean":          float(rm_lens.mean()),
                "eval/rm_len_std":           float(rm_lens.std()),
                "eval/rm_len_min":           float(rm_lens.min()),
                "eval/rm_len_max":           float(rm_lens.max()),
            }
            if self.ref_model is not None and kls.size > 0:
                metrics["eval/kl_to_ref"] = float(kls.mean())

            if self.debug_enabled and cached_prompts:
                print("=" * 120)
                print(
                    f"[EVAL DEBUG step={global_step}] "
                    f"printing {len(cached_prompts)} sample(s)"
                )
                for i in range(len(cached_prompts)):
                    print("-" * 120)
                    print(f"[prompt] {cached_prompts[i][:300]}")
                    print(f"[response] {cached_responses[i][:500]}")
                print("=" * 120)

            if wandb.run is not None:
                wandb.log(metrics, step=global_step)
            return metrics
        finally:
            if was_training:
                policy_model.train()
            if self.ref_model is not None and ref_was_training:
                self.ref_model.train()
