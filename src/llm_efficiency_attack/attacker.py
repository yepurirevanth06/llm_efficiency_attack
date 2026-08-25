"""Core attack logic: gradient-guided, budget-constrained token substitution
that increases how many tokens a Hugging Face model generates.

Algorithm (generalized from NMTSloth's three stages -- see README for the
mapping):

1. **Find critical tokens.** Backprop the objective (default:
   ``-log P(EOS)`` summed over a short horizon) into the input embeddings.
   Rank input positions by gradient norm; these are the tokens with the
   most leverage over generation length.
2. **Input mutation.** For each of the top ``max_perturbed_tokens``
   positions (most influential first), evaluate ``num_candidates`` nearest
   neighbors of the current token in embedding space as replacements
   (optionally filtered by a cosine-similarity floor so substitutions stay
   close to the original word), and greedily keep whichever single swap
   increases the objective the most.
3. **Evaluate.** After the search, run real generation on both the
   original and perturbed input and report the resulting length/step
   increase in ``logs`` (the efficiency-degradation-detection stage).

The loop runs for ``max_iterations`` rounds, re-ranking critical tokens each
round so later swaps account for earlier ones.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

import numpy as np
import torch

from .adapters import ModelAdapter, build_adapter
from .config import AttackConfig
from .metrics import measure_efficiency_damage
from .objective import get_objective

logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Attacker:
    """General-purpose white-box efficiency-attack toolbox.

    Parameters
    ----------
    model:
        A Hugging Face ``PreTrainedModel`` (causal or seq2seq LM). The
        specific architecture is handled transparently by an internal
        :class:`~llm_efficiency_attack.adapters.ModelAdapter`; callers never
        need to know or care which one was picked.

    Example
    -------
    >>> from llm_efficiency_attack import Attacker
    >>> attack = Attacker(model)
    >>> adv_x, logs = attack.run(x, config)

    ``tokenizer`` is optional and only needed to override auto-detection
    (e.g. in tests, where a small hand-built model has no Hub path to load
    a matching tokenizer from).
    """

    def __init__(self, model, tokenizer=None):
        self._raw_model = model
        self._raw_tokenizer = tokenizer or self._load_tokenizer(model)

    @staticmethod
    def _load_tokenizer(model):
        name_or_path = getattr(model.config, "_name_or_path", None) or getattr(
            model, "name_or_path", None
        )
        if not name_or_path:
            raise ValueError(
                "Could not auto-detect a tokenizer for this model (no "
                "config._name_or_path). Pass Attacker(model, tokenizer=tokenizer) explicitly."
            )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def run(self, x: str, config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Generate an adversarial input from ``x`` that costs more compute to run.

        Parameters
        ----------
        x:
            A benign text input.
        config:
            JSON-serializable dict of attack hyperparameters. See
            :class:`~llm_efficiency_attack.config.AttackConfig` for the full
            schema.

        Returns
        -------
        (adv_x, logs):
            ``adv_x`` is the perturbed text. ``logs`` is a structured dict
            with per-iteration objective values, the length/cost comparison
            between ``x`` and ``adv_x``, timing, and the effective config.
        """
        cfg = AttackConfig.from_dict(config)
        _set_seed(cfg.seed)

        start_time = time.time()
        adapter = build_adapter(self._raw_model, self._raw_tokenizer, cfg.device)
        objective_fn = get_objective(cfg.objective)

        input_ids = adapter.encode(x)
        seq_len = input_ids.shape[1]
        max_swaps = min(cfg.max_perturbed_tokens, seq_len)

        current_ids = input_ids.clone()
        swapped_positions: set[int] = set()
        iteration_logs: list[dict[str, Any]] = []

        for iteration in range(cfg.max_iterations):
            if len(swapped_positions) >= max_swaps:
                break

            importance = self._token_importance_scores(adapter, current_ids, objective_fn, cfg)
            # Rank candidate positions by |g_i| (paper Eq. 2: g_i is a signed
            # sum of partial derivatives; we rank by magnitude since either
            # sign indicates a token the objective is sensitive to -- the
            # actual swap direction is resolved exactly afterward by
            # _best_swap_at_position's real forward-pass evaluation, not by
            # the sign of g_i itself).
            special_ids = set(adapter.tokenizer.all_special_ids)
            candidate_positions = [
                pos
                for pos in torch.argsort(importance.abs(), descending=True).tolist()
                if pos not in swapped_positions
                and int(current_ids[0, pos]) not in special_ids
            ]
            if not candidate_positions:
                break
            position = candidate_positions[0]

            best_ids, best_objective, best_token_id = self._best_swap_at_position(
                adapter, current_ids, position, objective_fn, cfg
            )
            with torch.no_grad():
                baseline_objective = float(
                    objective_fn(adapter, current_ids, cfg.objective_horizon).item()
                )

            iteration_logs.append(
                {
                    "iteration": iteration,
                    "position": position,
                    "objective_before": baseline_objective,
                    "objective_after": best_objective,
                    "improved": best_objective > baseline_objective,
                }
            )

            if best_objective > baseline_objective:
                current_ids = best_ids
                swapped_positions.add(position)
                logger.info(
                    "iter %d: swapped position %d -> token_id %d (objective %.4f -> %.4f)",
                    iteration,
                    position,
                    best_token_id,
                    baseline_objective,
                    best_objective,
                )
            else:
                logger.info(
                    "iter %d: no improving swap found at position %d, stopping search",
                    iteration,
                    position,
                )
                break

        adv_x = adapter.decode(current_ids)
        damage = measure_efficiency_damage(adapter, x, adv_x, cfg.max_new_tokens)

        logs = {
            "config": cfg.to_dict(),
            "num_tokens_perturbed": len(swapped_positions),
            "perturbed_positions": sorted(swapped_positions),
            "iterations": iteration_logs,
            **damage.to_dict(),
            "wall_clock_seconds": time.time() - start_time,
        }
        return adv_x, logs

    # -- internals -----------------------------------------------------------

    def _token_importance_scores(
        self,
        adapter: ModelAdapter,
        input_ids: torch.Tensor,
        objective_fn,
        cfg: AttackConfig,
    ) -> torch.Tensor:
        """Token importance -- "find critical tokens" (paper Sec. 5.2, Eq. 2).

        Matches the paper's formula g_i = sum_j( d f(x) / d tk_i^j ): a
        *signed* sum of the objective's partial derivatives across each
        token's embedding dimensions. This is deliberately a sum, not an
        L2 norm -- a norm would discard the sign and only tell us "this
        token matters," not "which direction changing it would push the
        objective." Positions are later ranked by |g_i| (see `run`), since
        the exact best replacement token and its effect are resolved by a
        real forward pass in `_best_swap_at_position`, not by the sign of
        g_i alone.
        """
        adapter.model.zero_grad(set_to_none=True)
        embeds = adapter.model.get_input_embeddings()(input_ids).detach().clone()
        embeds.requires_grad_(True)

        score = self._score_from_embeds(adapter, embeds, cfg.objective_horizon)
        score.backward()

        grad = embeds.grad
        if grad is None:
            return torch.zeros(input_ids.shape[1])
        return grad.sum(dim=-1).squeeze(0).detach().cpu()

    def _score_from_embeds(self, adapter: ModelAdapter, embeds: torch.Tensor, horizon: int) -> torch.Tensor:
        """Same objective as `objective.neg_eos_logprob`, but starting from an
        explicit embeddings tensor so we can request its gradient directly.
        """
        from .adapters import CausalLMAdapter, Seq2SeqLMAdapter

        if isinstance(adapter, CausalLMAdapter):
            generated = embeds
            logprobs = []
            for _ in range(horizon):
                out = adapter.model(inputs_embeds=generated)
                next_logits = out.logits[:, -1, :]
                next_logprobs = torch.log_softmax(next_logits, dim=-1)
                logprobs.append(next_logprobs[0, adapter.eos_token_id])
                next_id = next_logits.argmax(dim=-1)
                next_embed = adapter.model.get_input_embeddings()(next_id).unsqueeze(1)
                generated = torch.cat([generated, next_embed], dim=1)
            return -torch.stack(logprobs).sum()

        if isinstance(adapter, Seq2SeqLMAdapter):
            encoder_outputs = adapter.model.get_encoder()(inputs_embeds=embeds)
            decoder_start_id = adapter.model.config.decoder_start_token_id
            if decoder_start_id is None:
                decoder_start_id = adapter.eos_token_id
            decoder_input_ids = torch.tensor([[decoder_start_id]], device=adapter.device)
            logprobs = []
            for _ in range(horizon):
                out = adapter.model(
                    encoder_outputs=encoder_outputs, decoder_input_ids=decoder_input_ids
                )
                next_logits = out.logits[:, -1, :]
                next_logprobs = torch.log_softmax(next_logits, dim=-1)
                logprobs.append(next_logprobs[0, adapter.eos_token_id])
                next_id = next_logits.argmax(dim=-1, keepdim=True)
                decoder_input_ids = torch.cat([decoder_input_ids, next_id], dim=1)
            return -torch.stack(logprobs).sum()

        raise TypeError(f"Unsupported adapter type: {type(adapter)}")

    def _best_swap_at_position(
        self,
        adapter: ModelAdapter,
        input_ids: torch.Tensor,
        position: int,
        objective_fn,
        cfg: AttackConfig,
    ) -> tuple[torch.Tensor, float, int]:
        """Try `num_candidates` embedding-nearest-neighbor replacements at
        `position` and return the one with the highest objective value --
        "input mutation" (stage 2).
        """
        embedding_matrix = adapter.embedding_matrix.detach()
        original_token_id = int(input_ids[0, position])
        original_embed = embedding_matrix[original_token_id]

        sims = torch.nn.functional.cosine_similarity(
            embedding_matrix, original_embed.unsqueeze(0), dim=-1
        )
        if cfg.min_cosine_similarity > 0:
            sims = torch.where(
                sims >= cfg.min_cosine_similarity, sims, torch.full_like(sims, -1.0)
            )
        # Exclude the token itself from its own candidate list.
        sims[original_token_id] = -1.0
        top_candidates = torch.topk(sims, k=min(cfg.num_candidates, sims.shape[0])).indices

        best_ids = input_ids
        best_token_id = original_token_id

        with torch.no_grad():
            best_score = float(objective_fn(adapter, input_ids, cfg.objective_horizon).item())
            for candidate_id in top_candidates.tolist():
                trial_ids = input_ids.clone()
                trial_ids[0, position] = candidate_id
                score = float(objective_fn(adapter, trial_ids, cfg.objective_horizon).item())
                if score > best_score:
                    best_score = score
                    best_ids = trial_ids
                    best_token_id = candidate_id

        return best_ids, best_score, best_token_id
