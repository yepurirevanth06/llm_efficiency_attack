"""Attack objectives.

An objective takes an adapter + input and returns a scalar "cost" score that
the search tries to *increase*. Kept as a small registry so a new objective
can be added without touching the attacker's search loop (requirement:
"keep the attack objective swappable").
"""

from __future__ import annotations

from typing import Callable

import torch

from .adapters import ModelAdapter

ObjectiveFn = Callable[[ModelAdapter, torch.Tensor, int], torch.Tensor]


def neg_eos_logprob(adapter: ModelAdapter, input_ids: torch.Tensor, horizon: int) -> torch.Tensor:
    """Sum of ``-log P(EOS)`` over the first ``horizon`` decode steps.

    This is the differentiable proxy for "the model keeps generating
    instead of stopping" used by NMTSloth/LLMEffiChecker: driving EOS
    probability down at every step delays termination, which increases the
    number of decode steps the model actually runs.
    """
    eos_logprobs = adapter.eos_logprobs(input_ids, horizon)
    return -eos_logprobs.sum()


_REGISTRY: dict[str, ObjectiveFn] = {
    "neg_eos_logprob": neg_eos_logprob,
}


def get_objective(name: str) -> ObjectiveFn:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown objective {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]
