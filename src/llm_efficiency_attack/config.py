"""Configuration schema, validation, and defaults for the efficiency attack.

The attack is entirely config-driven: every hyperparameter that controls the
search (how many tokens may be perturbed, how many candidate replacements to
consider, how many optimization iterations to run, etc.) lives in a single
JSON-serializable dict. ``AttackConfig`` validates that dict once, up front,
and fails loudly with a clear message on bad input rather than failing deep
inside the optimization loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


# Fields with (type, default). `None` default means "required unless a
# default is documented below".
_DEFAULTS: dict[str, Any] = {
    # --- search budget ---------------------------------------------------
    "max_perturbed_tokens": 3,       # how many input tokens may be replaced
    "num_candidates": 20,            # candidate replacement tokens per position
    "max_iterations": 10,            # outer greedy-search rounds
    # --- objective ---------------------------------------------------------
    "objective": "neg_eos_logprob",  # which cost-increasing objective to use
    "objective_horizon": 8,          # number of decode steps to score the objective over
    # --- decoding used for evaluation / logging --------------------------
    "max_new_tokens": 64,            # cap so evaluation on a slowed-down input terminates
    # --- reproducibility & runtime ----------------------------------------
    "seed": 0,
    "device": "cpu",
    # --- misc --------------------------------------------------------------
    "min_cosine_similarity": 0.0,    # optional embedding-space similarity floor
                                       # for candidate replacements (0 disables it)
}

_VALID_OBJECTIVES = {"neg_eos_logprob"}


@dataclass
class AttackConfig:
    """Validated, immutable view over the raw config dict.

    Parameters
    ----------
    max_perturbed_tokens:
        Maximum number of input tokens the attack is allowed to change.
        Bounds how far ``adv_x`` can drift from ``x`` (the perturbation budget).
    num_candidates:
        For each token chosen for replacement, how many nearest-neighbor
        candidate tokens (in embedding space) to evaluate.
    max_iterations:
        Number of greedy search rounds. Each round picks the single best
        (position, candidate) swap found so far and applies it.
    objective:
        Name of the objective function driving the search. Currently
        ``"neg_eos_logprob"``: the negative log-probability of the
        end-of-sequence token, summed over ``objective_horizon`` decode
        steps. Maximizing this pushes the model away from stopping early.
    objective_horizon:
        Number of future decode steps the objective looks at when scoring
        a candidate input. Larger = more faithful to true generation
        length, but more expensive per candidate evaluation.
    max_new_tokens:
        Cap on generated tokens when actually running `.generate(...)` for
        evaluation/logging, so a successfully-slowed-down input can't hang.
    seed:
        Random seed. The same config + inputs must give the same result.
    device:
        Torch device string, e.g. ``"cpu"`` or ``"cuda"``.
    min_cosine_similarity:
        If > 0, candidate replacement tokens whose embedding cosine
        similarity to the original token is below this threshold are
        discarded (keeps substitutions "close" to the original word).
    """

    max_perturbed_tokens: int
    num_candidates: int
    max_iterations: int
    objective: str
    objective_horizon: int
    max_new_tokens: int
    seed: int
    device: str
    min_cosine_similarity: float

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "AttackConfig":
        """Validate ``raw`` against the schema and fill in defaults.

        Raises
        ------
        ValueError
            If ``raw`` contains an unknown key, a value of the wrong type,
            or a value outside its valid range.
        """
        if not isinstance(raw, dict):
            raise ValueError(f"config must be a dict, got {type(raw).__name__}")

        unknown = set(raw) - set(_DEFAULTS)
        if unknown:
            raise ValueError(
                f"Unknown config field(s): {sorted(unknown)}. "
                f"Valid fields are: {sorted(_DEFAULTS)}"
            )

        merged = {**_DEFAULTS, **raw}

        def _require_positive_int(name: str) -> int:
            value = merged[name]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"config['{name}'] must be a positive int, got {value!r}")
            return value

        max_perturbed_tokens = _require_positive_int("max_perturbed_tokens")
        num_candidates = _require_positive_int("num_candidates")
        max_iterations = _require_positive_int("max_iterations")
        objective_horizon = _require_positive_int("objective_horizon")
        max_new_tokens = _require_positive_int("max_new_tokens")

        seed = merged["seed"]
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(f"config['seed'] must be an int, got {seed!r}")

        device = merged["device"]
        if not isinstance(device, str) or not device:
            raise ValueError(f"config['device'] must be a non-empty string, got {device!r}")

        objective = merged["objective"]
        if objective not in _VALID_OBJECTIVES:
            raise ValueError(
                f"config['objective'] must be one of {sorted(_VALID_OBJECTIVES)}, got {objective!r}"
            )

        min_cosine_similarity = merged["min_cosine_similarity"]
        if not isinstance(min_cosine_similarity, (int, float)) or isinstance(
            min_cosine_similarity, bool
        ):
            raise ValueError(
                f"config['min_cosine_similarity'] must be a number, got {min_cosine_similarity!r}"
            )
        if not (0.0 <= float(min_cosine_similarity) <= 1.0):
            raise ValueError("config['min_cosine_similarity'] must be within [0, 1]")

        return AttackConfig(
            max_perturbed_tokens=max_perturbed_tokens,
            num_candidates=num_candidates,
            max_iterations=max_iterations,
            objective=objective,
            objective_horizon=objective_horizon,
            max_new_tokens=max_new_tokens,
            seed=seed,
            device=device,
            min_cosine_similarity=float(min_cosine_similarity),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict (used to echo the effective config in logs)."""
        return asdict(self)
