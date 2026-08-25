"""Reusable cost-metric helper.

Kept separate from :class:`~llm_efficiency_attack.attacker.Attacker` so it
can be unit tested in isolation and reused outside the attack loop (e.g. to
score externally generated adversarial examples).
"""

from __future__ import annotations

from dataclasses import dataclass

from .adapters import ModelAdapter


@dataclass(frozen=True)
class EfficiencyDamage:
    """Comparison between a benign and adversarial input's generation cost."""

    clean_generated_tokens: int
    adv_generated_tokens: int

    @property
    def length_increase(self) -> int:
        return self.adv_generated_tokens - self.clean_generated_tokens

    @property
    def length_increase_ratio(self) -> float:
        if self.clean_generated_tokens == 0:
            return float("nan")
        return self.adv_generated_tokens / self.clean_generated_tokens

    def to_dict(self) -> dict:
        return {
            "clean_generated_tokens": self.clean_generated_tokens,
            "adv_generated_tokens": self.adv_generated_tokens,
            "length_increase": self.length_increase,
            "length_increase_ratio": self.length_increase_ratio,
        }


def measure_efficiency_damage(
    adapter: ModelAdapter, x: str, adv_x: str, max_new_tokens: int
) -> EfficiencyDamage:
    """Run real generation on ``x`` and ``adv_x`` and compare output length.

    Output length (number of decode steps taken before EOS or
    ``max_new_tokens``) is used as the cost proxy, since it's directly
    proportional to wall-clock time and FLOPs for autoregressive decoding
    with a fixed per-step cost.
    """
    clean_ids = adapter.encode(x)
    adv_ids = adapter.encode(adv_x)
    clean_len = adapter.generate_length(clean_ids, max_new_tokens)
    adv_len = adapter.generate_length(adv_ids, max_new_tokens)
    return EfficiencyDamage(clean_generated_tokens=clean_len, adv_generated_tokens=adv_len)
