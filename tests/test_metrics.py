from llm_efficiency_attack.adapters import build_adapter
from llm_efficiency_attack.metrics import EfficiencyDamage, measure_efficiency_damage


def test_efficiency_damage_derived_fields():
    damage = EfficiencyDamage(clean_generated_tokens=4, adv_generated_tokens=10)
    assert damage.length_increase == 6
    assert damage.length_increase_ratio == 2.5


def test_efficiency_damage_handles_zero_clean_length():
    damage = EfficiencyDamage(clean_generated_tokens=0, adv_generated_tokens=5)
    assert damage.length_increase == 5
    import math

    assert math.isnan(damage.length_increase_ratio)


def test_measure_efficiency_damage_runs_on_real_model(tiny_model, tiny_tokenizer):
    adapter = build_adapter(tiny_model, tiny_tokenizer, device="cpu")
    damage = measure_efficiency_damage(
        adapter, x="the weather today", adv_x="the weather today is sunny", max_new_tokens=8
    )
    assert damage.clean_generated_tokens >= 0
    assert damage.adv_generated_tokens >= 0
    assert damage.to_dict()["length_increase"] == (
        damage.adv_generated_tokens - damage.clean_generated_tokens
    )
