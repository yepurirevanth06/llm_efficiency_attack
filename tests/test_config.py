import pytest

from llm_efficiency_attack.config import AttackConfig


def test_defaults_fill_in_when_config_is_empty():
    cfg = AttackConfig.from_dict({})
    assert cfg.max_perturbed_tokens == 3
    assert cfg.objective == "neg_eos_logprob"
    assert cfg.seed == 0


def test_overrides_are_applied():
    cfg = AttackConfig.from_dict({"max_perturbed_tokens": 5, "seed": 42})
    assert cfg.max_perturbed_tokens == 5
    assert cfg.seed == 42
    # untouched fields still default
    assert cfg.num_candidates == 20


def test_unknown_field_raises():
    with pytest.raises(ValueError, match="Unknown config field"):
        AttackConfig.from_dict({"not_a_real_field": 1})


@pytest.mark.parametrize("field", ["max_perturbed_tokens", "num_candidates", "max_iterations"])
def test_non_positive_int_fields_raise(field):
    with pytest.raises(ValueError, match="positive int"):
        AttackConfig.from_dict({field: 0})
    with pytest.raises(ValueError, match="positive int"):
        AttackConfig.from_dict({field: -1})
    with pytest.raises(ValueError, match="positive int"):
        AttackConfig.from_dict({field: 1.5})


def test_invalid_objective_raises():
    with pytest.raises(ValueError, match="objective"):
        AttackConfig.from_dict({"objective": "not_a_real_objective"})


def test_min_cosine_similarity_out_of_range_raises():
    with pytest.raises(ValueError, match="min_cosine_similarity"):
        AttackConfig.from_dict({"min_cosine_similarity": 1.5})


def test_config_not_a_dict_raises():
    with pytest.raises(ValueError, match="must be a dict"):
        AttackConfig.from_dict(["not", "a", "dict"])  # type: ignore[arg-type]


def test_to_dict_roundtrips_json_serializable():
    import json

    cfg = AttackConfig.from_dict({"seed": 7})
    json.dumps(cfg.to_dict())  # must not raise
