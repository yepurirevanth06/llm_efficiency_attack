from llm_efficiency_attack import Attacker


def _config(**overrides):
    base = {
        "max_perturbed_tokens": 2,
        "num_candidates": 5,
        "max_iterations": 3,
        "objective": "neg_eos_logprob",
        "objective_horizon": 3,
        "max_new_tokens": 8,
        "seed": 0,
        "device": "cpu",
    }
    base.update(overrides)
    return base


def test_run_returns_adv_x_and_logs(tiny_model, tiny_tokenizer):
    attack = Attacker(tiny_model, tokenizer=tiny_tokenizer)
    x = "the weather today is sunny"

    adv_x, logs = attack.run(x, _config())

    assert isinstance(adv_x, str)
    assert isinstance(logs, dict)
    for key in (
        "config",
        "num_tokens_perturbed",
        "perturbed_positions",
        "iterations",
        "clean_generated_tokens",
        "adv_generated_tokens",
        "length_increase",
        "wall_clock_seconds",
    ):
        assert key in logs


def test_num_perturbed_tokens_respects_budget(tiny_model, tiny_tokenizer):
    attack = Attacker(tiny_model, tokenizer=tiny_tokenizer)
    x = "the weather today is sunny and warm"

    _, logs = attack.run(x, _config(max_perturbed_tokens=2))

    assert logs["num_tokens_perturbed"] <= 2
    assert len(logs["perturbed_positions"]) == logs["num_tokens_perturbed"]


def test_same_seed_gives_same_result(tiny_model, tiny_tokenizer):
    attack = Attacker(tiny_model, tokenizer=tiny_tokenizer)
    x = "the weather today is sunny"

    adv_x_1, logs_1 = attack.run(x, _config(seed=123))
    adv_x_2, logs_2 = attack.run(x, _config(seed=123))

    assert adv_x_1 == adv_x_2
    assert logs_1["num_tokens_perturbed"] == logs_2["num_tokens_perturbed"]
    assert logs_1["perturbed_positions"] == logs_2["perturbed_positions"]


def test_config_is_validated_before_running(tiny_model, tiny_tokenizer):
    import pytest

    attack = Attacker(tiny_model, tokenizer=tiny_tokenizer)
    with pytest.raises(ValueError):
        attack.run("the weather today", _config(max_perturbed_tokens=-1))


def test_logs_echo_effective_config(tiny_model, tiny_tokenizer):
    attack = Attacker(tiny_model, tokenizer=tiny_tokenizer)
    _, logs = attack.run("the weather today", _config(seed=99))
    assert logs["config"]["seed"] == 99
