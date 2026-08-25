"""Exercises the Seq2SeqLMAdapter path (T5-style encoder-decoder), which is
otherwise only ever tested via the CausalLMAdapter path in
test_attacker_end_to_end.py. This confirms build_adapter's dispatch on
`model.config.is_encoder_decoder` actually picks the right adapter and that
the encoder/decoder gradient + generation plumbing works.
"""

from llm_efficiency_attack import Attacker
from llm_efficiency_attack.adapters import Seq2SeqLMAdapter, build_adapter


def test_build_adapter_picks_seq2seq_for_encoder_decoder_model(tiny_seq2seq_model, tiny_tokenizer):
    adapter = build_adapter(tiny_seq2seq_model, tiny_tokenizer, device="cpu")
    assert isinstance(adapter, Seq2SeqLMAdapter)


def test_attacker_runs_end_to_end_on_seq2seq_model(tiny_seq2seq_model, tiny_tokenizer):
    attack = Attacker(tiny_seq2seq_model, tokenizer=tiny_tokenizer)
    x = "the quick brown fox"

    config = {
        "max_perturbed_tokens": 2,
        "num_candidates": 5,
        "max_iterations": 3,
        "objective_horizon": 3,
        "max_new_tokens": 8,
        "seed": 0,
        "device": "cpu",
    }
    adv_x, logs = attack.run(x, config)

    assert isinstance(adv_x, str)
    assert logs["num_tokens_perturbed"] <= 2
    assert "clean_generated_tokens" in logs
    assert "adv_generated_tokens" in logs
