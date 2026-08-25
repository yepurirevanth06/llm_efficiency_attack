"""Quickstart: run the efficiency attack on a small real Hugging Face model.

Usage:
    pip install -e .
    python examples/quickstart.py

Model choice matters for this demo. Open-ended causal LMs (GPT-2 style)
don't have a well-defined "done" point for arbitrary prompts, so they often
run to `max_new_tokens` regardless of whether they're under attack -- there's
no early stopping to suppress. Translation/seq2seq models do have a
well-defined completion point (translation is finished), so a clean input
terminates quickly and the attack has real room to show a length increase.
This is also why the original NMTSloth paper evaluated exclusively on NMT
systems rather than open-ended causal LMs. We use one of the paper's own
three eval targets here for that reason.
"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from llm_efficiency_attack import Attacker

MODEL_NAME = "Helsinki-NLP/opus-mt-en-de"  # English->German NMT model (small, real, pretrained)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    x = "The weather today is sunny and warm, and I plan to go for a walk in the park this afternoon."
    config = {
        "max_perturbed_tokens": 3,
        "num_candidates": 30,
        "max_iterations": 6,
        "objective": "neg_eos_logprob",
        "objective_horizon": 10,
        "max_new_tokens": 64,
        "seed": 0,
        "device": "cpu",
    }

    # --- the exact snippet from the task -----------------------------------
    attack = Attacker(model)
    adv_x, logs = attack.run(x, config)
    # -------------------------------------------------------------------------

    print(f"clean input : {x!r}")
    print(f"adv input   : {adv_x!r}")
    print(f"clean generated tokens : {logs['clean_generated_tokens']}")
    print(f"adv generated tokens   : {logs['adv_generated_tokens']}")
    print(f"length increase        : {logs['length_increase']} "
          f"({logs['length_increase_ratio']:.2f}x)")
    print(f"tokens perturbed       : {logs['num_tokens_perturbed']}")

    if logs["length_increase"] <= 0:
        print(
            "\nNote: no length increase this run. Try increasing "
            "'max_perturbed_tokens', 'num_candidates', or trying a "
            "different seed/input -- the search is greedy and not "
            "guaranteed to find an improving swap on every input."
        )


if __name__ == "__main__":
    main()
