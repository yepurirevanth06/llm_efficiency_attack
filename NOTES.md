# Implementation Notes

## Approach

I read the NMTSloth paper (FSE '22) before writing any code. The paper's
core method has three stages — find critical tokens via gradient, mutate
the input, detect degradation by measuring real output length — targeting
NMT systems specifically. My job was to generalize that into a package
that works on any Hugging Face model (causal or seq2seq), not just NMT.

Design order: I isolated model-specific logic behind a `ModelAdapter`
interface first (`adapters.py`), then built the search loop (`attacker.py`)
against that interface only, so it never has to know whether it's talking
to GPT-2 or T5. `objective.py` and `metrics.py` are split out separately
so the objective is swappable and the cost measurement is independently
testable, per the task's requirements.

## Where this matches the paper, and where it deliberately doesn't

Full comparison is in the README's "How the attack works" section. Short
version: the central hypothesis (suppress EOS probability → delay
termination → more compute) and the three-stage structure are faithfully
reproduced, including the paper's exact signed importance-score formula
(`g_i = Σ_j ∂f(x)/∂tk_i^j`, Eq. 2). What I deliberately cut for scope:
character-level and structure-level perturbation (token-level only), beam
search (greedy decoding only), and the full-generation-length objective
(I use a short fixed horizon instead, for tractability on arbitrary
models/lengths).

## Problems I ran into, and how I resolved them

**Bug: `retain_grad()` on a non-grad tensor.** Early version of
`ModelAdapter.eos_logprobs` called `.retain_grad()` unconditionally, but
that method is used both for gradient computation *and* plain scoring
(inside the candidate-search loop, which runs under `torch.no_grad()`).
Caused 4 test failures immediately. Fix: removed the stray call from the
scoring path; gradient computation has its own separate leaf-tensor path
in `attacker._score_from_embeds`.

**Correctness gap: unsigned gradient norm instead of the paper's signed
sum.** My first version ranked "critical tokens" by L2 norm of the
gradient, which can't distinguish "this token, if changed, would suppress
EOS" from "would raise EOS probability" — it only measures magnitude, not
direction. Caught this on a close re-read of the paper's Eq. 2 and fixed
it to use the paper's actual formula (signed sum across embedding
dimensions, ranked by absolute value for prioritization, with the real
direction resolved by an exact forward-pass evaluation per candidate).

**Missing dependency: `sentencepiece`.** The quickstart demo originally
used a GPT-2 model, which doesn't need it. When I switched the demo to a
translation model (`Helsinki-NLP/opus-mt-en-de`) for a more meaningful
efficiency-degradation demo, its `MarianTokenizer` requires
`sentencepiece`, which wasn't in `pyproject.toml`'s pinned dependencies.
Added it after the install failed on a clean machine.

**Empirical finding: a bigger search budget can hurt real attack
strength.** Sweeping `max_perturbed_tokens` and `num_candidates` upward
on the same input *decreased* the real length increase (24→25 tokens
became 24→21). This is because the search only sees the differentiable
proxy objective while searching — the real generation length is only
checked after the fact. A larger budget let the search wander further
from what the proxy predicted would work, without any feedback loop to
catch it. This is a genuine limitation, not a config mistake: the proxy
and the ground truth aren't perfectly correlated, and this package reports
the ground truth (`clean_generated_tokens` / `adv_generated_tokens`,
measured with real `.generate()` calls) specifically so that gap is
visible rather than hidden behind an optimistic proxy score.

## What I'd do with more time

In priority order: (1) implement the paper's full objective
(`f(x) = (1/n)Σ(p_eos_i + p_oi_i)`, averaged over the actual generation
trajectory rather than a fixed short horizon) to close the proxy/ground-truth
gap described above; (2) add character-level and structure-level
perturbation as additional candidate-generation strategies alongside the
existing token-level one; (3) add beam search support to the decode
simulation, since the paper found beam size affects attack severity.
