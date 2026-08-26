"""Minimal white-box efficiency attack -- core algorithm only.
Finds the input token with the most gradient-leverage over EOS
probability, swaps it for a nearby embedding-space token if that delays
EOS, repeats under a token budget."""
import torch

def attack(model, tokenizer, x, config):
    torch.manual_seed(config.get("seed", 0))
    device = config.get("device", "cpu")
    model = model.to(device).eval()
    is_s2s = getattr(model.config, "is_encoder_decoder", False)
    eos_id = tokenizer.eos_token_id
    embed = model.get_input_embeddings()
    horizon = config.get("objective_horizon", 8)
    budget = config.get("max_perturbed_tokens", 3)
    n_cand = config.get("num_candidates", 20)
    cap = config.get("max_new_tokens", 32)

    def neg_eos_logprob(ids):
        # -log P(eos) over `horizon` greedy steps; differentiable outside
        # no_grad(), a cheap forward pass inside it -- serves both the
        # gradient step and candidate scoring below.
        e = embed(ids).detach().clone().requires_grad_(True)
        if is_s2s:
            enc = model.get_encoder()(inputs_embeds=e)
            start = model.config.decoder_start_token_id or eos_id
            dec = torch.tensor([[start]], device=device)
            score = 0.0
            for _ in range(horizon):
                logits = model(encoder_outputs=enc, decoder_input_ids=dec).logits[:, -1]
                logp = torch.log_softmax(logits, -1)
                score = score - logp[0, eos_id]
                dec = torch.cat([dec, logp.argmax(-1, keepdim=True)], 1)
        else:
            gen, score = e, 0.0
            for _ in range(horizon):
                logits = model(inputs_embeds=gen).logits[:, -1]
                logp = torch.log_softmax(logits, -1)
                score = score - logp[0, eos_id]
                gen = torch.cat([gen, embed(logp.argmax(-1)).unsqueeze(1)], 1)
        return e, score

    def gen_len(ids, cap=cap):
        kw = dict(max_new_tokens=cap, do_sample=False)
        if not is_s2s:
            kw.update(attention_mask=torch.ones_like(ids), pad_token_id=tokenizer.pad_token_id or eos_id)
        with torch.no_grad():
            out = model.generate(ids, **kw)
        return out.shape[1] - (ids.shape[1] if not is_s2s else 1)

    ids = tokenizer(x, return_tensors="pt").input_ids.to(device)
    clean_ids = ids.clone()
    W = embed.weight.detach()
    special = set(tokenizer.all_special_ids)
    changed, log = set(), []

    for _ in range(budget):
        e, score = neg_eos_logprob(ids)
        score.backward()
        importance = e.grad.sum(-1).squeeze(0)  # paper Eq. 2: signed sum, not norm
        order = torch.argsort(importance.abs(), descending=True).tolist()
        pos = next((p for p in order if p not in changed and ids[0, p].item() not in special), None)
        if pos is None:
            break

        with torch.no_grad():
            base = neg_eos_logprob(ids)[1].item()
            orig_id = ids[0, pos].item()
            sims = torch.nn.functional.cosine_similarity(W, W[orig_id].unsqueeze(0), dim=-1)
            sims[orig_id] = -1
            best_ids, best_score = ids, base
            for cand in torch.topk(sims, n_cand).indices.tolist():
                trial = ids.clone()
                trial[0, pos] = cand
                s = neg_eos_logprob(trial)[1].item()
                if s > best_score:
                    best_ids, best_score = trial, s

        log.append({"position": pos, "objective_before": base, "objective_after": best_score})
        if best_score <= base:
            break
        ids, changed = best_ids, changed | {pos}

    adv_x = tokenizer.decode(ids[0], skip_special_tokens=True)
    logs = {
        "iterations": log,
        "num_tokens_perturbed": len(changed),
        "clean_generated_tokens": gen_len(clean_ids),
        "adv_generated_tokens": gen_len(ids),
    }
    logs["length_increase"] = logs["adv_generated_tokens"] - logs["clean_generated_tokens"]
    return adv_x, logs

class Attacker:
    """Required public API: Attacker(model).run(x, config) -> (adv_x, logs)."""

    def __init__(self, model, tokenizer=None):
        self.model = model
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
        self.tokenizer = tokenizer

    def run(self, x, config):
        return attack(self.model, self.tokenizer, x, config)
