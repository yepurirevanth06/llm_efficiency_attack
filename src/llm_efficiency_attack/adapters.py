"""Model adapters.

This is the "isolate model-specific assumptions" seam required by the task.
The rest of the package (the attack loop, the objective, the cost metric)
never touches a Hugging Face model or tokenizer directly -- it only talks to
a :class:`ModelAdapter`. Supporting a new architecture family means adding a
new adapter subclass, not touching the attack logic.

Two families are supported out of the box:

- ``CausalLMAdapter`` for decoder-only models (GPT-2 style,
  ``AutoModelForCausalLM``).
- ``Seq2SeqLMAdapter`` for encoder-decoder models (T5/BART style,
  ``AutoModelForSeq2SeqLM``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class ModelAdapter(ABC):
    """Common interface the attack loop needs from any Hugging Face model."""

    def __init__(self, model, tokenizer, device: str):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    # -- tokenization ------------------------------------------------------
    def encode(self, text: str) -> torch.Tensor:
        """Tokenize ``text`` into a ``(1, seq_len)`` LongTensor on ``self.device``."""
        ids = self.tokenizer(text, return_tensors="pt").input_ids
        return ids.to(self.device)

    def decode(self, input_ids: torch.Tensor) -> str:
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    @property
    def embedding_matrix(self) -> torch.Tensor:
        """The input token embedding matrix, shape ``(vocab_size, hidden_dim)``."""
        return self.model.get_input_embeddings().weight

    @property
    def eos_token_id(self) -> int:
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError(
                "This tokenizer has no eos_token_id, which the length-based "
                "objective requires. Set tokenizer.eos_token explicitly."
            )
        return eos_id

    # -- the piece that differs between architectures -----------------------
    @abstractmethod
    def eos_logprobs(self, input_ids: torch.Tensor, horizon: int) -> torch.Tensor:
        """Return a ``(horizon,)`` tensor: log P(EOS) at each of the first
        ``horizon`` decode steps, computed with gradients enabled so the
        caller can backprop into ``input_ids``' embeddings.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_length(self, input_ids: torch.Tensor, max_new_tokens: int) -> int:
        """Run real (no-grad) generation and return the number of new tokens
        produced. This is the ground-truth cost metric the objective is a
        differentiable proxy for.
        """
        raise NotImplementedError


class CausalLMAdapter(ModelAdapter):
    """Adapter for decoder-only (causal) language models."""

    def eos_logprobs(self, input_ids: torch.Tensor, horizon: int) -> torch.Tensor:
        """Score-only path (no gradient needed here -- the attacker computes
        gradients separately via its own leaf-embeddings tensor)."""
        generated = self.model.get_input_embeddings()(input_ids)
        logprobs = []
        for _ in range(horizon):
            out = self.model(inputs_embeds=generated)
            next_logits = out.logits[:, -1, :]
            next_logprobs = torch.log_softmax(next_logits, dim=-1)
            logprobs.append(next_logprobs[0, self.eos_token_id])

            # Feed the (soft) most likely next token back in via its embedding
            # so gradients flow all the way back to the *input* embeddings.
            next_id = next_logits.argmax(dim=-1)
            next_embed = self.model.get_input_embeddings()(next_id).unsqueeze(1)
            generated = torch.cat([generated, next_embed], dim=1)

        return torch.stack(logprobs)

    @torch.no_grad()
    def generate_length(self, input_ids: torch.Tensor, max_new_tokens: int) -> int:
        prompt_len = input_ids.shape[1]
        output = self.model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id or self.eos_token_id,
        )
        return int(output.shape[1] - prompt_len)


class Seq2SeqLMAdapter(ModelAdapter):
    """Adapter for encoder-decoder (seq2seq) language models."""

    def eos_logprobs(self, input_ids: torch.Tensor, horizon: int) -> torch.Tensor:
        """Score-only path (no gradient needed here -- the attacker computes
        gradients separately via its own leaf-embeddings tensor)."""
        encoder_embeds = self.model.get_input_embeddings()(input_ids)
        encoder_outputs = self.model.get_encoder()(inputs_embeds=encoder_embeds)

        decoder_start_id = self.model.config.decoder_start_token_id
        if decoder_start_id is None:
            decoder_start_id = self.eos_token_id
        decoder_input_ids = torch.tensor([[decoder_start_id]], device=self.device)

        logprobs = []
        for _ in range(horizon):
            out = self.model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
            )
            next_logits = out.logits[:, -1, :]
            next_logprobs = torch.log_softmax(next_logits, dim=-1)
            logprobs.append(next_logprobs[0, self.eos_token_id])

            next_id = next_logits.argmax(dim=-1, keepdim=True)
            decoder_input_ids = torch.cat([decoder_input_ids, next_id], dim=1)

        return torch.stack(logprobs)

    @torch.no_grad()
    def generate_length(self, input_ids: torch.Tensor, max_new_tokens: int) -> int:
        output = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        # Seq2seq output includes the decoder start token; don't count it.
        return int(output.shape[1] - 1)


def build_adapter(model, tokenizer, device: str) -> ModelAdapter:
    """Pick the right adapter for ``model`` based on its architecture family.

    This is the single place that needs to change to support a new model
    family: add a branch (or a registry) here, plus a matching
    ``ModelAdapter`` subclass.
    """
    if getattr(model.config, "is_encoder_decoder", False):
        return Seq2SeqLMAdapter(model, tokenizer, device)
    return CausalLMAdapter(model, tokenizer, device)
