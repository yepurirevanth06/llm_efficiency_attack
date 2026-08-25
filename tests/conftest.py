"""Shared fixtures for the test suite.

We build a genuinely-real ``transformers.GPT2LMHeadModel`` (same class used
in production, just tiny and randomly initialized) paired with a small
in-memory word-level tokenizer built with the ``tokenizers`` library. This
keeps the test suite fast and network-free while still exercising the real
Hugging Face model/generation code paths end to end.
"""

from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    T5Config,
    T5ForConditionalGeneration,
)

_VOCAB = [
    "[PAD]", "[UNK]", "[BOS]", "[EOS]",
    "the", "a", "weather", "today", "is", "sunny", "and", "warm",
    "cat", "dog", "sat", "on", "mat", "ran", "fast", "slow", "quick",
    "brown", "fox", "jumps", "over", "lazy",
]


@pytest.fixture(scope="session")
def tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {tok: i for i, tok in enumerate(_VOCAB)}
    core = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    core.pre_tokenizer = Whitespace()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    return tokenizer


@pytest.fixture(scope="session")
def tiny_model(tiny_tokenizer) -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=len(_VOCAB),
        n_positions=32,
        n_embd=16,
        n_layer=2,
        n_head=2,
        bos_token_id=tiny_tokenizer.bos_token_id,
        eos_token_id=tiny_tokenizer.eos_token_id,
    )
    torch.manual_seed(0)
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


@pytest.fixture(scope="session")
def tiny_seq2seq_model(tiny_tokenizer) -> T5ForConditionalGeneration:
    config = T5Config(
        vocab_size=len(_VOCAB),
        d_model=16,
        d_ff=32,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=2,
        decoder_start_token_id=tiny_tokenizer.bos_token_id,
        eos_token_id=tiny_tokenizer.eos_token_id,
        pad_token_id=tiny_tokenizer.pad_token_id,
        is_encoder_decoder=True,
    )
    torch.manual_seed(0)
    model = T5ForConditionalGeneration(config)
    model.eval()
    return model
