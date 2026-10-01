"""A tiny Qwen2-shaped model, so the library trains and tests on a CPU with no downloads."""
from __future__ import annotations

from transformers import Qwen2Config, Qwen2ForCausalLM


def toy_model(tokenizer, hidden: int = 128, layers: int = 4) -> Qwen2ForCausalLM:
    cfg = Qwen2Config(vocab_size=len(tokenizer), hidden_size=hidden, intermediate_size=2 * hidden,
                      num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=512, tie_word_embeddings=True)
    m = Qwen2ForCausalLM(cfg)
    m.config.pad_token_id = tokenizer.pad_token_id
    return m
