from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import torch
from torch import nn


@dataclass
class MTPForwardOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor


class MTPModelSpec(NamedTuple):
    """Everything needed to attach a pretrained MTP ("NextN") head to a base model.

    build_head_config(text_config) -> head_config
        Derives the 1-layer, full-attention config for the head from the base
        model's own text_config.
    build_head(head_config) -> nn.Module
        Constructs the (randomly initialized) head module.
    load_pretrained_weights(head, model_name_or_path) -> nn.Module
        Fetches the checkpoint's native `mtp.*` weights and loads them into head.
    """

    build_head_config: Callable
    build_head: Callable
    load_pretrained_weights: Callable


MTP_MODEL_REGISTRY: dict[str, MTPModelSpec] = {}


def register_mtp_model(model_type: str):
    def _register(spec: MTPModelSpec) -> MTPModelSpec:
        MTP_MODEL_REGISTRY[model_type] = spec
        return spec

    return _register


def get_mtp_model_spec(model_type: str) -> MTPModelSpec:
    assert model_type in MTP_MODEL_REGISTRY, (
        f"No MTP head registered for model_type={model_type!r}. "
        f"Registered: {sorted(MTP_MODEL_REGISTRY)}"
    )
    return MTP_MODEL_REGISTRY[model_type]


class MTPDraftModel(nn.Module):
    """Generic wrapper around a registered MTP head plus the frozen
    embed_tokens/lm_head it shares with its base model.

    This is model-type agnostic: adding a new architecture only requires
    registering a new MTPModelSpec (see deepspec/modeling/mtp/qwen3_5), not a
    new draft-model class. Mirrors the embed_tokens/lm_head pre-allocate +
    initialize_embeddings_and_head(freeze=True) pattern used by the Eagle3
    draft models, so it plugs directly into BaseTrainer.build_models().
    """

    def __init__(self, head_config, *, model_type: str):
        super().__init__()
        spec = get_mtp_model_spec(model_type)
        self.head = spec.build_head(head_config)
        self.embed_tokens = nn.Embedding(head_config.vocab_size, head_config.hidden_size)
        self.lm_head = nn.Linear(head_config.hidden_size, head_config.vocab_size, bias=False)

    def initialize_embeddings_and_head(self, *, embed_tokens, lm_head, freeze: bool = True):
        assert self.embed_tokens.weight.shape == embed_tokens.weight.shape
        assert self.lm_head.weight.shape == lm_head.weight.shape
        with torch.no_grad():
            self.embed_tokens.weight.copy_(embed_tokens.weight.detach())
            self.lm_head.weight.copy_(lm_head.weight.detach())
        if freeze:
            self.embed_tokens.requires_grad_(False)
            self.lm_head.requires_grad_(False)

    def forward(self, *, target_last_hidden_states, input_ids, position_ids, attention_mask):
        next_token_embeds = self.embed_tokens(input_ids)
        head_hidden_states = self.head(
            target_last_hidden_states,
            next_token_embeds,
            position_ids,
            attention_mask,
        )
        return self.lm_head(head_hidden_states)


__all__ = [
    "MTPForwardOutput",
    "MTPModelSpec",
    "MTPDraftModel",
    "MTP_MODEL_REGISTRY",
    "register_mtp_model",
    "get_mtp_model_spec",
]
