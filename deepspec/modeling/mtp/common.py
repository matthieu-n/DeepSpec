from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, NamedTuple

import torch
from torch import nn
from transformers import PretrainedConfig


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

    _META_FILENAME = "mtp_meta.json"
    _WEIGHTS_FILENAME = "pytorch_model.bin"

    def __init__(self, head_config, *, model_type: str, target_layer_ids):
        super().__init__()
        self.model_type = model_type
        spec = get_mtp_model_spec(model_type)
        self.head = spec.build_head(head_config)
        self.embed_tokens = nn.Embedding(head_config.vocab_size, head_config.hidden_size)
        self.lm_head = nn.Linear(head_config.hidden_size, head_config.vocab_size, bias=False)
        self.target_layer_ids = [int(layer_id) for layer_id in target_layer_ids]
        self.config = head_config

    def initialize_embeddings_and_head(self, *, embed_tokens, lm_head, freeze: bool = True):
        assert self.embed_tokens.weight.shape == embed_tokens.weight.shape
        assert self.lm_head.weight.shape == lm_head.weight.shape
        with torch.no_grad():
            self.embed_tokens.weight.copy_(embed_tokens.weight.detach())
            self.lm_head.weight.copy_(lm_head.weight.detach())
        if freeze:
            self.embed_tokens.requires_grad_(False)
            self.lm_head.requires_grad_(False)

    def set_embedding_head_trainable(self, trainable: bool):
        self.embed_tokens.requires_grad_(trainable)
        self.lm_head.requires_grad_(trainable)

    def save_pretrained(self, save_directory, *, state_dict=None):
        os.makedirs(save_directory, exist_ok=True)
        if state_dict is None:
            state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, self._WEIGHTS_FILENAME))
        self.config.to_json_file(os.path.join(save_directory, "config.json"))
        with open(os.path.join(save_directory, self._META_FILENAME), "w") as f:
            json.dump(
                {"model_type": self.model_type, "target_layer_ids": self.target_layer_ids},
                f,
            )

    @classmethod
    def from_pretrained(cls, load_directory, *, dtype=None, attn_implementation=None):
        with open(os.path.join(load_directory, cls._META_FILENAME)) as f:
            meta = json.load(f)
        head_config = PretrainedConfig.from_json_file(
            os.path.join(load_directory, "config.json")
        )
        if attn_implementation is not None:
            head_config._attn_implementation = attn_implementation
        model = cls(
            head_config,
            model_type=meta["model_type"],
            target_layer_ids=meta["target_layer_ids"],
        )
        state_dict = torch.load(
            os.path.join(load_directory, cls._WEIGHTS_FILENAME),
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        if dtype is not None:
            model = model.to(dtype=dtype)
        return model

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
