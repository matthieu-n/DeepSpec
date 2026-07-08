from __future__ import annotations

import json
import re

import torch
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
)

from deepspec.modeling.eagle3.common import prepare_4d_causal_attention_mask
from deepspec.modeling.mtp.common import MTPModelSpec, register_mtp_model
from deepspec.modeling.mtp.qwen3_5.config import build_mtp_head_config


class Qwen3_5MTPHead(nn.Module):
    """The pretrained "NextN"/MTP module shipped with Qwen3.5/Qwen3.6 checkpoints.

    Reverse-engineered from sglang's `Qwen3_5ForCausalLMMTP` (its NEXTN speculative
    decoding path), since transformers itself only ever loads and discards the
    `mtp.*` weights (see `Qwen3_5PreTrainedModel._keys_to_ignore_on_load_unexpected`).

    Given the base model's final hidden state at position i and the embedding of
    the token at position i+1, predicts the token at position i+2: combine the two
    (embedding first, hidden second) through `fc`, run one extra full-attention
    decoder layer, and project through the base model's shared lm_head.
    """

    def __init__(self, head_config):
        super().__init__()
        hidden_size = int(head_config.hidden_size)
        eps = float(head_config.rms_norm_eps)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(hidden_size, eps=eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(hidden_size, eps=eps)
        self.fc = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.layer = Qwen3_5DecoderLayer(head_config, layer_idx=0)
        self.norm = Qwen3_5RMSNorm(hidden_size, eps=eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(head_config)

    def forward(self, hidden_states, next_token_embeds, position_ids, attention_mask):
        """
        hidden_states: [B, L, H] base model's final hidden state at positions 0..L-1
        next_token_embeds: [B, L, H] embed_tokens of the token at positions 1..L
            (i.e. already shifted by the caller to line up with hidden_states)
        position_ids: [B, L] natural sequence position of the *embedded* token
        attention_mask: [B, L] 1 for real tokens, 0 for padding
        """
        combined = self.fc(
            torch.cat(
                [
                    self.pre_fc_norm_embedding(next_token_embeds),
                    self.pre_fc_norm_hidden(hidden_states),
                ],
                dim=-1,
            )
        )
        position_embeddings = self.rotary_emb(combined, position_ids)
        causal_mask = prepare_4d_causal_attention_mask(
            attention_mask=attention_mask,
            dtype=combined.dtype,
            q_len=combined.shape[1],
            kv_len=combined.shape[1],
            past_seen_tokens=0,
            device=combined.device,
        )
        combined = self.layer(
            combined,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            position_ids=position_ids,
        )
        return self.norm(combined)


_WEIGHT_KEY_RENAMES = (
    (re.compile(r"^mtp\.layers\.0\."), "layer."),
    (re.compile(r"^mtp\.norm\."), "norm."),
    (re.compile(r"^mtp\.fc\."), "fc."),
    (re.compile(r"^mtp\.pre_fc_norm_embedding\."), "pre_fc_norm_embedding."),
    (re.compile(r"^mtp\.pre_fc_norm_hidden\."), "pre_fc_norm_hidden."),
)


def _remap_mtp_weight_key(key):
    for pattern, replacement in _WEIGHT_KEY_RENAMES:
        if pattern.match(key):
            return pattern.sub(replacement, key)
    return None


def load_pretrained_mtp_weights(head, model_name_or_path):
    """Loads the checkpoint's native `mtp.*` weights into `head`.

    transformers' Qwen3.5 loader lists `mtp.*` in `_keys_to_ignore_on_load_unexpected`
    and drops them silently, so we fetch and load them ourselves directly from the
    checkpoint's safetensors shards.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    index_path = hf_hub_download(
        repo_id=model_name_or_path, filename="model.safetensors.index.json"
    )
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    mtp_shard_files = sorted(
        {fname for key, fname in weight_map.items() if key.startswith("mtp.")}
    )
    assert mtp_shard_files, (
        f"No mtp.* weights found in {model_name_or_path}'s safetensors index. "
        "Is this a checkpoint that ships a native MTP head?"
    )

    state_dict = {}
    for fname in mtp_shard_files:
        shard_path = hf_hub_download(repo_id=model_name_or_path, filename=fname)
        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                if not key.startswith("mtp."):
                    continue
                mapped_key = _remap_mtp_weight_key(key)
                if mapped_key is None:
                    continue
                state_dict[mapped_key] = f.get_tensor(key)

    missing, unexpected = head.load_state_dict(state_dict, strict=False)
    assert not unexpected, f"Unexpected keys loading MTP head from {model_name_or_path}: {unexpected}"
    assert not missing, f"Missing keys loading MTP head from {model_name_or_path}: {missing}"
    return head


register_mtp_model("qwen3_5")(
    MTPModelSpec(
        build_head_config=build_mtp_head_config,
        build_head=Qwen3_5MTPHead,
        load_pretrained_weights=load_pretrained_mtp_weights,
    )
)


__all__ = ["Qwen3_5MTPHead", "load_pretrained_mtp_weights"]
