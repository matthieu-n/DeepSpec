from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4RMSNorm,
    Gemma4TextMLP,
    Gemma4TextRotaryEmbedding,
    apply_rotary_pos_emb as apply_gemma4_rotary_pos_emb,
)

from deepspec.modeling.eagle3.common import prepare_4d_causal_attention_mask
from deepspec.modeling.mtp.common import MTPModelSpec, register_mtp_model
from deepspec.modeling.mtp.gemma4.config import build_mtp_head_config


class Gemma4MTPAttention(nn.Module):
    """Standard single-input self-attention over the head's own combined hidden
    state, using Gemma4's "global" (full_attention) head_dim/kv-head hyperparameters.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_attention_heads = int(config.num_attention_heads)
        self.head_dim = int(config.global_head_dim)
        self.use_alternative_attention = bool(config.attention_k_eq_v)
        if self.use_alternative_attention:
            self.num_key_value_heads = int(config.num_global_key_value_heads)
        else:
            self.num_key_value_heads = int(config.num_key_value_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0, (
            "num_attention_heads must be divisible by the Gemma4 key/value head count."
        )
        self.scaling = 1.0
        self.attention_dropout = float(config.attention_dropout)

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=bool(config.attention_bias),
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=bool(config.attention_bias),
        )
        self.v_proj = None
        if not self.use_alternative_attention:
            self.v_proj = nn.Linear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=bool(config.attention_bias),
            )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=bool(config.attention_bias),
        )
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            with_scale=False,
        )

    def _repeat_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.num_key_value_groups == 1:
            return hidden_states
        return hidden_states.repeat_interleave(self.num_key_value_groups, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len = hidden_states.shape[:-1]
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_attention_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        if self.use_alternative_attention:
            v = k
        else:
            v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = self.v_norm(v).transpose(1, 2)

        cos, sin = position_embeddings
        q = apply_gemma4_rotary_pos_emb(q, cos, sin, unsqueeze_dim=1)
        k = apply_gemma4_rotary_pos_emb(k, cos, sin, unsqueeze_dim=1)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            is_causal=attention_mask is None,
            scale=self.scaling,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)


class Gemma4MTPDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert not bool(config.enable_moe_block), (
            "Gemma4MTPDecoderLayer does not support MoE blocks; the from-scratch "
            "MTP head is dense-only."
        )
        assert int(config.hidden_size_per_layer_input) == 0, (
            "Gemma4MTPDecoderLayer does not support per-layer input gates."
        )
        self.self_attn = Gemma4MTPAttention(config)
        self.mlp = Gemma4TextMLP(config, layer_idx=0)
        self.input_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return residual + hidden_states


class Gemma4MTPHead(nn.Module):
    """A from-scratch "NextN"/MTP module for google/gemma-4-26B-A4B-it.

    Unlike Qwen3.5/Qwen3.6, this checkpoint ships no pretrained MTP head, so the
    head is randomly initialized and trained from scratch (see
    `load_pretrained_mtp_weights`, a no-op). Architecturally it mirrors the Qwen3.5
    MTP head: combine the base model's final hidden state at position i and the
    embedding of the token at position i+1 through `fc`, run one extra dense
    full-attention decoder layer, and project through the base model's shared
    lm_head (done by the generic `MTPDraftModel` wrapper, not here).
    """

    def __init__(self, head_config):
        super().__init__()
        hidden_size = int(head_config.hidden_size)
        eps = float(head_config.rms_norm_eps)
        self.pre_fc_norm_embedding = Gemma4RMSNorm(hidden_size, eps=eps)
        self.pre_fc_norm_hidden = Gemma4RMSNorm(hidden_size, eps=eps)
        self.fc = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.layer = Gemma4MTPDecoderLayer(head_config)
        self.norm = Gemma4RMSNorm(hidden_size, eps=eps)
        self.rotary_emb = Gemma4TextRotaryEmbedding(head_config, layer_type="full_attention")

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
        position_embeddings = self.rotary_emb(
            combined, position_ids, layer_type="full_attention"
        )
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
        )
        return self.norm(combined)


def load_pretrained_mtp_weights(head, model_name_or_path):
    """No-op: google/gemma-4-26B-A4B-it has no pretrained mtp.* weights, so the
    head keeps its random initialization and trains from scratch.
    """
    del model_name_or_path
    return head


register_mtp_model("gemma4")(
    MTPModelSpec(
        build_head_config=build_mtp_head_config,
        build_head=Gemma4MTPHead,
        load_pretrained_weights=load_pretrained_mtp_weights,
    )
)


__all__ = ["Gemma4MTPHead", "load_pretrained_mtp_weights"]
