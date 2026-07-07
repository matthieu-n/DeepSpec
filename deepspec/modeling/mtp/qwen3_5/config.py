import copy


def build_mtp_head_config(text_config):
    """Qwen3.5/Qwen3.6's MTP head is a single decoder layer, always full-attention
    (confirmed via the checkpoint's mtp.* weight keys: only self_attn.*, never the
    linear-attention/Mamba weights present in the main backbone's linear_attention
    layers). Everything else (hidden_size, rope params, attention hyperparameters)
    matches the base model's own full-attention layers exactly.
    """
    head_config = copy.deepcopy(text_config)
    head_config.num_hidden_layers = 1
    head_config.layer_types = ["full_attention"]
    head_config._attn_implementation = text_config._attn_implementation or "sdpa"
    return head_config


__all__ = ["build_mtp_head_config"]
