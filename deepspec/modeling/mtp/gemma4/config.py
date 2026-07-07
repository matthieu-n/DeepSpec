import copy


def build_mtp_head_config(text_config):
    """google/gemma-4-26B-A4B-it ships no pretrained MTP/NextN head, so this head
    is trained from scratch. Model it as a single full-attention ("global") decoder
    layer matching the base model's own full-attention layers exactly: same
    hidden_size and rope_parameters, same attention hyperparameters (the Gemma4
    "global" head_dim/kv-head count used for full_attention layers). MoE is disabled
    to keep the from-scratch head simple, mirroring the Gemma4 Eagle3 prototype's own
    restriction (`assert not config.enable_moe_block`).
    """
    head_config = copy.deepcopy(text_config)
    head_config.num_hidden_layers = 1
    head_config.layer_types = ["full_attention"]
    head_config.enable_moe_block = False
    head_config._attn_implementation = text_config._attn_implementation or "sdpa"
    return head_config


__all__ = ["build_mtp_head_config"]
