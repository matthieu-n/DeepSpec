import copy
import json

from deepspec.modeling.dspark.common import validate_target_layer_ids


TRAIN_ATTN_IMPLEMENTATION = "flex_attention"


def get_qwen3_text_config(target_config):
    model_type = str(target_config.model_type)
    if model_type == "qwen3_5":
        text_config = target_config.text_config
        assert str(text_config.model_type) == "qwen3_5_text", (
            "Qwen3.5 DSpark expects target_config.text_config.model_type to be "
            f"'qwen3_5_text', got {text_config.model_type!r}."
        )
        return copy.deepcopy(text_config)
    return copy.deepcopy(target_config)


def build_draft_config(
    target_config,
    model_args,
):
    target_text_config = get_qwen3_text_config(target_config)
    num_target_layers = int(target_text_config.num_hidden_layers)
    num_draft_layers = int(model_args.num_draft_layers)
    layer_types = ["full_attention"] * num_draft_layers
    assert "target_layer_ids" in model_args, "target_layer_ids must be provided."
    target_layer_ids = validate_target_layer_ids(
        model_args.target_layer_ids,
        num_target_layers,
    )

    confidence_head_alpha = float(model_args.confidence_head_alpha)
    assert confidence_head_alpha >= 0.0
    enable_confidence_head = confidence_head_alpha > 0.0
    if enable_confidence_head:
        assert "confidence_head_with_markov" in model_args, (
            "confidence_head_with_markov must be provided when "
            "confidence_head_alpha > 0."
        )
    markov_rank = int(model_args.markov_rank)
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank > 0:
        assert "markov_head_type" in model_args, (
            "markov_head_type must be provided when markov_rank > 0."
        )

    draft_checkpoint_config = None
    draft_checkpoint = getattr(model_args, "draft_model_name_or_path", None)
    if draft_checkpoint:
        draft_checkpoint_config = get_pretrained_draft_attn_config(draft_checkpoint)

    draft_config = target_text_config
    draft_config.architectures = ["Qwen3DSparkModel"]
    draft_config.target_model_type = str(target_config.model_type)
    draft_config.num_target_layers = num_target_layers
    draft_config.num_hidden_layers = num_draft_layers
    draft_config.block_size = int(model_args.block_size)
    draft_config.tie_word_embeddings = False
    draft_config.layer_types = layer_types
    draft_config._attn_implementation = TRAIN_ATTN_IMPLEMENTATION
    draft_config.mask_token_id = int(model_args.mask_token_id)
    draft_config.target_layer_ids = target_layer_ids
    draft_config.num_anchors = int(model_args.num_anchors)
    draft_config.enable_confidence_head = enable_confidence_head
    if enable_confidence_head:
        draft_config.confidence_head_with_markov = bool(
            model_args.confidence_head_with_markov
        )
    draft_config.markov_rank = markov_rank
    if markov_rank > 0:
        draft_config.markov_head_type = str(model_args.markov_head_type)

    # A pretrained draft checkpoint (e.g. a DFlash draft trained externally)
    # may use an attention geometry that differs from the target model's own
    # config, since the draft is a much smaller network. Override the
    # target-derived attention fields with the checkpoint's own values so the
    # freshly-constructed module's tensor shapes match the checkpoint's
    # state_dict exactly.
    if draft_checkpoint_config is not None:
        for key, value in draft_checkpoint_config.items():
            setattr(draft_config, key, value)

    # draft_config started life as a deepcopy of the *target's* text config
    # (get_qwen3_text_config above), so for hybrid targets (e.g. Qwen3.5's
    # mamba+attention text_config) it still carries the target-only fields
    # below. Qwen3DSparkAttention/modeling.py never reads them -- the draft
    # is a plain attention-only stack -- so they're dead weight that doesn't
    # belong on a draft checkpoint and, when a pretrained draft checkpoint
    # (e.g. modal-labs/Qwen3.5-27B-DFlash) has none of them, makes the
    # fine-tuned checkpoint's config.json diverge from the one it started
    # training from.
    _TARGET_ONLY_FIELDS = (
        "attn_output_gate",
        "full_attention_interval",
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_value_head_dim",
        "mamba_ssm_dtype",
        "mlp_only_layers",
        "mtp_num_hidden_layers",
        "mtp_use_dedicated_embeddings",
        "partial_rotary_factor",
    )
    for field in _TARGET_ONLY_FIELDS:
        if hasattr(draft_config, field):
            delattr(draft_config, field)

    return draft_config


_DRAFT_ATTN_CONFIG_KEYS = (
    "model_type",
    "head_dim",
    "num_attention_heads",
    "num_key_value_heads",
    "sliding_window",
    "use_sliding_window",
    "max_window_layers",
    "rope_parameters",
    # layer_types must come from the pretrained draft checkpoint, not the
    # hardcoded all-full_attention default above -- the checkpoint's own
    # per-layer sliding/full pattern determines which layers apply
    # config.sliding_window at runtime (see Qwen3DSparkAttention.__init__ in
    # modeling.py). Silently forcing full_attention on every layer changes
    # the draft's attention geometry away from the checkpoint it started
    # from.
    "layer_types",
)


def get_pretrained_draft_attn_config(draft_model_name_or_path):
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(
        repo_id=draft_model_name_or_path, filename="config.json"
    )
    with open(config_path) as f:
        raw_config = json.load(f)
    return {
        key: raw_config[key] for key in _DRAFT_ATTN_CONFIG_KEYS if key in raw_config
    }


__all__ = [
    "build_draft_config",
    "get_pretrained_draft_attn_config",
]
