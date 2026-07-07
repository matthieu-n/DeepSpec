from deepspec.modeling.mtp.common import (
    MTP_MODEL_REGISTRY,
    MTPForwardOutput,
    MTPModelSpec,
    get_mtp_model_spec,
    register_mtp_model,
)

# Importing registers this model_type's MTPModelSpec as a side effect.
import deepspec.modeling.mtp.qwen3_5  # noqa: E402,F401

__all__ = [
    "MTP_MODEL_REGISTRY",
    "MTPForwardOutput",
    "MTPModelSpec",
    "get_mtp_model_spec",
    "register_mtp_model",
]
