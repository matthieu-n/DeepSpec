from deepspec.data import CacheCollator
from deepspec.modeling.mtp.common import MTPDraftModel, get_mtp_model_spec
from deepspec.modeling.mtp.loss import compute_mtp_loss
from deepspec.trainer.base_trainer import BaseTrainer

# Importing registers each model_type's MTPModelSpec as a side effect.
import deepspec.modeling.mtp.qwen3_5  # noqa: E402,F401


class MTPTrainer(BaseTrainer):
    """Finetunes a pretrained model's native MTP ("NextN") head on a target
    data distribution, starting from the checkpoint's own mtp.* weights.

    Model-type agnostic: `_build_draft_model` dispatches on
    `target_config.model_type` through the MTP model registry
    (deepspec/modeling/mtp/common.py), so supporting a new architecture is a
    matter of registering a new MTPModelSpec, not touching this trainer.
    """

    data_collator_cls = CacheCollator

    def _build_draft_model(self, *, target_config, model_args):
        model_type = str(target_config.model_type)
        spec = get_mtp_model_spec(model_type)
        text_config = target_config.get_text_config()
        head_config = spec.build_head_config(text_config)
        draft_model = MTPDraftModel(
            head_config,
            model_type=model_type,
            target_layer_ids=model_args.target_layer_ids,
        )
        spec.load_pretrained_weights(
            draft_model.head, str(model_args.target_model_name_or_path)
        )
        return draft_model

    def run_batch(self, batch, tag: str = "train"):
        return compute_mtp_loss(model=self.model, batch=batch)


__all__ = ["MTPTrainer"]
