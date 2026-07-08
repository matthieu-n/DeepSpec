import os

from deepspec.trainer import MTPTrainer


BASE_TB_DIR = os.path.expanduser("~/tensorboard")
# Persisted on the shared PVC (not /work, which is an emptyDir GC'd with the
# Job after ttlSecondsAfterFinished) so checkpoints survive job cleanup.
BASE_CKPT_DIR = "/share/dflash-logits-cache/checkpoints"
project_name = "deepspec"
exp_name = "mtp_qwen36_27b_code_intelligence"
seed = 42

model = dict(
    target_model_name_or_path="Qwen/Qwen3.6-27B",
    # Only used by scripts/data/prepare_target_cache.py to build the offline
    # cache this trainer reads from. MTP training only consumes
    # target_last_hidden_states (the final post-norm hidden state), not any
    # intermediate layer, so this is a single placeholder layer id purely to
    # satisfy the cache writer's non-empty target_layer_ids requirement.
    target_layer_ids=[0],
)

train = dict(
    trainer_cls=MTPTrainer,
    lr=1.0e-5,
    warmup_ratio=0.03,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=8,
    num_train_epochs=3,
    max_train_steps=None,
    max_grad_norm=1.0,
    # Training itself never re-runs the 27B backbone (target_last_hidden_states
    # comes from the offline cache) -- embed_tokens + lm_head + the MTP head
    # are the only resident weights (~4GB in bf16), so no_shard is fine even
    # on a single GPU.
    sharding_strategy="no_shard",
    torch_compile=False,
)

logging = dict(
    logging_steps=1,
    checkpointing_steps=50,
    # Independent eval cadence (base_trainer.py calls self.evaluate() on this
    # cadence in addition to the checkpointing_steps-coupled eval, plus once
    # unconditionally before the first training step).
    eval_steps=2,
)

data = dict(
    target_cache_path=None,
    val_target_cache_path=None,
    chat_template="qwen",
    # code-intelligence traces carry full file contents in the user turn;
    # p95 is ~7.4k chars (~2-3k tokens) but the tail reaches ~28k chars, so
    # size for that tail rather than truncating training examples.
    max_length=12288,
    num_workers=4,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name = str(cfg["project_name"])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(
        BASE_CKPT_DIR,
        project_name,
        exp_name,
    )
    logging_cfg["tensorboard_dir"] = os.path.join(
        BASE_TB_DIR,
        project_name,
        exp_name,
    )
    cfg["logging"] = logging_cfg
    return cfg
