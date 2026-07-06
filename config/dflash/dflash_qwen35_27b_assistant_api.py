import os

from deepspec.trainer import Qwen3DSparkTrainer


BASE_TB_DIR = os.path.expanduser("~/tensorboard")
BASE_CKPT_DIR = os.path.expanduser("~/checkpoints")
project_name = "deepspec"
exp_name = "dflash_block16_qwen35_27b_assistant_api"
seed = 42

model = dict(
    target_model_name_or_path="Qwen/Qwen3.5-27B",
    draft_model_name_or_path="modal-labs/Qwen3.5-27B-DFlash",
    block_size=16,
    num_draft_layers=6,
    target_layer_ids=[1, 10, 18, 27, 35, 44, 52, 61],
    mask_token_id=248077,
    # Reduced from 512: only 200 assistant_api samples, and full 512 anchors
    # at max_length=8192 OOM'd a single 96GB GPU together with
    # torch_compile's graph-capture memory spike.
    num_anchors=128,

    # Disable markov head.
    markov_rank=0,

    # Disable confidence head.
    confidence_head_alpha=0.0,

    # CE-only loss.
    loss_decay_gamma=4.0,
    ce_loss_alpha=1.0,
    l1_loss_alpha=0.0,
)

train = dict(
    trainer_cls=Qwen3DSparkTrainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    # Scaled down from the 512 used for the larger gsm8k-style recipes: only
    # 200 assistant_api samples here, so 512 would give <1 step/epoch.
    global_batch_size=16,
    num_train_epochs=5,
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    # Disabled: torch.compile's graph-capture memory spike was the proximate
    # trigger of the CUDA OOM at max_length=8192 on a single 96GB GPU; not
    # worth the speed tradeoff for a 200-sample/5-epoch run.
    torch_compile=False,
)

logging = dict(
    logging_steps=5,
    checkpointing_steps=20,
    mlflow_tracking_uri="https://mlflow.us1.staging.dog",
    mlflow_experiment_name="qwen-3.5-27B-dflash",
)

data = dict(
    target_cache_path=None,
    chat_template="qwen",
    # assistant_api system+user prompts run much longer than gsm8k (up to
    # ~29k chars / ~7k+ tokens) -- default 4096 would truncate many of them.
    max_length=8192,
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
