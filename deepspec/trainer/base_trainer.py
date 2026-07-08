from contextlib import nullcontext
import math
import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

from deepspec.data import CacheDataset, validate_train_cache
from deepspec.data.cuda_prefetcher import CUDAPrefetcher
from deepspec.utils import (
    BF16Optimizer,
    StatelessResumableDistributedSampler,
    ensure_dir,
    init_dist,
    is_global_main_process,
    print_on_global_main,
    print_on_local_main,
)
from deepspec.utils.metrics import flush as flush_metrics, reset as reset_metrics
from deepspec.trainer.ckpt_manager import (
    discover_latest_checkpoint,
    load_resume_draft_model,
    load_training_state,
    save_checkpoint,
)
import deepspec.utils.training_logger as training_logger
from deepspec.utils.hfai_suspend import SuspendController


_PRECISION_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

_SHARDING_STRATEGIES = {
    "full_shard": ShardingStrategy.FULL_SHARD,
    "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    "no_shard": ShardingStrategy.NO_SHARD,
    "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    "hybrid_shard_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
    "_hybrid_shard_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
}

_HYBRID_STRATEGIES = (
    ShardingStrategy.HYBRID_SHARD,
    ShardingStrategy._HYBRID_SHARD_ZERO2,
)


def _build_fsdp_kwargs(
    *, sharding_strategy_name: str, precision_dtype, world_size: int
) -> dict:
    sharding_strategy = _SHARDING_STRATEGIES[sharding_strategy_name]
    fsdp_kwargs = dict(
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=precision_dtype,
            buffer_dtype=precision_dtype,
        ),
        sharding_strategy=sharding_strategy,
    )
    if sharding_strategy in _HYBRID_STRATEGIES:
        devices_per_node = torch.cuda.device_count()
        fsdp_kwargs["device_mesh"] = init_device_mesh(
            "cuda",
            (world_size // devices_per_node, devices_per_node),
            mesh_dim_names=("replicate", "shard"),
        )
    return fsdp_kwargs


def _compute_gradient_accumulation_steps(
    *, world_size: int, local_batch_size: int, global_batch_size: int
) -> int:
    denom = world_size * local_batch_size
    assert global_batch_size % denom == 0, (
        "global_batch_size must be divisible by world_size * local_batch_size: "
        f"global_batch_size={global_batch_size}, world_size={world_size}, "
        f"local_batch_size={local_batch_size}"
    )
    return global_batch_size // denom


def _compute_samples_per_epoch(*, dataset_size: int, global_batch_size: int) -> int:
    samples_per_epoch = (dataset_size // global_batch_size) * global_batch_size
    assert samples_per_epoch > 0, (
        "train dataset is too small to form one full global batch: "
        f"dataset_size={dataset_size}, global_batch_size={global_batch_size}"
    )
    return samples_per_epoch


def _compute_training_schedule(
    *,
    world_size: int,
    dataset_size: int,
    local_batch_size: int,
    global_batch_size: int,
    num_train_epochs: int,
    max_train_steps=None,
) -> tuple[int, int, int, int, int, int, int]:
    gradient_accumulation_steps = _compute_gradient_accumulation_steps(
        world_size=world_size,
        local_batch_size=local_batch_size,
        global_batch_size=global_batch_size,
    )
    samples_per_epoch = _compute_samples_per_epoch(
        dataset_size=dataset_size,
        global_batch_size=global_batch_size,
    )
    per_rank_samples_per_epoch = samples_per_epoch // world_size
    micro_batches_per_epoch = per_rank_samples_per_epoch // local_batch_size
    steps_per_epoch = micro_batches_per_epoch // gradient_accumulation_steps
    if max_train_steps is None:
        resolved_max_train_steps = int(num_train_epochs) * steps_per_epoch
        resolved_num_train_epochs = int(num_train_epochs)
    else:
        resolved_max_train_steps = int(max_train_steps)
        resolved_num_train_epochs = math.ceil(
            resolved_max_train_steps / steps_per_epoch
        )
    return (
        gradient_accumulation_steps,
        samples_per_epoch,
        per_rank_samples_per_epoch,
        micro_batches_per_epoch,
        steps_per_epoch,
        resolved_max_train_steps,
        resolved_num_train_epochs,
    )


def _launch_eval(
    *,
    target_model_name_or_path: str,
    checkpoint_dir: str,
    step: int,
    tensorboard_dir: str,
    exp_name: str,
) -> None:
    print("You can use this function to launch your auto eval script!")

class BaseTrainer:
    data_collator_cls = None

    def __init__(self, local_rank, args):
        self.args = args
        self.device, self.global_rank, self.world_size = init_dist(local_rank)
        self.precision_dtype = _PRECISION_DTYPES[self.args.train.precision]
        self.checkpoint_dir_root = self.args.logging.checkpoint_dir
        self.resume_checkpoint_dir = discover_latest_checkpoint(
            self.checkpoint_dir_root
        )
        self.suspend_controller = SuspendController(device=self.device)
        self.next_micro_step = 0

        if is_global_main_process(): ensure_dir(self.checkpoint_dir_root)
        training_logger.init(
            logging_steps=int(self.args.logging.logging_steps),
            tensorboard_dir=self.args.logging.tensorboard_dir,
            mlflow_tracking_uri=getattr(self.args.logging, "mlflow_tracking_uri", None),
            mlflow_experiment_name=getattr(
                self.args.logging, "mlflow_experiment_name", None
            ),
            mlflow_run_name=self.args.exp_name,
        )

        self.draft_model, self.tokenizer = self.build_models()
        if self.resume_checkpoint_dir is not None:
            self.draft_model = load_resume_draft_model(
                resume_checkpoint_dir=self.resume_checkpoint_dir,
                draft_model=self.draft_model,
                device=self.device,
                precision_dtype=self.precision_dtype,
                global_rank=self.global_rank,
            )
        self.model = self.draft_model
        if self.args.train.torch_compile:
            print_on_local_main("Compiling training model with torch.compile...")
            self.model = torch.compile(self.model, dynamic=True)
        self.model = self._wrap_with_fsdp(self.model)

        self.train_dataset = CacheDataset(cache_dir=self.args.data.target_cache_path)
        validate_train_cache(
            train_dataset=self.train_dataset,
            draft_model=self.draft_model,
            target_model_name_or_path=self.args.model.target_model_name_or_path,
        )

        val_target_cache_path = getattr(self.args.data, "val_target_cache_path", None)
        self.val_dataset = None
        if val_target_cache_path:
            self.val_dataset = CacheDataset(cache_dir=val_target_cache_path)
            validate_train_cache(
                train_dataset=self.val_dataset,
                draft_model=self.draft_model,
                target_model_name_or_path=self.args.model.target_model_name_or_path,
            )

        (
            self.gradient_accumulation_steps,
            self.samples_per_epoch,
            self.per_rank_samples_per_epoch,
            self.micro_batches_per_epoch,
            self.steps_per_epoch,
            self.max_train_steps,
            self.args.train.num_train_epochs,
        ) = _compute_training_schedule(
            world_size=self.world_size,
            dataset_size=len(self.train_dataset),
            local_batch_size=int(self.args.train.local_batch_size),
            global_batch_size=int(self.args.train.global_batch_size),
            num_train_epochs=int(self.args.train.num_train_epochs),
            max_train_steps=self.args.train.max_train_steps,
        )

        self.optimizer = BF16Optimizer(
            self.draft_model,
            lr=float(self.args.train.lr),
            total_steps=self.max_train_steps,
            warmup_ratio=float(self.args.train.warmup_ratio),
            weight_decay=float(self.args.train.weight_decay),
        )
        if self.resume_checkpoint_dir is not None:
            resume_state = load_training_state(
                resume_checkpoint_dir=self.resume_checkpoint_dir,
                optimizer=self.optimizer,
                global_rank=self.global_rank,
                world_size=self.world_size,
                local_batch_size=int(self.args.train.local_batch_size),
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                micro_batches_per_epoch=self.micro_batches_per_epoch,
            )
            self.next_micro_step = resume_state.next_micro_step
        else:
            print_on_local_main("Training from scratch.")
        self.info_board()

    @property
    def global_step(self):
        return self.next_micro_step // self.gradient_accumulation_steps

    def info_board(self):
        print_on_local_main("***** Running training *****")
        print_on_local_main(f"  Train dataset size = {len(self.train_dataset)}")
        print_on_local_main(f"  Num train epochs = {self.args.train.num_train_epochs}")
        print_on_local_main(f"  Samples per epoch = {self.samples_per_epoch}")
        print_on_local_main(f"  Local batch size = {self.args.train.local_batch_size}")
        print_on_local_main(f"  Global batch size = {self.args.train.global_batch_size}")
        print_on_local_main(f"  Gradient accumulation steps = {self.gradient_accumulation_steps}")
        print_on_local_main(f"  Steps per epoch = {self.steps_per_epoch}")
        print_on_local_main(f"  Max train steps = {self.max_train_steps}")

    def build_models(self):
        model_args = self.args.model

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.target_model_name_or_path,
        )
        target_config = AutoConfig.from_pretrained(
            model_args.target_model_name_or_path,
        )

        draft_model = self._build_draft_model(
            target_config=target_config,
            model_args=model_args,
        )
        draft_checkpoint = getattr(model_args, "draft_model_name_or_path", None)
        if draft_checkpoint:
            self._load_pretrained_draft_weights(draft_model, draft_checkpoint)
        draft_model = draft_model.to(device=self.device, dtype=self.precision_dtype)

        # Training only uses the target checkpoint to initialize frozen draft
        # embeddings and lm_head weights. VLM-wrapped targets (e.g. Qwen3.5)
        # register under AutoModelForImageTextToText, not AutoModelForCausalLM.
        target_model_cls = (
            AutoModelForImageTextToText
            if str(target_config.model_type) in ("qwen3_5",)
            else AutoModelForCausalLM
        )
        target_model = target_model_cls.from_pretrained(
            model_args.target_model_name_or_path,
            dtype=self.precision_dtype,
        ).to(device="cpu").eval()
        target_embed_tokens = target_model.get_input_embeddings()
        target_lm_head = target_model.get_output_embeddings()
        assert (target_lm_head is not None) and (target_embed_tokens is not None)
        draft_model.initialize_embeddings_and_head(
            embed_tokens=target_embed_tokens,
            lm_head=target_lm_head,
            freeze=True,
        )
        del target_model
        return draft_model, tokenizer

    def _build_draft_model(self, *, target_config, model_args):
        raise NotImplementedError

    def _load_pretrained_draft_weights(self, draft_model, draft_model_name_or_path):
        # embed_tokens/lm_head are intentionally absent from external DFlash
        # draft checkpoints, since build_models() always overwrites them from
        # the frozen target model right after this call.
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        ckpt_path = hf_hub_download(
            repo_id=draft_model_name_or_path, filename="model.safetensors"
        )
        state_dict = load_file(ckpt_path)
        missing, unexpected = draft_model.load_state_dict(state_dict, strict=False)
        assert not unexpected, (
            f"Unexpected keys loading {draft_model_name_or_path}: {unexpected}"
        )
        # MoE submodules (Gemma4's router/experts + extra layernorms) are not
        # part of the published DFlash draft checkpoint format and are
        # expected to be freshly initialized, then learned during fine-tuning.
        moe_missing_markers = (".router.", ".experts.", "post_feedforward_layernorm_1", "post_feedforward_layernorm_2", "pre_feedforward_layernorm_2")
        allowed_missing = {"embed_tokens.weight", "lm_head.weight"}
        unexpected_missing = [
            key
            for key in missing
            if key not in allowed_missing and not any(marker in key for marker in moe_missing_markers)
        ]
        assert not unexpected_missing, (
            f"Unexpected missing keys loading {draft_model_name_or_path}: {unexpected_missing}"
        )
        print_on_local_main(
            f"Initialized draft model from pretrained checkpoint "
            f"{draft_model_name_or_path} (missing={sorted(missing)})"
        )

    def _wrap_with_fsdp(self, model):
        fsdp_kwargs = _build_fsdp_kwargs(
            sharding_strategy_name=self.args.train.sharding_strategy,
            precision_dtype=self.precision_dtype,
            world_size=self.world_size,
        )
        return FSDP(model, **fsdp_kwargs)

    def _build_train_dataloader(self, start_offset_samples=0, num_samples=None):
        sampler = StatelessResumableDistributedSampler(
            dataset=self.train_dataset,
            num_replicas=self.world_size,
            rank=self.global_rank,
            total_size=self.samples_per_epoch,
            start_global_offset_samples=start_offset_samples,
            num_samples=num_samples,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.args.train.local_batch_size),
            sampler=sampler,
            collate_fn=self.data_collator_cls(),
            num_workers=int(self.args.data.num_workers),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

    def run_batch(self, batch, tag: str = "train"):
        raise NotImplementedError

    def _build_eval_dataloader(self):
        sampler = torch.utils.data.DistributedSampler(
            self.val_dataset,
            num_replicas=self.world_size,
            rank=self.global_rank,
            shuffle=False,
            drop_last=False,
        )
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.args.train.local_batch_size),
            sampler=sampler,
            collate_fn=self.data_collator_cls(),
            num_workers=int(self.args.data.num_workers),
            pin_memory=True,
            drop_last=False,
        )

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return
        self.model.eval()
        # Drop any train metrics accumulated since the last flush so they
        # don't get mixed into this eval pass's summary.
        reset_metrics()
        dataloader = self._build_eval_dataloader()
        prefetcher = CUDAPrefetcher(dataloader, self.device)
        for batch in prefetcher:
            self.run_batch(batch, tag="val")
        summary = flush_metrics()
        training_logger.log_eval_summary(summary, global_step=self.global_step)
        self.model.train()

    def _checkpoint_kwargs(self):
        return dict(
            model=self.model,
            draft_model=self.draft_model,
            optimizer=self.optimizer,
            checkpoint_dir_root=self.checkpoint_dir_root,
            train_config=self.args,
            next_micro_step=self.next_micro_step,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            global_rank=self.global_rank,
            world_size=self.world_size,
            local_batch_size=int(self.args.train.local_batch_size),
        )

    def save_and_eval_checkpoint(self, *, is_final: bool = False):
        checkpoint_dir = save_checkpoint(**self._checkpoint_kwargs())
        self.evaluate()
        if is_global_main_process():
            # log_artifacts/log_final_checkpoint are best-effort (see
            # training_logger.log_artifacts): FSDP full-state-dict checkpoints
            # run 33GB+ and a stalled upload previously wedged training
            # indefinitely at 0% GPU util (see qwen-3-5-experiments.md).
            # Checkpoints are already durable on the shared JuiceFS PVC, so a
            # failed/timed-out mlflow upload here is not fatal.
            artifact_path = f"checkpoints/step_{self.global_step}"
            training_logger.log_artifacts(checkpoint_dir, artifact_path=artifact_path)
            if is_final:
                training_logger.log_final_checkpoint(
                    checkpoint_dir, artifact_path, step=self.global_step
                )
            _launch_eval(
                target_model_name_or_path=self.args.model.target_model_name_or_path,
                checkpoint_dir=checkpoint_dir,
                step=self.global_step,
                tensorboard_dir=self.args.logging.tensorboard_dir,
                exp_name=self.args.exp_name,
            )
        dist.barrier()
        return checkpoint_dir

    def _save_and_suspend(self):
        print_on_global_main("Saving checkpoint before suspending...")
        save_checkpoint(**self._checkpoint_kwargs())
        dist.barrier()
        if is_global_main_process():
            print_on_global_main("Going to suspend...")
            self.suspend_controller.go_suspend()
        dist.barrier()

    def train(self):
        self.model.train()
        if self.global_step == 0:
            self.evaluate()
        if self.global_step >= self.max_train_steps:
            return

        # Baseline eval before any optimizer step, logged at global_step=0,
        # so val/accept_rate@pos in MLflow has a pre-finetuning reference
        # point to diff future steps against. Skipped on resume (a mid-run
        # restart is not "before training").
        if self.next_micro_step == 0:
            self.evaluate()

        local_batch_size = int(self.args.train.local_batch_size)
        total_micro_steps = self.max_train_steps * self.gradient_accumulation_steps
        remaining_micro_steps = total_micro_steps - self.next_micro_step
        remaining_samples = remaining_micro_steps * local_batch_size

        dataloader = self._build_train_dataloader(
            start_offset_samples=self.next_micro_step * local_batch_size,
            num_samples=remaining_samples,
        )
        prefetcher = CUDAPrefetcher(dataloader, self.device)
        training_logger.start_session(global_step=self.global_step)

        with self.suspend_controller.monitoring():
            for batch in prefetcher:
                should_sync = (
                    (self.next_micro_step + 1) % self.gradient_accumulation_steps == 0
                )
                sync_context = nullcontext() if should_sync else self.model.no_sync()
                with sync_context:
                    loss = self.run_batch(batch) / self.gradient_accumulation_steps
                    loss.backward()
                self.next_micro_step += 1

                if not should_sync:
                    continue

                grad_norm = FSDP.clip_grad_norm_(
                    self.model,
                    float(self.args.train.max_grad_norm),
                )
                self.optimizer.step()
                training_logger.on_optimizer_step(
                    global_step=self.global_step,
                    next_micro_step=self.next_micro_step,
                    micro_batches_per_epoch=self.micro_batches_per_epoch,
                    max_train_steps=self.max_train_steps,
                    learning_rate=self.optimizer.get_learning_rate(),
                    grad_norm=grad_norm.item(),
                )

                eval_steps = getattr(self.args.logging, "eval_steps", None)
                if self.global_step % int(self.args.logging.checkpointing_steps) == 0:
                    self.save_and_eval_checkpoint()
                elif eval_steps and self.global_step % int(eval_steps) == 0:
                    self.evaluate()

                if self.suspend_controller.requested():
                    self._save_and_suspend()
                    return

        self.save_and_eval_checkpoint(is_final=True)

    def clean_up(self):
        training_logger.close()
        dist.barrier()
        dist.destroy_process_group()
