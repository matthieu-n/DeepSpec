import time
from typing import Optional

from torch.utils.tensorboard import SummaryWriter

from deepspec.utils import ensure_dir, is_global_main_process, print_on_global_main
from deepspec.utils.metrics import add_metric, flush, reset


_writer: Optional[SummaryWriter] = None
_logging_steps: int = 1
_session_start_wall: Optional[float] = None
_session_start_step: int = 0
_mlflow_run = None


def init(
    *,
    logging_steps: int,
    tensorboard_dir: Optional[str] = None,
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_experiment_name: Optional[str] = None,
    mlflow_run_name: Optional[str] = None,
) -> None:
    global _writer, _logging_steps, _mlflow_run
    _logging_steps = int(logging_steps)
    if tensorboard_dir is not None and is_global_main_process():
        ensure_dir(tensorboard_dir)
        _writer = SummaryWriter(tensorboard_dir)
    if mlflow_tracking_uri is not None and is_global_main_process():
        import mlflow

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment_name)
        _mlflow_run = mlflow.start_run(run_name=mlflow_run_name)


def start_session(*, global_step: int) -> None:
    global _session_start_wall, _session_start_step
    reset()
    _session_start_wall = time.time()
    _session_start_step = int(global_step)


def on_optimizer_step(
    *,
    global_step: int,
    next_micro_step: int,
    micro_batches_per_epoch: int,
    max_train_steps: int,
    learning_rate: float,
    grad_norm: float,
):
    add_metric("lr", learning_rate, reduction="last", tag="train")
    add_metric("grad_norm", grad_norm, reduction="last", tag="train")

    if global_step % _logging_steps != 0:
        return None

    summary = flush()
    if is_global_main_process():
        _write_scalars(summary, global_step=global_step)
        _print_summary(
            summary=summary,
            global_step=global_step,
            next_micro_step=next_micro_step,
            micro_batches_per_epoch=micro_batches_per_epoch,
            max_train_steps=max_train_steps,
        )
    return summary


def log_eval_summary(summary: dict, *, global_step: int) -> None:
    if not summary:
        return
    if is_global_main_process():
        _write_scalars(summary, global_step=global_step)
        loss_text = ""
        if "val/loss" in summary:
            loss_text = f" loss={summary['val/loss']:.4f}"
        print_on_global_main(f"[eval] step={global_step}{loss_text}")


def log_artifacts(local_dir: str, artifact_path: str) -> None:
    if _mlflow_run is None or not is_global_main_process():
        return
    import mlflow

    # Checkpoints are already durable on the shared PVC (BASE_CKPT_DIR) --
    # this upload is best-effort convenience only. It has no timeout in
    # mlflow's HTTP client and previously wedged training indefinitely at
    # 0% GPU util after a multi-GB checkpoint dir stalled mid-upload (see
    # qwen-3-5-experiments.md, code-intelligence run stuck at step 20).
    # MLFLOW_HTTP_REQUEST_TIMEOUT (set in the job env) bounds each HTTP
    # call so a stalled connection raises instead of hanging forever.
    try:
        mlflow.log_artifacts(local_dir, artifact_path=artifact_path)
    except Exception as exc:
        print_on_global_main(
            f"[training_logger] log_artifacts failed, continuing without "
            f"mlflow checkpoint upload: {exc!r}"
        )


def log_final_checkpoint(local_dir: str, artifact_path: str, *, step: int) -> None:
    """Log the training-final checkpoint under a stable ``final_checkpoint``
    artifact path (in addition to its per-step ``checkpoints/step_N`` copy)
    and tag the run so the final checkpoint is discoverable without having to
    know the last step number.
    """
    if _mlflow_run is None or not is_global_main_process():
        return
    import mlflow

    log_artifacts(local_dir, artifact_path="final_checkpoint")
    mlflow.set_tag("final_checkpoint_step", str(step))
    mlflow.set_tag("final_checkpoint_path", artifact_path)


def close() -> None:
    global _writer, _mlflow_run
    if _writer is not None:
        _writer.close()
        _writer = None
    if _mlflow_run is not None:
        import mlflow

        mlflow.end_run()
        _mlflow_run = None


def _write_scalars(summary, *, global_step: int) -> None:
    if _writer is not None:
        for key, value in summary.items():
            _writer.add_scalar(key, value, global_step)
    if _mlflow_run is not None:
        import mlflow

        # MLflow metric names allow only alphanumerics, '_-. :/' -- unlike
        # TensorBoard, so per-position metrics like "accept_rate@3" need '@'
        # replaced before logging here.
        mlflow_summary = {key.replace("@", "_"): value for key, value in summary.items()}
        mlflow.log_metrics(mlflow_summary, step=global_step)


def _print_summary(
    *,
    summary,
    global_step: int,
    next_micro_step: int,
    micro_batches_per_epoch: int,
    max_train_steps: int,
) -> None:
    session_start_wall = _session_start_wall
    if session_start_wall is None:
        session_start_wall = time.time()
    current_epoch = next_micro_step // micro_batches_per_epoch + 1
    session_elapsed = time.time() - session_start_wall
    completed_session_steps = global_step - _session_start_step
    remaining_steps = max(max_train_steps - global_step, 0)
    remaining_min = (
        session_elapsed * remaining_steps / max(completed_session_steps, 1)
    ) / 60
    loss_text = ""
    if "train/loss" in summary:
        loss_text = f" loss={summary['train/loss']:.4f}"
    print_on_global_main(
        f"epoch={current_epoch} "
        f"step={global_step}/{max_train_steps}"
        f"{loss_text} "
        f"| elapsed={session_elapsed / 60:.1f}min"
        f" | remaining={remaining_min:.1f}min"
    )
