import argparse
from dataclasses import dataclass
import json
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepspec.data import ConversationCollator
from deepspec.data.target_cache_dataset import (
    AsyncTargetCacheWriter,
    LocalCacheWriteSummary,
    atomic_json_dump,
    build_global_target_cache_shard_map,
    build_target_cache_manifest,
    cleanup_target_cache_tmp_dir,
    compute_local_sample_range,
    finalize_target_cache_index,
    load_local_cache_write_summary,
    prepare_target_cache_output_dir,
    rename_local_target_cache_shards,
    write_target_cache_manifest,
)
from deepspec.data.jsonl_dataset import JsonLineDataset
from deepspec.utils import (
    CustomJSONEncoder,
    get_git_diff,
    get_git_sha,
    init_dist,
    is_global_main_process,
    load_config,
    main_process_first,
    parse_opts_to_config,
    print_on_global_main,
    print_on_local_main,
    seed_all,
)

os.environ["USE_TORCH"] = "true"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# PyTorch 2.10 Inductor still reads the legacy allow_tf32 flag while compiling.
torch.set_float32_matmul_precision("high")

# TEMPORARY: see DSPARK_DEBUG_NAN in DeepSpec's deepspec/modeling/dspark/gemma4/modeling.py.
# Traces which target-model layer first produces NaN/Inf, to root-cause NaN
# loss on from-scratch training. Remove once the NaN root cause is fixed.
_DEBUG_NAN = os.environ.get("DSPARK_DEBUG_NAN") == "1"


def _debug_nan(tag: str, tensor: torch.Tensor) -> None:
    if not _DEBUG_NAN:
        return
    t = tensor.detach().float()
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    flag = "BAD" if (has_nan or has_inf) else "ok"
    finite = t[torch.isfinite(t)]
    lo = finite.min().item() if finite.numel() else float("nan")
    hi = finite.max().item() if finite.numel() else float("nan")
    print(
        f"[DSPARK_DEBUG_NAN] {flag} {tag}: nan={has_nan} inf={has_inf} "
        f"finite_min={lo:.4g} finite_max={hi:.4g}",
        flush=True,
    )
    if has_nan or has_inf:
        per_sample_bad = (
            (~torch.isfinite(t)).reshape(t.shape[0], -1).any(dim=-1).tolist()
            if t.dim() >= 2
            else None
        )
        if per_sample_bad is not None:
            print(f"[DSPARK_DEBUG_NAN] {tag}: per_sample_bad={per_sample_bad}", flush=True)


@dataclass(frozen=True)
class TargetForwardResult:
    target_hidden_states: torch.Tensor
    target_last_hidden_states: torch.Tensor


_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _dequantize_fp8_target_weights_(model: torch.nn.Module) -> int:
    # TEMPORARY: RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic's quantization_config
    # sets format="dense" (FP8 needs no bit-packing), so transformers'
    # CompressedTensorsHfQuantizer._process_model_after_weight_loading skips
    # ModelCompressor.decompress_model() entirely (it's gated on
    # is_quantization_compressed, which is False for dense format). Quantized
    # Linear weights are left as raw fp8 tensors with a `weight_scale` buffer
    # and compressed_tensors' generic calibration forward hook still attached
    # -- fine for a real fp8 GEMM kernel (vLLM/SGLang), but that hook's plain
    # F.linear(bf16_input, fp8_weight) crashes with a dtype mismatch here.
    # Dequantize every affected Linear's weight to bf16 in place and drop the
    # hook (an instance-attribute override of `forward`) so it falls back to
    # plain nn.Linear.forward on real bf16 x bf16 matmuls.
    num_dequantized = 0
    for module in model.modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        weight_scale = getattr(module, "weight_scale", None)
        if weight_scale is None or module.weight.dtype not in _FP8_DTYPES:
            continue
        dequantized = module.weight.detach().to(torch.float32)
        zero_point = getattr(module, "weight_zero_point", None)
        if zero_point is not None:
            dequantized = dequantized - zero_point.detach().to(torch.float32)
        dequantized = dequantized * weight_scale.detach().to(torch.float32)
        module.weight = torch.nn.Parameter(
            dequantized.to(torch.bfloat16), requires_grad=False
        )
        module.__dict__.pop("forward", None)
        num_dequantized += 1
    return num_dequantized


def _get_target_backbone(target_model):
    model_type = str(target_model.config.model_type)
    if model_type in ("gemma4", "gemma4_unified", "qwen3_5"):
        if hasattr(target_model, "language_model"):
            return target_model.language_model
        if hasattr(target_model, "model") and hasattr(target_model.model, "language_model"):
            return target_model.model.language_model
        assert False, f"{model_type} target model must expose a text language_model."
    return getattr(target_model, "model", target_model)


def _get_target_hidden_size(target_model) -> int:
    model_type = str(target_model.config.model_type)
    if model_type in ("gemma4", "gemma4_unified", "qwen3_5"):
        return int(target_model.config.text_config.hidden_size)
    return int(target_model.config.hidden_size)


def _get_hook_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"Unsupported target hook output type: {type(output)!r}")


def run_target_forward_with_hooks(
    *,
    target_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_layer_ids,
):
    backbone = _get_target_backbone(target_model)
    layer_modules = backbone.layers
    target_layer_ids = [int(layer_id) for layer_id in target_layer_ids]
    captured_hidden_states = {}
    handles = []

    def capture_layer(layer_id: int):
        def hook(_module, _inputs, output):
            # .clone() (not just .detach()) so a later in-place op elsewhere in the
            # forward pass can't retroactively corrupt this snapshot before it's
            # printed post-hoc by _debug_nan() below.
            captured_hidden_states[layer_id] = _get_hook_tensor(output).detach().clone()

        return hook

    # TEMPORARY: trace every layer beyond the configured target_layer_ids, plus the
    # final norm's input/output, to find where NaN/Inf appears between the last
    # captured layer and target_output.last_hidden_state. See _DEBUG_NAN above.
    extra_debug_layer_ids = []
    norm_debug = {}

    def norm_pre_hook(_module, inputs):
        norm_debug["pre"] = inputs[0].detach()

    def norm_hook(_module, _inputs, output):
        norm_debug["post"] = output.detach()

    # TEMPORARY: the NaN onset layer moves between runs (layer2 in one run,
    # layer4 in the next) depending on the freshly-regenerated training data,
    # so hardcoding sub-module hooks on a single layer index misses it. Trace
    # every decoder layer's sub-modules, in real time (not deferred), so
    # whichever layer first produces NaN/Inf in a given run is caught.
    def make_submodule_hook(tag: str):
        def hook(_module, _inputs, output):
            _debug_nan(tag, _get_hook_tensor(output))

        return hook

    def make_router_hook(layer_id: int):
        def hook(_module, _inputs, output):
            router_probabilities, top_k_weights, top_k_index = output
            _debug_nan(f"target.layer{layer_id}.router_probabilities", router_probabilities)
            _debug_nan(f"target.layer{layer_id}.router_top_k_weights", top_k_weights.float())

        return hook

    # TEMPORARY: router_probabilities (softmax output) is the very FIRST bad
    # tensor observed (as early as layer0), even though the router's input
    # (the same finite post-attention residual the clean mlp branch also
    # reads) is finite. Hook router.proj directly to see whether the raw
    # pre-softmax expert_scores logits are already NaN/Inf -- softmax itself
    # is numerically stable (subtracts max internally) and shouldn't produce
    # NaN from finite input, so if expert_scores is already bad, the bug is
    # in router.proj's matmul (possibly its FP8 dequant/GEMM), not softmax.
    def make_router_proj_hook(layer_id: int):
        def hook(module, inputs, output):
            _debug_nan(f"target.layer{layer_id}.router_proj_out", output)
            # TEMPORARY: router_proj_in is confirmed small (+-30) and
            # router.proj.weight was confirmed small (+-0.15) at setup time,
            # yet router_proj_out is ~1e37/inf -- mathematically impossible
            # for a plain matmul of those magnitudes (hidden_dim=2816 caps it
            # around +-5000). Recompute the matmul manually in fp32 using the
            # EXACT tensors captured at call time (not a separately-timed
            # setup-time read) to tell whether (a) the weight is already
            # huge at call time despite reading small at setup (would point
            # at in-place corruption/aliasing between setup and forward), or
            # (b) the manual recompute is clean (would prove self.proj's real
            # forward isn't doing plain matmul at all -- e.g. a quantized
            # GEMM kernel with a dequant-scale bug).
            if layer_id == 0:
                x = inputs[0].detach()
                w = module.weight.detach()
                print(
                    f"[DSPARK_DEBUG_NAN] target.layer0.router.proj CALL-TIME "
                    f"type={type(module)!r} mro={[c.__name__ for c in type(module).__mro__]} "
                    f"weight_dtype={w.dtype} weight_shape={tuple(w.shape)} "
                    f"input_dtype={x.dtype} input_shape={tuple(x.shape)}",
                    flush=True,
                )
                quant_attrs = [
                    a for a in dir(module) if "scale" in a.lower() or "zero_point" in a.lower()
                ]
                print(
                    f"[DSPARK_DEBUG_NAN] target.layer0.router.proj quant_attrs={quant_attrs}",
                    flush=True,
                )
                for attr in quant_attrs:
                    val = getattr(module, attr)
                    if isinstance(val, torch.Tensor):
                        _debug_nan(f"target.layer0.router.proj.{attr}", val.float())
                    else:
                        print(
                            f"[DSPARK_DEBUG_NAN] target.layer0.router.proj.{attr} = {val!r}",
                            flush=True,
                        )
                _debug_nan("target.layer0.router.proj.weight_at_call", w.float())
                manual = torch.matmul(x.float(), w.float().t())
                _debug_nan(
                    "target.layer0.router_proj_out_manual_recompute", manual
                )

        return hook

    # TEMPORARY: router_proj_out is already ~1e37-1e38 (near bf16 overflow)
    # even in batches where router.proj.weight is confirmed small/normal
    # (+-0.15) -- but Gemma4RMSNorm's _norm() is scale-invariant by
    # construction (x * rsqrt(mean(x**2)+eps) always yields ~unit RMS per
    # token, computed in fp32), and router.scale is a uniform ~30-34 gain
    # times a fixed scalar_root_size (~0.019), so normed*scale*scalar_root
    # should never exceed roughly hidden_dim**0.5 * 34 * 0.019 ~= 34 in the
    # worst case. There is no legitimate path from that math to a 1e37
    # matmul output. Hook router's raw input, router.norm's own output, and
    # router.proj's actual input (post scale-multiply, pre-matmul) to find
    # exactly which step introduces the explosion.
    def make_router_pre_hook(layer_id: int):
        def hook(_module, inputs):
            _debug_nan(f"target.layer{layer_id}.router_raw_input", inputs[0])

        return hook

    def make_router_norm_hook(layer_id: int):
        def hook(_module, _inputs, output):
            _debug_nan(f"target.layer{layer_id}.router_norm_out", output)

        return hook

    def make_router_proj_pre_hook(layer_id: int):
        def hook(_module, inputs):
            _debug_nan(f"target.layer{layer_id}.router_proj_in", inputs[0])

        return hook

    def make_realtime_pre_hook(layer_id: int):
        def hook(_module, inputs):
            _debug_nan(f"target.layer{layer_id}.realtime_raw_input", inputs[0])

        return hook

    try:
        if -1 in target_layer_ids:
            handles.append(
                backbone.embed_tokens.register_forward_hook(capture_layer(-1))
            )
        for layer_id in target_layer_ids:
            if layer_id < 0:
                continue
            handles.append(
                layer_modules[layer_id].register_forward_hook(capture_layer(layer_id))
            )
        if _DEBUG_NAN:
            covered = set(target_layer_ids)
            extra_debug_layer_ids = [
                layer_id
                for layer_id in range(len(layer_modules))
                if layer_id not in covered
            ]
            for layer_id in extra_debug_layer_ids:
                handles.append(
                    layer_modules[layer_id].register_forward_hook(
                        capture_layer(layer_id)
                    )
                )
            handles.append(backbone.norm.register_forward_pre_hook(norm_pre_hook))
            handles.append(backbone.norm.register_forward_hook(norm_hook))

            for layer_id, layer in enumerate(layer_modules):
                handles.append(
                    layer.register_forward_pre_hook(make_realtime_pre_hook(layer_id))
                )
                handles.append(
                    layer.input_layernorm.register_forward_hook(
                        make_submodule_hook(f"target.layer{layer_id}.input_layernorm_out")
                    )
                )
                handles.append(
                    layer.self_attn.register_forward_hook(
                        make_submodule_hook(f"target.layer{layer_id}.self_attn_out")
                    )
                )
                handles.append(
                    layer.post_attention_layernorm.register_forward_hook(
                        make_submodule_hook(f"target.layer{layer_id}.post_attention_layernorm_out")
                    )
                )
                handles.append(
                    layer.pre_feedforward_layernorm.register_forward_hook(
                        make_submodule_hook(f"target.layer{layer_id}.pre_feedforward_layernorm_out")
                    )
                )
                handles.append(
                    layer.mlp.register_forward_hook(
                        make_submodule_hook(f"target.layer{layer_id}.mlp_out")
                    )
                )
                if layer.enable_moe_block:
                    handles.append(
                        layer.post_feedforward_layernorm_1.register_forward_hook(
                            make_submodule_hook(
                                f"target.layer{layer_id}.post_feedforward_layernorm_1_out"
                            )
                        )
                    )
                    handles.append(
                        layer.router.register_forward_hook(make_router_hook(layer_id))
                    )
                    handles.append(
                        layer.router.register_forward_pre_hook(
                            make_router_pre_hook(layer_id)
                        )
                    )
                    handles.append(
                        layer.router.norm.register_forward_hook(
                            make_router_norm_hook(layer_id)
                        )
                    )
                    handles.append(
                        layer.router.proj.register_forward_pre_hook(
                            make_router_proj_pre_hook(layer_id)
                        )
                    )
                    handles.append(
                        layer.router.proj.register_forward_hook(
                            make_router_proj_hook(layer_id)
                        )
                    )
                    if layer_id == 0:
                        print(
                            f"[DSPARK_DEBUG_NAN] target.layer0.router.proj type="
                            f"{type(layer.router.proj)!r} weight_dtype="
                            f"{layer.router.proj.weight.dtype}",
                            flush=True,
                        )
                        _debug_nan(
                            "target.layer0.router.proj.weight",
                            layer.router.proj.weight.float(),
                        )
                        _debug_nan(
                            "target.layer0.router.scale", layer.router.scale.float()
                        )
                        _debug_nan(
                            "target.layer0.router.per_expert_scale",
                            layer.router.per_expert_scale.float(),
                        )
                    handles.append(
                        layer.pre_feedforward_layernorm_2.register_forward_hook(
                            make_submodule_hook(
                                f"target.layer{layer_id}.pre_feedforward_layernorm_2_out"
                            )
                        )
                    )
                    handles.append(
                        layer.experts.register_forward_hook(
                            make_submodule_hook(f"target.layer{layer_id}.experts_out")
                        )
                    )
                    handles.append(
                        layer.post_feedforward_layernorm_2.register_forward_hook(
                            make_submodule_hook(
                                f"target.layer{layer_id}.post_feedforward_layernorm_2_out"
                            )
                        )
                    )
                handles.append(
                    layer.post_feedforward_layernorm.register_forward_hook(
                        make_submodule_hook(f"target.layer{layer_id}.post_feedforward_layernorm_out")
                    )
                )

        with torch.no_grad():
            _debug_nan("target.input_ids", input_ids.float())
            _debug_nan("target.attention_mask", attention_mask.float())
            target_output = target_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
            )
            for layer_id in target_layer_ids:
                _debug_nan(f"target.layer{layer_id}", captured_hidden_states[layer_id])
            for layer_id in extra_debug_layer_ids:
                _debug_nan(f"target.layer{layer_id}", captured_hidden_states[layer_id])
            if _DEBUG_NAN:
                _debug_nan("target.norm_input", norm_debug["pre"])
                _debug_nan("target.norm_output", norm_debug["post"])
            target_last_hidden_states = target_output.last_hidden_state.detach()
            _debug_nan("target.last_hidden_state", target_last_hidden_states)
            if _DEBUG_NAN:
                bad_pos = ~torch.isfinite(target_last_hidden_states).all(dim=-1)
                is_pad = attention_mask == 0
                print(
                    "[DSPARK_DEBUG_NAN] target.last_hidden_state bad_at_pad="
                    f"{(bad_pos & is_pad).sum().item()} bad_at_real="
                    f"{(bad_pos & ~is_pad).sum().item()} total_pad="
                    f"{is_pad.sum().item()} total_real={(~is_pad).sum().item()}",
                    flush=True,
                )
            target_hidden_states = torch.cat(
                [captured_hidden_states[layer_id] for layer_id in target_layer_ids],
                dim=-1,
            )
    finally:
        for handle in handles:
            handle.remove()
        captured_hidden_states.clear()

    return TargetForwardResult(
        target_hidden_states=target_hidden_states,
        target_last_hidden_states=target_last_hidden_states,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--opts", action="append", default=[])
    parser.add_argument(
        "--train-data-path",
        action="append",
        required=True,
        help="Training JSONL path. Repeat this argument to use multiple files.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-loss-tokens", type=int, default=14)
    parser.add_argument("--max-shard-bytes", type=int, default=64 * 1024**3)
    parser.add_argument("--local-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    cli_args = parser.parse_args()
    config = parse_opts_to_config(cli_args.opts, load_config(cli_args.config))
    return cli_args, config


def _write_manifest(
    *,
    output_dir: str,
    config,
    train_data_paths,
    target_layer_ids,
    hidden_size: int,
    min_loss_tokens: int,
    shards,
):
    num_samples = sum(
        int(
            load_local_cache_write_summary(
                os.path.join(output_dir, "_tmp", f"rank_{rank}")
            )["num_local_samples"]
        )
        for rank in range(dist.get_world_size())
    )
    manifest = build_target_cache_manifest(
        num_samples=num_samples,
        shards=shards,
        target_layer_ids=target_layer_ids,
        hidden_size=hidden_size,
        extra_fields={
            "target_model_name_or_path": str(config.model.target_model_name_or_path),
            "source_jsonl_paths": [str(path) for path in train_data_paths],
            "chat_template": str(config.data.chat_template),
            "max_length": int(config.data.max_length),
            "min_loss_tokens": int(min_loss_tokens),
            "project_name": (
                str(config.get("project_name"))
                if config.get("project_name") is not None
                else None
            ),
            "exp_name": (
                str(config.get("exp_name"))
                if config.get("exp_name") is not None
                else None
            ),
            "git_sha": str(get_git_sha()),
        },
    )
    write_target_cache_manifest(output_dir=output_dir, manifest=manifest)


def _print_prepare_progress(*, global_rank: int, processed_samples: int, total_samples: int):
    print(
        f"[prepare rank {global_rank}] {processed_samples}/{total_samples} samples",
        flush=True,
    )


def main(local_rank: int):
    cli_args, config = parse_args()
    train_data_paths = list(cli_args.train_data_path)
    target_layer_ids = [int(layer_id) for layer_id in config.model.target_layer_ids]
    min_loss_tokens = int(cli_args.min_loss_tokens)
    seed_all(int(config.seed))
    device, global_rank, world_size = init_dist(local_rank)
    output_dir = os.path.abspath(cli_args.output_dir)
    print_on_local_main(json.dumps(config, indent=4, cls=CustomJSONEncoder), flush=True)
    print_on_local_main(
        json.dumps(
            {
                "train_data_path": train_data_paths,
                "output_dir": output_dir,
                "target_layer_ids": target_layer_ids,
                "min_loss_tokens": min_loss_tokens,
                "max_shard_bytes": int(cli_args.max_shard_bytes),
                "local_batch_size": int(cli_args.local_batch_size),
                "num_workers": int(cli_args.num_workers),
            },
            indent=4,
        ),
        flush=True,
    )
    if global_rank == 0:
        prepare_target_cache_output_dir(output_dir)
    dist.barrier()

    rank_dir = os.path.join(output_dir, "_tmp", f"rank_{global_rank}")
    os.makedirs(rank_dir, exist_ok=True)

    with main_process_first():
        dataset = JsonLineDataset(data_paths=train_data_paths)

    local_start, local_end = compute_local_sample_range(
        num_samples=len(dataset),
        rank=global_rank,
        world_size=world_size,
    )
    local_total_samples = local_end - local_start

    local_subset = Subset(dataset, range(local_start, local_end))
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.target_model_name_or_path,
    )
    # gemma4's sliding-window (window=1024) attention layers hit a known
    # SDPA mask-generation issue that yields fully-masked rows near the
    # window boundary -- softmax over an all -inf row is 0/0 = NaN, which
    # then cascades through the residual stream. eager attention builds
    # the mask correctly and avoids it.
    #
    # The FP8-quantized MoE experts default to the "grouped_mm" dispatch
    # (transformers.integrations.finegrained_fp8.fp8_grouped_mm_experts_forward),
    # which sorts tokens by expert and relies on a post-hoc sentinel-row mask
    # to zero out uninitialized rows from the grouped GEMM. On this checkpoint
    # that NaN leaks past the mask: layer-by-layer tracing (DSPARK_DEBUG_NAN)
    # shows every decoder layer's output is finite through layer 1, then 100%
    # NaN starting at layer 2 and for every layer after, regardless of padding
    # (confirmed even on batches with zero padding tokens). Forcing the eager
    # per-expert loop (transformers.integrations.finegrained_fp8.FP8Experts.forward)
    # avoids the grouped-mm sentinel-masking path entirely.
    #
    # ROOT CAUSE of the actual NaN (forcing eager above did not fix it): this
    # checkpoint's config.json quantization_config.ignore explicitly excludes
    # every `model.language_model.layers.{i}.router.proj` from FP8
    # quantization (confirmed both in the checkpoint's config.json and in its
    # llm-compressor recipe.yaml: `ignore: [..., 're:.*router', ...]`) --
    # router.proj is meant to stay a plain, unquantized bf16 nn.Linear.
    # Loading with `AutoModel.from_pretrained` resolves to the bare `Gemma4Model`
    # backbone, whose module tree is rooted one level shallower than the
    # wrapper the ignore-list paths are written against (runtime path
    # `language_model.layers.{i}.router.proj` vs. the checkpoint's
    # `model.language_model.layers.{i}.router.proj`). compressed_tensors'
    # `apply_quantization_config` ignore matching is a plain string/regex
    # match against module names, so with `AutoModel` every single ignore
    # entry silently fails to match, and router.proj gets treated as a
    # generic quantization target: it's left with an uninitialized
    # `weight_scale` buffer (no such tensor exists in the checkpoint for a
    # module that was never meant to be quantized) that is garbage/NaN, and
    # the compressed_tensors dequant hook then corrupts router.proj's real
    # weight with that NaN scale on first use -- reproduced deterministically
    # offline (CPU, meta device, no checkpoint download) by diffing
    # `AutoModel.from_pretrained` against `AutoModelForCausalLM.from_pretrained`
    # against this exact checkpoint's quantization_config.
    # Loading via `AutoModelForCausalLM` resolves to the full
    # `Gemma4ForConditionalGeneration` wrapper instead, whose module tree
    # root matches the checkpoint's ignore-list paths exactly, so
    # quantization is applied correctly and router.proj stays untouched.
    # `.model` then extracts the same `Gemma4Model` backbone object that
    # `AutoModel.from_pretrained` used to hand back directly, so nothing
    # downstream (forward call, `_get_target_backbone`, etc.) needs to change.
    target_model = AutoModelForCausalLM.from_pretrained(
        config.model.target_model_name_or_path,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        experts_implementation="eager",
    ).model.to(device=device).eval()
    num_fp8_dequantized = _dequantize_fp8_target_weights_(target_model)
    if _DEBUG_NAN:
        print(
            f"[DSPARK_DEBUG_NAN] dequantized {num_fp8_dequantized} FP8 Linear "
            "weights to bf16 (dense-format checkpoint, transformers skipped "
            "decompress_model)",
            flush=True,
        )
    target_hidden_size = _get_target_hidden_size(target_model)
    train_collator = ConversationCollator(
        tokenizer=tokenizer,
        chat_template=config.data.chat_template,
        max_length=config.data.max_length,
        min_loss_tokens=min_loss_tokens,
    )
    dataloader = DataLoader(
        local_subset,
        batch_size=int(cli_args.local_batch_size),
        collate_fn=train_collator,
        num_workers=int(cli_args.num_workers),
        pin_memory=True,
        drop_last=False,
    )
    writer = AsyncTargetCacheWriter(
        rank_dir=rank_dir,
        max_shard_bytes=int(cli_args.max_shard_bytes),
        max_queue_size=int(cli_args.local_batch_size) * 4,
    )

    processed_local_samples = 0
    last_progress_printed = 0
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                processed_local_samples = min(
                    (batch_idx + 1) * int(cli_args.local_batch_size),
                    local_total_samples,
                )
                should_print_progress = (
                    processed_local_samples - last_progress_printed >= 100
                    or processed_local_samples == local_total_samples
                )
                if batch is None:
                    if should_print_progress:
                        _print_prepare_progress(
                            global_rank=global_rank,
                            processed_samples=processed_local_samples,
                            total_samples=local_total_samples,
                        )
                        last_progress_printed = processed_local_samples
                    continue
                batch = {
                    key: value.to(device, non_blocking=True)
                    for key, value in batch.items()
                }
                target_result = run_target_forward_with_hooks(
                    target_model=target_model,
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    target_layer_ids=target_layer_ids,
                )
                seq_lens = batch["attention_mask"].sum(dim=1).tolist()
                for sample_idx_in_batch, seq_len in enumerate(seq_lens):
                    seq_len = int(seq_len)
                    writer.write_sample(
                        input_ids=batch["input_ids"][sample_idx_in_batch, :seq_len],
                        attention_mask=batch["attention_mask"][
                            sample_idx_in_batch, :seq_len
                        ],
                        loss_mask=batch["loss_mask"][sample_idx_in_batch, :seq_len],
                        target_hidden_states=target_result.target_hidden_states[
                            sample_idx_in_batch, :seq_len
                        ],
                        target_last_hidden_states=target_result.target_last_hidden_states[
                            sample_idx_in_batch, :seq_len
                        ],
                    )
                if should_print_progress:
                    _print_prepare_progress(
                        global_rank=global_rank,
                        processed_samples=processed_local_samples,
                        total_samples=local_total_samples,
                    )
                    last_progress_printed = processed_local_samples
    finally:
        writer.close()
    del target_model
    torch.cuda.empty_cache()
    dataset.close()
    summary = LocalCacheWriteSummary(
        global_rank=global_rank,
        source_sample_start=local_start,
        source_sample_end=local_end,
        num_local_samples=writer.num_local_samples,
        num_local_shards=len(writer.local_shard_files),
        local_shard_files=list(writer.local_shard_files),
    )
    atomic_json_dump(summary.to_json(), os.path.join(rank_dir, "summary.json"))
    dist.barrier()

    shard_map = None
    summaries = None
    if is_global_main_process():
        summaries = [
            load_local_cache_write_summary(
                os.path.join(output_dir, "_tmp", f"rank_{rank}")
            )
            for rank in range(world_size)
        ]
        shard_map, shards = build_global_target_cache_shard_map(summaries)
    broadcast_payload = [shard_map]
    dist.broadcast_object_list(broadcast_payload, src=0)
    shard_map = broadcast_payload[0]
    local_summary = load_local_cache_write_summary(rank_dir)
    rename_local_target_cache_shards(
        output_dir=output_dir,
        rank_dir=rank_dir,
        summary=local_summary,
        shard_map=shard_map,
    )
    dist.barrier()

    if is_global_main_process():
        assert summaries is not None
        num_valid_samples = finalize_target_cache_index(
            output_dir=output_dir,
            summaries=summaries,
            shard_map=shard_map,
        )
        _write_manifest(
            output_dir=output_dir,
            config=config,
            train_data_paths=train_data_paths,
            target_layer_ids=target_layer_ids,
            hidden_size=target_hidden_size,
            min_loss_tokens=min_loss_tokens,
            shards=shards,
        )
        cleanup_target_cache_tmp_dir(output_dir)
        print_on_global_main(
            f"Prepared target cache at {output_dir} with "
            f"{num_valid_samples}/{len(dataset)} valid samples."
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    if os.path.exists(".git"):
        print(f"git status:", "\n\n".join(get_git_sha(detail_info=True)))
        print("git diff:", get_git_diff())
    torch.multiprocessing.spawn(main, nprocs=torch.cuda.device_count())
