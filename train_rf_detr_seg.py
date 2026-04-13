#!/usr/bin/env python3

import datetime
import os
import time
import copy
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple

# Must be called BEFORE importing rfdetr or transformers
from utils.distributed_utils import setup_cluster_env, get_rank, rank_zero_print

setup_cluster_env()

import hydra
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from hydra.utils import to_absolute_path
from lightning.pytorch.profilers import AdvancedProfiler, SimpleProfiler
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from rfdetr.assets.model_weights import ModelWeights, download_pretrain_weights
from rfdetr import RFDETRSegSmall, RFDETRSegMedium, RFDETRSegLarge
from rfdetr.models.lwdetr import build_criterion_and_postprocessors

from data.rf_detr_data_module import RFDETRDataModule
from models.rf_detr_seg_lightning_module import RFDETRSegLightningModule
from utils.test_only_checkpoint_restore import (
    _load_ckpt,
    _load_selected_weights,
    _merge_test_only_config_from_ckpt,
    _resolve_ckpt_path,
    _select_eval_weights_source,
)
from utils.train_utils import BackupToNASCallback

warnings.filterwarnings("ignore", category=FutureWarning)
OmegaConf.register_new_resolver(
    "extract_name", lambda path: path.split("/")[-1], replace=True
)
OmegaConf.register_new_resolver("oc.eval", eval, replace=True)

torch.set_float32_matmul_precision("medium")

_DEFAULT_PRETRAIN_WEIGHTS_BY_SIZE: Dict[str, str] = {
    "small": "rf-detr-seg-small.pth",
    "medium": "rf-detr-seg-medium.pth",
    "large": "rf-detr-seg-large.pth",
}

_PRETRAIN_DOWNLOAD_WAIT_TIMEOUT_SEC = 300
_PRETRAIN_DOWNLOAD_WAIT_POLL_SEC = 2


def _extract_checkpoint_num_queries(checkpoint: Dict[str, Any]) -> Optional[int]:
    args_obj = checkpoint.get("args")
    if hasattr(args_obj, "num_queries"):
        try:
            return int(args_obj.num_queries)
        except (TypeError, ValueError):
            return None
    if isinstance(args_obj, dict) and "num_queries" in args_obj:
        try:
            return int(args_obj["num_queries"])
        except (TypeError, ValueError):
            return None
    return None


def _init_rows_from_pretrained_stats(
    base_tensor: torch.Tensor, n_rows: int
) -> torch.Tensor:
    if n_rows <= 0:
        return base_tensor.new_empty((0, base_tensor.shape[1]))
    mean = base_tensor.mean(dim=0, keepdim=True)
    std = base_tensor.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (
        mean + torch.randn(n_rows, base_tensor.shape[1], dtype=base_tensor.dtype) * std
    )


def _build_query_compatible_pretrain_weights(
    pretrain_weights: str,
    requested_num_queries: Optional[int],
    run_save_dir: str,
) -> str:
    if requested_num_queries is None:
        return pretrain_weights

    checkpoint = torch.load(pretrain_weights, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        rank_zero_print(
            "[Startup] Query-compat skipped: pretrain checkpoint is not a dict."
        )
        return pretrain_weights

    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict):
        rank_zero_print(
            "[Startup] Query-compat skipped: checkpoint has no 'model' state_dict."
        )
        return pretrain_weights

    ref_key = "refpoint_embed.weight"
    query_key = "query_feat.weight"
    if ref_key not in model_state or query_key not in model_state:
        rank_zero_print(
            "[Startup] Query-compat skipped: checkpoint missing query tensors."
        )
        return pretrain_weights

    ref_tensor = model_state[ref_key]
    query_tensor = model_state[query_key]
    if (
        not isinstance(ref_tensor, torch.Tensor)
        or not isinstance(query_tensor, torch.Tensor)
        or ref_tensor.ndim != 2
        or query_tensor.ndim != 2
        or ref_tensor.shape[0] != query_tensor.shape[0]
    ):
        rank_zero_print(
            "[Startup] Query-compat skipped: unexpected query tensor shape(s)."
        )
        return pretrain_weights

    source_num_queries = _extract_checkpoint_num_queries(checkpoint)
    if source_num_queries is None or source_num_queries <= 0:
        rank_zero_print(
            "[Startup] Query-compat skipped: could not infer source num_queries from checkpoint args."
        )
        return pretrain_weights

    source_rows = int(ref_tensor.shape[0])
    if source_rows % source_num_queries != 0:
        rank_zero_print(
            "[Startup] Query-compat skipped: query rows are not divisible by source num_queries."
        )
        return pretrain_weights

    rows_per_query = source_rows // source_num_queries
    target_rows = int(requested_num_queries) * rows_per_query
    if target_rows <= 0 or target_rows == source_rows:
        return pretrain_weights

    new_ref = ref_tensor.new_empty((target_rows, ref_tensor.shape[1]))
    new_query = query_tensor.new_empty((target_rows, query_tensor.shape[1]))
    copy_rows = min(source_rows, target_rows)
    new_ref[:copy_rows] = ref_tensor[:copy_rows]
    new_query[:copy_rows] = query_tensor[:copy_rows]
    if target_rows > source_rows:
        extra_rows = target_rows - source_rows
        new_ref[source_rows:] = _init_rows_from_pretrained_stats(ref_tensor, extra_rows)
        new_query[source_rows:] = _init_rows_from_pretrained_stats(
            query_tensor, extra_rows
        )

    compatible_ckpt = copy.deepcopy(checkpoint)
    compatible_model_state = dict(model_state)
    compatible_model_state[ref_key] = new_ref
    compatible_model_state[query_key] = new_query
    compatible_ckpt["model"] = compatible_model_state

    if hasattr(compatible_ckpt.get("args"), "num_queries"):
        compatible_ckpt["args"].num_queries = int(requested_num_queries)

    compat_dir = os.path.join(run_save_dir, "compat_pretrain")
    os.makedirs(compat_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pretrain_weights))[0]
    compat_path = os.path.join(
        compat_dir, f"{base_name}_q{int(requested_num_queries)}.pth"
    )

    rank = get_rank()
    if rank == 0:
        torch.save(compatible_ckpt, compat_path)
    else:
        deadline = time.time() + _PRETRAIN_DOWNLOAD_WAIT_TIMEOUT_SEC
        while time.time() < deadline and not os.path.exists(compat_path):
            time.sleep(_PRETRAIN_DOWNLOAD_WAIT_POLL_SEC)

    if not os.path.exists(compat_path):
        raise FileNotFoundError(
            f"Failed to prepare query-compatible pretrain weights at '{compat_path}'."
        )

    rank_zero_print(
        "[Startup] Query-compat pretrain prepared: "
        f"source_queries={source_num_queries}, requested_queries={int(requested_num_queries)}, "
        f"rows_per_query={rows_per_query}, copied_rows={copy_rows}, total_rows={target_rows}."
    )
    rank_zero_print(f"[Startup] Using query-compatible pretrain weights: {compat_path}")
    return compat_path


def _apply_rfdetr_finetune_mode(
    nn_model: torch.nn.Module, finetune_mode: str
) -> Tuple[int, int, List[str]]:
    mode = str(finetune_mode).lower()
    named_params = list(nn_model.named_parameters())

    if mode == "full":
        for _, param in named_params:
            param.requires_grad = True
    else:
        for _, param in named_params:
            param.requires_grad = False

        allow_prefixes: List[str] = [
            "refpoint_embed.",
            "query_feat.",
            "class_embed.",
            "bbox_embed.",
            "transformer.enc_out_class_embed.",
            "transformer.enc_out_bbox_embed.",
            "segmentation_head.",
        ]
        if mode == "queries_decoder_head":
            allow_prefixes.append("transformer.decoder.")
        elif mode != "queries_head":
            raise ValueError(
                "Unsupported model.rfdetr.finetune_mode. "
                "Use one of: full, queries_head, queries_decoder_head."
            )

        for name, param in named_params:
            if any(name.startswith(prefix) for prefix in allow_prefixes):
                param.requires_grad = True

    trainable = [name for name, p in named_params if p.requires_grad]
    trainable_params = sum(p.numel() for _, p in named_params if p.requires_grad)
    total_params = sum(p.numel() for _, p in named_params)
    return trainable_params, total_params, trainable


def _maybe_raise_max_detections(
    config: DictConfig, requested_num_select: Optional[int]
) -> None:
    if requested_num_select is None:
        return
    current_max_dets = int(config.model.max_detections)
    if current_max_dets >= int(requested_num_select):
        return
    config.model.max_detections = int(requested_num_select)
    rank_zero_print(
        "[Startup] Raised model.max_detections to match high-query run: "
        f"{current_max_dets} -> {int(requested_num_select)}"
    )


def _get_profiler(config: DictConfig):
    ptype = config.training.profiler.type
    if ptype == "simple":
        return SimpleProfiler(dirpath="profiler_logs", filename="rfdetr_profile")
    if ptype == "advanced":
        return AdvancedProfiler(dirpath="profiler_logs", filename="rfdetr_profile")
    return None


def _get_model_class(size_name: str):
    size_name = str(size_name).lower()
    if size_name == "small":
        return RFDETRSegSmall
    if size_name == "medium":
        return RFDETRSegMedium
    if size_name == "large":
        return RFDETRSegLarge
    raise ValueError(
        f"Unsupported RF-DETR Seg size: {size_name}. Allowed sizes are 'small', 'medium', 'large'."
    )


def _ensure_pretrain_weights_available(pretrain_weights: str) -> str:
    weight_name = os.path.basename(pretrain_weights)
    resolved_local_path = os.path.realpath(pretrain_weights)
    known_hosted_weight = ModelWeights.from_filename(weight_name) is not None

    # Any explicit path (absolute or nested relative) is treated as custom
    # and must already exist.
    dirname = os.path.dirname(pretrain_weights)
    is_explicit_custom_path = dirname not in ("", ".")

    if is_explicit_custom_path or not known_hosted_weight:
        if os.path.exists(resolved_local_path):
            rank_zero_print(
                f"[Startup] Using RF-DETR pretrain weights at: {resolved_local_path}"
            )
            return resolved_local_path
        raise FileNotFoundError(
            "RF-DETR pretrain weights file not found. "
            f"Configured value: '{pretrain_weights}', resolved path: '{resolved_local_path}'. "
            "For custom weights, provide an existing file path. "
            "For hosted defaults, use a known filename like 'rf-detr-medium.pth'."
        )

    if os.path.exists(resolved_local_path):
        rank_zero_print(
            f"[Startup] Using RF-DETR pretrain weights at: {resolved_local_path}"
        )
        return resolved_local_path

    rank = get_rank()
    if rank == 0:
        rank_zero_print(
            f"[Startup] RF-DETR pretrain weights missing at '{resolved_local_path}'. "
            f"Auto-downloading hosted weights '{weight_name}'."
        )
        download_pretrain_weights(weight_name)
    else:
        deadline = time.time() + _PRETRAIN_DOWNLOAD_WAIT_TIMEOUT_SEC
        while time.time() < deadline and not os.path.exists(resolved_local_path):
            time.sleep(_PRETRAIN_DOWNLOAD_WAIT_POLL_SEC)
        if not os.path.exists(resolved_local_path):
            # Fallback in case rank 0 download failed or is not visible yet.
            download_pretrain_weights(weight_name)

    if not os.path.exists(resolved_local_path):
        raise FileNotFoundError(
            f"RF-DETR pretrain weights '{weight_name}' are still missing after download attempts. "
            f"Expected local path: '{resolved_local_path}'."
        )

    rank_zero_print(
        f"[Startup] Using RF-DETR pretrain weights at: {resolved_local_path}"
    )
    return resolved_local_path


def _build_rfdetr_model_kwargs(
    config: DictConfig, num_classes: int, device: str
) -> Dict[str, Any]:
    rfdetr_cfg = config.model.rfdetr
    size = str(rfdetr_cfg.size).lower()
    if size not in _DEFAULT_PRETRAIN_WEIGHTS_BY_SIZE:
        raise ValueError(f"Unsupported RF-DETR size: {size}")

    raw_pretrain_weights = rfdetr_cfg.get("pretrain_weights")
    if raw_pretrain_weights in (None, ""):
        pretrain_weights = _DEFAULT_PRETRAIN_WEIGHTS_BY_SIZE[size]
        rank_zero_print(
            f"[Startup] `model.rfdetr.pretrain_weights` is unset; using size-matched default '{pretrain_weights}'."
        )
    else:
        pretrain_weights = str(raw_pretrain_weights)
        lower_name = os.path.basename(pretrain_weights).lower()
        inferred_weight_size = next(
            (
                candidate
                for candidate in ("small", "medium", "large")
                if candidate in lower_name
            ),
            None,
        )
        if inferred_weight_size is not None and inferred_weight_size != size:
            raise ValueError(
                "RF-DETR size/weights mismatch: "
                f"`model.rfdetr.size={size}` but `model.rfdetr.pretrain_weights={pretrain_weights}` "
                f"looks like {inferred_weight_size} weights. "
                f"Use '{_DEFAULT_PRETRAIN_WEIGHTS_BY_SIZE[size]}' for size={size}, or switch size."
            )

    pretrain_weights = _ensure_pretrain_weights_available(pretrain_weights)

    kwargs: Dict[str, Any] = {
        "pretrain_weights": pretrain_weights,
        "resolution": int(config.model.input_size),
        "num_classes": int(num_classes),
        "device": device,
    }

    requested_num_queries = rfdetr_cfg.get("num_queries")
    requested_num_select = rfdetr_cfg.get("num_select")

    if size == "large":
        if requested_num_queries is not None or requested_num_select is not None:
            rank_zero_print(
                "[Startup] WARNING: RF-DETR large does not support configurable "
                "`num_queries`/`num_select`; ignoring these overrides."
            )
        return kwargs

    if requested_num_queries is not None:
        kwargs["num_queries"] = int(requested_num_queries)
        if requested_num_select is None:
            requested_num_select = int(requested_num_queries)
            rank_zero_print(
                f"[Startup] `model.rfdetr.num_select` not set; defaulting to num_queries={requested_num_select}."
            )

    if requested_num_select is not None:
        kwargs["num_select"] = int(requested_num_select)

    return kwargs


def _log_effective_model_config(size_name: str, kwargs: Dict[str, Any]) -> None:
    rank_zero_print(f"[Startup] Building RF-DETR model ({size_name})...")
    rank_zero_print("[Startup] Effective RF-DETR model config:")
    for key in sorted(kwargs.keys()):
        rank_zero_print(f"  - {key}: {kwargs[key]}")


def _build_model_to_coco_map(coco_gt, label_map: Dict[int, str]) -> Dict[int, int]:
    name_to_model_id = {str(name): int(idx) for idx, name in label_map.items()}
    model_to_coco: Dict[int, int] = {}

    if coco_gt is None or not getattr(coco_gt, "cats", None):
        for model_id in sorted(name_to_model_id.values()):
            model_to_coco[model_id] = model_id
        rank_zero_print(
            "[Startup] WARNING: COCO categories unavailable. Falling back to identity model->COCO mapping."
        )
        return model_to_coco

    unmatched_coco_names = []
    for coco_cat_id, cat_info in coco_gt.cats.items():
        cat_name = cat_info.get("name")
        if cat_name in name_to_model_id:
            model_id = name_to_model_id[cat_name]
            if model_id not in model_to_coco:
                model_to_coco[model_id] = int(coco_cat_id)
        else:
            unmatched_coco_names.append(str(cat_name))

    for model_id in sorted(name_to_model_id.values()):
        if model_id not in model_to_coco:
            model_to_coco[model_id] = model_id

    rank_zero_print(f"[Startup] RF-DETR model_to_coco mapping: {model_to_coco}")
    if unmatched_coco_names:
        rank_zero_print(
            f"[Startup] NOTE: COCO categories not present in model label_map: {sorted(set(unmatched_coco_names))}"
        )

    return model_to_coco


def _setup_logger(config: DictConfig):
    wandb_cfg = config.logging.wandb
    if not wandb_cfg.enabled:
        return None
    cfg_for_log = OmegaConf.to_container(config, resolve=True)
    return WandbLogger(
        project=wandb_cfg.project,
        name=config.run_name,
        tags=list(wandb_cfg.tags),
        notes=wandb_cfg.notes,
        group=wandb_cfg.get("group"),
        config=cfg_for_log,
        save_dir=os.getcwd(),
        reinit="finish_previous",
    )


def _setup_callbacks(config: DictConfig):
    ckpt_cfg = config.checkpointing
    ckpt_dir = os.path.join(to_absolute_path(ckpt_cfg.save_dir), "ckpts")
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="rfdetr-seg-epoch{epoch:02d}-val_segm_map{"
            + ckpt_cfg.monitor.replace("/", "_")
            + ":.4f}",
            monitor=ckpt_cfg.monitor,
            mode=ckpt_cfg.mode,
            save_top_k=ckpt_cfg.save_top_k,
            save_last=ckpt_cfg.save_last,
            every_n_epochs=ckpt_cfg.every_n_epochs,
            auto_insert_metric_name=False,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        ModelSummary(max_depth=3),
        EarlyStopping(
            monitor="val/segm_map_ema"
            if hasattr(config.model, "ema") and config.model.ema.enabled
            else "val/segm_map",
            patience=20,
            mode=ckpt_cfg.mode,
            verbose=True,
        ),
    ]

    if hasattr(config.model, "ema") and config.model.ema.enabled:
        from utils.ema import EMACallback

        warmup_steps = config.model.ema.get("warmup_steps", 0)
        tau = config.model.ema.get("tau", 2000)
        rank_zero_print(
            f"💡 EMA enabled: Adding EMACallback with decay={config.model.ema.decay}, "
            f"tau={tau}, warmup_steps={warmup_steps}"
        )
        callbacks.append(
            EMACallback(
                decay=config.model.ema.decay, tau=tau, warmup_steps=warmup_steps
            )
        )

        ema_monitor = "val/segm_map_ema"
        callbacks.append(
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="rfdetr-seg-ema-{epoch:02d}-val_segm_map_ema{"
                + ema_monitor.replace("/", "_")
                + ":.4f}",
                monitor=ema_monitor,
                mode=ckpt_cfg.mode,
                save_top_k=ckpt_cfg.save_top_k,
                save_last=False,
                every_n_epochs=ckpt_cfg.every_n_epochs,
                auto_insert_metric_name=False,
                verbose=True,
            )
        )

    if "backup_dir" in ckpt_cfg and ckpt_cfg.backup_dir:
        callbacks.append(
            BackupToNASCallback(backup_dir=to_absolute_path(ckpt_cfg.backup_dir))
        )
    return callbacks


def _resolve_run_name(config: DictConfig):
    unique_id = (
        os.environ.get("SLURM_JOB_ID")
        or os.environ.get("TORCHELASTIC_RUN_ID")
        or os.environ.get("WANDB_RUN_ID")
        or ""
    )
    timestamp = datetime.datetime.now().strftime("%H-%M")
    unique_id = (
        f"{unique_id}_{timestamp}"
        if unique_id
        else datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    )
    config.run_name = f"{config.run_name}_{unique_id}"


@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: DictConfig):
    test_only_checkpoint: Optional[Dict[str, Any]] = None
    test_only_weight_source: Optional[Literal["ema", "regular"]] = None
    early_ckpt_path: Optional[str] = None

    OmegaConf.set_struct(config, False)
    _resolve_run_name(config)

    if config.checkpointing.monitor == "val/map":
        config.checkpointing.monitor = "val/segm_map"

    hydra_cfg = HydraConfig.get()

    job_overrides = hydra_cfg.overrides.task
    for override in job_overrides:
        if "=" in override:
            key, value = override.split("=", 1)
            short_key = key.split(".")[-1]
            if short_key == "load_from_checkpoint":
                continue
            tag = f"{short_key}={value}"
            config.logging.wandb.tags.append(tag)
            rank_zero_print(f"   -> Added WandB tag: {tag}")

    if hydra_cfg.mode == RunMode.MULTIRUN:
        rank_zero_print(f"Detected Hydra Sweep (Job {hydra_cfg.job.num})")
        config.run_name = f"{config.run_name}_run{hydra_cfg.job.num}"
        if not config.logging.wandb.get("group"):
            sweep_id = os.path.basename(os.path.normpath(hydra_cfg.sweep.dir))
            config.logging.wandb.group = f"sweep_{sweep_id}"

    if config.debug:
        rank_zero_print(
            f"{'!' * 80}\n[DEBUG] Running in DEBUG/OVERFIT mode\n{'!' * 80}"
        )
        config.trainer.num_overfit_samples = 1
        config.data.eval_batch_size = config.data.batch_size
        config.run_name = f"DEBUG_{config.run_name}"

    base_save_dir = to_absolute_path(config.checkpointing.save_dir)
    run_save_dir = os.path.join(base_save_dir, config.run_name)
    config.checkpointing.save_dir = run_save_dir

    if config.test_only:
        early_ckpt_path = _resolve_ckpt_path(config, run_save_dir=run_save_dir)
        if not early_ckpt_path:
            raise ValueError(
                "test_only=true requires a valid checkpoint path. "
                "Check for typos/trailing punctuation and pass "
                "'initialization.load_from_checkpoint=/abs/path/model.ckpt'."
            )

        test_only_checkpoint = _load_ckpt(early_ckpt_path)
        config = _merge_test_only_config_from_ckpt(config, test_only_checkpoint)
        OmegaConf.set_struct(config, False)
        config.initialization.load_from_checkpoint = early_ckpt_path
        config.checkpointing.save_dir = run_save_dir

        if (
            hasattr(config, "model")
            and hasattr(config.model, "ema")
            and config.model.ema is not None
        ):
            config.model.ema.enabled = False
            rank_zero_print(
                "test_only: disabled EMA callback/branch for single selected-model evaluation."
            )

        test_only_weight_source = _select_eval_weights_source(
            early_ckpt_path, test_only_checkpoint, config=config
        )
        rank_zero_print(
            f"test_only: selected checkpoint weight source = {test_only_weight_source.upper()}"
        )

    OmegaConf.set_struct(config, True)

    eval_mode = config.get("eval_inference", {}).get("mode", "whole")
    rank_zero_print(f"--- Eval Inference Mode: {eval_mode.upper()} ---")
    if eval_mode in ["sliced", "both"]:
        rank_zero_print(OmegaConf.to_yaml(config.eval_inference.sahi))
        rank_zero_print("---------------------------")

    pl.seed_everything(config.seed, workers=True)

    dataset_path = to_absolute_path(config.data.path)
    rank_zero_print(f"[Startup] Using RF-DETR dataset path: {dataset_path}")

    label_map = {int(k): v for k, v in config.model.label_map.items()}
    class_names = [label_map[idx] for idx in sorted(label_map.keys())]

    rf_model_cls = _get_model_class(config.model.rfdetr.size)
    model_kwargs = _build_rfdetr_model_kwargs(
        config=config,
        num_classes=len(label_map),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    requested_num_queries = model_kwargs.get("num_queries")
    if "pretrain_weights" in model_kwargs:
        model_kwargs["pretrain_weights"] = _build_query_compatible_pretrain_weights(
            pretrain_weights=str(model_kwargs["pretrain_weights"]),
            requested_num_queries=(
                int(requested_num_queries)
                if requested_num_queries is not None
                else None
            ),
            run_save_dir=run_save_dir,
        )
    _maybe_raise_max_detections(
        config=config,
        requested_num_select=(
            int(model_kwargs["num_select"]) if "num_select" in model_kwargs else None
        ),
    )
    _log_effective_model_config(config.model.rfdetr.size, model_kwargs)

    rf_wrapper = rf_model_cls(**model_kwargs)

    if hasattr(rf_wrapper, "model") and hasattr(rf_wrapper.model, "args"):
        rf_wrapper.model.args.class_names = class_names
        rf_wrapper.model.args.num_classes = len(label_map)
    if hasattr(rf_wrapper.model, "class_names"):
        rf_wrapper.model.class_names = class_names

    rf_wrapper.model.reinitialize_detection_head(len(label_map))

    finetune_mode = str(config.model.rfdetr.get("finetune_mode", "queries_head"))
    trainable_params, total_params, trainable_names = _apply_rfdetr_finetune_mode(
        rf_wrapper.model.model, finetune_mode
    )
    rank_zero_print(
        "[Startup] RF-DETR finetune mode: "
        f"{finetune_mode} ({trainable_params:,}/{total_params:,} params trainable)"
    )
    if trainable_names:
        preview = trainable_names[:20]
        for name in preview:
            rank_zero_print(f"  - trainable: {name}")
        if len(trainable_names) > len(preview):
            rank_zero_print(
                f"  - ... and {len(trainable_names) - len(preview)} more trainable tensors"
            )
    else:
        rank_zero_print(
            "[Startup] WARNING: No trainable parameters selected by finetune mode."
        )

    base_args = rf_wrapper.model.args
    data_module = RFDETRDataModule(
        dataset_path=dataset_path, config=config, base_args=base_args
    )
    data_module.setup("fit")
    data_module.setup("test")

    for coco_gt in [data_module.val_coco_gt, data_module.test_coco_gt]:
        if coco_gt is not None and "info" not in coco_gt.dataset:
            coco_gt.dataset["info"] = {}

    model_to_coco = _build_model_to_coco_map(data_module.val_coco_gt, label_map)

    criterion, postprocess = build_criterion_and_postprocessors(data_module.args)
    lightning_model = RFDETRSegLightningModule(
        model=rf_wrapper.model.model,
        criterion=criterion,
        postprocess=postprocess,
        config=config,
        model_to_coco=model_to_coco,
        val_coco_gt=data_module.val_coco_gt,
        test_coco_gt=data_module.val_coco_gt
        if config.debug
        else data_module.test_coco_gt,
        val_image_root=data_module.val_image_root,
        test_image_root=data_module.test_image_root,
    )

    logger = _setup_logger(config)
    if logger:
        logger.experiment.save("models/rf_detr_seg_lightning_module.py")
        logger.watch(lightning_model, log="gradients", log_freq=500)
        rank_zero_print("✓ WandB logger watching model for gradients")

    callbacks = _setup_callbacks(config)
    profiler = _get_profiler(config)

    trainer = pl.Trainer(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
        precision=config.trainer.precision,
        strategy=config.trainer.strategy,
        max_epochs=config.trainer.max_epochs,
        log_every_n_steps=config.trainer.log_every_n_steps,
        val_check_interval=config.trainer.val_check_interval,
        gradient_clip_val=config.trainer.max_grad_norm,
        gradient_clip_algorithm=config.trainer.gradient_clip_algo,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        deterministic=config.trainer.deterministic,
        benchmark=config.trainer.benchmark,
        enable_progress_bar=True,
        callbacks=callbacks,
        logger=logger,
        overfit_batches=config.trainer.num_overfit_samples if config.debug else 0,
        limit_train_batches=config.data.limit_train_batches if not config.debug else 10,
        limit_val_batches=config.data.limit_val_batches if not config.debug else 10,
        limit_test_batches=config.data.limit_test_batches if not config.debug else 10,
        profiler=None if config.debug else profiler,
        plugins=[SLURMEnvironment(auto_requeue=False)]
        if "SLURM_JOB_ID" in os.environ
        else None,
    )

    ckpt_path = _resolve_ckpt_path(config, run_save_dir=config.checkpointing.save_dir)

    rank_zero_print(OmegaConf.to_yaml(config))
    if config.test_only:
        if not ckpt_path:
            raise ValueError(
                "test_only=true requires initialization.load_from_checkpoint"
            )
        if test_only_checkpoint is None:
            test_only_checkpoint = _load_ckpt(ckpt_path)
        if test_only_weight_source is None:
            test_only_weight_source = _select_eval_weights_source(
                ckpt_path, test_only_checkpoint, config=config
            )

        rank_zero_print(
            f"Loading {test_only_weight_source.upper()} weights for test-only evaluation..."
        )
        missing_keys, unexpected_keys = _load_selected_weights(
            lightning_model, test_only_checkpoint, test_only_weight_source
        )
        if missing_keys:
            rank_zero_print(
                f"⚠️  Missing keys during test-only load: {missing_keys[:10]} ..."
            )
        if unexpected_keys:
            rank_zero_print(
                f"⚠️  Unexpected keys during test-only load: {unexpected_keys[:10]} ..."
            )

        trainer.test(lightning_model, datamodule=data_module)
    else:
        trainer.fit(
            lightning_model,
            datamodule=data_module,
            ckpt_path=ckpt_path,
            weights_only=False,
        )

        best_path = None

        if hasattr(config.model, "ema") and config.model.ema.enabled:
            for cb in trainer.callbacks:
                if isinstance(cb, ModelCheckpoint) and cb.monitor == "val/segm_map_ema":
                    if cb.best_model_path:
                        best_path = cb.best_model_path
                        rank_zero_print(
                            f"🎯 Selected BEST EMA checkpoint (monitor: {cb.monitor}): {best_path}"
                        )
                    break

        if not best_path:
            for cb in trainer.callbacks:
                if (
                    isinstance(cb, ModelCheckpoint)
                    and cb.monitor == config.checkpointing.monitor
                ):
                    if cb.best_model_path:
                        best_path = cb.best_model_path
                        rank_zero_print(
                            f"🎯 Selected BEST REGULAR checkpoint (monitor: {cb.monitor}): {best_path}"
                        )
                    break

        eval_ckpt = best_path if best_path else "best"
        trainer.test(
            lightning_model,
            datamodule=data_module,
            ckpt_path=eval_ckpt,
            weights_only=False,
        )

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if trainer.is_global_zero:
        wandb.finish()


if __name__ == "__main__":
    main()
