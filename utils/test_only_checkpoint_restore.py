from __future__ import annotations

import os
from typing import Any, Dict, Literal, Optional, Tuple

import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from utils.distributed_utils import rank_zero_print


def _set_nested_value(cfg: DictConfig, dotted_key: str, value: Any) -> None:
    """Set nested DictConfig value, creating intermediate dicts as needed."""
    if value is None:
        return
    node = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in node or node[part] is None:
            node[part] = OmegaConf.create({})
        node = node[part]
    node[parts[-1]] = value


def _resolve_ckpt_path(
    config: DictConfig, run_save_dir: Optional[str] = None
) -> Optional[str]:
    """Resolve checkpoint path from manual override or auto-resume."""
    ckpt_path = config.initialization.get("load_from_checkpoint")
    if ckpt_path:
        raw = str(ckpt_path).strip().strip('"').strip("'")
        raw = raw.rstrip(".,;:)]}>")

        candidates = []
        for candidate in [
            raw,
            os.path.expanduser(raw),
            os.path.expandvars(raw),
            to_absolute_path(raw),
        ]:
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        for candidate in candidates:
            if os.path.exists(candidate):
                rank_zero_print(f"🔄 Manual checkpoint path: {candidate}")
                return candidate

        rank_zero_print("WARNING: Provided checkpoint path was not found.")
        rank_zero_print(f"  raw: {raw}")
        for idx, candidate in enumerate(candidates):
            rank_zero_print(f"  candidate[{idx}]: {candidate}")
        return None

    if run_save_dir and config.initialization.get("auto_resume", False):
        last_ckpt = os.path.join(run_save_dir, "ckpts", "last.ckpt")
        if os.path.exists(last_ckpt):
            rank_zero_print(f"Auto-Resume: Found existing 'last.ckpt' at {last_ckpt}")
            return last_ckpt
        rank_zero_print(
            f"Auto-Resume: No 'last.ckpt' found in {last_ckpt}. Starting fresh."
        )

    return None


def _load_ckpt(ckpt_path: str) -> Dict[str, Any]:
    """Load checkpoint on CPU with trusted full object unpickling."""
    rank_zero_print(f"Loading checkpoint object from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint to be a dict, got {type(checkpoint)}")
    return checkpoint


def _merge_test_only_config_from_ckpt(
    current_cfg: DictConfig, ckpt: Dict[str, Any]
) -> DictConfig:
    """Use checkpoint config as base and re-apply runtime/test-only overrides."""
    hp = ckpt.get("hyper_parameters", {})
    ckpt_cfg_raw = hp.get("config") if isinstance(hp, dict) else None
    if ckpt_cfg_raw is None:
        rank_zero_print(
            "WARNING: No 'hyper_parameters.config' found in checkpoint. Using current CLI config."
        )
        return current_cfg

    ckpt_cfg = OmegaConf.create(
        OmegaConf.to_container(ckpt_cfg_raw, resolve=False)
        if OmegaConf.is_config(ckpt_cfg_raw)
        else ckpt_cfg_raw
    )
    merged_cfg = OmegaConf.create(OmegaConf.to_container(ckpt_cfg, resolve=False))
    OmegaConf.set_struct(merged_cfg, False)

    # Runtime/test-only override policy
    merged_cfg.test_only = current_cfg.test_only
    merged_cfg.debug = current_cfg.debug
    merged_cfg.seed = current_cfg.seed
    merged_cfg.run_name = current_cfg.run_name
    merged_cfg.val_name = current_cfg.val_name
    merged_cfg.test_name = current_cfg.test_name
    merged_cfg.train_name = current_cfg.train_name

    _set_nested_value(
        merged_cfg,
        "initialization.load_from_checkpoint",
        current_cfg.initialization.get("load_from_checkpoint"),
    )

    if hasattr(current_cfg, "trainer") and current_cfg.trainer is not None:
        merged_cfg.trainer = OmegaConf.create(
            OmegaConf.to_container(current_cfg.trainer, resolve=False)
        )

    if hasattr(current_cfg, "data") and current_cfg.data is not None:
        _set_nested_value(merged_cfg, "data.path", current_cfg.data.get("path"))
        _set_nested_value(
            merged_cfg,
            "data.limit_test_batches",
            current_cfg.data.get("limit_test_batches"),
        )
        _set_nested_value(
            merged_cfg, "data.num_workers", current_cfg.data.get("num_workers")
        )
        _set_nested_value(
            merged_cfg, "data.batch_size", current_cfg.data.get("batch_size")
        )
        _set_nested_value(
            merged_cfg, "data.eval_batch_size", current_cfg.data.get("eval_batch_size")
        )


    if hasattr(current_cfg, "checkpointing") and current_cfg.checkpointing is not None:
        merged_cfg.checkpointing = OmegaConf.create(
            OmegaConf.to_container(current_cfg.checkpointing, resolve=False)
        )

    if hasattr(current_cfg, "logging") and current_cfg.logging is not None:
        merged_cfg.logging = OmegaConf.create(
            OmegaConf.to_container(current_cfg.logging, resolve=False)
        )

    if (
        hasattr(current_cfg, "eval_inference")
        and current_cfg.eval_inference is not None
    ):
        merged_cfg.eval_inference = OmegaConf.create(
            OmegaConf.to_container(current_cfg.eval_inference, resolve=False)
        )

    if hasattr(current_cfg, "inference") and current_cfg.inference is not None:
        merged_cfg.inference = OmegaConf.create(
            OmegaConf.to_container(current_cfg.inference, resolve=False)
        )

    rank_zero_print(
        "Using checkpoint config as base for test_only; reapplied runtime/test CLI overrides."
    )
    return merged_cfg


def _extract_regular_state_dict(ckpt: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get regular model state dict from Lightning checkpoint or raw state dict checkpoint."""
    state_dict = ckpt.get("state_dict")
    if isinstance(state_dict, dict):
        return state_dict

    # Native YOLOv5 checkpoint (Ultralytics format)
    if "model" in ckpt and hasattr(ckpt["model"], "state_dict"):
        raw_sd = ckpt["model"].float().state_dict()
        return {f"model.{k}": v for k, v in raw_sd.items()}

    # Fallback: checkpoint may itself be a state_dict-like mapping
    tensor_items = [
        (k, v)
        for k, v in ckpt.items()
        if isinstance(k, str) and isinstance(v, torch.Tensor)
    ]
    if tensor_items and all(
        "." in k for k, _ in tensor_items[: min(8, len(tensor_items))]
    ):
        return ckpt
    return None


def _select_eval_weights_source(
    ckpt_path: str, ckpt: Dict[str, Any], config: Optional[DictConfig] = None
) -> Literal["ema", "regular"]:
    """Choose EMA vs regular weights for evaluation using path hint + key availability."""
    path_has_ema = "ema" in os.path.basename(ckpt_path).lower()
    if config is not None and config.get("inference", {}).get("use_ema", False):
        path_has_ema = True

    has_lightning_ema = isinstance(ckpt.get("ema_state_dict"), dict)
    has_native_ema = (
        "ema" in ckpt and ckpt["ema"] is not None and hasattr(ckpt["ema"], "ema")
    )
    has_native_model = "model" in ckpt and hasattr(ckpt["model"], "state_dict")
    has_ema = has_lightning_ema or has_native_ema or has_native_model

    has_regular = _extract_regular_state_dict(ckpt) is not None

    if path_has_ema and has_ema:
        return "ema"
    if path_has_ema and not has_ema and has_regular:
        rank_zero_print(
            "WARNING: Checkpoint path suggests EMA but 'ema_state_dict' missing. Falling back to regular weights."
        )
        return "regular"
    if (not path_has_ema) and has_regular:
        return "regular"
    if (not path_has_ema) and (not has_regular) and has_ema:
        rank_zero_print("WARNING: Regular state_dict missing; using EMA weights.")
        return "ema"

    raise ValueError(
        "Checkpoint does not contain usable weights. Expected one of: "
        "'state_dict' (regular) or 'ema_state_dict' (EMA)."
    )


def _load_selected_weights(
    lightning_module: Any,
    ckpt: Dict[str, Any],
    source: Literal["ema", "regular"],
) -> Tuple[list, list]:
    """Load selected evaluation weights and return missing/unexpected keys."""
    if source == "ema":
        if isinstance(ckpt.get("ema_state_dict"), dict):
            ema_state = ckpt.get("ema_state_dict")
        elif "ema" in ckpt and ckpt["ema"] is not None and hasattr(ckpt["ema"], "ema"):
            ema_state = ckpt["ema"].ema.float().state_dict()
        elif "model" in ckpt and hasattr(ckpt["model"], "state_dict"):
            ema_state = ckpt["model"].float().state_dict()
        else:
            raise ValueError(
                "Requested EMA weights but checkpoint has no valid EMA state."
            )

        if hasattr(lightning_module, "model") and hasattr(
            lightning_module.model, "load_state_dict"
        ):
            current_state = lightning_module.model.state_dict()
            ema_state = {
                k: v
                for k, v in ema_state.items()
                if k in current_state and v.shape == current_state[k].shape
            }
            result = lightning_module.model.load_state_dict(ema_state, strict=False)
        else:
            current_state = lightning_module.state_dict()
            ema_state = {
                k: v
                for k, v in ema_state.items()
                if k in current_state and v.shape == current_state[k].shape
            }
            result = lightning_module.load_state_dict(ema_state, strict=False)
    else:
        regular_state = _extract_regular_state_dict(ckpt)
        if regular_state is None:
            raise ValueError(
                "Requested regular weights but checkpoint has no valid 'state_dict'."
            )

        current_state = lightning_module.state_dict()
        regular_state = {
            k: v
            for k, v in regular_state.items()
            if k in current_state and v.shape == current_state[k].shape
        }
        result = lightning_module.load_state_dict(regular_state, strict=False)

    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    return missing, unexpected
