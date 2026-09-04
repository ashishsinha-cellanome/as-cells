#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import sys
import re
from pathlib import Path
from datetime import datetime

import torch
from omegaconf import OmegaConf


_INTERP_RE = re.compile(r"\$\{([^}]+)\}")


def _resolve_interpolations(value: str, cfg_dict: dict, cwd: Path) -> str:
    if not isinstance(value, str):
        return value

    def _replace(match):
        key = match.group(1)
        if key == "hydra:runtime.cwd":
            return str(cwd)
        if ":" in key:
            # Unsupported resolver type; keep as-is
            return match.group(0)
        resolved = _get_nested(cfg_dict, key)
        if resolved is None:
            return match.group(0)
        return str(resolved)

    prev = None
    cur = value
    for _ in range(5):
        if cur == prev:
            break
        prev = cur
        cur = _INTERP_RE.sub(_replace, cur)
    return cur


def _select_config(ckpt: dict):
    hyper = ckpt.get("hyper_parameters") or {}
    cfg = hyper.get("config")
    if cfg is None:
        raise ValueError("Checkpoint missing hyper_parameters.config")
    return cfg


def _to_plain_cfg(cfg):
    if isinstance(cfg, dict):
        return cfg
    return OmegaConf.to_container(cfg, resolve=False)


def _get_nested(cfg_dict: dict, dotted_key: str, default=None):
    cur = cfg_dict
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _select_weights(ckpt_path: Path, ckpt: dict):
    name_lower = ckpt_path.name.lower()
    has_ema = isinstance(ckpt.get("ema_state_dict"), dict)
    if "ema" in name_lower and has_ema:
        return "ema", ckpt["ema_state_dict"]
    if isinstance(ckpt.get("state_dict"), dict):
        return "regular", ckpt["state_dict"]
    if has_ema:
        return "ema", ckpt["ema_state_dict"]
    raise ValueError("Checkpoint has no usable state_dict or ema_state_dict")


def _normalize_state_dict(sd: dict):
    out = {}
    for k, v in sd.items():
        if not isinstance(k, str):
            continue
        if not k.startswith("model."):
            continue
        if k.startswith("model.model."):
            k = "model." + k[len("model.model.") :]
        out[k] = v
    return out


def _label_map_to_names(label_map) -> list:
    if label_map is None:
        return []
    if not isinstance(label_map, dict):
        label_map = OmegaConf.to_container(label_map, resolve=False)
    items = sorted(label_map.items(), key=lambda kv: int(kv[0]))
    return [str(v) for _, v in items]


def _load_detection_model(repo_path: Path, model_cfg_path: Path, nc: int):
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    from models.yolo import DetectionModel  # type: ignore

    model = DetectionModel(cfg=str(model_cfg_path), nc=nc)
    return model


def convert_ckpt(ckpt_path: Path, out_dir: Path | None):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = _select_config(ckpt)
    cfg_dict = _to_plain_cfg(cfg)

    cwd = Path.cwd()
    repo_path = _get_nested(cfg_dict, "model.yolov5.repo_path")
    model_cfg = _get_nested(cfg_dict, "model.yolov5.model_cfg")
    label_map = _get_nested(cfg_dict, "model.label_map")

    repo_path = (
        Path(_resolve_interpolations(repo_path, cfg_dict, cwd)).expanduser().resolve()
    )
    model_cfg = _resolve_interpolations(model_cfg, cfg_dict, cwd)
    model_cfg_path = Path(model_cfg)
    if not model_cfg_path.is_absolute():
        model_cfg_path = repo_path / model_cfg_path
    model_cfg_path = model_cfg_path.expanduser().resolve()

    if not repo_path.exists():
        raise FileNotFoundError(f"YOLOv5 repo not found at: {repo_path}")
    if not model_cfg_path.exists():
        raise FileNotFoundError(f"YOLOv5 model cfg not found at: {model_cfg_path}")

    names = _label_map_to_names(label_map)
    nc = len(names) if names else int(_get_nested(cfg_dict, "model.yolov5.nc") or 0)
    if nc == 0:
        raise ValueError("Could not determine number of classes from config label_map")

    model = _load_detection_model(repo_path, model_cfg_path, nc)
    model.nc = nc
    if names:
        model.names = names

    source, sd = _select_weights(ckpt_path, ckpt)
    sd = _normalize_state_dict(sd)

    result = model.load_state_dict(sd, strict=False)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    if missing or unexpected:
        print(
            f"[WARN] {ckpt_path.name}: missing={len(missing)} unexpected={len(unexpected)} ({source})"
        )
    else:
        print(f"[INFO] {ckpt_path.name}: loaded weights ({source})")

    opt = ckpt.get("opt")
    if opt is None:
        opt = cfg_dict if cfg_dict is not None else None

    save_dict = {
        "epoch": ckpt.get("epoch"),
        "best_fitness": ckpt.get("best_fitness"),
        "model": model,
        "ema": None,
        "updates": ckpt.get("updates"),
        "optimizer": ckpt.get("optimizer"),
        "wandb_id": ckpt.get("wandb_id"),
        "opt": opt,
        "date": ckpt.get("date") or datetime.now().strftime("%Y-%m-%d"),
    }

    if out_dir is None:
        out_dir = ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (ckpt_path.stem + ".pt")
    torch.save(save_dict, out_path)
    print(f"[OK] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Lightning YOLOv5 .ckpt to Ultralytics .pt for models/yolov5/val.py"
    )
    parser.add_argument(
        "--ckpt", nargs="+", required=True, help="Path(s) to .ckpt file(s)"
    )
    parser.add_argument(
        "--out-dir", default=None, help="Optional output directory for .pt files"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    for p in args.ckpt:
        convert_ckpt(Path(p).expanduser().resolve(), out_dir)


if __name__ == "__main__":
    main()
