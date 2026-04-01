"""Project-local Python startup hooks.

This repo uses ``uv`` wheels for CUDA-enabled PyTorch. Some NVIDIA runtime
libraries are installed under ``.venv/site-packages/nvidia/.../lib`` instead of
system library paths, so ``torch`` can fail to import with missing ``.so``
errors unless those libraries are preloaded first.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


def _candidate_lib_dirs() -> list[Path]:
    prefixes = []
    if hasattr(sys, "prefix"):
        prefixes.append(Path(sys.prefix))
    if hasattr(sys, "base_prefix"):
        prefixes.append(Path(sys.base_prefix))

    seen: set[Path] = set()
    lib_dirs: list[Path] = []
    for prefix in prefixes:
        site_packages = prefix / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        nvidia_root = site_packages / "nvidia"
        for rel in (
            Path("cu13/lib"),
            Path("cudnn/lib"),
            Path("cusparselt/lib"),
            Path("nccl/lib"),
            Path("nvshmem/lib"),
        ):
            lib_dir = nvidia_root / rel
            if lib_dir.is_dir() and lib_dir not in seen:
                seen.add(lib_dir)
                lib_dirs.append(lib_dir)
    return lib_dirs


def _prepend_ld_library_path(lib_dirs: list[Path]) -> None:
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [str(path) for path in lib_dirs]
    if current:
        parts.append(current)
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)


def _preload_shared_objects(lib_dirs: list[Path]) -> None:
    # Load the packaged NVIDIA libraries into the process before torch imports.
    # This avoids relying on site-specific loader configuration in SLURM shells.
    for lib_dir in lib_dirs:
        for so_path in sorted(lib_dir.glob("*.so*")):
            try:
                ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue


_LIB_DIRS = _candidate_lib_dirs()
if _LIB_DIRS:
    _prepend_ld_library_path(_LIB_DIRS)
    _preload_shared_objects(_LIB_DIRS)
