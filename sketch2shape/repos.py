"""Clone optional third-party inference code (Hunyuan3D / TripoSR) on first use."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

from .config import PROJECT_ROOT, TRIPOSR_GIT, HUNYUAN_GIT

THIRD_PARTY = PROJECT_ROOT / "third_party"


def _git_clone(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(dest)],
        check=True,
    )


def ensure_repo(name: str, url: str) -> Path:
    dest = THIRD_PARTY / name
    if (dest / ".git").is_dir() or (dest / "tsr").is_dir() or (dest / "hy3dgen").is_dir():
        _prepend_path(dest)
        return dest
    print(f">>> Cloning {name} from {url}", flush=True)
    _git_clone(url, dest)
    _prepend_path(dest)
    return dest


def _prepend_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def ensure_hunyuan() -> Optional[Path]:
    try:
        return ensure_repo("Hunyuan3D-2", HUNYUAN_GIT)
    except Exception as exc:
        print(f"Hunyuan3D-2 clone/import skipped: {exc}", flush=True)
        return None


def ensure_triposr() -> Optional[Path]:
    try:
        return ensure_repo("TripoSR", TRIPOSR_GIT)
    except Exception as exc:
        print(f"TripoSR clone skipped: {exc}", flush=True)
        return None
