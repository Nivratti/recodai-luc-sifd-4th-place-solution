from __future__ import annotations

import os
import sys
import site
from pathlib import Path
from typing import Iterable, List, Set


EXCLUDE_DIR_NAMES = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".idea", ".vscode",
    "runs", "outputs", "out", "logs",
    "data", "datasets",
    "wheelhouse", "kaggle_bundle", "dist", "build",
    ".venv", "venv", "env",
}

MARKER_FILES = ("pyproject.toml", "setup.py", "setup.cfg")


def _is_probably_python_root(dirpath: Path) -> bool:
    return any((dirpath / m).exists() for m in MARKER_FILES)


def _has_any_py_package(dirpath: Path) -> bool:
    # Flat-layout package: <root>/<pkg>/__init__.py
    # Keep it permissive.
    for child in dirpath.iterdir():
        if child.is_dir() and (child / "__init__.py").exists():
            return True
    return False


def discover_python_paths(repo_root: Path, max_depth: int = 7) -> List[Path]:
    repo_root = repo_root.resolve()
    found: List[Path] = []
    seen: Set[Path] = set()

    # Always include repo root (helps "tools.*" style imports)
    candidates = [repo_root]

    # If top-level src exists, include it
    if (repo_root / "src").is_dir():
        candidates.append(repo_root / "src")

    def add(p: Path):
        p = p.resolve()
        if p not in seen and p.exists() and p.is_dir():
            seen.add(p)
            found.append(p)

    # Apply initial candidates
    for c in candidates:
        add(c)

    # Walk repo tree (bounded depth)
    base_depth = len(repo_root.parts)
    for dirpath, dirnames, filenames in os.walk(repo_root):
        d = Path(dirpath)

        # depth guard
        depth = len(d.parts) - base_depth
        if depth > max_depth:
            dirnames[:] = []
            continue

        # prune excluded dirs
        dirnames[:] = [x for x in dirnames if x not in EXCLUDE_DIR_NAMES and not x.startswith(".")]

        # if this dir looks like a python project root, add src/ or itself
        if _is_probably_python_root(d):
            if (d / "src").is_dir():
                add(d / "src")
            elif _has_any_py_package(d):
                add(d)

    return found


def write_pth(paths: Iterable[Path], name: str = "recodai_bundle_paths.pth") -> Path:
    # Try system site-packages first (pip writes here usually)
    site_pkgs = site.getsitepackages()
    target_dir = Path(site_pkgs[0]) if site_pkgs else Path(sys.prefix) / "lib"

    target_dir.mkdir(parents=True, exist_ok=True)
    pth_path = target_dir / name

    text = "\n".join(str(p) for p in paths) + "\n"
    pth_path.write_text(text, encoding="utf-8")
    return pth_path


def activate_paths(paths: List[Path], also_sys_path: bool = True) -> None:
    # Make it effective immediately (without kernel restart)
    if also_sys_path:
        for p in reversed(paths):
            sp = str(p)
            if sp not in sys.path:
                sys.path.insert(0, sp)


def main():
    repo_root = Path(os.environ.get("REPO_ROOT", "")).expanduser()
    if not repo_root.exists():
        raise SystemExit(f"REPO_ROOT not found: {repo_root}")

    paths = discover_python_paths(repo_root)
    pth = write_pth(paths)

    activate_paths(paths, also_sys_path=True)

    print(f"[OK] Discovered {len(paths)} python paths")
    print(f"[OK] Wrote .pth: {pth}")
    # Optional: print a few
    for p in paths[:15]:
        print(" -", p)


if __name__ == "__main__":
    main()