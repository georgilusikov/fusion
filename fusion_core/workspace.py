"""Isolated temporary workspaces for agent-mode model calls."""

from __future__ import annotations

import contextlib
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Iterator

from .config import WORKTREE_LOCK

# --- safe agent workspaces -------------------------------------------------

def _snapshot_ignore(_directory: str, names: list[str]) -> set[str]:
    blocked = {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".aws",
        ".ssh",
        ".gnupg",
        "dist",
        "build",
    }
    return {
        name for name in names
        if name in blocked or name.startswith(".env") or name.endswith(".pyc")
    }


def _remove_snapshot_symlinks(root: Path) -> None:
    for path in root.rglob("*"):
        try:
            if path.is_symlink():
                path.unlink()
        except OSError:
            continue


def _make_tree_read_only(root: Path) -> None:
    paths = [root, *root.rglob("*")]
    for path in sorted(paths, key=lambda item: len(item.parts), reverse=True):
        try:
            if path.is_symlink():
                continue
            current = stat.S_IMODE(path.stat().st_mode)
            path.chmod(current & ~0o222)
        except OSError:
            continue


def _make_tree_writable(root: Path) -> None:
    if not root.exists():
        return
    paths = [root, *root.rglob("*")]
    for path in paths:
        try:
            if path.is_symlink():
                continue
            current = stat.S_IMODE(path.stat().st_mode)
            path.chmod(current | 0o700)
        except OSError:
            continue


def _copy_read_only_snapshot(source: Path, workspace: Path) -> None:
    shutil.copytree(
        source, workspace, ignore=_snapshot_ignore, symlinks=True,
        ignore_dangling_symlinks=True,
    )
    _remove_snapshot_symlinks(workspace)
    _make_tree_read_only(workspace)


def _git_root(source: Path) -> Path | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return Path(proc.stdout.strip()).resolve()


@contextlib.contextmanager
def prepared_workspace(mode: str, source: Path) -> Iterator[Path]:
    """Yield an isolated agent workspace and remove it afterwards.

    temp: empty directory.
    snapshot: copied source tree; copied files are read-only and secrets/caches excluded.
    worktree: detached temporary git worktree; falls back to snapshot when unavailable.
    """
    source = source.resolve()
    base = Path(tempfile.mkdtemp(prefix="fusion_workspace_"))
    workspace = base / "workspace"
    worktree_root: Path | None = None
    worktree_added = False
    try:
        if mode == "temp":
            workspace.mkdir()
        elif mode == "snapshot":
            if not source.is_dir():
                raise ValueError(f"workspace source is not a directory: {source}")
            _copy_read_only_snapshot(source, workspace)
        elif mode == "worktree":
            worktree_root = _git_root(source)
            if worktree_root is None:
                if not source.is_dir():
                    raise ValueError(f"workspace source is not a directory: {source}")
                _copy_read_only_snapshot(source, workspace)
            else:
                try:
                    with WORKTREE_LOCK:
                        subprocess.run(
                            ["git", "-C", str(worktree_root), "worktree", "add", "--detach", str(workspace), "HEAD"],
                            capture_output=True,
                            text=True,
                            timeout=60,
                            check=True,
                        )
                    worktree_added = True
                except (OSError, subprocess.SubprocessError):
                    _make_tree_writable(workspace)
                    shutil.rmtree(workspace, ignore_errors=True)
                    _copy_read_only_snapshot(source, workspace)
        else:
            raise ValueError(f"unknown agent workspace mode: {mode}")
        yield workspace
    finally:
        if worktree_added and worktree_root is not None:
            try:
                with WORKTREE_LOCK:
                    subprocess.run(
                        ["git", "-C", str(worktree_root), "worktree", "remove", "--force", str(workspace)],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
            except OSError:
                pass
        _make_tree_writable(base)
        shutil.rmtree(base, ignore_errors=True)


