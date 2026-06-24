from __future__ import annotations

import contextlib
import io
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import fusion


class FusionWorkspaceTests(unittest.TestCase):
    def test_snapshot_is_isolated_and_excludes_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "code.py").write_text("print('ok')")
            (source / ".env").write_text("SECRET=yes")
            (source / ".env.local").write_text("SECRET=also")
            try:
                os.symlink("/etc/passwd", source / "escape")
            except (OSError, NotImplementedError):
                pass
            with fusion.prepared_workspace("snapshot", source) as workspace:
                copied = workspace / "code.py"
                self.assertTrue(copied.exists())
                self.assertFalse((workspace / ".env").exists())
                self.assertFalse((workspace / ".env.local").exists())
                self.assertFalse((workspace / "escape").exists())
                self.assertEqual(stat.S_IMODE(copied.stat().st_mode) & 0o222, 0)
                self.assertEqual(stat.S_IMODE(workspace.stat().st_mode) & 0o222, 0)
                self.assertNotEqual(workspace.resolve(), source.resolve())

    def test_worktree_is_disposable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "repo"
            source.mkdir()
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.com"], check=True)
            subprocess.run(["git", "-C", str(source), "config", "user.name", "Fusion Test"], check=True)
            tracked = source / "tracked.txt"
            tracked.write_text("original")
            subprocess.run(["git", "-C", str(source), "add", "tracked.txt"], check=True)
            subprocess.run(["git", "-C", str(source), "commit", "-qm", "initial"], check=True)
            with fusion.prepared_workspace("worktree", source) as workspace:
                workspace_path = workspace
                (workspace / "tracked.txt").write_text("changed")
                self.assertEqual(tracked.read_text(), "original")
            self.assertFalse(workspace_path.exists())

    def test_current_directory_agent_mode_is_not_accepted(self) -> None:
        parser = fusion.build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["hello", "--agent-workspace", "current"])


if __name__ == "__main__":
    unittest.main()
