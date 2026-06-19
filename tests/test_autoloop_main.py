"""Tests for autonomous_loop main() entry point."""

from unittest.mock import patch


class TestAutoLoopMain:
    def test_main_prepare_only(self):
        from autoresearch.experiment.autonomous_loop import main

        with patch("sys.argv", ["autoresearch", "--prepare-only"]):
            main()

    def test_main_with_args(self):
        import os
        import tempfile

        from autoresearch.experiment.autonomous_loop import main

        f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        f.write("test text\n")
        f.close()
        try:
            with patch(
                "sys.argv", ["autoresearch", "-i", f.name, "-n", "1", "--prepare-only"]
            ):
                main()
        finally:
            os.unlink(f.name)
