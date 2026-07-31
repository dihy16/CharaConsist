import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run_colab_remote  # noqa: E402


class ModelSnapshotTests(unittest.TestCase):
    def test_complete_snapshot_skips_hugging_face_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "FLUX.1-dev"
            model_path.mkdir()
            (model_path / "model_index.json").write_text("{}", encoding="utf-8")

            with patch.object(run_colab_remote, "download_model_snapshot") as download:
                downloaded = run_colab_remote.ensure_model_snapshot(
                    model_path, "owner/model", None
                )

            self.assertFalse(downloaded)
            download.assert_not_called()

    def test_incomplete_snapshot_requires_a_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "HF_TOKEN"):
                run_colab_remote.ensure_model_snapshot(
                    Path(tmp) / "FLUX.1-dev", "owner/model", None
                )

    def test_incomplete_snapshot_downloads_once_then_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "FLUX.1-dev"

            def create_complete_snapshot(path, _repo, _token):
                path.mkdir()
                (path / "model_index.json").write_text("{}", encoding="utf-8")

            with patch.object(
                run_colab_remote,
                "download_model_snapshot",
                side_effect=create_complete_snapshot,
            ) as download:
                downloaded = run_colab_remote.ensure_model_snapshot(
                    model_path, "owner/model", "test-token"
                )

            self.assertTrue(downloaded)
            download.assert_called_once_with(model_path, "owner/model", "test-token")


if __name__ == "__main__":
    unittest.main()
