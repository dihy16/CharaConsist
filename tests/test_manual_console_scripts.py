import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONSOLE_DIR = REPO_ROOT / "scripts" / "colab_console"


class ManualConsoleScriptTests(unittest.TestCase):
    def test_task_specs_use_remote_posix_output_paths(self):
        cases = (
            ("lambda", "1a_anchor_verb.txt", "1", "lambda_1p00/seed_2025/bg_fg/1a_anchor_verb"),
            ("component", "1a_anchor_verb.txt", "full", "component_ablation/full/seed_2025/bg_fg/1a_anchor_verb"),
            ("role", "2b_transfer_roles.txt", "1", "role_action_ablation/role_bias_1p00/seed_2025/bg_fg/2b_transfer_roles"),
            ("binding", "2b_final_action_binding.txt", "1:0.5", "action_binding_ablation/beta_1p00_gamma_0p50/seed_2025/bg_fg/2b_final_action_binding"),
            ("routing", "2b_final_action_binding.txt", "hard:1:0.5", "entity_routing_ablation/routing_hard/beta_1p00_gamma_0p50/seed_2025/bg_fg/2b_final_action_binding"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            for mode, prompt, value, expected in cases:
                output = Path(temp_dir) / f"{mode}.json"
                subprocess.run(
                    [
                        sys.executable,
                        str(CONSOLE_DIR / "write_manual_task.py"),
                        "--output", str(output),
                        "--mode", mode,
                        "--prompt", prompt,
                        "--value", value,
                        "--seed", "2025",
                    ],
                    check=True,
                )
                task = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(task["relative_output"], expected)

    def test_documented_shell_scripts_exist_without_inline_python(self):
        manual = (REPO_ROOT / "docs" / "MANUAL_COLAB_CONSOLE_RUN.md").read_text(encoding="utf-8")
        scripts = sorted(CONSOLE_DIR.glob("*.sh"))
        self.assertGreaterEqual(len(scripts), 10)
        for script in scripts:
            if script.name == "manual_common.sh":
                continue
            self.assertIn(script.name, manual)
            self.assertNotIn("<<'PY'", script.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
