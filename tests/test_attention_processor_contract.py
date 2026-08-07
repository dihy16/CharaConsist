import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSOR_SOURCE = (
    PROJECT_ROOT / "models" / "attention_processor_characonsist.py"
).read_text(encoding="utf-8")


class AttentionProcessorContractTests(unittest.TestCase):
    def test_all_processors_accept_shared_action_gate_kwarg(self):
        tree = ast.parse(PROCESSOR_SOURCE)
        class_names = {"FluxAttnProcessor2_0", "CharaConsistAttnProcessor2_0"}
        signatures = {}
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in class_names:
                call = next(
                    child
                    for child in node.body
                    if isinstance(child, ast.FunctionDef) and child.name == "__call__"
                )
                signatures[node.name] = {argument.arg for argument in call.args.args}

        self.assertEqual(set(signatures), class_names)
        for parameters in signatures.values():
            self.assertIn("action_gate_strength", parameters)
            self.assertIn("role_action_bias_strength", parameters)
            self.assertIn("entity_routing_mode", parameters)


if __name__ == "__main__":
    unittest.main()
