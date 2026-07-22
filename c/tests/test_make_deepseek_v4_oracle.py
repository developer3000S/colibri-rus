import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "tools" / "make_deepseek_v4_oracle.py"
SPEC = importlib.util.spec_from_file_location("make_deepseek_v4_oracle", SCRIPT)
ORACLE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ORACLE)


class DSparkIdentityTests(unittest.TestCase):
    def test_make_target_uses_supported_validation_flags(self):
        makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text(
            encoding="utf-8"
        )
        target = makefile.split("deepseek-v4-oracle: deepseek-v4", 1)[1]
        target = target.split("\nelse\n", 1)[0]
        self.assertIn("--validate", target)
        self.assertIn("--teacher-forcing", target)
        self.assertIn("--check-dspark", target)
        self.assertNotIn("--continuation", target)

    def run_validation(self, generated: list[list[int]]) -> int:
        with tempfile.TemporaryDirectory() as directory:
            oracle_path = Path(directory) / "oracle.json"
            oracle_path.write_text(
                json.dumps({"prompt_ids": [1], "full_ids": [1, 2, 3]}),
                encoding="utf-8",
            )
            outputs = list(generated)

            def fake_run(binary, args, env=None):
                if "--record-oracle" in args:
                    output = Path(args[args.index("--record-oracle") + 1])
                    tokens = outputs.pop(0)
                    output.write_text(
                        json.dumps({
                            "prompt_ids": [1],
                            "full_ids": [1, *tokens],
                        }),
                        encoding="utf-8",
                    )
                return types.SimpleNamespace(returncode=0)

            with mock.patch.object(ORACLE, "run_c", side_effect=fake_run):
                return ORACLE.validate(
                    Path("deepseek_v4"),
                    "/model",
                    oracle_path,
                    teacher_forcing=2,
                    greedy=2,
                    memory_gb=None,
                    check_dspark=True,
                )

    def test_empty_outputs_do_not_pass_identity(self):
        self.assertNotEqual(self.run_validation([[], []]), 0)

    def test_exact_full_length_outputs_pass_identity(self):
        self.assertEqual(self.run_validation([[2, 3], [2, 3]]), 0)


if __name__ == "__main__":
    unittest.main()
