import importlib.machinery
import importlib.util
import argparse
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent.parent
CLI = HERE / "v4"


def load_cli():
    loader = importlib.machinery.SourceFileLoader("v4_cli", str(CLI))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class V4CliTest(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(CLI), *args],
            cwd=HERE,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )

    def test_chat_help_uses_explicit_model_and_ram(self):
        result = self.run_cli("chat", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--model", result.stdout)
        self.assertIn("--ram", result.stdout)

    def test_model_is_required(self):
        result = self.run_cli("chat", "--ram", "24")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--model", result.stderr)

    def test_missing_model_is_reported_before_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / "missing")
            result = self.run_cli("chat", "--model", missing, "--ram", "24")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("model not found", result.stderr)

    def test_multi_turn_prompt_format(self):
        cli = load_cli()
        history = cli.new_history("Be concise.")
        prompt = cli.turn_prompt(history, "Hello", False)
        self.assertEqual(
            prompt,
            cli.BOS + "Be concise." + cli.USER + "Hello" + cli.ASSISTANT + "</think>",
        )
        history = cli.completed_history(prompt, "Hi!")
        next_prompt = cli.turn_prompt(history, "Again", True)
        self.assertEqual(
            next_prompt,
            prompt + "Hi!" + cli.EOS + cli.USER + "Again" + cli.ASSISTANT + "<think>",
        )

    def test_compact_stats_uses_wall_ttft_and_ram_plan(self):
        cli = load_cli()
        log = (
            "ram_tiers available=24.00GiB dense=resident projected=23.95GiB\n"
            "prefill_timing startup=2.0s wall_to_first=15.500s "
            "target_tok_s=2.0 combined_tok_s=1.250000\n"
            "decode_timing tokens=4 seconds=4.0 tok_s=1.000000\n"
            "summary dspark_rounds=1 proposed=4 accepted=2 rate=0.500 enabled=1\n"
        )
        self.assertEqual(
            cli.compact_stats(log),
            "RAM 23.95/24.00 GiB | TTFT 15.50s | prefill 1.250 tok/s | "
            "decode 1.000 tok/s | DSpark acceptance 50.0%",
        )

    def test_engine_command_prefers_prompt_file(self):
        cli = load_cli()
        args = argparse.Namespace(model=str(HERE), ngen=32, ram=0, stop_sentence=False)
        command = cli.engine_command(args, raw=True, prompt_file=r"C:\tmp\prompt.txt")
        self.assertIn("--prompt-file", command)
        self.assertIn(r"C:\tmp\prompt.txt", command)
        self.assertNotIn("你好", command)

    def test_write_prompt_file_is_utf8(self):
        cli = load_cli()
        path = cli.write_prompt_file("你好")
        try:
            self.assertEqual(Path(path).read_text(encoding="utf-8"), "你好")
        finally:
            Path(path).unlink(missing_ok=True)

if __name__ == "__main__":
    unittest.main()
