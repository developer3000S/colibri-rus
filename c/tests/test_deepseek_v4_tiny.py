#!/usr/bin/env python3
"""Dependency-free token-exact checks for the committed tiny V4 fixture."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path


def run(label: str, command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
    )
    if result.returncode:
        raise AssertionError(
            f"{label} failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def flat_oracle(case: dict[str, object]) -> dict[str, object]:
    return {
        "prompt_ids": case["prompt_ids"],
        "full_ids": case["greedy_full_ids"],
        "tf_pred": case["teacher_forcing_ids"],
    }


def check_target(
    binary: Path,
    model: Path,
    draft: Path,
    name: str,
    case: dict[str, object],
    temporary: Path,
) -> None:
    oracle = temporary / f"{name}.json"
    oracle.write_text(json.dumps(flat_oracle(case)), encoding="utf-8")
    full = case["greedy_full_ids"]
    generated = case["greedy_new_ids"]
    result = run(
        f"target {name}",
        [
            binary.as_posix(),
            model.as_posix(),
            "--oracle",
            oracle.as_posix(),
            "--teacher-forcing",
            str(len(full)),
            "--greedy",
            str(len(generated)),
            "--draft-model",
            draft.as_posix(),
        ],
    )
    expected_tf = f"{len(full)}/{len(full)} positions"
    expected_greedy = f"{len(generated)}/{len(generated)} tokens"
    if expected_tf not in result.stdout:
        raise AssertionError(f"target {name}: missing exact teacher-forcing result")
    if expected_greedy not in result.stdout:
        raise AssertionError(f"target {name}: missing exact greedy result")
    print(f"PASS target {name}: teacher forcing and greedy token-exact")


def check_prefix_is_rejected(
    binary: Path, model: Path, draft: Path, case: dict[str, object], temporary: Path
) -> None:
    truncated = flat_oracle(case)
    truncated["full_ids"] = truncated["full_ids"][:-1]
    truncated["tf_pred"] = truncated["tf_pred"][:-1]
    oracle = temporary / "truncated-prefix.json"
    oracle.write_text(json.dumps(truncated), encoding="utf-8")
    result = subprocess.run(
        [
            binary.as_posix(),
            model.as_posix(),
            "--oracle",
            oracle.as_posix(),
            "--teacher-forcing",
            str(len(truncated["full_ids"])),
            "--greedy",
            str(len(case["greedy_new_ids"])),
            "--draft-model",
            draft.as_posix(),
        ],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
    )
    if result.returncode == 0 or "greedy length mismatch" not in result.stderr:
        raise AssertionError("truncated greedy prefix was not rejected")
    print("PASS target greedy: truncated prefix rejected")


def token_prompt(ids: list[int]) -> str:
    return "".join(f"<t{token:03d}>" for token in ids)


def check_session(
    binary: Path,
    model: Path,
    draft: Path,
    name: str,
    case: dict[str, object],
    temporary: Path,
    dspark: bool,
    ordinal: int = 0,
) -> list[int]:
    record = temporary / f"{name}-{'dspark' if dspark else 'target'}-{ordinal}.json"
    command = [
        binary.as_posix(),
        model.as_posix(),
        token_prompt(case["prompt_ids"]),
        "--raw-prompt",
        "--max-tokens",
        str(case["max_new_tokens"]),
        "--record-oracle",
        record.as_posix(),
        "--draft-model",
        draft.as_posix(),
    ]
    if not dspark:
        command.append("--no-dspark")
    result = run(f"session {name} dspark={dspark}", command)
    actual = json.loads(record.read_text(encoding="utf-8"))
    expected_prompt = case["prompt_ids"]
    expected_full = case["greedy_full_ids"]
    if actual.get("prompt_ids") != expected_prompt:
        raise AssertionError(
            f"session {name}: tokenized prompt mismatch: "
            f"expected {expected_prompt}, got {actual.get('prompt_ids')}"
        )
    if actual.get("full_ids") != expected_full:
        raise AssertionError(
            f"session {name} dspark={dspark}: exact output mismatch: "
            f"expected {expected_full}, got {actual.get('full_ids')}"
        )
    stats = re.search(
        r"v4_tokens prompt=(\d+) generated=(\d+).*?"
        r"speculative_rounds=(\d+) proposed=(\d+) accepted=(\d+).*?enabled=(\d+)",
        result.stderr,
    )
    if not stats:
        raise AssertionError(f"session {name}: missing generation statistics")
    prompt_count, generated_count, rounds, proposed, _accepted, enabled = (
        int(value) for value in stats.groups()
    )
    if prompt_count != len(expected_prompt) or generated_count != case["max_new_tokens"]:
        raise AssertionError(
            f"session {name}: length mismatch: prompt={prompt_count} generated={generated_count}"
        )
    if dspark and (enabled != 1 or rounds < 1 or proposed < 1):
        raise AssertionError(
            f"session {name}: DSpark was not exercised: "
            f"enabled={enabled} rounds={rounds} proposed={proposed}"
        )
    if not dspark and (enabled != 0 or rounds or proposed):
        raise AssertionError(
            f"session {name}: drafting was not disabled: "
            f"enabled={enabled} rounds={rounds} proposed={proposed}"
        )
    mode = "DSpark" if dspark else "target session"
    print(f"PASS {mode} {name}: exact IDs and exact length")
    return actual["full_ids"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()

    binary = args.binary.resolve()
    fixture = args.fixture.resolve()
    draft = fixture / "dspark"
    reference = json.loads((fixture / "ref.json").read_text(encoding="utf-8"))
    if reference.get("source") != "transformers":
        raise AssertionError("tiny oracle must come from independent Transformers")
    cases = reference["cases"]

    with tempfile.TemporaryDirectory(prefix="colibri-v4-tiny-") as directory:
        temporary = Path(directory)
        check_target(binary, fixture, draft, "short", cases["short"], temporary)
        check_prefix_is_rejected(binary, fixture, draft, cases["short"], temporary)
        check_target(binary, fixture, draft, "compressed", cases["compressed"], temporary)
        check_target(binary, fixture, draft, "long", cases["long"], temporary)

        # Repeated process-level engine/session lifecycles also validate that
        # drafting is genuinely disabled on the ordinary runtime path.
        for ordinal in range(3):
            check_session(
                binary,
                fixture,
                draft,
                "short",
                cases["short"],
                temporary,
                dspark=False,
                ordinal=ordinal,
            )
        # The 72-token case crosses the target and DSpark 64-token prefill
        # chunk boundary and verifies absolute-position continuity.
        target_long = check_session(
            binary,
            fixture,
            draft,
            "long",
            cases["long"],
            temporary,
            dspark=False,
        )
        dspark_long = check_session(
            binary,
            fixture,
            draft,
            "long",
            cases["long"],
            temporary,
            dspark=True,
        )
        if dspark_long != target_long:
            raise AssertionError("DSpark output differs from target-only output")
        print("PASS DSpark enabled/disabled identity: exact full sequence")

    print("PASS tiny DeepSeek V4 oracle: all checks completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
