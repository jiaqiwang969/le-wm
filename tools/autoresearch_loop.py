import argparse
import csv
import json
import os
import re
import subprocess
from pathlib import Path

from autoresearch_common import (
    DEFAULT_BENCHMARK_NAME,
    DEFAULT_MAX_STEPS,
    benchmark_config,
    benchmark_names,
    experiment_dir as common_experiment_dir,
    flatten_cli_overrides,
    tag_dir as common_tag_dir,
)


MAX_EXPERIMENTS = 20
MAX_CONSECUTIVE_CRASHES = 3
MAX_CONSECUTIVE_NON_IMPROVING = 8
TARGET_SUCCESS_RATE = 90.0
EXP_ID_PATTERN = re.compile(r"^exp_(\d+)$")


def load_results_rows(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def resolve_stablewm_home(env=None) -> Path:
    env = env or os.environ
    return Path(env.get("STABLEWM_HOME", Path.home() / "stablewm")).expanduser()


def tag_root(
    stablewm_home: Path,
    tag: str,
    *,
    benchmark: str = DEFAULT_BENCHMARK_NAME,
) -> Path:
    return common_tag_dir(Path(stablewm_home), tag, benchmark=benchmark)


def results_table_path(
    stablewm_home: Path,
    tag: str,
    *,
    benchmark: str = DEFAULT_BENCHMARK_NAME,
) -> Path:
    return tag_root(stablewm_home, tag, benchmark=benchmark) / "results.tsv"


def experiment_dir(
    stablewm_home: Path,
    tag: str,
    exp_id: str,
    *,
    benchmark: str = DEFAULT_BENCHMARK_NAME,
) -> Path:
    return common_experiment_dir(Path(stablewm_home), tag, exp_id, benchmark=benchmark)


def next_exp_id(rows: list[dict]) -> str:
    highest = 0
    for row in rows:
        match = EXP_ID_PATTERN.match(row.get("exp_id", ""))
        if match is None:
            continue
        highest = max(highest, int(match.group(1)))
    return f"exp_{highest + 1:04d}"


def _parse_float(value):
    if value in ("", None):
        return None
    return float(value)


def _best_success_rate(rows: list[dict]):
    scores = [
        _parse_float(row.get("success_rate"))
        for row in rows
        if row.get("status") != "crash"
    ]
    scores = [score for score in scores if score is not None]
    return None if not scores else max(scores)


def _count_trailing_status(rows: list[dict], status: str) -> int:
    count = 0
    for row in reversed(rows):
        if row.get("status") != status:
            break
        count += 1
    return count


def stop_decision(rows: list[dict], target_success_rate: float = TARGET_SUCCESS_RATE) -> dict:
    best_success_rate = _best_success_rate(rows)
    decision = {
        "stop": False,
        "reason": None,
        "total_experiments": len(rows),
        "consecutive_crashes": _count_trailing_status(rows, "crash"),
        "consecutive_non_improving": _count_trailing_status(rows, "discard"),
        "best_success_rate": best_success_rate,
    }

    if best_success_rate is not None and best_success_rate >= target_success_rate:
        decision["stop"] = True
        decision["reason"] = "target_success_rate"
    elif len(rows) >= MAX_EXPERIMENTS:
        decision["stop"] = True
        decision["reason"] = "max_experiments"
    elif decision["consecutive_crashes"] >= MAX_CONSECUTIVE_CRASHES:
        decision["stop"] = True
        decision["reason"] = "consecutive_crashes"
    elif decision["consecutive_non_improving"] >= MAX_CONSECUTIVE_NON_IMPROVING:
        decision["stop"] = True
        decision["reason"] = "consecutive_non_improving"

    return decision


def build_train_stage_command(
    repo_root: Path,
    tag: str,
    exp_id: str,
    max_steps: int,
    stablewm_home: Path,
    *,
    benchmark: str = DEFAULT_BENCHMARK_NAME,
    extra_overrides=None,
) -> list[str]:
    command = [
        "python3",
        "tools/run_candidate.py",
        "--benchmark",
        benchmark,
        "--tag",
        tag,
        "--exp-id",
        exp_id,
        "--repo-root",
        str(Path(repo_root)),
        "--stablewm-home",
        str(Path(stablewm_home)),
        "--max-steps",
        str(max_steps),
    ]
    for override in flatten_cli_overrides(extra_overrides):
        command.extend(["--override", override])
    return command


def build_eval_stage_command(
    repo_root: Path,
    tag: str,
    exp_id: str,
    stablewm_home: Path,
    *,
    benchmark: str = DEFAULT_BENCHMARK_NAME,
    extra_overrides=None,
) -> list[str]:
    command = [
        "python3",
        "tools/eval_candidate.py",
        "--benchmark",
        benchmark,
        "--tag",
        tag,
        "--exp-id",
        exp_id,
        "--repo-root",
        str(Path(repo_root)),
        "--stablewm-home",
        str(Path(stablewm_home)),
    ]
    for override in flatten_cli_overrides(extra_overrides):
        command.extend(["--override", override])
    return command


def build_results_stage_command(
    tag: str,
    exp_id: str,
    stablewm_home: Path,
    description: str,
    *,
    benchmark: str = DEFAULT_BENCHMARK_NAME,
    status: str | None = None,
    success_rate: float | None = None,
    eval_wall_sec: float | None = None,
) -> list[str]:
    command = [
        "python3",
        "tools/update_results.py",
        "--benchmark",
        benchmark,
        "--tag",
        tag,
        "--exp-id",
        exp_id,
        "--stablewm-home",
        str(Path(stablewm_home)),
        "--description",
        description,
    ]
    if status is not None:
        command.extend(["--status", status])
    if success_rate is not None:
        command.extend(["--success-rate", str(success_rate)])
    if eval_wall_sec is not None:
        command.extend(["--eval-wall-sec", str(eval_wall_sec)])
    return command


def parse_last_json_object(stdout: str) -> dict:
    decoder = json.JSONDecoder()
    last_payload = None
    for match in re.finditer(r"{", stdout):
        try:
            payload, end = decoder.raw_decode(stdout[match.start() :])
        except json.JSONDecodeError:
            continue
        if stdout[match.start() + end :].strip():
            continue
        last_payload = payload
    if last_payload is None:
        raise ValueError("Could not parse trailing JSON object from stdout")
    return last_payload


def run_json_command(command: list[str], cwd: Path, env: dict) -> dict:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    result = {
        "command": command,
        "wrapper_returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }
    if stdout:
        try:
            result["output"] = parse_last_json_object(stdout)
        except ValueError as exc:
            result["output"] = None
            result["parse_error"] = str(exc)
    else:
        result["output"] = None
    return result


def _summary_status_from_results(results_stage: dict, fallback: str) -> str:
    output = results_stage.get("output") or {}
    row = output.get("row") or {}
    return row.get("status", fallback)


def _summary_success_rate(results_stage: dict, eval_stage: dict | None):
    output = results_stage.get("output") or {}
    row = output.get("row") or {}
    success_rate = row.get("success_rate")
    if success_rate not in ("", None):
        return _parse_float(success_rate)
    if eval_stage is None:
        return None
    return _parse_float((eval_stage.get("output") or {}).get("success_rate"))


def run_loop(
    tag: str,
    description: str,
    repo_root: Path,
    stablewm_home: Path,
    max_steps: int,
    *,
    benchmark: str = DEFAULT_BENCHMARK_NAME,
    target_success_rate: float = TARGET_SUCCESS_RATE,
    exp_id: str | None = None,
    dry_run: bool = False,
    train_overrides=None,
    eval_overrides=None,
    command_runner=run_json_command,
) -> dict:
    repo_root = Path(repo_root).expanduser().resolve()
    stablewm_home = Path(stablewm_home).expanduser().resolve()
    rows = load_results_rows(results_table_path(stablewm_home, tag, benchmark=benchmark))
    decision = stop_decision(rows, target_success_rate=target_success_rate)

    summary = {
        "benchmark": benchmark,
        "tag": tag,
        "description": description,
        "repo_root": str(repo_root),
        "stablewm_home": str(stablewm_home),
        "stop_decision_before_run": decision,
    }

    if decision["stop"]:
        summary["status"] = "stopped"
        summary["reason"] = decision["reason"]
        return summary

    exp_id = exp_id or next_exp_id(rows)
    exp_dir = experiment_dir(stablewm_home, tag, exp_id, benchmark=benchmark)
    train_command = build_train_stage_command(
        repo_root=repo_root,
        tag=tag,
        exp_id=exp_id,
        max_steps=max_steps,
        stablewm_home=stablewm_home,
        benchmark=benchmark,
        extra_overrides=train_overrides,
    )
    eval_command = build_eval_stage_command(
        repo_root=repo_root,
        tag=tag,
        exp_id=exp_id,
        stablewm_home=stablewm_home,
        benchmark=benchmark,
        extra_overrides=eval_overrides,
    )
    results_command = build_results_stage_command(
        tag=tag,
        exp_id=exp_id,
        stablewm_home=stablewm_home,
        description=description,
        benchmark=benchmark,
    )
    summary.update(
        {
            "exp_id": exp_id,
            "exp_dir": str(exp_dir),
        }
    )

    if dry_run:
        summary.update(
            {
                "status": "dry_run",
                "train_command": train_command,
                "eval_command": eval_command,
                "results_command": results_command,
            }
        )
        return summary

    env = os.environ.copy()
    env["STABLEWM_HOME"] = str(stablewm_home)

    train_stage = command_runner(train_command, cwd=repo_root, env=env)
    summary["train"] = train_stage
    train_output = train_stage.get("output") or {}
    actual_exp_dir = Path(train_output.get("exp_dir", exp_dir))
    object_checkpoint = train_output.get("object_checkpoint") or train_output.get("latest_object_ckpt")
    object_checkpoint_exists = train_output.get("object_checkpoint_exists")
    if object_checkpoint_exists is None:
        object_checkpoint_exists = bool(object_checkpoint)
    train_failed = (
        train_stage.get("wrapper_returncode", 0) != 0
        or train_output.get("returncode") != 0
        or not object_checkpoint
        or not object_checkpoint_exists
    )

    if train_failed:
        crash_command = build_results_stage_command(
            tag=tag,
            exp_id=exp_id,
            stablewm_home=stablewm_home,
            description=description,
            benchmark=benchmark,
            status="crash",
        )
        results_stage = command_runner(crash_command, cwd=repo_root, env=env)
        summary["results"] = results_stage
        summary["status"] = _summary_status_from_results(results_stage, "crash")
        summary["reason"] = "missing_object_ckpt" if not object_checkpoint_exists else "training_failed"
        summary["success_rate"] = None
        summary["exp_dir"] = str(actual_exp_dir)
        return summary

    eval_stage = command_runner(eval_command, cwd=repo_root, env=env)
    summary["eval"] = eval_stage
    eval_output = eval_stage.get("output") or {}
    eval_failed = (
        eval_stage.get("wrapper_returncode", 0) != 0
        or eval_output.get("returncode") != 0
        or eval_output.get("success_rate") is None
    )

    if eval_failed:
        results_command = build_results_stage_command(
            tag=tag,
            exp_id=exp_id,
            stablewm_home=stablewm_home,
            description=description,
            benchmark=benchmark,
            status="crash",
        )
    else:
        results_command = build_results_stage_command(
            tag=tag,
            exp_id=exp_id,
            stablewm_home=stablewm_home,
            description=description,
            benchmark=benchmark,
            success_rate=eval_output.get("success_rate"),
            eval_wall_sec=eval_output.get("eval_wall_sec"),
        )

    results_stage = command_runner(results_command, cwd=repo_root, env=env)
    summary["results"] = results_stage
    summary["status"] = _summary_status_from_results(
        results_stage,
        "crash" if eval_failed else "discard",
    )
    summary["success_rate"] = _summary_success_rate(results_stage, None if eval_failed else eval_stage)
    summary["exp_dir"] = str(actual_exp_dir)
    if not eval_failed and eval_output.get("eval_wall_sec") is not None:
        summary["eval_wall_sec"] = eval_output.get("eval_wall_sec")
    if not eval_failed and eval_output.get("evaluation_time") is not None:
        summary["evaluation_time"] = eval_output.get("evaluation_time")
    if eval_failed:
        summary["reason"] = "evaluation_failed"
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one native autoresearch candidate loop for LeWM."
    )
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK_NAME, choices=benchmark_names())
    parser.add_argument("--tag", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--stablewm-home", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--target-success-rate", type=float, default=TARGET_SUCCESS_RATE)
    parser.add_argument("--exp-id", default=None)
    parser.add_argument(
        "--train-override",
        action="append",
        default=[],
        help="Additional Hydra override passed through to tools/run_candidate.py.",
    )
    parser.add_argument(
        "--eval-override",
        action="append",
        default=[],
        help="Additional Hydra override passed through to tools/eval_candidate.py.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    stablewm_home = args.stablewm_home or resolve_stablewm_home()
    max_steps = args.max_steps
    if max_steps is None:
        max_steps = benchmark_config(args.benchmark).default_max_steps
    summary = run_loop(
        tag=args.tag,
        description=args.description,
        repo_root=args.repo_root,
        stablewm_home=stablewm_home,
        max_steps=max_steps,
        benchmark=args.benchmark,
        target_success_rate=args.target_success_rate,
        exp_id=args.exp_id,
        dry_run=args.dry_run,
        train_overrides=args.train_override,
        eval_overrides=args.eval_override,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
