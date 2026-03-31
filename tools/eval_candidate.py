from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from autoresearch_common import (
    DEFAULT_BENCHMARK_NAME,
    DEFAULT_MODEL_NAME,
    benchmark_config,
    benchmark_names,
    experiment_dir,
    flatten_cli_overrides,
    policy_name_for_model as common_policy_name_for_model,
    repo_root_from_arg,
    stablewm_home_from_arg,
    stream_subprocess_output,
    summary_path,
    write_json,
)


SUCCESS_RATE_PATTERN = re.compile(r"'success_rate':\s*([0-9]+(?:\.[0-9]+)?)")
EVAL_TIME_PATTERN = re.compile(r"evaluation_time:\s*([0-9]+(?:\.[0-9]+)?)\s+seconds")


def results_file_for_experiment(
    exp_dir: Path,
    filename: str | None = None,
    *,
    benchmark: str = DEFAULT_BENCHMARK_NAME,
) -> Path:
    resolved_filename = filename or benchmark_config(benchmark).default_results_filename
    return exp_dir / resolved_filename


def policy_name_for_model(*, stablewm_home: Path, exp_dir: Path, model_name: str) -> str:
    return common_policy_name_for_model(stablewm_home, exp_dir, model_name)


def build_eval_command(
    *,
    repo_root: Path,
    policy_name: str,
    output_filename: str | None = None,
    benchmark: str = DEFAULT_BENCHMARK_NAME,
    overrides: list[str] | None = None,
) -> list[str]:
    config = benchmark_config(benchmark)
    resolved_output_filename = output_filename or config.default_results_filename
    cmd = [
        "python3",
        str(repo_root / "eval.py"),
        f"--config-name={config.eval_config_name}",
        f"policy={policy_name}",
        f"output.filename={resolved_output_filename}",
    ]
    cmd.extend(flatten_cli_overrides(overrides))
    return cmd


def parse_eval_metrics(results_text: str) -> dict[str, float]:
    success_match = SUCCESS_RATE_PATTERN.search(results_text)
    time_match = EVAL_TIME_PATTERN.search(results_text)
    if success_match is None:
        raise ValueError("Could not parse success_rate from results text")
    if time_match is None:
        raise ValueError("Could not parse evaluation_time from results text")
    return {
        "success_rate": float(success_match.group(1)),
        "evaluation_time": float(time_match.group(1)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official eval for one candidate.")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK_NAME, choices=benchmark_names())
    parser.add_argument("--tag", required=True)
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--repo-root")
    parser.add_argument("--stablewm-home")
    parser.add_argument("--results-file")
    parser.add_argument("--output-filename")
    parser.add_argument("--override", action="append", default=[], dest="overrides")
    parser.add_argument("--parse-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_arg(args.repo_root)
    stablewm_home = stablewm_home_from_arg(args.stablewm_home)
    exp_dir = experiment_dir(stablewm_home, args.tag, args.exp_id, benchmark=args.benchmark)
    results_path = Path(args.results_file) if args.results_file else results_file_for_experiment(
        exp_dir,
        args.output_filename,
        benchmark=args.benchmark,
    )
    eval_log = exp_dir / "eval.log"
    summary_file = summary_path(exp_dir, "eval")

    if args.parse_only:
        metrics = parse_eval_metrics(results_path.read_text(encoding="utf-8"))
        payload = {
            "results_file": str(results_path),
            **metrics,
        }
        print(json.dumps(payload, indent=2))
        return 0

    policy_name = policy_name_for_model(
        stablewm_home=stablewm_home,
        exp_dir=exp_dir,
        model_name=args.model_name,
    )
    cmd = build_eval_command(
        repo_root=repo_root,
        policy_name=policy_name,
        output_filename=args.output_filename,
        benchmark=args.benchmark,
        overrides=args.overrides,
    )
    exp_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["STABLEWM_HOME"] = str(stablewm_home)

    start = time.monotonic()
    returncode = stream_subprocess_output(cmd, cwd=repo_root, env=env, log_path=eval_log)
    wall_sec = time.monotonic() - start

    summary = {
        "benchmark": args.benchmark,
        "command": cmd,
        "policy_path": policy_name,
        "results_file": str(results_path),
        "eval_log": str(eval_log),
        "summary_path": str(summary_file),
        "returncode": returncode,
        "eval_wall_sec": wall_sec,
    }

    if returncode == 0 and results_path.exists():
        summary.update(parse_eval_metrics(results_path.read_text(encoding="utf-8")))

    write_json(summary_file, summary)
    print(json.dumps(summary, indent=2))
    return 0 if returncode == 0 else returncode


if __name__ == "__main__":
    sys.exit(main())
