from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from autoresearch_common import (
    DEFAULT_BENCHMARK_NAME,
    DEFAULT_MAX_STEPS,
    DEFAULT_MODEL_NAME,
    benchmark_config,
    benchmark_names,
    experiment_dir as common_experiment_dir,
    flatten_cli_overrides,
    repo_root_from_arg,
    stablewm_home_from_arg,
    stream_subprocess_output,
    subdir_for_experiment,
    summary_path,
    write_json,
)

MIN_MAX_STEPS = 2


def experiment_dir(
    stablewm_home: Path,
    tag: str,
    exp_id: str,
    *,
    benchmark: str = DEFAULT_BENCHMARK_NAME,
) -> Path:
    return common_experiment_dir(stablewm_home, tag, exp_id, benchmark=benchmark)


def validate_max_steps(max_steps: int) -> int:
    if max_steps < MIN_MAX_STEPS:
        raise ValueError(
            f"max_steps must be >= {MIN_MAX_STEPS}; max_steps=1 crashes the current scheduler setup"
        )
    return max_steps


def build_train_command(
    *,
    repo_root: Path,
    exp_dir: Path,
    max_steps: int = DEFAULT_MAX_STEPS,
    overrides: list[str] | None = None,
    stablewm_home: Path | None = None,
    benchmark: str = DEFAULT_BENCHMARK_NAME,
) -> list[str]:
    validated_max_steps = validate_max_steps(max_steps)
    resolved_stablewm_home = stablewm_home or exp_dir.parents[3]
    config = benchmark_config(benchmark)
    cmd = [
        "python3",
        str(repo_root / "train.py"),
        f"data={config.train_data_alias}",
        "wandb.enabled=False",
        f"+trainer.max_steps={validated_max_steps}",
        f"subdir={subdir_for_experiment(resolved_stablewm_home, exp_dir)}",
    ]
    cmd.extend(flatten_cli_overrides(overrides))
    return cmd


def artifact_paths(exp_dir: Path) -> dict[str, Path]:
    return {
        "train_log": exp_dir / "train.log",
        "config": exp_dir / "config.yaml",
        "object_checkpoint": exp_dir / f"{DEFAULT_MODEL_NAME}_object.ckpt",
        "weights_checkpoint": exp_dir / "lewm_weights.ckpt",
        "summary": summary_path(exp_dir, "train"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one bounded training candidate.")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK_NAME, choices=benchmark_names())
    parser.add_argument("--tag", required=True)
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--repo-root")
    parser.add_argument("--stablewm-home")
    parser.add_argument("--override", action="append", default=[], dest="overrides")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_arg(args.repo_root)
    stablewm_home = stablewm_home_from_arg(args.stablewm_home)
    exp_dir = experiment_dir(stablewm_home, args.tag, args.exp_id, benchmark=args.benchmark)
    paths = artifact_paths(exp_dir)
    max_steps = args.max_steps
    if max_steps is None:
        max_steps = benchmark_config(args.benchmark).default_max_steps
    try:
        cmd = build_train_command(
            repo_root=repo_root,
            exp_dir=exp_dir,
            max_steps=max_steps,
            overrides=args.overrides,
            stablewm_home=stablewm_home,
            benchmark=args.benchmark,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    summary = {
        "benchmark": args.benchmark,
        "command": cmd,
        "exp_dir": str(exp_dir),
        "object_checkpoint": str(paths["object_checkpoint"]),
        "weights_checkpoint": str(paths["weights_checkpoint"]),
        "train_log": str(paths["train_log"]),
        "summary_path": str(paths["summary"]),
    }

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0

    exp_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["STABLEWM_HOME"] = str(stablewm_home)

    start = time.monotonic()
    returncode = stream_subprocess_output(cmd, cwd=repo_root, env=env, log_path=paths["train_log"])
    wall_sec = time.monotonic() - start

    summary.update(
        {
            "returncode": returncode,
            "train_wall_sec": wall_sec,
            "config_exists": paths["config"].exists(),
            "object_checkpoint_exists": paths["object_checkpoint"].exists(),
            "weights_checkpoint_exists": paths["weights_checkpoint"].exists(),
        }
    )
    write_json(paths["summary"], summary)
    print(json.dumps(summary, indent=2))
    return returncode


if __name__ == "__main__":
    sys.exit(main())
