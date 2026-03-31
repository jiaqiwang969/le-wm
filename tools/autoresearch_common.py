from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    train_data_alias: str
    eval_config_name: str
    default_results_filename: str
    default_max_steps: int


DEFAULT_BENCHMARK_NAME = "pusht"
DEFAULT_MODEL_NAME = "lewm_epoch_1"

BENCHMARK_CONFIGS = {
    "pusht": BenchmarkConfig(
        name="pusht",
        train_data_alias="pusht",
        eval_config_name="pusht.yaml",
        default_results_filename="pusht_results.txt",
        default_max_steps=2000,
    ),
    "cube": BenchmarkConfig(
        name="cube",
        train_data_alias="ogb",
        eval_config_name="cube.yaml",
        default_results_filename="ogb_cube_results.txt",
        default_max_steps=500,
    ),
}

BENCHMARK_NAME = DEFAULT_BENCHMARK_NAME
DEFAULT_RESULTS_FILENAME = BENCHMARK_CONFIGS[DEFAULT_BENCHMARK_NAME].default_results_filename
DEFAULT_MAX_STEPS = BENCHMARK_CONFIGS[DEFAULT_BENCHMARK_NAME].default_max_steps


def benchmark_names() -> list[str]:
    return sorted(BENCHMARK_CONFIGS)


def benchmark_config(benchmark: str = DEFAULT_BENCHMARK_NAME) -> BenchmarkConfig:
    try:
        return BENCHMARK_CONFIGS[benchmark]
    except KeyError as exc:
        supported = ", ".join(benchmark_names())
        raise ValueError(f"Unsupported benchmark '{benchmark}'. Supported: {supported}") from exc


def repo_root_from_arg(repo_root: str | Path | None = None) -> Path:
    if repo_root is not None:
        return Path(repo_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def stablewm_home_from_arg(
    stablewm_home: str | Path | None = None,
    *,
    env: dict[str, str] | None = None,
    home: str | Path | None = None,
) -> Path:
    if stablewm_home is not None:
        return Path(stablewm_home).expanduser().resolve()

    runtime_env = env if env is not None else os.environ
    configured = runtime_env.get("STABLEWM_HOME")
    if configured:
        return Path(configured).expanduser().resolve()

    home_dir = Path(home).expanduser().resolve() if home is not None else Path.home()
    preferred = home_dir / "stablewm"
    if preferred.exists():
        return preferred

    fallback = home_dir / ".stable-wm"
    if fallback.exists():
        return fallback

    return fallback


def tag_dir(stablewm_home: Path, tag: str, *, benchmark: str = DEFAULT_BENCHMARK_NAME) -> Path:
    return stablewm_home / "autoresearch" / benchmark_config(benchmark).name / tag


def experiment_dir(
    stablewm_home: Path,
    tag: str,
    exp_id: str,
    *,
    benchmark: str = DEFAULT_BENCHMARK_NAME,
) -> Path:
    return tag_dir(stablewm_home, tag, benchmark=benchmark) / exp_id


def subdir_for_experiment(stablewm_home: Path, exp_dir: Path) -> str:
    return exp_dir.relative_to(stablewm_home).as_posix()


def policy_name_for_model(stablewm_home: Path, exp_dir: Path, model_name: str) -> str:
    return f"{subdir_for_experiment(stablewm_home, exp_dir)}/{model_name}"


def summary_path(exp_dir: Path, name: str) -> Path:
    return exp_dir / f"{name}_summary.json"


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_commit_with_dirty_suffix(repo_root: Path) -> str:
    try:
        head = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo_root), "diff", "--quiet"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        untracked = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "non-git-worktree"

    if dirty != 0 or untracked:
        return f"{head}-dirty"
    return head


def stream_subprocess_output(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> int:
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND: " + " ".join(cmd) + "\n\n")
        with subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                log_file.write(line)
            return proc.wait()


def flatten_cli_overrides(values: Iterable[str] | None) -> list[str]:
    return [value for value in (values or []) if value]
