from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import eval_candidate
from autoresearch_common import (
    DEFAULT_BENCHMARK_NAME,
    DEFAULT_MODEL_NAME,
    benchmark_names,
    experiment_dir,
    git_commit_with_dirty_suffix,
    policy_name_for_model,
    repo_root_from_arg,
    stablewm_home_from_arg,
    summary_path,
    tag_dir,
)


RESULT_COLUMNS = [
    "exp_id",
    "parent_exp_id",
    "commit",
    "status",
    "success_rate",
    "train_wall_sec",
    "eval_wall_sec",
    "peak_vram_gb",
    "policy_path",
    "description",
]

LEGACY_RESULT_COLUMNS = [
    "timestamp_utc",
    "exp_id",
    "status",
    "success_rate",
    "evaluation_time",
    "description",
    "exp_dir",
    "results_file",
]


def results_path_for_tag(
    stablewm_home: Path,
    tag: str,
    *,
    benchmark: str = DEFAULT_BENCHMARK_NAME,
) -> Path:
    return tag_dir(stablewm_home, tag, benchmark=benchmark) / "results.tsv"


def _read_tabular_lines(results_path: Path) -> list[str]:
    return [
        line.rstrip("\n")
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _current_row_from_values(values: list[str]) -> dict[str, str]:
    return dict(zip(RESULT_COLUMNS, values))


def _policy_path_from_exp_dir(results_path: Path, exp_dir_raw: str, exp_id: str) -> str:
    stablewm_home = results_path.parents[3]
    exp_dir = Path(exp_dir_raw) if exp_dir_raw else results_path.parent / exp_id
    try:
        return policy_name_for_model(stablewm_home, exp_dir, DEFAULT_MODEL_NAME)
    except ValueError:
        derived_exp_dir = results_path.parent / exp_id
        return policy_name_for_model(stablewm_home, derived_exp_dir, DEFAULT_MODEL_NAME)


def _current_row_from_legacy_values(results_path: Path, values: list[str]) -> dict[str, str]:
    legacy_row = dict(zip(LEGACY_RESULT_COLUMNS, values))
    exp_id = legacy_row.get("exp_id", "")
    return {
        "exp_id": exp_id,
        "parent_exp_id": "",
        "commit": "",
        "status": legacy_row.get("status", ""),
        "success_rate": legacy_row.get("success_rate", ""),
        "train_wall_sec": "",
        "eval_wall_sec": legacy_row.get("evaluation_time", ""),
        "peak_vram_gb": "",
        "policy_path": _policy_path_from_exp_dir(
            results_path,
            legacy_row.get("exp_dir", ""),
            exp_id,
        ),
        "description": legacy_row.get("description", ""),
    }


def _normalized_existing_rows(results_path: Path) -> list[dict[str, str]]:
    lines = _read_tabular_lines(results_path)
    if not lines:
        return []

    header = lines[0].split("\t")
    if header == RESULT_COLUMNS:
        with results_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))

    if header != LEGACY_RESULT_COLUMNS:
        raise ValueError(f"Unsupported results.tsv schema in {results_path}")

    rows = []
    for index, line in enumerate(lines[1:], start=2):
        values = line.split("\t")
        if len(values) == len(LEGACY_RESULT_COLUMNS):
            rows.append(_current_row_from_legacy_values(results_path, values))
            continue
        if len(values) == len(RESULT_COLUMNS):
            rows.append(_current_row_from_values(values))
            continue
        raise ValueError(
            f"Unsupported row width in {results_path}:{index}; got {len(values)} columns"
        )
    return rows


def _write_results_table(results_path: Path, rows: list[dict[str, str]]) -> None:
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in RESULT_COLUMNS})


def ensure_results_table(results_path: Path) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if not results_path.exists():
        _write_results_table(results_path, [])
        return

    lines = _read_tabular_lines(results_path)
    if not lines:
        _write_results_table(results_path, [])
        return

    header = lines[0].split("\t")
    if header == RESULT_COLUMNS:
        return

    backup_path = results_path.with_suffix(".tsv.legacy.bak")
    backup_path.write_text(results_path.read_text(encoding="utf-8"), encoding="utf-8")
    _write_results_table(results_path, _normalized_existing_rows(results_path))


def load_results(results_path: Path) -> list[dict[str, str]]:
    ensure_results_table(results_path)
    if not results_path.exists():
        return []
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader)


def append_result(results_path: Path, row: dict[str, str]) -> None:
    ensure_results_table(results_path)
    full_row = {column: row.get(column, "") for column in RESULT_COLUMNS}
    with results_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, delimiter="\t")
        writer.writerow(full_row)


def _success_value(row: dict[str, str]) -> float | None:
    raw = row.get("success_rate", "")
    if raw in ("", None):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def best_result(rows: list[dict[str, str]]) -> dict[str, str] | None:
    successful_rows = [row for row in rows if _success_value(row) is not None and row.get("status") != "crash"]
    if not successful_rows:
        return None
    return max(successful_rows, key=lambda row: _success_value(row) or float("-inf"))


def classify_result(previous_rows: list[dict[str, str]], success_rate: float | None) -> str:
    if success_rate is None:
        return "crash"
    current_best = best_result(previous_rows)
    if current_best is None:
        return "keep"
    return "keep" if success_rate > float(current_best["success_rate"]) else "discard"


def update_best_link(tag_root: Path, exp_dir: Path) -> Path:
    best_link = tag_root / "best"
    if best_link.is_symlink() or best_link.exists():
        best_link.unlink()
    best_link.symlink_to(exp_dir.resolve(), target_is_directory=True)
    return best_link


def _load_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_row(
    *,
    stablewm_home: Path,
    repo_root: Path,
    benchmark: str,
    tag: str,
    exp_id: str,
    previous_rows: list[dict[str, str]],
    description: str,
    explicit_status: str | None,
    explicit_success_rate: float | None,
    explicit_train_wall_sec: float | None,
    explicit_eval_wall_sec: float | None,
    explicit_policy_path: str | None,
    explicit_parent_exp_id: str | None,
    explicit_peak_vram_gb: str | None,
    explicit_commit: str | None,
    model_name: str,
) -> tuple[dict[str, str], Path]:
    exp_dir = experiment_dir(stablewm_home, tag, exp_id, benchmark=benchmark)
    train_summary = _load_summary(summary_path(exp_dir, "train"))
    eval_summary = _load_summary(summary_path(exp_dir, "eval"))

    success_rate = explicit_success_rate
    if success_rate is None:
        success_rate = eval_summary.get("success_rate")

    status = explicit_status or classify_result(previous_rows, success_rate)
    parent_exp_id = explicit_parent_exp_id
    if parent_exp_id is None:
        current_best = best_result(previous_rows)
        parent_exp_id = current_best["exp_id"] if current_best else ""

    policy_path = explicit_policy_path or eval_summary.get("policy_path", "")
    if not policy_path:
        object_ckpt = exp_dir / f"{model_name}_object.ckpt"
        if object_ckpt.exists():
            policy_path = policy_name_for_model(stablewm_home, exp_dir, model_name)

    row = {
        "exp_id": exp_id,
        "parent_exp_id": parent_exp_id or "",
        "commit": explicit_commit or git_commit_with_dirty_suffix(repo_root),
        "status": status,
        "success_rate": "" if success_rate is None else str(success_rate),
        "train_wall_sec": str(
            explicit_train_wall_sec
            if explicit_train_wall_sec is not None
            else train_summary.get("train_wall_sec", "")
        ),
        "eval_wall_sec": str(
            explicit_eval_wall_sec
            if explicit_eval_wall_sec is not None
            else eval_summary.get("eval_wall_sec", "")
        ),
        "peak_vram_gb": explicit_peak_vram_gb or "",
        "policy_path": policy_path,
        "description": description,
    }
    return row, exp_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append one candidate result to results.tsv.")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK_NAME, choices=benchmark_names())
    parser.add_argument("--tag", required=True)
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--status", choices=["keep", "discard", "crash"])
    parser.add_argument("--success-rate", type=float)
    parser.add_argument("--train-wall-sec", type=float)
    parser.add_argument("--eval-wall-sec", type=float)
    parser.add_argument("--policy-path")
    parser.add_argument("--parent-exp-id")
    parser.add_argument("--peak-vram-gb")
    parser.add_argument("--commit")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--repo-root")
    parser.add_argument("--stablewm-home")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_arg(args.repo_root)
    stablewm_home = stablewm_home_from_arg(args.stablewm_home)
    results_path = results_path_for_tag(stablewm_home, args.tag, benchmark=args.benchmark)
    previous_rows = load_results(results_path)
    row, exp_dir = build_row(
        stablewm_home=stablewm_home,
        repo_root=repo_root,
        benchmark=args.benchmark,
        tag=args.tag,
        exp_id=args.exp_id,
        previous_rows=previous_rows,
        description=args.description,
        explicit_status=args.status,
        explicit_success_rate=args.success_rate,
        explicit_train_wall_sec=args.train_wall_sec,
        explicit_eval_wall_sec=args.eval_wall_sec,
        explicit_policy_path=args.policy_path,
        explicit_parent_exp_id=args.parent_exp_id,
        explicit_peak_vram_gb=args.peak_vram_gb,
        explicit_commit=args.commit,
        model_name=args.model_name,
    )

    payload = {
        "benchmark": args.benchmark,
        "exp_dir": str(exp_dir),
        "results_path": str(results_path),
        "row": row,
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    append_result(results_path, row)
    if row["status"] == "keep":
        payload["best_link"] = str(
            update_best_link(tag_dir(stablewm_home, args.tag, benchmark=args.benchmark), exp_dir)
        )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
