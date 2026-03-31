import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"


def load_tool_module(name: str):
    module_path = TOOLS_DIR / f"{name}.py"
    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))
    spec = importlib.util.spec_from_file_location(f"tools.{name}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AutoresearchLoopHelpersTest(unittest.TestCase):
    def test_tag_root_uses_cube_namespace_when_requested(self):
        autoresearch_loop = load_tool_module("autoresearch_loop")

        tag_root = autoresearch_loop.tag_root(
            Path("/tmp/stablewm"),
            "smoke-loop",
            benchmark="cube",
        )

        self.assertEqual(tag_root, Path("/tmp/stablewm/autoresearch/cube/smoke-loop"))

    def test_next_exp_id_is_exp_0001_when_results_are_empty(self):
        autoresearch_loop = load_tool_module("autoresearch_loop")

        exp_id = autoresearch_loop.next_exp_id([])

        self.assertEqual(exp_id, "exp_0001")

    def test_next_exp_id_increments_latest_recorded_experiment(self):
        autoresearch_loop = load_tool_module("autoresearch_loop")

        exp_id = autoresearch_loop.next_exp_id(
            [
                {"exp_id": "exp_0001"},
                {"exp_id": "exp_0009"},
            ]
        )

        self.assertEqual(exp_id, "exp_0010")

    def test_load_results_rows_reads_existing_results_tsv(self):
        autoresearch_loop = load_tool_module("autoresearch_loop")

        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "results.tsv"
            results_path.write_text(
                "\t".join(
                    [
                        "timestamp_utc",
                        "exp_id",
                        "status",
                        "success_rate",
                        "evaluation_time",
                        "description",
                        "exp_dir",
                        "results_file",
                    ]
                )
                + "\n"
                + "\t".join(
                    [
                        "2026-03-30T00:00:00+00:00",
                        "exp_0001",
                        "keep",
                        "98.0",
                        "488.0",
                        "baseline",
                        "/tmp/stablewm/autoresearch/pusht/smoke/exp_0001",
                        "/tmp/stablewm/autoresearch/pusht/smoke/exp_0001/pusht_results.txt",
                    ]
                )
                + "\n"
            )

            rows = autoresearch_loop.load_results_rows(results_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["exp_id"], "exp_0001")
        self.assertEqual(rows[0]["success_rate"], "98.0")

    def test_stop_when_total_experiments_reaches_limit(self):
        autoresearch_loop = load_tool_module("autoresearch_loop")
        rows = [{"exp_id": f"exp_{index:04d}", "status": "discard"} for index in range(1, 21)]

        decision = autoresearch_loop.stop_decision(rows)

        self.assertTrue(decision["stop"])
        self.assertEqual(decision["reason"], "max_experiments")

    def test_stop_after_three_consecutive_crashes(self):
        autoresearch_loop = load_tool_module("autoresearch_loop")
        rows = [
            {"exp_id": "exp_0001", "status": "keep", "success_rate": "70.0"},
            {"exp_id": "exp_0002", "status": "crash", "success_rate": ""},
            {"exp_id": "exp_0003", "status": "crash", "success_rate": ""},
            {"exp_id": "exp_0004", "status": "crash", "success_rate": ""},
        ]

        decision = autoresearch_loop.stop_decision(rows)

        self.assertTrue(decision["stop"])
        self.assertEqual(decision["reason"], "consecutive_crashes")

    def test_stop_after_eight_consecutive_non_improving_runs(self):
        autoresearch_loop = load_tool_module("autoresearch_loop")
        rows = [{"exp_id": "exp_0001", "status": "keep", "success_rate": "80.0"}]
        rows.extend(
            {
                "exp_id": f"exp_{index:04d}",
                "status": "discard",
                "success_rate": "79.0",
            }
            for index in range(2, 10)
        )

        decision = autoresearch_loop.stop_decision(rows)

        self.assertTrue(decision["stop"])
        self.assertEqual(decision["reason"], "consecutive_non_improving")

    def test_stop_when_target_success_rate_is_reached(self):
        autoresearch_loop = load_tool_module("autoresearch_loop")
        rows = [
            {"exp_id": "exp_0001", "status": "keep", "success_rate": "89.0"},
            {"exp_id": "exp_0002", "status": "keep", "success_rate": "90.0"},
        ]

        decision = autoresearch_loop.stop_decision(rows)

        self.assertTrue(decision["stop"])
        self.assertEqual(decision["reason"], "target_success_rate")

    def test_stop_decision_respects_custom_target_success_rate(self):
        autoresearch_loop = load_tool_module("autoresearch_loop")
        rows = [
            {"exp_id": "exp_0001", "status": "keep", "success_rate": "96.0"},
        ]

        decision = autoresearch_loop.stop_decision(rows, target_success_rate=97.0)

        self.assertFalse(decision["stop"])
        self.assertIsNone(decision["reason"])

    def test_train_stage_command_calls_run_candidate_tool(self):
        autoresearch_loop = load_tool_module("autoresearch_loop")

        command = autoresearch_loop.build_train_stage_command(
            repo_root=Path("/repo/le-wm"),
            tag="smoke-loop",
            exp_id="exp_0007",
            max_steps=2,
            stablewm_home=Path("/tmp/stablewm"),
            benchmark="cube",
        )

        self.assertEqual(command[:2], ["python3", "tools/run_candidate.py"])
        self.assertIn("--benchmark", command)
        self.assertIn("cube", command)
        self.assertIn("smoke-loop", command)
        self.assertIn("--exp-id", command)
        self.assertIn("exp_0007", command)
        self.assertIn("--max-steps", command)
        self.assertIn("2", command)
        self.assertIn("--stablewm-home", command)
        self.assertIn("/tmp/stablewm", command)

    def test_train_stage_command_forwards_training_overrides(self):
        autoresearch_loop = load_tool_module("autoresearch_loop")

        command = autoresearch_loop.build_train_stage_command(
            repo_root=Path("/repo/le-wm"),
            tag="warmstart",
            exp_id="exp_0001",
            max_steps=2000,
            stablewm_home=Path("/tmp/stablewm"),
            extra_overrides=[
                "warm_start.enabled=True",
                "warm_start.checkpoint=pusht/lewm_weights.ckpt",
            ],
        )

        self.assertEqual(command.count("--override"), 2)
        self.assertIn("warm_start.enabled=True", command)
        self.assertIn("warm_start.checkpoint=pusht/lewm_weights.ckpt", command)

    def test_eval_stage_command_calls_eval_candidate_tool(self):
        autoresearch_loop = load_tool_module("autoresearch_loop")

        command = autoresearch_loop.build_eval_stage_command(
            repo_root=Path("/repo/le-wm"),
            tag="smoke-loop",
            exp_id="exp_0007",
            stablewm_home=Path("/tmp/stablewm"),
            benchmark="cube",
            extra_overrides=["eval.num_eval=10", "eval.goal_offset_steps=10"],
        )

        self.assertEqual(command[:2], ["python3", "tools/eval_candidate.py"])
        self.assertIn("--benchmark", command)
        self.assertIn("cube", command)
        self.assertIn("--tag", command)
        self.assertIn("smoke-loop", command)
        self.assertIn("--exp-id", command)
        self.assertIn("exp_0007", command)
        self.assertNotIn("--object-ckpt", command)
        self.assertEqual(command[-2:], ["--override", "eval.goal_offset_steps=10"])
        self.assertIn("--stablewm-home", command)
        self.assertIn("/tmp/stablewm", command)

    def test_results_stage_command_calls_update_results_tool(self):
        autoresearch_loop = load_tool_module("autoresearch_loop")

        command = autoresearch_loop.build_results_stage_command(
            tag="smoke-loop",
            exp_id="exp_0007",
            stablewm_home=Path("/tmp/stablewm"),
            description="controller smoke",
            benchmark="cube",
            status="discard",
            success_rate=2.0,
            eval_wall_sec=487.37,
        )

        self.assertEqual(command[:2], ["python3", "tools/update_results.py"])
        self.assertIn("--benchmark", command)
        self.assertIn("cube", command)
        self.assertIn("smoke-loop", command)
        self.assertIn("--exp-id", command)
        self.assertIn("exp_0007", command)
        self.assertIn("--description", command)
        self.assertIn("controller smoke", command)
        self.assertNotIn("--exp-dir", command)
        self.assertIn("--status", command)
        self.assertIn("discard", command)
        self.assertIn("--success-rate", command)
        self.assertIn("2.0", command)
        self.assertIn("--eval-wall-sec", command)
        self.assertIn("487.37", command)

    def test_extracts_summary_json_from_mixed_stdout(self):
        autoresearch_loop = load_tool_module("autoresearch_loop")

        payload = autoresearch_loop.parse_last_json_object(
            "line 1\nline 2\n{\n  \"returncode\": 0,\n  \"success_rate\": 60.0\n}\n"
        )

        self.assertEqual(payload["returncode"], 0)
        self.assertEqual(payload["success_rate"], 60.0)


class AutoresearchLoopExecutionTest(unittest.TestCase):
    def test_run_loop_stops_early_when_target_success_rate_is_already_met(self):
        autoresearch_loop = load_tool_module("autoresearch_loop")

        with tempfile.TemporaryDirectory() as tmpdir:
            stablewm_home = Path(tmpdir) / "stablewm"
            results_path = stablewm_home / "autoresearch" / "cube" / "smoke-loop" / "results.tsv"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            results_path.write_text(
                "\t".join(
                    [
                        "timestamp_utc",
                        "exp_id",
                        "status",
                        "success_rate",
                        "evaluation_time",
                        "description",
                        "exp_dir",
                        "results_file",
                    ]
                )
                + "\n"
                + "\t".join(
                    [
                        "2026-03-30T00:00:00+00:00",
                        "exp_0001",
                        "keep",
                        "98.0",
                        "488.0",
                        "baseline",
                        str(stablewm_home / "autoresearch" / "cube" / "smoke-loop" / "exp_0001"),
                        str(stablewm_home / "autoresearch" / "cube" / "smoke-loop" / "exp_0001" / "ogb_cube_results.txt"),
                    ]
                )
                + "\n"
            )

            summary = autoresearch_loop.run_loop(
                tag="smoke-loop",
                description="should stop",
                repo_root=Path("/repo/le-wm"),
                stablewm_home=stablewm_home,
                max_steps=2,
                benchmark="cube",
                command_runner=lambda *args, **kwargs: self.fail("command runner should not be used"),
            )

        self.assertEqual(summary["status"], "stopped")
        self.assertEqual(summary["reason"], "target_success_rate")

    def test_run_loop_allows_progress_when_custom_target_is_higher(self):
        autoresearch_loop = load_tool_module("autoresearch_loop")

        with tempfile.TemporaryDirectory() as tmpdir:
            stablewm_home = Path(tmpdir) / "stablewm"
            results_path = stablewm_home / "autoresearch" / "cube" / "smoke-loop" / "results.tsv"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            results_path.write_text(
                "\t".join(
                    [
                        "timestamp_utc",
                        "exp_id",
                        "status",
                        "success_rate",
                        "evaluation_time",
                        "description",
                        "exp_dir",
                        "results_file",
                    ]
                )
                + "\n"
                + "\t".join(
                    [
                        "2026-03-30T00:00:00+00:00",
                        "exp_0001",
                        "keep",
                        "96.0",
                        "488.0",
                        "baseline",
                        str(stablewm_home / "autoresearch" / "cube" / "smoke-loop" / "exp_0001"),
                        str(stablewm_home / "autoresearch" / "cube" / "smoke-loop" / "exp_0001" / "ogb_cube_results.txt"),
                    ]
                )
                + "\n"
            )

            summary = autoresearch_loop.run_loop(
                tag="smoke-loop",
                description="should continue",
                repo_root=Path("/repo/le-wm"),
                stablewm_home=stablewm_home,
                max_steps=2,
                benchmark="cube",
                target_success_rate=97.0,
                dry_run=True,
                command_runner=lambda *args, **kwargs: self.fail("dry-run should not execute child tools"),
            )

        self.assertEqual(summary["status"], "dry_run")
        self.assertEqual(summary["exp_id"], "exp_0002")

    def test_run_loop_dry_run_allocates_exp_id_without_running_commands(self):
        autoresearch_loop = load_tool_module("autoresearch_loop")

        with tempfile.TemporaryDirectory() as tmpdir:
            stablewm_home = Path(tmpdir) / "stablewm"

            summary = autoresearch_loop.run_loop(
                tag="smoke-loop",
                description="dry-run",
                repo_root=Path("/repo/le-wm"),
                stablewm_home=stablewm_home,
                max_steps=2,
                benchmark="cube",
                dry_run=True,
                command_runner=lambda *args, **kwargs: self.fail("dry-run should not execute child tools"),
            )

        self.assertEqual(summary["status"], "dry_run")
        self.assertEqual(summary["exp_id"], "exp_0001")
        self.assertIn("train_command", summary)
        self.assertIn("eval_command", summary)
        self.assertIn("results_command", summary)
        self.assertIn("--benchmark", summary["train_command"])
        self.assertIn("cube", summary["train_command"])

    def test_run_loop_records_crash_when_training_emits_no_object_checkpoint(self):
        autoresearch_loop = load_tool_module("autoresearch_loop")
        calls = []

        def fake_runner(command, cwd, env):
            calls.append(command)
            tool_name = command[1]
            if tool_name == "tools/run_candidate.py":
                return {
                    "wrapper_returncode": 0,
                    "output": {
                        "status": "completed",
                        "returncode": 0,
                        "exp_dir": "/tmp/stablewm/autoresearch/cube/smoke-loop/exp_0001",
                        "object_checkpoint": "/tmp/stablewm/autoresearch/cube/smoke-loop/exp_0001/lewm_epoch_1_object.ckpt",
                        "object_checkpoint_exists": False,
                    },
                }
            if tool_name == "tools/update_results.py":
                return {
                    "wrapper_returncode": 0,
                    "output": {
                        "row": {
                            "status": "crash",
                            "success_rate": "",
                        }
                    },
                }
            self.fail(f"unexpected tool invocation: {tool_name}")

        summary = autoresearch_loop.run_loop(
            tag="smoke-loop",
            description="missing checkpoint",
            repo_root=Path("/repo/le-wm"),
            stablewm_home=Path("/tmp/stablewm"),
            max_steps=2,
            benchmark="cube",
            command_runner=fake_runner,
        )

        self.assertEqual(summary["status"], "crash")
        self.assertEqual([command[1] for command in calls], ["tools/run_candidate.py", "tools/update_results.py"])
        self.assertIn("--status", calls[-1])
        self.assertIn("crash", calls[-1])

    def test_run_loop_records_eval_metrics_after_successful_candidate(self):
        autoresearch_loop = load_tool_module("autoresearch_loop")
        calls = []

        def fake_runner(command, cwd, env):
            calls.append(command)
            tool_name = command[1]
            if tool_name == "tools/run_candidate.py":
                return {
                    "wrapper_returncode": 0,
                    "output": {
                        "status": "completed",
                        "returncode": 0,
                        "exp_dir": "/tmp/stablewm/autoresearch/cube/smoke-loop/exp_0001",
                        "object_checkpoint": "/tmp/stablewm/autoresearch/cube/smoke-loop/exp_0001/lewm_epoch_1_object.ckpt",
                        "object_checkpoint_exists": True,
                    },
                }
            if tool_name == "tools/eval_candidate.py":
                return {
                    "wrapper_returncode": 0,
                    "output": {
                        "status": "completed",
                        "returncode": 0,
                        "success_rate": 2.0,
                        "eval_wall_sec": 487.37,
                    },
                }
            if tool_name == "tools/update_results.py":
                return {
                    "wrapper_returncode": 0,
                    "output": {
                        "row": {
                            "status": "discard",
                            "success_rate": "2.0",
                        }
                    },
                }
            self.fail(f"unexpected tool invocation: {tool_name}")

        summary = autoresearch_loop.run_loop(
            tag="smoke-loop",
            description="successful eval",
            repo_root=Path("/repo/le-wm"),
            stablewm_home=Path("/tmp/stablewm"),
            max_steps=2,
            benchmark="cube",
            command_runner=fake_runner,
        )

        self.assertEqual(summary["status"], "discard")
        self.assertEqual(summary["success_rate"], 2.0)
        self.assertEqual(
            [command[1] for command in calls],
            [
                "tools/run_candidate.py",
                "tools/eval_candidate.py",
                "tools/update_results.py",
            ],
        )
        self.assertIn("--success-rate", calls[-1])
        self.assertIn("2.0", calls[-1])
        self.assertIn("--eval-wall-sec", calls[-1])
        self.assertIn("487.37", calls[-1])

    def test_run_loop_forwards_training_overrides_to_train_stage(self):
        autoresearch_loop = load_tool_module("autoresearch_loop")
        calls = []

        def fake_runner(command, cwd, env):
            calls.append(command)
            tool_name = command[1]
            if tool_name == "tools/run_candidate.py":
                return {
                    "wrapper_returncode": 0,
                    "output": {
                        "status": "completed",
                        "returncode": 0,
                        "exp_dir": "/tmp/stablewm/autoresearch/pusht/warmstart/exp_0001",
                        "latest_object_ckpt": None,
                    },
                }
            if tool_name == "tools/update_results.py":
                return {
                    "wrapper_returncode": 0,
                    "output": {"row": {"status": "crash", "success_rate": ""}},
                }
            self.fail(f"unexpected tool invocation: {tool_name}")

        autoresearch_loop.run_loop(
            tag="warmstart",
            description="warm-start smoke",
            repo_root=Path("/repo/le-wm"),
            stablewm_home=Path("/tmp/stablewm"),
            max_steps=2000,
            benchmark="cube",
            train_overrides=[
                "warm_start.enabled=True",
                "warm_start.checkpoint=pusht/lewm_weights.ckpt",
            ],
            command_runner=fake_runner,
        )

        self.assertIn("--override", calls[0])
        self.assertIn("warm_start.enabled=True", calls[0])
        self.assertIn("warm_start.checkpoint=pusht/lewm_weights.ckpt", calls[0])


if __name__ == "__main__":
    unittest.main()
