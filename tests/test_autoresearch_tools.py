import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


import eval_candidate  # noqa: E402
import run_candidate  # noqa: E402
import update_results  # noqa: E402
from autoresearch_common import git_commit_with_dirty_suffix, stablewm_home_from_arg  # noqa: E402


class RunCandidateHelpersTest(unittest.TestCase):
    def test_experiment_dir_uses_autoresearch_pusht_layout(self):
        stablewm_home = Path("/tmp/stablewm")

        exp_dir = run_candidate.experiment_dir(stablewm_home, tag="smoke", exp_id="exp_0001")

        self.assertEqual(
            exp_dir,
            Path("/tmp/stablewm/autoresearch/pusht/smoke/exp_0001"),
        )

    def test_experiment_dir_uses_autoresearch_cube_layout_when_requested(self):
        stablewm_home = Path("/tmp/stablewm")

        exp_dir = run_candidate.experiment_dir(
            stablewm_home,
            tag="smoke",
            exp_id="exp_0001",
            benchmark="cube",
        )

        self.assertEqual(
            exp_dir,
            Path("/tmp/stablewm/autoresearch/cube/smoke/exp_0001"),
        )

    def test_train_command_forces_benchmark_overrides(self):
        repo_root = Path("/repo/le-wm")
        stablewm_home = Path("/tmp/stablewm")
        exp_dir = run_candidate.experiment_dir(stablewm_home, tag="smoke", exp_id="exp_0001")

        cmd = run_candidate.build_train_command(
            repo_root=repo_root,
            exp_dir=exp_dir,
            max_steps=2000,
            overrides=["warm_start.enabled=True"],
        )

        self.assertEqual(cmd[:2], ["python3", str(repo_root / "train.py")])
        self.assertIn("data=pusht", cmd)
        self.assertIn("wandb.enabled=False", cmd)
        self.assertIn("+trainer.max_steps=2000", cmd)
        self.assertIn("subdir=autoresearch/pusht/smoke/exp_0001", cmd)
        self.assertIn("warm_start.enabled=True", cmd)

    def test_train_command_uses_cube_benchmark_configuration(self):
        repo_root = Path("/repo/le-wm")
        stablewm_home = Path("/tmp/stablewm")
        exp_dir = run_candidate.experiment_dir(
            stablewm_home,
            tag="smoke",
            exp_id="exp_0001",
            benchmark="cube",
        )

        cmd = run_candidate.build_train_command(
            repo_root=repo_root,
            exp_dir=exp_dir,
            max_steps=100,
            benchmark="cube",
        )

        self.assertEqual(cmd[:2], ["python3", str(repo_root / "train.py")])
        self.assertIn("data=ogb", cmd)
        self.assertIn("wandb.enabled=False", cmd)
        self.assertIn("+trainer.max_steps=100", cmd)
        self.assertIn("subdir=autoresearch/cube/smoke/exp_0001", cmd)

    def test_build_train_command_rejects_max_steps_below_two(self):
        repo_root = Path("/repo/le-wm")
        stablewm_home = Path("/tmp/stablewm")
        exp_dir = run_candidate.experiment_dir(stablewm_home, tag="smoke", exp_id="exp_0001")

        with self.assertRaisesRegex(ValueError, "max_steps"):
            run_candidate.build_train_command(
                repo_root=repo_root,
                exp_dir=exp_dir,
                max_steps=1,
            )


class EvalCandidateHelpersTest(unittest.TestCase):
    def test_parse_eval_metrics_from_results_text(self):
        results_text = """
==== RESULTS ====
metrics: {'success_rate': 95.0, 'episode_successes': array([ True, False]), 'seeds': None}
evaluation_time: 123.45 seconds
"""

        metrics = eval_candidate.parse_eval_metrics(results_text)

        self.assertEqual(metrics["success_rate"], 95.0)
        self.assertEqual(metrics["evaluation_time"], 123.45)

    def test_policy_name_is_relative_to_stablewm_home(self):
        stablewm_home = Path("/tmp/stablewm")
        exp_dir = stablewm_home / "autoresearch" / "pusht" / "smoke" / "exp_0001"

        policy_name = eval_candidate.policy_name_for_model(
            stablewm_home=stablewm_home,
            exp_dir=exp_dir,
            model_name="lewm_epoch_1",
        )

        self.assertEqual(policy_name, "autoresearch/pusht/smoke/exp_0001/lewm_epoch_1")

    def test_results_file_defaults_to_cube_name_when_requested(self):
        exp_dir = Path("/tmp/stablewm/autoresearch/cube/smoke/exp_0001")

        results_path = eval_candidate.results_file_for_experiment(exp_dir, benchmark="cube")

        self.assertEqual(results_path, exp_dir / "ogb_cube_results.txt")

    def test_eval_command_uses_cube_config_when_requested(self):
        repo_root = Path("/repo/le-wm")

        cmd = eval_candidate.build_eval_command(
            repo_root=repo_root,
            policy_name="autoresearch/cube/smoke/exp_0001/lewm_epoch_1",
            benchmark="cube",
        )

        self.assertEqual(cmd[:2], ["python3", str(repo_root / "eval.py")])
        self.assertIn("--config-name=cube.yaml", cmd)
        self.assertIn("policy=autoresearch/cube/smoke/exp_0001/lewm_epoch_1", cmd)
        self.assertIn("output.filename=ogb_cube_results.txt", cmd)

    def test_eval_command_appends_cli_overrides(self):
        repo_root = Path("/repo/le-wm")

        cmd = eval_candidate.build_eval_command(
            repo_root=repo_root,
            policy_name="autoresearch/cube/smoke/exp_0001/lewm_epoch_1",
            benchmark="cube",
            overrides=["eval.num_eval=10", "eval.goal_offset_steps=10"],
        )

        self.assertEqual(cmd[-2:], ["eval.num_eval=10", "eval.goal_offset_steps=10"])


class UpdateResultsHelpersTest(unittest.TestCase):
    def test_results_table_append_and_best_tracking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "results.tsv"

            update_results.ensure_results_table(results_path)
            update_results.append_result(
                results_path,
                {
                    "exp_id": "exp_0001",
                    "parent_exp_id": "",
                    "commit": "abc123-dirty",
                    "status": "keep",
                    "success_rate": "92.0",
                    "train_wall_sec": "100.0",
                    "eval_wall_sec": "200.0",
                    "peak_vram_gb": "",
                    "policy_path": "autoresearch/pusht/smoke/exp_0001/lewm_epoch_1",
                    "description": "baseline",
                },
            )
            update_results.append_result(
                results_path,
                {
                    "exp_id": "exp_0002",
                    "parent_exp_id": "exp_0001",
                    "commit": "def456-dirty",
                    "status": "discard",
                    "success_rate": "91.0",
                    "train_wall_sec": "101.0",
                    "eval_wall_sec": "201.0",
                    "peak_vram_gb": "",
                    "policy_path": "autoresearch/pusht/smoke/exp_0002/lewm_epoch_1",
                    "description": "worse",
                },
            )

            rows = update_results.load_results(results_path)
            best_row = update_results.best_result(rows)

            self.assertEqual(len(rows), 2)
            self.assertEqual(best_row["exp_id"], "exp_0001")

    def test_classify_result_returns_keep_discard_and_crash(self):
        previous_rows = [
            {
                "exp_id": "exp_0001",
                "parent_exp_id": "",
                "commit": "abc123",
                "status": "keep",
                "success_rate": "92.0",
                "train_wall_sec": "100.0",
                "eval_wall_sec": "200.0",
                "peak_vram_gb": "",
                "policy_path": "autoresearch/pusht/smoke/exp_0001/lewm_epoch_1",
                "description": "baseline",
            }
        ]

        self.assertEqual(update_results.classify_result(previous_rows, 95.0), "keep")
        self.assertEqual(update_results.classify_result(previous_rows, 91.0), "discard")
        self.assertEqual(update_results.classify_result(previous_rows, None), "crash")

    def test_update_best_symlink_points_to_kept_experiment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stablewm_home = Path(tmpdir)
            tag_dir = stablewm_home / "autoresearch" / "pusht" / "smoke"
            exp_dir = tag_dir / "exp_0003"
            exp_dir.mkdir(parents=True)

            best_link = update_results.update_best_link(tag_dir, exp_dir)

            self.assertEqual(best_link, tag_dir / "best")
            self.assertTrue(best_link.is_symlink())
            self.assertEqual(best_link.resolve(), exp_dir.resolve())

    def test_results_path_for_cube_tag_uses_cube_namespace(self):
        stablewm_home = Path("/tmp/stablewm")

        results_path = update_results.results_path_for_tag(
            stablewm_home,
            "smoke",
            benchmark="cube",
        )

        self.assertEqual(
            results_path,
            Path("/tmp/stablewm/autoresearch/cube/smoke/results.tsv"),
        )

    def test_ensure_results_table_migrates_legacy_schema(self):
        legacy_header = "\t".join(
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
        legacy_row = "\t".join(
            [
                "2026-03-30T08:34:15.675132+00:00",
                "exp_0001",
                "keep",
                "2.0",
                "487.3710792064667",
                "legacy smoke",
                "/tmp/stablewm/autoresearch/pusht/smoke/exp_0001",
                "/tmp/stablewm/autoresearch/pusht/smoke/exp_0001/pusht_results.txt",
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            stablewm_home = Path(tmpdir) / "stablewm"
            results_path = stablewm_home / "autoresearch" / "pusht" / "smoke" / "results.tsv"
            results_path.parent.mkdir(parents=True)
            results_path.write_text(f"{legacy_header}\n{legacy_row}\n", encoding="utf-8")

            update_results.ensure_results_table(results_path)
            rows = update_results.load_results(results_path)

            backup_path = results_path.with_suffix(".tsv.legacy.bak")
            self.assertTrue(backup_path.exists())
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["exp_id"], "exp_0001")
            self.assertEqual(rows[0]["status"], "keep")
            self.assertEqual(rows[0]["eval_wall_sec"], "487.3710792064667")
            self.assertEqual(
                rows[0]["policy_path"],
                "autoresearch/pusht/smoke/exp_0001/lewm_epoch_1",
            )

    def test_ensure_results_table_recovers_mixed_legacy_and_current_rows(self):
        legacy_header = "\t".join(
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
        legacy_row = "\t".join(
            [
                "2026-03-30T08:34:15.675132+00:00",
                "exp_0001",
                "keep",
                "2.0",
                "487.3710792064667",
                "legacy smoke",
                "/tmp/stablewm/autoresearch/pusht/smoke/exp_0001",
                "/tmp/stablewm/autoresearch/pusht/smoke/exp_0001/pusht_results.txt",
            ]
        )
        current_row_under_legacy_header = "\t".join(
            [
                "exp_0003",
                "exp_0001",
                "remote-non-git",
                "discard",
                "2.0",
                "24.540774044115096",
                "562.3367116670124",
                "",
                "autoresearch/pusht/smoke/exp_0003/lewm_epoch_1",
                "fresh smoke",
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            stablewm_home = Path(tmpdir) / "stablewm"
            results_path = stablewm_home / "autoresearch" / "pusht" / "smoke" / "results.tsv"
            results_path.parent.mkdir(parents=True)
            results_path.write_text(
                f"{legacy_header}\n{legacy_row}\n{current_row_under_legacy_header}\n",
                encoding="utf-8",
            )

            update_results.ensure_results_table(results_path)
            rows = update_results.load_results(results_path)

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["exp_id"], "exp_0001")
            self.assertEqual(rows[1]["exp_id"], "exp_0003")
            self.assertEqual(rows[1]["commit"], "remote-non-git")
            self.assertEqual(rows[1]["status"], "discard")
            self.assertEqual(rows[1]["policy_path"], "autoresearch/pusht/smoke/exp_0003/lewm_epoch_1")


class AutoresearchCommonHelpersTest(unittest.TestCase):
    def test_stablewm_home_prefers_existing_legacy_compatible_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            preferred = home / "stablewm"
            preferred.mkdir()

            resolved = stablewm_home_from_arg(env={}, home=home)

            self.assertEqual(resolved, preferred.resolve())

    def test_git_commit_returns_non_git_when_repo_has_no_git_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)

            commit = git_commit_with_dirty_suffix(repo_root)

            self.assertEqual(commit, "non-git-worktree")

    @mock.patch.dict("os.environ", {"STABLEWM_HOME": "/env/stablewm"}, clear=True)
    def test_stablewm_home_uses_env_before_default_locations(self):
        resolved = stablewm_home_from_arg()
        self.assertEqual(resolved, Path("/env/stablewm"))


if __name__ == "__main__":
    unittest.main()
