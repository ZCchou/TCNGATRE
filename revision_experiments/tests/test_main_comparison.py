from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_all_models_all_datasets as runner
from revision_experiments.analysis.main_comparison import summarize_main_comparison
from revision_experiments.main_comparison.deterministic_entrypoint import configure_determinism


class MainComparisonPlanningTests(unittest.TestCase):
    def test_legacy_job_shape_and_commands_are_preserved(self):
        args = runner.parse_args(
            [
                "--models", "USAD", "TCNGATRE",
                "--datasets", "simulate",
                "--stages", "train", "infer", "eval",
                "--dry-run",
            ]
        )
        jobs, _ = runner.build_job_specs(args)
        self.assertEqual(len(jobs), 5)
        self.assertTrue(all(job.seed is None for job in jobs))
        self.assertTrue(all(job.run_root is None for job in jobs))
        self.assertEqual(jobs[0].command[-2:], ["--dataset", "simulate"])
        self.assertEqual(Path(jobs[0].command[1]).name, "train_usad.py")

    def test_full_five_seed_matrix_has_unique_isolated_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = runner.parse_args(
                ["--seeds", "0", "1", "2", "3", "4", "--result-root", temporary]
            )
            jobs, _ = runner.build_job_specs(args)
            run_rows = runner._run_rows(jobs)
            self.assertEqual(len(jobs), 255)
            self.assertEqual(len(run_rows), 90)
            self.assertEqual(len({row["run_id"] for row in run_rows}), 90)
            self.assertEqual(len({row["run_root"] for row in run_rows}), 90)
            expected_root = Path(temporary).resolve()
            self.assertTrue(
                all(Path(row["run_root"]).resolve().is_relative_to(expected_root) for row in run_rows)
            )

    def test_model_seed_environment_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = runner.parse_args(
                ["--datasets", "simulate", "--seeds", "3", "--result-root", temporary]
            )
            jobs, _ = runner.build_job_specs(args)
            by_model = {job.model: job for job in jobs if job.stage == "train"}
            expected = {
                "USAD": "UAV_USAD_SEED",
                "Recurrent_AE": "UAV_RAE_SEED",
                "TranAD": "UAV_TRANAD_SEED",
                "OmniAnomaly": "UAV_OA_SEED",
                "BeatGAN": "UAV_BEATGAN_SEED",
                "TCNGATRE": "UAV_TCNGATRE_SPLIT_SEED",
            }
            for model, env_name in expected.items():
                self.assertEqual(by_model[model].env_overrides[env_name], "3")
                self.assertEqual(by_model[model].env_overrides["PYTHONHASHSEED"], "3")
            self.assertIn("UAV_TCNGATRE_GRAPH_DIR", by_model["TCNGATRE"].env_overrides)

    def test_smoke_mode_sets_one_epoch_and_disables_plots(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = runner.parse_args(
                [
                    "--datasets", "simulate", "--seeds", "0", "--smoke",
                    "--result-root", temporary,
                ]
            )
            jobs, _ = runner.build_job_specs(args)
            train_jobs = [job for job in jobs if job.stage == "train"]
            self.assertEqual(len(train_jobs), 6)
            for job in train_jobs:
                names = runner.MODEL_ENV[job.model]
                self.assertEqual(job.env_overrides[names["epochs"]], "1")
                self.assertEqual(job.env_overrides[names["plot"]], "0")
                self.assertEqual(job.env_overrides["PYTHONUNBUFFERED"], "1")
            self.assertEqual(
                next(job for job in train_jobs if job.model == "TCNGATRE")
                .env_overrides["UAV_TCNGATRE_SAMPLE_STRIDE"],
                "64",
            )


class DeterminismTests(unittest.TestCase):
    def test_reapplying_same_seed_repeats_random_streams(self):
        configure_determinism(7)
        first = (random.random(), np.random.rand(), torch.rand(3))
        configure_determinism(7)
        second = (random.random(), np.random.rand(), torch.rand(3))
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        torch.testing.assert_close(first[2], second[2])


class MainComparisonSummaryTests(unittest.TestCase):
    @staticmethod
    def _create_run(result_root: Path, seed: int, f1_offset: float) -> dict:
        run_dir = result_root / "simulate" / "USAD" / f"seed_{seed}"
        analysis = (
            run_dir
            / runner.INFER_OUTPUT_NAMES["USAD"]
            / "score_threshold_analysis"
        )
        analysis.mkdir(parents=True)
        value = 0.70 + f1_offset
        pd.DataFrame(
            [
                {
                    "threshold_method": "spot",
                    "label_col": "label_any",
                    "precision": value,
                    "recall": value,
                    "f1": value,
                    "fpr": 1.0 - value,
                    "auroc": value,
                    "average_precision": value,
                    "num_samples": 100,
                    "positives": 20,
                    "negatives": 80,
                }
            ]
        ).to_csv(analysis / "summary_metrics.csv", index=False)
        (run_dir / "DONE.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
        return {
            "run_id": f"simulate/USAD/seed_{seed}",
            "dataset": "simulate",
            "model": "USAD",
            "seed": seed,
            "smoke": False,
            "run_root": str(run_dir),
            "stages": "train,infer",
        }

    def test_micro_and_seed_summaries_are_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            result_root = Path(temporary)
            runs = [self._create_run(result_root, 0, 0.0), self._create_run(result_root, 1, 0.02)]
            payload = summarize_main_comparison(result_root, runs)
            self.assertEqual(payload["complete_runs"], 2)
            primary = pd.read_csv(result_root / "summary" / "primary_metrics_all_runs.csv")
            seed_summary = pd.read_csv(result_root / "summary" / "primary_metrics_seed_summary.csv")
            self.assertAlmostEqual(float(primary.loc[0, "f1"]), 0.7)
            self.assertEqual(primary.loc[0, "aggregation"], "micro_over_all_windows")
            self.assertEqual(int(seed_summary.loc[0, "seed_count"]), 2)
            self.assertTrue((result_root / "run_status.csv").is_file())


if __name__ == "__main__":
    unittest.main()
