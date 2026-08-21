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
from revision_experiments.core.integrity import load_approved_changes
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

    def test_default_seeded_profile_keeps_metrics_and_disables_expensive_plots(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = runner.parse_args(
                ["--models", "TCNGATRE", "--datasets", "simulate", "--seeds", "0", "--result-root", temporary]
            )
            jobs, _ = runner.build_job_specs(args)
            train = next(job for job in jobs if job.stage == "train")
            self.assertEqual(train.determinism, "seeded")
            self.assertFalse(train.plots)
            self.assertNotIn("CUBLAS_WORKSPACE_CONFIG", train.env_overrides)
            self.assertEqual(train.env_overrides["UAV_TCNGATRE_PLOT_SCORES"], "0")
            self.assertIn("seeded", train.command)

    def test_tcngatre_gps_uses_default_batch_32(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = runner.parse_args(
                [
                    "--models", "USAD", "TCNGATRE",
                    "--datasets", "gpsdata", "simulate",
                    "--seeds", "0",
                    "--result-root", temporary,
                ]
            )
            jobs, _ = runner.build_job_specs(args)
            train_jobs = [job for job in jobs if job.stage == "train"]
            gps_tcngatre = next(
                job for job in train_jobs
                if job.model == "TCNGATRE" and job.dataset == "gpsdata"
            )
            self.assertEqual(gps_tcngatre.env_overrides["UAV_TCNGATRE_BATCH_SIZE"], "32")
            gps_row = next(
                row for row in runner._run_rows(jobs)
                if row["model"] == "TCNGATRE" and row["dataset"] == "gpsdata"
            )
            self.assertEqual(gps_row["batch_size"], 32)
            self.assertTrue(
                all(
                    "UAV_TCNGATRE_BATCH_SIZE" not in job.env_overrides
                    for job in train_jobs
                    if not (job.model == "TCNGATRE" and job.dataset == "gpsdata")
                )
            )

    def test_only_alfa_tcngatre_uses_larger_formal_stride(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = runner.parse_args(
                [
                    "--models", "USAD", "TCNGATRE",
                    "--datasets", "alfa", "gpsdata", "simulate",
                    "--seeds", "0",
                    "--result-root", temporary,
                ]
            )
            jobs, _ = runner.build_job_specs(args)
            train_jobs = [job for job in jobs if job.stage == "train"]
            alfa_tcngatre = next(
                job for job in train_jobs
                if job.model == "TCNGATRE" and job.dataset == "alfa"
            )
            self.assertEqual(
                alfa_tcngatre.env_overrides["UAV_TCNGATRE_SAMPLE_STRIDE"], "16"
            )
            self.assertTrue(
                all(
                    "UAV_TCNGATRE_SAMPLE_STRIDE" not in job.env_overrides
                    for job in train_jobs
                    if job.model == "TCNGATRE" and job.dataset != "alfa"
                )
            )
            self.assertTrue(
                all(
                    "UAV_TCNGATRE_SAMPLE_STRIDE" not in job.env_overrides
                    for job in train_jobs if job.model != "TCNGATRE"
                )
            )
            run_rows = runner._run_rows(jobs)
            stride_by_run = {
                row["run_id"]: row["sample_stride"]
                for row in run_rows if row["model"] == "TCNGATRE"
            }
            self.assertEqual(stride_by_run["alfa/TCNGATRE/seed_0"], 16)
            self.assertEqual(stride_by_run["gpsdata/TCNGATRE/seed_0"], 4)
            self.assertEqual(stride_by_run["simulate/TCNGATRE/seed_0"], 4)

            alfa_run_root = Path(alfa_tcngatre.run_root)
            alfa_run_root.mkdir(parents=True, exist_ok=True)
            (alfa_run_root / "DONE.json").write_text(
                json.dumps({"status": "complete", "sample_stride": 4}), encoding="utf-8"
            )
            alfa_tcngatre.stage_marker.parent.mkdir(parents=True, exist_ok=True)
            alfa_tcngatre.stage_marker.write_text(
                json.dumps({"status": "ok", "signature": "old_stride_4"}), encoding="utf-8"
            )
            runner._prepare_seeded_outputs(
                jobs, Path(temporary) / "batch", force=False
            )
            self.assertFalse((alfa_run_root / "DONE.json").exists())

    def test_tcngatre_stage_signatures_include_fixed_data_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = runner.parse_args(
                [
                    "--models", "USAD", "TCNGATRE",
                    "--datasets", "alfa",
                    "--seeds", "0",
                    "--result-root", temporary,
                ]
            )
            jobs, _ = runner.build_job_specs(args)
            train_by_model = {job.model: job for job in jobs if job.stage == "train"}
            self.assertIsNone(train_by_model["USAD"].data_protocol_signature)
            self.assertEqual(len(train_by_model["TCNGATRE"].data_protocol_signature), 64)
            self.assertNotEqual(
                train_by_model["TCNGATRE"].signature,
                train_by_model["USAD"].signature,
            )

    def test_strict_profile_and_explicit_plots_are_available(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = runner.parse_args(
                [
                    "--models", "TCNGATRE", "--datasets", "simulate", "--seeds", "0",
                    "--determinism", "strict", "--plots", "--result-root", temporary,
                ]
            )
            jobs, _ = runner.build_job_specs(args)
            train = next(job for job in jobs if job.stage == "train")
            self.assertEqual(train.env_overrides["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
            self.assertEqual(train.env_overrides["UAV_TCNGATRE_PLOT_SCORES"], "1")

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

    def test_strict_mode_enables_deterministic_algorithms(self):
        state = configure_determinism(7, mode="strict")
        self.assertTrue(state["deterministic_algorithms"])
        self.assertTrue(state["cudnn_deterministic"])
        configure_determinism(7, mode="seeded")


class IntegrityApprovalTests(unittest.TestCase):
    def test_multiple_exact_cross_platform_hashes_are_loaded(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "approved.json"
            old_a, old_b, new_a, new_b = ("a" * 64, "b" * 64, "c" * 64, "d" * 64)
            path.write_text(
                json.dumps(
                    {
                        "changes": [
                            {
                                "path": "runner.py",
                                "old_sha256": old_a,
                                "new_sha256": new_a,
                                "accepted_old_sha256": [old_a, old_b],
                                "accepted_new_sha256": [new_a, new_b],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            approval = load_approved_changes(path)["runner.py"]
            self.assertEqual(set(approval["accepted_old_sha256"]), {old_a, old_b})
            self.assertEqual(set(approval["accepted_new_sha256"]), {new_a, new_b})


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
