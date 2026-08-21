from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from revision_experiments.analysis.robustness import corrupt_array
from revision_experiments.analysis.summarize import summarize_experiment_matrix
from revision_experiments.analysis.statistics import holm_adjust, paired_sign_permutation, rank_biserial
from revision_experiments.baselines.common_data import CommonDataBundle, FlightWindowDataset, window_starts
from revision_experiments.baselines.export_common_data import (
    EXPORT_PROFILE,
    EXPORT_SCHEMA_VERSION,
    split_fingerprint,
    validate_common_data,
)
from revision_experiments.core.config import (
    load_protocol,
    make_config,
    resolve_experiment_selection,
)
from revision_experiments.core.engine import data_protocol_payload
from revision_experiments.core.engine import set_model_seed
from revision_experiments.scoring.aggregators import aggregate_channels
from revision_experiments.scoring.local_anomaly import inject_local_anomaly


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def test_mean_parity(self):
        np.testing.assert_allclose(aggregate_channels(self.values, "mean"), self.values.mean(axis=1))

    def test_topk_and_quantile(self):
        np.testing.assert_allclose(aggregate_channels(self.values, "topk_1"), [3.0, 6.0])
        np.testing.assert_allclose(aggregate_channels(self.values, "topk_5"), self.values.mean(axis=1))
        self.assertEqual(aggregate_channels(self.values, "quantile_95").shape, (2,))


class InjectionAndRobustnessTests(unittest.TestCase):
    def test_local_injections_do_not_modify_source(self):
        source = np.ones((20, 4), dtype=float)
        for kind in ("bias", "drift", "freeze", "noise"):
            changed, labels = inject_local_anomaly(source, [1], 5, 4, kind, np.ones(4), 3.0, 0)
            np.testing.assert_allclose(source, 1.0)
            self.assertEqual(int(labels.sum()), 4)
            self.assertEqual(changed.shape, source.shape)

    def test_corruptions_are_finite_and_shape_preserving(self):
        source = np.ones((2, 12, 5), dtype=np.float32)
        for kind, level in (("gaussian", 0.1), ("missing", 0.3), ("channel_dropout", 3), ("downsample", 4)):
            changed, _ = corrupt_array(source, kind, level, 1)
            self.assertEqual(changed.shape, source.shape)
            self.assertTrue(np.isfinite(changed).all())


class StatisticsTests(unittest.TestCase):
    def test_permutation_and_effect(self):
        result = paired_sign_permutation(np.array([1.0, 2.0, 3.0]))
        self.assertTrue(0.0 <= result["p_value"] <= 1.0)
        self.assertAlmostEqual(rank_biserial(np.array([1.0, 2.0, 3.0])), 1.0)

    def test_holm_is_monotone_in_sorted_order(self):
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        self.assertTrue(all(0.0 <= value <= 1.0 for value in adjusted))


class CoreAblationMatrixTests(unittest.TestCase):
    EXPECTED_SELECTION = {
        "ex01": ["full", "tcn_only", "late_graph", "static_only", "dynamic_only"],
        "ex02": ["fusion_learned_scalar", "prior_random_fixed"],
    }

    def test_core_preset_expands_to_105_unique_runs(self):
        from revision_experiments.run_revision import _seeds, _task_rows

        protocol = load_protocol()
        selection = resolve_experiment_selection(protocol, preset="core_ablation")
        self.assertEqual(selection, self.EXPECTED_SELECTION)
        rows = _task_rows(
            [], ["alfa", "gpsdata", "simulate"], [0, 1, 2, 3, 4], False,
            preset="core_ablation",
        )
        self.assertEqual(len(rows), 105)
        self.assertEqual(len({row["run_dir"] for row in rows}), 105)
        with self.assertRaisesRegex(ValueError, "Duplicate model seeds"):
            _seeds("0,0")

    def test_variant_filter_rejects_unmatched_names(self):
        protocol = load_protocol()
        selected = resolve_experiment_selection(
            protocol, preset="core_ablation", variants=["full", "prior_random_fixed"]
        )
        self.assertEqual(selected, {"ex01": ["full"], "ex02": ["prior_random_fixed"]})
        with self.assertRaisesRegex(ValueError, "not in the selected experiments"):
            resolve_experiment_selection(
                protocol, preset="core_ablation", variants=["single_hop"]
            )

    @staticmethod
    def _write_fake_run(root: Path, experiment: str, variant: str, seed: int) -> None:
        cfg = make_config(experiment, "simulate", variant, seed)
        protocol_hash = data_protocol_payload(cfg.to_legacy())["data_protocol_hash"]
        run_dir = root / "protocol_v1" / experiment / "simulate" / variant / f"seed_{seed}"
        analysis = run_dir / "infer_tcngatre_failure" / "score_threshold_analysis"
        analysis.mkdir(parents=True)
        value = 0.8 if variant == "full" else 0.7
        primary = {
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
            "tp": 16,
            "fp": 4,
            "tn": 76,
            "fn": 4,
        }
        (analysis / "primary_metrics.json").write_text(json.dumps(primary), encoding="utf-8")
        pd.DataFrame(
            [
                {"flight": "simulate_0", "threshold_method": "spot", "label_col": "label_any", "f1": value},
                {"flight": "simulate_1", "threshold_method": "spot", "label_col": "label_any", "f1": value - 0.05},
            ]
        ).to_csv(analysis / "per_flight_total_score_threshold_methods.csv", index=False)
        (run_dir / "DONE.json").write_text(
            json.dumps({
                "status": "complete",
                "config_hash": cfg.config_hash,
                "data_protocol_hash": protocol_hash,
            }),
            encoding="utf-8",
        )

    def test_summary_is_complete_and_statistically_usable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection = {"ex01": ["full", "tcn_only"]}
            for variant in selection["ex01"]:
                for seed in (0, 1):
                    self._write_fake_run(root, "ex01", variant, seed)
            report = summarize_experiment_matrix(
                "protocol_v1",
                selection,
                ["simulate"],
                [0, 1],
                results_root=root,
                preset_name="test_core",
                n_resamples=100,
            )
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["complete_runs"], 4)
            output = root / "protocol_v1" / "summary" / "test_core"
            summary = pd.read_csv(output / "primary_metrics_seed_summary.csv")
            significance = pd.read_csv(output / "paired_significance.csv")
            self.assertTrue((summary["seed_count"] == 2).all())
            self.assertEqual(significance.loc[0, "status"], "complete")
            self.assertTrue(np.isfinite(float(significance.loc[0, "p_value_holm"])))

            stale_done = (
                root / "protocol_v1" / "ex01" / "simulate" / "tcn_only"
                / "seed_1" / "DONE.json"
            )
            payload = json.loads(stale_done.read_text(encoding="utf-8"))
            payload["data_protocol_hash"] = "stale_9_1_16_protocol"
            stale_done.write_text(json.dumps(payload), encoding="utf-8")
            incomplete = summarize_experiment_matrix(
                "protocol_v1",
                selection,
                ["simulate"],
                [0, 1],
                results_root=root,
                preset_name="test_stale",
                n_resamples=20,
            )
            self.assertEqual(incomplete["status"], "incomplete")
            missing = pd.read_csv(
                root / "protocol_v1" / "summary" / "test_stale"
                / "missing_experiment_cells.csv"
            )
            self.assertEqual(missing.loc[0, "reason"], "data_protocol_hash_mismatch")


class RevisionTrainingParityTests(unittest.TestCase):
    def test_formal_config_keeps_native_training_and_early_stopping(self):
        cfg = make_config("ex01", "alfa", "full", 0)
        legacy = cfg.to_legacy()
        self.assertEqual(cfg.epochs, 100)
        self.assertEqual(cfg.batch_size, 128)
        self.assertEqual(cfg.lookback, 128)
        self.assertEqual(cfg.stride, 16)
        self.assertEqual(cfg.d_model, 64)
        self.assertEqual(cfg.tcn_layers, 5)
        self.assertEqual(cfg.tcn_blocks, 4)
        self.assertTrue(legacy.cross_dim_loss_enabled)
        self.assertEqual(legacy.early_stop_patience, 5)
        self.assertAlmostEqual(legacy.early_stop_min_delta, 1e-4)

    def test_only_alfa_formal_revision_runs_use_larger_stride(self):
        self.assertEqual(make_config("ex01", "alfa", "full", 0).stride, 16)
        self.assertEqual(make_config("ex01", "gpsdata", "full", 0).stride, 4)
        self.assertEqual(make_config("ex01", "simulate", "full", 0).stride, 4)
        self.assertEqual(make_config("ex01", "alfa", "full", 0, smoke=True).stride, 64)

    def test_seeded_mode_does_not_force_slow_deterministic_kernels(self):
        previous_algorithms = torch.are_deterministic_algorithms_enabled()
        previous_deterministic = torch.backends.cudnn.deterministic
        previous_benchmark = torch.backends.cudnn.benchmark
        try:
            state = set_model_seed(3)
            self.assertFalse(state["deterministic_algorithms"])
            self.assertFalse(state["cudnn_deterministic"])
            self.assertFalse(state["cudnn_benchmark"])
        finally:
            torch.use_deterministic_algorithms(previous_algorithms)
            torch.backends.cudnn.deterministic = previous_deterministic
            torch.backends.cudnn.benchmark = previous_benchmark


class BaselineCommonDataTests(unittest.TestCase):
    @staticmethod
    def _write_valid_export(root: Path) -> Path:
        dataset_root = root / "toy"
        dataset_root.mkdir()
        records = []
        for split, value in (("train", 0.0), ("validation", 1.0), ("failure", 2.0)):
            split_root = dataset_root / split
            split_root.mkdir()
            path = split_root / f"{split}.npz"
            np.savez_compressed(
                path,
                time=np.arange(8, dtype=np.float32),
                values=np.full((8, 2), value, np.float32),
            )
            records.append({"flight": split, "path": str(path), "rows": 8, "channels": 2})
        manifest = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "export_profile": EXPORT_PROFILE,
            "dataset": "toy",
            "data_split_seed": 64,
            "split_fingerprint": split_fingerprint({
                "train": ["train"], "validation": ["validation"], "failure": ["failure"]
            }),
            "split_policy": {"source": "test", "prefail_normal_policy": "train_only"},
            "graph_profile": {
                "source": "normal training flights only",
                "max_points_per_pair": 200000,
                "include_flights": ["train"],
                "num_input_files": 1,
            },
            "nodes": ["a", "b"],
            "normalization_source": "normal training flights only",
            "labels_exported": False,
            "train": [records[0]],
            "validation": [records[1]],
            "failure": [records[2]],
        }
        (dataset_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return dataset_root

    def test_windows_never_cross_flights_and_scaler_is_train_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root = root / "toy"
            dataset_root.mkdir()
            records = []
            for flight, value in (("f0", 0.0), ("f1", 10.0), ("failure", 1000.0)):
                path = dataset_root / f"{flight}.npz"
                np.savez_compressed(path, time=np.arange(8), values=np.full((8, 2), value, np.float32))
                records.append({"flight": flight, "path": str(path), "rows": 8, "channels": 2})
            manifest = {
                "schema_version": EXPORT_SCHEMA_VERSION,
                "export_profile": EXPORT_PROFILE,
                "dataset": "toy",
                "data_split_seed": 64,
                "split_fingerprint": split_fingerprint({
                    "train": ["f0", "f1"], "validation": ["validation"], "failure": ["failure"]
                }),
                "split_policy": {"source": "test", "prefail_normal_policy": "train_only"},
                "graph_profile": {
                    "source": "normal training flights only",
                    "max_points_per_pair": 200000,
                    "include_flights": ["f0", "f1"],
                    "num_input_files": 2,
                },
                "nodes": ["a", "b"],
                "normalization_source": "normal training flights only",
                "labels_exported": False,
                "train": records[:2],
                "validation": [{**records[0], "flight": "validation"}],
                "failure": [records[2]],
            }
            (dataset_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            bundle = CommonDataBundle("toy", root=root)
            standardizer = bundle.fit_standardizer()
            np.testing.assert_allclose(standardizer.mean, [5.0, 5.0])
            windows = FlightWindowDataset(bundle, "train", standardizer, window=4, stride=2)
            self.assertEqual(len(windows), 6)
            for item in windows:
                values = item.numpy()
                self.assertTrue(np.all(values == values[0, 0]))

    def test_window_start_cap_is_deterministic(self):
        first = window_starts(100, 10, 2, max_windows=7)
        second = window_starts(100, 10, 2, max_windows=7)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 7)

    def test_export_validation_checks_arrays_and_label_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root = self._write_valid_export(root)
            payload = validate_common_data("toy", root, verify_arrays=True)
            self.assertFalse(payload["labels_exported"])

            manifest_path = dataset_root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["labels_exported"] = True
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "labels_exported"):
                validate_common_data("toy", root, verify_arrays=True)

    def test_tcngatre_fixed_splits_include_prefail_normal_training_flights(self):
        from data.alfa_shared import build_flight_path_maps

        repo_root = Path(__file__).resolve().parents[2]
        expected = {
            "alfa": (29, 1, 16, 20),
            "gpsdata": (1, 1, 2, 0),
            "simulate": (8, 2, 2, 0),
        }
        for dataset, counts in expected.items():
            train, validation, failure, metadata = build_flight_path_maps(
                repo_root / "dataset" / dataset
            )
            prefail_count = len(metadata["prefail_normal"])
            self.assertEqual((len(train), len(validation), len(failure), prefail_count), counts)
            self.assertFalse(metadata["expected_count_mismatches"])

    def test_graph_cache_requires_exact_training_flight_provenance(self):
        from tcngatre_runtime import graph_cache_matches

        with tempfile.TemporaryDirectory() as temporary:
            graph_root = Path(temporary)
            (graph_root / "keep_columns.json").write_text('["a", "b"]', encoding="utf-8")
            (graph_root / "adjacency_dense.csv").write_text("node\n", encoding="utf-8")
            (graph_root / "edges_mic.csv").write_text(
                "src,dst,mic,overlap,kept\na,b,0.5,100,1\n", encoding="utf-8"
            )
            metadata_path = graph_root / "build_metadata.json"
            metadata_path.write_text(json.dumps({
                "include_flights": ["train_a", "train_b"],
                "num_input_files": 2,
                "num_pair_results": 1,
            }), encoding="utf-8")
            self.assertTrue(graph_cache_matches(graph_root, ["train_b", "train_a"]))
            self.assertFalse(graph_cache_matches(graph_root, ["train_a"]))

    def test_mic_progress_does_not_swallow_worker_failures(self):
        from util.build_set_a_graph import progress_iter

        def broken_items():
            yield 1
            raise RuntimeError("worker failed")

        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            list(progress_iter(broken_items(), total=2, desc="test", progress_every=1))


if __name__ == "__main__":
    unittest.main()
