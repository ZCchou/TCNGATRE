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
from revision_experiments.baselines.mstgcnet_model import MSTGCNetApprox
from revision_experiments.baselines.mstgcnet_native_evaluation import (
    atssd,
    confusion_metrics,
    point_adjust,
)
from revision_experiments.baselines.reproduction_utils import accumulation_groups
from revision_experiments.baselines.tsae_uav_model import TSAEUAV
from revision_experiments.core.config import (
    load_protocol,
    make_config,
    resolve_experiment_selection,
)
from revision_experiments.core.engine import data_protocol_payload
from revision_experiments.core.engine import set_model_seed
from revision_experiments.models.variants import build_revision_model
from revision_experiments.posthoc.parity import compare_scores
from revision_experiments.scoring.aggregators import aggregate_channels
from revision_experiments.scoring.local_anomaly import inject_local_anomaly


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def test_mean_parity(self):
        np.testing.assert_allclose(aggregate_channels(self.values, "mean"), self.values.mean(axis=1))

    def test_score_parity_accepts_float32_rounding_at_large_scale(self):
        result = compare_scores([4_444_446.0], [4_444_445.617853752], atol=5e-6)
        self.assertTrue(result.passed)
        self.assertGreater(result.max_abs_error, 0.3)
        self.assertLess(result.max_rel_error, 1e-6)

    def test_score_parity_rejects_material_relative_error(self):
        self.assertFalse(compare_scores([100.0], [99.0], atol=5e-6).passed)

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
        self.assertEqual(make_config("ex01", "alfa", "full", 0).batch_size, 128)
        self.assertEqual(make_config("ex01", "gpsdata", "full", 0).batch_size, 32)
        self.assertEqual(make_config("ex01", "simulate", "full", 0).batch_size, 128)
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

    def test_static_only_removes_dynamic_attention_parameters(self):
        static_cfg = make_config("ex01", "simulate", "static_only", 0, smoke=True)
        full_cfg = make_config("ex01", "simulate", "full", 0, smoke=True)
        self.assertEqual(
            static_cfg.variant_revision,
            "lightweight_terminal_static_smoothing_v3",
        )
        self.assertEqual(full_cfg.variant_revision, "")

        device = torch.device("cpu")
        static_model = build_revision_model(
            static_cfg, static_cfg.to_legacy(), num_nodes=4, device=device
        )
        full_model = build_revision_model(
            full_cfg, full_cfg.to_legacy(), num_nodes=4, device=device
        )
        static_names = {name for name, _ in static_model.named_parameters()}
        self.assertFalse(
            any("graph_corrections" in name and ".dyn." in name for name in static_names)
        )
        self.assertLess(
            sum(parameter.numel() for parameter in static_model.parameters()),
            sum(parameter.numel() for parameter in full_model.parameters()),
        )
        self.assertEqual(
            static_model._correction_positions,
            [static_model.num_blocks - 1],
        )
        self.assertEqual(len(static_model.graph_corrections), 1)
        self.assertEqual(
            sum(parameter.numel() for parameter in static_model.graph_corrections.parameters()),
            0,
        )

        x = torch.randn(2, 16, 4, 1)
        a = torch.rand(4, 4)
        a.fill_diagonal_(0.0)
        m = torch.ones(4, 4)
        prediction, aux = static_model(x, a, m, short_patch=8)
        self.assertEqual(tuple(prediction.shape), (2, 4, 4, 1))
        self.assertEqual(aux["A_dyn"].numel(), 1)
        torch.testing.assert_close(aux["A_fuse"], aux["A_static"])
        prediction.square().mean().backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in static_model.parameters())
        )


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


class ReviewerBaselineReproductionTests(unittest.TestCase):
    def test_ex03_filter_expands_to_two_real_adapter_runs(self):
        from revision_experiments.run_revision import _task_rows

        rows = _task_rows(
            ["ex03"], ["alfa"], [0], True, variants=["mstgcnet", "tsae_uav"]
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["variant"] for row in rows}, {"mstgcnet", "tsae_uav"})

    def test_ex03_gps_simulate_expands_to_twenty_formal_runs(self):
        from revision_experiments.run_revision import _task_rows

        rows = _task_rows(
            ["ex03"], ["gpsdata", "simulate"], [0, 1, 2, 3, 4], False,
            variants=["mstgcnet", "tsae_uav"],
        )
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({row["run_dir"] for row in rows}), 20)

    def test_accumulation_groups_preserve_short_final_group_size(self):
        loader = [torch.zeros(32, 2), torch.zeros(32, 2), torch.zeros(5, 2)]
        groups = list(accumulation_groups(loader, 2))
        self.assertEqual([count for _, count in groups], [64, 5])

    def test_mstgcnet_native_metrics_use_micro_confusion_counts(self):
        labels = np.array([0, 1, 1, 0, 1, 1], dtype=np.int8)
        scores = np.array([0.0, 1.0, 0.5, 0.0, 2.0, 0.2])
        prediction = point_adjust(labels, atssd(scores, window_size=2, alpha=0.5))
        metrics = confusion_metrics(labels, prediction, scores)
        self.assertAlmostEqual(metrics["precision"], metrics["tp"] / max(metrics["tp"] + metrics["fp"], 1))
        self.assertAlmostEqual(metrics["recall"], metrics["tp"] / max(metrics["tp"] + metrics["fn"], 1))

    def test_tsae_uav_forward_backward_is_finite(self):
        model = TSAEUAV(channels=12, d_model=64, top_k=3, layers=2)
        values = torch.randn(2, 16, 12)
        reconstructed = model(values)
        self.assertEqual(tuple(reconstructed.shape), tuple(values.shape))
        self.assertTrue(torch.isfinite(reconstructed).all())
        reconstructed.square().mean().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_mstgcnet_approx_forward_backward_and_sparse_graph(self):
        model = MSTGCNetApprox(nodes=12, d_model=16, heads=2)
        values = torch.randn(2, 96, 12)
        reconstructed, balance, adjacencies = model(values)
        self.assertEqual(tuple(reconstructed.shape), tuple(values.shape))
        self.assertEqual(len(adjacencies), 3)
        self.assertTrue(torch.isfinite(reconstructed).all())
        self.assertTrue(torch.isfinite(balance))
        for adjacency in adjacencies:
            self.assertEqual(tuple(adjacency.shape), (12, 12))
            torch.testing.assert_close(adjacency.sum(dim=-1), torch.ones(12))
        for layer in model.layers:
            for expert in layer.experts:
                neighbor_index, neighbor_weight = expert._causal_knn()
                self.assertEqual(neighbor_index.shape[1], 5)
                self.assertTrue(torch.isfinite(neighbor_weight).all())
                self.assertTrue(
                    (
                        expert.patch_index[neighbor_index]
                        <= expert.patch_index[:, None]
                    ).all()
                )
        (reconstructed.square().mean() + 0.01 * balance).backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    @staticmethod
    def _write_reviewer_metric_files(run_dir: Path, value: float) -> None:
        analysis = run_dir / "infer_tcngatre_failure" / "score_threshold_analysis"
        analysis.mkdir(parents=True)
        pd.DataFrame([{
            "threshold_method": "spot", "label_col": "label_any",
            "precision": value, "recall": value, "f1": value,
            "fpr": 1.0 - value, "auroc": value,
            "average_precision": value,
        }]).to_csv(analysis / "summary_metrics.csv", index=False)
        pd.DataFrame([
            {
                "flight": f"flight_{index}", "threshold_method": "spot",
                "label_col": "label_any", "f1": value - 0.01 * index,
                "average_precision": value - 0.005 * index,
            }
            for index in range(2)
        ]).to_csv(
            analysis / "per_flight_total_score_threshold_methods.csv", index=False
        )

    def test_reviewer_summary_uses_formal_ex03_and_main_comparison_only(self):
        from revision_experiments.summarize_main_comparison import (
            _tcngatre_data_protocol_signature,
        )
        from revision_experiments.summarize_reviewer_baselines import (
            _canonical_split,
            summarize,
        )

        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            results = temporary_root / "protocol_v1"
            reference = temporary_root / "main_reference"
            seeds = [0, 1]
            for baseline, value, classification in (
                ("mstgcnet", 0.6, "released_scaffold_engineering_reimplementation"),
                ("tsae_uav", 0.7, "paper_based_protocol_compatible_reimplementation"),
            ):
                for seed in seeds:
                    cfg = make_config("ex03", "alfa", baseline, seed)
                    protocol = data_protocol_payload(cfg.to_legacy())
                    run = results / "ex03" / "alfa" / baseline / f"seed_{seed}"
                    self._write_reviewer_metric_files(run, value + 0.01 * seed)
                    (run / "DONE.json").write_text(json.dumps({
                        "status": "complete", "config_hash": cfg.config_hash,
                        "data_protocol_hash": protocol["data_protocol_hash"],
                        "reproduction_classification": classification,
                    }), encoding="utf-8")
            reference_protocol = data_protocol_payload(
                make_config("ex03", "alfa", "mstgcnet", 0).to_legacy()
            )
            reference_split = _canonical_split(reference_protocol)
            for seed in seeds:
                run = reference / f"seed_{seed}"
                self._write_reviewer_metric_files(run, 0.8 + 0.01 * seed)
                (run / "split_flights.json").write_text(
                    json.dumps({
                        "train_flights": reference_split["train"],
                        "validation_flights": reference_split["validation"],
                        "failure_flights_scored_only": reference_split["failure"],
                    }), encoding="utf-8",
                )
                (run / "DONE.json").write_text(json.dumps({
                    "status": "complete", "sample_stride": 16, "batch_size": 128,
                    "data_protocol_signature": _tcngatre_data_protocol_signature("alfa"),
                }), encoding="utf-8")
            output = temporary_root / "summary"
            report = summarize(
                datasets=["alfa"], seeds=seeds, tcngatre_root=reference,
                output_root=output, n_resamples=20, results_root=results,
            )
            self.assertEqual(report["status"], "complete")
            self.assertIn("main_comparison", report["reference"])
            significance = pd.read_csv(output / "paired_significance.csv")
            self.assertEqual(len(significance), 4)
            self.assertTrue(
                (significance["mean_difference_baseline_minus_tcngatre"] < 0).all()
            )


if __name__ == "__main__":
    unittest.main()
