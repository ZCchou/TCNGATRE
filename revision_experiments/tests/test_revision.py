from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from revision_experiments.analysis.robustness import corrupt_array
from revision_experiments.analysis.statistics import holm_adjust, paired_sign_permutation, rank_biserial
from revision_experiments.baselines.common_data import CommonDataBundle, FlightWindowDataset, window_starts
from revision_experiments.baselines.export_common_data import (
    EXPORT_PROFILE,
    EXPORT_SCHEMA_VERSION,
    validate_common_data,
)
from revision_experiments.core.config import make_config
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


class RevisionTrainingParityTests(unittest.TestCase):
    def test_formal_config_keeps_native_training_and_early_stopping(self):
        cfg = make_config("ex01", "alfa", "full", 0)
        legacy = cfg.to_legacy()
        self.assertEqual(cfg.epochs, 100)
        self.assertEqual(cfg.batch_size, 128)
        self.assertEqual(cfg.lookback, 128)
        self.assertEqual(cfg.d_model, 64)
        self.assertEqual(cfg.tcn_layers, 5)
        self.assertEqual(cfg.tcn_blocks, 4)
        self.assertTrue(legacy.cross_dim_loss_enabled)
        self.assertEqual(legacy.early_stop_patience, 5)
        self.assertAlmostEqual(legacy.early_stop_min_delta, 1e-4)

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
            "graph_profile": {
                "source": "normal training flights only",
                "max_points_per_pair": 200000,
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
                "dataset": "toy", "nodes": ["a", "b"], "normalization_source": "normal training flights only",
                "labels_exported": False, "train": records[:2], "validation": [records[0]], "failure": [records[2]],
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


if __name__ == "__main__":
    unittest.main()
