from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from revision_experiments.posthoc.data import (
    ChannelStatistics,
    FlightArray,
    corrupt_full_flights,
    inject_events,
)
from revision_experiments.posthoc.ex05 import scenario_specs
from revision_experiments.posthoc.ex07 import _period_indices
from revision_experiments.posthoc.evaluation import normalize_vector_columns
from revision_experiments.posthoc.summarize import summarize_posthoc
from revision_experiments.posthoc.source import _resolve_primary_metrics


def _flight() -> FlightArray:
    values = np.arange(480, dtype=np.float32).reshape(40, 12) / 480.0
    return FlightArray("normal", np.arange(40, dtype=np.float32), values)


def _stats() -> ChannelStatistics:
    return ChannelStatistics(
        median=np.full(12, 0.5, dtype=np.float32),
        std=np.full(12, 0.2, dtype=np.float32),
        robust_scale=np.full(12, 0.1, dtype=np.float32),
    )


class PosthocTests(unittest.TestCase):
    def test_primary_metrics_falls_back_to_main_comparison_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run = Path(temp_dir) / "seed_0"
            analysis = run / "infer_tcngatre_failure" / "score_threshold_analysis"
            analysis.mkdir(parents=True)
            root_primary = run / "primary_metrics.json"
            root_primary.write_text("{}", encoding="utf-8")
            self.assertEqual(_resolve_primary_metrics(run, analysis), root_primary)
            analysis_primary = analysis / "primary_metrics.json"
            analysis_primary.write_text("{}", encoding="utf-8")
            self.assertEqual(_resolve_primary_metrics(run, analysis), analysis_primary)

    def test_robustness_corruptions_are_deterministic_finite_and_non_mutating(self) -> None:
        original = _flight()
        source = {original.flight: original}
        baseline = original.values.copy()
        for kind, level in (
            ("gaussian", 0.05), ("missing", 0.2),
            ("channel_dropout", 3), ("downsample", 4),
        ):
            first, manifest_first = corrupt_full_flights(source, kind, level, _stats(), 3)
            second, manifest_second = corrupt_full_flights(source, kind, level, _stats(), 3)
            np.testing.assert_array_equal(first["normal"].values, second["normal"].values)
            self.assertEqual(first["normal"].values.shape, baseline.shape)
            self.assertTrue(np.isfinite(first["normal"].values).all())
            self.assertEqual(manifest_first, manifest_second)
        np.testing.assert_array_equal(original.values, baseline)

    def test_synthetic_events_are_reproducible_and_do_not_mutate_source(self) -> None:
        original = _flight()
        baseline = original.values.copy()
        first, labels_first, events_first = inject_events(
            original, 2, "bias", 3.0, 4, _stats().robust_scale, 123
        )
        second, labels_second, events_second = inject_events(
            original, 2, "bias", 3.0, 4, _stats().robust_scale, 123
        )
        np.testing.assert_array_equal(first.values, second.values)
        np.testing.assert_array_equal(labels_first, labels_second)
        np.testing.assert_array_equal(original.values, baseline)
        self.assertEqual(events_first, events_second)
        self.assertEqual(len(events_first), 3)
        self.assertTrue(all(0 <= row["start_index"] < row["end_index_exclusive"] <= len(original.time) for row in events_first))

    def test_scenario_matrix_and_period_partition(self) -> None:
        self.assertEqual(len(scenario_specs(10, 5)), 42)
        frame = pd.DataFrame({
            "t_start": np.arange(12),
            "label_any": [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            "graph_row": np.arange(100, 112),
        })
        periods = _period_indices(frame)
        self.assertIsNotNone(periods)
        assert periods is not None
        np.testing.assert_array_equal(periods["during"], [104, 105, 106])
        np.testing.assert_array_equal(periods["before"], [101, 102, 103])
        np.testing.assert_array_equal(periods["after"], [107, 108, 109])

    def test_legacy_numpy_vector_is_canonicalized(self) -> None:
        frame = pd.DataFrame({
            "sensor_score_vec": ["[0.1  0.2\n 0.3]"],
            "value_residual_vec": ["[0.1, 0.2, 0.3]"],
        })
        normalized = normalize_vector_columns(frame)
        self.assertEqual(json.loads(normalized.loc[0, "sensor_score_vec"]), [0.1, 0.2, 0.3])

    def test_summarizer_detects_complete_small_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            run = tmp_path / "ex05" / "simulate" / "seed_0"
            run.mkdir(parents=True)
            (run / "DONE.json").write_text("{}", encoding="utf-8")
            pd.DataFrame([
                {
                    "dataset": "simulate", "seed": 0, "aggregation_method": method,
                    "precision": 1.0, "recall": 1.0, "f1": 1.0, "fpr": 0.0,
                    "auroc": 1.0, "average_precision": 1.0,
                }
                for method in ("mean", "max", "topk_1", "topk_3", "topk_5", "quantile_90", "quantile_95")
            ]).to_csv(run / "real_failure_aggregation.csv", index=False)
            pd.DataFrame([
                {
                    "dataset": "simulate", "seed": 0, "scenario_id": f"scenario_{scenario}",
                    "aggregation_method": method, "precision": 1.0, "recall": 1.0,
                    "f1": 1.0, "average_precision": 1.0, "event_recall": 1.0,
                    "event_miss_rate": 0.0, "mean_detection_delay": 0.0,
                    "channel_hit_at_k": 1.0,
                }
                for scenario in range(42)
                for method in ("mean", "max", "topk_1", "topk_3", "topk_5", "quantile_90", "quantile_95")
            ]).to_csv(run / "synthetic_event_metrics.csv", index=False)
            result = summarize_posthoc(tmp_path, ["ex05"], ["simulate"], [0], require_complete=True)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["completed_units"], 1)
            summary = json.loads((tmp_path / "summary" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["missing_units"], 0)


if __name__ == "__main__":
    unittest.main()
