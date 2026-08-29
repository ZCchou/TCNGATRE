from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from revision_experiments.summarize_mstgcnet_five_seed import summarize


class MSTGCNetFiveSeedSummaryTests(unittest.TestCase):
    def test_complete_five_seed_summary_and_confusion_recalculation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mstgcnet"
            for seed in range(5):
                run = root / f"seed_{seed}"
                native = run / "native_evaluation"
                native.mkdir(parents=True)
                tp, fp, tn, fn = 10 + seed, 2, 20, 3
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = 2 * precision * recall / (precision + recall)
                metrics = {
                    "num_samples": tp + fp + tn + fn,
                    "positives": tp + fn,
                    "negatives": tn + fp,
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "accuracy": (tp + tn) / (tp + fp + tn + fn),
                    "fpr": fp / (fp + tn),
                    "auroc": 0.7 + seed * 0.01,
                    "average_precision": 0.6 + seed * 0.01,
                }
                (native / "primary_metrics.json").write_text(
                    json.dumps(metrics), encoding="utf-8"
                )
                (run / "DONE.json").write_text(
                    json.dumps({"status": "complete"}), encoding="utf-8"
                )
                (run / "config_resolved.json").write_text(
                    json.dumps({
                        "baseline_parameters": {
                            "parameter_profile": "paper_faithful",
                            "window": 96,
                            "train_stride": 16,
                            "validation_stride": 16,
                            "score_stride": 16,
                            "epochs": 50,
                            "early_stop_patience": 10,
                            "effective_batch_size": 128,
                            "physical_batch_size": 8,
                            "gradient_accumulation_steps": 16,
                        }
                    }),
                    encoding="utf-8",
                )
                (run / "best.pt").write_bytes(f"seed-{seed}".encode())
                pd.DataFrame([{"epoch": value} for value in range(1, 6)]).to_csv(
                    run / "history.csv", index=False
                )

            output = Path(temporary) / "summary"
            result = summarize(
                run_root=root,
                output_dir=output,
                dataset="alfa",
                seeds=[0, 1, 2, 3, 4],
                require_complete=True,
            )
            self.assertEqual(result["status"], "complete")
            all_runs = pd.read_csv(output / "mstgcnet_five_seed_all_runs.csv")
            summary = pd.read_csv(output / "mstgcnet_five_seed_summary.csv")
            self.assertEqual(len(all_runs), 5)
            self.assertEqual(int(summary.loc[0, "seed_count"]), 5)
            self.assertAlmostEqual(
                float(summary.loc[0, "f1_mean"]), float(all_runs["f1"].mean())
            )
            self.assertAlmostEqual(
                float(summary.loc[0, "f1_sample_sd"]), float(all_runs["f1"].std(ddof=1))
            )


if __name__ == "__main__":
    unittest.main()
