from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revision_experiments.core.config import make_config  # noqa: E402


DEFAULT_RESULTS = REPO_ROOT / "revision_results" / "protocol_v1"
MODELS = ("gcad", "m2ad")
DATASETS = ("alfa", "gpsdata", "simulate")
METRICS = ("precision", "recall", "f1", "accuracy", "fpr", "auroc", "average_precision")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_metrics(row: dict, tolerance: float = 1e-12) -> None:
    tp, fp, tn, fn = (int(row[key]) for key in ("tp", "fp", "tn", "fn"))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    expected = {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, np.finfo(float).eps),
        "accuracy": (tp + tn) / max(tp + fp + tn + fn, 1),
        "fpr": fp / max(fp + tn, 1),
    }
    for key, value in expected.items():
        actual = float(row[key])
        if not math.isfinite(actual) or abs(actual - value) > tolerance:
            raise ValueError(f"{key} mismatch: stored={actual}, recomputed={value}")
    for key in ("auroc", "average_precision"):
        actual = float(row[key])
        if not math.isfinite(actual) or not 0 <= actual <= 1:
            raise ValueError(f"invalid {key}: {actual}")


def _collect(results_root: Path, datasets: list[str], seeds: list[int]):
    rows, flights, statuses, missing = [], [], [], []
    for dataset in datasets:
        for model in MODELS:
            for seed in seeds:
                run_dir = results_root / "ex09" / dataset / model / f"seed_{seed}"
                run_id = f"ex09/{dataset}/{model}/seed_{seed}"
                try:
                    done = _read_json(run_dir / "DONE.json")
                    if done.get("status") != "complete":
                        raise ValueError("DONE status is not complete")
                    expected = make_config("ex09", dataset, model, seed, smoke=False)
                    if done.get("config_hash") != expected.config_hash:
                        raise ValueError("formal configuration hash mismatch")
                    metric_path = run_dir / "infer_tcngatre_failure" / "score_threshold_analysis" / "primary_metrics.json"
                    primary = _read_json(metric_path)
                    if primary.get("threshold_method") != "spot" or primary.get("label_col") != "label_any":
                        raise ValueError("primary protocol is not SPOT + label_any")
                    _verify_metrics(primary)
                    primary.update({
                        "run_id": run_id, "dataset": dataset, "model": model,
                        "seed": seed, "source": str(metric_path),
                        "source_commit": done.get("source_commit"),
                        "config_hash": done.get("config_hash"),
                        "adapter_config_hash": done.get("adapter_config_hash"),
                    })
                    rows.append(primary)
                    flight_path = run_dir / "infer_tcngatre_failure" / "score_threshold_analysis" / "per_flight_total_score_threshold_methods.csv"
                    current = pd.read_csv(flight_path)
                    current = current.loc[
                        current["threshold_method"].astype(str).str.lower().eq("spot")
                        & current["label_col"].astype(str).eq("label_any")
                    ].copy()
                    if current["flight"].duplicated().any():
                        raise ValueError("duplicate per-flight primary rows")
                    current.insert(0, "seed", seed)
                    current.insert(0, "model", model)
                    current.insert(0, "dataset", dataset)
                    flights.append(current)
                    statuses.append({"run_id": run_id, "status": "complete", "reason": ""})
                except Exception as exc:
                    reason = repr(exc)
                    statuses.append({"run_id": run_id, "status": "missing_or_invalid", "reason": reason})
                    missing.append({"run_id": run_id, "reason": reason})
    return (
        pd.DataFrame(rows),
        pd.concat(flights, ignore_index=True) if flights else pd.DataFrame(),
        pd.DataFrame(statuses),
        pd.DataFrame(missing, columns=["run_id", "reason"]),
    )


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, model), group in frame.groupby(["dataset", "model"], sort=True):
        record = {
            "dataset": dataset, "model": model,
            "seed_count": int(group["seed"].nunique()),
            "seeds": ",".join(map(str, sorted(group["seed"].astype(int).unique()))),
        }
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(record)
    result = pd.DataFrame(rows)
    average_rows = []
    for model, group in frame.groupby("model", sort=True):
        per_seed = group.groupby("seed", sort=True)
        if group["dataset"].nunique() != 3 or any(part["dataset"].nunique() != 3 for _, part in per_seed):
            continue
        seed_mean = per_seed[list(METRICS)].mean()
        record = {"dataset": "average", "model": model, "seed_count": len(seed_mean), "seeds": ",".join(map(str, seed_mean.index))}
        for metric in METRICS:
            values = seed_mean[metric].to_numpy(dtype=float)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        average_rows.append(record)
    if average_rows:
        result = pd.concat([result, pd.DataFrame(average_rows)], ignore_index=True)
    return result


def _paper_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, record in summary.iterrows():
        row = {"Dataset": record["dataset"], "Method": "GCAD" if record["model"] == "gcad" else "M²AD"}
        for metric in METRICS:
            row["AP" if metric == "average_precision" else metric.upper()] = (
                f"{record[f'{metric}_mean']:.4f} ± {record[f'{metric}_std']:.4f}"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _json_records(frame: pd.DataFrame) -> list[dict]:
    """Return JSON-safe records while preserving numeric cell types."""
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def _collect_native_m2ad(results_root: Path, datasets: list[str], seeds: list[int]) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        for seed in seeds:
            path = results_root / "ex09" / dataset / "m2ad" / f"seed_{seed}" / "native_evaluation" / "primary_metrics.json"
            if not path.is_file():
                continue
            row = _read_json(path)
            _verify_metrics(row)
            row.update({"dataset": dataset, "model": "m2ad", "seed": seed, "source": str(path)})
            rows.append(row)
    return pd.DataFrame(rows)


def _write_report(output: Path, summary: pd.DataFrame, native_summary: pd.DataFrame, complete: bool, expected: int, actual: int) -> None:
    labels = {"gcad": "GCAD", "m2ad": "M²AD"}
    dataset_labels = {"alfa": "ALFA", "gpsdata": "GPSData", "simulate": "Simulate", "average": "Average"}
    lines = [
        "# GCAD 与 M²AD 复现结果汇总",
        "",
        f"运行完整性：{'通过' if complete else '未通过'}（{actual}/{expected} 个正式运行）。",
        "",
        "## 五种子论文结果",
        "",
        "| 数据集 | 方法 | Precision | Recall | F1 | FPR | AUROC | AP |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    order = {"alfa": 0, "gpsdata": 1, "simulate": 2, "average": 3}
    records = summary.sort_values(["dataset", "model"], key=lambda x: x.map(order) if x.name == "dataset" else x)
    for _, row in records.iterrows():
        def cell(metric: str) -> str:
            return f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_std']:.4f}"
        lines.append(
            f"| {dataset_labels.get(row['dataset'], row['dataset'])} | {labels.get(row['model'], row['model'])} | "
            f"{cell('precision')} | {cell('recall')} | {cell('f1')} | {cell('fpr')} | "
            f"{cell('auroc')} | {cell('average_precision')} |"
        )
    lines.extend([
        "",
        "## 复现口径",
        "",
        "- GCAD 复用官方 TSMixerRevIN 预测器和基于梯度的 Granger 因果图偏离分数。",
        "- M²AD 复用官方 LSTM 预测器、逐传感器 GMM 与 Gamma 校准组件，以组合 Fisher 统计量作为连续异常分数。",
        "- 两个方法均使用固定训练、验证和故障航班划分；论文主表采用 `label_any + causal EMA + flight-wise SPOT` 完成在线决策。",
        "- 表中为五个模型种子的均值与样本标准差；未使用 point adjustment 或故障标签校准。",
        "",
        "M²AD 的原生 GMM/Gamma 校准结果另行保存，用于核对官方决策流程，不与主表的统一决策结果混用。",
    ])
    if not native_summary.empty:
        lines.extend([
            "", "## M²AD 原生 Gamma 校准结果（诊断）", "",
            "| 数据集 | F1 | Precision | Recall | FPR | AUROC | AP |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for _, row in native_summary.loc[native_summary["dataset"] != "average"].iterrows():
            def native_cell(metric: str) -> str:
                return f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_std']:.4f}"
            lines.append(
                f"| {dataset_labels.get(row['dataset'], row['dataset'])} | {native_cell('f1')} | "
                f"{native_cell('precision')} | {native_cell('recall')} | {native_cell('fpr')} | "
                f"{native_cell('auroc')} | {native_cell('average_precision')} |"
            )
    (output / "REPRODUCTION_RESULTS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize GCAD and M2AD five-seed results.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS))
    parser.add_argument("--output-root", default=str(DEFAULT_RESULTS / "summary" / "gcad_m2ad_2025"))
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs, flights, statuses, missing = _collect(Path(args.results_root).resolve(), args.datasets, args.seeds)
    summary = _summarize(runs) if not runs.empty else pd.DataFrame()
    paper = _paper_table(summary) if not summary.empty else pd.DataFrame()
    native_runs = _collect_native_m2ad(Path(args.results_root).resolve(), args.datasets, args.seeds)
    native_summary = _summarize(native_runs) if not native_runs.empty else pd.DataFrame()
    expected = len(args.datasets) * len(MODELS) * len(args.seeds)
    complete = len(runs) == expected and missing.empty
    provenance = pd.DataFrame([
        {"Method": "GCAD", "Year": 2025, "Official commit": "e3e0c039468c105edf798747269ba87c309b573f", "Raw score": "Gradient-causality graph deviation", "Decision": "causal EMA + flight-wise SPOT", "Point adjustment": "No"},
        {"Method": "M²AD", "Year": 2025, "Official commit": "05ac998e55123c51c4a4dd47ad31343bc3c25c23", "Raw score": "GMM-combined Fisher statistic", "Decision": "causal EMA + flight-wise SPOT", "Native decision": "Gamma p-value < 0.001", "Point adjustment": "No"},
    ])
    runs.to_csv(output / "primary_metrics_all_runs.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "primary_metrics_seed_summary.csv", index=False, encoding="utf-8-sig")
    paper.to_csv(output / "paper_table.csv", index=False, encoding="utf-8-sig")
    flights.to_csv(output / "per_flight_primary_all_runs.csv", index=False, encoding="utf-8-sig")
    statuses.to_csv(output / "run_status.csv", index=False, encoding="utf-8-sig")
    missing.to_csv(output / "missing_experiment_cells.csv", index=False, encoding="utf-8-sig")
    provenance.to_csv(output / "reproduction_protocol.csv", index=False, encoding="utf-8-sig")
    native_runs.to_csv(output / "m2ad_native_metrics_all_runs.csv", index=False, encoding="utf-8-sig")
    native_summary.to_csv(output / "m2ad_native_metrics_seed_summary.csv", index=False, encoding="utf-8-sig")
    (output / "paper_table.tex").write_text(
        paper.to_latex(index=False, escape=False, float_format=lambda value: f"{value:.4f}"),
        encoding="utf-8",
    )
    workbook_data = {
        "paper_table": _json_records(paper),
        "seed_summary": _json_records(summary),
        "all_runs": _json_records(runs),
        "run_status": _json_records(statuses),
        "reproduction_protocol": _json_records(provenance),
        "m2ad_native_runs": _json_records(native_runs),
        "m2ad_native_summary": _json_records(native_summary),
    }
    (output / "workbook_data.json").write_text(
        json.dumps(workbook_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "status": "complete" if complete else "incomplete",
        "expected_runs": expected, "complete_runs": len(runs), "missing_runs": len(missing),
        "datasets": args.datasets, "seeds": args.seeds, "output_root": str(output),
        "primary_protocol": "label_any + causal EMA + flight-wise SPOT + pooled-window confusion counts",
    }
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        with pd.ExcelWriter(output / "GCAD_M2AD_paper_results.xlsx", engine="openpyxl") as writer:
            paper.to_excel(writer, sheet_name="论文主表", index=False)
            summary.to_excel(writer, sheet_name="五种子汇总", index=False)
            runs.to_excel(writer, sheet_name="逐种子结果", index=False)
            native_summary.to_excel(writer, sheet_name="M2AD原生Gamma汇总", index=False)
            native_runs.to_excel(writer, sheet_name="M2AD原生Gamma逐种子", index=False)
            statuses.to_excel(writer, sheet_name="运行状态", index=False)
            provenance.to_excel(writer, sheet_name="复现协议", index=False)
    except ImportError as exc:
        print(f"[WARN] Excel not generated because openpyxl is unavailable: {exc}", flush=True)
    _write_report(output, summary, native_summary, complete, expected, len(runs))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 2 if args.require_complete and not complete else 0


if __name__ == "__main__":
    raise SystemExit(main())
