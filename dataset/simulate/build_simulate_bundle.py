from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW_NORMAL_DIR = ROOT / "normal"
RAW_ABNORMAL_DIR = ROOT / "abnormal"
RAW_MANIFEST_PATH = ROOT / "manifest.csv"

WIDE_ROOT_NAME = "wide_flights_set_simulate_faultflag_rawsplit"
LABELS_DIR_NAME = "wide_flights_failure_labels"

WIDE_ROOT = ROOT / WIDE_ROOT_NAME
NO_FAILURE_DIR = WIDE_ROOT / "No_Failure"
FAILURE_DIR = WIDE_ROOT / "Failure"
LABELS_DIR = ROOT / LABELS_DIR_NAME
DATASET_MANIFEST_PATH = ROOT / "dataset_manifest.json"

FEATURE_COLUMNS = [
    "u_cmd",
    "position",
    "velocity",
    "accel",
    "torque",
    "current",
    "voltage",
]
TRAIN_FLIGHTS = [f"normal_{idx:02d}" for idx in range(1, 9)]
VAL_FLIGHTS = [f"normal_{idx:02d}" for idx in range(9, 11)]
PHYSICAL_TRIM_LEADING_SEC = 2.0
PHYSICAL_TRIM_TIME_AXIS = "rebased_to_zero"


def parse_time_to_seconds(series: pd.Series) -> pd.Series:
    numeric = series.astype(str).str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)
    return pd.to_numeric(numeric, errors="coerce")


def load_raw_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = ["Time", *FEATURE_COLUMNS, "fault_flag"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    out = pd.DataFrame({"t": parse_time_to_seconds(df["Time"])})
    for column in FEATURE_COLUMNS:
        out[column] = pd.to_numeric(df[column], errors="coerce")
    out["fault_flag"] = pd.to_numeric(df["fault_flag"], errors="coerce").fillna(0.0)
    if "fault_strength" in df.columns:
        out["fault_strength"] = pd.to_numeric(df["fault_strength"], errors="coerce").fillna(0.0)
    else:
        out["fault_strength"] = out["fault_flag"].astype(float)
    out = out.dropna(subset=["t"])
    if out.empty:
        raise ValueError(f"No valid rows after parsing time in {csv_path}")
    out = out.sort_values("t", kind="mergesort").reset_index(drop=True)

    raw_active = out.loc[out["fault_flag"].astype(float) > 0.5, "t"]
    raw_first_positive = float(raw_active.iloc[0]) if not raw_active.empty else float("nan")

    raw_t0 = float(out["t"].iloc[0])
    trim_anchor = raw_t0 + float(PHYSICAL_TRIM_LEADING_SEC)
    trimmed = out.loc[out["t"].astype(float) >= trim_anchor - 1e-9].copy()
    if trimmed.empty:
        raise ValueError(f"No rows remain after trimming first {PHYSICAL_TRIM_LEADING_SEC:g}s in {csv_path}")
    trimmed["t"] = trimmed["t"].astype(float) - trim_anchor
    trimmed.loc[trimmed["t"].abs() < 1e-9, "t"] = 0.0
    trimmed = trimmed.reset_index(drop=True)
    trimmed.attrs["raw_first_positive_t"] = raw_first_positive
    trimmed.attrs["trim_anchor_sec"] = trim_anchor
    return trimmed


def write_processed_csv(df: pd.DataFrame, out_path: Path):
    payload = df[["t", *FEATURE_COLUMNS]].copy()
    payload.to_csv(out_path, index=False, encoding="utf-8")


def write_label_csv(df: pd.DataFrame, out_path: Path):
    payload = pd.DataFrame(
        {
            "t": df["t"].astype(float),
            "anomaly_label": (df["fault_flag"].astype(float) > 0.5).astype(int),
        }
    )
    payload.to_csv(out_path, index=False, encoding="utf-8")


def infer_fault_mode(stem: str) -> str:
    prefix = "abnormal_"
    mode = stem[len(prefix):] if stem.startswith(prefix) else stem
    return mode if mode else "fault"


def build_raw_manifest(abnormal_fault_info: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx in range(1, 11):
        rows.append(
            {
                "name": f"normal_{idx:02d}",
                "is_fault": 0,
                "fault_mode": "none",
                "fault_start": float("nan"),
                "raw_fault_start_before_trim_sec": float("nan"),
                "seed": 202600 + idx,
                "file": f"normal_{idx:02d}.csv",
            }
        )
    for stem in sorted(abnormal_fault_info):
        info = abnormal_fault_info[stem]
        rows.append(
            {
                "name": stem,
                "is_fault": 1,
                "fault_mode": infer_fault_mode(stem),
                "fault_start": float(info["trimmed_first_positive_t"]),
                "raw_fault_start_before_trim_sec": float(info["raw_first_positive_t"]),
                "seed": "",
                "file": f"{stem}.csv",
            }
        )
    return pd.DataFrame(rows)


def build_manifest(failure_total: int) -> dict:
    return {
        "manifest_version": 1,
        "dataset_kind": "simulate_shared_wide_csv",
        "wide_root_discovery": {
            "mode": "scan_for_no_failure_and_failure",
        },
        "labels_dirname": LABELS_DIR_NAME,
        "prefail_normal_suffix": "__prefail_normal",
        "prefail_normal_policy": "train_only",
        "failure_label_time_offset_sec": 0.0,
        "trim_leading_sec": 0.0,
        "physical_trim_leading_sec": float(PHYSICAL_TRIM_LEADING_SEC),
        "physical_trim_time_axis": PHYSICAL_TRIM_TIME_AXIS,
        "legacy_train_flights": TRAIN_FLIGHTS,
        "legacy_val_flights": VAL_FLIGHTS,
        "expected_counts": {
            "no_failure_total": 10,
            "classic_no_failure": 10,
            "prefail_normal": 0,
            "failure_total": int(failure_total),
            "train_normal": 8,
            "val_normal": 2,
        },
    }


def reset_generated_dirs():
    for path in [NO_FAILURE_DIR, FAILURE_DIR, LABELS_DIR]:
        if path.exists():
            shutil.rmtree(path)
    NO_FAILURE_DIR.mkdir(parents=True, exist_ok=True)
    FAILURE_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)


def verify_fault_alignment(raw_manifest: pd.DataFrame, abnormal_fault_info: dict[str, dict[str, float]]):
    rows: list[dict] = []
    for _, row in raw_manifest.loc[raw_manifest["is_fault"] == 1].iterrows():
        name = str(row["name"])
        fault_start = float(row["fault_start"])
        info = abnormal_fault_info[name]
        raw_first_positive_t = float(info["raw_first_positive_t"])
        trimmed_first_positive_t = float(info["trimmed_first_positive_t"])
        rows.append(
            {
                "flight": name,
                "raw_fault_start_before_trim_sec": raw_first_positive_t,
                "trimmed_manifest_fault_start": fault_start,
                "trimmed_fault_flag_first_positive_t": trimmed_first_positive_t,
                "physical_trim_leading_sec": float(PHYSICAL_TRIM_LEADING_SEC),
                "abs_delta_sec": abs(trimmed_first_positive_t - fault_start),
            }
        )
    return pd.DataFrame(rows).sort_values("flight", kind="mergesort").reset_index(drop=True)


def main():
    reset_generated_dirs()

    abnormal_fault_info: dict[str, dict[str, float]] = {}

    for csv_path in sorted(RAW_NORMAL_DIR.glob("*.csv")):
        stem = csv_path.stem
        frame = load_raw_frame(csv_path)
        write_processed_csv(frame, NO_FAILURE_DIR / f"{stem}.csv")

    for csv_path in sorted(RAW_ABNORMAL_DIR.glob("*.csv")):
        stem = csv_path.stem
        frame = load_raw_frame(csv_path)
        write_processed_csv(frame, FAILURE_DIR / f"{stem}.csv")
        write_label_csv(frame, LABELS_DIR / f"{stem}.csv")
        active = frame.loc[frame["fault_flag"].astype(float) > 0.5, "t"]
        if active.empty:
            raise ValueError(f"Fault file has no positive fault_flag rows: {csv_path}")
        abnormal_fault_info[stem] = {
            "raw_first_positive_t": float(frame.attrs.get("raw_first_positive_t", float("nan"))),
            "trimmed_first_positive_t": float(active.iloc[0]),
        }

    raw_manifest = build_raw_manifest(abnormal_fault_info)
    raw_manifest.to_csv(RAW_MANIFEST_PATH, index=False, encoding="utf-8")

    manifest_payload = build_manifest(failure_total=len(abnormal_fault_info))
    DATASET_MANIFEST_PATH.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    alignment_df = verify_fault_alignment(raw_manifest=raw_manifest, abnormal_fault_info=abnormal_fault_info)
    alignment_df.to_csv(ROOT / "fault_alignment_check.csv", index=False, encoding="utf-8")

    print(f"[OK] wrote manifest: {DATASET_MANIFEST_PATH}")
    print(f"[OK] no_failure={len(list(NO_FAILURE_DIR.glob('*.csv')))} failure={len(list(FAILURE_DIR.glob('*.csv')))}")
    print(f"[OK] labels={len(list(LABELS_DIR.glob('*.csv')))} alignment_report={ROOT / 'fault_alignment_check.csv'}")


if __name__ == "__main__":
    main()
