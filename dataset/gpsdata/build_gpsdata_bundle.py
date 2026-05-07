from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "gpsdata"
RAW_NO_FAILURE = RAW_ROOT / "No_Failure"
RAW_FAILURE = RAW_ROOT / "Failure"

WIDE_ROOT_NAME = "wide_flights_set_gps_labelsplit"
LABELS_DIR_NAME = "wide_flights_failure_labels"

WIDE_ROOT = ROOT / WIDE_ROOT_NAME
NO_FAILURE_DIR = WIDE_ROOT / "No_Failure"
FAILURE_DIR = WIDE_ROOT / "Failure"
LABELS_DIR = ROOT / LABELS_DIR_NAME
DATASET_MANIFEST_PATH = ROOT / "dataset_manifest.json"
SPLIT_REPORT_PATH = ROOT / "split_report.json"

NO_FAILURE_RAW_NAME = "Benign Flight.csv"
FAILURE_FILE_MAP = {
    "GPS Jamming.csv": "gps_jamming",
    "GPS Spoofing.csv": "gps_spoofing",
}
TRAIN_FLIGHT = "benign_flight_train"
VAL_FLIGHT = "benign_flight_val"
VAL_RATIO = 0.20
TIMESTAMP_DIVISOR = 1_000_000.0


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip().lstrip("\ufeff") for col in out.columns]
    return out


def resolve_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"t", "timestamp", "time_utc_usec", "label"}
    return [str(col) for col in df.columns if str(col) not in excluded]


def parse_raw_frame(csv_path: Path, feature_columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    df = sanitize_columns(pd.read_csv(csv_path, engine="python"))
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing 'timestamp' column in {csv_path}")

    if feature_columns is None:
        feature_columns = resolve_feature_columns(df)
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns in {csv_path}: {missing}")

    out = pd.DataFrame()
    out["t"] = (
        pd.to_numeric(df["timestamp"], errors="coerce") - float(pd.to_numeric(df["timestamp"], errors="coerce").iloc[0])
    ) / TIMESTAMP_DIVISOR
    for column in feature_columns:
        out[column] = pd.to_numeric(df[column], errors="coerce")

    if "label" in df.columns:
        out["anomaly_label"] = (pd.to_numeric(df["label"], errors="coerce").fillna(0.0) > 0.5).astype(int)
    else:
        out["anomaly_label"] = 0

    out = out.dropna(subset=["t"]).reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No valid rows after parsing timestamp in {csv_path}")
    return out, list(feature_columns)


def rebase_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["t"] = pd.to_numeric(out["t"], errors="coerce") - float(pd.to_numeric(out["t"], errors="coerce").iloc[0])
    return out.reset_index(drop=True)


def write_processed_csv(df: pd.DataFrame, out_path: Path, feature_columns: list[str]) -> None:
    payload = df[["t", *feature_columns]].copy()
    payload.to_csv(out_path, index=False, encoding="utf-8")


def write_label_csv(df: pd.DataFrame, out_path: Path) -> None:
    payload = pd.DataFrame(
        {
            "t": pd.to_numeric(df["t"], errors="coerce").astype(float),
            "anomaly_label": pd.to_numeric(df["anomaly_label"], errors="coerce").fillna(0).astype(int),
        }
    )
    payload.to_csv(out_path, index=False, encoding="utf-8")


def reset_generated_dirs() -> None:
    for path in [WIDE_ROOT, LABELS_DIR]:
        if path.exists():
            shutil.rmtree(path)
    NO_FAILURE_DIR.mkdir(parents=True, exist_ok=True)
    FAILURE_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)


def build_manifest(feature_columns: list[str]) -> dict:
    return {
        "manifest_version": 1,
        "dataset_kind": "gpsdata_shared_wide_csv",
        "wide_root_dirname": WIDE_ROOT_NAME,
        "wide_root_discovery": {
            "mode": "scan_for_no_failure_and_failure",
        },
        "labels_dirname": LABELS_DIR_NAME,
        "prefail_normal_suffix": "__prefail_normal",
        "prefail_normal_policy": "train_only",
        "failure_label_time_offset_sec": 0.0,
        "trim_leading_sec": 0.0,
        "legacy_train_flights": [TRAIN_FLIGHT],
        "legacy_val_flights": [VAL_FLIGHT],
        "expected_counts": {
            "no_failure_total": 2,
            "classic_no_failure": 2,
            "prefail_normal": 0,
            "failure_total": 2,
            "train_normal": 1,
            "val_normal": 1,
        },
        "feature_columns": list(feature_columns),
        "split_policy": {
            "source_no_failure_file": NO_FAILURE_RAW_NAME,
            "mode": "single_flight_contiguous_80_20",
            "train_ratio": 0.8,
            "val_ratio": VAL_RATIO,
            "train_flight_name": TRAIN_FLIGHT,
            "val_flight_name": VAL_FLIGHT,
        },
    }


def main() -> None:
    reset_generated_dirs()

    benign_path = RAW_NO_FAILURE / NO_FAILURE_RAW_NAME
    benign_df, feature_columns = parse_raw_frame(benign_path)
    split_index = max(1, min(len(benign_df) - 1, int(len(benign_df) * (1.0 - VAL_RATIO))))
    train_df = rebase_time(benign_df.iloc[:split_index].copy())
    val_df = rebase_time(benign_df.iloc[split_index:].copy())

    write_processed_csv(train_df, NO_FAILURE_DIR / f"{TRAIN_FLIGHT}.csv", feature_columns=feature_columns)
    write_processed_csv(val_df, NO_FAILURE_DIR / f"{VAL_FLIGHT}.csv", feature_columns=feature_columns)
    write_label_csv(train_df, LABELS_DIR / f"{TRAIN_FLIGHT}.csv")
    write_label_csv(val_df, LABELS_DIR / f"{VAL_FLIGHT}.csv")

    failure_reports: list[dict] = []
    for raw_name, output_stem in FAILURE_FILE_MAP.items():
        raw_path = RAW_FAILURE / raw_name
        failure_df, current_features = parse_raw_frame(raw_path, feature_columns=feature_columns)
        if list(current_features) != list(feature_columns):
            raise ValueError(f"Feature columns mismatch in {raw_path}")
        failure_df = rebase_time(failure_df)
        write_processed_csv(failure_df, FAILURE_DIR / f"{output_stem}.csv", feature_columns=feature_columns)
        write_label_csv(failure_df, LABELS_DIR / f"{output_stem}.csv")
        positive = failure_df.loc[failure_df["anomaly_label"] > 0, "t"]
        failure_reports.append(
            {
                "flight": output_stem,
                "rows": int(len(failure_df)),
                "positive_rows": int(failure_df["anomaly_label"].sum()),
                "first_positive_t": (None if positive.empty else float(positive.iloc[0])),
                "last_positive_t": (None if positive.empty else float(positive.iloc[-1])),
            }
        )

    manifest_payload = build_manifest(feature_columns=feature_columns)
    DATASET_MANIFEST_PATH.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    split_report = {
        "no_failure_source": {
            "file": str(benign_path.name),
            "rows_total": int(len(benign_df)),
            "rows_train": int(len(train_df)),
            "rows_val": int(len(val_df)),
            "train_ratio_actual": float(len(train_df) / max(len(benign_df), 1)),
            "val_ratio_actual": float(len(val_df) / max(len(benign_df), 1)),
            "train_t_end": float(train_df["t"].iloc[-1]) if len(train_df) > 0 else None,
            "val_t_end": float(val_df["t"].iloc[-1]) if len(val_df) > 0 else None,
        },
        "failure_flights": failure_reports,
        "feature_count": int(len(feature_columns)),
        "feature_columns": list(feature_columns),
    }
    SPLIT_REPORT_PATH.write_text(
        json.dumps(split_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] wrote manifest: {DATASET_MANIFEST_PATH}")
    print(f"[OK] wide_root={WIDE_ROOT}")
    print(f"[OK] labels_root={LABELS_DIR}")
    print(
        "[OK] no_failure_train_rows="
        f"{len(train_df)} no_failure_val_rows={len(val_df)} failure_files={len(FAILURE_FILE_MAP)}"
    )


if __name__ == "__main__":
    main()
