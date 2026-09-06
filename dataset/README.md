# Dataset layout

This repository retains the flight-level split manifests used by TCNGATRE but does not redistribute the flight CSV files.

Prepare each dataset in the following form:

```text
dataset/
  <dataset_name>/
    dataset_manifest.json
    <wide_root>/
      No_Failure/
        <normal_train_or_validation_flight>.csv
      Failure/
        <failure_flight>.csv
    wide_flights_failure_labels/
      <matching_flight_name>.csv
```

Supported dataset names are `alfa`, `gpsdata`, and `simulate`.

The manifest may set `wide_root_dirname` explicitly. If it is omitted, TCNGATRE requires exactly one directory beneath the dataset root that contains both `No_Failure` and `Failure` subdirectories. File stems must match the names listed in `legacy_train_flights` and `legacy_val_flights`.

The evaluation label directory must contain one CSV per scored flight. Label alignment is performed by the window end and forecast horizon; labels are not used to train the model or calibrate the flight-wise SPOT threshold.

To use a data directory outside this repository, set `UAV_TCNGATRE_DATA_ROOT` and, if necessary, `UAV_TCNGATRE_LABELS_ROOT` before running the training, inference, or evaluation entry points.

