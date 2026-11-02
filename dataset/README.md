# Dataset layout

This directory contains the complete local ALFA, GPSData, and Simulate bundle used by this project: prepared flight CSV files, failure labels, split manifests, preprocessing metadata, and preparation scripts. It is the project's prepared dataset bundle, rather than a complete mirror of the upstream raw-data collections.

| Dataset | Normal training flights | Normal validation flights | Failure flights | Bundle files |
| --- | ---: | ---: | ---: | ---: |
| ALFA | 29 | 1 | 16 | 99 |
| GPSData | 1 | 1 | 2 | 14 |
| Simulate | 8 | 2 | 2 | 31 |

The 144 bundle files total approximately 568.6 MiB. File counts include labels, metadata, preparation scripts, and locally retained source/intermediate CSV files, so they exceed the number of scored flights. `SHA256SUMS.txt` records the exact hashes of these 144 files. The existing manifests determine which prepared directories and flights are loaded; do not combine the source/intermediate CSV files with the prepared flight splits.

The prepared inputs follow this layout:

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

