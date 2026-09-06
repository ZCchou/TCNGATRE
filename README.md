# TCNGATRE

Official implementation of **TCNGATRE: Knowledge-Guided Temporal and Graph Forecasting with Dynamic Sensor-Dependency Refinement for UAV Anomaly Detection**.

TCNGATRE combines a causal temporal convolutional forecasting backbone with two interleaved graph-correction stages. A static sensor-dependency prior is estimated from normal training flights using the maximal information coefficient (MIC). Sample-adaptive graph attention refines this prior, and the fused graph guides gated multi-hop message passing before future sensor trajectories are predicted. Forecast residuals are converted into online anomaly decisions using causal smoothing and flight-wise SPOT.

## Repository scope

The `main` branch contains only the proposed model and the code required to train, infer, and evaluate it:

- `TCNGATRE/model/`: temporal and graph forecasting network.
- `TCNGATRE/data/`: flight-level split and sliding-window data loaders.
- `TCNGATRE/util/`: causal preprocessing and MIC graph construction.
- `TCNGATRE/utils/`: normalisation, output, and threshold helpers.
- `common/threshold_methods.py`: SPOT and metric implementation used by evaluation.
- `dataset/*/dataset_manifest.json`: dataset split definitions used in the paper.

Baseline, ablation, hyperparameter, post-hoc analysis, generated-result, and manuscript files are intentionally excluded from this branch.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m pip install -r requirements.txt
```

`minepy` is required for MIC graph estimation. If a binary wheel is not available for the local Python version, install it from conda-forge.

## Data preparation

The flight CSV files are not stored in this code-only branch. Place the prepared ALFA, GPSData, or Simulate files under `dataset/` using the layout described in [`dataset/README.md`](dataset/README.md). The included manifests preserve the flight-level splits used in the paper.

Each input CSV must contain a monotonic time column followed by numeric sensor columns. Normal training and validation flights are placed in `No_Failure`, and scored fault flights are placed in `Failure`. Failure-label CSV files use matching flight stems.

## Training, inference, and evaluation

Run commands from the model directory:

```bash
cd TCNGATRE

python train_tcngatre.py --dataset alfa
python infer_tcngatre.py --dataset alfa
python eval_tcngatre.py --dataset alfa
```

Replace `alfa` with `gpsdata` or `simulate` as needed. Training automatically constructs or validates the MIC graph using only the normal training flights declared by the selected manifest. Outputs are written to `TCNGATRE/runs/tcngatre_<dataset>/` by default.

The main configuration is defined in `TCNGATRE/tcngatreconfig.py`. It can be overridden through the documented `UAV_TCNGATRE_*` environment variables, including data paths, output paths, model dimensions, window settings, training parameters, and threshold settings.

## Expected outputs

After a complete run, the default run directory contains:

- the MIC graph and its provenance metadata;
- training configuration, history, normalisation statistics, and `best.pt`;
- validation and failure-flight forecasts and residual scores;
- window-level labels, threshold decisions, and aggregate metrics.

## Reproducibility notes

- Static MIC graphs are built from normal training flights only.
- Normalisation statistics are fitted on the normal training split.
- Flight-level splits are read from the dataset manifests.
- The model uses four causal TCN blocks, with graph correction after the second and fourth blocks.
- Evaluation uses causal score smoothing and flight-wise SPOT without failure-label threshold calibration.

## Full research code

The complete experimental framework, including baselines and revision analyses, remains available on the `codex-revision-20260821` branch.

