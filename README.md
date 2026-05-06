# TCNGATRE

Code release for **TCNGATRE**, an interleaved temporal-graph forecasting framework for UAV multivariate sensor anomaly detection, together with the unified baseline evaluation code used in this project.

## Repository Scope

This repository is organized as a **code-first release**. Local-only assets that are useful for experimentation but should not be uploaded to GitHub are intentionally excluded from the public package, including:

- raw and processed datasets under `dataset/`
- model checkpoints and inference outputs under `*/runs/`
- batch logs and local cache files
- ad hoc generated figures used during analysis

The root `.gitignore` keeps these assets local by default.

## Included Code

- `TCNGATRE/`: proposed method
- `USAD/`, `Recurrent_AE/`, `TranAD/`, `OmniAnomaly/`, `BeatGAN/`: baseline implementations
- `common/`: shared utilities for preprocessing, thresholding, evaluation, and plotting
- `ablation/`: ablation experiment variants and summarization scripts
- `hparam/`: hyperparameter sensitivity configurations and summarization scripts
- `run_all_models_all_datasets.py`: unified multi-model execution entry
- `summarize_all_model_results.py`: aggregate experiment summaries
- `plot_all_model_metrics.py`: metric plotting utilities

## Local Data Convention

The training and evaluation code expects datasets to be placed locally under:

- `dataset/alfa`
- `dataset/gpsdata`
- `dataset/simulate`

These directories are treated as local data mounts and are not part of the GitHub release.

## Clean GitHub Export

To create a GitHub-friendly snapshot that contains code only, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\export_clean_repo.ps1 -TargetDir F:\TCNGATRE_github_clean -Force -InitGit
```

This command copies the publishable code into `F:\TCNGATRE_github_clean`, excludes datasets and experiment artifacts, and initializes a fresh Git repository there.

## Basic Usage

Install dependencies:

```powershell
pip install -r requirements.txt
```

Example training / inference commands:

```powershell
cd TCNGATRE
python train_tcngatre.py --dataset alfa
python infer_tcngatre.py --dataset gpsdata
python eval_tcngatre.py --dataset simulate
```

Baseline scripts follow the same dataset convention inside their corresponding subdirectories.
