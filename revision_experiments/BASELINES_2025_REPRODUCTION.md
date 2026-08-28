# GCAD and M2AD 2025 baseline reproduction

This isolated experiment adds two official-code 2025 baselines without changing
the legacy model implementations or raw datasets.

## Sources

- GCAD (AAAI 2025): <https://github.com/Tc99m/GCAD>, pinned at
  `e3e0c039468c105edf798747269ba87c309b573f`.
- M2AD (AISTATS 2025): <https://github.com/sarahmish/M2AD>, pinned at
  `05ac998e55123c51c4a4dd47ad31343bc3c25c23`.

GCAD retains the official TSMixerRevIN predictor and the gradient-derived causal
graph deviation score. M2AD retains the official LSTM forecasting network,
point-error calculation, sensor-wise GMM, and Gamma calibration components. Its
combined Fisher statistic is evaluated under the project protocol: `label_any`,
causal EMA, and flight-wise SPOT without failure-label calibration or point
adjustment. The native Gamma decision (`p < 0.001`) is saved separately as a
diagnostic result and is not mixed into the common-protocol paper table.

## Data protocol

| Dataset | Normal train | Normal validation | Failure | Channels |
|---|---:|---:|---:|---:|
| ALFA | 29 | 1 | 16 | 12 |
| GPSData | 1 | 1 | 2 | 45 |
| Simulate | 8 | 2 | 2 | 7 |

Normalization statistics are fitted only on normal training flights. Windows
never cross flight boundaries. Five model seeds are used: 0, 1, 2, 3, and 4.

## Commands

```text
python revision_experiments/run_revision.py fetch-baselines --baselines m2ad
python revision_experiments/run_revision.py prepare-baseline-data --datasets all
python -u revision_experiments/run_revision.py run --experiments ex09 --variants m2ad --datasets all --seeds 0 --smoke --manifest-name ex09_m2ad_smoke.csv
python -u revision_experiments/run_revision.py run --experiments ex09 --variants m2ad --datasets all --seeds 0,1,2,3,4 --manifest-name ex09_m2ad_formal_5seed.csv
python revision_experiments/summarize_2025_baselines.py --datasets alfa gpsdata simulate --seeds 0 1 2 3 4 --require-complete
```

Formal output is stored under `revision_results/protocol_v1/ex09/`. The summary
is stored under `revision_results/protocol_v1/summary/gcad_m2ad_2025/`.
