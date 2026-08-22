# TCNGATRE revision experiments

This directory is an add-only experiment harness for the KNOSYS revision.  It
imports the legacy TCNGATRE data/model helpers read-only and writes all generated
files under `revision_results/`. SHA-256 legacy auditing is available through
`verify-legacy` or `UAV_STRICT_LEGACY_INTEGRITY=1`, but normal experiment runs do
not enable the platform-sensitive audit gate and therefore are not blocked by
line-ending differences. Approved hashes remain recorded in
`manifests/approved_legacy_changes.json` for explicit audits.

## Commands

```powershell
python revision_experiments/run_revision.py snapshot
python revision_experiments/run_revision.py doctor
python revision_experiments/run_revision.py verify-legacy
python revision_experiments/run_revision.py fetch-baselines
python revision_experiments/run_revision.py prepare-baseline-data --datasets all
python revision_experiments/run_revision.py audit-adapters --seed 0
python revision_experiments/run_revision.py smoke --preset core_ablation --datasets all
python revision_experiments/run_revision.py run --preset core_ablation --datasets all --seeds 0,1,2,3,4 --dry-run
python revision_experiments/run_revision.py summarize --preset core_ablation --datasets all --seeds 0,1,2,3,4 --require-complete
```

## Focused reviewer ablation

The `core_ablation` preset removes duplicate or low-priority configurations and
keeps seven reviewer-facing variants: Full, TCN-only, late graph fusion,
static-only, dynamic-only, learned scalar fusion, and a shuffled-MIC prior.
This expands to 7 variants x 3 datasets x 5 seeds = 105 runs. Repeating the
same run command resumes partial checkpoints and reuses a completed run only
when both its resolved configuration hash and current data-protocol hash match.
The `static_only` variant is an explicitly lightweight static-graph baseline:
after the final TCN block it applies one parameter-free convex smoothing step
between each sensor state and its MIC-neighbour aggregate. It has no dynamic
attention, learned message projection, gate, or multi-hop/interleaved graph
correction. This is a compound lightweight baseline rather than a strict
equal-capacity graph-source ablation; `fusion_static` remains available for the
equal-capacity static-fusion comparison.
Formal ALFA ablation runs use window stride 16 to reduce the 29-flight training
window count; GPSData and Simulate retain stride 4. Formal GPSData ablation
runs use batch size 32, while ALFA and Simulate retain batch size 128. Smoke
stride and batch size remain 64 and 32, respectively.

```powershell
python -u revision_experiments/run_revision.py run --preset core_ablation --datasets all --seeds 0,1,2,3,4 --manifest-name core_ablation_5seed.csv
python revision_experiments/run_revision.py summarize --preset core_ablation --datasets all --seeds 0,1,2,3,4 --require-complete
```

The focused summary is written under
`revision_results/protocol_v1/summary/core_ablation/`. It contains the audited
run-status matrix, missing/invalid cells, every Micro `SPOT + label_any` seed
result, mean/std/count tables, per-flight inputs, and paired F1 bootstrap,
permutation, rank-biserial, and Holm-corrected significance results. Runs from
the former ALFA 9/1/16 protocol are rejected by their data-protocol hash and
cannot enter the statistical tables.

## Six-model repeated runs

The existing all-model runner keeps its original behavior when `--seeds` is
omitted. Seeded mode isolates every model/dataset/seed run, supports stage-level
resume, and writes only under `revision_results/protocol_v1/main_comparison/`.

```powershell
python run_all_models_all_datasets.py --seeds 0 1 2 3 4 --dry-run
python run_all_models_all_datasets.py --datasets simulate --seeds 0 --smoke --keep-going
python run_all_models_all_datasets.py --seeds 0 1 2 3 4 --keep-going
```

Seeded formal runs default to the performance-oriented `--determinism seeded`
policy: Python, NumPy, PyTorch and CUDA RNG seeds are fixed, while globally
forcing slow deterministic CUDA kernels is avoided. This matches the purpose
of five-seed statistical repetition; exact bitwise reruns can still be requested
with `--determinism strict`. Native per-flight plotting is disabled by default
because it does not affect scores or Micro metrics; pass `--plots` when those
diagnostic figures are specifically needed. Model architecture, epoch counts,
optimizer settings, data splits, inference scores and thresholds are unchanged
from the original single-seed commands; the GPSData memory override below is
the only seeded batch-size exception.

GPSData has substantially more sensor nodes than ALFA. Seeded TCNGATRE runs on
GPSData therefore use a targeted physical batch size of 32 to reduce graph
memory use while retaining better GPU utilization than the former value 16;
every other model/dataset keeps its native batch size. The
override is recorded in each stage signature and provenance file and can be
changed explicitly with `--tcngatre-gps-batch-size`.

Seeded formal TCNGATRE runs on ALFA use sample stride 16 for training,
validation, and failure inference, reducing the window count to roughly one
quarter of stride 4. TCNGATRE on GPSData and Simulate, and every other model,
remain unchanged. The ALFA value can be overridden with
`--tcngatre-alfa-sample-stride`; it is recorded in manifests, provenance,
stage signatures, and completion markers so stride-4 ALFA results are rerun
instead of being mixed into the new summary.

The primary comparison collector uses only the Micro result selected from the
global `summary_metrics.csv` row with `threshold_method=spot` and
`label_col=label_any`. Per-flight files remain native diagnostic artifacts but
are not used in the five-seed statistical summary.

Existing comparison results can be audited and summarized independently of the
latest run manifest:

```powershell
python revision_experiments/summarize_main_comparison.py
python revision_experiments/summarize_main_comparison.py --require-complete
```

The standalone command checks the complete 6 models x 3 datasets x 5 seeds =
90-cell matrix by default. It rejects missing or non-finite Micro metrics and
stale TCNGATRE results whose data protocol, ALFA stride, or GPSData batch size
does not match the current formal profile. Tables are written to
`revision_results/protocol_v1/main_comparison/summary/`.

CATCH and CAROTS are launched through the Python executables pinned in
`envs/runtime_paths.json`.  Their adapters use official model/loss code at the
audited commit, boundary-safe flight windows, train-normal-only scaling, and
the same flightwise causal EMA/SPOT evaluator as TCNGATRE.

Baseline common data is prepared automatically on first launch. The explicit
`prepare-baseline-data` command is available for preflight checks. It always
uses the full protocol graph settings, writes no failure labels, validates all
exported arrays, and reuses the validated export across models and seeds.

The manifests enforce fixed flight lists. ALFA uses 29 normal training flights
(9 legacy normal flights plus 20 `__prefail_normal` segments), 1 normal
validation flight, and 16 failure test flights. MIC graphs use the training
list only. Cached graphs, checkpoints, completed-run markers, and common
baseline exports without the current split fingerprint are treated as stale.

Copy `envs/runtime_paths.example.json` to `envs/runtime_paths.json` and replace
the placeholders with the absolute Python executable paths of the two isolated
environments.  The local runtime-path file is intentionally excluded from Git.

The default full protocol uses data split seed 64 and model seeds 0--4.  Smoke
runs use a deliberately small architecture, one epoch, and a sparse sampling
stride.  Smoke metrics demonstrate executable plumbing only; they are not
paper results.

Official third-party repositories are kept in `_external/` and never imported
into the legacy Python namespace.  Each source is audited and its resolved Git
commit is recorded before an adapter is allowed to run.
