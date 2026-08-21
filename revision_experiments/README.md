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
python revision_experiments/run_revision.py smoke --datasets all
python revision_experiments/run_revision.py run --experiments ex01,ex02 --datasets all --seeds 0,1,2,3,4 --dry-run
python revision_experiments/run_revision.py summarize --protocol protocol_v1
```

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
GPSData therefore use a targeted physical batch size of 16 to avoid quadratic
graph-memory OOMs; every other model/dataset keeps its native batch size. The
override is recorded in each stage signature and provenance file and can be
changed explicitly with `--tcngatre-gps-batch-size`.

The primary comparison collector uses only the Micro result selected from the
global `summary_metrics.csv` row with `threshold_method=spot` and
`label_col=label_any`. Per-flight files remain native diagnostic artifacts but
are not used in the five-seed statistical summary.

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
