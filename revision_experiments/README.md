# TCNGATRE revision experiments

This directory is an add-only experiment harness for the KNOSYS revision.  It
imports the legacy TCNGATRE data/model helpers read-only, writes all generated
files under `revision_results/`, and checks a SHA-256 snapshot of every tracked
legacy file before and after a run.

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

CATCH and CAROTS are launched through the Python executables pinned in
`envs/runtime_paths.json`.  Their adapters use official model/loss code at the
audited commit, boundary-safe flight windows, train-normal-only scaling, and
the same flightwise causal EMA/SPOT evaluator as TCNGATRE.

Baseline common data is prepared automatically on first launch. The explicit
`prepare-baseline-data` command is available for preflight checks. It always
uses the full protocol graph settings, writes no failure labels, validates all
exported arrays, and reuses the validated export across models and seeds.

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
