from __future__ import annotations

import json
from pathlib import Path

from revision_experiments.core.config import load_protocol, make_config, resolve_experiment_selection


def test_ex09_expands_to_two_2025_baselines() -> None:
    protocol = load_protocol()
    assert resolve_experiment_selection(protocol, experiments=["ex09"]) == {
        "ex09": ["gcad", "m2ad"]
    }


def test_ex09_formal_matrix_has_30_unique_run_directories() -> None:
    runs = {
        str(make_config("ex09", dataset, model, seed).run_dir)
        for dataset in ("alfa", "gpsdata", "simulate")
        for model in ("gcad", "m2ad")
        for seed in range(5)
    }
    assert len(runs) == 30


def test_2025_sources_are_commit_pinned() -> None:
    path = Path(__file__).resolve().parents[1] / "baselines" / "baseline_sources.json"
    sources = json.loads(path.read_text(encoding="utf-8"))
    assert sources["gcad"]["commit"] == "e3e0c039468c105edf798747269ba87c309b573f"
    assert sources["m2ad"]["commit"] == "05ac998e55123c51c4a4dd47ad31343bc3c25c23"
    assert sources["gcad"]["year"] == sources["m2ad"]["year"] == 2025
