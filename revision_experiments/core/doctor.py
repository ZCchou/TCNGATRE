from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from revision_experiments.analysis.robustness import corrupt_array
from revision_experiments.models.variants import assert_graph_probabilities, build_revision_model
from revision_experiments.scoring.aggregators import AGGREGATORS, aggregate_channels

from .config import make_config
from .integrity import verify_snapshot
from .paths import PACKAGE_ROOT, REPO_ROOT, RESULTS_ROOT, ensure_import_paths
from .provenance import environment_payload, write_json

ensure_import_paths()

from data.stgtcn_window_dataset import resolve_flight_splits  # noqa: E402
from model.tcngatre import STGraphTCN  # noqa: E402


def _python_files() -> list[Path]:
    excluded = {"_external", "dataset", "revision_results"}
    return [
        path for path in REPO_ROOT.rglob("*.py")
        if not any(part in excluded for part in path.relative_to(REPO_ROOT).parts)
    ]


def _alfa4hz_hits() -> list[str]:
    terms = ("alfa4hz", "alfa_4hz")
    hits = []
    for path in _python_files():
        if PACKAGE_ROOT in path.parents:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if any(term in text for term in terms):
            hits.append(str(path.relative_to(REPO_ROOT)))
    return hits


def _synthetic_variant_checks() -> list[dict]:
    variants = [
        "full", "tcn_only", "static_only", "dynamic_only", "late_graph", "single_hop",
        "no_cross_dim", "prior_mic_fixed", "prior_identity_fixed", "prior_random_fixed",
        "fusion_static", "fusion_dynamic", "fusion_learned_scalar", "fusion_sample_gate",
        "fusion_concat_mlp",
    ]
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for variant in variants:
        experiment = "ex02" if variant.startswith(("prior_", "fusion_")) else "ex01"
        cfg = make_config(experiment, "simulate", variant, seed=0, smoke=True)
        legacy_cfg = cfg.to_legacy()
        model = build_revision_model(cfg, legacy_cfg, 7, device)
        model.train()
        x = torch.randn(2, cfg.lookback, 7, 1, device=device)
        a = torch.rand(7, 7, device=device)
        a = 0.5 * (a + a.T)
        a.fill_diagonal_(0.0)
        m = torch.ones_like(a)
        prediction, aux = model(x, a, m, short_patch=cfg.short_patch)
        loss = prediction.square().mean()
        loss.backward()
        gradients = sum(1 for parameter in model.parameters() if parameter.grad is not None)
        if gradients == 0:
            raise AssertionError(f"No gradients for {variant}")
        if variant != "tcn_only":
            assert_graph_probabilities(aux, atol=2e-5)
        results.append({
            "variant": variant,
            "shape": list(prediction.shape),
            "grad_tensors": gradients,
            "status": "passed",
        })
        del model, x, a, m, prediction, aux, loss
    return results


def _full_parity() -> dict:
    cfg = make_config("ex01", "simulate", "full", seed=0, smoke=True)
    legacy_cfg = cfg.to_legacy()
    device = torch.device("cpu")
    original = STGraphTCN(
        num_nodes_hint=7, in_feat=1, d_model=legacy_cfg.d_model,
        short_kernel=legacy_cfg.short_kernel, tcn_layers=legacy_cfg.tcn_layers,
        tcn_blocks=legacy_cfg.tcn_blocks, dropout=legacy_cfg.dropout,
        eta=legacy_cfg.graph_eta, beta=legacy_cfg.graph_beta, out_feat=1,
        horizon=legacy_cfg.horizon_out, graph_gate_init=legacy_cfg.graph_gate_init,
        interleave_every=legacy_cfg.interleave_every, num_hops=legacy_cfg.graph_num_hops,
    ).to(device)
    adapter = build_revision_model(cfg, legacy_cfg, 7, device)
    adapter.load_state_dict(original.state_dict(), strict=True)
    original.eval()
    adapter.eval()
    torch.manual_seed(7)
    x = torch.randn(2, cfg.lookback, 7, 1)
    a = torch.rand(7, 7)
    a = 0.5 * (a + a.T)
    a.fill_diagonal_(0)
    m = torch.ones_like(a)
    with torch.no_grad():
        left, left_aux = original(x, a, m, short_patch=cfg.short_patch)
        right, right_aux = adapter(x, a, m, short_patch=cfg.short_patch)
    max_error = float(torch.max(torch.abs(left - right)))
    aux_errors = {key: float(torch.max(torch.abs(left_aux[key] - right_aux[key]))) for key in ("A_static", "A_dyn", "A_fuse")}
    if max_error > 1e-6 or max(aux_errors.values()) > 1e-6:
        raise AssertionError(f"Full adapter parity failed: {max_error}, {aux_errors}")
    return {"prediction_max_abs_error": max_error, "aux_max_abs_errors": aux_errors, "status": "passed"}


def run_doctor() -> dict:
    integrity = verify_snapshot()
    ast_errors = []
    files = _python_files()
    for path in files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:
            ast_errors.append({"path": str(path), "error": repr(exc)})
    if ast_errors:
        raise RuntimeError(json.dumps(ast_errors, ensure_ascii=False, indent=2))
    alfa_hits = _alfa4hz_hits()
    if alfa_hits:
        raise RuntimeError(f"alfa4hz references found: {alfa_hits}")
    datasets = {}
    for dataset in ("alfa", "gpsdata", "simulate"):
        legacy_cfg = make_config("ex01", dataset, "full", 0, smoke=True).to_legacy()
        train, validation, failure = resolve_flight_splits(dataset_root=Path(legacy_cfg.data_root))
        datasets[dataset] = {"train": len(train), "validation": len(validation), "failure": len(failure)}
        if not train or not validation or not failure:
            raise RuntimeError(f"Incomplete split for {dataset}: {datasets[dataset]}")

    scores = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    aggregation_checks = {method: aggregate_channels(scores, method).tolist() for method in AGGREGATORS}
    if not np.allclose(aggregate_channels(scores, "mean"), scores.mean(axis=1)):
        raise AssertionError("Mean aggregation parity failed")
    corruption_checks = {}
    for kind, level in (("gaussian", 0.05), ("missing", 0.2), ("channel_dropout", 1), ("downsample", 2)):
        corrupted, metadata = corrupt_array(np.ones((2, 8, 4), np.float32), kind, level, 0)
        if corrupted.shape != (2, 8, 4) or not np.isfinite(corrupted).all():
            raise AssertionError(f"Corruption check failed: {kind}")
        corruption_checks[kind] = metadata

    report = {
        "status": "passed",
        "legacy_integrity": integrity,
        "environment": environment_payload(REPO_ROOT),
        "python_files_parsed": len(files),
        "alfa4hz_references": alfa_hits,
        "datasets": datasets,
        "full_adapter_parity": _full_parity(),
        "variant_checks": _synthetic_variant_checks(),
        "aggregation_checks": aggregation_checks,
        "corruption_checks": corruption_checks,
        "disk_free_bytes": shutil.disk_usage(REPO_ROOT).free,
        "pythondontwritebytecode": os.environ.get("PYTHONDONTWRITEBYTECODE", ""),
    }
    output = RESULTS_ROOT / "protocol_v1" / "doctor_report.json"
    write_json(output, report)
    return report
