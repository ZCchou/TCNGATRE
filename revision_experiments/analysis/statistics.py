from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd


def hierarchical_paired_bootstrap(
    pairs: pd.DataFrame,
    value_a: str,
    value_b: str,
    flight_col: str = "flight",
    seed_col: str = "seed",
    n_resamples: int = 10000,
    random_seed: int = 20260821,
) -> dict:
    required = {flight_col, seed_col, value_a, value_b}
    if not required.issubset(pairs.columns):
        raise ValueError(f"Missing columns: {sorted(required.difference(pairs.columns))}")
    clean = pairs.dropna(subset=list(required)).copy()
    flights = clean[flight_col].drop_duplicates().to_numpy()
    if len(flights) == 0:
        raise ValueError("No paired observations")
    rng = np.random.default_rng(int(random_seed))
    estimates = np.empty(int(n_resamples), dtype=np.float64)
    for idx in range(int(n_resamples)):
        sampled_flights = rng.choice(flights, size=len(flights), replace=True)
        differences: list[float] = []
        for flight in sampled_flights:
            group = clean.loc[clean[flight_col] == flight]
            seeds = group[seed_col].drop_duplicates().to_numpy()
            sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
            for seed in sampled_seeds:
                row = group.loc[group[seed_col] == seed].iloc[0]
                differences.append(float(row[value_a]) - float(row[value_b]))
        estimates[idx] = float(np.mean(differences))
    observed = float(np.mean(clean[value_a] - clean[value_b]))
    return {
        "mean_difference": observed,
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "n_resamples": int(n_resamples),
        "n_flights": int(len(flights)),
    }


def paired_sign_permutation(differences: np.ndarray, n_resamples: int = 10000, seed: int = 20260821) -> dict:
    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    values = values[np.abs(values) > 0]
    if values.size == 0:
        return {"p_value": 1.0, "observed": 0.0, "exact": True, "n": 0}
    observed = float(abs(values.mean()))
    if values.size <= 16:
        permutations = itertools.product((-1.0, 1.0), repeat=values.size)
        stats = np.fromiter((abs(np.mean(values * np.asarray(signs))) for signs in permutations), float)
        exact = True
    else:
        rng = np.random.default_rng(int(seed))
        signs = rng.choice((-1.0, 1.0), size=(int(n_resamples), values.size))
        stats = np.abs(np.mean(signs * values[None, :], axis=1))
        exact = False
    p_value = float((np.sum(stats >= observed - 1e-15) + (0 if exact else 1)) / (len(stats) + (0 if exact else 1)))
    return {"p_value": p_value, "observed": observed, "exact": exact, "n": int(values.size)}


def rank_biserial(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values) & (values != 0)]
    if values.size == 0:
        return 0.0
    ranks = pd.Series(np.abs(values)).rank(method="average").to_numpy(dtype=np.float64)
    positive = float(ranks[values > 0].sum())
    negative = float(ranks[values < 0].sum())
    denominator = positive + negative
    return 0.0 if denominator <= 0 else (positive - negative) / denominator


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    total = len(p)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * float(p[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def seed_summary(frame: pd.DataFrame, group_cols: list[str], metric: str) -> pd.DataFrame:
    return frame.groupby(group_cols, dropna=False)[metric].agg(["mean", "std", "count"]).reset_index()
