from __future__ import annotations

from typing import Mapping, Sequence


def resolve_rate_groups(
    sensor_names: Sequence[str],
    rate_groups: Mapping[str, Sequence[str]] | None,
) -> tuple[list[str], list[list[int]]]:
    names = [str(x) for x in sensor_names]
    if rate_groups is None or len(rate_groups) <= 0:
        return ["all_rate"], [list(range(len(names)))]

    remaining = set(range(len(names)))
    group_names: list[str] = []
    group_indices: list[list[int]] = []

    for group_name, patterns in rate_groups.items():
        group_name = str(group_name)
        pats = [str(x).strip().lower() for x in patterns if str(x).strip() != ""]
        matched: list[int] = []
        for idx, name in enumerate(names):
            if idx not in remaining:
                continue
            lname = name.lower()
            if any(pat in lname for pat in pats):
                matched.append(idx)
        group_names.append(group_name)
        group_indices.append(matched)
        for idx in matched:
            remaining.discard(idx)

    if remaining:
        group_names.append("other_rate")
        group_indices.append(sorted(remaining))

    if len(group_names) <= 0:
        return ["all_rate"], [list(range(len(names)))]
    return group_names, group_indices


def build_group_tensors(
    value_seq,
    mask_seq,
    group_indices: Sequence[Sequence[int]],
):
    import numpy as np

    value_arr = np.asarray(value_seq, dtype=np.float32)
    mask_arr = np.asarray(mask_seq, dtype=np.float32)
    if value_arr.ndim != 2 or mask_arr.shape != value_arr.shape:
        raise ValueError(
            f"group tensors expect [T, D] arrays with same shape, got {value_arr.shape} vs {mask_arr.shape}"
        )
    history_steps, _ = value_arr.shape
    num_groups = max(len(group_indices), 1)
    max_group_dim = max(1, max((len(idx) for idx in group_indices), default=0))

    group_value = np.zeros((history_steps, num_groups, max_group_dim), dtype=np.float32)
    group_mask = np.zeros((history_steps, num_groups, max_group_dim), dtype=np.float32)
    group_available = np.zeros((history_steps, num_groups), dtype=np.float32)

    for g_idx, sensor_idx in enumerate(group_indices):
        if len(sensor_idx) <= 0:
            continue
        sensor_idx = list(sensor_idx)
        width = len(sensor_idx)
        group_value[:, g_idx, :width] = value_arr[:, sensor_idx]
        group_mask[:, g_idx, :width] = mask_arr[:, sensor_idx]
        group_available[:, g_idx] = (group_mask[:, g_idx, :width].sum(axis=-1) > 0.0).astype(np.float32)

    return group_value, group_mask, group_available
