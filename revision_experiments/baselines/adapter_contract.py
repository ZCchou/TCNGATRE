from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaselineAdapter(ABC):
    """No baseline can join the comparison without satisfying this contract."""

    @abstractmethod
    def prepare(self, dataset: str, output_dir: Path) -> dict:
        """Export train-normal, validation-normal and failure data without label leakage."""

    @abstractmethod
    def fit(self, seed: int, output_dir: Path) -> Path:
        """Return the checkpoint path from normal-only training."""

    @abstractmethod
    def score(self, checkpoint: Path, output_dir: Path) -> Path:
        """Return a CSV containing flight/t_start/t_end/raw_total_score."""

    @abstractmethod
    def provenance(self) -> dict:
        """Return source URL, pinned commit, environment, and effective hyperparameters."""
