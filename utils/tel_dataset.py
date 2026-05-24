"""
utils/tel_dataset.py
─────────────────────
PyTorch Dataset backed by Apache Arrow (pyarrow), reading the Silver
Parquet written by 08_tel_preprocess.py.

No Spark, no extra deps beyond pyarrow (ships with Databricks Runtime).

Public API
──────────
  F1TelemetryDataset(silver_path, seasons, session_types=None, labelled_only=False)
    __len__()                  → number of laps
    __getitem__(idx)           → (x: FloatTensor (6, 1024), y: int label or -1)
    label_distribution()       → {race_position: count} dict (labelled rows only)
"""

from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from typing import Optional

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

N_CHANNELS = 6
SEQ_LEN    = 1024


class F1TelemetryDataset(Dataset):
    """
    Reads the Silver Parquet (written by 08_tel_preprocess.py) into memory via
    PyArrow and serves (telemetry_tensor, label) pairs to a PyTorch DataLoader.

    Parameters
    ----------
    silver_path : str
        Root directory of the Silver Parquet (e.g. ``/Volumes/.../telemetry/clean/silver``).
        PyArrow discovers all ``season=*/`` partition sub-directories automatically.
    seasons : list[int]
        Which season partitions to load (e.g. ``[2024, 2025]``).
        Passed as a partition filter — avoids loading unneeded files.
    session_types : list[str] | None
        If provided, only laps from these session types are kept
        (e.g. ``["R", "Q"]`` for fine-tuning; ``None`` for pre-training).
    labelled_only : bool
        If True, rows where ``label_position`` is null are dropped.
        Set True for fine-tuning (needs race position labels),
        False for pre-training (uses all laps).
    """

    def __init__(
        self,
        silver_path:   str,
        seasons:       list[int],
        session_types: Optional[list[str]] = None,
        labelled_only: bool = False,
    ):
        self.silver_path   = silver_path
        self.seasons       = seasons
        self.session_types = session_types
        self.labelled_only = labelled_only

        # ── Build partition filter ────────────────────────────────────────────
        # PyArrow partition filters use list-of-tuples OR-logic between entries
        # within the same column, and AND-logic across columns.
        filters: list[tuple] = []

        # Filter to requested seasons
        if len(seasons) == 1:
            filters.append(("season", "=", seasons[0]))
        else:
            # pq.read_table accepts list-of-lists for OR conditions per column
            # We'll apply season filter post-load below to keep things simple
            pass

        # ── Load table ────────────────────────────────────────────────────────
        dataset = pq.ParquetDataset(
            silver_path,
            filters=filters if filters else None,
        )
        table = dataset.read(
            columns=["driver", "season", "event", "session_type",
                     "lap_number", "channels", "label_position"]
        )

        # ── Apply season filter (multi-season case) ───────────────────────────
        if len(seasons) > 1:
            import pyarrow.compute as pc
            season_col  = table.column("season")
            season_mask = pc.is_in(season_col, value_set=pa_array(seasons))
            table = table.filter(season_mask)

        # ── Apply session_type filter ─────────────────────────────────────────
        if session_types is not None:
            import pyarrow as pa
            import pyarrow.compute as pc
            stype_col  = table.column("session_type")
            stype_mask = pc.is_in(stype_col,
                                  value_set=pa.array(session_types, type=pa.string()))
            table = table.filter(stype_mask)

        # ── Apply labelled_only filter ────────────────────────────────────────
        if labelled_only:
            import pyarrow.compute as pc
            label_col  = table.column("label_position")
            valid_mask = pc.is_valid(label_col)
            table = table.filter(valid_mask)

        self._table = table
        self._len   = len(table)

        print(
            f"F1TelemetryDataset: {self._len:,} laps loaded "
            f"| seasons={seasons} | sessions={session_types} | labelled_only={labelled_only}"
        )

    # ── PyTorch Dataset interface ─────────────────────────────────────────────

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Returns
        -------
        x : FloatTensor of shape (N_CHANNELS, SEQ_LEN) = (6, 1024)
            Normalised telemetry tensor ready for the MAE encoder.
        y : int
            Race position label (1–20), or -1 if not labelled.
            Pre-training callers should ignore y.
        """
        row = self._table.slice(idx, 1)

        # channels is stored as a list-of-lists: [[ch0_val0, ...], [ch1_val0, ...], ...]
        # PyArrow returns it as a nested ListArray; convert via numpy.
        channels_raw = row.column("channels")[0].as_py()   # Python list of 6 lists
        x = np.array(channels_raw, dtype=np.float32)       # (6, 1024)

        if x.shape != (N_CHANNELS, SEQ_LEN):
            # Guard against malformed rows — replace with zeros so the batch
            # doesn't crash; the zero tensor will produce a finite but high loss
            # that the model learns to avoid (rare in practice).
            x = np.zeros((N_CHANNELS, SEQ_LEN), dtype=np.float32)

        x_tensor = torch.from_numpy(x)   # (6, 1024), float32

        label_raw = row.column("label_position")[0].as_py()
        y = int(label_raw) - 1 if label_raw is not None else -1
        # Convert 1-indexed race position to 0-indexed class for CrossEntropyLoss.
        # -1 is the ignore_index sentinel; set ignore_index=-1 in your loss function.

        return x_tensor, y

    # ── Utility methods ───────────────────────────────────────────────────────

    def label_distribution(self) -> dict[int, int]:
        """
        Return {race_position (1-indexed): lap_count} for labelled rows.
        Used by 10_mae_finetune.py to compute inverse-frequency class weights.

        Returns an empty dict if labelled_only=False and no labels are present.
        """
        import pyarrow.compute as pc

        label_col = self._table.column("label_position")
        # Drop nulls
        valid = self._table.filter(pc.is_valid(label_col))
        if len(valid) == 0:
            return {}

        labels = valid.column("label_position").to_pylist()
        dist: dict[int, int] = {}
        for pos in labels:
            if pos is not None:
                dist[int(pos)] = dist.get(int(pos), 0) + 1
        return dist


# ── PRIVATE HELPER ────────────────────────────────────────────────────────────

def pa_array(values: list[int]):
    """Thin wrapper so the multi-season branch doesn't need an extra import."""
    import pyarrow as pa
    return pa.array(values, type=pa.int32())
