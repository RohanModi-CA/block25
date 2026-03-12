from __future__ import annotations

import json
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .derived import derive_spacing_dataset
from .io import load_track2_dataset
from .models import DatasetSelection, SignalRecord, Track2Dataset


def load_dataset_selection(path: str | Path) -> OrderedDict[str, DatasetSelection]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f, object_pairs_hook=OrderedDict)

    if not isinstance(cfg, dict) or len(cfg) == 0:
        raise ValueError("Top-level JSON must be a non-empty object keyed by dataset stem")

    validated: OrderedDict[str, DatasetSelection] = OrderedDict()
    required = {"include", "discards", "pair_ids"}

    for dataset_name, entry in cfg.items():
        if not isinstance(dataset_name, str) or not dataset_name:
            raise ValueError("Each top-level JSON key must be a non-empty dataset string")
        if not isinstance(entry, dict):
            raise ValueError(f"Dataset '{dataset_name}' entry must be an object")

        missing = required.difference(entry.keys())
        if missing:
            raise ValueError(
                f"Dataset '{dataset_name}' is missing required key(s): {sorted(missing)}"
            )

        include = entry["include"]
        discards = entry["discards"]
        pair_ids = entry["pair_ids"]

        if not isinstance(include, bool):
            raise ValueError(f"Dataset '{dataset_name}' field 'include' must be boolean")
        if not isinstance(discards, list):
            raise ValueError(f"Dataset '{dataset_name}' field 'discards' must be a list")
        if not isinstance(pair_ids, list):
            raise ValueError(f"Dataset '{dataset_name}' field 'pair_ids' must be a list")

        discards_out: list[int] = []
        for idx in discards:
            if not isinstance(idx, int) or idx < 0:
                raise ValueError(
                    f"Dataset '{dataset_name}' has invalid discard index {idx!r}; expected non-negative int"
                )
            discards_out.append(int(idx))

        pair_ids_out: list[int] = []
        for pair_id in pair_ids:
            if not isinstance(pair_id, int):
                raise ValueError(
                    f"Dataset '{dataset_name}' has invalid pair_id {pair_id!r}; expected int"
                )
            pair_ids_out.append(int(pair_id))

        validated[dataset_name] = DatasetSelection(
            include=include,
            discards=discards_out,
            pair_ids=pair_ids_out,
        )

    return validated


def _build_bond_signal_records_for_dataset(
    dataset_name: str,
    selection: DatasetSelection,
    track2: Track2Dataset,
) -> list[SignalRecord]:
    spacing = derive_spacing_dataset(track2)
    n_pairs = int(spacing.spacing_matrix.shape[1])

    remaining_local_indices = [
        local_idx for local_idx in range(n_pairs) if local_idx not in set(selection.discards)
    ]

    if len(remaining_local_indices) != len(selection.pair_ids):
        raise ValueError(
            f"Dataset '{dataset_name}' has {len(remaining_local_indices)} remaining local bonds after discards "
            f"but {len(selection.pair_ids)} config pair_ids were provided; these lengths must match exactly"
        )

    records: list[SignalRecord] = []
    for local_idx, requested_pair_id in zip(remaining_local_indices, selection.pair_ids):
        label = spacing.pair_labels[local_idx] if local_idx < len(spacing.pair_labels) else "?"
        records.append(
            SignalRecord(
                dataset_name=dataset_name,
                entity_id=int(requested_pair_id),
                local_index=int(local_idx),
                label=str(label).lower(),
                signal_kind="bond",
                source_path=track2.track2_path,
                t=track2.frame_times_s,
                y=np.asarray(spacing.spacing_matrix[:, local_idx], dtype=float),
            )
        )
    return records


def _build_site_signal_records_for_dataset(
    dataset_name: str,
    selection: DatasetSelection,
    track2: Track2Dataset,
) -> list[SignalRecord]:
    x_positions = np.asarray(track2.x_positions, dtype=float)
    n_frames, n_blocks = x_positions.shape
    n_pairs = max(0, n_blocks - 1)

    remaining_local_bonds = [i for i in range(n_pairs) if i not in set(selection.discards)]
    if len(remaining_local_bonds) != len(selection.pair_ids):
        raise ValueError(
            f"Dataset '{dataset_name}' has {len(remaining_local_bonds)} remaining local bonds after discards "
            f"but {len(selection.pair_ids)} config pair_ids were provided; these lengths must match exactly"
        )

    site_mapping: dict[int, int] = {}
    for local_bond_idx, global_bond_id in zip(remaining_local_bonds, selection.pair_ids):
        left_site = int(global_bond_id)
        right_site = int(global_bond_id + 1)

        if local_bond_idx in site_mapping and site_mapping[local_bond_idx] != left_site:
            raise ValueError(
                f"Dataset '{dataset_name}' has conflicting site mapping for local block {local_bond_idx}"
            )
        if (local_bond_idx + 1) in site_mapping and site_mapping[local_bond_idx + 1] != right_site:
            raise ValueError(
                f"Dataset '{dataset_name}' has conflicting site mapping for local block {local_bond_idx + 1}"
            )

        site_mapping[local_bond_idx] = left_site
        site_mapping[local_bond_idx + 1] = right_site

    records: list[SignalRecord] = []
    for local_block_idx, site_id in sorted(site_mapping.items()):
        label = track2.block_colors[local_block_idx] if local_block_idx < len(track2.block_colors) else "?"
        records.append(
            SignalRecord(
                dataset_name=dataset_name,
                entity_id=int(site_id),
                local_index=int(local_block_idx),
                label=str(label).lower(),
                signal_kind="site",
                source_path=track2.track2_path,
                t=track2.frame_times_s,
                y=np.asarray(x_positions[:, local_block_idx], dtype=float),
            )
        )
    return records


def build_configured_bond_signals(
    config: OrderedDict[str, DatasetSelection],
    *,
    track_data_root: str | None = None,
    allow_duplicate_ids: bool = False,
) -> list[SignalRecord]:
    records: list[SignalRecord] = []
    seen_ids: set[int] = set()

    for dataset_name, selection in config.items():
        if not selection.include:
            continue

        track2 = load_track2_dataset(dataset=dataset_name, track_data_root=track_data_root)
        dataset_records = _build_bond_signal_records_for_dataset(dataset_name, selection, track2)

        for record in dataset_records:
            if (not allow_duplicate_ids) and (record.entity_id in seen_ids):
                warnings.warn(
                    f"Skipping duplicate bond id {record.entity_id} from dataset '{dataset_name}' "
                    "because an earlier occurrence was already accepted"
                )
                continue
            records.append(record)
            seen_ids.add(record.entity_id)

    return records


def build_configured_site_signals(
    config: OrderedDict[str, DatasetSelection],
    *,
    track_data_root: str | None = None,
    allow_duplicate_ids: bool = False,
) -> list[SignalRecord]:
    records: list[SignalRecord] = []
    seen_ids: set[int] = set()

    for dataset_name, selection in config.items():
        if not selection.include:
            continue

        track2 = load_track2_dataset(dataset=dataset_name, track_data_root=track_data_root)
        dataset_records = _build_site_signal_records_for_dataset(dataset_name, selection, track2)

        for record in dataset_records:
            if (not allow_duplicate_ids) and (record.entity_id in seen_ids):
                warnings.warn(
                    f"Skipping duplicate site id {record.entity_id} from dataset '{dataset_name}' "
                    "because an earlier occurrence was already accepted"
                )
                continue
            records.append(record)
            seen_ids.add(record.entity_id)

    return records
