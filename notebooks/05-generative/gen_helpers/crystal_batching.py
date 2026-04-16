"""Batching helpers for the crystal diffusion notebook."""

from __future__ import annotations

import numpy as np
import torch


CRYSTAL_BATCH_TENSOR_KEYS = (
    "frac0",
    "atom_tokens0",
    "lattice0",
    "continuous_conditions",
    "composition_conditions",
    "spacegroup_conditions",
    "num_atoms",
    "batch_idx",
)


def prepare_crystal_item(record):
    return {
        "name": record["name"],
        "formula": record["formula"],
        "num_atoms": record["num_atoms"],
        "frac0": record["frac0"].copy(),
        "atom_tokens0": record["atom_tokens0"].copy(),
        "lattice0": record["lattice0_norm"].copy(),
        "continuous_conditions": record["conditions_norm"].copy(),
        "composition_conditions": record["composition_cond"].copy(),
        "spacegroup_conditions": int(record["spacegroup_number"]),
    }


def collate_crystals(records):
    items = [prepare_crystal_item(record) for record in records]
    num_atoms = torch.tensor([x["num_atoms"] for x in items], dtype=torch.long)
    frac0 = torch.tensor(np.concatenate([x["frac0"] for x in items], axis=0), dtype=torch.float32)
    atom_tokens0 = torch.tensor(np.concatenate([x["atom_tokens0"] for x in items], axis=0), dtype=torch.long)
    lattice0 = torch.tensor(np.stack([x["lattice0"] for x in items], axis=0), dtype=torch.float32)
    continuous_conditions = torch.tensor(
        np.stack([x["continuous_conditions"] for x in items], axis=0),
        dtype=torch.float32,
    )
    composition_conditions = torch.tensor(
        np.stack([x["composition_conditions"] for x in items], axis=0),
        dtype=torch.float32,
    )
    spacegroup_conditions = torch.tensor(
        [x["spacegroup_conditions"] for x in items],
        dtype=torch.long,
    )
    batch_idx = torch.repeat_interleave(torch.arange(len(items), dtype=torch.long), num_atoms)
    return {
        "frac0": frac0,
        "atom_tokens0": atom_tokens0,
        "lattice0": lattice0,
        "continuous_conditions": continuous_conditions,
        "composition_conditions": composition_conditions,
        "spacegroup_conditions": spacegroup_conditions,
        "num_atoms": num_atoms,
        "batch_idx": batch_idx,
        "names": [x["name"] for x in items],
        "formulas": [x["formula"] for x in items],
    }


def move_crystal_batch_to_device(batch, device):
    moved = {key: batch[key].to(device) for key in CRYSTAL_BATCH_TENSOR_KEYS}
    moved["names"] = batch["names"]
    moved["formulas"] = batch["formulas"]
    return moved


__all__ = [
    "CRYSTAL_BATCH_TENSOR_KEYS",
    "collate_crystals",
    "move_crystal_batch_to_device",
    "prepare_crystal_item",
]
