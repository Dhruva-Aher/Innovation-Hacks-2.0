from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


TARGET_COLUMNS = [
    "strength",
    "comfort",
    "sustainability",
    "breathability",
    "durability",
    "cost",
]


@dataclass(frozen=True)
class DatasetBundle:
    frame: pd.DataFrame
    source: str


def scaffold_from_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return scaffold if scaffold else smiles


def load_csv_dataset(path: str | Path) -> DatasetBundle:
    frame = pd.read_csv(path)
    if "smiles" not in frame.columns:
        raise ValueError("CSV must contain a 'smiles' column")
    missing = [c for c in TARGET_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"CSV must contain target columns: {missing}")
    return DatasetBundle(frame=frame.copy(), source=str(path))


def _base_demo_rows():
    # Wider starter pool for better synthetic coverage in demo mode.
    return [
        "CCCCCCCCCC",
        "COC(=O)c1ccc(cc1)C(=O)OCC",
        "NCCCCCC(=O)",
        "OC[C@H]1O[C@H](O)[C@@H](O)[C@H](O)[C@@H]1O",
        "CC(C)C[C@H](N)C(=O)N[C@@H](CS)C(=O)O",
        "CC[C@H](C)[C@H](N)C(=O)N[C@@H](C)C(=O)O",
        "CC(C#N)C",
        "CC(C)CCC(C)C",
        "CCOCCOCCOC",
        "c1ccc(cc1)CC",
        "CC(=O)O",
        "CCO",
        "CC(C)O",
        "CC(C)(C)CO",
        "OCCO",
        "CCN(CC)CC",
        "CCOC(=O)C",
        "CC(C)C(=O)O",
        "c1ccncc1",
        "c1ccccc1O",
        "CCCCO",
        "CCSCC",
        "CNC(=O)O",
    ]


def make_demo_dataset(n_samples: int = 600, random_state: int = 42) -> DatasetBundle:
    from .descriptors import MolecularDescriptorEngine

    rng = np.random.default_rng(random_state)
    engine = MolecularDescriptorEngine()
    smiles_pool = _base_demo_rows()
    rows = []

    for _ in range(n_samples):
        smiles = rng.choice(smiles_pool)
        desc = engine.calculate_descriptors(smiles)

        strength = 2.0 + 2.4 * desc["backbone_rigidity"] + 1.8 * desc["crystallinity_index"] + 0.6 * np.log1p(desc["molecular_weight"])
        comfort = 7.5 - 2.0 * desc["backbone_rigidity"] + 2.2 * desc["hydrogen_bonding_density"] + 2.0 * desc["chain_flexibility"] + 2.0 * desc["polarity_ratio"]
        sustainability = 6.5 + 2.0 * desc["hydrogen_bonding_density"] + 1.2 * (1.0 / (1.0 + desc["structural_complexity"])) + 1.2 * desc["polarity_ratio"]
        breathability = 5.0 + 2.5 * desc["chain_flexibility"] + 3.5 * desc["polarity_ratio"] + 1.5 * desc["surface_area_ratio"]
        durability = 3.5 + 3.2 * desc["backbone_rigidity"] + 1.8 * desc["crystallinity_index"] + 1.0 * (desc["structural_complexity"] / 25.0)
        cost = 1.5 + 1.4 * desc["structural_complexity"] / 25.0 + 0.5 * desc["aromatic_rings"] + 0.3 * np.log1p(desc["molecular_weight"])

        targets = np.array([strength, comfort, sustainability, breathability, durability, cost], dtype=float)
        targets += rng.normal(0, 0.25, size=targets.shape)
        targets = np.clip(targets, 0.5, 10.0)

        rows.append({"smiles": smiles, **{k: float(v) for k, v in zip(TARGET_COLUMNS, targets)}})

    frame = pd.DataFrame(rows)
    frame["scaffold"] = frame["smiles"].map(scaffold_from_smiles)
    return DatasetBundle(frame=frame, source="demo")


def attach_scaffolds(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["scaffold"] = out["smiles"].map(scaffold_from_smiles)
    return out
