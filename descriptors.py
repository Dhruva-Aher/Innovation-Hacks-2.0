from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


@dataclass(frozen=True)
class DescriptorResult:
    smiles: str
    valid: bool
    descriptors: Dict[str, float]


class MolecularDescriptorEngine:
    """Calculate RDKit descriptors and a few textile-relevant derived features."""

    def __init__(self) -> None:
        self.descriptor_functions = {
            # Core molecular properties
            "molecular_weight": Descriptors.MolWt,
            "exact_molecular_weight": Descriptors.ExactMolWt,
            "heavy_atom_count": Descriptors.HeavyAtomCount,
            "num_heteroatoms": Descriptors.NumHeteroatoms,
            # Structural descriptors
            "ring_count": Descriptors.RingCount,
            "aromatic_rings": Descriptors.NumAromaticRings,
            "rotatable_bonds": Descriptors.NumRotatableBonds,
            "saturated_rings": Descriptors.NumSaturatedRings,
            "aliphatic_rings": Descriptors.NumAliphaticRings,
            # Chemical properties
            "logp": Descriptors.MolLogP,
            "tpsa": Descriptors.TPSA,
            "molar_refractivity": Descriptors.MolMR,
            # Hydrogen bonding
            "h_bond_donors": Descriptors.NumHDonors,
            "h_bond_acceptors": Descriptors.NumHAcceptors,
            # Complexity / topology
            "bertz_complexity": Descriptors.BertzCT,
            "balaban_j": Descriptors.BalabanJ,
            "avg_ipc": Descriptors.AvgIpc,
            "fraction_csp3": Descriptors.FractionCSP3,
            "chi0": Descriptors.Chi0,
            "chi1": Descriptors.Chi1,
            "chi0v": Descriptors.Chi0v,
            "chi1v": Descriptors.Chi1v,
            "kappa1": Descriptors.Kappa1,
            "kappa2": Descriptors.Kappa2,
            "kappa3": Descriptors.Kappa3,
        }

    def smiles_to_mol(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return mol

    def calculate_descriptors(self, smiles: str) -> Dict[str, float]:
        mol = self.smiles_to_mol(smiles)
        descriptors: Dict[str, float] = {}

        for name, func in self.descriptor_functions.items():
            try:
                descriptors[name] = float(func(mol))
            except Exception:
                descriptors[name] = 0.0

        descriptors.update(self._calculate_textile_descriptors(mol))
        return descriptors

    def _calculate_textile_descriptors(self, mol) -> Dict[str, float]:
        heavy_atoms = max(1, Descriptors.HeavyAtomCount(mol))
        bonds = max(1, mol.GetNumBonds())
        mw = max(1e-8, Descriptors.MolWt(mol))
        mr = max(1e-8, Descriptors.MolMR(mol))
        rings = max(0, Descriptors.RingCount(mol))

        chain_flexibility = Descriptors.NumRotatableBonds(mol) / bonds
        polarity_ratio = Descriptors.TPSA(mol) / mw
        crystallinity_index = Descriptors.NumAromaticRings(mol) / max(1, rings) if rings else 0.0
        hydrogen_bonding_density = (
            Descriptors.NumHDonors(mol) + Descriptors.NumHAcceptors(mol)
        ) / heavy_atoms
        backbone_rigidity = (
            Descriptors.NumAromaticRings(mol) + Descriptors.NumSaturatedRings(mol)
        ) / heavy_atoms
        structural_complexity = Descriptors.BertzCT(mol) / heavy_atoms
        surface_area_ratio = Descriptors.TPSA(mol) / mr

        return {
            "chain_flexibility": float(chain_flexibility),
            "polarity_ratio": float(polarity_ratio),
            "crystallinity_index": float(crystallinity_index),
            "hydrogen_bonding_density": float(hydrogen_bonding_density),
            "backbone_rigidity": float(backbone_rigidity),
            "structural_complexity": float(structural_complexity),
            "surface_area_ratio": float(surface_area_ratio),
        }

    def get_feature_names(self) -> List[str]:
        sample = self.calculate_descriptors("CCO")
        return list(sample.keys())

    def get_feature_vector(self, smiles: str) -> np.ndarray:
        return np.array(list(self.calculate_descriptors(smiles).values()), dtype=float)

    def descriptors_dataframe(self, smiles_list: List[str]) -> pd.DataFrame:
        rows = []
        for smiles in smiles_list:
            try:
                row = self.calculate_descriptors(smiles)
                row["smiles"] = smiles
                row["is_valid"] = True
            except Exception:
                row = {name: np.nan for name in self.get_feature_names()}
                row["smiles"] = smiles
                row["is_valid"] = False
            rows.append(row)
        return pd.DataFrame(rows)
