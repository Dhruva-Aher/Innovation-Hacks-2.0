from __future__ import annotations

import argparse

from app.ml_system import FabricMLPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict fabric properties from a SMILES string")
    parser.add_argument("smiles", type=str, help="Input SMILES")
    parser.add_argument("--artifacts", type=str, default="artifacts", help="Directory containing trained model files")
    args = parser.parse_args()

    predictor = FabricMLPredictor.load(args.artifacts)
    output = predictor.predict_with_uncertainty(args.smiles)
    meta = output.pop("__meta__", None)
    for prop, values in output.items():
        print(f"{prop}: {values['prediction']:.2f} +/- {values['uncertainty']:.2f}")
    if meta:
        print(f"novelty_score: {meta.get('novelty_score', 0.0):.2f}")


if __name__ == "__main__":
    main()
