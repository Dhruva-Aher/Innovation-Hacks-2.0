from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from app.ml_system import FabricMLPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fabric property ML model")
    parser.add_argument("--csv", type=str, default=None, help="Path to a labeled CSV with smiles and target columns")
    parser.add_argument("--demo-samples", type=int, default=600, help="Number of demo rows if no CSV is supplied")
    parser.add_argument("--artifacts", type=str, default="artifacts", help="Directory to save trained model files")
    args = parser.parse_args()

    predictor = FabricMLPredictor()

    if args.csv:
        frame = pd.read_csv(args.csv)
        report = predictor.fit(frame)
    else:
        report = predictor.fit_demo(n_samples=args.demo_samples)

    predictor.save(args.artifacts)
    print("Training complete")
    print(report)
    for target, metric in report.metrics.items():
        print(f"{target}: MAE={metric['mae']:.3f}, R2={metric['r2']:.3f}")


if __name__ == "__main__":
    main()
