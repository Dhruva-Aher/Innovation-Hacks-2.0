from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from .data import TARGET_COLUMNS, attach_scaffolds, make_demo_dataset
from .descriptors import MolecularDescriptorEngine


@dataclass
class FitReport:
    rows: int
    features: int
    metrics: Dict[str, Dict[str, float]]
    source: str


class FabricMLPredictor:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.engine = MolecularDescriptorEngine()
        self.models: Dict[str, RandomForestRegressor] = {}
        self.feature_names: List[str] = self.engine.get_feature_names()
        self.training_columns: List[str] = TARGET_COLUMNS.copy()
        self.is_fitted = False
        self.report: FitReport | None = None
        self.feature_means: np.ndarray | None = None
        self.feature_stds: np.ndarray | None = None

    def _feature_matrix(self, smiles_list: List[str]) -> np.ndarray:
        vectors = [self.engine.get_feature_vector(smiles) for smiles in smiles_list]
        return np.asarray(vectors, dtype=float)

    def _prepare_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = ["smiles", *TARGET_COLUMNS]
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        out = attach_scaffolds(frame)
        out = out.dropna(subset=["smiles", *TARGET_COLUMNS]).reset_index(drop=True)
        return out

    def fit(self, frame: pd.DataFrame) -> FitReport:
        df = self._prepare_frame(frame)
        X = self._feature_matrix(df["smiles"].tolist())
        y = df[TARGET_COLUMNS].astype(float).values
        groups = df["scaffold"].tolist()

        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=self.random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        self.models = {}
        metrics: Dict[str, Dict[str, float]] = {}
        self.feature_means = X_train.mean(axis=0)
        self.feature_stds = X_train.std(axis=0)
        self.feature_stds = np.where(self.feature_stds == 0, 1.0, self.feature_stds)

        for col_idx, target in enumerate(TARGET_COLUMNS):
            model = RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=1,
                random_state=self.random_state,
                n_jobs=-1,
            )
            model.fit(X_train, y_train[:, col_idx])
            pred = model.predict(X_test)
            metrics[target] = {
                "mae": float(mean_absolute_error(y_test[:, col_idx], pred)),
                "r2": float(r2_score(y_test[:, col_idx], pred)),
            }
            self.models[target] = model

        self.is_fitted = True
        self.report = FitReport(
            rows=int(len(df)),
            features=int(X.shape[1]),
            metrics=metrics,
            source="frame",
        )
        return self.report

    def fit_demo(self, n_samples: int = 600) -> FitReport:
        bundle = make_demo_dataset(n_samples=n_samples, random_state=self.random_state)
        report = self.fit(bundle.frame)
        self.report = FitReport(rows=report.rows, features=report.features, metrics=report.metrics, source=bundle.source)
        return self.report

    def predict(self, smiles: str) -> Dict[str, float]:
        if not self.is_fitted:
            raise RuntimeError("Model is not trained. Call fit() or load().")

        vector = self.engine.get_feature_vector(smiles).reshape(1, -1)
        predictions = {}
        for target, model in self.models.items():
            value = float(model.predict(vector)[0])
            predictions[target] = float(np.clip(value, 0.5, 10.0))
        return predictions

    def predict_with_uncertainty(self, smiles: str) -> Dict[str, Dict[str, float]]:
        if not self.is_fitted:
            raise RuntimeError("Model is not trained. Call fit() or load().")

        vector = self.engine.get_feature_vector(smiles).reshape(1, -1)
        result: Dict[str, Dict[str, float]] = {}
        novelty = self.novelty_score(smiles)

        for target, model in self.models.items():
            tree_preds = np.array([tree.predict(vector)[0] for tree in model.estimators_], dtype=float)
            mean = float(np.clip(tree_preds.mean(), 0.5, 10.0))
            std = float(tree_preds.std(ddof=1) if len(tree_preds) > 1 else 0.0)
            # Slightly inflate uncertainty for out-of-domain molecules.
            adjusted_uncertainty = float(std * (1.0 + 0.75 * novelty))
            result[target] = {"prediction": mean, "uncertainty": adjusted_uncertainty}
        result["__meta__"] = {"novelty_score": float(novelty)}
        return result

    def evaluate_frame(self, frame: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        if not self.is_fitted:
            raise RuntimeError("Model is not trained. Call fit() first.")
        df = self._prepare_frame(frame)
        X = self._feature_matrix(df["smiles"].tolist())
        y = df[TARGET_COLUMNS].astype(float).values
        metrics: Dict[str, Dict[str, float]] = {}
        for col_idx, target in enumerate(TARGET_COLUMNS):
            pred = self.models[target].predict(X)
            metrics[target] = {
                "mae": float(mean_absolute_error(y[:, col_idx], pred)),
                "r2": float(r2_score(y[:, col_idx], pred)),
            }
        return metrics

    def get_feature_importance(self, target: str, top_n: int = 10) -> List[Tuple[str, float]]:
        if target not in self.models:
            raise ValueError(f"Unknown target: {target}")
        model = self.models[target]
        if not hasattr(model, "feature_importances_"):
            return []
        pairs = list(zip(self.feature_names, model.feature_importances_.tolist()))
        pairs.sort(key=lambda item: item[1], reverse=True)
        return pairs[:top_n]

    def novelty_score(self, smiles: str) -> float:
        if not self.is_fitted or self.feature_means is None or self.feature_stds is None:
            return 0.0
        vector = self.engine.get_feature_vector(smiles)
        z = np.abs((vector - self.feature_means) / self.feature_stds)
        # Squash to [0, 1.5] roughly; higher means more out-of-domain.
        return float(np.clip(np.nanmean(z) / 3.0, 0.0, 1.5))

    def save(self, directory: str | Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("Nothing to save; fit the model first.")
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models, path / "models.joblib")
        joblib.dump(self.feature_names, path / "feature_names.joblib")
        meta = {
            "random_state": self.random_state,
            "targets": TARGET_COLUMNS,
            "report": None if self.report is None else self.report.__dict__,
            "feature_means": None if self.feature_means is None else self.feature_means.tolist(),
            "feature_stds": None if self.feature_stds is None else self.feature_stds.tolist(),
        }
        (path / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "FabricMLPredictor":
        path = Path(directory)
        predictor = cls()
        predictor.models = joblib.load(path / "models.joblib")
        predictor.feature_names = joblib.load(path / "feature_names.joblib")
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        predictor.random_state = int(meta.get("random_state", 42))
        predictor.is_fitted = True
        if meta.get("report"):
            predictor.report = FitReport(**meta["report"])
        if meta.get("feature_means") is not None:
            predictor.feature_means = np.array(meta["feature_means"], dtype=float)
        if meta.get("feature_stds") is not None:
            predictor.feature_stds = np.array(meta["feature_stds"], dtype=float)
        return predictor
