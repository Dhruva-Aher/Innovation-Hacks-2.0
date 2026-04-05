from app.descriptors import MolecularDescriptorEngine
from app.ml_system import FabricMLPredictor


def test_descriptor_engine():
    engine = MolecularDescriptorEngine()
    feats = engine.calculate_descriptors("CCO")
    assert len(feats) >= 25
    assert feats["molecular_weight"] > 0


def test_train_and_predict_demo():
    predictor = FabricMLPredictor()
    report = predictor.fit_demo(n_samples=120)
    assert report.rows == 120
    pred = predictor.predict("CCOCCO")
    assert set(pred.keys())
    assert len(pred) == 6
