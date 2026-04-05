# Fabric Property Prediction Project

A descriptor-based machine learning pipeline for predicting fabric-related properties from SMILES strings, with a polished Streamlit dashboard and a FastAPI backend.

## What it does
- Parses arbitrary SMILES with RDKit
- Builds a 29-feature molecular descriptor vector
- Trains one Random Forest model per target property
- Uses scaffold-aware splitting for a more realistic evaluation
- Returns predictions plus tree-based uncertainty estimates
- Exposes a FastAPI service and a Streamlit UI

## Targets
- strength
- comfort
- sustainability
- breathability
- durability
- cost

## Important note
This project is ready for real experimental data, but the built-in demo data is synthetic and only for smoke tests and development. For genuine predictions, train on labeled experimental measurements.

## Project layout
- `app/descriptors.py` — RDKit descriptor engine
- `app/data.py` — dataset loading and demo dataset generation
- `app/ml_system.py` — training, prediction, saving, loading
- `app/api.py` — FastAPI service
- `ui/streamlit_app.py` — polished dashboard
- `train.py` — CLI training entry point
- `predict.py` — CLI inference entry point
- `tests/test_smoke.py` — smoke tests

## Install
```bash
pip install -r requirements.txt
```

## Train with demo data
```bash
python train.py --demo-samples 600
```

## Train on your own CSV
Your CSV must contain:
- `smiles`
- `strength`
- `comfort`
- `sustainability`
- `breathability`
- `durability`
- `cost`

```bash
python train.py --csv your_data.csv
```

## Predict from CLI
```bash
python predict.py "CCOCCOCCOC"
```

## Run API
```bash
uvicorn app.api:app --reload
```

## Run the Streamlit UI
```bash
streamlit run ui/streamlit_app.py
```

## Deploy with Docker
```bash
docker build -t fabric-ai-studio .
docker run -p 8501:8501 fabric-ai-studio
```

## Example CSV format
```csv
smiles,strength,comfort,sustainability,breathability,durability,cost
CCOCCOCCOC,6.2,7.0,5.5,6.8,6.0,2.7
```
