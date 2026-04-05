from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data import TARGET_COLUMNS, load_csv_dataset, make_demo_dataset
from app.ml_system import FabricMLPredictor


APP_TITLE = "Fabric AI Studio"
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    .stApp {
        background: linear-gradient(180deg, #f8fbff 0%, #ffffff 42%, #f4f7fb 100%);
    }
    .hero {
        padding: 1.4rem 1.5rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
        color: white;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
        line-height: 1.1;
    }
    .hero p {
        margin: 0.35rem 0 0 0;
        font-size: 1rem;
        opacity: 0.92;
    }
    .metric-card {
        padding: 1rem 1rem 0.9rem 1rem;
        border-radius: 18px;
        background: white;
        border: 1px solid rgba(148, 163, 184, 0.22);
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
    }
    .muted {
        color: #64748b;
        font-size: 0.95rem;
    }
    .pill {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: #e0f2fe;
        color: #075985;
        font-weight: 600;
        font-size: 0.82rem;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }
    .small-note {
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_or_create_predictor() -> FabricMLPredictor:
    if (ARTIFACTS / "models.joblib").exists():
        return FabricMLPredictor.load(ARTIFACTS)
    return FabricMLPredictor()


def ensure_state() -> None:
    if "predictor" not in st.session_state:
        st.session_state.predictor = load_or_create_predictor()
    if "report" not in st.session_state:
        st.session_state.report = getattr(st.session_state.predictor, "report", None)
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>🧵 Fabric AI Studio</h1>
            <p>Predict textile-relevant properties from SMILES, inspect uncertainty, and train on real labeled data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def property_figure(predictions: Dict[str, Dict[str, float]]) -> go.Figure:
    labels = TARGET_COLUMNS
    values = [predictions[t]["prediction"] for t in labels]
    fig = go.Figure(
        data=go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name="Prediction",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
    )
    return fig


def uncertainty_figure(predictions: Dict[str, Dict[str, float]]) -> go.Figure:
    labels = TARGET_COLUMNS
    values = [predictions[t]["uncertainty"] for t in labels]
    fig = go.Figure(
        data=[go.Bar(x=labels, y=values)],
        layout=go.Layout(height=320, margin=dict(l=20, r=20, t=40, b=20)),
    )
    fig.update_yaxes(title_text="Tree variance")
    return fig


def importance_figure(pairs: List[Tuple[str, float]], title: str) -> go.Figure:
    if not pairs:
        return go.Figure()
    features = [p[0] for p in pairs][::-1]
    values = [p[1] for p in pairs][::-1]
    fig = go.Figure(go.Bar(x=values, y=features, orientation="h"))
    fig.update_layout(height=360, title=title, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def display_report(report) -> None:
    if not report:
        st.info("Train the model to see evaluation metrics.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Training rows", f"{report.rows}")
    c2.metric("Features", f"{report.features}")
    c3.metric("Source", report.source)
    st.subheader("Validation metrics")
    cols = st.columns(3)
    for idx, target in enumerate(TARGET_COLUMNS):
        with cols[idx % 3]:
            metric = report.metrics.get(target, {})
            st.markdown(
                f"""
                <div class="metric-card">
                    <strong>{target.title()}</strong><br/>
                    <span class="muted">MAE: {metric.get('mae', 0):.3f}</span><br/>
                    <span class="muted">R²: {metric.get('r2', 0):.3f}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")


def set_model(pred: FabricMLPredictor) -> None:
    st.session_state.predictor = pred
    st.session_state.report = getattr(pred, "report", None)


def train_demo_model(n_samples: int) -> None:
    with st.spinner("Training demo model..."):
        pred = FabricMLPredictor()
        pred.fit_demo(n_samples=n_samples)
        pred.save(ARTIFACTS)
        set_model(pred)
        st.success("Demo model trained and saved.")


def train_from_upload(uploaded_file) -> None:
    with st.spinner("Training from uploaded CSV..."):
        temp_path = Path(st.session_state.get("uploaded_csv_path", ARTIFACTS / "uploaded.csv"))
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_bytes(uploaded_file.getvalue())
        bundle = load_csv_dataset(temp_path)
        pred = FabricMLPredictor()
        pred.fit(bundle.frame)
        pred.save(ARTIFACTS)
        set_model(pred)
        st.success("Custom dataset trained and saved.")


ensure_state()
render_hero()

with st.sidebar:
    st.header("Control panel")
    st.caption("Train, load, and deploy from one place.")
    if st.button("Load saved model", use_container_width=True):
        if (ARTIFACTS / "models.joblib").exists():
            set_model(FabricMLPredictor.load(ARTIFACTS))
            st.success("Saved model loaded.")
        else:
            st.warning("No saved model found yet.")

    demo_samples = st.slider("Demo samples", min_value=60, max_value=2000, value=600, step=60)
    if st.button("Train demo model", use_container_width=True):
        train_demo_model(demo_samples)

    uploaded = st.file_uploader("Upload labeled CSV", type=["csv"])
    if uploaded is not None:
        st.caption("Expected columns: smiles, strength, comfort, sustainability, breathability, durability, cost")
        if st.button("Train on uploaded CSV", use_container_width=True):
            train_from_upload(uploaded)

    st.divider()
    st.markdown("**Examples**")
    examples = [
        ("Polyethylene glycol", "CCOCCOCCOC"),
        ("Cellulose unit", "OC[C@H]1O[C@H](O)[C@@H](O)[C@H](O)[C@@H]1O"),
        ("Aromatic chain", "c1ccc(cc1)CC"),
        ("Simple alkane", "CCCCCCCCCCCC"),
    ]
    for label, smiles in examples:
        if st.button(label, use_container_width=True, key=f"ex_{label}"):
            st.session_state.smiles_input = smiles

main_tabs = st.tabs(["Predict", "Model quality", "Dataset format"])

with main_tabs[0]:
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.subheader("Predict from SMILES")
        smiles = st.text_input(
            "Enter a valid SMILES string",
            value=st.session_state.get("smiles_input", "CCOCCOCCOC"),
            key="smiles_input",
            help="Any valid SMILES works. The model parses it with RDKit, computes descriptors, and predicts properties.",
        )
        go_btn = st.button("Run prediction", type="primary")

        predictor = st.session_state.predictor
        if go_btn:
            try:
                result = predictor.predict_with_uncertainty(smiles)
                st.session_state.last_prediction = result
            except Exception as exc:
                st.error(str(exc))

        result = st.session_state.last_prediction
        if result:
            meta = result.get("__meta__", {})
            if meta:
                novelty = meta.get("novelty_score", 0.0)
                if novelty > 0.75:
                    st.warning("This molecule looks somewhat outside the training distribution, so uncertainty is higher.")
                elif novelty > 0.35:
                    st.info("This molecule is moderately different from the training set.")
                else:
                    st.success("This molecule is close to the training distribution.")

            top_cols = st.columns(3)
            for idx, target in enumerate(TARGET_COLUMNS):
                with top_cols[idx % 3]:
                    vals = result[target]
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <strong>{target.title()}</strong><br/>
                            <span style="font-size:1.8rem; font-weight:700;">{vals['prediction']:.2f}</span>
                            <span class="muted"> / 10</span><br/>
                            <span class="muted">Uncertainty: ±{vals['uncertainty']:.2f}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.write("")

            st.plotly_chart(property_figure(result), use_container_width=True)
            st.plotly_chart(uncertainty_figure(result), use_container_width=True)

            with st.expander("Feature importance for each target"):
                target = st.selectbox("Choose a property", TARGET_COLUMNS, index=0)
                pairs = predictor.get_feature_importance(target, top_n=10)
                st.plotly_chart(importance_figure(pairs, f"Top features for {target}"), use_container_width=True)
                st.dataframe(pd.DataFrame(pairs, columns=["feature", "importance"]), use_container_width=True)

    with right:
        st.subheader("What the model sees")
        if result:
            try:
                from app.descriptors import MolecularDescriptorEngine
                engine = MolecularDescriptorEngine()
                df = pd.DataFrame([engine.calculate_descriptors(smiles)]).T.reset_index()
                df.columns = ["feature", "value"]
                st.dataframe(df, use_container_width=True, height=650)
            except Exception as exc:
                st.error(str(exc))
        else:
            st.info("Run a prediction to inspect the descriptor vector.")

with main_tabs[1]:
    st.subheader("Model quality")
    display_report(st.session_state.report)
    st.caption("The demo model is useful for development and UI testing. Train on real experimental labels for genuine predictions.")

with main_tabs[2]:
    st.subheader("Training data format")
    st.markdown(
        """
        Your CSV should include the following columns:

        - `smiles`
        - `strength`
        - `comfort`
        - `sustainability`
        - `breathability`
        - `durability`
        - `cost`
        """
    )
    sample_df = make_demo_dataset(n_samples=12).frame.drop(columns=["scaffold"])
    st.dataframe(sample_df, use_container_width=True)
    st.download_button(
        "Download sample CSV",
        sample_df.to_csv(index=False).encode("utf-8"),
        file_name="sample_fabric_training_data.csv",
        mime="text/csv",
    )
    st.markdown(
        """
        <div class="small-note">
            Deploy with Docker or run locally with Streamlit. The app loads the trained model from <code>artifacts/</code> if present.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption("Built on RDKit + scikit-learn + Streamlit")
