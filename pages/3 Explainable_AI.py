
import streamlit as st
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(page_title="Explainable AI", layout="wide")
st.title("🧠 Explainable AI")

st.markdown("""
This page explains **how input features influence predictions** using SHAP.

✔ Fast SHAP (DeepExplainer when possible)  
✔ Time-step aggregated explanations  
✔ Human-readable explanations  
✔ Stable for any number of samples  
""")

# ------------------------------------------------------------------
# Load model & data
# ------------------------------------------------------------------
@st.cache_resource
def load_model_and_data():
    model = load_model("models/mlp_cement_model.keras")
    scaler = joblib.load("models/minmax_scaler.pkl")
    meta = joblib.load("models/metadata.pkl")
    data = pd.read_csv("data/clean_data.csv", index_col=0)
    return model, scaler, meta, data

model, scaler, meta, data = load_model_and_data()

columns = meta["columns"]
look_back = meta["look_back"]

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")
n_samples = st.sidebar.slider(
    "Number of samples to explain",
    min_value=5,
    max_value=10,
    value=10,
)

sample_idx = st.sidebar.slider(
    "Sample index",
    min_value=0,
    max_value=n_samples - 1,
    value=0,
)

# ------------------------------------------------------------------
# Prepare time-series windows
# ------------------------------------------------------------------
def make_windows(df, cols, look_back):
    X = []
    for i in range(len(df) - look_back):
        window = scaler.transform(df[cols].iloc[i:i + look_back])
        X.append(window.flatten())
    return np.array(X)

X_all = make_windows(data, columns, look_back)

X_background = X_all[:50]
X_test = X_all[-n_samples:]

# ------------------------------------------------------------------
# Feature names
# ------------------------------------------------------------------
feature_names = [
    f"{col}_t-{look_back - i - 1}"
    for i in range(look_back)
    for col in columns
]

# ------------------------------------------------------------------
# SHAP computation (NO caching – stable)
# ------------------------------------------------------------------
st.info("⏳ Computing SHAP values...")

with st.spinner("Running SHAP..."):
    try:
        explainer = shap.DeepExplainer(model, X_background)
        shap_values = explainer.shap_values(X_test)
        expected_value = explainer.expected_value
        explainer_name = "DeepExplainer (fast)"
    except Exception:
        explainer = shap.KernelExplainer(model.predict, X_background)
        shap_values = explainer.shap_values(X_test)
        expected_value = explainer.expected_value
        explainer_name = "KernelExplainer (slow)"

st.success(f"✅ SHAP computed using {explainer_name}")

# ------------------------------------------------------------------
# Normalize SHAP outputs (CRITICAL)
# ------------------------------------------------------------------
# Handle list output
if isinstance(shap_values, list):
    shap_values = shap_values[0]

shap_values = np.array(shap_values)

# Remove last dim if (n_samples, n_features, 1)
if shap_values.ndim == 3:
    shap_values = shap_values[:, :, 0]

# HARD ALIGNMENT (prevents ALL slider bugs)
shap_values = shap_values[:len(X_test)]

# Expected value → scalar float
if isinstance(expected_value, (list, np.ndarray)):
    expected_value = float(np.array(expected_value).flatten()[0])
else:
    expected_value = float(expected_value)

# ------------------------------------------------------------------
# 1️⃣ SHAP Summary Plot
# ------------------------------------------------------------------
st.subheader("📌 Feature Importance Summary")

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=feature_names,
    max_display=20,
    show=False,
)
st.pyplot(fig)
plt.clf()

# ------------------------------------------------------------------
# 2️⃣ Time-step Aggregated SHAP
# ------------------------------------------------------------------
st.subheader("📊 Time-step Aggregated Importance")

grouped = {}
for i, fname in enumerate(feature_names):
    base = fname.split("_t-")[0]
    grouped.setdefault(base, []).append(i)

agg_importance = {
    feature: np.mean(np.abs(shap_values[:, idxs]))
    for feature, idxs in grouped.items()
}

agg_df = (
    pd.DataFrame.from_dict(agg_importance, orient="index", columns=["Importance"])
    .sort_values("Importance", ascending=False)
)

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(agg_df.index, agg_df["Importance"])
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Aggregated Feature Importance")
ax.invert_yaxis()
st.pyplot(fig)
plt.clf()

# ------------------------------------------------------------------
# 3️⃣ Waterfall Plot (single prediction)
# ------------------------------------------------------------------
st.subheader("🧩 Individual Prediction Explanation")

single_shap = np.array(shap_values[sample_idx]).reshape(-1)
single_data = np.array(X_test[sample_idx]).reshape(-1)

explanation = shap.Explanation(
    values=single_shap,
    base_values=expected_value,
    data=single_data,
    feature_names=feature_names,
)

fig, ax = plt.subplots(figsize=(10, 8))
shap.plots.waterfall(explanation, max_display=15, show=False)
st.pyplot(fig)
plt.clf()

# ------------------------------------------------------------------
# 4️⃣ Human-readable explanations
# ------------------------------------------------------------------
st.subheader("🧠 Human-Readable Explanation")

top_k = 5
top_idx = np.argsort(np.abs(single_shap))[::-1][:top_k]

for idx in top_idx:
    impact = single_shap[idx]
    direction = "increased" if impact > 0 else "decreased"
    st.markdown(
        f"- **{feature_names[idx]}** {direction} the prediction by **{impact:.3f}**"
    )

# ------------------------------------------------------------------
# Info
# ------------------------------------------------------------------
with st.expander("ℹ️ How to interpret"):
    st.markdown("""
- Positive SHAP → pushes prediction higher  
- Negative SHAP → pushes prediction lower  
- Aggregated SHAP → combines all time lags  
- Waterfall → full story of one prediction  

This implementation is **stable, production-safe, and slider-proof**.
""")
