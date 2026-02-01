import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from statsmodels.tsa.arima.model import ARIMAResults

st.set_page_config(page_title="Forecasting", layout="wide")
st.title("📊 Forecasting")

# Load all models and data
@st.cache_resource
def load_assets():
    arima = ARIMAResults.load("models/arima_cement.pkl")
    mlp   = load_model("models/mlp_cement_model.keras")
    lstm  = load_model("models/lstm_cement_model.keras")
    scaler = joblib.load("models/minmax_scaler.pkl")
    meta   = joblib.load("models/metadata.pkl")
    data   = pd.read_csv("data/clean_data.csv", parse_dates=True, index_col=0)
    return arima, mlp, lstm, scaler, meta, data

arima, mlp, lstm, scaler, meta, data = load_assets()

# Sidebar controls
st.sidebar.header("⚙️ Forecast Controls")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["ARIMA", "MLP", "LSTM"]
)
forecast_steps = st.sidebar.slider(
    "Forecast Horizon (months)",
    1, 36, 12
)

# Extract metadata
TARGET = meta["target"]
columns = meta["columns"]
target_idx = columns.index(TARGET)
look_back = meta["look_back"]

# Recursive forecasting function for neural networks
def recursive_forecast(model, last_window, steps, target_idx, is_lstm=True):
    """
    Generate multi-step forecasts recursively using neural network models.
    
    Args:
        model: Trained Keras model (MLP or LSTM)
        last_window: Last known window of scaled data
        steps: Number of future steps to forecast
        target_idx: Index of target variable in feature array
        is_lstm: Boolean indicating if model is LSTM (3D input) or MLP (2D input)
    
    Returns:
        Array of predicted values (scaled)
    """
    preds = []
    window = last_window.copy()
    
    for _ in range(steps):
        # Predict next step
        if is_lstm:
            p = model.predict(window[np.newaxis, :, :], verbose=0)[0, 0]
        else:
            p = model.predict(window.reshape(1, -1), verbose=0)[0, 0]
        
        preds.append(p)
        
        # Update window: shift and insert new prediction
        window = np.roll(window, -1, axis=0)
        window[-1, target_idx] = p
    
    return np.array(preds)

# Generate forecasts
last_window = scaler.transform(data[columns].iloc[-look_back:])

if model_choice == "ARIMA":
    preds = arima.forecast(steps=forecast_steps).values
else:
    preds_scaled = recursive_forecast(
        lstm if model_choice == "LSTM" else mlp,
        last_window,
        forecast_steps,
        target_idx,
        is_lstm=(model_choice == "LSTM")
    )
    preds = scaler.inverse_transform(
        np.column_stack([preds_scaled] * len(columns))
    )[:, target_idx]

# Create forecast dates
last_date = data.index[-1]
forecast_dates = pd.date_range(start=last_date, periods=forecast_steps + 1, freq='MS')[1:]

# Display results
st.subheader(f"{model_choice} Forecast")
st.write(f"Forecasting **{forecast_steps} months** ahead from {last_date.strftime('%Y-%m')}")

# Create forecast dataframe
forecast_df = pd.DataFrame({
    'Date': forecast_dates,
    'Predicted Value': preds
})
forecast_df.set_index('Date', inplace=True)

# Plot
fig, ax = plt.subplots(figsize=(14, 6))

# Plot historical data
historical_lookback = min(60, len(data))
ax.plot(data.index[-historical_lookback:], 
        data[TARGET].iloc[-historical_lookback:], 
        label='Historical Data', 
        linewidth=2, 
        color='blue')

# Plot forecast
ax.plot(forecast_dates, 
        preds, 
        label=f'{model_choice} Forecast', 
        linewidth=2, 
        color='red', 
        linestyle='--', 
        marker='o', 
        markersize=4)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel(TARGET, fontsize=12)
ax.set_title(f'{model_choice} Forecast for {TARGET}', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()

st.pyplot(fig)

# Display forecast table
st.subheader("Forecast Values")
st.dataframe(forecast_df.style.format({'Predicted Value': '{:.2f}'}), use_container_width=True)

# Download forecast
csv = forecast_df.to_csv()
st.download_button(
    label="📥 Download Forecast as CSV",
    data=csv,
    file_name=f"{model_choice}_forecast_{forecast_steps}months.csv",
    mime="text/csv"
)

# Model information
with st.expander("ℹ️ Model Information"):
    st.markdown(f"""
    **Selected Model:** {model_choice}
    
    **Model Details:**
    - **ARIMA**: Statistical time series model using AutoRegressive Integrated Moving Average
    - **MLP**: Multi-Layer Perceptron neural network with dense layers
    - **LSTM**: Long Short-Term Memory recurrent neural network for sequence modeling
    
    **Target Variable:** {TARGET}
    
    **Features Used:** {', '.join(columns)}
    
    **Look-back Window:** {look_back} months
    """)
