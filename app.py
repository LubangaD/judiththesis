import streamlit as st

st.set_page_config(page_title="Manufacturing Forecasting", layout="wide")

st.title("📈 Manufacturing Activity Forecasting in Kenya")

st.markdown("""
### Overview
This study develops and compares ARIMA, MLP, and LSTM models to forecast Kenya’s manufacturing sector activity using high-frequency production indicators. 
The research demonstrates the superiority of deep learning models in capturing nonlinear dynamics and provides short- to medium-term projections to support strategic planning.”

### Objectives
- Model monthly manufacturing activity using **statistical and deep learning methods**
- Compare predictive performance of **ARIMA, MLP, and LSTM**
- Generate **future forecasts (2024–2026)**
- Provide **explainable AI insights** for transparency

### Data
- Monthly manufacturing indicators from official statistics
- Period covered: **historical data up to 2023**

### Models Implemented
- **ARIMA** – linear statistical baseline
- **MLP** – feed-forward neural network
- **LSTM** – deep learning sequence model

👉 Use the sidebar to navigate through the application.
""")

st.info("This tool is designed for research, policy analysis, and decision support.")
