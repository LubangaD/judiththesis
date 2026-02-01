# 📈 Manufacturing Activity Forecasting in Kenya

A comprehensive Streamlit application for forecasting manufacturing activity in Kenya using cement production as a proxy indicator. This project compares statistical and deep learning approaches to time series forecasting.

## 🎯 Overview

This application provides:
- **Comparative forecasting** using ARIMA, MLP, and LSTM models
- **Interactive predictions** for 1-36 months ahead
- **Explainable AI** insights using SHAP values
- **Visual analysis** of historical trends and future projections

## 🏗️ Project Structure

```
cement-forecast-app/
│
├── app.py                          # Main entry point (intro page)
├── pages/
│   ├── 1_📊_Forecasting.py        # Model predictions and visualization
│   └── 2_🧠_Explainable_AI.py     # SHAP-based model explanations
│
├── models/                         # Pre-trained models (not included in repo)
│   ├── arima_cement.pkl           # ARIMA model
│   ├── mlp_cement_model.keras     # MLP neural network
│   ├── lstm_cement_model.keras    # LSTM neural network
│   ├── minmax_scaler.pkl          # Feature scaler
│   └── metadata.pkl               # Model metadata
│
├── data/
│   └── clean_data.csv             # Historical manufacturing data
│
├── utils/
│   └── forecasting.py             # Reusable utility functions
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd cement-forecast-app
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your models and data:**
   - Place your trained models in the `models/` directory
   - Place your historical data in the `data/` directory
   - Ensure the following files exist:
     - `models/arima_cement.pkl`
     - `models/mlp_cement_model.keras`
     - `models/lstm_cement_model.keras`
     - `models/minmax_scaler.pkl`
     - `models/metadata.pkl`
     - `data/clean_data.csv`

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📊 Features

### 1. Home Page
- Overview of the project objectives
- Information about the data and methodology
- Navigation to other pages

### 2. Forecasting Page
- **Model Selection**: Choose between ARIMA, MLP, or LSTM
- **Forecast Horizon**: Select 1-36 months ahead
- **Interactive Visualization**: View historical data and predictions
- **Download Results**: Export forecasts as CSV
- **Model Comparison**: Compare different approaches

### 3. Explainable AI Page
- **SHAP Values**: Understand feature importance
- **Summary Plots**: See overall feature impact patterns
- **Waterfall Charts**: Detailed explanation of individual predictions
- **Feature Ranking**: Identify the most influential features

## 🔧 Models

### ARIMA (AutoRegressive Integrated Moving Average)
- **Type**: Statistical time series model
- **Strengths**: Captures linear trends and seasonality
- **Use Case**: Baseline statistical approach

### MLP (Multi-Layer Perceptron)
- **Type**: Feed-forward neural network
- **Strengths**: Learns non-linear patterns
- **Architecture**: Dense layers with dropout regularization

### LSTM (Long Short-Term Memory)
- **Type**: Recurrent neural network
- **Strengths**: Captures long-term dependencies in sequences
- **Architecture**: LSTM layers optimized for time series

## 📁 Data Format

The application expects the following data structure:

**clean_data.csv**:
- Date index (datetime format)
- Target variable (e.g., cement production)
- Additional features used in modeling

**metadata.pkl** should contain:
```python
{
    'target': 'target_column_name',
    'columns': ['feature1', 'feature2', ...],
    'look_back': 12  # number of historical months used
}
```

## 🛠️ Development

### Adding New Models

1. Train your model following the same preprocessing pipeline
2. Save the model in the `models/` directory
3. Update the forecasting page to include your model
4. Ensure compatibility with the existing data format

### Customizing Features

- Modify `utils/forecasting.py` to add new utility functions
- Update `metadata.pkl` if adding new features
- Adjust the SHAP analysis for different feature sets

## 📈 Usage Examples

### Basic Forecasting
1. Select a model from the sidebar
2. Choose your forecast horizon (e.g., 12 months)
3. View the predictions and visualization
4. Download results if needed

### Understanding Predictions
1. Navigate to the Explainable AI page
2. Review the SHAP summary plot for overall feature importance
3. Select individual samples to see detailed explanations
4. Use insights to validate model behavior

## 🔍 Troubleshooting

### Common Issues

**Error: Model file not found**
- Ensure all model files are in the `models/` directory
- Check file names match exactly (case-sensitive)

**Error: Shape mismatch**
- Verify your data has the same features used during training
- Check that the look_back value matches your model configuration

**Slow SHAP computation**
- Reduce the number of samples in the Explainable AI page
- SHAP calculations are computationally intensive for large datasets

## 📚 References

- **ARIMA**: Box, G. E. P., & Jenkins, G. M. (1976). Time Series Analysis: Forecasting and Control
- **LSTM**: Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory
- **SHAP**: Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🙏 Acknowledgments

- Kenya National Bureau of Statistics for manufacturing data
- Anthropic for Claude AI assistance
- Streamlit team for the excellent framework
- SHAP library contributors for explainability tools

---

**Note**: This application is designed for research, policy analysis, and decision support. Forecasts should be interpreted with appropriate caution and validated against domain expertise.
