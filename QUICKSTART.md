# 🚀 Quick Start Guide

Get your cement forecasting application up and running in minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Basic familiarity with command line

## Installation Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- Streamlit (web framework)
- TensorFlow (deep learning)
- Statsmodels (ARIMA)
- SHAP (explainability)
- And more...

### 2. Setup Sample Data (Optional)

If you're testing the app without your own data:

```bash
python setup.py
```

This creates:
- Sample historical data (`data/clean_data.csv`)
- Sample metadata (`models/metadata.pkl`)

### 3. Train Models

Train the ARIMA, MLP, and LSTM models:

```bash
python train_models.py
```

This will:
- Load your data
- Train all three models
- Save trained models to `models/` directory
- Display performance metrics

Training takes 5-15 minutes depending on your hardware.

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Using Your Own Data

### Step 1: Prepare Your Data

Create `data/clean_data.csv` with this structure:

```csv
Date,Cement,GDP,Inflation,Construction,Imports
2010-01-01,1250.5,450000,2.3,580.2,230.1
2010-02-01,1280.3,452000,2.5,590.5,235.4
...
```

Requirements:
- Date column as index
- Numeric columns only
- No missing values
- Monthly frequency recommended

### Step 2: Update Metadata

Edit the metadata dictionary in `train_models.py`:

```python
target_col = 'YourTargetColumn'  # Your target variable name
feature_cols = ['Feature1', 'Feature2', ...]  # Your features
```

### Step 3: Train and Run

```bash
python train_models.py
streamlit run app.py
```

## Troubleshooting

### Issue: "No module named 'tensorflow'"
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: "Model file not found"
**Solution**: Train models first
```bash
python train_models.py
```

### Issue: "ValueError: Shape mismatch"
**Solution**: Ensure your data has the same features used during training

### Issue: SHAP is slow
**Solution**: Reduce the number of samples in the Explainable AI page (sidebar slider)

## Navigation

Once the app is running:

1. **Home Page**: Overview and introduction
2. **📊 Forecasting**: 
   - Select a model (ARIMA/MLP/LSTM)
   - Choose forecast horizon
   - View predictions and charts
   - Download results

3. **🧠 Explainable AI**:
   - Understand feature importance
   - See how features affect predictions
   - Analyze individual predictions

## Tips

- Start with ARIMA for a quick baseline
- Use LSTM for complex patterns
- Compare all models to find the best
- Longer forecast horizons = more uncertainty
- SHAP analysis helps validate model behavior

## Next Steps

- Experiment with different forecast horizons
- Compare model performance
- Analyze feature importance
- Download and share forecasts
- Customize the app for your needs

## Getting Help

- Check the main README.md for detailed documentation
- Review model training output for performance metrics
- Enable Streamlit debug mode: `streamlit run app.py --logger.level=debug`

## Performance Optimization

For faster training:
- Reduce number of epochs in `train_models.py`
- Use a smaller `look_back` value
- Sample your data if it's very large

For faster SHAP:
- Reduce number of samples in the UI
- Use fewer background samples in the code

---

**Ready to forecast? Start with:**
```bash
python setup.py
python train_models.py
streamlit run app.py
```

Happy forecasting! 📈
