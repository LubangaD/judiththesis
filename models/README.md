# Metadata File Structure

This file documents the expected structure of `metadata.pkl` for the application.

## Creating metadata.pkl

You can create this file using the following Python code:

```python
import joblib

metadata = {
    'target': 'Cement',  # Name of the target variable column
    'columns': ['Cement', 'Feature1', 'Feature2', ...],  # All feature columns
    'look_back': 12,  # Number of historical months to use for prediction
    'model_info': {
        'arima_order': (1, 1, 1),  # ARIMA(p, d, q) order
        'mlp_architecture': [64, 32, 16],  # Hidden layer sizes
        'lstm_units': 50,  # Number of LSTM units
    },
    'data_info': {
        'train_start': '2010-01-01',
        'train_end': '2023-12-01',
        'frequency': 'MS',  # Month Start
        'n_features': 5,  # Total number of features
    }
}

# Save metadata
joblib.dump(metadata, 'models/metadata.pkl')
```

## Required Fields

- **target** (str): Name of the target variable to forecast
- **columns** (list): List of all feature column names in order
- **look_back** (int): Number of previous timesteps used as input

## Optional Fields

- **model_info** (dict): Information about model architectures
- **data_info** (dict): Information about training data
- Any other metadata you want to store for reference

## Example

```python
{
    'target': 'Cement',
    'columns': ['Cement', 'GDP', 'Inflation', 'Construction', 'Imports'],
    'look_back': 12
}
```

## Loading Metadata

The application loads metadata like this:

```python
import joblib
meta = joblib.load('models/metadata.pkl')
target = meta['target']
columns = meta['columns']
look_back = meta['look_back']
```
